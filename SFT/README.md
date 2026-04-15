# SFT

## 1. 文档定位

本文档是 `SFT/` 模块的唯一主文档，同时承担以下三种角色：

- Spec：需求规格说明
- Technical Design：技术实现设计
- Project Plan：开发排期与进度管理

本模块后续所有需求变更，必须先更新本文档中的 Spec，再同步更新 Design 和 Plan，最后才进入实现与联调。

## 2. 模块目标

`SFT` 模块负责在 `extracter/` 已沉淀的高质量结构化样本基础上，对基础模型进行监督微调，将“金融现象理解 -> 研究逻辑推导 -> 因子公式表达 -> Python 实现”这一研究范式注入模型。

该模块的目标不是让模型背诵固定因子模板，而是让模型稳定学习以下能力：

- 根据给定的市场现象、研究观察和任务约束，生成合理的因子研究思路
- 将研究思路组织成一致、规范、可追溯的 `reasoning / factor_formula / factor_python` 输出
- 在受限字段空间下完成近似建模，而不是发明文档外字段
- 同时保持经济含义解释、数学表达和代码实现之间的一致性

当前基线方案：

- 基础模型：`Qwen3-0.6B`
- 微调方式：`LoRA / QLoRA`
- 训练形式：`instruction tuning / chat-style supervised fine-tuning`

模块内需要同时沉淀：

- README / 设计文档
- 数据构建结果
- 训练配置
- 模型权重与评估结果
- 推理 demo 与案例材料

## 3. 范围定义

### 3.1 In Scope

- 基于 `extracter/output/samples.jsonl` 构建 SFT 训练数据
- 对 Extracter 输出样本进行二次清洗、去重和难度分层
- 将样本转换为统一的 chat-format 训练数据
- 进行 SFT 训练、验证和测试集评估
- 提供自动评估与人工抽样复核所需材料
- 提供最小可用的推理 demo
- 为后续 DPO 阶段沉淀统一的数据接口和候选输出基础

### 3.2 Out of Scope

- `extracter/` 模块的数据生产与 PDF 解析
- DPO 训练与偏好对构建
- 因子回测收益评估与实盘接入
- Agent 编排、任务调度与多工具执行系统
- 分钟级、Tick 级或任何日内数据驱动的训练目标扩展

## 4. 核心设计原则

- 范式注入优先：训练目标是学习研究范式，而不是只记忆输出模板
- 数据质量优先于样本量：训练前的数据清洗比单纯堆样本更重要
- 训练与推理对齐：训练期和推理期尽量复用同一套 system / user / assistant 结构
- 报告级切分防泄漏：按 `report_title` 划分数据集，避免同报告样本跨集合泄漏
- 结构化输出优先：模型输出必须稳定收敛到固定 JSON 结构
- 字段一致性优先：`inspiration`、`reasoning`、`factor_formula`、`factor_python` 和字段约束必须前后一致
- 约束遵守优先：模型必须学习字段白名单、日频限制和不可用字段处理规则

## 5. 输入与输出

### 5.1 输入

- Extracter 样本：`extracter/output/samples.jsonl`
- 数据字典：`extracter/data_dict.md`
- 模块 README 中固化的 Prompt 约束和训练规则
- 模型训练配置

### 5.2 输出

`SFT/` 模块需要产出至少以下内容：

- `data/train.jsonl`：训练集
- `data/val.jsonl`：验证集
- `data/test.jsonl`：测试集
- `configs/train_config.yaml`：训练配置
- `checkpoints/`：SFT Adapter 或模型权重
- `output/eval_report.json`：自动评估结果
- `output/case_studies.md`：人工复核案例与分析记录
- 推理 demo 所需脚本或调用样例

说明：

- 上述路径是模块内建议产物路径，用于约束后续实现方向
- 本文档阶段只初始化规范，不要求这些文件当前已存在

## 6. 数据契约与训练样本格式

### 6.1 原始输入样本契约

SFT 的原始输入样本沿用 Extracter 输出 schema：

```python
{
  "sample_id": "report_id_factor_idx",
  "report_title": "string",
  "report_date": "string|null",
  "broker": "string|null",
  "inspiration": "string",
  "reasoning": "string",
  "factor_formula": "string",
  "factor_python": "string",
  "required_inputs": ["string"],
  "inavailable_inputs": ["string"]
}
```

字段含义以 `extracter/README.md` 中的定义为准；SFT 阶段不重定义字段语义，只在训练构造时决定如何映射到 prompt 和目标输出。

进行预处理后保留干净样本，并加入难度分层字段 `class`（简单、中等、困难）、数据生产版本 `version`（以 `sft_<timestamp>` 形式生成），同时提前计算：

- `length_input`：`inspiration` 的字符长度
- `length_output`：`reasoning + factor_formula + factor_python` 的字符总长度

```python
{
  "sample_id": "report_id_factor_idx",
  "report_title": "string",
  "report_date": "string|null",
  "broker": "string|null",
  "class": "string|null",
  "version": "string|null",
  "inspiration": "string",
  "reasoning": "string",
  "factor_formula": "string",
  "factor_python": "string",
  "required_inputs": ["string"],
  "inavailable_inputs": ["string"],
  "length_input": 0,
  "length_output": 0
}
```



### 6.2 训练任务定义

SFT 采用 chat-style supervised fine-tuning。每条样本由“输入任务描述 + 输出目标答案”构成，模型在单次响应中同时完成：

- 动机提炼
- 金融逻辑推理
- 数学表达生成
- Python 实现生成
- 输入字段与不可用字段约束对齐

输入信息原则上包括：

1. 任务描述
2. 现象描述或研究观察
3. 可用字段约束
4. 生成要求

输出目标为完整结构化结果，而不是单独预测某个字段。

### 6.3 Chat 训练样本格式

当前实现统一使用 `prompt / completion / metadata` 结构的 chat JSONL，便于训练、评估和推理复用同一套模板。单条样本形式如下：

```python
{
  "prompt": [
    {"role": "system", "content": "...系统约束..."},
    {"role": "user", "content": "...任务描述..."}
  ],
  "completion": [
    {"role": "assistant", "content": "...目标 JSON 输出..."}
  ],
  "metadata": {
    "sample_id": "...",
    "report_title": "...",
    "report_date": "...",
    "broker": "...",
    "inspiration": "...",
    "class": "...",
    "version": "..."
  }
}
```

其中：

- `prompt`：由 `system + user` 组成，承载稳定规则、现象描述、字段白名单和生成要求
- `completion`：只包含 assistant 目标 JSON，用于 SFT 监督
- `metadata`：保留样本追踪信息，不进入 assistant 学习目标

### 6.4 Assistant 输出契约

SFT 的监督目标是 assistant 侧完整 JSON。输出结构保持如下形式：

```python
{
  "reasoning": "...",
  "factor_formula": "...",
  "factor_python": "def compute_factor(...):\n    ...",
  "required_inputs": ["..."],
  "inavailable_inputs": []
}
```

要求：

- 输出必须是合法 JSON
- 不附带前后说明文字
- 字段间语义必须一致
- `factor_python` 必须保持为单个 `compute_factor` 函数

### 6.5 评估结果数据结构

评估中，将数据集中的元信息和SFT数据信息拼接合成最总的评估数据

 ```python
 {
   "inspiration": "...",
   "sample_id": "...",
   "report_title": "...",
   "report_date": "...",
   "broker": "...",
   "reasoning": "...",
   "factor_formula": "...",
   "factor_python": "def compute_factor(...):\n    ...",
   "required_inputs": ["..."],
   "inavailable_inputs": []
 }
 ```



## 7. 数据构建与清洗设计

### 7.1 数据处理流程

SFT 数据构建流程如下：

1. 读取 `extracter/output/samples.jsonl`
2. 执行高质量过滤和重复去重
3. 按难度做样本分层（当前使用启发式规则实现）
4. 构造统一的 chat-format 样本
5. 按报告级切分 `train / val / test`
6. 序列化为训练框架可消费的数据格式
7. 记录数据版本号和样本统计信息

### 7.2 清洗目标

当前实现先使用本地规则完成一轮可重复的数据清洗，后续可再叠加基于大模型的精细质量评审。

SFT 的清洗目标不是判断样本“能否交付”，而是判断“是否适合作为模型模仿对象”。

在进入训练集前，需要删除以下类型样本：

- `inspiration` 或 `reasoning` 明显空泛、套话化的样本
- `factor_formula` 与 `factor_python` 不一致的样本
- 代码虽可解析但经济含义明显错误的样本
- 字段约束不自洽的样本
- 严重重复样本
- 过度依赖原文转述语句的样本
- 因子逻辑与日频输入约束明显冲突的样本

### 7.3 难度分层

当前实现使用启发式难度分层，为后续 curriculum training 预留可替换接口。

建议按难度组织 SFT 数据，而不是简单混合训练。

基础模式样本：

- 价格相对位置类
- 波动率类
- 成交量占比类
- 简单量价关系类

中等复杂度样本：

- 多字段组合因子
- 滚动统计类因子
- 需要对不可用字段进行替代表达的样本

高复杂度样本：

- 带有经济含义转译的复杂表达
- 研报定义较抽象、需要转化为可执行公式的样本
- 同时涉及输入约束、替代逻辑和实现细节的样本

训练上采用“先基础、后混合”的 curriculum 思路，提高格式学习和一致性学习的稳定性。

### 7.4 数据切分

数据切分按报告级别进行，而不是随机按样本切分。

固定规则：

- `Train = 80%`
- `Validation = 10%`
- `Test = 10%`

切分约束：

- 同一 `report_title` 只能出现在一个集合中
- 不同集合中的券商、年份和主题分布尽量均衡
- 需要显式记录切分版本，保证实验可复现

## 8. Prompt 设计

Prompt设计可以参考数据提取部分的要求，以带来一致文件的结果生成

### 8.1 System Prompt 职责

`system` 负责固化长期不变的规则，内容包括但不限于：

- 你是量化研究因子生成助手
- 输出必须为结构化 JSON
- 只可使用 `data_dict.md` 中允许的字段
- 仅依赖日度数据
- 不得使用 `paused`
- `reasoning` 必须是因子定义说明，而不是来源复述
- `factor_python` 必须是单个 `compute_factor` 函数
- 输入是宽表向量化数据，不允许逐股票 `for` 循环

要求：

- `system` 只放稳定规则，不承载样本级信息
- 长度应受控，避免挤占上下文

### 8.2 User Prompt 职责

`user` 负责描述单条任务上下文，原则上包含：

- 任务说明
- 现象描述或研究目标
- 可用字段集合
- 生成要求

建议结构：

```text
任务：根据给定现象与约束，生成单因子样本。
现象描述：
...

可用字段：
...

生成要求：
...
```

### 8.3 Assistant Prompt 职责

`assistant` 不承载说明性文本，只返回最终监督目标 JSON。其职责是让模型学习：

- 从自然语言现象到因子定义的映射
- 从金融逻辑到数学表达的映射
- 从公式到 Python 实现的映射
- 字段约束遵守能力

训练时的 loss 只对 `assistant` 输出部分计算：

- `system` 和 `user` 作为条件输入
- `assistant` JSON 作为监督目标

## 9. 训练实现设计

### 9.1 目标产物

SFT 阶段需要交付：

- 训练集、验证集、测试集
- 训练配置文件
- 训练脚本
- 最终 SFT Adapter 或全量模型权重
- 自动评估报告
- 推理 demo 与案例集

### 9.2 模型与训练方式

当前锁定的基线方案：

- 模型：`Qwen3-0.6B`
- 微调范式：`LoRA / QLoRA`
- 训练框架：`TRL`

选择原则：

- 成本低
- 迭代快
- 便于与后续 DPO 共用底座
- 尽量保留基础模型的通用能力

训练目标仍采用标准 next-token prediction，但监督焦点是 assistant 侧的结构化结果。

### 9.3 模块划分

`SFT/` 后续实现建议至少包含以下职责模块：

- Dataset Builder：负责清洗、去重、分层、切分和 chat-format 数据构造
- Prompt Builder：负责统一 system / user / assistant 模板，保证训练与推理分布尽量一致
- Trainer：负责基于 `TRL` 的 `SFTTrainer` / 后续 `DPOTrainer` 进行模型加载、LoRA/QLoRA 配置、训练、验证和 checkpoint 保存
- Evaluator：负责自动评估、抽样分析和人工复核导出
- Inference Demo：负责对给定现象生成因子草案，供后续 Agent 集成调用

### 9.4 训练过程关注点

虽然底层是语言建模损失，但该任务关注的有效学习信号包括：

- 输出格式学习
- 字段一致性学习
- 金融逻辑到公式的映射学习
- 公式到代码实现的映射学习
- 约束条件遵守能力

因此数据质量、一致性和标签稳定性比单纯扩展样本量更重要。

## 10. 训练配置原则

### 10.1 输入长度设计

单条样本通常包含：

- `system` 规则
- `user` 任务与字段约束
- `assistant` 全量 JSON

因此需要保证：

- `max_prompt_length` 能完整保留 `system + user`
- `max_seq_length` 能覆盖 `assistant` 中的代码段和 JSON 尾部

若长度受限，优先保留：

- `system` 规则
- 用户现象描述
- 可用字段约束
- `assistant` 输出完整性

不应截断 assistant JSON 尾部，否则会破坏训练分布。

### 10.2 超参数原则

当前阶段只锁定原则，不预设具体数值：

- 使用较小学习率，避免破坏基础模型语义能力
- 保持足够的 warmup
- 使用 validation loss 与结构化指标共同早停
- 优先稳定收敛，而不是追求极限训练轮数
- 同步记录数据版本、模型版本和训练配置版本

### 10.3 训练轮次原则

建议从少量 epoch 起步，再依据验证集结果决定是否继续。该任务模板性较强，过多 epoch 容易导致：

- 输出模板僵化
- 语言表达单一
- 对训练样本过拟合
- 对新现象的泛化能力下降

## 11. 评估与验收标准

### 11.1 自动评估指标

SFT 阶段不能只看 loss，至少应覆盖以下三层评估：格式正确性、实现正确性、研究合理性。

自动评估指标包括：

- 结构化输出成功率
  - 是否为合法 JSON
  - 是否包含全部字段
  - 字段类型是否正确
- 代码可解析率
  - 对 `factor_python` 执行 `ast.parse`
- 字段白名单合规率
  - `required_inputs` 是否均在白名单中
  - `factor_formula` 是否引用非法字段
  - `factor_python` 是否使用非法字段
  - 是否出现 `paused` 或分钟级暗含字段
- 参数覆盖率
  - 函数参数是否被 `required_inputs` 覆盖
- 约束遵守率
  - 输出是否能满足后续规则校验和回测接入前置要求

### 11.2 语义一致性评估

除自动规则外，还需要评估字段间一致性，可采用规则检查与 LLM-as-judge 结合的方式：

- `inspiration` 与 `reasoning` 是否描述同一类现象
- `reasoning` 与 `factor_formula` 是否一致
- `factor_formula` 与 `factor_python` 是否一致
- `required_inputs` 与代码参数是否一致

### 11.3 最终验收标准

SFT 阶段最小验收标准如下：

- 模型能稳定输出合法结构化结果
- 大部分样本代码可解析并通过规则校验
- 模型对字段空间约束有稳定遵循能力
- 模型能从自然语言现象描述生成较合理的因子定义
- 相较未微调基座模型，在领域任务上有明显提升

## 12. 推理接口

SFT 模型在部署时，承担“研究草案生成器”的角色。建议对外提供统一推理接口。

### 12.1 输入

- 现象描述
- 可用字段集合
- 任务约束

### 12.2 输出

- 结构化因子定义 JSON

### 12.3 接口要求

- 推理期复用训练期的 Prompt 模板，只用user输入来替换inspiration
- 输出格式与训练期 assistant 目标保持一致
- 结果可直接进入后续规则校验、人工复核或 DPO 候选收集流程

## 13. 模块划分与建议目录

当前仅初始化文档，以下为后续实现建议目录，不代表已创建：

```text
SFT/
├── README.md
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── configs/
│   └── train_config.yaml
├── checkpoints/
├── output/
│   ├── eval_report.json
│   └── case_studies.md
└── ...
```

目录职责约束：

- `data/`：放训练、验证、测试数据及数据版本统计
- `configs/`：放训练配置
- `checkpoints/`：放训练中间产物和最终权重
- `output/`：放评估报告、案例复盘和人工复核材料

## 14. 开发计划

### 14.1 里程碑

| 里程碑 | 目标 | 状态 |
|------|------|------|
| M0 | 明确 Spec / Design / Plan，初始化 README | Completed |
| M1 | 完成 Extracter 样本质量复查与清洗规则固化 | Completed |
| M2 | 完成 chat-format 数据构造与数据切分 | Completed |
| M3 | 跑通基础 SFT 训练链路 | In Progress |
| M4 | 完成自动评估与最小推理 demo | Pending |
| M5 | 为 DPO 准备候选输出与案例沉淀 | Pending |

### 14.2 交付顺序

1. 固化 README 中的需求、数据契约和评估标准
2. 完成 Extracter 样本质量复查和 SFT 清洗规则
3. 构造 chat-format 训练数据并完成报告级切分
4. 跑通基础 SFT 训练
5. 建立自动评估和人工抽样复核流程
6. 输出最小可用推理 demo 和案例材料
7. 收集验证集候选输出，为 DPO 阶段做准备

### 14.3 阶段规划

Phase 1：数据准备

- 完成 Extracter 输出样本的质量复查
- 建立 SFT 清洗规则
- 生成第一版 `train / val / test`

Phase 2：训练打通

- 完成 chat-format 数据构造
- 跑通基础 SFT 训练
- 完成最小可用推理 demo

Phase 3：评估与迭代

- 建立自动评估脚本
- 人工抽样评估案例
- 分析失败类型并回流修改数据与 Prompt

Phase 4：为 DPO 做准备

- 收集 SFT 模型在验证集上的多个候选输出
- 标注偏好对
- 沉淀 DPO 数据构建流程

### 14.4 进度维护规则

- 每次需求变更必须先更新本文档
- 每个里程碑开始前补充输入、输出和验收标准
- 每个里程碑结束后更新状态与实际偏差
