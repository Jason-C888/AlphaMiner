# Extracter

## 1. 文档定位

本文档是 `extracter/` 模块的唯一主文档，同时承担以下三种角色：

- Spec：需求规格说明
- Technical Design：技术实现设计
- Project Plan：开发排期与进度管理

本模块后续所有需求变更，必须先更新本文档中的 Spec，再同步更新 Design 和 Plan，最后才进入实现与联调。

## 2. 模块目标

`extracter` 模块负责从 `研报/` 目录中的券商研报 PDF 中，提取以“单因子”为单位的高质量结构化样本，为后续 SFT 与 DPO 阶段提供训练数据基础。

默认技术路线：

- PDF 文本抽取：`PyPDF2`
- 主流程抽取：兼容 OpenAI Chat Completions 的 LLM 接口
- 开发语言：Python

模块内需要同时沉淀：

- README / 设计文档
- 代码
- 中间产物
- 最终样本

## 3. 范围定义

### 3.1 In Scope

- 扫描 `研报/` 下的 PDF 文件并建立清单
- 对 PDF 的文本可用性与质量进行评分
- 在候选研报中定位高置信的因子相关片段
- 调用 LLM 生成结构化单因子样本
- 对生成结果做字段、代码、输入字段和约束校验
- 输出候选列表、最终样本与失败记录

### 3.2 Out of Scope

- SFT 训练实现
- DPO 训练实现
- 因子回测与收益评估
- 分钟级、Tick 级或任何日内数据驱动的因子落地
- 非 `extracter/data_dict.md` 字段空间外的数据接入

## 4. 核心设计原则

- 先筛选、后生成：候选发现与样本生成必须解耦
- 失败显式记录：任何失败都必须进入失败记录，不能静默丢弃
- 文档优先：需求、设计、计划三者统一维护在本文档
- 白名单驱动：`required_inputs` 与 `factor_python` 只能引用数据字典允许的字段
- 日频优先：因子实现只允许依赖日度数据，可以降频
- 向量化优先：因子代码必须面向宽表输入，禁止逐股票 `for` 循环
- 可复核：最终输出既支持程序消费，也支持人工复核

## 5. 输入与输出

### 5.1 输入

- 研报 PDF：`../研报/`
- 数据字典：`extracter/data_dict.md`
- 环境配置：默认 `extracter/.env`
- CLI 运行参数

### 5.2 输出

输出目录默认位于 `extracter/output/`，至少包括：

- `candidates.csv`：筛选后的候选报告清单，至少包含报告名称、评分、排序等元信息
- `samples.jsonl`：最终通过校验的样本
- `failures.csv`：失败记录

## 6. 数据样本契约

单条样本的目标结构如下：

```python
{
  "sample_id": "report_id_factor_idx",
  "report_title": "string",
  "report_date": "string|null",
  "broker": "string|null",
  "inspiration": "string",
  "reasoning": "string",
  "factor_python": "string",
  "required_inputs": ["string"],
  "inavailable_inputs": ["string"]
}
```

### 6.1 字段语义

- `sample_id`：单样本唯一标识，默认由报告标识与因子序号组成
- `report_title`：可追溯主键之一，当前追溯主要依赖该字段
- `report_date`：报告日期，只允许从文件名与首页文本提取，无法提取时为 `null`
- `broker`：券商名，只允许从文件名与首页文本提取，无法提取时为 `null`
- `inspiration`：因子动机，不是全文摘要
- `reasoning`：围绕金融现象、量化逻辑和因子定义的摘要式推理，必须以明确的因子数学表达式收尾
- `factor_python`：由 LLM 生成的、可解析的单函数 Python 因子实现
- `required_inputs`：执行该因子所需的具体字段列名
- `inavailable_inputs`：数据字典中找不到、但原始因子逻辑需要的字段

### 6.2 文本内容约束

`inspiration` 与 `reasoning` 必须满足：

- 直接描述金融现象、因子含义、市场规律和量化逻辑
- 不允许出现“研报指出”“报告认为”“文中提到”“作者认为”等来源转述
- 不允许出现“根据研报”“本报告”“本文”“这一章节”等元叙述
- `reasoning` 不是读后总结，而应更接近因子定义说明

### 6.3 Python 代码约束

`factor_python` 必须满足：

- 函数名固定为 `compute_factor`
- 输入是向量化宽表数据，行是交易日，列是股票代码
- 输出可 reshape 为：
  - `index = DatetimeIndex`
  - `columns = 股票代码`
- 不允许逐股票 `for` 循环
- 可以使用中间变量
- 不允许注释
- 不允许 `print` / `logging`
- 默认基于前复权行情数据

### 6.4 数据频率与窗口约束

- 只允许依赖日度数据
- 分钟级、Tick 级、其他日内数据若被原逻辑依赖，必须进入 `inavailable_inputs`
- 若表达式包含滚动均值、滚动求和、滚动标准差等时间窗口，默认窗口不得超过一年数据范围
- 若原定义明显依赖更长窗口，优先改写为一年内近似版本；无法近似时写入 `inavailable_inputs`

### 6.5 缺失值与字段限制

- 聚合和算术计算必须显式满足 `NaN` 安全
- 优先使用 `pandas` 默认 `skipna=True` 语义或 `numpy.nanmean` / `numpy.nanstd` / `numpy.nansum`
- 不允许使用 `paused` 作为输入或过滤条件
- 行情字段只允许使用 `price` 表日度字段
- 基本面字段只允许使用 `valuation`、`balance`、`income`、`indicator` 表字段
- `required_inputs` 不允许写表名，只允许写具体变量列名
- 不允许新增数据字典外字段名

## 7. 流程设计

模块分为两个可独立执行的阶段。

### 7.1 Stage 1: Discovery

目标：从全量研报中发现优先进入生成流程的高质量候选报告。

执行顺序：

1. `parse pdf`
   - 扫描 `研报/` 下的 PDF
   - 提取文本并建立文档清单
2. `rating reports`
   - 对文档进行预处理与质量评估
3. `candidate discovery`
   - 从全部研报中优先筛出 5 到 10 篇文本层较好、结构清晰、主题明确的报告
   - 在单篇研报内部定位可能包含“因子构建 / 逻辑推导 / 公式定义”的高置信章节或片段
   - 候选片段仅作为报告评分和样本生成时的中间过程数据，不写入 `candidates.csv`

阶段输出：

- `output/candidates.csv`

`candidates.csv` 的最小字段集合要求：

- `report_title`：报告名称
- `score`：报告评分
- `rank`：候选排序

### 7.2 Stage 2: Generation

目标：对候选报告进行结构化因子样本生成与校验导出，候选片段只作为生成阶段的中间输入。

执行顺序：

1. `parse pdf`
   - 读取 `output/candidates.csv` 中的候选项
   - 重新解析相关 PDF 或片段
2. `async llm and sample generation`
   - 调用兼容 OpenAI Chat Completions 的 LLM 接口
   - 并发生成结构化样本
3. `validation and export`
   - 对结构、字段、代码、输入合法性进行校验
   - 输出最终样本与失败记录

阶段输出：

- `output/samples.jsonl`
- `output/failures.csv`

## 8. 模块设计

### 8.1 CLI

`extracter` 模块需要提供统一 CLI 入口。

建议参数如下：

- `--stage`
  - 取值：`discovery` / `generate`
- `--env-file`
  - 默认：`extracter/.env`
- `--output-path`
  - 默认：`extracter/output`
- `--max-factors-per-report`
  - 单篇研报允许抽取的最大不重复因子数量
- `--max-samples-generation`
  - 在 Discovery 阶段表示候选数量上限
  - 在 Generation 阶段表示最终样本数量上限
- `--max-qps`
  - 每秒最大 LLM 调用数量

### 8.2 pipeline.py

职责：流程编排与阶段切换。

要求：

- 支持按阶段单独执行
- 支持 Discovery 与 Generation 解耦运行
- 所有失败统一进入失败记录
- 不允许在阶段之间隐式吞错

### 8.3 Parser

目录建议：

- `parser/pdf_parser.py`
  - 使用 `PyPDF2` 对 PDF 进行文本解析
- `parser/data_dict_parser.py`
  - 解析 `extracter/data_dict.md`
  - 提供字段白名单查询
- `parser/parser_utils.py`
  - 解析相关公共方法

### 8.4 Validation

目录建议：

- `validation/report_rating.py`
  - 报告级评分
  - 候选研报排序
  - 章节级候选片段发现
  - 候选文件仅输出报告名称，片段信息不落盘到 `candidates.csv`
- `validation/result_validation.py`
  - 模型输出到可交付样本的转换与校验

### 8.5 LLM Client

建议文件：`llm_client.py`

职责：

- 调用兼容 OpenAI Chat Completions 的接口
- 支持异步并发
- 支持 QPS 限流
- 强制结构化 JSON 输出
- 约束 `required_inputs` 在数据字典字段空间内
- 约束 `factor_python` 为单个可解析函数

### 8.6 Configs

建议文件：`configs.py`

职责：

- 统一维护默认路径
- 统一维护阈值参数、QPS、候选数量和评分阈值
- 统一维护模型与提示词相关配置入口
- 统一维护输出文件命名：`candidates.csv`、`samples.jsonl`、`failures.csv`

### 8.7 Utils

目录建议：`utils/`

职责：

- 公共数据结构
- 序列化与导出辅助
- 通用校验工具
- 测试辅助函数

## 9. 评分与校验规则

### 9.1 报告级评分

目标不是学术排序，而是判断“哪篇报告值得进入高质量生产流程”。

当前评分信号：

| 信号 | 含义 | 作用 |
|------|------|------|
| `text_extractable` | 文本是否可抽取 | 保障流程可执行 |
| `text_length` | 文本长度是否足够 | 排除信息量不足文档 |
| `section_signal_count` | 是否存在明显章节结构 | 提高后续定位成功率 |
| `keyword_signal_count` | 是否包含因子/模型/选股等信号 | 提升主题相关性 |
| `broker_priority` | 华泰/海通/东方等优先 | 倾向结构规范报告 |
| `garble_ratio` | 乱码比例 | 过滤低质量文档 |

### 9.2 章节级候选评分

用于在单篇报告内部定位最可能承载“单因子定义”的片段。

当前使用的信号：

| 信号 | 含义 |
|------|------|
| 正向关键词 | 因子、模型、构建、定义、公式、回归、打分等 |
| 负向惩罚 | 市场综述、回测、业绩、宏观、年度策略等 |
| 文本长度 | 片段过短通常不足以支持完整因子抽取 |

### 9.3 样本结果校验

最少需要覆盖以下检查：

- 核心字段非空
- `factor_python` 可通过 `ast.parse`
- 函数参数需被 `required_inputs` 覆盖
- `required_inputs` 必须存在于 `extracter/data_dict.md`
- `required_inputs` 只能是具体字段列名，不能是表名
- 禁止 `paused`
- 禁止分钟级或其他日内数据依赖
- 禁止数据字典外字段
- 聚合函数必须满足 `NaN` 安全要求

## 10. LLM 生成要求

LLM 生成阶段应遵循以下原则：

- 输入为候选报告及其相关片段，而不是整库无约束生成
- 输出必须是结构化 JSON
- 每个样本必须完整包含所有目标字段
- `inspiration` 与 `reasoning` 必须符合非元叙述约束
- `required_inputs` 必须尽量映射到 `data_dict.md`
- `required_inputs` 只能输出字段列名，不能输出表名
- 无法满足的数据依赖必须明确写入 `inavailable_inputs`
- `factor_python` 必须是面向真实量化宽表输入的可执行函数定义

## 11. 目录规划

当前建议目录结构如下：

```text
extracter/
├── README.md
├── .env
├── data_dict.md
├── configs.py
├── cli.py
├── pipeline.py
├── llm_client.py
├── parser/
│   ├── pdf_parser.py
│   ├── data_dict_parser.py
│   └── parser_utils.py
├── validation/
│   ├── report_rating.py
│   └── result_validation.py
├── utils/
└── output/
    ├── candidates.csv
    ├── samples.jsonl
    └── failures.csv
```

## 12. 开发计划

### 12.1 里程碑

| 里程碑 | 目标 | 状态 |
|------|------|------|
| M0 | 明确 Spec / Design / Plan，初始化 README | Completed |
| M1 | 完成目录骨架与配置约束梳理 | Completed |
| M2 | 完成 PDF 解析与数据字典解析 | Completed |
| M3 | 完成报告级评分与候选发现 | Completed |
| M4 | 完成 LLM 样本生成链路 | In Progress |
| M5 | 完成结果校验与导出 | In Progress |
| M6 | 完成最小可运行链路联调 | In Progress |

### 12.2 交付顺序

1. 固化 README 中的需求、命名和输出规范
2. 建立最小目录结构与配置入口
3. 先打通 Discovery
4. 再打通 Generation
5. 最后补齐验证、失败记录与人工复核视图

### 12.3 进度维护规则

- 每次需求变更必须更新本文档
- 每个里程碑开始前补充输入、输出、验收标准
- 每个里程碑结束后更新状态与实际偏差

## 13. 验收标准

`extracter` 模块的最小验收标准如下：

- 可扫描 `研报/` 目录并建立候选输入
- 可输出候选清单
- 可对候选项调用 LLM 生成结构化样本
- 可校验 `factor_python` 语法和输入字段合法性
- 可输出 `samples.jsonl`
- 任意失败均可在失败记录中定位
