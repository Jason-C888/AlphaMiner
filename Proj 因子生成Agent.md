# Proj 因子生成Agent

## 项目背景

因子挖掘（Alpha Factor Discovery）是量化投资研究的核心问题之一。传统因子构建主要依赖研究员基于市场经验与经济学直觉，对金融现象进行归纳总结，并进一步形式化为可计算的因子表达式。这一过程通常遵循“**现象观察 → 逻辑推理 → 数学建模 → 回测验证**”的研究范式。尽管该方法在过去数十年中积累了大量有效因子，但其本质上高度依赖专家经验，存在可扩展性差、效率低以及难以系统化的问题。

本项目希望提出一种新的研究框架：**基于推理范式对齐的因子挖掘大模型 Agent**。我们的核心思想是，将量化研究员的思考流程显式建模，并通过大模型进行学习与复现。具体而言，我们的工作包括：

- **（1）基于 SFT 的推理范式注入**：构建“现象—推理及因子公式—Python代码实现”的结构化数据，对模型进行监督微调，使其学习因子构建的思维链；
- **（2）基于 DPO 的偏好对齐**：利用量化研究员对因子优劣的隐式评判标准，构建偏好数据，对模型进行对齐，使其输出符合实际研究偏好；
- **（3）Agent 化因子研究系统**：设计一个能够接受现象与观察，进行推理、因子构建与验证的智能体，实现从研究到落地的自动化流程。

与现有方法相比，本文的关键创新在于：将因子挖掘问题从“搜索问题”转化为“**推理与知识建模问题**”。通过引入结构化推理与人类偏好对齐，我们的方法在提升因子生成效率的同时，也显著增强了结果的可解释性与经济意义。



## 项目设计

1. 实验将包含三个阶段
   - extracter模块：从6G研报数据中提取SFT数据
   - SFT模块：进行SFT注入思考范式
   - DPO模块：进行DPO对齐因子研究偏好

### Extracter

1. 目标：从`研报/` 目录中的研报 PDF 中，沉淀出以“单因子”为单位的高质量样本。方案基于Python进行开发，默认采用“PDF 文本抽取（PyPDF2） + LLM 主流程”的路线。readme文档、代码、中间产物存放在extracter/文件夹下

2. Readme的要求：该 README 不是普通说明文档，而是一个**工程级 Spec + 技术设计文档 + 项目管理文档**，必须满足以下要求：

   - 该 README 同时承担三种角色：

     - **Spec（需求规格说明）**

     - **Technical Design（技术实现设计）**

     - **Project Plan（开发排期与进度管理）**

   - 文档必须支持项目的**持续迭代更新**，每次需求变更必须先更新 Spec，再更新 Design 和 Plan。

3. 数据样本定义标准

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
     "inavailable_inputs" ["string"]
   }
   ```

   - 默认约束：
     - `inspiration` 是因子动机，不是全文摘要。
     - `reasoning` 是“研报证据驱动的推理”，可以提取摘要式逻辑，不要求逐字复刻原文，使用的数据尽量来自data_dict.md
        - 必须基于研报内容做摘要式推理，并以推导出的明确的因子数学表达式收尾
        - 内容要像因子定义说明，而不是读后总结
        - `inspiration` 和 `reasoning`必须直接描述金融现象、因子含义、市场规律和量化逻辑，不允许提到“研报”、“报告”，不允许写“研报指出”“报告认为”“文中提到”“作者认为”“该章节说明”等来源转述语句。
        - 不要总结写作过程，不要评价研报，不要出现“根据研报/从研报来看/本文/本报告/这一章节”等元叙述。
     - `factor_python` 由 LLM 根据研报内容生成，不要求研报原文已提供代码。
     - `required_inputs` 必须是可执行层面的字段名或数据表需求。
        - 只可以使用data_dict.md数据字典中提供的字段，如果没有请选择其中可以起到替代作用的
     - `inavailable_inputs`: data_dict.md无法找到但是需要的字段
     - 可追溯性当前主要依赖 `report_title`

   - 生成的Python代码约束

     - 数据频率限制：生成的因子必须仅依赖日度数据，其他的分钟级、tick 级或其他日内频率数据要纳入`inavailable_inputs`但不要硬造字段名或违规代码。

     - 滚动窗口限制：所提供的数据只有一年，凡是表达式中出现滚动均值、滚动求和、滚动标准差等时间窗口算子，默认窗口不得超出限制一年得限制；若研报原始定义明显依赖更长窗口，优先改写为不超过一年的近似版本，否则标记入`inavailable_inputs`

     - 代码格式约束：禁止使用注释和任何 print / logging 输出，函数名为compute_factor

     - 缺失值处理：表达式中的聚合和算术计算必须显式跳过 `NaN`，优先使用 `pandas` 默认 `skipna=True` 语义或 `numpy` 的 `nanmean`、`nanstd`、`nansum` 等安全写法；不得因为单个缺失值导致整段截面或时间窗口无意义失效。

     - 停牌字段限制：不允许使用 `paused` 字段作为输入或过滤条件；停牌股票已通过价格/成交量字段中的 `NaN` 体现，相关逻辑统一依赖缺失值传播与跳过机制处理。

     - 字段白名单限制：`required_inputs` 和 `factor_python` 中允许引用的字段，必须严格来自文档中列出的字段集合。
       - 行情字段只允许使用 `extracter/data_dict.md` 的 `price` 表日度字段。
       - 基本面字段只允许使用 `extracter/data_dict.md` 中与聚宽文档一致的 `valuation`、`balance`、`income`、`indicator` 表字段。
       - 不允许新增文档外字段名，不允许把 `paused` 继续视为可用字段，不允许以分钟数据衍生出的日频中间字段充当输入

        - 输入为向量化数据（numpy / pandas Series）, 参数是 required_inputs 和 inavailable_inputs 中列出的字段英文名, 每一个输入是一个宽表，行是交易日，列是股票代码,数值是因子值

        - 不允许 for 循环逐股票计算

        - 可使用中间变量（如 vwap = money / volume）

        - 因子计算默认基于“前复权行情数据”

        - 最终因子需可被 reshape 为：
            - index = DatetimeIndex（交易日）
            - columns = 股票代码（如 000001.XSHE / 600000.XSHG）
            
        - 参考形式：
            
            - ```python
                 def compute_factor(close, volume, ...):
                     return factor_df
                 ```

   4. 项目主流程

      - 候选文本筛选阶段

        - PDF解析与文本结构解析

        - PDF筛选与打分

      - 信息抽取预处理阶段

        - PDF解析与文本结构解析
        - LLM因子抽取样本生成

        - 校验与后处理

   5. 主要模块

      - CLI：Extracter模块的入口

        - 输入参数
          - stage：dicovery/generate分别执行对应步骤
          - env-file：默认extracter/.env
          - output path: extracter/默认output
          - max-factors-per-report：每篇研报允许抽取的最大不重复因子数量
          - max_samples_generation: 对于第一阶段是生成的candidates的数量，对于第二阶段是生成的最终样本数量
          - max-qps：每秒钟最大llm调用的数量

      - pipeline.py ：负责流程编排，可以输入参数选择调用流程

        - Stage1: Discovery
          - 执行顺序：
            - parse pdf：扫描研报/下的PDF，建立文档清单
            - rating reports：文档预处理与质量评估
            - candidate discovery :从全部研报中优先筛出 5 到 10 篇文本层较好、结构清晰、主题明确的报告,在单篇研报内部定位可能包含“因子构建 / 逻辑推导 / 公式定义”的高置信章节或片段。
          - output中生成candiates.csv
        - Stage2: Generation
          - 执行顺序
            - parse pdf:读取output中的candidates.csv中的样本，进行解析
            - async LLM and sample generation：调用兼容 OpenAI Chat Completions 的 LLM 接口，生成结构化样本。
            - validation and export：通过程序校验结构正确性，再输出人工复核视图和失败记录。
        - 实现要求：
          - 候选筛选与样本生成解耦，可分别调用
          - 所有失败显式进入失败记录output/failures.csv，而不是静默丢弃

      - Parser文件夹

        - pdf_parser.py: 使用PyPDF2对pdf进行解析，提取出文本信息
        - Data_dcit_parser.py: 用于解析提供data_dict.md文本，其中用md表格的格式存储提取样本的factor_python的可用字段
        - parser_utils: parser模块的通用utils

      - Validation文件夹

        - Report_rating.py: 实现报告级评分PDF筛选与打分、候选研报排序和章节级候选因子发现

          - 报告级评分：报告级评分的目标不是做学术意义上的排序模型，而是判断“哪篇报告值得进入 Phase 1 的高质量生产流程”。当前评分信号如下：

          | 信号                   | 含义                         | 作用               |
          | ---------------------- | ---------------------------- | ------------------ |
          | `text_extractable`     | 文本是否可抽取               | 保障流程可执行     |
          | `text_length`          | 文本长度是否足够             | 排除信息量不足文档 |
          | `section_signal_count` | 是否存在明显章节结构         | 提高后续定位成功率 |
          | `keyword_signal_count` | 是否包含因子/模型/选股等信号 | 提升主题相关性     |
          | `broker_priority`      | 华泰/海通/东方等优先         | 倾向结构规范报告   |
          | `garble_ratio`         | 乱码比例                     | 过滤低质量文档     |

          - 章节级候选评分章节级评分用于在一篇报告内部找出最可能承载“单因子定义”的片段。当前使用的信号：

          | 信号       | 含义                                       |
          | ---------- | ------------------------------------------ |
          | 正向关键词 | 因子、模型、构建、定义、公式、回归、打分等 |
          | 负向惩罚   | 市场综述、回测、业绩、宏观、年度策略等     |
          | 文本长度   | 片段太短时通常不足以支持完整因子抽取       |

        - Result_valiation.py: 负责把模型输出转化为可交付样本

          - 校验内容
            - 核心字段非空
            - `factor_python` 可通过 `ast.parse`
            - 函数参数需被 `required_inputs` 覆盖
            - `required_inputs` 必须存在于 `extracter/data_dict.md` 或表级需求中
            - 禁止 `paused`、分钟级/日内数据依赖和文档外字段
            - 对聚合函数的缺失值处理进行 `NaN` 安全检查
          - 最终产生结果sample.jsonl存储在output中
          - 失败的结果记入output/failures.csv

      - LLM_client.py：实现LLM调用和样本生成

        - 调用兼容 OpenAI 的 Chat Completions 接口，按 DeepSeek 接入。使用异步调用实现并发，设置限流器，允许输入QPS限流参数
        - 设计要点：
          - 设计提示词
          - 提取的数据要符合前文定义的样本标准，包含所有字段
          - 输出强制为结构化 JSON。
          - `required_inputs` 被约束在 `extracter/data_dict.md` 字段空间内。
          - `factor_python` 必须是单个可解析函数，并面向真实量化场景的向量化宽表输入

      - Utils文件夹（可选）：其他对项目有用的辅助文件，比如测试函数，数据结构等

      - Configs.py：项目参数

      - Output文件夹：装在项目的生成文件，需要包含

        - failures.jsonl: 中间任意缓解失败的样本
          - stage：失败的stage
          - report_title：处理失败样本所属于的报告名称
          - reason：处理失败的原因
            - reason_type:
              - PARSE_ERROR
              - NO_FACTOR_FOUND
              - INVALID_JSON
              - INVALID_CODE
              - LOW_QUALITY
              - OTHER
        - candiates.csv：存储Report_rating生成的候选报告名称
          - 候选报告名称
          - 排序
        - samples.jsonl: 最终生成结果json文件，生成的样本服从数据样本定义标准要求

## SFT

## DPO

