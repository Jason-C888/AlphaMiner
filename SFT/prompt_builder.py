from __future__ import annotations

import json
from textwrap import dedent

from extracter.parser.data_dict_parser import DataDictionary


SYSTEM_PROMPT = dedent(
    """
    你是量化研究因子生成助手。

    你必须严格遵守以下规则：
    - 输出必须是结构化 JSON，不附带额外解释
    - 只可使用给定字段白名单中的字段
    - 仅依赖日度数据
    - 不得使用 paused
    - reasoning 必须是因子定义说明，不得包含来源复述
    - factor_python 必须是单个 compute_factor 函数
    - 输入是宽表向量化数据，不允许逐股票 for 循环
    - 若原始逻辑依赖不可用字段，应做合理近似或写入 inavailable_inputs
    """
).strip()


def build_allowed_fields_text(data_dictionary: DataDictionary) -> str:
    lines: list[str] = []
    for table_name in sorted(data_dictionary.tables):
        table = data_dictionary.tables[table_name]
        fields = [
            field_name
            for field_name in table.fields
            if data_dictionary.has_field(field_name)
        ]
        lines.append(f"- {table_name}: {', '.join(fields)}")
    return "\n".join(lines)


def build_user_prompt(sample: dict, allowed_fields_text: str) -> str:
    inspiration = sample["inspiration"] or "无"
    return dedent(
        f"""
        任务：根据给定现象与约束，生成单因子样本。

        现象描述：
        {inspiration}

        可用字段：
        {allowed_fields_text}

        生成要求：
        1. 输出 inspiration / reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs
        2. 仅使用日频字段
        3. 代码必须为向量化宽表形式
        4. 若原始逻辑依赖不可用字段，请近似替代或写入 inavailable_inputs
        5. 输出 JSON
        """
    ).strip()


def build_assistant_content(sample: dict) -> str:
    payload = {
        "sample_id": sample["sample_id"],
        "report_title": sample["report_title"],
        "report_date": sample["report_date"],
        "broker": sample["broker"],
        "inspiration": sample["inspiration"],
        "reasoning": sample["reasoning"],
        "factor_formula": sample["factor_formula"],
        "factor_python": sample["factor_python"],
        "required_inputs": sample["required_inputs"],
        "inavailable_inputs": sample["inavailable_inputs"],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_prompt_messages(sample: dict, allowed_fields_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample, allowed_fields_text)},
    ]


def build_completion_messages(sample: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": build_assistant_content(sample),
        }
    ]


def build_prompt_completion_record(sample: dict, allowed_fields_text: str) -> dict:
    return {
        "prompt": build_prompt_messages(sample, allowed_fields_text),
        "completion": build_completion_messages(sample),
    }
