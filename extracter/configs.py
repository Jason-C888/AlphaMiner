from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


DEFAULT_ENV_FILE = ".env"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_STAGE = "discovery"
DEFAULT_MAX_FACTORS_PER_REPORT = 3
DEFAULT_MAX_SAMPLES_GENERATION = 10
DEFAULT_MAX_QPS = 2.0
DEFAULT_GENERATION_SYSTEM_PROMPT = dedent(
    """
    你是量化因子抽取助手。
    你必须只输出合法 JSON。
    你需要从候选片段中抽取单因子样本。
    inspiration 和 reasoning 必须直接描述金融现象、因子含义、市场规律和量化逻辑。
    不要出现 研报、报告、文中、作者认为、根据研报、本报告、本文、这一章节 等来源转述或元叙述。
    reasoning 必须以清晰的因子数学表达式或定义收尾。
    required_inputs 只能包含给定字段白名单中的具体字段列名，不能输出表名。
    required_inputs 必须尽可能少，且每个参数都必须在 compute_factor 中实际使用。
    report_date 和 broker 不允许臆造。
    factor_python 必须定义单个 compute_factor 函数，参数与 required_inputs 完全一致。
    禁止使用 paused、分钟级数据、日内数据、print、logging、注释。
    """
).strip()
DEFAULT_GENERATION_USER_PROMPT_TEMPLATE = dedent(
    """
    报告标题: {report_title}
    报告日期: {report_date}
    券商: {broker}
    最多输出样本数: {max_factors_per_report}

    输出格式必须是 JSON 对象，形如：
    {{"samples": [{{"inspiration": "...", "reasoning": "...", "factor_python": "...", "required_inputs": ["close"], "inavailable_inputs": []}}]}}

    允许使用的字段白名单:
    {allowed_fields}

    候选片段:
    {sections_text}
    """
).strip()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class PromptConfig:
    generation_system_prompt: str
    generation_user_prompt_template: str


@dataclass(frozen=True)
class RuntimeConfig:
    stage: str
    env_file: Path
    output_dir: Path
    report_dir: Path
    data_dict_path: Path
    max_factors_per_report: int
    max_samples_generation: int
    max_qps: float
    llm: LLMConfig
    prompts: PromptConfig


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _int_env(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None:
        return default
    return int(value)


def build_runtime_config(
    *,
    stage: str,
    env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    max_factors_per_report: int = DEFAULT_MAX_FACTORS_PER_REPORT,
    max_samples_generation: int = DEFAULT_MAX_SAMPLES_GENERATION,
    max_qps: float = DEFAULT_MAX_QPS,
) -> RuntimeConfig:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    resolved_env_file = base_dir / DEFAULT_ENV_FILE if env_file is None else Path(env_file)
    env_values = _load_env_file(resolved_env_file)
    output_dir = base_dir / DEFAULT_OUTPUT_DIR if output_path is None else Path(output_path)
    llm = LLMConfig(
        base_url=env_values.get("LLM_BASE_URL"),
        api_key=env_values.get("LLM_API_KEY"),
        model=env_values.get("LLM_MODEL"),
        timeout=_int_env(env_values, "LLM_TIMEOUT", 120),
        max_retries=_int_env(env_values, "LLM_MAX_RETRIES", 2),
    )
    prompts = PromptConfig(
        generation_system_prompt=DEFAULT_GENERATION_SYSTEM_PROMPT,
        generation_user_prompt_template=DEFAULT_GENERATION_USER_PROMPT_TEMPLATE,
    )
    return RuntimeConfig(
        stage=stage,
        env_file=resolved_env_file,
        output_dir=output_dir,
        report_dir=project_root / "研报",
        data_dict_path=base_dir / "data_dict.md",
        max_factors_per_report=max_factors_per_report,
        max_samples_generation=max_samples_generation,
        max_qps=max_qps,
        llm=llm,
        prompts=prompts,
    )
