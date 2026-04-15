from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib import error, request

from .inference_config import InferenceConfig, resolve_api_key
from .prompt_builder import parse_model_output


@dataclass(frozen=True)
class BackendResponse:
    raw_text: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    backend_type: str
    model_name: str


class InferenceBackendError(RuntimeError):
    pass


class InferenceBackend:
    def generate(self, messages: list[dict[str, str]]) -> BackendResponse:
        raise NotImplementedError


class OpenAICompatBackend(InferenceBackend):
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._api_key = resolve_api_key(config)

    def generate(self, messages: list[dict[str, str]]) -> BackendResponse:
        if not self._config.backend.base_url or not self._config.backend.model:
            raise InferenceBackendError("OpenAI-compatible backend requires base_url and model.")
        if not self._api_key:
            raise InferenceBackendError("API key is missing. Set it in the config or environment.")

        payload = {
            "model": self._config.backend.model,
            "messages": messages,
            "temperature": self._config.generation.temperature,
            "max_tokens": self._config.generation.max_new_tokens,
        }
        body = _request_with_retries(
            url=_resolve_chat_completions_url(self._config.backend.base_url),
            payload=payload,
            timeout=self._config.generation.timeout,
            max_retries=self._config.generation.max_retries,
            api_key=self._api_key,
        )
        try:
            response_payload = json.loads(body)
            raw_text = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise InferenceBackendError(
                "OpenAI-compatible response is not a valid chat completions payload."
            ) from exc

        parsed_json, parse_error = parse_model_output(raw_text)
        return BackendResponse(
            raw_text=raw_text,
            parsed_json=parsed_json,
            parse_error=parse_error,
            backend_type="openai_compat",
            model_name=self._config.backend.model,
        )


class LocalHFBackend(InferenceBackend):
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None

    def generate(self, messages: list[dict[str, str]]) -> BackendResponse:
        model, tokenizer = self._load_model_and_tokenizer()
        input_ids = self._build_input_ids(tokenizer, messages)
        generate_kwargs = {
            "max_new_tokens": self._config.generation.max_new_tokens,
            "do_sample": self._config.generation.temperature > 0,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = self._config.generation.temperature
        outputs = model.generate(input_ids=input_ids, **generate_kwargs)
        generated_tokens = outputs[0][input_ids.shape[-1] :]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        parsed_json, parse_error = parse_model_output(raw_text)
        return BackendResponse(
            raw_text=raw_text,
            parsed_json=parsed_json,
            parse_error=parse_error,
            backend_type="local_hf",
            model_name=self._model_name,
        )

    @property
    def _model_name(self) -> str:
        if self._config.backend.adapter_path is not None:
            return str(self._config.backend.adapter_path)
        if self._config.backend.base_model_path is not None:
            return str(self._config.backend.base_model_path)
        return "local_hf"

    def _build_input_ids(self, tokenizer: Any, messages: list[dict[str, str]]):
        import torch

        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt_text = "\n\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            ) + "\n\nassistant:"
            encoded = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        return encoded.to(self._resolve_device(torch))

    def _load_model_and_tokenizer(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self._config.backend.base_model_path is None:
            raise InferenceBackendError("local_hf backend requires base_model_path.")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self._config.runtime.trust_remote_code,
        }
        if self._config.runtime.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_resolve_torch_dtype(self._config.runtime.torch_dtype),
            )
            model_kwargs["device_map"] = "auto"
        elif self._config.runtime.device == "auto":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(self._config.runtime.torch_dtype)
        else:
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(self._config.runtime.torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            self._config.backend.base_model_path,
            trust_remote_code=self._config.runtime.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._config.backend.base_model_path,
            **model_kwargs,
        )
        if self._config.backend.adapter_path is not None:
            model = PeftModel.from_pretrained(model, self._config.backend.adapter_path)

        if self._config.runtime.device != "auto" and not self._config.runtime.load_in_4bit:
            model = model.to(self._config.runtime.device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def _resolve_device(self, torch_module: Any) -> str:
        if self._config.runtime.device != "auto":
            return self._config.runtime.device
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"


def build_inference_backend(config: InferenceConfig) -> InferenceBackend:
    backend_type = config.backend.type
    if backend_type == "openai_compat":
        return OpenAICompatBackend(config)
    if backend_type == "local_hf":
        return LocalHFBackend(config)
    raise ValueError(f"Unsupported inference backend type: {backend_type}")


def _request_with_retries(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    api_key: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = InferenceBackendError(
                f"LLM request failed with HTTP {exc.code}: {detail}"
            )
        except error.URLError as exc:
            last_error = InferenceBackendError(f"LLM request failed: {exc.reason}")
        if attempt == max_retries:
            break
    if last_error is None:
        raise InferenceBackendError("LLM request failed for an unknown reason.")
    raise last_error


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _resolve_torch_dtype(name: str):
    import torch

    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)
