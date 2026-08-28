import json
import math
import os
import requests
from collections.abc import Mapping
from typing import Any, Dict, Sequence


LOGPROB_SUMMARY_KEYS = (
    "logprob_available",
    "sequence_logprob_sum",
    "sequence_logprob_mean",
    "sequence_token_count",
    "logprob_unavailable_reason",
)


def unavailable_logprob_summary(reason: str, token_count: int = 0) -> Dict[str, Any]:
    return {
        "logprob_available": False,
        "sequence_logprob_sum": None,
        "sequence_logprob_mean": None,
        "sequence_token_count": token_count,
        "logprob_unavailable_reason": reason,
    }


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def summarize_choice_logprobs(choice) -> Dict[str, Any]:
    """Summarize OpenAI/vLLM chat choice token log-probabilities."""
    if choice is None:
        return unavailable_logprob_summary("Model API returned no choice.")

    request_reason = _field(choice, "_logprob_unavailable_reason")
    logprobs = _field(choice, "logprobs")
    if logprobs is None:
        return unavailable_logprob_summary(
            request_reason or "Model API response omitted choice.logprobs."
        )

    content = _field(logprobs, "content")
    if content is None:
        return unavailable_logprob_summary(
            request_reason or "Model API response omitted choice.logprobs.content."
        )
    if not isinstance(content, (list, tuple)):
        return unavailable_logprob_summary(
            "choice.logprobs.content is not a token list."
        )
    if not content:
        return unavailable_logprob_summary("choice.logprobs.content is empty.")

    values = []
    invalid_indices = []
    for index, token_entry in enumerate(content):
        value = _field(token_entry, "logprob")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            invalid_indices.append(index)
        else:
            values.append(float(value))

    token_count = len(content)
    if invalid_indices:
        preview = ", ".join(str(index) for index in invalid_indices[:5])
        suffix = "..." if len(invalid_indices) > 5 else ""
        return unavailable_logprob_summary(
            f"Missing or invalid token logprob at content index(es): {preview}{suffix}.",
            token_count=token_count,
        )

    sequence_sum = sum(values)
    return {
        "logprob_available": True,
        "sequence_logprob_sum": sequence_sum,
        "sequence_logprob_mean": sequence_sum / token_count,
        "sequence_token_count": token_count,
        "logprob_unavailable_reason": None,
    }


def select_logprob_summary(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy only persisted logprob summary fields from a result mapping."""
    return {key: source.get(key) for key in LOGPROB_SUMMARY_KEYS}


def read_text_with_encoding_fallback(path, encodings: Sequence[str] = ("utf-8", "gb18030", "cp1252", "latin-1")):
    # 部分 LaTeX 中间文件来自 GBK/GB18030 环境，固定用 UTF-8 读取会因破折号、引号等字符失败。
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error

def load_jsonl(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith('.jsonl'):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file format: {path}")

def save_jsonl(data, path):
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith('.jsonl'):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported path: {path}")


def candidate_run_metadata(args, dataset, output_dir):
    """Build metadata for a seeded candidate run without changing old runs."""
    seed = getattr(args, "seed", None)
    if seed is None:
        return None
    return {
        "dataset": dataset,
        "model_name": getattr(args, "model_name", None),
        "url": getattr(args, "url", None),
        "table_format": getattr(args, "table_format", None),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "seed": seed,
        "base_seed": getattr(args, "base_seed", None),
        "sample_index": getattr(args, "sample_index", None),
        "candidate_id": getattr(args, "candidate_id", None),
        "save_logprobs": bool(getattr(args, "save_logprobs", False)),
        "output_dir": os.path.abspath(output_dir),
    }


def save_candidate_run_metadata(args, dataset, output_dir):
    metadata = candidate_run_metadata(args, dataset, output_dir)
    if metadata is not None:
        save_jsonl(metadata, os.path.join(output_dir, "run_metadata.json"))
    return metadata

def _chat_completions_url(url):
    base_url = os.environ.get("BASE_URL")
    if not base_url:
        base_url = url or "localhost:8000"

    base_url = base_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    if base_url.endswith("/chat/completions"):
        return base_url
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return f"{base_url}/chat/completions"


def _auth_headers():
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _api_error_message(response):
    if not isinstance(response, Mapping):
        return "API returned a response without choices"
    error = response.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or error)
    if error:
        return str(error)
    return "API returned a response without choices"


def model_resp(
    url, 
    messages, 
    model_params=None,
    model_name=None,
):  
    model_key = (model_name or "").lower()
    skip_token_flag = model_key.startswith("gemma-3")

    parameters = dict(
        messages=messages,
        model=model_name or "model",
        max_tokens=4096,
        temperature=0.95,
        top_p=0.6,
        skip_special_tokens=True if skip_token_flag else False,
        spaces_between_special_tokens=False,
        chat_template_kwargs={"enable_thinking": False},
    )

    if model_params:
        parameters.update(model_params)

    requested_logprobs = bool(parameters.get("logprobs"))

    for _ in range(3):
        try:
            resp = requests.post(
                url=_chat_completions_url(url),
                json=parameters,
                headers=_auth_headers(),
                verify=False,
            ).json()

            if resp.get("choices"):
                choice = resp["choices"][0]
                if requested_logprobs and choice.get("logprobs") is None:
                    choice = dict(choice)
                    choice["_logprob_unavailable_reason"] = (
                        "Model API response omitted choice.logprobs."
                    )
                return choice

            if requested_logprobs:
                fallback_parameters = dict(parameters)
                fallback_parameters.pop("logprobs", None)
                fallback_parameters.pop("top_logprobs", None)
                fallback_resp = requests.post(
                    url=_chat_completions_url(url),
                    json=fallback_parameters,
                    headers=_auth_headers(),
                    verify=False,
                ).json()
                if fallback_resp.get("choices"):
                    choice = dict(fallback_resp["choices"][0])
                    choice["_logprob_unavailable_reason"] = (
                        "Logprobs request was rejected; generation retried without "
                        f"logprobs: {_api_error_message(resp)}"
                    )
                    return choice
        except:
            import traceback
            traceback.print_exc()
            pass
        
    return None
