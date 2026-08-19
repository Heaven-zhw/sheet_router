import json
import os
import requests
from typing import Sequence


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

    for _ in range(3):
        try:
            resp = requests.post(
                url=_chat_completions_url(url),
                json=parameters,
                headers=_auth_headers(),
                verify=False,
            ).json()

            if 'choices' in resp:
                resp = resp['choices'][0]
                return resp
        except:
            import traceback
            traceback.print_exc()
            pass
        
    return None
