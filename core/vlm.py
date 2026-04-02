"""VLM 调用封装：支持纯文本和带图片的调用，以及 JSON 解析重试。"""

import json
import os
import time
from typing import Optional

import httpx
from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME, VLM_MAX_RETRIES, VLM_TIMEOUT, RATE_LIMIT_DELAY

# VLM API 可直连不走代理，但 SerpAPI / Jina 需要代理。
# 用 trust_env=False 让 httpx 忽略代理环境变量，仅对 VLM client 生效。
_vlm_http = httpx.Client(trust_env=False)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=_vlm_http)


def call_vlm(
    system_prompt: str,
    user_content: str | list,
    image_b64: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.5,
    max_retries: int = VLM_MAX_RETRIES,
) -> str:
    """调用 VLM，返回原始文本。支持纯文本和图文混合。"""
    # 构造 user message
    if image_b64 is not None:
        if isinstance(user_content, str):
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": user_content},
                ],
            }
        else:
            user_msg = {"role": "user", "content": user_content}
    else:
        user_msg = {"role": "user", "content": user_content}

    messages = [{"role": "system", "content": system_prompt}, user_msg]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=VLM_TIMEOUT,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            time.sleep(RATE_LIMIT_DELAY)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"VLM 调用 {max_retries} 次均失败: {e}") from e
    return ""


def extract_json(raw: str) -> Optional[dict]:
    """从可能包含 markdown 包裹的文本中提取 JSON 对象。"""
    # 先尝试去除 markdown 代码块
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n")
        cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 回退：花括号匹配
    try:
        start = raw.index("{")
        depth, i = 0, start
        while i < len(raw):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start: i + 1])
            i += 1
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def call_vlm_json(
    system_prompt: str,
    user_content: str | list,
    image_b64: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.5,
    max_attempts: int = 3,
) -> Optional[dict]:
    """调用 VLM 并解析 JSON 返回，失败时自动重试。"""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_vlm(system_prompt, user_content, image_b64, max_tokens, temperature)
        except RuntimeError:
            continue
        if not raw:
            continue
        data = extract_json(raw)
        if data is not None:
            return data
    return None
