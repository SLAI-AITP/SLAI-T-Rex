from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    timeout_seconds: float = 180
    disable_proxy: bool = False


class ChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        proxies = {"http": None, "https": None} if self.config.disable_proxy else None
        started = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=self.config.timeout_seconds,
            proxies=proxies,
        )
        latency = time.time() - started
        response.raise_for_status()
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": content, "raw": payload, "latency_seconds": latency, "usage": payload.get("usage")}


def config_from_dict(data: dict[str, Any], *, base_url: str | None = None) -> LLMConfig:
    key_env = str(data.get("api_key_env") or "LLM_API_KEY")
    return LLMConfig(
        base_url=str(base_url or data.get("base_url") or "http://localhost:8000/v1"),
        model=str(data.get("model") or "your-model"),
        api_key=os.environ.get(key_env, ""),
        temperature=float(data.get("temperature", 0.7)),
        top_p=float(data.get("top_p", 0.9)),
        max_tokens=int(data.get("max_tokens", 4096)),
        timeout_seconds=float(data.get("timeout_seconds", 180)),
        disable_proxy=bool(data.get("disable_proxy", False)),
    )


def base_urls_from_dict(data: dict[str, Any]) -> list[str]:
    values = data.get("base_urls")
    if isinstance(values, list):
        urls = [str(value).strip() for value in values if str(value).strip()]
    elif isinstance(values, str):
        urls = [part.strip() for part in values.split(",") if part.strip()]
    else:
        urls = []
    if not urls:
        urls = [str(data.get("base_url") or "http://localhost:8000/v1")]
    return urls
