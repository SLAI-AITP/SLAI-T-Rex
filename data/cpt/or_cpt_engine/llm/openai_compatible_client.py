from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from or_cpt_engine.schemas.common import LLMConfig, LLMStageConfig
from or_cpt_engine.utils.logging import log_model_failure_case

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("LLM response did not contain a JSON object")


@dataclass(slots=True)
class ResolvedLLMEndpoint:
    name: str
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_sec: int = 120
    trust_env: bool = False


@dataclass(slots=True)
class _AsyncEndpointSession:
    endpoint: ResolvedLLMEndpoint
    client: httpx.AsyncClient
    consecutive_failures: int = 0
    unhealthy_until: float = 0.0


class SimpleOpenAICompatibleClient:
    def __init__(self, llm_config: LLMConfig, stage_config: LLMStageConfig):
        provider = llm_config.providers[stage_config.provider]
        self.endpoints = _load_endpoints(provider)
        self.timeout = provider.timeout_sec
        self.max_retries = provider.max_retries
        self.stage_config = stage_config
        self._cursor = 0

    def chat(self, prompt: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        last_error: Exception | None = None
        total_attempts = max(1, (self.max_retries + 1) * len(self.endpoints))
        for attempt in range(total_attempts):
            endpoint = self._next_endpoint()
            try:
                started = time.perf_counter()
                response = httpx.post(
                    f"{endpoint.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {endpoint.api_key}"},
                    json={
                        "model": endpoint.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.stage_config.temperature,
                        "max_tokens": self.stage_config.max_tokens,
                    },
                    timeout=endpoint.timeout_sec,
                    trust_env=endpoint.trust_env,
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                return {
                    "request_id": request_id,
                    "content": content,
                    "raw_response": payload,
                    "latency_sec": time.perf_counter() - started,
                    "model": endpoint.model,
                    "endpoint": endpoint.name,
                    "attempt": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(2 ** (attempt // max(len(self.endpoints), 1)), 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def _next_endpoint(self) -> ResolvedLLMEndpoint:
        endpoint = self.endpoints[self._cursor % len(self.endpoints)]
        self._cursor += 1
        return endpoint


class AsyncOpenAICompatibleClient:
    def __init__(
        self,
        llm_config: LLMConfig,
        stage_config: LLMStageConfig,
        *,
        concurrency_per_endpoint: int | None = None,
    ):
        provider = llm_config.providers[stage_config.provider]
        self.provider = provider
        self.stage_config = stage_config
        self.max_retries = max(0, int(provider.max_retries))
        self.endpoint_failure_threshold = max(1, int(provider.endpoint_failure_threshold))
        self.endpoint_cooldown_seconds = max(1, int(provider.endpoint_cooldown_seconds))
        self.per_endpoint_concurrency = max(1, int(concurrency_per_endpoint or provider.concurrency_per_endpoint))
        self._sessions = [
            _AsyncEndpointSession(
                endpoint=endpoint,
                client=httpx.AsyncClient(timeout=endpoint.timeout_sec, trust_env=endpoint.trust_env),
            )
            for endpoint in _load_endpoints(provider)
        ]
        self._slot_queue: asyncio.Queue[_AsyncEndpointSession] = asyncio.Queue()
        self._rebuild_slot_queue()

    async def close(self) -> None:
        for session in self._sessions:
            await session.client.aclose()

    def endpoint_count(self) -> int:
        return len(self._sessions)

    def total_concurrency_capacity(self) -> int:
        return max(1, self.endpoint_count() * self.per_endpoint_concurrency)

    async def chat(
        self,
        prompt: str,
        *,
        log_context: dict[str, str] | None = None,
        stage_config: LLMStageConfig | None = None,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        last_error: Exception | None = None
        total_attempts = max(1, (self.max_retries + 1) * self.endpoint_count())
        first_attempt_started = time.perf_counter()
        active_stage_config = stage_config or self.stage_config
        for attempt in range(total_attempts):
            session = await self._acquire_endpoint()
            started = time.perf_counter()
            response: httpx.Response | None = None
            try:
                response = await session.client.post(
                    f"{session.endpoint.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {session.endpoint.api_key}"},
                    json={
                        "model": session.endpoint.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": active_stage_config.temperature,
                        "max_tokens": active_stage_config.max_tokens,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage") or {}
                content = payload["choices"][0]["message"]["content"]
                latency = time.perf_counter() - started
                self._mark_endpoint_success(session)
                self._log_model_attempt(request_id, attempt, session, prompt, content, latency, "ok",
                                        http_status=response.status_code, usage=usage, log_context=log_context)
                self._log_model_summary(request_id, "success",
                                        total_latency=time.perf_counter() - first_attempt_started,
                                        retry_count=attempt, session=session, log_context=log_context)
                return {
                    "request_id": request_id,
                    "content": content,
                    "raw_response": payload,
                    "latency_sec": latency,
                    "model": session.endpoint.model,
                    "endpoint": session.endpoint.name,
                    "attempt": attempt + 1,
                    "http_status": response.status_code,
                    "usage": usage,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = response.status_code if response is not None else None
                latency = time.perf_counter() - started
                self._mark_endpoint_failure(session, exc, status_code)
                self._log_model_attempt(request_id, attempt, session, prompt, "", latency, "error",
                                        http_status=status_code, error=str(exc)[:500], log_context=log_context)
                if attempt < total_attempts - 1:
                    await asyncio.sleep(min(2 ** (attempt // max(self.endpoint_count(), 1)), 5))
            finally:
                self._slot_queue.put_nowait(session)
        self._log_model_summary(request_id, "failed",
                                total_latency=time.perf_counter() - first_attempt_started,
                                retry_count=total_attempts, error=str(last_error)[:500], log_context=log_context)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    @staticmethod
    def _log_model_attempt(
        request_id: str,
        attempt: int,
        session: _AsyncEndpointSession,
        prompt: str,
        content: str,
        latency_sec: float,
        status: str,
        *,
        http_status: int | None = None,
        usage: dict[str, Any] | None = None,
        error: str | None = None,
        log_context: dict[str, str] | None = None,
    ) -> None:
        ctx = log_context or {}
        fields = _format_model_log_fields(
            request_id=request_id,
            attempt=attempt + 1,
            model=session.endpoint.model,
            endpoint_name=session.endpoint.name,
            endpoint_base_url=session.endpoint.base_url,
            latency_sec=latency_sec,
            status=status,
            http_status=http_status,
            usage=usage,
            prompt_chars=len(prompt),
            completion_chars=len(content) if content else 0,
            error=error,
            **ctx,
        )
        logging.getLogger("or_cpt_engine.model_call").info("model_api_call %s", fields)
        if status == "error":
            log_model_failure_case(
                request_id=request_id,
                attempt=attempt + 1,
                model=session.endpoint.model,
                endpoint_name=session.endpoint.name,
                endpoint_base_url=session.endpoint.base_url,
                latency_seconds=round(float(latency_sec), 3),
                status=status,
                http_status=http_status,
                prompt_chars=len(prompt),
                completion_chars=len(content) if content else 0,
                error_message=error or "",
                error_type=_classify_model_error_for_log(error, http_status, latency_sec),
                **ctx,
            )

    @staticmethod
    def _log_model_summary(
        request_id: str,
        status: str,
        *,
        total_latency: float,
        retry_count: int = 0,
        error: str | None = None,
        log_context: dict[str, str] | None = None,
        session: _AsyncEndpointSession | None = None,
    ) -> None:
        ctx = log_context or {}
        parts = [f"request_id={request_id}", f"status={status}"]
        if "prompt_name" in ctx:
            parts.append(f"prompt_name={ctx['prompt_name']}")
        if "instance_id" in ctx:
            parts.append(f"instance_id={ctx['instance_id']}")
        if session is not None:
            parts.append(f"endpoint_name={session.endpoint.name}")
            parts.append(f"endpoint_base_url={session.endpoint.base_url}")
        parts.append(f"total_latency_seconds={total_latency:.3f}")
        parts.append(f"retry_count={retry_count}")
        if error:
            parts.append(f"error={error}")
        logging.getLogger("or_cpt_engine.model_call").info("model_call_summary %s", " ".join(parts))

    def _rebuild_slot_queue(self) -> None:
        self._slot_queue = asyncio.Queue(maxsize=self.total_concurrency_capacity())
        for _ in range(self.per_endpoint_concurrency):
            for session in self._sessions:
                self._slot_queue.put_nowait(session)

    async def _acquire_endpoint(self) -> _AsyncEndpointSession:
        deferred: list[_AsyncEndpointSession] = []
        while True:
            queue_size = self._slot_queue.qsize()
            soonest_recovery_at: float | None = None
            for _ in range(queue_size):
                session = await self._slot_queue.get()
                now = time.monotonic()
                if session.unhealthy_until <= now:
                    for skipped in deferred:
                        self._slot_queue.put_nowait(skipped)
                    return session
                deferred.append(session)
                if soonest_recovery_at is None or session.unhealthy_until < soonest_recovery_at:
                    soonest_recovery_at = session.unhealthy_until

            for skipped in deferred:
                self._slot_queue.put_nowait(skipped)
            deferred.clear()

            if soonest_recovery_at is None:
                await asyncio.sleep(0.05)
                continue
            await asyncio.sleep(max(0.05, min(soonest_recovery_at - time.monotonic(), 1.0)))

    def _mark_endpoint_success(self, session: _AsyncEndpointSession) -> None:
        was_unhealthy = session.unhealthy_until > 0.0
        had_failures = session.consecutive_failures > 0
        session.consecutive_failures = 0
        session.unhealthy_until = 0.0
        if was_unhealthy or had_failures:
            logger.info(
                "or_cpt_model_endpoint_health endpoint_name=%s endpoint_base_url=%s status=recovered",
                session.endpoint.name,
                session.endpoint.base_url,
            )

    def _mark_endpoint_failure(
        self,
        session: _AsyncEndpointSession,
        error: Exception,
        http_status: int | None,
    ) -> None:
        if not self._is_retryable_endpoint_failure(error, http_status):
            return
        session.consecutive_failures += 1
        if session.consecutive_failures < self.endpoint_failure_threshold:
            return
        session.consecutive_failures = 0
        session.unhealthy_until = time.monotonic() + self.endpoint_cooldown_seconds
        logger.warning(
            "or_cpt_model_endpoint_health endpoint_name=%s endpoint_base_url=%s status=quarantined cooldown_seconds=%s reason=%s",
            session.endpoint.name,
            session.endpoint.base_url,
            self.endpoint_cooldown_seconds,
            str(error).replace("\n", "\\n")[:200],
        )

    @staticmethod
    def _is_retryable_endpoint_failure(error: Exception, http_status: int | None) -> bool:
        if http_status is not None and http_status >= 500:
            return True
        return isinstance(error, (httpx.TransportError, httpx.TimeoutException))


def _load_endpoints(provider) -> list[ResolvedLLMEndpoint]:
    if provider.endpoint_pool_path:
        pool_path = Path(provider.endpoint_pool_path)
        with pool_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        endpoints = payload.get("endpoints") or []
        if not endpoints:
            raise RuntimeError(f"LLM endpoint pool is empty: {pool_path}")
        resolved: list[ResolvedLLMEndpoint] = []
        for index, endpoint in enumerate(endpoints, start=1):
            base_url = _normalize_base_url(str(endpoint.get("base_url") or ""))
            resolved.append(
                ResolvedLLMEndpoint(
                    name=str(endpoint.get("name") or f"endpoint_{index}"),
                    base_url=base_url,
                    model=str(endpoint.get("model_name") or endpoint.get("model") or provider.model),
                    api_key=str(endpoint.get("api_key") or "EMPTY"),
                    timeout_sec=int(endpoint.get("timeout_seconds") or provider.timeout_sec),
                    trust_env=bool(endpoint.get("trust_env") if endpoint.get("trust_env") is not None else provider.trust_env),
                )
            )
        return resolved

    base_url = os.getenv(provider.base_url_env)
    api_key = os.getenv(provider.api_key_env, "EMPTY")
    if not base_url:
        raise RuntimeError(f"missing LLM base URL env var: {provider.base_url_env}")
    return [
        ResolvedLLMEndpoint(
            name="env_default",
            base_url=_normalize_base_url(base_url),
            model=provider.model,
            api_key=api_key,
            timeout_sec=provider.timeout_sec,
            trust_env=provider.trust_env,
        )
    ]


def _format_model_log_fields(
    request_id: str,
    attempt: int,
    model: str,
    endpoint_name: str,
    endpoint_base_url: str,
    latency_sec: float,
    status: str,
    *,
    http_status: int | None = None,
    usage: dict[str, Any] | None = None,
    prompt_chars: int = 0,
    completion_chars: int = 0,
    error: str | None = None,
    instance_id: str = "",
    prompt_name: str = "",
) -> str:
    fields = [
        f"request_id={request_id}",
        f"attempt={attempt}",
        f"model={model}",
        f"endpoint_name={endpoint_name}",
        f"endpoint_base_url={endpoint_base_url}",
        f"latency_seconds={latency_sec:.3f}",
        f"status={status}",
    ]
    if http_status is not None:
        fields.append(f"http_status={http_status}")
    if instance_id:
        fields.append(f"instance_id={instance_id}")
    if prompt_name:
        fields.append(f"prompt_name={prompt_name}")
    fields.append(f"prompt_chars={prompt_chars}")
    fields.append(f"completion_chars={completion_chars}")
    usage = usage or {}
    fields.append(f"prompt_tokens={usage.get('prompt_tokens', 'n/a')}")
    fields.append(f"completion_tokens={usage.get('completion_tokens', 'n/a')}")
    fields.append(f"total_tokens={usage.get('total_tokens', 'n/a')}")
    if error:
        fields.append(f"error={error}")
    return " ".join(fields)


def _classify_model_error_for_log(error: str | None, http_status: int | None, latency_sec: float) -> str:
    text = (error or "").lower()
    if "timeout" in text or "timed out" in text or "readtimeout" in text:
        return "TIMEOUT"
    if "rate limit" in text or http_status == 429:
        return "RATE_LIMIT"
    if http_status is not None and http_status >= 500:
        return "HTTP_5XX"
    if http_status is not None and http_status >= 400:
        return "HTTP_4XX"
    if latency_sec >= 120.0:
        return "LIKELY_TIMEOUT"
    if "connection" in text or "connect" in text or "network" in text:
        return "CONNECTIVITY"
    if "endpoint_unavailable" in text or "unavailable" in text:
        return "ENDPOINT_UNAVAILABLE"
    return "UNKNOWN"


def is_retryable_llm_error(exc: Exception) -> bool:
    """Return True when an LLM-call exception should trigger a requeue rather than a permanent rejection.

    This mirrors cpt_cleaner's RequeueDirective semantics: transient infrastructure
    failures (network, 5xx, timeout) are retryable; permanent failures (4xx, bad
    response format) are not.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    message = str(exc).lower()
    if "llm request failed after retries" in message:
        return True
    return False


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip()
    if not normalized:
        raise ValueError("endpoint base_url must not be empty")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    normalized = normalized.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized
