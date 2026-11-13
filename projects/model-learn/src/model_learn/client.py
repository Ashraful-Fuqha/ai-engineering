import json
import time
from typing import Any, Callable
import httpx

from .config import settings
from .models import (
    ChatCompletionRequest,
    ExecutionResult,
    StreamMetrics,
    TokenUsage,
)
from .pricing import calculate_cost


class LlmApiError(Exception):
    """Base exception for LLM API errors."""
    def __init__(self, status_code: int, message: str, body: Any = None):
        super().__init__(f"LLM API Error [{status_code}]: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body


class LlmClient:
    """Raw HTTP client for OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.default_timeout_seconds

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def execute(
        self,
        request: ChatCompletionRequest,
        on_chunk: Callable[[str], None] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> ExecutionResult:
        """
        Execute request in either streaming or non-streaming mode.
        If on_chunk is provided and request.stream is True, on_chunk is called for each token.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        if request.stream:
            # Ensure stream_options requests token usage in final SSE frame
            if request.stream_options is None:
                request.stream_options = {"include_usage": True}

        payload = request.model_dump(exclude_none=True)

        # Provider Compatibility Boundary:
        # Google Gemini's OpenAI gateway strictly validates JSON fields and rejects 'seed'
        if "googleapis.com" in self.base_url and "seed" in payload:
            payload.pop("seed")

        if client is not None:
            return await self._run(client, url, headers, payload, request.stream, on_chunk)

        async with httpx.AsyncClient(timeout=self.timeout) as managed_client:
            return await self._run(managed_client, url, headers, payload, request.stream, on_chunk)

    async def _run(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        is_stream: bool,
        on_chunk: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        if is_stream:
            return await self._execute_streaming(client, url, headers, payload, on_chunk)
        return await self._execute_non_streaming(client, url, headers, payload)

    async def _execute_non_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> ExecutionResult:
        t_start = time.perf_counter()
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                self._handle_error_response(response)
        except httpx.HTTPStatusError as exc:
            self._handle_error_response(exc.response)

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0

        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason")
        system_fingerprint = data.get("system_fingerprint")

        raw_usage = data.get("usage", {})
        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
        total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
        cached_tokens = raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        # Fallback estimation if usage is missing
        if prompt_tokens == 0 and content:
            completion_tokens = max(1, len(content) // 4)

        model = data.get("model", payload.get("model", ""))
        cost = calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
        tps = (completion_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        return ExecutionResult(
            model=model,
            stream=False,
            full_content=content,
            finish_reason=finish_reason,
            system_fingerprint=system_fingerprint,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
            ),
            metrics=StreamMetrics(
                ttft_ms=total_latency_ms,
                total_latency_ms=round(total_latency_ms, 2),
                tokens_per_second=round(tps, 2),
                estimated_cost_usd=cost,
            ),
        )

    async def _execute_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        on_chunk: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        t_start = time.perf_counter()
        t_first_token: float | None = None
        full_content_parts: list[str] = []
        finish_reason: str | None = None
        system_fingerprint: str | None = None
        raw_usage: dict[str, Any] = {}
        model = payload.get("model", "")

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error_response(response)

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    
                    if not line.startswith("data:"):
                        continue
                    
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "model" in chunk_json:
                        model = chunk_json["model"]
                    if "system_fingerprint" in chunk_json:
                        system_fingerprint = chunk_json["system_fingerprint"]

                    if "usage" in chunk_json and chunk_json["usage"]:
                        raw_usage = chunk_json["usage"]

                    choices = chunk_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "finish_reason" in choices[0] and choices[0]["finish_reason"]:
                            finish_reason = choices[0]["finish_reason"]

                        content_delta = delta.get("content")
                        if content_delta:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()

                            full_content_parts.append(content_delta)
                            if on_chunk:
                                on_chunk(content_delta)

        except httpx.HTTPStatusError as exc:
            self._handle_error_response(exc.response)

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        ttft_ms = ((t_first_token - t_start) * 1000.0) if t_first_token is not None else total_latency_ms

        full_content = "".join(full_content_parts)

        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
        total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
        cached_tokens = raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        if completion_tokens == 0 and full_content:
            completion_tokens = max(1, len(full_content) // 4)

        gen_time_sec = max(0.001, (total_latency_ms - ttft_ms) / 1000.0)
        tps = (completion_tokens / gen_time_sec) if completion_tokens > 0 else 0.0

        cost = calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)

        return ExecutionResult(
            model=model,
            stream=True,
            full_content=full_content,
            finish_reason=finish_reason,
            system_fingerprint=system_fingerprint,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
            ),
            metrics=StreamMetrics(
                ttft_ms=round(ttft_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                tokens_per_second=round(tps, 2),
                estimated_cost_usd=cost,
            ),
        )

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Parse structured API error payload or throw with status code."""
        try:
            body = response.json()
            error_info = body.get("error", {})
            message = error_info.get("message", response.text)
        except Exception:
            body = response.text
            message = response.text or response.reason_phrase

        if response.status_code == 401:
            raise LlmApiError(401, f"Authentication Failed (Invalid API Key): {message}", body)
        elif response.status_code == 429:
            raise LlmApiError(429, f"Rate Limit or Quota Exceeded: {message}", body)
        elif response.status_code == 400:
            raise LlmApiError(400, f"Bad Request (Context length or invalid params): {message}", body)
        else:
            raise LlmApiError(response.status_code, f"API Error [{response.status_code}]: {message}", body)
