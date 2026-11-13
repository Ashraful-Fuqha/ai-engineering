import httpx
import pytest
import respx

from model_learn.client import LlmApiError, LlmClient
from model_learn.models import ChatCompletionRequest, Message
from model_learn.pricing import calculate_cost
from model_learn.redact import redact_sensitive_data


@pytest.fixture
def mock_client():
    return LlmClient(
        api_key="test-api-key-12345",
        base_url="https://api.mockllm.com/v1",
    )


async def test_non_streaming_success(mock_client, respx_mock):
    """Test non-streaming mode parsing, usage, and metrics."""
    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello world!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )

    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="Hi")],
        stream=False,
    )

    result = await mock_client.execute(req)

    assert result.stream is False
    assert result.full_content == "Hello world!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.metrics.total_latency_ms > 0
    assert result.metrics.estimated_cost_usd > 0


async def test_streaming_sse_parsing(mock_client, respx_mock):
    """Test SSE streaming parser with multi-chunk delta reconstruction."""
    sse_body = (
        'data: {"choices": [{"delta": {"content": "Async "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "streaming "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "works!"}, "finish_reason": "stop"}]}\n\n'
        'data: {"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}\n\n'
        'data: [DONE]\n\n'
    )

    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=sse_body,
        )
    )

    collected_tokens = []

    def on_chunk(t: str):
        collected_tokens.append(t)

    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="Test stream")],
        stream=True,
    )

    result = await mock_client.execute(req, on_chunk=on_chunk)

    assert result.stream is True
    assert result.full_content == "Async streaming works!"
    assert "".join(collected_tokens) == "Async streaming works!"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    assert result.finish_reason == "stop"
    assert result.metrics.ttft_ms >= 0


async def test_invalid_credentials_error_401(mock_client, respx_mock):
    """Test controlled handling of invalid API key."""
    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "API key not valid. Please pass a valid API key."}},
        )
    )

    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="Hi")],
        stream=False,
    )

    with pytest.raises(LlmApiError) as exc_info:
        await mock_client.execute(req)

    assert exc_info.value.status_code == 401
    assert "Invalid API Key" in exc_info.value.message


async def test_rate_limit_error_429(mock_client, respx_mock):
    """Test controlled handling of rate limit or quota exceeded."""
    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Resource has been exhausted (e.g. check quota)."}},
        )
    )

    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="Hi")],
        stream=False,
    )

    with pytest.raises(LlmApiError) as exc_info:
        await mock_client.execute(req)

    assert exc_info.value.status_code == 429
    assert "Quota Exceeded" in exc_info.value.message


def test_credential_redaction():
    """Ensure sensitive credentials and tokens are redacted from telemetry logs."""
    raw_telemetry = {
        "headers": {
            "Authorization": "Bearer AIzaSySecret123456",
            "Content-Type": "application/json",
        },
        "config": {
            "api_key": "AIzaSySecret123456",
            "gemini_api_key": "AIzaSySecret123456",
            "endpoint_url": "https://api.example.com?key=AIzaSySecret123456&foo=bar",
        },
        "public_data": "Normal message",
    }

    sanitized = redact_sensitive_data(raw_telemetry)

    assert sanitized["headers"]["Authorization"] == "[REDACTED]"
    assert sanitized["config"]["api_key"] == "[REDACTED]"
    assert sanitized["config"]["gemini_api_key"] == "[REDACTED]"
    assert "AIzaSySecret123456" not in str(sanitized)
    assert sanitized["config"]["endpoint_url"] == "https://api.example.com?key=[REDACTED]&foo=bar"
    assert sanitized["public_data"] == "Normal message"


def test_cost_calculation():
    """Verify Gemini 2.5 Flash token pricing math."""
    cost = calculate_cost("gemini-2.5-flash", 1_000_000, 1_000_000, cached_tokens=0)
    assert pytest.approx(cost, rel=1e-4) == 0.375
