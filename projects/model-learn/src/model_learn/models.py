from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class StreamMetrics(BaseModel):
    ttft_ms: float = Field(0.0, description="Time to first token in milliseconds")
    total_latency_ms: float = Field(0.0, description="Total request latency in milliseconds")
    tokens_per_second: float = Field(0.0, description="Tokens generated per second after first token")
    estimated_cost_usd: float = Field(0.0, description="Estimated total cost in USD")


class ExecutionResult(BaseModel):
    model: str
    stream: bool
    full_content: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    metrics: StreamMetrics = Field(default_factory=StreamMetrics)
    system_fingerprint: str | None = None
