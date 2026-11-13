import json
from typing import Any
import httpx
from pydantic import ValidationError

from src.config import settings
from src.exceptions import (
    ApiClientError,
    ApiParseError,
    ApiResponseError,
    ApiTimeoutError,
)
from src.models import Post


async def fetch_post(
    post_id: int,
    base_url: str | None = None,
    timeout: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> Post:
    """
    Fetch a single post from the upstream API, with explicit timeout and error boundaries.
    
    Accepts an optional `client` for connection pooling or test mocking.
    """
    resolved_base_url = (base_url or settings.api_base_url).rstrip("/")
    resolved_timeout = timeout if timeout is not None else settings.api_timeout_seconds
    url = f"{resolved_base_url}/posts/{post_id}"

    # Helper to perform the actual request on an AsyncClient instance
    async def _do_request(c: httpx.AsyncClient) -> Post:
        try:
            response = await c.get(url, timeout=resolved_timeout)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ApiTimeoutError(f"Request to {url} timed out after {resolved_timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise ApiResponseError(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise ApiClientError(f"Network error while calling {url}: {exc}") from exc

        # Parse JSON and validate with Pydantic model
        try:
            payload: Any = response.json()
            return Post.model_validate(payload)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise ApiParseError(f"Failed to parse or validate response from {url}: {exc}") from exc

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient() as managed_client:
        return await _do_request(managed_client)
