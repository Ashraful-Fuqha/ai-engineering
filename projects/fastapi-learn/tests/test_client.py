import httpx
import pytest

from src.client import fetch_post
from src.db import get_post, save_post
from src.exceptions import (
    ApiParseError,
    ApiResponseError,
    ApiTimeoutError,
)
from src.models import Post


async def test_fetch_post_success(respx_mock):
    """Case 1: Upstream responds 200 with valid JSON matching our Pydantic model."""
    respx_mock.get("https://jsonplaceholder.typicode.com/posts/1").mock(
        return_value=httpx.Response(
            200,
            json={
                "userId": 99,
                "id": 1,
                "title": "Clean Architecture in Python",
                "body": "Pydantic + HTTPX + SQLite is great.",
            },
        )
    )

    post = await fetch_post(1)

    assert isinstance(post, Post)
    assert post.id == 1
    assert post.user_id == 99
    assert post.title == "Clean Architecture in Python"


async def test_fetch_post_timeout(respx_mock):
    """Case 2: Upstream times out. Should raise controlled ApiTimeoutError."""
    respx_mock.get("https://jsonplaceholder.typicode.com/posts/1").mock(
        side_effect=httpx.TimeoutException("Connection timed out")
    )

    with pytest.raises(ApiTimeoutError) as exc_info:
        await fetch_post(1)

    assert "timed out" in str(exc_info.value).lower()


async def test_fetch_post_non_2xx(respx_mock):
    """Case 3: Upstream returns 404 or 500. Should raise ApiResponseError with status code."""
    respx_mock.get("https://jsonplaceholder.typicode.com/posts/404").mock(
        return_value=httpx.Response(404, text="Post Not Found")
    )

    with pytest.raises(ApiResponseError) as exc_info:
        await fetch_post(404)

    assert exc_info.value.status_code == 404
    assert "Post Not Found" in exc_info.value.detail


async def test_fetch_post_malformed_json(respx_mock):
    """Case 4: Upstream returns 200 OK, but broken non-JSON text."""
    respx_mock.get("https://jsonplaceholder.typicode.com/posts/1").mock(
        return_value=httpx.Response(200, text="<html><body>Bad Gateway</body></html>")
    )

    with pytest.raises(ApiParseError):
        await fetch_post(1)


async def test_fetch_post_schema_validation_failure(respx_mock):
    """Case 5: Upstream returns valid JSON, but missing required fields or wrong schema."""
    respx_mock.get("https://jsonplaceholder.typicode.com/posts/1").mock(
        return_value=httpx.Response(
            200,
            json={"title": "Missing ID and userId"},
        )
    )

    with pytest.raises(ApiParseError):
        await fetch_post(1)


async def test_sqlite_persistence_roundtrip(tmp_path):
    """Case 6: Persist a Post into SQLite and read it back."""
    test_db = str(tmp_path / "test_posts.db")
    sample_post = Post(id=42, userId=7, title="Testing DB", body="Persisted successfully")

    await save_post(sample_post, db_path=test_db)
    loaded_post = await get_post(42, db_path=test_db)

    assert loaded_post is not None
    assert loaded_post.id == 42
    assert loaded_post.user_id == 7
    assert loaded_post.title == "Testing DB"
