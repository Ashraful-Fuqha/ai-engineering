import argparse
import asyncio
import sys
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.client import fetch_post
from src.db import get_post, save_post
from src.exceptions import ApiClientError, ApiParseError, ApiResponseError, ApiTimeoutError
from src.models import Post

app = FastAPI(title="HTTP + Pydantic Mini-Service")


class SyncResponse(BaseModel):
    message: str
    post: Post


@app.get("/")
async def root():
    return {"status": "ok", "message": "Service is running"}


@app.post("/posts/{post_id}/sync", response_model=SyncResponse)
async def sync_post_endpoint(post_id: int):
    """
    Fetch post from external API, validate with Pydantic, and persist to SQLite.
    Maps domain errors into clean HTTP status codes.
    """
    try:
        post = await fetch_post(post_id)
        await save_post(post)
        return SyncResponse(message="Post fetched and persisted successfully", post=post)
    except ApiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        )
    except ApiResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream returned error: {exc.status_code}",
        )
    except ApiParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream sent invalid data: {exc}",
        )
    except ApiClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to communicate with external API: {exc}",
        )


@app.get("/posts/{post_id}", response_model=Post)
async def get_persisted_post(post_id: int):
    """Read persisted post from SQLite."""
    post = await get_post(post_id)
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found in database")
    return post


async def run_cli(post_id: int):
    """CLI runner to fetch and persist a post without starting the HTTP server."""
    print(f"Fetching post {post_id} from upstream API...")
    try:
        post = await fetch_post(post_id)
        print(f"Validated with Pydantic:\n  {post}")
        await save_post(post)
        print(f"Successfully persisted post {post.id} to SQLite!")
    except ApiTimeoutError as e:
        print(f"Timeout Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ApiResponseError as e:
        print(f"API Error (Status {e.status_code}): {e.detail}", file=sys.stderr)
        sys.exit(1)
    except ApiParseError as e:
        print(f"Parse/Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and persist a post")
    parser.add_argument("post_id", type=int, nargs="?", default=1, help="Post ID to fetch (default: 1)")
    args = parser.parse_args()
    asyncio.run(run_cli(args.post_id))
