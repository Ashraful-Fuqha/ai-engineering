# FastAPI & Async Python Microservice Foundations

An asynchronous Python microservice and CLI client that fetches external JSON resources, validates payloads using Pydantic schemas, and persists structured data to a local SQLite database using `aiosqlite`.

## Features
- **Async HTTP Client**: Built on `httpx.AsyncClient` with custom error hierarchy.
- **Data Validation**: Strict Pydantic models for domain models.
- **Asynchronous Persistence**: Non-blocking SQLite storage via `aiosqlite`.
- **Comprehensive Testing**: Offline test suite using `pytest-asyncio` and `respx` mocking.

## Quick Start

```bash
# Run tests
uv run pytest -v

# Fetch and store post #1
uv run python main.py 1
```

