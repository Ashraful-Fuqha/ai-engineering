# Python & FastAPI Fundamentals: Engineering Notes

> A reference guide for building robust, typed, asynchronous Python services — mapped directly to TypeScript / Next.js mental models.

---

## 1. TypeScript $\leftrightarrow$ Python Rosetta Stone

| TypeScript / Next.js Stack | Modern Python / FastAPI Stack | Core Purpose & Mental Model |
| :--- | :--- | :--- |
| **`pnpm` / `package.json`** | **`uv` / `pyproject.toml`** | Fast Rust-based package manager, manages lockfile (`uv.lock`) and virtual environment (`.venv`). |
| **`zod` (`z.object({...})`)** | **Pydantic (`BaseModel`)** | Class-based schema definition. Validates, coerces types at runtime, and handles JSON serialization/deserialization. |
| **`process.env` + `zod`** | **`pydantic-settings` (`BaseSettings`)** | Strongly-typed environment variables. Automatically reads `.env` and system env with fallback defaults. |
| **`fetch` / `axios`** | **`httpx.AsyncClient`** | Modern HTTP client supporting `async/await` and HTTP/2. Managed via Python context managers (`async with`). |
| **`Promise` / `async/await`** | **`asyncio` Coroutines / `async def`** | Single-threaded event loop cooperative multitasking. Never run blocking sync code inside `async def`. |
| **Vitest / Jest + MSW** | **`pytest` + `pytest-asyncio` + `respx`** | Test runner with fixture injection. `respx` intercepts HTTPX calls at the transport level (identical to MSW). |
| **Better-SQLite3 / Drizzle** | **`aiosqlite`** | Non-blocking asynchronous SQLite bindings with parameterized queries. |
| **Express / Next.js Route Handlers**| **FastAPI (`@app.get`, `@app.post`)** | Modern ASGI web framework with automatic OpenAPI/Swagger generation and Pydantic validation out of the box. |

---

## 2. Core Concepts Mastered

### A. Python Packaging & Namespaces
- **`pyproject.toml`**: The modern Python standard (PEP 518/621) replacing legacy `setup.py` and `requirements.txt`.
- **`__init__.py`**: Turns a directory into an explicit Python package so modules can be cleanly imported across folders (e.g. `from src.client import fetch_post`).
- **Virtual Environment (`.venv`)**: Isolates project dependencies per directory instead of installing globally on the host machine.

### B. Async/Await & Event Loop Dynamics
- Python's `asyncio` runs an **Event Loop** on a single thread.
- `await` yields control back to the event loop while waiting for network I/O or disk operations.
- **Rule of Thumb**: Never call blocking functions (e.g., standard `time.sleep()`, synchronous `requests.get()`, or blocking file operations) inside an `async def`. Doing so freezes the entire event loop. Use `asyncio.sleep()` and `httpx.AsyncClient` instead.

### C. Context Managers (`async with`)
In TypeScript, socket cleanup often relies on explicit teardown or garbage collection. In Python, the `async with` block guarantees that sockets, connections, and database handles are closed immediately when exiting the block — even if an exception is thrown:
```python
async with httpx.AsyncClient() as client:
    response = await client.get(url)
# Client socket is closed immediately here
```

### D. Pydantic Models & Field Aliasing
APIs frequently use `camelCase` (`userId`), while Python strictly follows `snake_case` (`user_id`). Pydantic bridges this effortlessly:
```python
from pydantic import BaseModel, ConfigDict, Field

class Post(BaseModel):
    id: int
    user_id: int = Field(alias="userId")
    title: str
    body: str

    model_config = ConfigDict(populate_by_name=True)
```
- **Parsing**: `Post.model_validate(raw_dict)` parses and validates.
- **Dumping**: `post.model_dump()` turns it back into a Python `dict`.
- **JSON**: `post.model_dump_json()` outputs a JSON string.

### E. Error Boundaries & Exception Chaining
Never leak raw third-party library exceptions (like `httpx.ConnectTimeout`) to callers.
- Subclass `Exception` to create domain-specific errors (`ApiTimeoutError`, `ApiResponseError`, `ApiParseError`).
- Use Python's `raise ... from exc` syntax to preserve original stack traces (exception chaining):
  ```python
  except httpx.TimeoutException as exc:
      raise ApiTimeoutError("Request timed out") from exc
  ```

### F. SQLite Persistence (`aiosqlite`)
- Use parameterized queries (`?`) to prevent SQL injection.
- Implement idempotency with upserts:
  ```sql
  INSERT INTO posts (id, user_id, title, body)
  VALUES (?, ?, ?, ?)
  ON CONFLICT(id) DO UPDATE SET
      user_id = excluded.user_id,
      title = excluded.title,
      body = excluded.body;
  ```

### G. Isolated Testing (Mocking Without Network)
Real unit and integration tests should never depend on third-party uptime or live networks.
- `respx_mock` intercepts HTTPX calls before they touch the operating system's network stack.
- `tmp_path` is a built-in pytest fixture providing a fresh, temporary directory per test run, ensuring test databases are completely isolated and automatically cleaned up.

---

## 3. Project Architecture

```text
fastapi-learn/
├── .env.example             # Template for required environment variables
├── .gitignore               # Ignores .venv, __pycache__, .env, *.db
├── pyproject.toml           # Project metadata, dependencies & pytest config
├── uv.lock                  # Deterministic dependency lockfile
├── main.py                  # Dual-purpose entry point (FastAPI server + CLI)
├── src/
│   ├── __init__.py          # Marks src as an importable package
│   ├── config.py            # Typed settings via pydantic-settings
│   ├── exceptions.py        # Domain-level custom exceptions (Error Boundaries)
│   ├── models.py            # Pydantic data schemas
│   ├── client.py            # Resilient HTTP client with timeouts
│   └── db.py                # Asynchronous SQLite persistence
└── tests/
    ├── __init__.py
    └── test_client.py       # 6 unit & integration tests covering all edge cases
```

---

## 4. Command Quick Reference

### Virtual Environment & Dependencies
```bash
# Add a production dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>
```

### Running Tests
```bash
# Run all tests with verbose output
uv run pytest -v

# Run a specific test
uv run pytest -k "test_fetch_post_timeout"
```

### Running the CLI
```bash
# Fetch post 1, validate, and persist to SQLite
uv run python main.py 1

# Fetch a non-existent post to test the controlled error boundary
uv run python main.py 99999
```

### Running the FastAPI Server
```bash
# Start development server with hot-reload
uv run fastapi dev main.py
```
- **API Base**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`

### Triggering Endpoints via cURL
```bash
# Trigger POST to sync external data into SQLite
curl -X POST http://127.0.0.1:8000/posts/1/sync

# Read persisted record from SQLite
curl http://127.0.0.1:8000/posts/1
```
