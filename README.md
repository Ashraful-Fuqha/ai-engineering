# Applied AI Engineering: Foundations to Production Agents

A unified monorepo documenting the end-to-end engineering journey of building robust, type-safe, and autonomous AI systems. Built using **modern Python 3.13**, **`uv` workspaces**, **Pydantic v2**, and **FastAPI**.

---

## 🏗️ Monorepo Architecture

This monorepo uses **`uv` workspaces** to isolate dependencies, CLIs, and test suites across projects while sharing a single virtual environment, lockfile, and root `.env`.

```text
ai-engineering/
├── .env.example                     # Shared environment template
├── .gitignore                       # Multi-package ignore rules
├── pyproject.toml                   # Root workspace manifest ([tool.uv.workspace])
├── README.md                        # Master portfolio overview
└── projects/
    ├── fastapi-learn/               # Milestone 1: HTTP & Pydantic Microservice
    ├── model-learn/                 # Milestone 2: Raw Model API, SSE Streaming & Cost Meter
    └── log-extractor/               # Milestone 3: Structured Outputs & Diagnostic Contracts
```

---

## 📦 Projects Overview

### 1. `fastapi-learn` (Microservice Foundations)
* **Goal**: Build an async microservice and CLI client that calls external endpoints, validates payloads with Pydantic, and persists to SQLite.
* **Key Concepts**: Python packaging (`uv`), Type hints, Async/await event loop, Error boundaries, Non-blocking SQLite (`aiosqlite`), Pytest fixture mocking with `respx`.
* **Documentation**: See [fastapi-learn/notes.md](projects/fastapi-learn/notes.md).

### 2. `model-learn` (The Raw Model API)
* **Goal**: Understand the direct LLM provider wire protocol via HTTPX without black-box agent frameworks.
* **Key Concepts**: Server-Sent Events (SSE) parsing, Token economics, Sampling mathematics (Logits, Softmax, Temperature, Top-p, PRNG Seeds), Time-To-First-Token (TTFT) timing, Generation speed (Tokens/Sec), Versioned cost calculations, and Credential redaction in telemetry.
* **Documentation**: See [model-learn/notes.md](projects/model-learn/notes.md).

### 3. `log-extractor` (Structured Outputs & Contracts)
* **Goal**: Convert messy Computer Vision and Slurm job logs into evidence-backed, type-safe diagnostic reports (`CVFailureReport`).
* **Key Concepts**: Schema-Constrained Grammars (CFG decoding), Layered Validation (Layer 1 JSON Schema + Layer 2 Pydantic domain rules), Anti-hallucination contracts (verbatim evidence verification), 1-Shot Bounded Repair Retries (self-healing), Indirect prompt injection defense, and Rate-limit backoff engineering.
* **Documentation**: See [log-extractor/notes.md](projects/log-extractor/notes.md).

---

## 🚀 Quick Start Guide

### Prerequisites
Install the `uv` package manager (Rust-based, blazing fast):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation
Clone the monorepo and synchronize the unified virtual environment:
```bash
git clone git@github.com:Ashraful-Fuqha/ai-engineering.git
cd ai-engineering

# Copy template and configure your API keys
cp .env.example .env

# Single-command install of all workspace dependencies
uv sync
```

---

## 🧪 Running Tests & CLIs Across Projects

You can run commands from the monorepo root using `uv run --package <name>`, or `cd` into any project folder:

### 1. Log Extractor (Structured Outputs)
```bash
# Run offline unit tests
uv run --package log-extractor pytest projects/log-extractor/tests -v

# Extract from a single log
uv run --package log-extractor log-extractor extract --log "RuntimeError: CUDA out of memory on GPU 0"

# Run evaluation benchmark against synthetic test fixtures
uv run --package log-extractor log-extractor evaluate --limit 5
```

### 2. Model Learn (Streaming & Cost Meter)
```bash
# Run unit tests
uv run --package model-learn pytest projects/model-learn/tests -v

# Live stream prompt with TTFT and cost telemetry
uv run --package model-learn model-learn --prompt "Explain Server-Sent Events" --save-log
```

### 3. FastAPI Learn (Microservice & Persistence)
```bash
# Run unit tests
uv run --package fastapi-learn pytest projects/fastapi-learn/tests -v

# Run CLI fetch and persist
uv run --package fastapi-learn python projects/fastapi-learn/main.py 1
```
