# Raw LLM Streaming & Cost Meter CLI

A direct wire-level HTTP client for LLM streaming and cost metering using Python, HTTPX, and Server-Sent Events (SSE).

## Features
- Direct HTTP wire protocol (no framework abstraction).
- Server-Sent Events (SSE) streaming with live token rendering.
- Real-time performance telemetry: Time-To-First-Token (TTFT), Total Latency, and Tokens Per Second.
- Versioned token cost calculation.
- Credential redaction in telemetry logs.
