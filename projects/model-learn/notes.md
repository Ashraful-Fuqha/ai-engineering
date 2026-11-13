# The Raw Model API: Architecture, Wire Protocols & Cost Metering

> A comprehensive engineering guide to understanding the LLM provider boundary at the raw HTTP/SSE wire level before introducing higher-level agent frameworks.

---

## 1. Why Start with Raw HTTP (The Provider Boundary)?

Frameworks like LangChain, LlamaIndex, or CrewAI provide convenient abstractions, but they hide critical architectural realities:
1. **Network Latency & Buffering**: Inability to see whether delays come from pre-fill (TTFT), token generation, or proxy buffering.
2. **Hidden Economics**: Obscuring token counts, especially modern "reasoning/thinking tokens" that consume budget without appearing in output text.
3. **The "OpenAI-Compatible" Fallacy**: Assuming all providers support the exact same JSON schema fields (e.g. Google Gemini rejecting the `seed` field).
4. **Streaming Failures**: Not knowing how to handle network drops, incomplete SSE chunks, or mid-stream rate limits.

---

## 2. Wire Protocols: OpenAI Compatibility vs Native Provider APIs

### The Dual-Gateway Reality
Most leading model providers expose two formats:

```
                  ┌──────────────────────────────────────────────┐
                  │            Client Application                │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   [OpenAI-Compatible Wire Format]                 [Native Provider Format]
   e.g. POST /v1/chat/completions                  e.g. POST /v1beta/models/...
   - Standard across OpenAI, Groq,                 - Google: contents -> parts
     OpenRouter, Gemini, DeepSeek                  - Anthropic: content_block_delta
   - Bearer Token in HTTP Headers                  - API keys often leaked in URL query
```

### The Google Gemini Case Study
* **Native Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=AIzaSy...`
  * *Security risk*: API key is passed in the URL query string, where it is logged by HTTP proxies, CDNs, and server access logs.
* **OpenAI-Compatible Gateway**: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
  * *Security win*: Key is sent in `Authorization: Bearer <API_KEY>` header.
  * *Compatibility win*: Allows the same client to switch between Gemini, Groq, OpenAI, or local vLLM with just an env variable change.

#### The "Compatibility Illusion" Gotcha
Even on OpenAI-compatible endpoints, schema validation differs:
```json
// Rejected by Google Gemini with HTTP 400:
// "Invalid JSON payload received. Unknown name 'seed': Cannot find field."
{
  "model": "gemini-2.5-flash",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.0,
  "seed": 42
}
```
**Engineering Lesson**: Always build provider-specific compatibility filters to strip unsupported fields (like `seed`) before dispatching payloads.

---

## 3. Message Roles & ChatML Under the Hood

LLMs do not understand "conversations"—they are autoregressive next-token predictors. The provider's backend converts JSON message arrays into Chat Markup Language (ChatML) templates before passing them to the tokenizer:

```text
<|im_start|>system
You are an expert software engineer.<|im_end|>
<|im_start|>user
Write a hello world program.<|im_end|>
<|im_start|>assistant
```

### Role Taxonomy

| Role | Purpose | Best Practices & Secrets |
| :--- | :--- | :--- |
| **`system`** | Sets foundational rules, output format, tool schemas, and safety boundaries. | Modern providers cache this prefix (Prompt Caching), giving a 50–90% cost reduction on subsequent requests. |
| **`user`** | Raw input from end users or client applications. | Can contain multimodal parts (text, image URLs, base64 documents). |
| **`assistant`** | Prior model responses. | Can be "pre-filled" in few-shot prompting to force the model to start its reply with a specific token (e.g. `{"` for JSON). |
| **`tool`** | Function execution return payload. | Requires a matching `tool_call_id` to correlate the output to the model's prior request. |

---

## 4. Token Economics & The "Thinking Tokens" Mystery

### Tokenization (Byte-Pair Encoding)
* 1 token $\approx$ 4 characters or 0.75 words in standard English prose.
* Code, JSON brackets, punctuation, whitespace, and non-Latin scripts (Arabic, Hindi, Japanese) consume 2–5x more tokens per character.

### The Context Budget Equation
$$\text{Total Tokens} = T_{\text{system}} + T_{\text{history}} + T_{\text{tools}} + T_{\text{retrieved\_context}} + T_{\text{reasoning}} + T_{\text{output}}$$

### The Reasoning / Thinking Tokens Discovery
In our live experiment with `gemini-2.5-flash`, the API reported:
* **Prompt Tokens**: 24
* **Visible Completion Tokens**: 82
* **Total Tokens**: **1,444**

$$\text{Internal Thinking Tokens} = 1444 - (24 + 82) = \mathbf{1,338\text{ tokens}}$$

**What is happening?**
Modern reasoning models (Gemini 2.5 Flash, OpenAI o1/o3-mini, DeepSeek R1) generate an internal Chain of Thought (CoT) to plan and evaluate candidate responses before outputting visible text.
* **Billing Impact**: You pay for internal thinking tokens even though they are never rendered to the user.
* **Latency Impact**: Thinking tokens run sequentially, driving up Time-To-First-Token (TTFT) from ~300ms to 8–10 seconds.

---

## 5. Sampling Math: From Logits to Determinism

```text
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│  Neural Network │ Logits│  Temperature    │       │     Softmax      │ Probabilities
│  Final Layer    ├──────►│  Logit / T      ├──────►│  P(i) = e^zi/Σe  ├─────────────►
└─────────────────┘       └─────────────────┘       └──────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│ Selected Token  │◄──────┤   PRNG Dice     │◄──────┤      Top-P       │
│                 │       │   (Seed: X0)    │       │ (Nucleus Filter) │
└─────────────────┘       └─────────────────┘       └──────────────────┘
```

### 1. Logits
The model outputs an unnormalized vector of scores $z \in \mathbb{R}^{V}$ across its vocabulary $V$ ($\sim 100,000$ tokens).

### 2. Softmax
Converts raw logits into normalized probabilities summing to 1.0 (100%):
$$P(\text{token}_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### 3. Temperature ($T$)
Modifies logits prior to softmax: $\tilde{z}_i = \frac{z_i}{T}$
* **$T \to 0.0$ (Greedy Decoding)**: Largest logit explodes toward $1.0$, completely suppressing all lower-ranked tokens. Essential for JSON, code, and deterministic classification.
* **High $T$ ($1.0 \to 1.5$)**: Compresses differences between logits, giving less probable words a fighting chance. Used for creative writing.

### 4. Top-P (Nucleus Sampling)
Sorts tokens descending by probability and cuts off the tail once the cumulative sum reaches $p$ (e.g. 0.90):
$$\sum_{i \in \text{Candidates}} P(i) \ge p$$
Dynamically expands candidate pool when the model is uncertain, and collapses it to a single token when confident.

### 5. What is a `Seed` Fundamentally?
A computer CPU/GPU cannot generate real random numbers. It uses a **Pseudo-Random Number Generator (PRNG)** formula:
$$X_{n+1} = (a \cdot X_n + c) \pmod m$$
* **$X_0$ is the Seed**: The initial integer starting the formula.
* If you set `seed = 42`, the sequence of random numbers ($X_1, X_2, X_3, \dots$) generated at each token step is mathematically frozen.

---

## 6. HTTP Streaming & The Server-Sent Events (SSE) Protocol

### The 3 Layers of Streaming

```text
[Layer 1: Raw TCP Network Chunks]
Socket bytes: 'data: {"choices": [{"delta": {"con' ... 'tent": "Hello"}}]}\n\n'
       │ (Chunk boundaries do NOT align with JSON boundaries!)
       ▼
[Layer 2: Server-Sent Events (SSE) Framing]
Framed by double newlines (`\n\n`), each payload prefixed by `data: `
       │
       ▼
[Layer 3: Semantic JSON Deltas]
Parsed chunk objects containing:
  - `delta.content`: Fragment of generated text
  - `finish_reason`: "stop" | "length" | "content_filter"
  - `usage`: Final token counts (when stream_options: {"include_usage": true})
  - `data: [DONE]`: Stream termination sentinel
```

### Python HTTPX Streaming Implementation
```python
async with client.stream("POST", url, headers=headers, json=payload) as response:
    response.raise_for_status()
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break  # End of stream
        
        chunk = json.loads(data_str)
        delta_text = chunk["choices"][0]["delta"].get("content", "")
        # Render token immediately to UI
```

---

## 7. Performance & Cost Instrumentation

### Latency Formulas
1. **Time-To-First-Token (TTFT)**:
   $$\text{TTFT} = t_{\text{first\_token\_received}} - t_{\text{request\_sent}}$$
   *Measures network RTT + prompt prefill time + internal reasoning passes.*
2. **Total Latency**:
   $$\text{Latency}_{\text{total}} = t_{\text{stream\_closed}} - t_{\text{request\_sent}}$$
3. **Generation Speed (Tokens Per Second - TPS)**:
   $$\text{TPS} = \frac{\text{Completion Tokens}}{(\text{Latency}_{\text{total}} - \text{TTFT}) / 1000}$$

### Versioned Pricing Calculation
$$\text{Cost} = \left(\frac{T_{\text{uncached\_prompt}}}{10^6} \times P_{\text{input}}\right) + \left(\frac{T_{\text{cached\_prompt}}}{10^6} \times P_{\text{cached}}\right) + \left(\frac{T_{\text{completion}}}{10^6} \times P_{\text{output}}\right)$$

*Example (Gemini 2.5 Flash Rate Card)*:
* Input: \$0.075 / 1M tokens
* Cached Input: \$0.01875 / 1M tokens
* Output: \$0.30 / 1M tokens

---

## 8. Security & Secret Redaction Patterns

### The Redactor Architecture
Never persist raw telemetry logs that might leak credentials.
```python
# Pattern that masks API keys while protecting valid metrics
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api[-_]?key|authorization|bearer|secret|password|credential|access[-_]?token)",
    re.IGNORECASE,
)
```
* **Critical Bug Trap**: Matching the bare word `token` accidentally redacts `prompt_tokens`, `completion_tokens`, and `tokens_per_second`. Always match specific variants (`access_token`, `auth_token`).
* **URL Sanitization**: Strip query parameter keys using regex: `re.sub(r"([?&]key=)([^&]+)", r"\1[REDACTED]", url)`.

---

## 9. Offline Unit Testing with `respx`

Production LLM test suites should never make live network calls or burn API credits:
```python
@pytest.fixture
def mock_client():
    return LlmClient(api_key="test-key", base_url="https://api.mock.com")

async def test_streaming_sse_parsing(mock_client, respx_mock):
    sse_body = (
        'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "world!"}}]}\n\n'
        'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post("https://api.mock.com/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_body, headers={"Content-Type": "text/event-stream"})
    )

    result = await mock_client.execute(ChatCompletionRequest(...))
    assert result.full_content == "Hello world!"
    assert result.usage.prompt_tokens == 5
```

---

## 10. CLI Command Quick Reference

### Running the CLI Meter
```bash
# Live streaming with real-time token rendering and telemetry
uv run model-learn --prompt "Explain Server-Sent Events" --save-log

# Non-streaming mode (compares latency against streaming)
uv run model-learn --prompt "Explain Server-Sent Events" --no-stream

# Greedy deterministic decoding
uv run model-learn --prompt "Invent 3 quirky names" --temperature 0.0 --seed 42

# Reading prompt from an external file
uv run model-learn --file prompt.txt --save-log
```

### Running the Offline Test Suite
```bash
# Run all 6 unit tests with verbose reporting
uv run pytest -v
```

---

## 11. Project Layout

```text
model-learn/
├── .env.example                     # Environment template (GEMINI_API_KEY, MODEL)
├── .gitignore                       # Ignores .env, logs/, *.pyc, .venv/
├── pyproject.toml                   # uv packaging & pytest pythonpath configuration
├── notes.md                         # This comprehensive engineering guide
├── src/
│   └── model_learn/
│       ├── __init__.py              # Package entry point
│       ├── config.py                # Pydantic Settings (GEMINI_API_KEY, GEMINI_MODEL)
│       ├── pricing.py               # Versioned model pricing catalog ($/M tokens)
│       ├── models.py                # Pydantic data schemas (Request, Usage, Metrics)
│       ├── redact.py                # Credential & secret redactor
│       ├── client.py                # HTTPX streaming client & SSE line parser
│       └── cli.py                   # Terminal interface with telemetry dashboard
└── tests/
    └── test_client.py               # 6 offline respx unit tests
```
