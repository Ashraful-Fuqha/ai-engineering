# CV / Slurm Log Diagnostic Extractor

A robust diagnostic extraction pipeline that converts messy Computer Vision stack traces, CUDA crashes, and Slurm job logs into evidence-backed, type-safe Pydantic reports (`CVFailureReport`).

## Core Capabilities
- **Schema-Constrained Decoding**: Enforces strict JSON Schema contracts at inference time.
- **Layered Validation**: Pydantic domain rules verifying memory bounds, image dimensions, and verbatim evidence.
- **Anti-Hallucination**: Strictly outputs `null` for missing evidence without inventing fake dimensions.
- **1-Shot Bounded Repair**: Automatically re-prompts with validation error tracebacks to heal recoverable schema errors.
- **Adversarial Defense**: Immunized against indirect prompt injections embedded in raw log buffers.
- **Benchmark Evaluation**: Includes 32 labeled synthetic test fixtures with automated accuracy scoring.

## Quick Start

```bash
# Run unit tests
uv run pytest -v

# Extract from a single log string
uv run log-extractor extract --log "RuntimeError: CUDA out of memory..."

# Run benchmark evaluation
uv run log-extractor evaluate --limit 5
```

