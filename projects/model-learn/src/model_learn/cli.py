import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

from .client import LlmApiError, LlmClient
from .config import settings
from .models import ChatCompletionRequest, Message
from .redact import redact_sensitive_data


def print_metrics_summary(result) -> None:
    """Print an eye-pleasing, structured performance and cost report."""
    metrics = result.metrics
    usage = result.usage
    print("\n" + "=" * 55)
    print("📊 EXECUTION & COST TELEMETRY")
    print("=" * 55)
    print(f"• Model             : {result.model}")
    print(f"• Mode              : {'STREAMING (SSE)' if result.stream else 'NON-STREAMING'}")
    if result.system_fingerprint:
        print(f"• System Fingerprint: {result.system_fingerprint}")
    print("-" * 55)
    print(f"• TTFT (First Token): {metrics.ttft_ms:.1f} ms")
    print(f"• Total Latency     : {metrics.total_latency_ms:.1f} ms")
    print(f"• Generation Speed  : {metrics.tokens_per_second:.1f} tokens/sec")
    print("-" * 55)
    print(f"• Prompt Tokens     : {usage.prompt_tokens} (cached: {usage.cached_tokens})")
    print(f"• Completion Tokens : {usage.completion_tokens}")
    print(f"• Total Tokens      : {usage.total_tokens}")
    print(f"• Estimated Cost    : ${metrics.estimated_cost_usd:.6f} USD")
    print("=" * 55 + "\n")


def save_telemetry_log(result, prompt: str, system_prompt: str | None) -> Path:
    """Save redacted telemetry JSON to logs/ directory."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"telemetry_{timestamp}.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "model": result.model,
        "streaming": result.stream,
        "request": {
            "system_prompt": system_prompt,
            "prompt": prompt,
        },
        "response": {
            "content": result.full_content,
            "finish_reason": result.finish_reason,
            "system_fingerprint": result.system_fingerprint,
        },
        "usage": result.usage.model_dump(),
        "metrics": result.metrics.model_dump(),
    }
    
    # Ensure sensitive credentials and keys are never saved in logs
    safe_data = redact_sensitive_data(data)
    log_file.write_text(json.dumps(safe_data, indent=2), encoding="utf-8")
    return log_file


async def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Raw LLM Streaming & Cost Meter CLI (Direct HTTP Wire Protocol)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Prompt text to send to the model",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to a text file containing the prompt",
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default="You are an expert software engineer and helpful AI assistant.",
        help="System prompt instructions",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=settings.gemini_model,
        help=f"Target model (default: {settings.gemini_model})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 2.0, default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed integer for sampling determinism",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (use traditional non-streaming HTTP POST)",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save redacted request & response telemetry as JSON in logs/",
    )

    args = parser.parse_args()

    # Determine prompt text
    prompt_text = ""
    if args.file:
        if not args.file.exists():
            print(f"Error: Prompt file '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        prompt_text = args.file.read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt_text = args.prompt.strip()
    else:
        print("Enter your prompt (Press Enter twice or Ctrl+D to submit):")
        lines = []
        try:
            while True:
                line = input()
                if not line and lines and not lines[-1]:
                    break
                lines.append(line)
        except EOFError:
            pass
        prompt_text = "\n".join(lines).strip()

    if not prompt_text:
        print("Error: No prompt provided. Use --prompt, --file, or interactive input.", file=sys.stderr)
        sys.exit(1)

    # Validate API key
    if not settings.gemini_api_key:
        print(
            "Error: No API key found!\n"
            "Please add GEMINI_API_KEY to your .env file in this directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    messages = [
        Message(role="system", content=args.system),
        Message(role="user", content=prompt_text),
    ]

    is_stream = not args.no_stream
    request = ChatCompletionRequest(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stream=is_stream,
    )

    client = LlmClient()

    if args.seed is not None and "googleapis.com" in settings.gemini_base_url:
        print("ℹ️  Note: Google Gemini's gateway ignores 'seed'. Using temperature=0.0 for deterministic greedy decoding.")

    print(f"\n🚀 Sending request to {args.model} (stream={is_stream})...\n")

    def on_chunk(token: str) -> None:
        print(token, end="", flush=True)

    try:
        result = await client.execute(
            request,
            on_chunk=on_chunk if is_stream else None,
        )
        if not is_stream:
            print(result.full_content)
        else:
            print()  # newline after streaming finishes

        print_metrics_summary(result)

        if args.save_log:
            log_path = save_telemetry_log(result, prompt_text, args.system)
            print(f"📁 Telemetry saved to: {log_path} (Credentials safely redacted)")

    except LlmApiError as err:
        print(f"\n❌ {err.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n❌ Unexpected error: {err}", file=sys.stderr)
        sys.exit(1)


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
