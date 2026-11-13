import argparse
import asyncio
import json
from pathlib import Path
import sys

from .config import settings
from .evaluator import evaluate_fixtures
from .extractor import DiagnosticExtractor
from .fixtures import FIXTURES


def print_report(res) -> None:
    """Print an eye-pleasing terminal visualization of the CVFailureReport."""
    if not res.success or res.report is None:
        print("\n❌ EXTRACTION FAILED")
        print(f"• Error     : {res.error_message}")
        print(f"• Refusal   : {res.refusal}")
        print(f"• Truncated : {res.truncated}")
        print(f"• Attempts  : {res.attempts}\n")
        return

    r = res.report
    print("\n" + "=" * 65)
    print("📋 CV / SLURM FAILURE DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"• Pipeline Step        : {r.pipeline_step.value}")
    print(f"• Error Classification : {r.error_class.value}")
    print(f"• Human Review Needed  : {'⚠️ YES' if r.requires_human_review else '✅ NO'}")
    print(f"• Extraction Attempts  : {res.attempts} {'(1-shot repair succeeded!)' if res.attempts > 1 else ''}")
    print("-" * 65)

    if r.image_dimensions:
        dims = r.image_dimensions
        print(f"• Image Dimensions     : {dims.width}x{dims.height} (C={dims.channels}, B={dims.batch_size})")
    else:
        print("• Image Dimensions     : [None in log - Not Hallucinated]")

    if r.memory:
        mem = r.memory
        print(f"• Memory Usage         : {mem.allocated_mb:.1f} MB allocated (GPU {mem.device_id})")
    else:
        print("• Memory Usage         : [None in log - Not Hallucinated]")

    print("-" * 65)
    print("• Verbatim Evidence Quoted From Log:")
    for i, ev in enumerate(r.evidence, 1):
        print(f"   [{i}] \"{ev}\"")

    print("\n• Diagnostic Hypotheses:")
    for i, hyp in enumerate(r.hypotheses, 1):
        print(f"   ({i}) {hyp}")

    print("\n• Suggested Action:")
    print(f"   👉 {r.suggested_action}")
    print("=" * 65 + "\n")


async def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="CV / Slurm Log Diagnostic Extractor (Schema-Constrained Outputs & Evidence Contracts)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command 1: extract
    extract_parser = subparsers.add_parser("extract", help="Extract diagnostic report from a single log")
    extract_parser.add_argument("--log", "-l", type=str, help="Raw log string to analyze")
    extract_parser.add_argument("--file", "-f", type=Path, help="Path to raw log file")
    extract_parser.add_argument("--save-json", action="store_true", help="Save extracted JSON report to reports/")

    # Command 2: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark suite on synthetic labeled fixtures")
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit number of fixtures to test")
    eval_parser.add_argument("--concurrency", type=int, default=1, help="Parallel requests (default: 1)")
    eval_parser.add_argument("--delay", type=float, default=4.5, help="Delay in seconds between requests for free-tier 15 RPM quota (default: 4.5s)")
    eval_parser.add_argument("--save-report", action="store_true", help="Save evaluation metrics JSON")

    args = parser.parse_args()

    # Validate API key
    if not settings.gemini_api_key:
        print(
            "Error: No API key found!\n"
            "Please ensure GEMINI_API_KEY is present in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    extractor = DiagnosticExtractor()

    if args.command == "extract" or (args.command is None and (getattr(args, "log", None) or getattr(args, "file", None))):
        raw_text = ""
        if getattr(args, "file", None):
            if not args.file.exists():
                print(f"Error: Log file '{args.file}' not found.", file=sys.stderr)
                sys.exit(1)
            raw_text = args.file.read_text(encoding="utf-8").strip()
        elif getattr(args, "log", None):
            raw_text = args.log.strip()
        else:
            print("Enter log text (Press Enter twice or Ctrl+D to submit):")
            lines = []
            try:
                while True:
                    line = input()
                    if not line and lines and not lines[-1]:
                        break
                    lines.append(line)
            except EOFError:
                pass
            raw_text = "\n".join(lines).strip()

        if not raw_text:
            print("Error: No log text provided.", file=sys.stderr)
            sys.exit(1)

        print("\n🔍 Analyzing log with schema-constrained decoding...")
        res = await extractor.extract(raw_text)
        print_report(res)

        if getattr(args, "save_json", False) and res.report:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / "latest_diagnostic_report.json"
            report_file.write_text(res.report.model_dump_json(indent=2), encoding="utf-8")
            print(f"📁 Saved report to: {report_file}")

    elif args.command == "evaluate":
        test_fixtures = FIXTURES[:args.limit] if args.limit else FIXTURES
        print(f"\n🚀 Running Benchmark on {len(test_fixtures)} fixtures (delay={args.delay}s to stay under 15 RPM)...\n")

        completed = 0
        def on_prog(fixture, res):
            nonlocal completed
            completed += 1
            
            status = res.get("status")
            if status == "error":
                err_snippet = str(res.get("error", "Unknown error"))[:45]
                badge = f"⏳ 429/ERR: {err_snippet}..."
            elif res.get("error_match"):
                if res.get("attempts", 1) > 1:
                    badge = "🛠️ REPAIRED (Att. 2)"
                else:
                    badge = "✅ PASS"
            else:
                got = res.get("extracted_error", "Unknown")
                exp = res.get("expected_error", "Unknown")
                badge = f"❌ MISMATCH (got {got}, expected {exp})"

            print(f"[{completed:>2}/{len(test_fixtures)}] {badge:<28} : {fixture.id}")

        metrics = await evaluate_fixtures(
            extractor,
            fixtures=test_fixtures,
            concurrency=args.concurrency,
            delay_between_requests=args.delay,
            on_progress=on_prog,
        )

        print("\n" + "=" * 65)
        print("📊 BENCHMARK ACCURACY & ROBUSTNESS SCORECARD")
        print("=" * 65)
        print(f"• Total Test Cases           : {metrics.total_cases}")
        print(f"• Schema Validity Rate       : {metrics.schema_validity_rate:.1f}% ({metrics.schema_valid_count}/{metrics.total_cases})")
        print(f"• Error Class Accuracy       : {metrics.error_accuracy:.1f}% ({metrics.error_class_correct}/{metrics.total_cases})")
        print(f"• Pipeline Step Accuracy     : {metrics.step_accuracy:.1f}% ({metrics.pipeline_step_correct}/{metrics.total_cases})")
        print(f"• Anti-Hallucination Rate    : {metrics.anti_hallucination_rate:.1f}% ({metrics.anti_hallucination_valid}/{metrics.total_cases})")
        print(f"• Verbatim Evidence Validity : {metrics.evidence_validity_rate:.1f}% ({metrics.evidence_verbatim_valid}/{metrics.total_cases})")
        print(f"• Prompt Injections Defended : {metrics.injection_defense_success}/4")
        print(f"• 1-Shot Repairs Triggered   : {metrics.repair_triggered_count}")
        if metrics.error_count > 0:
            print(f"• Rate-Limited / Network Errs: {metrics.error_count}")
        print("=" * 65 + "\n")

        if args.save_report:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            eval_file = reports_dir / "benchmark_scorecard.json"
            eval_file.write_text(json.dumps(metrics.results, indent=2), encoding="utf-8")
            print(f"📁 Full evaluation results saved to: {eval_file}")

    else:
        parser.print_help()


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()

