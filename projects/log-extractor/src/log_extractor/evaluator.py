import asyncio
from dataclasses import dataclass, field
from typing import Any

from .extractor import DiagnosticExtractor, ExtractionResult
from .fixtures import FIXTURES, LogFixture


@dataclass
class EvaluationMetrics:
    total_cases: int = 0
    schema_valid_count: int = 0
    error_class_correct: int = 0
    pipeline_step_correct: int = 0
    evidence_verbatim_valid: int = 0
    anti_hallucination_valid: int = 0
    injection_defense_success: int = 0
    repair_triggered_count: int = 0
    refusal_or_truncation_count: int = 0
    error_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def schema_validity_rate(self) -> float:
        return (self.schema_valid_count / self.total_cases * 100) if self.total_cases else 0.0

    @property
    def error_accuracy(self) -> float:
        return (self.error_class_correct / self.total_cases * 100) if self.total_cases else 0.0

    @property
    def step_accuracy(self) -> float:
        return (self.pipeline_step_correct / self.total_cases * 100) if self.total_cases else 0.0

    @property
    def evidence_validity_rate(self) -> float:
        return (self.evidence_verbatim_valid / self.total_cases * 100) if self.total_cases else 0.0

    @property
    def anti_hallucination_rate(self) -> float:
        return (self.anti_hallucination_valid / self.total_cases * 100) if self.total_cases else 0.0


async def evaluate_fixtures(
    extractor: DiagnosticExtractor,
    fixtures: list[LogFixture] | None = None,
    concurrency: int = 1,
    delay_between_requests: float = 2.0,
    on_progress: Any = None,
) -> EvaluationMetrics:
    """
    Run extraction against synthetic benchmark fixtures with rate-limit pacing
    and fault-tolerant error containment.
    """
    test_set = fixtures or FIXTURES
    metrics = EvaluationMetrics(total_cases=len(test_set))
    semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_single(fixture: LogFixture) -> dict[str, Any]:
        async with semaphore:
            case_result: dict[str, Any] = {
                "id": fixture.id,
                "title": fixture.title,
                "expected_error": fixture.expected_error.value,
                "is_adversarial": fixture.is_adversarial,
            }

            try:
                res: ExtractionResult = await extractor.extract(fixture.raw_log)
                case_result["success"] = res.success
                case_result["attempts"] = res.attempts

                if res.refusal or res.truncated:
                    metrics.refusal_or_truncation_count += 1
                    case_result["status"] = "refused_or_truncated"
                    if on_progress:
                        on_progress(fixture, case_result)
                    return case_result

                if not res.success or res.report is None:
                    case_result["status"] = "failed"
                    case_result["error"] = res.error_message
                    if on_progress:
                        on_progress(fixture, case_result)
                    return case_result

                # Successful schema validation
                metrics.schema_valid_count += 1
                if res.attempts > 1:
                    metrics.repair_triggered_count += 1

                rep = res.report
                case_result["extracted_error"] = rep.error_class.value
                case_result["extracted_step"] = rep.pipeline_step.value

                # 1. Error class match
                if rep.error_class == fixture.expected_error:
                    metrics.error_class_correct += 1
                    case_result["error_match"] = True
                else:
                    case_result["error_match"] = False

                # 2. Pipeline step match
                if rep.pipeline_step == fixture.expected_step:
                    metrics.pipeline_step_correct += 1
                    case_result["step_match"] = True
                else:
                    case_result["step_match"] = False

                # 3. Verbatim Evidence Check
                try:
                    rep.verify_evidence_verbatim(fixture.raw_log)
                    metrics.evidence_verbatim_valid += 1
                    case_result["evidence_valid"] = True
                except ValueError:
                    case_result["evidence_valid"] = False

                # 4. Anti-Hallucination check
                hallucinated = False
                if not fixture.expect_dimensions and rep.image_dimensions is not None:
                    hallucinated = True
                if not fixture.expect_memory and rep.memory is not None:
                    hallucinated = True

                if not hallucinated:
                    metrics.anti_hallucination_valid += 1
                    case_result["anti_hallucination"] = True
                else:
                    case_result["anti_hallucination"] = False

                # 5. Adversarial prompt injection defense
                if fixture.is_adversarial:
                    if rep.error_class == fixture.expected_error:
                        metrics.injection_defense_success += 1
                        case_result["injection_defended"] = True
                    else:
                        case_result["injection_defended"] = False

                case_result["status"] = "success"

            except Exception as err:
                metrics.error_count += 1
                case_result["status"] = "error"
                case_result["error"] = str(err)

            if on_progress:
                on_progress(fixture, case_result)

            # Pacing pause to respect rate limits
            if delay_between_requests > 0:
                await asyncio.sleep(delay_between_requests)

            return case_result

    # Run in sequential or controlled batches to stay under RPM quota
    results = []
    for f in test_set:
        res = await _evaluate_single(f)
        results.append(res)

    metrics.results = results
    return metrics

