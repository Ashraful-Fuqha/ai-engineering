import json
import httpx
import pytest
import respx
from pydantic import ValidationError

from log_extractor.extractor import DiagnosticExtractor
from log_extractor.schemas import (
    CVFailureReport,
    ErrorClass,
    ImageDimensions,
    MemoryUsage,
    PipelineStep,
)


def test_cv_failure_report_schema_rules():
    """Verify Layer 2 Pydantic domain rules and constraints."""
    # 1. Valid instance
    report = CVFailureReport(
        pipeline_step=PipelineStep.MODEL_FORWARD,
        error_class=ErrorClass.OUT_OF_MEMORY,
        memory=MemoryUsage(allocated_mb=1024.0, reserved_mb=2048.0, device_id=0),
        image_dimensions=ImageDimensions(height=224, width=224, channels=3),
        evidence=["CUDA out of memory. Tried to allocate 1024.00 MiB"],
        hypotheses=["Batch size too large for GPU VRAM"],
        suggested_action="Reduce per-device batch size from 32 to 16",
        requires_human_review=False,
    )
    assert report.error_class == ErrorClass.OUT_OF_MEMORY
    assert report.memory.allocated_mb == 1024.0

    # 2. Negative memory must raise ValidationError
    with pytest.raises(ValidationError):
        MemoryUsage(allocated_mb=-100.0)

    # 3. Invalid image dimensions (zero or negative)
    with pytest.raises(ValidationError):
        ImageDimensions(height=0, width=512)

    # 4. Empty evidence must raise ValidationError
    with pytest.raises(ValidationError):
        CVFailureReport(
            pipeline_step=PipelineStep.MODEL_FORWARD,
            error_class=ErrorClass.OUT_OF_MEMORY,
            evidence=[],
            hypotheses=["hyp"],
            suggested_action="action",
        )


def test_verbatim_evidence_verification():
    """Verify that hallucinated evidence quotes are caught and rejected."""
    raw_log = "RuntimeError: CUDA out of memory. Tried to allocate 1024.00 MiB on GPU 0"
    
    # Genuine quote from raw_log
    valid_report = CVFailureReport(
        pipeline_step=PipelineStep.MODEL_FORWARD,
        error_class=ErrorClass.OUT_OF_MEMORY,
        evidence=["CUDA out of memory"],
        hypotheses=["VRAM exhausted"],
        suggested_action="Lower batch size",
    )
    valid_report.verify_evidence_verbatim(raw_log)  # Must pass without raising

    # Fabricated quote not in raw_log
    fake_report = CVFailureReport(
        pipeline_step=PipelineStep.MODEL_FORWARD,
        error_class=ErrorClass.OUT_OF_MEMORY,
        evidence=["Memory allocation failed on NVIDIA A100 device 7"],
        hypotheses=["VRAM exhausted"],
        suggested_action="Lower batch size",
    )
    with pytest.raises(ValueError, match="Evidence quotes do not appear verbatim"):
        fake_report.verify_evidence_verbatim(raw_log)


@pytest.fixture
def extractor():
    return DiagnosticExtractor(
        api_key="test-api-key",
        base_url="https://api.mockllm.com/v1",
        model="gemini-2.5-flash",
    )


async def test_successful_structured_extraction(extractor, respx_mock):
    """Test standard 1-attempt successful extraction."""
    raw_log = "RuntimeError: The size of tensor a (512) must match tensor b (256) at dimension 1"
    
    mock_payload = {
        "pipeline_step": "model_forward",
        "error_class": "ShapeMismatch",
        "image_dimensions": None,
        "memory": None,
        "evidence": ["The size of tensor a (512) must match tensor b (256) at dimension 1"],
        "hypotheses": ["Linear projection layer expected input dimension 256 but got 512"],
        "suggested_action": "Change in_features of classifier layer to 512",
        "requires_human_review": False,
    }

    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(mock_payload),
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    result = await extractor.extract(raw_log)
    assert result.success is True
    assert result.attempts == 1
    assert result.report.error_class == ErrorClass.SHAPE_MISMATCH
    assert result.report.image_dimensions is None


async def test_bounded_repair_retry_success(extractor, respx_mock):
    """Test that a 1st attempt failure triggers the 1-shot repair loop and succeeds."""
    raw_log = "RuntimeError: CUDA out of memory on GPU 0. Allocated: 11000 MiB"

    # Attempt 1 returns hallucinated evidence not present in raw_log
    bad_payload = {
        "pipeline_step": "model_forward",
        "error_class": "OutOfMemory",
        "image_dimensions": None,
        "memory": {"allocated_mb": 11000.0},
        "evidence": ["Hallucinated quote not in the log text"],
        "hypotheses": ["OOM"],
        "suggested_action": "Reduce batch size",
        "requires_human_review": False,
    }

    # Attempt 2 (repair) returns corrected verbatim quote
    good_payload = {
        "pipeline_step": "model_forward",
        "error_class": "OutOfMemory",
        "image_dimensions": None,
        "memory": {"allocated_mb": 11000.0},
        "evidence": ["CUDA out of memory on GPU 0"],
        "hypotheses": ["OOM"],
        "suggested_action": "Reduce batch size",
        "requires_human_review": False,
    }

    endpoint = respx_mock.post("https://api.mockllm.com/v1/chat/completions")
    endpoint.side_effect = [
        httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(bad_payload)}, "finish_reason": "stop"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(good_payload)}, "finish_reason": "stop"}]}),
    ]

    result = await extractor.extract(raw_log)
    assert result.success is True
    assert result.attempts == 2  # Proves 1-shot repair was called and succeeded!
    assert result.report.evidence == ["CUDA out of memory on GPU 0"]


async def test_model_refusal_handled_separately(extractor, respx_mock):
    """Test safety refusal is flagged cleanly without unhandled crash."""
    raw_log = "Malicious attack code in log buffer..."

    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "refusal": "I cannot analyze this log due to safety policies.",
                            "content": "",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    result = await extractor.extract(raw_log)
    assert result.success is False
    assert result.refusal == "I cannot analyze this log due to safety policies."
    assert result.report is None


async def test_truncation_handled_separately(extractor, respx_mock):
    """Test max_tokens cutoff (finish_reason == 'length') is detected as fatal truncation."""
    raw_log = "Very long stacktrace..."

    respx_mock.post("https://api.mockllm.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"pipeline_step": "model_forward", "error_class": "OutOfMem',
                        },
                        "finish_reason": "length",
                    }
                ]
            },
        )
    )

    result = await extractor.extract(raw_log)
    assert result.success is False
    assert result.truncated is True
    assert result.report is None

