from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ErrorClass(str, Enum):
    OUT_OF_MEMORY = "OutOfMemory"
    SHAPE_MISMATCH = "ShapeMismatch"
    FILE_NOT_FOUND = "FileNotFound"
    CUDA_KERNEL_ERROR = "CudaKernelError"
    CORRUPTED_IMAGE = "CorruptedImage"
    CHECKPOINT_ERROR = "CheckpointError"
    TIMEOUT = "Timeout"
    UNKNOWN = "Unknown"


class PipelineStep(str, Enum):
    DATA_LOADING = "data_loading"
    AUGMENTATION = "augmentation"
    MODEL_FORWARD = "model_forward"
    LOSS_BACKWARD = "loss_backward"
    OPTIMIZER_STEP = "optimizer_step"
    CHECKPOINTING = "checkpointing"
    DISTRIBUTED_COMM = "distributed_comm"
    UNKNOWN = "unknown"


class ImageDimensions(BaseModel):
    """Image or tensor dimensions explicitly stated in the log."""
    height: int = Field(gt=0, description="Image height in pixels (must be > 0)")
    width: int = Field(gt=0, description="Image width in pixels (must be > 0)")
    channels: int | None = Field(default=None, gt=0, description="Channels count (e.g. 3 for RGB)")
    batch_size: int | None = Field(default=None, gt=0, description="Batch dimension if given")

    model_config = ConfigDict(extra="forbid")


class MemoryUsage(BaseModel):
    """GPU / Host memory statistics explicitly stated in the log."""
    allocated_mb: float = Field(ge=0.0, description="Memory allocated in MB (must be >= 0)")
    reserved_mb: float | None = Field(default=None, ge=0.0, description="Memory reserved by CUDA in MB")
    device_id: int | None = Field(default=None, ge=0, description="CUDA device index (e.g. 0)")

    model_config = ConfigDict(extra="forbid")


class CVFailureReport(BaseModel):
    """
    Structured, evidence-backed diagnostic report extracted from a CV / Slurm job log.
    Strictly forbids inventing information: missing dimensions or memory must be null.
    """
    pipeline_step: PipelineStep = Field(
        description="The stage in the CV training/inference pipeline where failure occurred"
    )
    error_class: ErrorClass = Field(
        description="Standardized error category"
    )
    image_dimensions: ImageDimensions | None = Field(
        default=None,
        description="Image dimensions ONLY if explicitly stated in the log text; otherwise null",
    )
    memory: MemoryUsage | None = Field(
        default=None,
        description="GPU/Host memory figures ONLY if explicitly mentioned in the log; otherwise null",
    )
    evidence: list[str] = Field(
        min_length=1,
        description="Exact verbatim substrings quoted directly from the log text supporting this diagnosis",
    )
    hypotheses: list[str] = Field(
        min_length=1,
        description="Probable root causes explaining why this failure occurred",
    )
    suggested_action: str = Field(
        min_length=5,
        description="Concrete, actionable fix recommended for the engineer",
    )
    requires_human_review: bool = Field(
        default=False,
        description="Flag indicating if the log was ambiguous, truncated, or unresolvable",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("evidence")
    @classmethod
    def validate_evidence_not_blank(cls, v: list[str]) -> list[str]:
        cleaned = [item.strip() for item in v if item and item.strip()]
        if not cleaned:
            raise ValueError("Evidence list cannot be empty or contain only blank strings.")
        return cleaned

    @model_validator(mode="after")
    def validate_unknown_flag(self) -> "CVFailureReport":
        """If the error class is Unknown, human review must always be flagged."""
        if self.error_class == ErrorClass.UNKNOWN or self.pipeline_step == PipelineStep.UNKNOWN:
            self.requires_human_review = True
        return self

    def verify_evidence_verbatim(self, raw_log: str) -> None:
        """
        Anti-Hallucination verification: Ensure every evidence quote is an
        exact verbatim substring of the raw input log.
        """
        missing_quotes: list[str] = []
        for quote in self.evidence:
            # Clean outer quotes if model wrapped them in "..."
            clean_quote = quote.strip(' \t\n\r"\'')
            if clean_quote not in raw_log:
                # Also try matching normalized whitespace
                normalized_quote = " ".join(clean_quote.split())
                normalized_log = " ".join(raw_log.split())
                if normalized_quote not in normalized_log:
                    missing_quotes.append(quote)
        
        if missing_quotes:
            raise ValueError(
                f"Evidence quotes do not appear verbatim in the raw log: {missing_quotes}"
            )

