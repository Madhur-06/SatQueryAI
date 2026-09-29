"""Structured errors. Every user-facing failure carries a stable code, a message and details."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    # inputs
    NO_INPUTS = "no_inputs"
    TOO_MANY_INPUTS = "too_many_inputs"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNREADABLE_FILE = "unreadable_file"
    INVALID_OVERRIDE = "invalid_override"
    UPLOAD_TOO_LARGE = "upload_too_large"
    MODALITY_UNKNOWN = "modality_unknown"
    MISSING_CRS = "missing_crs"
    CRS_MISMATCH = "crs_mismatch"
    NO_OVERLAP = "no_overlap"
    LOW_OVERLAP = "low_overlap"
    INCOMPATIBLE_PAIR = "incompatible_pair"
    BAND_ROLE_MISSING = "band_role_missing"
    MODALITY_MISMATCH = "modality_mismatch"
    # planning / execution
    TASK_INPUT_MISMATCH = "task_input_mismatch"
    NO_TOOL = "no_tool"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INVALID_PARAMS = "invalid_params"
    TOOL_FAILED = "tool_failed"
    INTERNAL_ERROR = "internal_error"


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    hint: str | None = None


class SatQueryError(Exception):
    """Raised for expected, user-explainable failures (bad inputs, impossible plans...)."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.info = ErrorInfo(code=code, message=message, details=details or {}, hint=hint)

    @property
    def code(self) -> ErrorCode:
        return self.info.code

    def __str__(self) -> str:
        return f"[{self.info.code}] {self.info.message}"
