"""Custom exception for content quality validation failures."""


class ContentQualityError(Exception):
    """Raised when transcription content fails quality validation.

    Caught by ``_ainvoke_with_retry`` and retried against the
    ``validation_attempts`` budget alongside ``pydantic.ValidationError``
    and ``InputTokensBelowThresholdError``.
    """

    def __init__(self, failure_type: str, detail: str) -> None:
        self.failure_type = failure_type
        self.detail = detail
        super().__init__(
            f"Content quality check failed ({failure_type}): {detail}"
        )
