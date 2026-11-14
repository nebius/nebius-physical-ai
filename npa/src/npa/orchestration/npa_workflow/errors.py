"""Shared errors for NPA workflow specs."""


class NpaWorkflowError(ValueError):
    """Raised when a workflow spec is malformed or inconsistent."""


class WorkflowExecutionError(NpaWorkflowError):
    """Preserve a workflow failure separately from its failure-handler outcome."""

    def __init__(
        self,
        primary_error: NpaWorkflowError,
        *,
        failure_handler: str = "",
        cleanup_error: NpaWorkflowError | None = None,
    ) -> None:
        self.primary_error = str(primary_error)
        self.failure_handler = failure_handler
        self.cleanup_error = str(cleanup_error) if cleanup_error is not None else ""
        message = self.primary_error
        if failure_handler:
            outcome = (
                f"failed: {self.cleanup_error}" if self.cleanup_error else "completed"
            )
            message = (
                f"{message}; failure handler {failure_handler!r} {outcome} "
                "(the original workflow failure remains authoritative)"
            )
        super().__init__(message)
