"""Custom exceptions for the application."""

from typing import Any


class AppExceptionError(Exception):
    """Base exception class for application-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppExceptionError):
    """Raised when input validation fails."""


class AgentTaskFailureError(AppExceptionError):
    """Raised when an agent task fails and should be retried."""

    def __init__(
        self,
        message: str,
        task_type: str | None = None,
        agent_type: str | None = None,
        field: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.task_type = task_type
        self.agent_type = agent_type
        self.field = field
        super().__init__(message, error_code, details)


class ResearchFailureError(AgentTaskFailureError):
    """Raised when research phase fails completely with no usable sources."""

    def __init__(
        self,
        message: str = "Research failed to gather any usable sources",
        task_type: str | None = None,
        field: str | None = None,
        research_iterations: int | None = None,
        error_code: str = "RESEARCH_FAILURE",
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if research_iterations is not None:
            details["research_iterations"] = research_iterations
        super().__init__(
            message=message,
            task_type=task_type,
            agent_type="REPORTER",
            field=field,
            error_code=error_code,
            details=details,
        )



