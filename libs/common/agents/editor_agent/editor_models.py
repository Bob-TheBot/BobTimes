"""Editor decision and action models."""

from typing import Any

from agents.types import EditorActionType
from pydantic import BaseModel, Field

# Avoid circular import - use forward reference


class EditorAction(BaseModel):
    """Action the editor wants to take."""
    reasoning: str  # Why this action was chosen (FIRST)
    action_type: EditorActionType
    parameters: dict[str, Any]


class EditorDecision(BaseModel):
    """Editor's decision for next step."""
    reasoning: str  # Overall reasoning for the decision (FIRST)
    action: EditorAction
    updated_state: dict[str, Any] | None = None  # Use dict to avoid circular import
    continue_cycle: bool


class StoryReview(BaseModel):
    """Review of a story draft."""
    reasoning: str  # Why this review decision was made (FIRST)
    story_id: str
    reporter_id: str
    is_approved: bool
    feedback: str | None = None
    required_changes: list[str] = Field(default_factory=list)


class NewspaperStatusResponse(BaseModel):
    """Response model for newspaper status."""
    content: dict[str, str] = {}
    error: str | None = None


class EditorInfoResponse(BaseModel):
    """Response model for editor information."""
    id: str
    tools: list[str]
    default_model_speed: str
    temperature: float
