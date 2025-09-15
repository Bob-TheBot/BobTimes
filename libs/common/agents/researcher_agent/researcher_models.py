"""Researcher-specific models and responses."""

from pydantic import BaseModel


class ResearcherInfoResponse(BaseModel):
    """Structured information about a researcher agent."""
    id: str
    field: str
    sub_section: str | None
    tools: list[str]
    default_model_speed: str
    temperature: float


class ResearcherExecutionResult(BaseModel):
    """Result of researcher task execution."""
    success: bool
    result_type: str  # "story_draft", "topic_list", "research_result"
    iterations_used: int
    research_iterations: int
    sources_collected: int
    facts_gathered: int
    error_message: str | None = None


class JournalistTaskSummary(BaseModel):
    """Summary of researcher task for logging and tracking."""
    task_type: str
    field: str
    sub_section: str | None
    topic: str | None
    description: str
    min_sources: int
    target_word_count: int
    researcher_id: str
