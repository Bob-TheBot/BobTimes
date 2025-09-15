"""Editor state management models and functionality."""

from agents.agent import StoryDraft
from agents.editor_agent.editor_tools import (
    empty_action_history,
    empty_reporters,
    empty_stories_collected,
    empty_topics_assigned,
    empty_topics_collected,
)
from agents.types import EditorGoal, FieldTopicRequest, JournalistField, StoryReviewStatus
from pydantic import BaseModel, Field


class EditorState(BaseModel):
    """Current state of the editor's news cycle orchestration."""
    # Current status
    current_goal: EditorGoal
    current_iteration: int = 0
    max_iterations: int = 20

    # Configuration
    requested_fields: list[FieldTopicRequest]
    stories_per_field: int

    # Progress tracking
    topics_collected: dict[JournalistField, list[str]] = Field(default_factory=empty_topics_collected)
    topics_assigned: dict[str, str] = Field(default_factory=empty_topics_assigned)  # reporter_id -> topic
    stories_collected: dict[str, StoryDraft] = Field(default_factory=empty_stories_collected)
    stories_review_status: dict[str, StoryReviewStatus] = Field(default_factory=dict)  # story_id -> StoryReviewStatus
    stories_in_revision: dict[str, str] = Field(default_factory=dict)  # story_id -> reporter_id
    stories_published: list[str] = Field(default_factory=list)

    # Retry tracking
    publish_retry_count: dict[str, int] = Field(default_factory=dict)  # story_id -> retry_count

    # Active reporters
    active_reporters: dict[str, JournalistField] = Field(default_factory=empty_reporters)

    # History
    action_history: list[str] = Field(default_factory=empty_action_history)

    # Error tracking
    tool_errors: list[str] = Field(default_factory=list)  # Recent tool execution errors
    last_failed_action: str | None = None  # Last action that failed


class EditorStateManager:
    """Helper class for managing editor state operations."""

    @staticmethod
    def get_approved_stories(state: EditorState) -> list[str]:
        """Get list of story IDs that are approved and ready for publishing."""
        approved: list[str] = []
        for story_id, review_status in state.stories_review_status.items():
            if review_status.approved and story_id not in state.stories_published:
                approved.append(story_id)
        return approved

    @staticmethod
    def get_stories_needing_review(state: EditorState) -> list[str]:
        """Get list of story IDs that need review."""
        needs_review: list[str] = []
        for story_id in state.stories_collected:
            if story_id not in state.stories_review_status:
                needs_review.append(story_id)
        return needs_review

    @staticmethod
    def get_stories_needing_revision(state: EditorState) -> list[str]:
        """Get list of story IDs that are in revision state."""
        return list(state.stories_in_revision.keys())

    @staticmethod
    def is_cycle_complete(state: EditorState) -> bool:
        """Check if the news cycle is complete based on state."""
        # Check if we have enough published stories per field
        published_count = len(state.stories_published)
        target_count = len(state.requested_fields) * state.stories_per_field

        return published_count >= target_count

    @staticmethod
    def get_progress_summary(state: EditorState) -> dict[str, int]:
        """Get a summary of current progress."""
        return {
            "topics_collected": sum(len(topics) for topics in state.topics_collected.values()),
            "topics_assigned": len(state.topics_assigned),
            "stories_collected": len(state.stories_collected),
            "stories_reviewed": len(state.stories_review_status),
            "stories_approved": len([r for r in state.stories_review_status.values() if r.approved]),
            "stories_published": len(state.stories_published),
            "stories_in_revision": len(state.stories_in_revision),
        }

    @staticmethod
    def add_tool_error(state: EditorState, action_type: str, error_message: str) -> None:
        """Add a tool error to the state for feedback to LLM."""
        error_entry = f"❌ {action_type}: {error_message}"
        state.tool_errors.append(error_entry)
        state.last_failed_action = action_type

        # Keep only last 5 errors to avoid overwhelming the prompt
        if len(state.tool_errors) > 5:
            state.tool_errors = state.tool_errors[-5:]

    @staticmethod
    def clear_old_errors(state: EditorState) -> None:
        """Clear errors older than 3 iterations to keep prompt clean."""
        # Only keep errors if they're recent (last few actions)
        if len(state.action_history) > 3:
            state.tool_errors = []
            state.last_failed_action = None
