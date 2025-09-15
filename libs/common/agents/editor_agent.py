"""Editor agent implementation - refactored for maintainability."""

# Re-export the main EditorAgent from the new modular structure
from agents.editor_agent.editor_agent_main import EditorAgent
from agents.editor_agent.editor_executor import EditorExecutor
from agents.editor_agent.editor_models import EditorAction, EditorDecision, EditorInfoResponse, NewspaperStatusResponse, StoryReview
from agents.editor_agent.editor_state import EditorState, EditorStateManager

# Keep backward compatibility
__all__ = [
    "EditorAction",
    "EditorAgent",
    "EditorDecision",
    "EditorExecutor",
    "EditorInfoResponse",
    "EditorState",
    "EditorStateManager",
    "NewspaperStatusResponse",
    "StoryReview"
]
