"""Reporter agent implementation - refactored for maintainability."""

# Re-export the main ReporterAgent from the new modular structure
from agents.reporter_agent.reporter_agent_main import ReporterAgent
from agents.reporter_agent.reporter_executor import ReporterTaskExecutor
from agents.reporter_agent.reporter_models import ReporterExecutionResult, ReporterInfoResponse, ReporterTaskSummary
from agents.reporter_agent.reporter_prompt import ReporterPromptBuilder
from agents.reporter_agent.reporter_state import ReporterState, ReporterStateManager, TaskPhase
from agents.reporter_agent.reporter_tools import ReporterToolRegistry

# Keep backward compatibility
__all__ = [
    "ReporterAgent",
    "ReporterExecutionResult",
    "ReporterInfoResponse",
    "ReporterPromptBuilder",
    "ReporterState",
    "ReporterStateManager",
    "ReporterTaskExecutor",
    "ReporterTaskSummary",
    "ReporterToolRegistry",
    "TaskPhase"
]
