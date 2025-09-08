"""Reporter state management models and functionality."""

from enum import StrEnum

from agents.models.search_models import SearchResultSummary
from agents.models.story_models import ResearchResult, StorySource, ToolCall
from agents.models.task_models import ReporterTask
from pydantic import BaseModel, Field


class TaskPhase(StrEnum):
    """Phase of task execution for reporter agents."""
    RESEARCH = "research"
    WRITING = "writing"
    COMPLETED = "completed"


class ReporterToolResult(BaseModel):
    """Result from tool execution for reporter agents."""
    tool_name: str
    success: bool
    result: str
    error: str | None = None
    iteration: int


def _default_research_results() -> list[ResearchResult]:
    return []


def _default_search_results() -> list[SearchResultSummary]:
    return []


def _default_sources() -> list[StorySource]:
    return []


def _default_tool_calls() -> list[ToolCall]:
    return []


def _default_tool_results() -> list[ReporterToolResult]:
    return []


class ReporterState(BaseModel):
    """Current state of a reporter agent's task execution."""

    # Task information
    current_task: ReporterTask
    task_phase: TaskPhase = TaskPhase.RESEARCH

    # Progress tracking
    iteration: int = 0
    max_iterations: int = 15
    research_iteration_count: int = 0

    # Research data
    research_results: list[ResearchResult] = Field(default_factory=_default_research_results)
    current_research: ResearchResult | None = None  # Most recent research result
    search_results: list[SearchResultSummary] = Field(default_factory=_default_search_results)
    accumulated_facts: list[str] = Field(default_factory=list)
    sources: list[StorySource] = Field(default_factory=_default_sources)

    # Tool execution history
    tool_calls: list[ToolCall] = Field(default_factory=_default_tool_calls)
    tool_results: list[ReporterToolResult] = Field(default_factory=_default_tool_results)

    # Error tracking
    errors: list[str] = Field(default_factory=list)
    last_error: str | None = None

    # Command history for agent actions
    command_history: list[str] = Field(default_factory=list)


class ReporterStateManager:
    """Helper class for managing reporter state operations."""

    @staticmethod
    def reset_for_new_task(state: ReporterState, task: ReporterTask) -> None:
        """Reset state for a new task execution."""
        state.current_task = task
        state.task_phase = TaskPhase.RESEARCH
        state.iteration = 0
        state.research_iteration_count = 0
        state.research_results.clear()
        state.search_results.clear()
        state.accumulated_facts.clear()
        state.sources.clear()
        state.tool_calls.clear()
        state.tool_results.clear()
        state.errors.clear()
        state.last_error = None
        state.command_history.clear()

    @staticmethod
    def advance_iteration(state: ReporterState) -> None:
        """Advance to next iteration."""
        state.iteration += 1

    @staticmethod
    def advance_research_iteration(state: ReporterState) -> None:
        """Advance research iteration counter."""
        state.research_iteration_count += 1

    @staticmethod
    def transition_to_writing_phase(state: ReporterState) -> None:
        """Transition from research to writing phase."""
        state.task_phase = TaskPhase.WRITING

    @staticmethod
    def mark_completed(state: ReporterState) -> None:
        """Mark task as completed."""
        state.task_phase = TaskPhase.COMPLETED

    @staticmethod
    def add_tool_call(state: ReporterState, tool_call: ToolCall) -> None:
        """Add a tool call to the history."""
        state.tool_calls.append(tool_call)

    @staticmethod
    def add_tool_result(state: ReporterState, tool_result: ReporterToolResult) -> None:
        """Add a tool result to the history."""
        tool_result.iteration = state.iteration
        state.tool_results.append(tool_result)

    @staticmethod
    def add_search_result(state: ReporterState, search_summary: SearchResultSummary) -> None:
        """Add search results to state."""
        state.search_results.append(search_summary)

    @staticmethod
    def set_current_research(state: ReporterState, research: ResearchResult) -> None:
        """Set the current research result."""
        # Set as current research
        state.current_research = research

        # Add to research results list
        state.research_results.append(research)

        # Accumulate facts
        state.accumulated_facts.extend(research.facts)

        # Accumulate sources (avoid duplicates)
        existing_urls = {source.url for source in state.sources}
        for source in research.sources:
            if source.url not in existing_urls:
                state.sources.append(source)
                existing_urls.add(source.url)

    @staticmethod
    def add_error(state: ReporterState, error_message: str) -> None:
        """Add an error to the state."""
        error_entry = f"Iteration {state.iteration}: {error_message}"
        state.errors.append(error_entry)
        state.last_error = error_message

        # Keep only last 5 errors to avoid overwhelming the prompt
        if len(state.errors) > 5:
            state.errors = state.errors[-5:]

    @staticmethod
    def add_command(state: ReporterState, command: str) -> None:
        """Add a command to the history."""
        command_entry = f"Iteration {state.iteration}: {command}"
        state.command_history.append(command_entry)

        # Keep only last 10 commands
        if len(state.command_history) > 10:
            state.command_history = state.command_history[-10:]

    @staticmethod
    def has_sufficient_research(state: ReporterState, min_sources: int) -> bool:
        """Check if we have sufficient research for the task."""
        return len(state.sources) >= min_sources and len(state.accumulated_facts) > 0

    @staticmethod
    def should_continue_research(state: ReporterState, max_research_iterations: int = 4) -> bool:
        """Check if we should continue research phase."""
        return (state.task_phase == TaskPhase.RESEARCH and
                state.research_iteration_count < max_research_iterations)

    @staticmethod
    def get_latest_research(state: ReporterState) -> ResearchResult | None:
        """Get the most recent research result."""
        return state.research_results[-1] if state.research_results else None

    @staticmethod
    def collect_all_sources_from_searches(state: ReporterState) -> list[StorySource]:
        """Collect all sources from search results."""
        from datetime import datetime

        from agents.models.story_models import StorySource

        all_sources: list[StorySource] = []
        seen_urls: set[str] = set()

        for search_summary in state.search_results:
            for result in search_summary.results:
                if hasattr(result, "url") and hasattr(result, "title") and result.url not in seen_urls:
                    source = StorySource(
                        url=result.url,
                        title=result.title,
                        summary=getattr(result, "snippet", None) or getattr(result, "body", None),
                        accessed_at=datetime.now()
                    )
                    all_sources.append(source)
                    seen_urls.add(result.url)

        return all_sources

    @staticmethod
    def get_progress_summary(state: ReporterState) -> dict[str, int | str]:
        """Get a summary of current progress."""
        return {
            "phase": state.task_phase.value,
            "iteration": state.iteration,
            "research_iterations": state.research_iteration_count,
            "research_results": len(state.research_results),
            "sources_collected": len(state.sources),
            "facts_collected": len(state.accumulated_facts),
            "tool_calls_made": len(state.tool_calls),
            "errors": len(state.errors)
        }
