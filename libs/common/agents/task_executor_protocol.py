"""Protocol for task execution to avoid circular dependencies."""

from typing import Protocol, runtime_checkable

from agents.models.story_models import ResearchResult, StoryDraft, TopicList
from agents.models.task_models import ReporterTask
from core.llm_service import ModelSpeed


@runtime_checkable
class TaskExecutor(Protocol):
    """Protocol for executing reporter tasks."""

    async def execute_reporter_task(self, task: ReporterTask, model_speed: ModelSpeed = ModelSpeed.FAST, max_retries: int = 2) -> StoryDraft | TopicList | ResearchResult:
        """Execute a reporter task with retry logic.
        
        Args:
            task: The reporter task to execute
            model_speed: Model speed preference for LLM operations
            max_retries: Maximum number of retry attempts on failure (default: 2)
            
        Returns:
            Task result (StoryDraft, TopicList, or ResearchResult)
        """
        ...
