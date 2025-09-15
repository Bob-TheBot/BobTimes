"""Task execution service for coordinating agent tasks without circular dependencies."""

import asyncio
from collections.abc import Callable
from typing import Any

from agents.image_processor import process_story_images
from agents.models.story_models import ResearchResult, StoryDraft, TopicList
from agents.models.task_models import ReporterTask
from agents.types import EconomicsSubSection, ReporterField, ScienceSubSection, TaskType, TechnologySubSection
from core.exceptions import ResearchFailureError
from core.llm_service import LLMService, ModelSpeed
from core.logging_service import get_logger

logger = get_logger(__name__)


class TaskExecutionService:
    """Service for executing tasks by dynamically spawning appropriate agents."""

    def __init__(self, create_reporter: Callable[[ReporterField, TechnologySubSection | EconomicsSubSection | ScienceSubSection |  None], Any], llm_service: LLMService | None = None):
        """Initialize the task execution service.

        Args:
            create_reporter: Factory function for creating reporter agents (field, sub_section)
            llm_service: LLM service for image generation
        """
        self.create_reporter = create_reporter
        self.llm_service = llm_service

    async def execute_reporter_task(self, task: ReporterTask, model_speed: ModelSpeed = ModelSpeed.FAST, max_retries: int = 2) -> StoryDraft | TopicList | ResearchResult:
        """Execute a reporter task by spawning a temporary reporter agent with retry logic.

        Args:
            task: The reporter task to execute
            model_speed: Model speed preference for LLM operations
            max_retries: Maximum number of retry attempts on failure (default: 2)

        Returns:
            Task result (StoryDraft, TopicList, or ResearchResult)
        """
        logger.info(
            f"🎯 Spawning reporter for task: {task.name.value}",
            task_type=task.name.value,
            field=task.field.value,
            sub_section=task.sub_section.value if task.sub_section else "none",
            description=task.description[:100] + "..." if len(task.description) > 100 else task.description,
            topic=(task.topic[:50] + "..." if task.topic and len(task.topic) > 50 else task.topic) if task.topic else "No specific topic",
            max_retries=max_retries
        )

        last_exception = None

        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            if is_retry:
                logger.info(
                    f"🔄 Retrying reporter task (attempt {attempt + 1}/{max_retries + 1})",
                    task_type=task.name.value,
                    field=task.field.value,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1
                )
                # Add a small delay between retries
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s...

            try:
                # Spawn reporter for this specific task with sub-section if available
                reporter = self.create_reporter(task.field, task.sub_section)

                logger.info(
                    f"👤 Reporter agent created for {task.field.value}",
                    field=task.field.value,
                    reporter_id=reporter.reporter_id if hasattr(reporter, "reporter_id") else "unknown",
                    attempt=attempt + 1 if is_retry else 1
                )

                # Execute the task with specified model speed
                result = await reporter.execute_task(task, model_speed=model_speed)

                # Post-process images for story tasks
                if task.name == TaskType.WRITE_STORY and isinstance(result, StoryDraft) and self.llm_service:
                    logger.info("🖼️ Processing images for story draft")
                    try:
                        await process_story_images(
                            story_draft=result,
                            search_results=reporter.search_results if hasattr(reporter, "search_results") else [],
                            llm_service=self.llm_service
                        )
                        logger.info(
                            "✅ Image processing completed",
                            images_count=len(result.suggested_images)
                        )
                    except Exception as e:
                        logger.exception(f"❌ Image processing failed: {e}")

                # Log detailed completion info based on result type
                if isinstance(result, StoryDraft):
                    logger.info(
                        "✅ Story task completed successfully",
                        task_type=task.name.value,
                        field=task.field.value,
                        title=result.title[:50] + "..." if len(result.title) > 50 else result.title,
                        word_count=result.word_count,
                        sources_count=len(result.sources),
                        images_count=len(result.suggested_images),
                        attempt=attempt + 1 if is_retry else 1
                    )
                elif isinstance(result, TopicList):
                    logger.info(
                        "✅ Topic collection task completed successfully",
                        task_type=task.name.value,
                        field=task.field.value,
                        topics_count=len(result.topics),
                        topics=", ".join(result.topics[:3]) + ("..." if len(result.topics) > 3 else ""),
                        attempt=attempt + 1 if is_retry else 1
                    )
                elif isinstance(result, ResearchResult):
                    logger.info(
                        "✅ Research task completed successfully",
                        task_type=task.name.value,
                        field=task.field.value,
                        sources_count=len(result.sources),
                        facts_count=len(result.facts) if hasattr(result, "facts") else 0,
                        attempt=attempt + 1 if is_retry else 1
                    )
                else:
                    logger.info(
                        "✅ Reporter task completed successfully",
                        task_type=task.name.value,
                        field=task.field.value,
                        result_type=type(result).__name__,
                        attempt=attempt + 1 if is_retry else 1
                    )

                return result

            except ResearchFailureError as e:
                last_exception = e
                logger.error(
                    f"❌ Reporter task failed due to research failure (attempt {attempt + 1}/{max_retries + 1}): {task.name.value}",
                    task_type=task.name.value,
                    field=task.field.value,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                    error_code=e.error_code or "RESEARCH_FAILURE",
                    error_message=e.message,
                    research_iterations=e.details.get("research_iterations", 0),
                    topic=e.details.get("topic", "Unknown"),
                    reporter_id=e.details.get("reporter_id", "Unknown"),
                    min_sources_required=e.details.get("min_sources_required", 0)
                )

                # If this is the last attempt, re-raise the exception
                if attempt >= max_retries:
                    logger.error(
                        f"❌ All retry attempts exhausted for research failure: {task.name.value}",
                        task_type=task.name.value,
                        field=task.field.value,
                        total_attempts=max_retries + 1
                    )
                    raise

                # Otherwise, continue to next retry iteration
                continue

            except Exception as e:
                last_exception = e
                logger.error(
                    f"❌ Reporter task failed (attempt {attempt + 1}/{max_retries + 1}): {task.name.value}",
                    task_type=task.name.value,
                    field=task.field.value,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                    error=str(e)
                )

                # If this is the last attempt, re-raise the exception
                if attempt >= max_retries:
                    logger.error(
                        f"❌ All retry attempts exhausted for task failure: {task.name.value}",
                        task_type=task.name.value,
                        field=task.field.value,
                        total_attempts=max_retries + 1
                    )
                    raise

                # Otherwise, continue to next retry iteration
                continue

        # This should never be reached due to the raise statements above
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Unexpected end of retry loop for task: {task.name.value}")


