"""Newspaper service - API entry point for news generation.

This service acts as a thin layer between the API and the editor agent.
All orchestration logic is handled by the EditorAgent.
"""

from uuid import uuid4

from agents.agent_factory import AgentFactory
from agents.editor_agent import EditorAgent
from agents.models.cycle_models import NewsCycle
from agents.models.performance_models import AgentPerformance, AgentType
from agents.task_execution_service import TaskExecutionService
from agents.editor_agent.editor_tools import NewspaperFileStore
from agents.types import CycleStatus, ReporterField, FieldTopicRequest
from core.config_service import ConfigService
from core.llm_service import LLMService
from core.logging_service import get_logger

from ..models.newspaper_models import NewspaperContent

logger = get_logger(__name__)


class NewspaperService:
    """Service for managing newspaper generation through the editor agent.
    
    This service is a thin API layer that delegates all orchestration
    to the EditorAgent. It maintains the current newspaper state for
    serving to the UI.
    """

    def __init__(self, config_service: ConfigService, llm_service: LLMService | None = None):
        """Initialize the newspaper service.

        Args:
            config_service: Configuration service instance
            llm_service: LLM service for image generation
        """
        self.config_service = config_service
        self.llm_service = llm_service
        self.agent_factory = AgentFactory(config_service, llm_service)
        self.task_service = TaskExecutionService(self.agent_factory.create_reporter, llm_service)
        self.editor: EditorAgent | None = None
        self.current_cycle: NewsCycle | None = None
        self.performance_metrics: dict[str, AgentPerformance] = {}
        
        logger.info("NewspaperService initialized")

    async def run_news_cycle(
        self,
        field_requests: list[FieldTopicRequest] | None = None,
        stories_per_field: int = 1,
        sequential: bool = True
    ) -> NewsCycle:
        """Run a complete news generation cycle using the editor's orchestration.

        Args:
            field_requests: List of FieldTopicRequest objects to create stories for
            stories_per_field: Number of stories to generate per field/sub-section
            sequential: Whether to generate stories sequentially or in parallel

        Returns:
            Completed news cycle with all stories and decisions
        """
        # Default field requests if none provided
        if field_requests is None:
            field_requests = [
                FieldTopicRequest(field=ReporterField.TECHNOLOGY),
                FieldTopicRequest(field=ReporterField.SCIENCE),
                FieldTopicRequest(field=ReporterField.ECONOMICS)
            ]
        
        logger.info(
            "Starting news cycle",
            field_requests=str([(req.field.value, req.sub_section) for req in field_requests]),
            stories_per_field=stories_per_field,
            sequential=sequential
        )

        try:
            # Initialize editor if needed
            if not self.editor:
                self.editor = EditorAgent(
                    task_service=self.task_service,
                    config_service=self.config_service
                )
                self.performance_metrics[self.editor.editor_id] = AgentPerformance(
                    agent_id=self.editor.editor_id,
                    agent_type=AgentType.EDITOR
                )

            # For now, delegate to editor's orchestration with just the fields
            # TODO: Update editor to handle sub-sections fully
            if sequential:
                # Process field requests one by one for sequential generation
                all_stories = []
                last_cycle = None
                for req in field_requests:
                    logger.info(f"Generating content for {req.field.value}" + (f" ({req.sub_section})" if req.sub_section else ""))
                    single_cycle = await self.editor.orchestrate_news_cycle(
                        requested_fields=[req],
                        stories_per_field=stories_per_field
                    )
                    all_stories.extend(single_cycle.published_stories)
                    last_cycle = single_cycle
                
                # Create combined cycle
                cycle = last_cycle or NewsCycle(cycle_id="empty", cycle_status=CycleStatus.COMPLETED)
                cycle.published_stories = all_stories
            else:
                # Delegate to editor's orchestration for parallel processing
                cycle = await self.editor.orchestrate_news_cycle(
                    requested_fields=field_requests,
                    stories_per_field=stories_per_field
                )

            # Store the cycle for serving to UI
            self.current_cycle = cycle

            logger.info(
                "News cycle completed",
                cycle_id=str(cycle.cycle_id or ""),
                stories_published=len(cycle.published_stories)
            )

            return cycle

        except Exception as e:
            logger.error(f"News cycle failed: {e}")
            
            # Create minimal failed cycle for API response
            if not self.current_cycle:
                self.current_cycle = NewsCycle(
                    cycle_id=f"failed-{uuid4().hex[:8]}",
                    cycle_status=CycleStatus.FAILED
                )
            else:
                self.current_cycle.cycle_status = CycleStatus.FAILED
            
            raise

    async def get_newspaper_content(self) -> NewspaperContent:
        """Get the current newspaper content for the UI.

        Returns:
            Current newspaper content with all published stories
        """
        if not self.current_cycle:
            # Fallback to reading from newspaper.json file
            return self._get_newspaper_content_from_file()

        # Convert the current cycle to newspaper content
        newspaper = NewspaperContent(
            title="Bob Times",
            tagline="AI-Generated News That Matters",
            stories=self.current_cycle.published_stories,
            metadata={
                "cycle_id": str(self.current_cycle.cycle_id or ""),
                "status": self.current_cycle.cycle_status.value,
                "generated_at": self.current_cycle.start_time.isoformat() if self.current_cycle.start_time else None,
                "total_stories": len(self.current_cycle.published_stories),
                "total_submissions": len(self.current_cycle.submissions),
                "total_decisions": len(self.current_cycle.editorial_decisions)
            }
        )

        logger.info(
            "Serving newspaper content",
            stories=len(newspaper.stories),
            cycle_id=newspaper.metadata.get("cycle_id", "unknown")
        )

        return newspaper

    def _get_newspaper_content_from_file(self) -> NewspaperContent:
        """Get newspaper content from the file store as fallback.

        Returns:
            Newspaper content from the file store
        """
        try:
            # Create file store instance
            file_store = NewspaperFileStore()

            logger.info(f"File store path: {file_store.path}")
            logger.info(f"File exists: {file_store.path.exists()}")

            # Test reading data through public method
            try:
                logger.info("Testing file store data loading...")
            except Exception as e:
                logger.error(f"Error testing file store: {e}")
                raise

            # Get current stories from file store
            logger.info("Calling list_current()...")
            current_data = file_store.list_current()
            logger.info("list_current() completed")

            logger.info(
                "File store data retrieved",
                front_page_count=len(current_data["front_page"]),
                sections_count=sum(len(stories) for stories in current_data["sections"].values())
            )
            logger.info(f"Sections available: {list(current_data['sections'].keys())}")

            # Convert Story objects to PublishedStory objects
            published_stories = []

            # Add front page stories
            for story in current_data["front_page"]:
                try:
                    published_story = story.to_published_story()
                    published_stories.append(published_story)
                    logger.info(f"Converted front page story: {story.id} - {story.title}")
                except Exception as e:
                    logger.error(f"Error converting front page story {story.id}: {e}")

            # Add stories from all sections (avoid duplicates)
            added_story_ids = {story.id for story in current_data["front_page"]}
            for section_stories in current_data["sections"].values():
                for story in section_stories:
                    if story.id not in added_story_ids:
                        try:
                            published_story = story.to_published_story()
                            published_stories.append(published_story)
                            added_story_ids.add(story.id)
                            logger.info(f"Converted section story: {story.id} - {story.title}")
                        except Exception as e:
                            logger.error(f"Error converting section story {story.id}: {e}")

            logger.info(
                "Serving newspaper content from file store",
                stories=len(published_stories)
            )

            return NewspaperContent(
                title="Bob Times",
                tagline="Your Source for News",
                stories=published_stories,
                metadata={
                    "status": "Loaded from file store",
                    "total_stories": len(published_stories),
                    "source": "newspaper.json"
                }
            )

        except Exception as e:
            logger.error(f"Error reading from newspaper file store: {e}")
            # Return empty newspaper as final fallback
            return NewspaperContent(
                title="Bob Times",
                tagline="Your Source for News",
                stories=[],
                metadata={
                    "status": "Error loading content",
                    "error": str(e)
                }
            )

    def get_performance_metrics(self) -> dict[str, AgentPerformance]:
        """Get performance metrics for all agents.
        
        Returns:
            Dictionary of agent performance metrics
        """
        return self.performance_metrics