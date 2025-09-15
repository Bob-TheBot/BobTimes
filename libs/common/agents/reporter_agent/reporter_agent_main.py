"""Main Reporter Agent implementation - refactored into focused components."""

from agents.agent import AgentConfig, BaseAgent
from agents.models.story_models import ResearchResult, StoryDraft, TopicList
from agents.models.task_models import JournalistTask
from agents.reporter_agent.reporter_executor import ReporterTaskExecutor
from agents.reporter_agent.reporter_models import ReporterInfoResponse
from agents.reporter_agent.reporter_prompt import ReporterPromptBuilder
from agents.reporter_agent.reporter_tools import ReporterToolRegistry
from agents.types import (
    EconomicsSubSection,
    JournalistField,
    ScienceSubSection,
    TechnologySubSection,
)
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger

logger = get_logger(__name__)


class ReporterAgent(BaseAgent):
    """Agent that researches and writes news stories in a specific field."""

    def __init__(
        self,
        field: JournalistField,
        sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None,
        reporter_id: str | None = None,
        config_service: ConfigService | None = None
    ) -> None:
        """Initialize reporter agent with a specific field of expertise.

        Args:
            field: The field this reporter specializes in
            sub_section: Optional sub-section within the field for specialized reporting
            reporter_id: Optional unique identifier for this reporter
            config_service: Configuration service instance
        """
        self.field = field
        self.sub_section = sub_section
        sub_id = f"-{sub_section}" if sub_section else ""
        self.reporter_id = reporter_id or f"reporter-{field.value}{sub_id}-001"

        # Initialize tool registry and prompt builder
        self.tool_registry = ReporterToolRegistry(config_service)
        self.prompt_builder = ReporterPromptBuilder(self.tool_registry)

        # Create configuration with system prompt
        config = AgentConfig(
            system_prompt=self.prompt_builder.create_system_prompt(field, sub_section),
            tools=self.tool_registry.get_all_tools(),
            max_iterations=15,
            temperature=0.7,
            default_model_speed=ModelSpeed.SLOW  # Reporting requires thorough research
        )

        super().__init__(config, config_service or ConfigService())

        # Initialize task executor
        self.executor = ReporterTaskExecutor(
            agent=self,
            tool_registry=self.tool_registry,
            prompt_builder=self.prompt_builder,
            reporter_id=self.reporter_id
        )

        logger.info(
            "Initialized reporter agent",
            field=field.value,
            sub_section=sub_section.value if sub_section else "None",
            reporter_id=self.reporter_id
        )

    async def execute_task(
        self,
        task: JournalistTask,
        model_speed: ModelSpeed | None = None
    ) -> StoryDraft | TopicList | ResearchResult:
        """Execute a reporter task using the task executor.

        Args:
            task: The reporting task from editor
            model_speed: Optional model speed override for this task
            
        Returns:
            Response based on task type (StoryDraft, TopicList, or ResearchResult)
        """
        return await self.executor.execute_task(task, model_speed)


    def get_reporter_info(self) -> ReporterInfoResponse:
        """Get information about this reporter.

        Returns:
            Structured reporter information
        """
        return ReporterInfoResponse(
            id=self.reporter_id,
            field=self.field.value,
            sub_section=self.sub_section.value if self.sub_section else None,
            tools=[tool.name for tool in self.config.tools],
            default_model_speed=self.config.default_model_speed.value,
            temperature=self.config.temperature
        )
