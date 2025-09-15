"""Agent factory for dynamic agent creation based on tasks."""

from agents.types import EconomicsSubSection, ReporterField, ScienceSubSection, TechnologySubSection
from core.config_service import ConfigService
from core.llm_service import LLMService
from core.logging_service import get_logger

# Import agents at module level - no lazy loading
from .editor_agent import EditorAgent
from .reporter_agent import ReporterAgent
from .task_execution_service import TaskExecutionService

logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating agents dynamically based on task requirements."""

    def __init__(self, config_service: ConfigService, llm_service: LLMService | None = None):
        """Initialize the agent factory.

        Args:
            config_service: Configuration service instance
            llm_service: LLM service for image generation
        """
        self.config_service = config_service
        self.llm_service = llm_service
        self._agent_counter = 0

        # Create task execution service with a bound method as the factory
        self.task_service = TaskExecutionService(self.create_reporter, llm_service)

    def create_reporter(
        self,
        field: ReporterField,
        sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None,
        reporter_id: str | None = None
    ) -> ReporterAgent:
        """Create a reporter agent for a specific field and optional sub-section.

        Args:
            field: The reporter field of expertise
            sub_section: Optional sub-section within the field for specialized reporting
            reporter_id: Optional custom reporter ID

        Returns:
            ReporterAgent instance configured for the field and sub-section
        """

        if reporter_id is None:
            self._agent_counter += 1
            sub_suffix = f"-{sub_section}" if sub_section else ""
            reporter_id = f"reporter-{field.value}{sub_suffix}-{self._agent_counter:03d}"

        logger.info(
            "Creating reporter agent",
            reporter_id=reporter_id,
            field=field.value,
            sub_section=sub_section or "none"
        )

        return ReporterAgent(
            field=field,
            sub_section=sub_section,
            reporter_id=reporter_id,
            config_service=self.config_service
        )

    def create_editor(self, editor_id: str | None = None) -> EditorAgent:
        """Create an editor agent.

        Args:
            editor_id: Optional custom editor ID

        Returns:
            EditorAgent instance
        """

        if editor_id is None:
            self._agent_counter += 1
            editor_id = f"editor-{self._agent_counter:03d}"

        logger.info(
            "Creating editor agent",
            editor_id=editor_id
        )

        return EditorAgent(
            task_service=self.task_service,
            editor_id=editor_id,
            config_service=self.config_service
        )
