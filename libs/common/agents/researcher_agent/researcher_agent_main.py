"""Main Researcher Agent implementation - refactored into focused components."""

from agents.agent import AgentConfig, BaseAgent
from agents.models.story_models import ResearchResult, StoryDraft, TopicList
from agents.models.task_models import JournalistTask
from agents.researcher_agent.researcher_executor import JournalistTaskExecutor
from agents.researcher_agent.researcher_models import ResearcherInfoResponse
from agents.researcher_agent.researcher_prompt import ResearcherConfiguration, ResearcherPromptBuilder
from agents.researcher_agent.researcher_tools import ResearcherToolRegistry
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


class ResearcherAgent(BaseAgent):
    """Agent that researches and writes news stories in a specific field."""

    def __init__(
        self,
        field: JournalistField,
        sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None,
        researcher_id: str | None = None,
        config_service: ConfigService | None = None
    ) -> None:
        """Initialize researcher agent with a specific field of expertise.

        Args:
            field: The field this researcher specializes in
            sub_section: Optional sub-section within the field for specialized reporting
            researcher_id: Optional unique identifier for this researcher
            config_service: Configuration service instance
        """
        self.field = field
        self.sub_section = sub_section
        sub_id = f"-{sub_section}" if sub_section else ""
        self.researcher_id = researcher_id or f"researcher-{field.value}{sub_id}-001"

        # Initialize tool registry and prompt builder
        self.tool_registry = ResearcherToolRegistry(config_service)
        self.prompt_builder = ResearcherPromptBuilder(self.tool_registry)

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
        self.executor = JournalistTaskExecutor(
            agent=self,
            tool_registry=self.tool_registry,
            prompt_builder=self.prompt_builder,
            researcher_id=self.researcher_id
        )

        logger.info(
            "Initialized researcher agent",
            field=field.value,
            sub_section=sub_section.value if sub_section else "None",
            researcher_id=self.researcher_id
        )

    async def execute_task(
        self,
        task: JournalistTask,
        model_speed: ModelSpeed | None = None
    ) -> StoryDraft | TopicList | ResearchResult:
        """Execute a researcher task using the task executor.

        Args:
            task: The reporting task from editor
            model_speed: Optional model speed override for this task
            
        Returns:
            Response based on task type (StoryDraft, TopicList, or ResearchResult)
        """
        return await self.executor.execute_task(task, model_speed)

    async def find_trending_topics(self, max_topics: int = 5) -> list[str]:
        """Find trending topics in the researcher's field.

        Args:
            max_topics: Maximum number of topics to return

        Returns:
            List of trending topic suggestions
        """
        logger.info(
            "Finding trending topics",
            field=self.field.value,
            sub_section=self.sub_section.value if self.sub_section else "none",
            max_topics=max_topics
        )

        # Use search tool to find trending topics
        search_tool = self.tool_registry.get_tool_by_name("search")
        if not search_tool:
            return []

        # Create search params for trending topics
        from agents.researcher_agent.researcher_tools import SearchParams, SearchType
        from utils.duckduckgo_search_tool import DDGTimeLimit

        search_query = self._create_search_query("trending news today")
        search_params = SearchParams(
            query=search_query,
            search_type=SearchType.NEWS,
            max_results=max_topics * 2,
            time_limit=DDGTimeLimit.week
        )

        search_result = await search_tool.execute(search_params)

        # Extract unique topics from results
        topics: list[str] = []
        seen_topics: set[str] = set()

        if search_result.success and search_result.news_results:
            for r in search_result.news_results:
                title = r.title
                if title and title not in seen_topics:
                    topics.append(title)
                    seen_topics.add(title)

                    if len(topics) >= max_topics:
                        break

        logger.info(f"Found {len(topics)} trending topics")
        return topics

    async def propose_multiple_topics(self, count: int = 5) -> list[str]:
        """Propose multiple topics for editor selection to prevent conflicts.

        Args:
            count: Number of topics to propose

        Returns:
            List of proposed topics for this researcher's field
        """
        logger.info(
            "Proposing multiple topics for editor selection",
            field=self.field.value,
            count=count
        )

        # Use the existing find_trending_topics method but get more topics
        # to give the editor more options for conflict prevention
        topics = await self.find_trending_topics(max_topics=count)

        logger.info(
            "Proposed topics for editor selection",
            field=self.field.value,
            topics_count=len(topics)
        )

        return topics

    def get_researcher_info(self) -> ResearcherInfoResponse:
        """Get information about this researcher.

        Returns:
            Structured researcher information
        """
        return ResearcherInfoResponse(
            id=self.researcher_id,
            field=self.field.value,
            sub_section=self.sub_section.value if self.sub_section else None,
            tools=[tool.name for tool in self.config.tools],
            default_model_speed=self.config.default_model_speed.value,
            temperature=self.config.temperature
        )

    def _create_search_query(self, base_query: str) -> str:
        """Create a search query optimized for the researcher's field and sub-section.

        Args:
            base_query: Base search terms

        Returns:
            Optimized search query with field and sub-section context
        """
        # Start with field context
        query_parts = [self.field.value, base_query]

        # Add sub-section specific terms if available
        if self.sub_section:
            sub_section_terms = self._get_sub_section_search_terms()
            if sub_section_terms:
                query_parts.insert(1, sub_section_terms)

        return " ".join(query_parts)

    def _get_sub_section_search_terms(self) -> str:
        """Get search terms specific to a sub-section using the configuration system.

        Returns:
            Search terms for the sub-section
        """
        if self.sub_section:
            return ResearcherConfiguration.get_sub_section_search_terms(self.sub_section)
        return ""
