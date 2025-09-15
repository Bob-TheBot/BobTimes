"""Main Editor Agent implementation - refactored into focused components."""

from agents.agent import (
    AgentConfig,
    BaseAgent,
    NewsCycle,
)
from agents.editor_agent.editor_executor import EditorExecutor
from agents.editor_agent.editor_models import EditorDecision, EditorInfoResponse, NewspaperStatusResponse

# Import refactored components
from agents.editor_agent.editor_state import EditorState, EditorStateManager
from agents.editor_agent.editor_tools import AssignTopicsTool, CollectStoryTool, CollectTopicsTool, PublishStoryTool, ReviewStoryTool
from agents.models.submission_models import PublishedStory
from agents.models.task_models import JournalistTask
from agents.task_execution_service import TaskExecutionService
from agents.types import AgentType, EditorGoal, FieldTopicRequest, NewspaperSection, JournalistField, StoryPriority, TaskType
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from utils.topic_memory_service import TopicMemoryService

logger = get_logger(__name__)


class EditorAgent(BaseAgent):
    """Agent that reviews story submissions and makes editorial decisions."""

    def __init__(
        self,
        task_service: TaskExecutionService,
        editor_id: str | None = None,
        config_service: ConfigService | None = None
    ):
        """Initialize editor agent.

        Args:
            task_service: Task execution service for spawning reporter agents
            editor_id: Optional unique identifier for this editor
            config_service: Configuration service instance
        """
        self.editor_id = editor_id or "editor-chief-001"
        self.task_service = task_service
        self.orchestrator = EditorExecutor(task_service, self.editor_id)
        self.config_service = config_service or ConfigService()

        # Initialize topic memory service
        try:
            self.topic_memory = TopicMemoryService(self.config_service)
            logger.info(f"🧠 [EDITOR-{self.editor_id}] Topic memory service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize topic memory service: {e}")
            self.topic_memory = None

        # Initialize base agent with placeholder system prompt
        config = AgentConfig(
            system_prompt="Editor agent placeholder - will be updated after initialization",
            temperature=0.5,  # Lower temperature for more consistent editorial decisions
            default_model_speed=ModelSpeed.SLOW,  # Editorial decisions require careful analysis
            tools=[
                CollectTopicsTool(task_executor=task_service),
                AssignTopicsTool(task_executor=task_service),
                CollectStoryTool(task_executor=task_service),
                ReviewStoryTool(task_executor=task_service),
                PublishStoryTool(task_executor=task_service)
            ]
        )

        super().__init__(
            config=config,
            config_service=self.config_service
        )

        # Now update the system prompt with tools available
        self.config.system_prompt = self._create_editorial_prompt()

        logger.info(
            f"🎭 [EDITOR-{self.editor_id}] Editor agent initialized",
            agent_type="EDITOR",
            editor_id=self.editor_id,
            model_speed=config.default_model_speed.value,
            temperature=str(config.temperature)
        )

    def create_researcher_task_with_forbidden_topics(
        self,
        task_type: TaskType,
        field: JournalistField,
        description: str,
        topic: str | None = None,
        days_back: int = 30
    ) -> JournalistTask:
        """Create a reporter task with forbidden topics injected into guidelines.

        Args:
            task_type: Type of task (find_topics, write_story)
            field: Reporter field
            description: Task description
            topic: Optional topic for write_story tasks
            days_back: Number of days to look back for forbidden topics

        Returns:
            ReporterTask with forbidden topics in guidelines
        """
        # Get forbidden topics as formatted text
        forbidden_text = ""
        if self.topic_memory:
            forbidden_text = self.topic_memory.get_forbidden_topics_as_text(days_back)
        else:
            forbidden_text = "Topic memory not available - no forbidden topics to avoid."

        # Create enhanced guidelines with forbidden topics
        enhanced_guidelines = f"""
{description}

{forbidden_text}

IMPORTANT: Ensure your proposed topics are unique and not similar to the forbidden topics listed above.
Focus on fresh angles and new developments that haven't been covered recently.
"""

        return JournalistTask(
            agent_type=AgentType.RESEARCHER,
            name=task_type,
            field=field,
            description=description,
            guidelines=enhanced_guidelines.strip(),
            topic=topic
        )

    def _create_editorial_prompt(self, state: EditorState | None = None) -> str:
        """Create the system prompt for the editor agent.

        Args:
            state: Optional editor state to include in prompt
            
        Returns:
            System prompt string with base instructions, tool descriptions, and optional state
        """
        # Base editorial instructions
        base_prompt = """You are an experienced newspaper editor orchestrating a complete news cycle from topic collection to story publication.

# WORKFLOW SEQUENCE
Your news cycle follows this exact sequence:
1. COLLECT TOPICS → 2. ASSIGN TOPICS → 3. COLLECT STORIES → 4. REVIEW STORIES → 5. PUBLISH APPROVED STORIES
"""

        # Add tool descriptions dynamically from available tools
        tool_section = "\n# AVAILABLE TOOLS\n"
        for tool_name, tool in self.tools.items():
            tool_section += f"\n## {tool_name}\n{tool.description.strip()}\n"

        # Decision logic and rules
        decision_section = f"""
# DECISION LOGIC
Analyze the current state and determine the next logical action:

- If no topics collected: USE collect_topics
- If topics collected but none assigned: USE assign_topics  
- If topics assigned but no stories collected: USE collect_story
- If stories collected but not reviewed: USE review_story
- If stories approved but not published: USE publish_story
- If rejected stories need revision: USE collect_story with editor_remarks

# OUTPUT FORMAT
Return EditorDecision with:
{EditorDecision.model_json_schema()}

# CRITICAL RULES
- NEVER skip the review step - all stories must be reviewed before publishing
- ONLY publish APPROVED stories
- Use exact field names: "technology", "economics", "science"  
- When assigning topics, you only need field parameter - available_topics will be provided automatically
- Continue until all requested coverage targets are met
- Analyze state carefully before each decision"""

        # Combine base prompt with tools and rules
        prompt = base_prompt + tool_section + decision_section

        # Add state information if provided
        if state:
            state_section = self._format_state_for_prompt(state)
            prompt += f"\n\n# CURRENT STATE\n{state_section}"
            prompt += "\n\n# YOUR TASK\nAnalyze the current state above and return an EditorDecision with your next action."

        return prompt

    def _format_state_for_prompt(self, state: EditorState) -> str:
        """Format editor state information for inclusion in system prompt.

        Args:
            state: Current editor state
            
        Returns:
            Formatted state string for the prompt
        """
        # Use the existing orchestrator method to format state
        # This keeps the formatting logic in one place
        return self.orchestrator.build_orchestration_prompt(state)

    async def _generate_editorial_decision(self, state: EditorState) -> EditorDecision:
        """Generate an editorial decision from the LLM using proper async flow.

        Args:
            state: Current editor state
            
        Returns:
            EditorDecision from the LLM
        """
        # Build complete system prompt with state
        system_prompt = self._create_editorial_prompt(state)

        # Create proper message structure - LLM service requires ending with user message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Based on the current state analysis above, make your editorial decision."}
        ]

        # Use async LLM service
        decision: EditorDecision = await self.llm_service.generate(
            messages=messages,
            response_type=EditorDecision,
            model_speed=ModelSpeed.SLOW,
            temperature=self.config.temperature
        )

        return decision

    async def orchestrate_news_cycle(
        self,
        requested_fields: list[FieldTopicRequest],
        stories_per_field: int = 1,
        max_iterations: int = 20
    ) -> NewsCycle:
        """Orchestrate a complete news cycle from topic collection to story publishing.

        Args:
            requested_fields: List of field/topic requests for coverage
            stories_per_field: Number of stories needed per field
            max_iterations: Maximum orchestration iterations

        Returns:
            Complete news cycle with published stories
        """
        # Initialize state
        state = EditorState(
            current_goal=EditorGoal.COLLECT_TOPICS,
            requested_fields=requested_fields,
            stories_per_field=stories_per_field,
            max_iterations=max_iterations
        )

        logger.info(
            f"🎬 [EDITOR-{self.editor_id}] Starting news cycle orchestration",
            agent_type="EDITOR",
            editor_id=self.editor_id,
            requested_fields=str([f.field.value for f in requested_fields]),
            stories_per_field=stories_per_field,
            max_iterations=max_iterations
        )

        # Orchestration loop
        while state.current_iteration < state.max_iterations:
            try:
                # Get decision from LLM with proper message structure
                decision: EditorDecision = await self._generate_editorial_decision(state)

                logger.info(
                    f"🎯 [EDITOR-{self.editor_id}] DECISION MADE | Action: {decision.action.action_type.value}",
                    agent_type="EDITOR",
                    editor_id=self.editor_id,
                    action_type=decision.action.action_type.value,
                    reasoning_preview=decision.reasoning[:100] + "..." if len(decision.reasoning) > 100 else decision.reasoning,
                    will_continue=decision.continue_cycle
                )

                # Execute the chosen action
                await self.orchestrator.execute_editor_action(decision.action, state)

                # Update iteration count
                state.current_iteration += 1

                # Record action in history
                state.action_history.append(
                    f"Iteration {state.current_iteration}: {decision.action.action_type.value} - {decision.reasoning[:100]}"
                )

                # Check if we should continue
                if not decision.continue_cycle:
                    logger.info(
                        f"🏁 [EDITOR-{self.editor_id}] Decided to complete the cycle",
                        agent_type="EDITOR",
                        editor_id=self.editor_id,
                        final_iteration=state.current_iteration,
                        reason="Editor chose to stop cycle"
                    )
                    break

                # Check if cycle is complete
                if EditorStateManager.is_cycle_complete(state):
                    logger.info(
                        f"🏁 [EDITOR-{self.editor_id}] Cycle automatically completed",
                        agent_type="EDITOR",
                        editor_id=self.editor_id,
                        final_iteration=state.current_iteration,
                        reason="Target story count reached"
                    )
                    break

            except Exception as e:
                logger.error(
                    f"❌ [EDITOR-{self.editor_id}] Error in orchestration loop iteration {state.current_iteration + 1}: {e}",
                    agent_type="EDITOR",
                    editor_id=self.editor_id,
                    iteration=state.current_iteration + 1,
                    error=str(e)
                )
                break

        # Create final news cycle from state
        cycle = self._create_news_cycle_from_state(state)

        logger.info(
            f"🎬 [EDITOR-{self.editor_id}] News cycle orchestration completed",
            agent_type="EDITOR",
            editor_id=self.editor_id,
            iterations=state.current_iteration,
            published_count=len(cycle.published_stories) if cycle.published_stories else 0,
            max_iterations_used=state.max_iterations
        )

        return cycle

    def _create_news_cycle_from_state(self, state: EditorState) -> NewsCycle:
        """Create a NewsCycle from the final editor state.
        
        Args:
            state: Final editor state
            
        Returns:
            NewsCycle object with all published stories and metadata
        """
        published_stories = []

        for story_id in state.stories_published:
            if story_id in state.stories_collected:
                story_draft = state.stories_collected[story_id]

                # Get the reporter who wrote this story from topics_assigned
                reporter_id = "system"
                for rid in state.topics_assigned:
                    if rid in state.active_reporters:
                        reporter_id = rid
                        break

                # Map field to section (field is the main categorization)
                section_map = {
                    "technology": NewspaperSection.TECHNOLOGY,
                    "economics": NewspaperSection.ECONOMICS,
                    "science": NewspaperSection.SCIENCE
                }
                section = section_map.get(story_draft.field.value, NewspaperSection.TECHNOLOGY)

                # Determine priority (could be based on word count, sources, etc.)
                priority = StoryPriority.HIGH if len(story_draft.sources) > 3 else StoryPriority.MEDIUM

                published_story = PublishedStory(
                    story_id=story_id,
                    title=story_draft.title,
                    field=story_draft.field,
                    content=story_draft.content,
                    summary=story_draft.summary,
                    keywords=story_draft.keywords,
                    sources=story_draft.sources,
                    images=story_draft.suggested_images or [],
                    author=reporter_id,  # Actual reporter ID
                    reporter_id=reporter_id,  # Same as author
                    section=section,  # Map from field
                    priority=priority,  # Determine from content quality
                    word_count=story_draft.word_count
                )
                published_stories.append(published_story)

        return NewsCycle(
            cycle_id=f"cycle-{self.editor_id}-{state.current_iteration}",
            published_stories=published_stories
        )

    async def get_newspaper_status(self) -> NewspaperStatusResponse:
        """Get current status of the newspaper."""
        try:
            # Simple status check - newspaper is always active since we use file-based storage
            from agents.editor_agent.editor_tools import NewspaperFileStore

            store = NewspaperFileStore()
            # Try to read the store to verify it's accessible
            current_data = store.list_current()

            return NewspaperStatusResponse(
                content={
                    "status": "active",
                    "stories_count": str(len(current_data["front_page"]) + sum(len(stories) for stories in current_data["sections"].values())),
                    "front_page_count": str(len(current_data["front_page"]))
                }
            )

        except Exception as e:
            return NewspaperStatusResponse(error=str(e))

    def get_info(self) -> EditorInfoResponse:
        """Get information about this editor agent."""
        return EditorInfoResponse(
            id=self.editor_id,
            tools=[tool.__class__.__name__ for tool in self.tools],
            default_model_speed=self.config.default_model_speed.value,
            temperature=self.config.temperature
        )
