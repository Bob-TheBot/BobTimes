"""Reporter agent task execution logic."""

from typing import Any

from agents.agent import BaseAgent
from agents.agent import ToolCall as BaseToolCall
from agents.models.story_models import AgentResponse, ResearchResult, StoryDraft, StorySource, TopicList
from agents.models.story_models import ToolCall as StoryToolCall
from agents.models.task_models import ReporterTask
from agents.reporter_agent.reporter_prompt import ReporterPromptBuilder
from agents.reporter_agent.reporter_state import ReporterState, ReporterStateManager, ReporterToolResult, TaskPhase
from agents.reporter_agent.reporter_tools import ReporterToolRegistry
from agents.types import TaskType
from core.exceptions import ResearchFailureError
from core.llm_service import ModelSpeed
from core.logging_service import get_logger

logger = get_logger(__name__)


class ReporterTaskExecutor:
    """Handles the execution logic for reporter tasks."""

    def __init__(
        self,
        agent: BaseAgent,
        tool_registry: ReporterToolRegistry,
        prompt_builder: ReporterPromptBuilder,
        reporter_id: str
    ):
        """Initialize the task executor.
        
        Args:
            agent: Base agent instance with LLM service
            tool_registry: Registry of reporter tools
            prompt_builder: Prompt building service
            reporter_id: Unique reporter identifier
        """
        self.agent = agent
        self.tool_registry = tool_registry
        self.prompt_builder = prompt_builder
        self.reporter_id = reporter_id

    async def execute_task(
        self,
        task: ReporterTask,
        model_speed: ModelSpeed | None = None
    ) -> StoryDraft | TopicList | ResearchResult:
        """Execute a reporter task with proper state management.
        
        Args:
            task: The task to execute
            model_speed: Optional model speed override
            
        Returns:
            Task result based on task type
            
        Raises:
            ResearchFailureError: If research phase fails completely
            ValueError: If task cannot be completed within iteration limits
        """
        # Initialize state for this task
        state = ReporterState(
            current_task=task,
            max_iterations=15
        )

        # For WRITE_STORY tasks, do NOT automatically inject memory sources
        # Let the agent choose to use fetch_from_memory tool instead
        if task.name == TaskType.WRITE_STORY and task.topic:
            logger.info(
                f"� [REPORTER-{self.reporter_id}] Starting WRITE_STORY task for topic: {task.topic}",
                topic=task.topic,
                note="Agent will choose whether to use fetch_from_memory tool or research fresh content"
            )

        logger.info(
            f"📰 [REPORTER-{self.reporter_id}] Starting task: {task.name.value}",
            agent_type="REPORTER",
            task_type=task.name.value,
            field=task.field.value,
            reporter_id=self.reporter_id,
            description=task.description[:100] + "..." if len(task.description) > 100 else task.description,
            topic=(task.topic[:100] + "..." if task.topic and len(task.topic) > 100 else task.topic or "No specific topic")
        )

        # Do NOT automatically load shared memories - let agent choose to use fetch_from_memory tool
        # self._load_shared_memories(state, task)  # Commented out to force tool usage

        # Main execution loop
        while state.iteration < state.max_iterations:
            ReporterStateManager.advance_iteration(state)


            # Check research limits for WRITE_STORY tasks
            if task.name == TaskType.WRITE_STORY and state.task_phase == TaskPhase.RESEARCH:
                if not self._check_research_limits(state, task):
                    break

            # Build complete prompt for current state
            # Use the system prompt from agent config (already contains tool definitions)
            system_prompt = self.agent.config.system_prompt
            task_prompt = self.prompt_builder.build_task_prompt(state)
            complete_prompt = f"{system_prompt}\n\n{task_prompt}"

            try:
                # Generate response from LLM
                response = await self._generate_agent_response(
                    complete_prompt, state, model_speed
                )


                # Handle tool calls
                if response.tool_call:
                    await self._handle_tool_execution(response.tool_call, state)
                    continue

                # Handle final responses
                if final_result := self._handle_final_response(response, state, task):
                    return final_result


            except Exception as e:
                error_msg = f"Failed in iteration {state.iteration}: {e}"
                logger.exception(error_msg)  # Use exception() to get full stack trace
                ReporterStateManager.add_error(state, error_msg)

                if state.iteration >= state.max_iterations:
                    raise

        raise ValueError(f"Failed to complete task within {state.max_iterations} iterations")

    def _check_research_limits(self, state: ReporterState, task: ReporterTask) -> bool:
        """Check if research limits have been exceeded.
        
        Args:
            state: Current reporter state
            task: Current task
            
        Returns:
            True if execution should continue, False if limits exceeded
            
        Raises:
            ResearchFailureError: If research has completely failed
        """
        if (state.research_iteration_count >= 4 and
            len(state.sources) < task.min_sources):

            logger.error(
                "Research iteration limit reached without sufficient sources",
                research_iterations=state.research_iteration_count,
                sources_found=len(state.sources),
                min_sources_required=task.min_sources
            )

            # Complete failure - no sources at all
            if len(state.sources) == 0:
                logger.error(
                    f"❌ [REPORTER-{self.reporter_id}] Research failed - no usable sources found",
                    agent_type="REPORTER",
                    task_type=task.name.value,
                    field=task.field.value,
                    reporter_id=self.reporter_id,
                    research_iterations=state.research_iteration_count,
                    facts_found=len(state.accumulated_facts),
                    topic=task.topic or "Unknown"
                )
                raise ResearchFailureError(
                    message=f"Research failed - no usable sources found after {state.research_iteration_count} iterations",
                    task_type=task.name.value,
                    field=task.field.value,
                    research_iterations=state.research_iteration_count,
                    details={
                        "topic": task.topic or "Unknown",
                        "reporter_id": self.reporter_id,
                        "min_sources_required": task.min_sources,
                        "facts_found": len(state.accumulated_facts)
                    }
                )

            return False

        return True

    async def _generate_agent_response(
        self,
        prompt: str,
        state: ReporterState,
        model_speed: ModelSpeed | None = None
    ) -> AgentResponse:
        """Generate structured response from LLM.
        
        Args:
            prompt: Complete prompt for the LLM
            state: Current reporter state
            model_speed: Optional model speed override
            
        Returns:
            Structured AgentResponse
        """
        # Determine effective model speed
        effective_model_speed = self._determine_model_speed(state, model_speed)

        try:
            response = await self.agent.generate_structured_response(
                prompt=prompt,
                response_type=AgentResponse,
                model_speed=effective_model_speed,
                temperature=self.agent.config.temperature
            )

            # Ensure iteration metadata is set
            response.iteration = state.iteration
            response.max_iterations = state.max_iterations

            return response

        except Exception as e:
            logger.error(f"Failed to generate agent response: {e}")
            # Return a basic response to avoid breaking the loop
            return AgentResponse(
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                reasoning=f"Error generating response: {e!s}"
            )

    async def _handle_tool_execution(self, tool_call: StoryToolCall, state: ReporterState) -> None:
        """Execute a tool call and update state.
        
        Args:
            tool_call: Tool call to execute
            state: Current reporter state
        """
        # Add tool call to state
        ReporterStateManager.add_tool_call(state, tool_call)
        ReporterStateManager.add_command(
            state, f"Tool call: {tool_call.name} with params: {tool_call.parameters}"
        )

        # Convert to BaseAgent ToolCall format
        base_tool_call = BaseToolCall(name=tool_call.name, arguments=tool_call.parameters)

        # Execute the tool
        result = await self.agent.execute_single_tool_call(base_tool_call)

        # Create tool result
        tool_result = ReporterToolResult(
            tool_name=tool_call.name,
            success=not isinstance(result, dict) or "error" not in result,
            result=str(result),  # Keep string for logging/display
            error=result.get("error") if isinstance(result, dict) and "error" in result else None,
            iteration=state.iteration
        )

        # Add result to state
        ReporterStateManager.add_tool_result(state, tool_result)

        # Handle unified tool results - all tools now return UnifiedToolResult
        if tool_result.success and hasattr(result, 'sources') and hasattr(result, 'topic_source_mapping'):
            await self._handle_unified_tool_result(result, state)
        elif tool_result.success and tool_call.name == "fetch_from_memory":
            # Handle fetch_from_memory results separately
            self._handle_fetch_from_memory_result(result, state)

        if tool_result.error:
            ReporterStateManager.add_error(state, f"Tool {tool_call.name} failed: {tool_result.error}")

    async def _handle_unified_tool_result(self, result: Any, state: ReporterState) -> None:
        """Handle unified tool results from any tool (search, youtube, scraper).

        Args:
            result: UnifiedToolResult from any tool
            state: Current reporter state
        """
        operation = getattr(result, "operation", "unknown")
        logger.info(
            f"🔧 [REPORTER-{self.reporter_id}] Processing unified tool result",
            operation=operation,
            sources_count=len(result.sources) if hasattr(result, 'sources') else 0,
            topics_count=len(result.topics_extracted) if hasattr(result, 'topics_extracted') else 0,
            has_topic_mapping=bool(getattr(result, 'topic_source_mapping', None))
        )

        # Store sources in SharedMemoryStore using topic mapping
        if hasattr(result, "sources") and result.sources:
            topic_mapping = getattr(result, "topic_source_mapping", None)

            # For search results, check if we need to enhance content with scraping
            if operation == "search" and topic_mapping:
                enhanced_sources, enhanced_mapping = await self._enhance_search_content(result.sources, topic_mapping, state)

                ReporterStateManager.save_sources_with_topics(
                    state=state,
                    sources=enhanced_sources,
                    topic_source_mapping=enhanced_mapping
                )

                logger.info(
                    f"🔗 [REPORTER-{self.reporter_id}] Stored {len(enhanced_sources)} enhanced sources from {operation} tool in SharedMemoryStore",
                    topics_linked=len(enhanced_mapping) if enhanced_mapping else 0
                )
            else:
                ReporterStateManager.save_sources_with_topics(
                    state=state,
                    sources=result.sources,
                    topic_source_mapping=topic_mapping
                )

                logger.info(
                    f"🔗 [REPORTER-{self.reporter_id}] Stored {len(result.sources)} sources from {operation} tool in SharedMemoryStore",
                    topics_linked=len(topic_mapping) if topic_mapping else 0
                )
        else:
            logger.warning(
                f"⚠️ [REPORTER-{self.reporter_id}] Tool result has no sources or sources are empty",
                operation=operation
            )

        # Handle YouTube-specific metadata if present
        if operation == "youtube" and hasattr(result, "metadata") and result.metadata:
            video_metadata = result.metadata.get("video_metadata", [])
            if video_metadata:
                # Add video metadata to state for later use
                for metadata in video_metadata:
                    # Avoid duplicates - metadata is a dict, so check video_id
                    if not any(existing.video_id == metadata["video_id"] for existing in state.youtube_video_metadata):
                        # Convert dict metadata to YouTubeVideo object
                        from utils.youtube_models import YouTubeVideo
                        from datetime import datetime
                        video_obj = YouTubeVideo(
                            video_id=metadata["video_id"],
                            title=metadata["title"],
                            description=metadata.get("description", ""),
                            channel_id=metadata["channel_id"],
                            channel_title=metadata["channel_title"],
                            published_at=datetime.fromisoformat(metadata["published_at"]),
                            duration="",  # Not available in metadata
                            url=metadata["url"],
                            thumbnail_url=None,  # Not available in metadata
                            transcript=None  # Not needed for metadata storage
                        )
                        state.youtube_video_metadata.append(video_obj)

                logger.info(
                    f"📹 [REPORTER-{self.reporter_id}] Stored {len(video_metadata)} video metadata entries for transcript retrieval",
                    total_metadata_entries=len(state.youtube_video_metadata)
                )

    async def _enhance_search_content(self, sources: list[StorySource], topic_mapping: dict[str, Any], state: ReporterState) -> tuple[list[StorySource], dict[str, Any]]:
        """Enhance search sources with full content by scraping when needed.

        Args:
            sources: Original sources from search
            topic_mapping: Topic to source mapping
            state: Current reporter state

        Returns:
            Tuple of (enhanced_sources, enhanced_topic_mapping)
        """
        enhanced_sources: list[StorySource] = []
        enhanced_mapping: dict[str, Any] = {}
        scraper_tool = None

        # Get scraper tool if available
        scraper_tool = self.tool_registry.get_tool_by_name("scrape")

        for source in sources:
            enhanced_source = source

            # Find corresponding topic mapping
            topic_info = None
            for topic_name, info in topic_mapping.items():
                if info.get('source') and info['source'].url == source.url:
                    topic_info = info
                    break

            # Check if we need to enhance content
            if topic_info and topic_info.get('needs_content_fetch', False) and scraper_tool:
                logger.info(
                    f"🔍 [REPORTER-{self.reporter_id}] Enhancing content for topic with scraping",
                    url=source.url,
                    current_content_length=topic_info.get('content_length', 0)
                )

                try:
                    # Create scraper params
                    from agents.reporter_agent.reporter_tools import ScrapeParams
                    scrape_params = ScrapeParams(url=source.url)

                    # Execute scraping
                    scrape_result = await scraper_tool.execute(scrape_params)

                    if scrape_result.success and scrape_result.sources:
                        # Use the scraped content to enhance the source
                        scraped_source = scrape_result.sources[0]
                        if scraped_source.content and len(scraped_source.content) > len(source.content or ''):
                            enhanced_source = scraped_source
                            logger.info(
                                f"✅ [REPORTER-{self.reporter_id}] Enhanced content via scraping",
                                url=source.url,
                                original_length=len(source.content or ''),
                                enhanced_length=len(scraped_source.content)
                            )
                        else:
                            logger.warning(
                                f"⚠️ [REPORTER-{self.reporter_id}] Scraping didn't improve content quality",
                                url=source.url
                            )
                    else:
                        logger.warning(
                            f"⚠️ [REPORTER-{self.reporter_id}] Failed to scrape content",
                            url=source.url,
                            error=scrape_result.error
                        )
                except Exception as e:
                    logger.error(
                        f"❌ [REPORTER-{self.reporter_id}] Error during content enhancement",
                        url=source.url,
                        error=str(e)
                    )

            enhanced_sources.append(enhanced_source)

            # Update topic mapping with enhanced source
            if topic_info:
                topic_name = None
                for name, info in topic_mapping.items():
                    if info.get('source') and info['source'].url == source.url:
                        topic_name = name
                        break

                if topic_name:
                    enhanced_mapping[topic_name] = {
                        'source': enhanced_source,
                        'query': topic_info.get('query'),
                        'needs_content_fetch': False,  # Content has been enhanced
                        'content_length': len(enhanced_source.content or enhanced_source.summary or '')
                    }

        # Filter out topics without sufficient content
        final_sources: list[StorySource] = []
        final_mapping: dict[str, Any] = {}

        for source in enhanced_sources:
            content_length = len(source.content or source.summary or '')
            if content_length >= 100:  # Minimum content threshold
                final_sources.append(source)

                # Find and keep corresponding topic mapping
                for topic_name, info in enhanced_mapping.items():
                    if info.get('source') and info['source'].url == source.url:
                        final_mapping[topic_name] = info
                        break
            else:
                logger.warning(
                    f"⚠️ [REPORTER-{self.reporter_id}] Filtering out topic with insufficient content",
                    url=source.url,
                    content_length=content_length,
                    title=source.title
                )

        logger.info(
            f"📊 [REPORTER-{self.reporter_id}] Content validation complete",
            original_sources=len(enhanced_sources),
            valid_sources=len(final_sources),
            filtered_out=len(enhanced_sources) - len(final_sources)
        )

        return final_sources, final_mapping

    def _handle_search_result(self, result: Any, state: ReporterState) -> None:
        """Handle search tool results and update state.
        
        Args:
            result: Search tool result
            state: Current reporter state
        """
        logger.info(
            f"🔍 [REPORTER-{self.reporter_id}] Processing search result",
            result_type=type(result).__name__,
            has_cleaned_results=hasattr(result, "cleaned_results"),
            result_preview=str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        )

        if hasattr(result, "cleaned_results"):
            # Convert search results to sources
            from agents.models.search_models import SearchResultSummary

            search_summary = SearchResultSummary(
                query=result.query,
                search_type=result.search_type,  # Already a string, don't call .value
                result_count=len(result.cleaned_results),
                results=result.cleaned_results
            )
            ReporterStateManager.add_search_result(state, search_summary)

            # Convert to story sources and save using generic method
            story_sources = ReporterToolRegistry.convert_search_results_to_sources(result.cleaned_results)
            sources_before = len(state.sources)

            # Use topic mapping from search results if available
            topic_mapping = getattr(result, "topic_source_mapping", None)
            ReporterStateManager.save_sources_with_topics(
                state=state,
                sources=story_sources,
                topic_source_mapping=topic_mapping
            )

            logger.info(
                f"🔗 [REPORTER-{self.reporter_id}] Saved {len(story_sources)} sources from search results",
                sources_before=sources_before,
                sources_after=len(state.sources),
                cleaned_results_count=len(result.cleaned_results)
            )
        else:
            logger.warning(
                f"⚠️ [REPORTER-{self.reporter_id}] Search result has no cleaned_results attribute"
            )

    def _handle_youtube_result(self, result: Any, state: ReporterState) -> None:
        """Handle YouTube tool results and update state.

        Args:
            result: YouTube tool result
            state: Current reporter state
        """
        logger.info(
            f"🎥 [REPORTER-{self.reporter_id}] Processing YouTube result",
            result_type=type(result).__name__,
            operation=getattr(result, "operation", "unknown"),
            success=getattr(result, "success", False),
            channels_searched=getattr(result, "channels_searched", 0),
            videos_found=getattr(result, "videos_found", 0)
        )

        if hasattr(result, "sources") and result.sources:
            # Use ReporterStateManager to store sources in SharedMemoryStore
            topic_mapping = getattr(result, "topic_source_mapping", None)
            ReporterStateManager.save_sources_with_topics(
                state=state,
                sources=result.sources,
                topic_source_mapping=topic_mapping
            )

            logger.info(
                f"🔗 [REPORTER-{self.reporter_id}] Stored {len(result.sources)} YouTube sources in SharedMemoryStore",
                topics_linked=len(topic_mapping) if topic_mapping else 0,
                operation=result.operation
            )
        else:
            logger.warning(
                f"⚠️ [REPORTER-{self.reporter_id}] YouTube result has no sources attribute or sources are empty"
            )

        # Store video metadata for later transcript retrieval (no API cost)
        if hasattr(result, "video_metadata") and result.video_metadata:
            # Add video metadata to state for later use
            for metadata in result.video_metadata:
                # Avoid duplicates - metadata is a dict, so check video_id
                if not any(existing.video_id == metadata["video_id"] for existing in state.youtube_video_metadata):
                    # Convert dict metadata to YouTubeVideo object
                    from utils.youtube_models import YouTubeVideo
                    from datetime import datetime
                    video_obj = YouTubeVideo(
                        video_id=metadata["video_id"],
                        title=metadata["title"],
                        description=metadata.get("description", ""),
                        channel_id=metadata["channel_id"],
                        channel_title=metadata["channel_title"],
                        published_at=datetime.fromisoformat(metadata["published_at"]),
                        duration="",  # Not available in metadata
                        url=metadata["url"],
                        thumbnail_url=None,  # Not available in metadata
                        transcript=None  # Not needed for metadata storage
                    )
                    state.youtube_video_metadata.append(video_obj)

            logger.info(
                f"📹 [REPORTER-{self.reporter_id}] Stored {len(result.video_metadata)} video metadata entries for transcript retrieval",
                total_metadata_entries=len(state.youtube_video_metadata),
                operation=result.operation
            )

    def _handle_fetch_from_memory_result(self, result: Any, state: ReporterState) -> None:
        """Handle fetch_from_memory tool results and inject sources into state.

        Args:
            result: FetchFromMemoryResult from the tool
            state: Current reporter state
        """
        logger.info(
            f"🧠 [REPORTER-{self.reporter_id}] Processing fetch_from_memory result",
            success=getattr(result, "success", False),
            topic_key=getattr(result, "topic_key", "unknown"),
            sources_count=getattr(result, "sources_count", 0)
        )

        if hasattr(result, "success") and result.success:
            # Get the actual sources from memory and inject them into state
            from agents.shared_memory_store import get_shared_memory_store
            memory_store = get_shared_memory_store()

            topic_sources = memory_store.get_sources_for_topic(result.topic_key)
            if topic_sources:
                # Add sources to state for LLM context
                for source in topic_sources:
                    if source.url not in [s.url for s in state.sources]:
                        state.sources.append(source)

                logger.info(
                    f"📚 [REPORTER-{self.reporter_id}] Injected {len(topic_sources)} sources from memory into context",
                    topic_key=result.topic_key,
                    total_sources_now=len(state.sources)
                )
            else:
                logger.warning(
                    f"⚠️ [REPORTER-{self.reporter_id}] No sources found in memory for topic: {result.topic_key}"
                )
        else:
            error_msg = getattr(result, "error", "Unknown error")
            logger.warning(
                f"⚠️ [REPORTER-{self.reporter_id}] fetch_from_memory failed: {error_msg}"
            )

    def _handle_final_response(
        self,
        response: AgentResponse,
        state: ReporterState,
        task: ReporterTask
    ) -> StoryDraft | TopicList | ResearchResult | None:
        """Handle final responses based on task type.
        
        Args:
            response: Agent response
            state: Current reporter state
            task: Current task
            
        Returns:
            Final result if task is complete, None to continue
        """
        # Handle FIND_TOPICS task
        if task.name == TaskType.FIND_TOPICS and response.topic_list:
            response.topic_list.field = task.field
            response.topic_list.sub_section = task.sub_section
            logger.info(
                "✅ Topics found successfully",
                task_type=task.name.value,
                field=task.field.value,
                topics_count=len(response.topic_list.topics),
                iteration=state.iteration
            )
            return response.topic_list

        # Handle RESEARCH_TOPIC task
        if task.name == TaskType.RESEARCH_TOPIC and response.research_result:
            response.research_result.field = task.field
            response.research_result.sub_section = task.sub_section

            # Add collected sources to research result
            self._enhance_research_result(response.research_result, state)

            logger.info(
                "✅ Research completed successfully",
                task_type=task.name.value,
                field=task.field.value,
                sources_count=len(response.research_result.sources),
                iteration=state.iteration
            )
            return response.research_result

        # Handle WRITE_STORY task responses
        if task.name == TaskType.WRITE_STORY:
            # Research result during writing task
            if response.research_result:
                response.research_result.field = task.field
                response.research_result.sub_section = task.sub_section

                self._enhance_research_result(response.research_result, state)
                ReporterStateManager.set_current_research(state, response.research_result)
                ReporterStateManager.advance_research_iteration(state)

                # Check if we have sufficient research to proceed to writing
                if ReporterStateManager.has_sufficient_research(state, task.min_sources):
                    ReporterStateManager.transition_to_writing_phase(state)

                logger.info(
                    f"✅ [REPORTER-{self.reporter_id}] Research phase completed",
                    sources_count=len(response.research_result.sources),
                    facts_count=len(response.research_result.facts),
                    research_iteration=state.research_iteration_count,
                    sufficient_sources=len(response.research_result.sources) >= task.min_sources
                )

                return None  # Continue to next iteration

            # Story draft (final result)
            if response.story_draft:
                response.story_draft.field = task.field
                response.story_draft.sub_section = task.sub_section
                response.story_draft.calculate_word_count()

                # Ensure all sources are included
                self._ensure_all_sources_in_story(response.story_draft, state)

                ReporterStateManager.mark_completed(state)

                logger.info(
                    f"✅ [REPORTER-{self.reporter_id}] Story draft completed successfully",
                    title=response.story_draft.title,
                    word_count=response.story_draft.word_count,
                    sources_count=len(response.story_draft.sources),
                    keywords_count=len(response.story_draft.keywords),
                    iteration=state.iteration
                )

                return response.story_draft

        return None

    def _enhance_research_result(self, research_result: ResearchResult, state: ReporterState) -> None:
        """Enhance research result with sources from search results.
        
        Args:
            research_result: Research result to enhance
            state: Current reporter state
        """
        # Add all collected sources, avoiding duplicates
        existing_urls = {source.url for source in research_result.sources}
        for source in state.sources:
            if source.url not in existing_urls:
                research_result.sources.append(source)

    def _ensure_all_sources_in_story(self, story_draft: StoryDraft, state: ReporterState) -> None:
        """Ensure all research sources are included in the story draft.

        Args:
            story_draft: Story draft to update
            state: Current reporter state
        """
        existing_urls = {source.url for source in story_draft.sources}
        sources_added = 0

        for source in state.sources:
            if source.url not in existing_urls:
                story_draft.sources.append(source)
                sources_added += 1

        if sources_added > 0:
            logger.info(
                f"📚 [REPORTER-{self.reporter_id}] Added {sources_added} research sources to story draft"
            )

    def _determine_model_speed(
        self,
        state: ReporterState,
        override: ModelSpeed | None
    ) -> ModelSpeed:
        """Determine appropriate model speed for current state.
        
        Args:
            state: Current reporter state
            override: Optional model speed override
            
        Returns:
            Model speed to use
        """
        if override:
            return override

        task = state.current_task

        # For WRITE_STORY tasks, use different speeds for different phases
        if task.name == TaskType.WRITE_STORY:
            if state.task_phase == TaskPhase.WRITING:
                # Use slow model for writing phase (better quality)
                logger.info(f"🐌 [REPORTER-{self.reporter_id}] Using SLOW model for story writing phase")
                return ModelSpeed.SLOW
            else:
                # Use fast model for research phase (efficiency)
                logger.info(f"⚡ [REPORTER-{self.reporter_id}] Using FAST model for research phase")
                return ModelSpeed.FAST

        # Default to agent's configured model speed
        return self.agent.config.default_model_speed

    # Removed _load_shared_memories method - no automatic memory injection
    # Agents must use fetch_from_memory tool to retrieve content when needed
