"""Editor orchestration logic for managing news cycle workflow."""

from datetime import datetime
from typing import Any

from agents.editor_agent.editor_models import EditorAction
from agents.editor_agent.editor_state import EditorState, EditorStateManager
from agents.editor_agent.editor_tools import (
    AssignTopicsParams,
    AssignTopicsTool,
    CollectStoryParams,
    CollectStoryTool,
    CollectTopicsParams,
    CollectTopicsTool,
    PublishStoryParams,
    PublishStoryTool,
    ReviewStoryParams,
    ReviewStoryTool,
)
from agents.types import EditorActionType, StoryReviewStatus
from core.logging_service import get_logger

logger = get_logger(__name__)


class EditorExecutor:
    """Handles the orchestration logic for editor workflow."""

    def __init__(self, task_service: Any, editor_id: str):
        """Initialize orchestrator.
        
        Args:
            task_service: Task execution service for spawning reporters
            editor_id: Unique identifier for this editor
        """
        self.task_service = task_service
        self.editor_id = editor_id

    def build_orchestration_prompt(self, state: EditorState) -> str:
        """Build orchestration prompt for LLM decision making with complete state information.

        Args:
            state: Current editor state

        Returns:
            Formatted prompt for the LLM with full state details
        """
        progress = EditorStateManager.get_progress_summary(state)

        prompt_parts = [
            f"CURRENT GOAL: {state.current_goal.value}",
            f"ITERATION: {state.current_iteration + 1}/{state.max_iterations}",
            "",
            "REQUESTED COVERAGE:",
            f"- Field Requests: {[f'{req.field.value}' + (f'/{req.sub_section.value}' if req.sub_section else '') for req in state.requested_fields]}",
            f"- Stories per field: {state.stories_per_field}",
            "",
            "PROGRESS SUMMARY:",
            f"- Topics collected: {progress['topics_collected']}",
            f"- Topics assigned: {progress['topics_assigned']}",
            f"- Stories collected: {progress['stories_collected']}",
            f"- Stories reviewed: {progress['stories_reviewed']} (approved: {progress['stories_approved']})",
            f"- Stories published: {progress['stories_published']}",
            f"- Stories in revision: {progress['stories_in_revision']}",
        ]

        # Add detailed topics collected information
        if state.topics_collected:
            prompt_parts.extend(["", "TOPICS COLLECTED BY FIELD:"])
            for field, topics in state.topics_collected.items():
                prompt_parts.append(f"  {field.value}:")
                for i, topic in enumerate(topics, 1):
                    prompt_parts.append(f"    {i}. {topic}")

        # Add detailed topic assignments
        if state.topics_assigned:
            prompt_parts.extend(["", "TOPIC ASSIGNMENTS:"])
            for reporter_id, topic in state.topics_assigned.items():
                field = state.active_reporters.get(reporter_id)
                if field:
                    field_str = field.value
                    prompt_parts.append(f"  Reporter {reporter_id} ({field_str}): {topic}")
                else:
                    prompt_parts.append(f"  Reporter {reporter_id} (unknown field): {topic}")

        # Add active reporters information
        if state.active_reporters:
            prompt_parts.extend(["", "ACTIVE REPORTERS:"])
            for reporter_id, field in state.active_reporters.items():
                prompt_parts.append(f"  {reporter_id}: {field.value}")

        # Add detailed story information
        if state.stories_collected:
            prompt_parts.extend(["", "DETAILED STORY STATUS:"])
            for story_id, story_draft in state.stories_collected.items():
                if story_id in state.stories_review_status:
                    status = state.stories_review_status[story_id]
                    if status.approved:
                        prompt_parts.append(f"  - {story_id}: ✅ APPROVED - Ready to publish")
                        prompt_parts.append(f"    Title: {story_draft.title}")
                        prompt_parts.append(f"    Field: {story_draft.field.value}")
                        prompt_parts.append(f"    Word count: {story_draft.word_count}")
                        if status.feedback:
                            prompt_parts.append(f"    Review feedback: {status.feedback}")
                    else:
                        prompt_parts.append(f"  - {story_id}: ❌ REJECTED - Needs revision")
                        prompt_parts.append(f"    Title: {story_draft.title}")
                        prompt_parts.append(f"    Field: {story_draft.field.value}")
                        if status.feedback:
                            prompt_parts.append(f"    Rejection feedback: {status.feedback}")
                        if status.required_changes:
                            prompt_parts.append(f"    Required changes: {', '.join(status.required_changes)}")
                else:
                    prompt_parts.extend([
                        f"  - {story_id}: ⏳ NEEDS REVIEW",
                        f"    Title: {story_draft.title}",
                        f"    Field: {story_draft.field.value}",
                        f"    Summary: {story_draft.summary[:150]}{'...' if len(story_draft.summary) > 150 else ''}",
                        f"    Word count: {story_draft.word_count}",
                        f"    Sources: {len(story_draft.sources)} sources",
                        f"    Keywords: {', '.join(story_draft.keywords[:5])}{'...' if len(story_draft.keywords) > 5 else ''}",
                    ])
                    # Include full story content for review
                    prompt_parts.append(f"    FULL CONTENT: {story_draft.content[:500]}{'...' if len(story_draft.content) > 500 else ''}")

        # Add stories in revision details
        if state.stories_in_revision:
            prompt_parts.extend(["", "STORIES IN REVISION:"])
            for story_id, reporter_id in state.stories_in_revision.items():
                prompt_parts.append(f"  {story_id}: Assigned to {reporter_id} for revision")

        # Add published stories
        if state.stories_published:
            prompt_parts.extend(["", "PUBLISHED STORIES:"])
            for story_id in state.stories_published:
                if story_id in state.stories_collected:
                    story = state.stories_collected[story_id]
                    prompt_parts.append(f"  {story_id}: {story.title} ({story.field.value})")
                else:
                    prompt_parts.append(f"  {story_id}: Published")

        # Add retry tracking information
        if state.publish_retry_count:
            prompt_parts.extend(["", "PUBLISH RETRY TRACKING:"])
            for story_id, retry_count in state.publish_retry_count.items():
                prompt_parts.append(f"  {story_id}: {retry_count} retries")

        # Add action history for context
        if state.action_history:
            prompt_parts.extend(["", "RECENT ACTION HISTORY:"])
            # Show last 5 actions
            recent_actions = state.action_history[-5:] if len(state.action_history) > 5 else state.action_history
            for action in recent_actions:
                prompt_parts.append(f"  - {action}")

        # Add recent tool errors for feedback
        if state.tool_errors:
            prompt_parts.extend(["", "⚠️  RECENT TOOL ERRORS - PLEASE LEARN FROM THESE:"])
            for error in state.tool_errors:
                prompt_parts.append(f"  - {error}")
            if state.last_failed_action:
                prompt_parts.append(f"  → LAST FAILED ACTION: {state.last_failed_action}")
                prompt_parts.append("  → Please correct the parameters or choose a different action")

        # Analyze current state and suggest next action
        next_action_analysis = self._analyze_next_action(state, progress)
        prompt_parts.extend([
            "",
            "NEXT ACTION ANALYSIS:",
            f"Based on current progress: {next_action_analysis}",
            ""
        ])

        # Decision instruction is now part of the system prompt, not needed here

        return "\n".join(prompt_parts)

    def _analyze_next_action(self, state: EditorState, progress: dict[str, int]) -> str:
        """Analyze the current state and suggest the logical next action.

        Args:
            state: Current editor state
            progress: Progress summary from EditorStateManager

        Returns:
            Analysis string explaining what the next logical action should be
        """
        target_stories = len(state.requested_fields) * state.stories_per_field

        # Check if cycle is complete
        if progress["stories_published"] >= target_stories:
            return f"CYCLE COMPLETE: Published {progress['stories_published']}/{target_stories} target stories. Consider ending the cycle."

        # Check for approved stories ready to publish
        approved_unpublished = progress["stories_approved"] - progress["stories_published"]
        if approved_unpublished > 0:
            return f"READY TO PUBLISH: {approved_unpublished} approved stories are waiting to be published. Next action should be 'publish_story'."

        # Check for stories needing review
        unreviewed_stories = progress["stories_collected"] - progress["stories_reviewed"]
        if unreviewed_stories > 0:
            return f"REVIEW NEEDED: {unreviewed_stories} stories have been collected but not yet reviewed. Next action should be 'review_story'."

        # Check for stories in revision
        if progress["stories_in_revision"] > 0:
            return f"REVISIONS PENDING: {progress['stories_in_revision']} stories are being revised by reporters. Next action should be 'collect_story' to get revised versions."

        # Check if we have assigned topics but no stories
        if progress["topics_assigned"] > 0 and progress["stories_collected"] == 0:
            return f"STORIES NEEDED: {progress['topics_assigned']} topics have been assigned to reporters but no stories collected yet. Next action should be 'collect_story'."

        # Check if we have topics but none assigned
        if progress["topics_collected"] > 0 and progress["topics_assigned"] == 0:
            return f"ASSIGNMENT NEEDED: {progress['topics_collected']} topics have been collected but none assigned to reporters. Next action should be 'assign_topics'."

        # Check if we need more topics
        if progress["topics_collected"] == 0:
            return "TOPICS NEEDED: No topics have been collected yet. Next action should be 'collect_topics' to gather story ideas for the requested fields."

        # Fallback analysis
        return f"STATE ANALYSIS: Topics: {progress['topics_collected']}, Assigned: {progress['topics_assigned']}, Stories: {progress['stories_collected']}, Reviewed: {progress['stories_reviewed']}, Published: {progress['stories_published']}/{target_stories}. Determine next logical step."

    def _update_current_goal(self, state: EditorState) -> None:
        """Update the current goal based on the actual progress state.

        Args:
            state: Current editor state (modified in place)
        """
        from agents.types import EditorGoal

        # Determine the appropriate goal based on current progress
        progress = EditorStateManager.get_progress_summary(state)

        # If we have published enough stories, we're done
        target_stories = len(state.requested_fields) * state.stories_per_field
        if progress["stories_published"] >= target_stories:
            state.current_goal = EditorGoal.COMPLETE_CYCLE

        # If we have approved stories ready to publish
        elif progress["stories_approved"] > 0 and progress["stories_approved"] > progress["stories_published"]:
            state.current_goal = EditorGoal.PUBLISH_STORIES

        # If we have stories that need review
        elif progress["stories_collected"] > progress["stories_reviewed"]:
            state.current_goal = EditorGoal.REVIEW_STORIES

        # If we have assigned topics but no stories collected yet
        elif progress["topics_assigned"] > 0 and progress["stories_collected"] == 0:
            state.current_goal = EditorGoal.COLLECT_STORIES

        # If we have topics collected but none assigned
        elif progress["topics_collected"] > 0 and progress["topics_assigned"] == 0:
            state.current_goal = EditorGoal.ASSIGN_TOPICS

        # If we have no topics collected yet
        elif progress["topics_collected"] == 0:
            state.current_goal = EditorGoal.COLLECT_TOPICS

        # Default fallback - evaluate what we have
        else:
            state.current_goal = EditorGoal.EVALUATE_TOPICS

    async def execute_editor_action(self, action: EditorAction, state: EditorState) -> None:
        """Execute an editor action using the appropriate tool.

        Args:
            action: The action to execute
            state: Current state (modified in place)
        """
        param_keys_str = ", ".join(action.parameters.keys()) if action.parameters else "none"

        logger.info(
            f"⚡ ACTION EXECUTE | {action.action_type.value.upper()}",
            action_type=action.action_type.value,
            reasoning_preview=action.reasoning[:80] + "..." if len(action.reasoning) > 80 else action.reasoning,
            param_keys=param_keys_str
        )

        try:
            if action.action_type == EditorActionType.COLLECT_TOPICS:
                await self._execute_collect_topics(action, state)
            elif action.action_type == EditorActionType.ASSIGN_TOPICS:
                await self._execute_assign_topics(action, state)
            elif action.action_type == EditorActionType.COLLECT_STORY:
                 await self._execute_collect_story(action, state)
            elif action.action_type == EditorActionType.REVIEW_STORY:
                await self._execute_review_story(action, state)
            elif action.action_type == EditorActionType.PUBLISH_STORY:
                await self._execute_publish_story(action, state)
            else:
                logger.error(f"Unknown action type: {action.action_type}")

        except Exception as e:
            error_message = str(e)

            # Enhance error message for common validation errors
            if "validation error" in error_message.lower():
                if action.action_type.value == "publish_story" and "story" in error_message and "Field required" in error_message:
                    error_message = 'publish_story requires only story_id parameter, not the full story object. Use: {"story_id": "story-123", "section": "technology", "priority": "medium"}'
                elif action.action_type.value == "review_story" and "story" in error_message and "Field required" in error_message:
                    error_message = 'review_story requires only story_id parameter, not the full story object. Use: {"story_id": "story-123"}'
                elif action.action_type.value == "assign_topics" and "available_topics" in error_message:
                    error_message = 'assign_topics requires field parameter only. Available topics are provided automatically. Use: {"field": "technology", "topics_to_assign": 2}'

            logger.error(
                f"❌ Failed to execute action: {action.action_type.value}",
                action_type=action.action_type.value,
                error=error_message
            )
            # Add error to state for LLM feedback
            EditorStateManager.add_tool_error(state, action.action_type.value, error_message)

    async def _execute_collect_topics(self, action: EditorAction, state: EditorState) -> None:
        """Execute collect topics action."""
        tool = CollectTopicsTool(task_executor=self.task_service)
        params = CollectTopicsParams(
            field_requests=state.requested_fields,
            topics_per_field=action.parameters.get("topics_per_field", 5)
        )

        result = await tool.execute(params)
        if result.success:
            state.topics_collected.update(result.topics_by_field)
            total_topics = sum(len(topics) for topics in result.topics_by_field.values())
            logger.info("✅ Successfully collected topics", total_topics=total_topics)
        else:
            error_msg = result.error or "Unknown error"
            logger.error("❌ Failed to collect topics", error=error_msg)
            EditorStateManager.add_tool_error(state, "collect_topics", error_msg)

    async def _execute_assign_topics(self, action: EditorAction, state: EditorState) -> None:
        """Execute assign topics action."""
        tool = AssignTopicsTool(task_executor=self.task_service)

        # Get the field from action parameters
        field = action.parameters.get("field")
        if not field:
            logger.error("❌ assign_topics action missing required 'field' parameter")
            return

        # Get available topics for this field from state
        available_topics = state.topics_collected.get(field, [])

        # Create params with available topics from state
        params_dict = action.parameters.copy()
        params_dict["available_topics"] = available_topics

        params = AssignTopicsParams(**params_dict)
        result = await tool.execute(params)
        if result.success and result.assignments:
            for assignment in result.assignments:
                state.topics_assigned[assignment.reporter_id] = assignment.topic
                state.active_reporters[assignment.reporter_id] = assignment.field
            logger.info("✅ Successfully assigned topics", assignments=len(result.assignments))
        else:
            error_msg = getattr(result, "error", "Unknown error") or "Unknown error"
            logger.error("❌ Failed to assign topics", error=str(error_msg))
            EditorStateManager.add_tool_error(state, "assign_topics", str(error_msg))

    async def _execute_collect_story(self, action: EditorAction, state: EditorState) -> None:
        """Execute collect story action."""
        tool = CollectStoryTool(task_executor=self.task_service)
        params = CollectStoryParams(**action.parameters)

        result = await tool.execute(params)
        if result.success and result.story:
            story_id = f"story-{datetime.now().timestamp()}"
            state.stories_collected[story_id] = result.story

            # If this was a revision, remove from revision tracking
            if story_id in state.stories_in_revision:
                del state.stories_in_revision[story_id]
                # Reset review status for revised story
                if story_id in state.stories_review_status:
                    del state.stories_review_status[story_id]

            logger.info("✅ Successfully collected story", story_id=story_id, title=result.story.title[:50])
        else:
            logger.error("❌ Failed to collect story", error=result.error or "Unknown error")

    async def _execute_review_story(self, action: EditorAction, state: EditorState) -> None:
        """Execute review story action."""
        tool = ReviewStoryTool(task_executor=self.task_service)

        # Get story_id from action parameters
        story_id = action.parameters.get("story_id")
        if not story_id:
            error_msg = "review_story action missing required 'story_id' parameter"
            logger.error(f"❌ {error_msg}")
            EditorStateManager.add_tool_error(state, "review_story", error_msg)
            return

        # Get the actual story from state
        if story_id not in state.stories_collected:
            error_msg = f"Story {story_id} not found in collected stories"
            logger.error(f"❌ {error_msg}")
            EditorStateManager.add_tool_error(state, "review_story", error_msg)
            return

        story_draft = state.stories_collected[story_id]

        # Create params with actual story from state
        params = ReviewStoryParams(
            story_id=story_id,
            story=story_draft
        )

        result = await tool.execute(params)
        if result.success:
            story_id = action.parameters.get("story_id", "unknown")
            state.stories_review_status[story_id] = StoryReviewStatus(
                approved=result.is_approved,
                feedback=result.feedback or "",
                required_changes=result.required_changes or [],
                timestamp=datetime.now(),
                reasoning=result.reasoning
            )

            if not result.is_approved:
                # Mark story for revision
                reporter_id = action.parameters.get("reporter_id", "unknown")
                state.stories_in_revision[story_id] = reporter_id

            logger.info("✅ Story review completed", story_id=story_id, approved=result.is_approved)
        else:
            error_msg = getattr(result, "error", "Unknown error") or "Unknown error"
            logger.error("❌ Failed to review story", error=str(error_msg))
            EditorStateManager.add_tool_error(state, "review_story", str(error_msg))

    async def _execute_publish_story(self, action: EditorAction, state: EditorState) -> None:
        """Execute publish story action with retry logic."""
        story_id = action.parameters.get("story_id")
        if not story_id:
            error_msg = "publish_story action missing required 'story_id' parameter"
            logger.error(f"❌ {error_msg}")
            EditorStateManager.add_tool_error(state, "publish_story", error_msg)
            return

        # Get the actual story from state
        if story_id not in state.stories_collected:
            error_msg = f"Story {story_id} not found in collected stories"
            logger.error(f"❌ {error_msg}")
            EditorStateManager.add_tool_error(state, "publish_story", error_msg)
            return

        story_draft = state.stories_collected[story_id]

        # Check if story is approved
        if story_id in state.stories_review_status:
            review_status = state.stories_review_status[story_id]
            if not review_status.approved:
                error_msg = f"Cannot publish rejected story {story_id}"
                logger.error(f"❌ {error_msg}")
                EditorStateManager.add_tool_error(state, "publish_story", error_msg)
                return

        tool = PublishStoryTool(task_executor=self.task_service)

        # Create params with actual story from state
        params = PublishStoryParams(
            story=story_draft,
            story_id=story_id,
            section=action.parameters.get("section", "technology"),
            priority=action.parameters.get("priority", "medium"),
            front_page=action.parameters.get("front_page", False)
        )

        result = await tool.execute(params)
        if result.published:
            state.stories_published.append(result.story_id)

            # Clean up tracking
            if story_id in state.stories_in_revision:
                del state.stories_in_revision[story_id]

            logger.info("✅ Story published successfully", story_id=result.story_id)
        else:
            # Handle publish failure with retry logic
            error_msg = getattr(result, "error", "Unknown error") or "Unknown error"
            is_retriable = "validation error" in error_msg.lower()
            retry_count = state.publish_retry_count.get(story_id, 0)

            if is_retriable and retry_count < 2:
                # Mark for retry
                state.publish_retry_count[story_id] = retry_count + 1

                # Reset review status to approved for retry
                if story_id in state.stories_review_status:
                    state.stories_review_status[story_id] = StoryReviewStatus(
                        approved=True,
                        feedback="Retry after validation error",
                        timestamp=datetime.now(),
                        reasoning="Retrying publication after fixing validation issues"
                    )

                logger.warning("🔄 Publish failed, will retry", story_id=str(story_id), retry_attempt=retry_count + 1)
            else:
                logger.error("❌ Publish failed permanently", story_id=str(story_id), error=str(error_msg))
