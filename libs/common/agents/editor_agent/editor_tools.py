"""Editor-specific tools for orchestrating news cycle."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.models.story_models import ImageSourceType, StoryDraft, StoryImage, StorySource, TopicList
from agents.models.submission_models import PublishedStory
from agents.models.task_models import JournalistTask
from agents.tools.base_tool import BaseTool
from agents.types import (
    AgentType,
    EconomicsSubSection,
    FieldTopicRequest,
    NewspaperSection,
    JournalistField,
    ScienceSubSection,
    StoryPriority,
    StoryStatus,
    TaskType,
    TechnologySubSection,
)
from core.config_service import ConfigService
from utils.topic_memory_service import TopicMemoryService
from utils.topic_memory_models import CoveredTopic
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..task_executor_protocol import TaskExecutor

logger = get_logger(__name__)


def _normalize_topic_name(topic: str) -> str:
    """Normalize topic name for consistent storage and retrieval."""
    return topic.strip().title()


# ============================================================================
# PUBLISHING INFRASTRUCTURE CLASSES (moved from newspaper.py)
# ============================================================================

class Story(BaseModel):
    """Represents a published story with full metadata."""
    id: str
    title: str
    field: JournalistField
    status: StoryStatus
    content: str
    published_date: str  # ISO format
    summary: str
    keywords: list[str] = Field(default_factory=list)
    author: str
    section: NewspaperSection
    priority: StoryPriority
    front_page: bool = False
    word_count: int = 0
    images: list[StoryImage] = Field(default_factory=lambda: [])
    sources: list[StorySource] = Field(default_factory=lambda: [])

    def model_post_init(self, __context: Any) -> None:
        """Calculate word count after initialization."""
        if self.content and self.word_count == 0:
            self.word_count = len(self.content.split())

    def to_published_story(self) -> PublishedStory:
        """Convert Story to PublishedStory for API responses."""
        return PublishedStory(
            story_id=self.id,
            title=self.title,
            content=self.content,
            summary=self.summary,
            author=self.author,
            reporter_id=self.author,  # Use author as reporter_id for compatibility
            field=self.field,
            section=NewspaperSection(self.section),
            priority=self.priority,
            sources=self.sources,
            keywords=self.keywords,
            images=self.images,
            published_at=datetime.fromisoformat(self.published_date),
            word_count=self.word_count,
            views=0,  # Default value for views
            editorial_decision=None  # No editorial decision for file-based stories
        )


class StoreLimits(BaseModel):
    front_page: int
    default_section: int
    per_section: dict[str, int] = Field(default_factory=dict)


class StoreMetadata(BaseModel):
    last_updated: str
    limits: StoreLimits


class StoreData(BaseModel):
    stories: dict[str, Story] = Field(default_factory=dict)
    front_page: list[str] = Field(default_factory=list)
    sections: dict[str, list[str]] = Field(default_factory=dict)
    metadata: StoreMetadata


class NewspaperFileStore:
    """JSON file-backed storage for newspaper stories with limits."""

    def __init__(
        self,
        path: str | Path | None = None,
        front_page_limit: int = 10,
        default_section_limit: int = 20,
        per_section_limits: dict[NewspaperSection, int] | None = None,
    ) -> None:
        if path:
            self.path = Path(path)
        else:
            # Resolve path relative to this file to avoid CWD issues
            self.path = (Path(__file__).resolve(
            ).parents[4] / "libs/common/data/newspaper.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.front_page_limit = int(front_page_limit)
        self.default_section_limit = int(default_section_limit)
        self.per_section_limits = per_section_limits or {}
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            limits = StoreLimits(
                front_page=self.front_page_limit,
                default_section=self.default_section_limit,
                per_section={k.value: int(v)
                             for k, v in self.per_section_limits.items()},
            )
            data = StoreData(
                stories={},
                front_page=[],
                sections={field.value: [] for field in JournalistField},
                metadata=StoreMetadata(
                    last_updated=datetime.now().isoformat(),
                    limits=limits,
                ),
            )
            self._write(data)

    def _read(self) -> StoreData:
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Handle backward compatibility: add empty images and sources fields to stories that don't have them
        if "stories" in raw:
            for _, story_data in raw["stories"].items():
                if "images" not in story_data:
                    story_data["images"] = []
                if "sources" not in story_data:
                    story_data["sources"] = []

        return StoreData.model_validate(raw)

    def _write(self, data: StoreData) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data.model_dump(mode="json"),
                      f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def _priority_rank(self, priority: StoryPriority) -> int:
        order = {
            StoryPriority.BREAKING: 1,
            StoryPriority.HIGH: 2,
            StoryPriority.MEDIUM: 3,
            StoryPriority.LOW: 4,
        }
        return order.get(priority, 5)

    def publish(self, story: Story) -> None:
        data = self._read()
        stories = data.stories
        sections = data.sections
        front_page = data.front_page

        stories[story.id] = story

        section_key = story.section.value
        ids = sections.get(section_key, [])
        if story.id not in ids:
            ids.append(story.id)

        def key_func(story_id: str) -> tuple[int, str]:
            s = stories[story_id]
            return (self._priority_rank(s.priority), s.published_date)

        # Sort and limit section stories
        ids = sorted(ids, key=key_func, reverse=True)
        section_capacity = self._section_capacity(data, story.section)
        sections[section_key] = ids[:section_capacity]

        # Handle front page stories
        if story.front_page:
            if story.id not in front_page:
                front_page.append(story.id)
            front_page = sorted(front_page, key=key_func, reverse=True)
            front_page_limit = data.metadata.limits.front_page
            data.front_page = front_page[:front_page_limit]

        # Clean up orphaned stories that are no longer in any section or front page
        all_active_ids = set(data.front_page)
        for section_ids in data.sections.values():
            all_active_ids.update(section_ids)

        # Remove stories that are no longer referenced anywhere
        orphaned_stories = [
            sid for sid in stories.keys() if sid not in all_active_ids]
        for orphaned_id in orphaned_stories:
            del stories[orphaned_id]

        data.metadata.last_updated = datetime.now().isoformat()
        self._write(data)

    def _section_capacity(self, data: StoreData, section: NewspaperSection) -> int:
        per_section = data.metadata.limits.per_section
        return int(per_section.get(section.value, data.metadata.limits.default_section))

    def list_current(self) -> dict[str, Any]:
        """List current stories on front page and by sections."""
        data = self._read()

        def to_story(sid: str) -> Story:
            if sid not in data.stories:
                raise KeyError(f"Story {sid} not found")
            return data.stories[sid]

        # Get front page stories
        front_page = []
        for sid in data.front_page:
            try:
                story = to_story(sid)
                front_page.append(story)
            except Exception as e:
                logger.error(f"Error processing front page story {sid}: {e}")

        # If no cover stories, fill front page with latest stories up to limit
        if not front_page:
            all_story_ids = list(data.stories.keys())
            if all_story_ids:
                def key_func(story_id: str) -> tuple[int, str]:
                    s = data.stories[story_id]
                    return (self._priority_rank(s.priority), s.published_date)

                # Sort by priority and date (newest first)
                sorted_ids = sorted(all_story_ids, key=key_func, reverse=True)
                # Take up to front page limit
                front_page_ids = sorted_ids[:data.metadata.limits.front_page]
                for sid in front_page_ids:
                    try:
                        story = to_story(sid)
                        front_page.append(story)
                    except Exception as e:
                        logger.error(
                            f"Error processing fallback story {sid}: {e}")

        sections_map: dict[NewspaperSection, list[Story]] = {}
        for section_name, ids in data.sections.items():
            try:
                section_enum = NewspaperSection(section_name)
                section_stories = []
                for sid in ids:
                    try:
                        story = to_story(sid)
                        section_stories.append(story)
                    except Exception as e:
                        logger.error(
                            f"Error processing section story {sid}: {e}")
                sections_map[section_enum] = section_stories
            except Exception as e:
                logger.error(f"Error processing section {section_name}: {e}")

        return {"front_page": front_page, "sections": sections_map}

    def get_history(self, limit: int) -> list[Story]:
        """Get historical stories limited by count."""
        data = self._read()
        all_ids = list(data.stories.keys())
        all_ids.sort(
            key=lambda sid: data.stories[sid].published_date, reverse=True)
        return [data.stories[sid] for sid in all_ids[:limit]]

    def get_section(self, section: NewspaperSection) -> list[Story]:
        """Get stories from a specific section."""
        data = self._read()
        ids = data.sections.get(section.value, [])
        return [data.stories[sid] for sid in ids]

    def remove(self, story_id: str) -> bool:
        """Remove a story from the newspaper."""
        data = self._read()
        if story_id not in data.stories:
            return False

        del data.stories[story_id]

        for section_name, ids in list(data.sections.items()):
            data.sections[section_name] = [
                sid for sid in ids if sid != story_id]

        data.front_page = [sid for sid in data.front_page if sid != story_id]

        data.metadata.last_updated = datetime.now().isoformat()
        self._write(data)
        return True


# ============================================================================
# COLLECT TOPICS TOOL
# ============================================================================

class CollectTopicsParams(BaseModel):
    """Parameters for collecting topics."""
    field_requests: list[FieldTopicRequest]
    topics_per_field: int = 5


class CollectTopicsResult(BaseModel):
    """Result of topic collection."""
    reasoning: str  # Explanation of topic collection (FIRST)
    topics_by_field: dict[JournalistField, list[str]]
    success: bool
    error: str | None = None


class CollectTopicsTool(BaseTool):
    """Collect topic suggestions from reporters for specified fields."""

    name: str = "collect_topics"
    description: str = """
Collect trending topic suggestions from reporters for specified field requests.
Each field request may include a sub-section and will have a temporary reporter created to gather topics.

Parameters:
- field_requests: List of FieldTopicRequest objects with field and optional sub_section
- topics_per_field: Number of topics to collect per field request (default: 5)

Usage: <tool>collect_topics</tool><args>{"field_requests": [{"field": "technology", "sub_section": "ai_tools"}], "topics_per_field": 5}</args>

Returns: CollectTopicsResult with topics organized by field
"""
    params_model: type[BaseModel] | None = CollectTopicsParams
    task_executor: TaskExecutor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> CollectTopicsResult:
        """Execute topic collection from reporters."""
        if not isinstance(params, CollectTopicsParams):
            return CollectTopicsResult(
                reasoning="Invalid parameters provided",
                topics_by_field={},
                success=False,
                error="Invalid parameters type"
            )

        logger.info(
            "🔍 Starting topic collection from reporters",
            requests_count=len(params.field_requests),
            topics_per_field=params.topics_per_field,
            requests=", ".join([f"{req.field.value}" + (
                f"/{req.sub_section.value}" if req.sub_section else "") for req in params.field_requests])
        )

        topics_by_field = {}

        for request in params.field_requests:
            sub_section_desc = f"/{request.sub_section.value}" if request.sub_section else ""
            try:
                logger.info(
                    f"📰 Requesting topics from {request.field.value}{sub_section_desc} reporter...")

                # Create task for finding topics
                base_guidelines = "Focus on current, newsworthy topics with broad appeal"
                try:
                    config_service = ConfigService()
                    forbidden_text = TopicMemoryService(config_service).get_forbidden_topics_as_text(days_back=30)
                except Exception as e:
                    logger.exception("Failed to retrieve forbidden topics for guidelines")
                    forbidden_text = "No recent topics to avoid."

                combined_guidelines = f"{base_guidelines}\n\n{forbidden_text}\n\nIMPORTANT: Avoid duplicates."

                task = JournalistTask(
                    agent_type=AgentType.RESEARCHER,
                    name=TaskType.FIND_TOPICS,
                    field=request.field,
                    sub_section=request.sub_section,
                    description=f"Find {params.topics_per_field} trending topics in {request.field.value}{sub_section_desc}",
                    guidelines=combined_guidelines
                )

                # Execute task via task service (which spawns a researcher)
                # Topic collection uses FAST model for efficiency
                result = await self.task_executor.execute_researcher_task(task, model_speed=model_speed)

                if isinstance(result, TopicList):
                    topics_by_field[request.field] = result.topics
                    logger.info(
                        f"✅ Received topics from {request.field.value}{sub_section_desc} reporter",
                        field=request.field.value,
                        sub_section=request.sub_section.value if request.sub_section else "none",
                        topics_count=len(result.topics),
                        topics=", ".join(
                            result.topics[:3]) + ("..." if len(result.topics) > 3 else "")
                    )
                else:
                    logger.warning(
                        f"⚠️  Unexpected result type from {request.field.value}{sub_section_desc} reporter",
                        field=request.field.value,
                        sub_section=request.sub_section.value if request.sub_section else "none",
                        result_type=type(result).__name__
                    )
                    topics_by_field[request.field] = []

            except Exception as e:
                logger.error(
                    f"❌ Failed to collect topics from {request.field.value}{sub_section_desc} reporter",
                    field=request.field.value,
                    sub_section=request.sub_section.value if request.sub_section else "none",
                    error=str(e)
                )
                topics_by_field[request.field] = []

        return CollectTopicsResult(
            reasoning=f"Collected topics from {len(topics_by_field)} fields with {params.topics_per_field} topics each",
            topics_by_field=topics_by_field,
            success=True
        )


# ============================================================================
# SELECT TOPICS TOOL (from reporter-provided topics)
# ============================================================================

class SelectTopicsParams(BaseModel):
    """Parameters for selecting topics from reporter-provided topics."""
    field: JournalistField = Field(description="Field to select topics from")
    available_topics: list[str] = Field(
        description="Topics provided by reporters for this field")
    max_topics: int = Field(
        default=3, description="Maximum number of topics to select")
    stories_per_field: int = Field(
        default=1, description="Number of stories needed for this field")


class SelectTopicsResult(BaseModel):
    """Result of topic selection from reporter-provided topics."""
    reasoning: str = Field(
        description="Editor's reasoning for topic selection")
    selected_topics: list[str] = Field(
        description="Topics selected for story assignment")
    rejected_topics: list[str] = Field(description="Topics not selected")
    success: bool = Field(description="Whether the selection was successful")


class SelectTopicsTool(BaseTool):
    """Select topics from reporter-provided topics for story assignment.

    This tool allows the editor to intelligently select which topics from the
    reporter's TopicList should be assigned for story writing, based on editorial
    criteria like newsworthiness, relevance, and resource constraints.
    """

    name: str = "select_topics"
    description: str = """
    Select topics from reporter-provided topics for story assignment.
    The editor reviews topics submitted by reporters and selects the most newsworthy ones.

    Parameters:
    - field: ReporterField (technology/economics/science) to select topics from
    - available_topics: List of topics provided by reporters for this field
    - max_topics: Maximum number of topics to select (default: 3)
    - stories_per_field: Number of stories needed for this field (default: 1)

    The tool will intelligently select topics based on:
    - Newsworthiness and current relevance
    - Audience interest and engagement potential
    - Editorial balance across different sub-topics
    - Resource constraints (stories_per_field limit)

    Usage: <tool>select_topics</tool><args>{"field": "technology", "available_topics": ["AI Breakthrough", "Quantum Computing", "5G Networks"], "max_topics": 2}</args>

    Returns: SelectTopicsResult with selected topics and reasoning
    """
    params_model: type[BaseModel] | None = SelectTopicsParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.SLOW) -> SelectTopicsResult:
        """Select topics from reporter-provided topics for assignment."""
        if not isinstance(params, SelectTopicsParams):
            return SelectTopicsResult(
                reasoning="Invalid parameters provided",
                selected_topics=[],
                rejected_topics=[],
                success=False
            )

        if not params.available_topics:
            return SelectTopicsResult(
                reasoning=f"No topics available from reporters for field {params.field.value}",
                selected_topics=[],
                rejected_topics=[],
                success=False
            )

        # Determine how many topics to select (limited by stories_per_field and max_topics)
        topics_to_select = min(len(params.available_topics),
                               params.stories_per_field, params.max_topics)

        if topics_to_select == 0:
            return SelectTopicsResult(
                reasoning="No topics needed - stories_per_field is 0",
                selected_topics=[],
                rejected_topics=params.available_topics,
                success=True
            )

        # Deduplicate against recently covered topics using TopicMemoryService
        try:
            config_service = ConfigService()
            memory = TopicMemoryService(config_service)
            # Exact-title forbidden set (recent topics)
            forbidden_titles_lower = {t.title.lower().strip() for t in memory.get_forbidden_topics(days_back=30)}
            # Similarity check against memory for semantic overlaps
            similarity_map = memory.check_topic_similarity(params.available_topics)
        except Exception:
            logger.exception("Failed to initialize topic memory for selection; proceeding without memory filter")
            forbidden_titles_lower: set[str] = set()
            similarity_map: dict[str, list[str]] = {t: [] for t in params.available_topics}

        # Filter available topics: exclude exact matches or high-overlap similarities
        allowed: list[str] = []
        forbidden_exact_count = 0
        forbidden_similar_count = 0
        for topic in params.available_topics:
            low = topic.lower().strip()
            if low in forbidden_titles_lower:
                forbidden_exact_count += 1
                continue
            similar = similarity_map.get(topic, [])
            if len(similar) > 0:
                forbidden_similar_count += 1
                continue
            allowed.append(topic)

        selected_topics = allowed[:topics_to_select]
        # Everything not selected is rejected (includes filtered and leftovers)
        selected_set = set(selected_topics)
        rejected_topics = [t for t in params.available_topics if t not in selected_set]

        if len(selected_topics) == 0:
            reasoning = (
                f"All {len(params.available_topics)} proposed topics for {params.field.value} were duplicates or overlapped with recently covered topics. "
                f"Rejected {forbidden_exact_count} exact matches and {forbidden_similar_count} similar topics."
            )
            logger.info(
                "📝 Topic selection yielded no unique topics",
                field=params.field.value,
                available_count=len(params.available_topics),
                rejected_count=len(rejected_topics),
                forbidden_exact=forbidden_exact_count,
                forbidden_similar=forbidden_similar_count,
            )
            return SelectTopicsResult(
                reasoning=reasoning,
                selected_topics=[],
                rejected_topics=rejected_topics,
                success=True
            )

        reasoning = (
            f"Selected {len(selected_topics)} topics from {len(params.available_topics)} available topics for {params.field.value}. "
            f"Deduplicated against recently covered topics (rejected {forbidden_exact_count} exact and {forbidden_similar_count} similar). "
            f"Resource constraint: max {params.stories_per_field} stories per field."
        )

        logger.info(
            "📝 Topic selection completed",
            field=params.field.value,
            available_count=len(params.available_topics),
            selected_count=len(selected_topics),
            rejected_count=len(rejected_topics),
            forbidden_exact=forbidden_exact_count,
            forbidden_similar=forbidden_similar_count,
            selected_topics=", ".join(selected_topics)
        )

        return SelectTopicsResult(
            reasoning=reasoning,
            selected_topics=selected_topics,
            rejected_topics=rejected_topics,
            success=True
        )

# ============================================================================
# ASSIGN TOPICS TOOL
# ============================================================================

class AssignTopicsParams(BaseModel):
    """Parameters for creating topic assignments from memory topics."""
    field: JournalistField  # Field to assign topics from
    selected_topics: list[str]  # Topics selected from memory to assign
    priority: StoryPriority = StoryPriority.MEDIUM  # Priority level for assignments


class TopicAssignment(BaseModel):
    """Single topic assignment that will spawn a reporter."""
    reporter_id: str  # Generated ID for the reporter that will be spawned
    field: JournalistField
    topic: str
    priority: int  # Priority for this story
    guidelines: str | None = None


def _empty_rejected_topics() -> dict[JournalistField, list[str]]:
    """Typed default factory for rejected topics."""
    return {}


def empty_topics_collected() -> dict[JournalistField, list[str]]:
    """Typed default factory for topics collected."""
    return {}


def empty_topics_assigned() -> dict[str, str]:
    """Typed default factory for topics assigned."""
    return {}


def empty_stories_collected() -> dict[str, StoryDraft]:
    """Typed default factory for stories collected."""
    return {}


def empty_reporters() -> dict[str, JournalistField]:
    """Typed default factory for active reporters."""
    return {}


def empty_action_history() -> list[str]:
    """Typed default factory for action history."""
    return []


class AssignTopicsResult(BaseModel):
    """Result of topic selection and assignment."""
    reasoning: str  # Overall selection strategy (FIRST)
    assignments: list[TopicAssignment]  # Topics that will spawn reporters
    rejected_topics: dict[JournalistField, list[str]] = Field(
        default_factory=_empty_rejected_topics)
    success: bool


class AssignTopicsTool(BaseTool):
    """Create topic assignments for reporters to be spawned.

    This tool takes already-selected topics and creates assignments that will
    be used to spawn reporter agents. It does NOT select topics - that's
    done by the editor using select_topics_for_coverage.
    """

    name: str = "assign_topics"
    description: str = """
Create topic assignments from topics Collected by the reporters.
Each assignment will spawn a reporter agent to write the story.

Parameters:
- field: ReporterField (technology/economics/science) to assign topics from
- selected_topics: List of topics selected from memory (these have validated content)
- priority: StoryPriority level (low/medium/high/breaking, defaults to medium)

Usage: <tool>assign_topics</tool><args>{"field": "technology", "selected_topics": ["AI Breakthrough In Healthcare", "New Quantum Processor"], "priority": "high"}</args>

Returns: AssignTopicsResult with reporter assignments ready for spawning
"""
    params_model: type[BaseModel] | None = AssignTopicsParams
    task_executor: TaskExecutor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> AssignTopicsResult:
        """Create assignments from already-selected topics for spawning reporters."""
        # model_speed parameter available for future use if needed
        if not isinstance(params, AssignTopicsParams):
            return AssignTopicsResult(
                reasoning="Invalid parameters provided",
                assignments=[],
                success=False
            )

        try:
            # Use the selected topics from memory
            if not params.selected_topics:
                return AssignTopicsResult(
                    reasoning=f"No topics selected for field {params.field.value}",
                    assignments=[],
                    success=False
                )

            topics_to_assign = params.selected_topics

            # Convert topics to assignments that will spawn reporters
            assignments: list[TopicAssignment] = []
            for i, topic in enumerate(topics_to_assign):
                # Generate unique reporter ID based on field and index
                reporter_id = f"reporter-{params.field.value}-{i+1:03d}"

                # Convert StoryPriority enum to int for priority field
                priority_value = {"low": 1, "medium": 5, "high": 8, "breaking": 10}.get(
                    params.priority.value, 5
                )

                assignments.append(
                    TopicAssignment(
                        reporter_id=reporter_id,
                        field=params.field,
                        topic=topic,
                        priority=priority_value,
                        guidelines=f"Write a comprehensive news story about: {topic}"
                    )
                )

            logger.info(
                "Created topic assignments",
                total_assignments=len(assignments),
                field=params.field.value,
                topics_assigned=len(topics_to_assign),
                selected_topics=len(params.selected_topics)
            )

            return AssignTopicsResult(
                reasoning=f"Created {len(assignments)} assignments from {params.field.value} topics for reporter spawning",
                assignments=assignments,
                rejected_topics={},  # No rejections in this simplified approach
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to create assignments: {e}")
            return AssignTopicsResult(
                reasoning=f"Failed to create assignments: {e!s}",
                assignments=[],
                success=False
            )


# ============================================================================
# COLLECT STORY TOOL
# ============================================================================

class CollectStoryParams(BaseModel):
    """Parameters for collecting a story from a reporter."""
    reporter_id: str
    field: JournalistField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None
    topic: str
    guidelines: str | None = None
    editor_remarks: str | None = None  # For revisions


class CollectStoryResult(BaseModel):
    """Result of story collection."""
    reasoning: str  # Explanation of story collection (FIRST)
    story: StoryDraft | None = None
    reporter_id: str
    success: bool
    error: str | None = None


class CollectStoryTool(BaseTool):
    """Collect a story from a specific reporter."""

    name: str = "collect_story"
    description: str = """
Request a story from a reporter on their assigned topic.
Can include editor remarks for revisions.

Parameters:
- reporter_id: ID of the reporter
- field: ReporterField of the reporter
- sub_section: Optional sub-section within the field
- topic: Topic to write about
- guidelines: Optional writing guidelines
- editor_remarks: Optional feedback for revisions

Usage: <tool>collect_story</tool><args>{"reporter_id": "reporter-tech-001", "field": "technology", "sub_section": "ai_tools", "topic": "AI breakthrough"}</args>

Returns: CollectStoryResult with the story draft
"""
    params_model: type[BaseModel] | None = CollectStoryParams
    task_executor: TaskExecutor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> CollectStoryResult:
        """Execute story collection from reporter."""
        if not isinstance(params, CollectStoryParams):
            return CollectStoryResult(
                reasoning="Invalid parameters provided",
                reporter_id="unknown",
                success=False,
                error="Invalid parameters type"
            )

        try:
            # Normalize topic name for consistent SharedMemoryStore lookup
            normalized_topic = _normalize_topic_name(params.topic)

            # Create story task
            task = JournalistTask(
                agent_type=AgentType.REPORTER,
                name=TaskType.WRITE_STORY,
                field=params.field,
                sub_section=params.sub_section,
                description=f"Write story about: {normalized_topic}",
                topic=normalized_topic,
                guidelines=params.guidelines,
                editor_remarks=params.editor_remarks,
                min_sources=1,
                target_word_count=500,
                require_images=True
            )

            # Execute task via task service (which spawns reporter)
            # Story writing uses FAST model for content generation
            result = await self.task_executor.execute_reporter_task(task, model_speed=model_speed)
            if isinstance(result, StoryDraft):
                return CollectStoryResult(
                    reasoning=f"Successfully collected story from {params.reporter_id} about {params.topic}",
                    story=result,
                    reporter_id=params.reporter_id,
                    success=True
                )
            else:
                return CollectStoryResult(
                    reasoning=f"Reporter returned unexpected result type: {type(result).__name__}",
                    reporter_id=params.reporter_id,
                    success=False,
                    error="Unexpected result type"
                )

        except Exception as e:
            logger.error(
                f"Failed to collect story from {params.reporter_id}: {e}")
            return CollectStoryResult(
                reasoning="Failed to collect story due to error",
                reporter_id=params.reporter_id,
                success=False,
                error=str(e)
            )


# ============================================================================
# REVIEW STORY TOOL
# ============================================================================

class ReviewStoryParams(BaseModel):
    """Parameters for reviewing a story."""
    story_id: str
    story: StoryDraft


class ReviewStoryResult(BaseModel):
    """Result of story review."""
    reasoning: str  # Explanation of review decision (FIRST)
    is_approved: bool
    feedback: str | None = None
    required_changes: list[str] = Field(default_factory=list)
    success: bool


class ReviewStoryTool(BaseTool):
    """Review a story draft for quality and publication readiness."""

    name: str = "review_story"
    description: str = """
Review a story draft for quality, accuracy, and publication readiness.
Provides feedback if revision is needed.

Parameters:
- story_id: Unique identifier for the story to review (story will be retrieved from state automatically)

Usage: <tool>review_story</tool><args>{"story_id": "story-123"}</args>

Returns: ReviewStoryResult with approval status and feedback
"""
    params_model: type[BaseModel] | None = ReviewStoryParams
    task_executor: TaskExecutor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.SLOW) -> ReviewStoryResult:
        """Execute story review."""
        # model_speed parameter available for future use if needed
        if not isinstance(params, ReviewStoryParams):
            return ReviewStoryResult(
                reasoning="Invalid parameters provided",
                is_approved=False,
                success=False
            )

        story = params.story

        # Basic quality checks
        issues = []

        if story.word_count < 100:
            issues.append("Story is too short (minimum 100 words)")

        if len(story.sources) < 1:
            issues.append("Insufficient sources (minimum 1 required)")

        if not story.summary:
            issues.append("Missing story summary")

        if issues:
            return ReviewStoryResult(
                reasoning=f"Story needs revision due to {len(issues)} issues",
                is_approved=False,
                feedback="Please address the following issues:",
                required_changes=issues,
                success=True
            )

        return ReviewStoryResult(
            reasoning="Story meets all quality standards",
            is_approved=True,
            feedback="Story approved - meets all quality standards",
            success=True
        )


# ============================================================================
# PUBLISH STORY TOOL
# ============================================================================

class PublishStoryParams(BaseModel):
    """Parameters for publishing a story."""
    story: StoryDraft
    story_id: str | None = None  # Optional explicit story ID to use
    section: str = "national"
    priority: str = "medium"
    front_page: bool = False  # Whether this should be a cover story


class PublishStoryResult(BaseModel):
    """Result of story publication."""
    reasoning: str  # Explanation of publication (FIRST)
    story_id: str
    published: bool
    error: str | None = None


class PublishStoryTool(BaseTool):
    """Publish an approved story to the newspaper."""

    name: str = "publish_story"
    description: str = """
Publish an approved story to the newspaper.

Parameters:
- story_id: Unique identifier for the story to publish (story will be retrieved from state automatically)
- section: Newspaper section (optional, default: "technology")
- priority: Story priority (optional, default: "medium")
- front_page: Whether this should be a cover story (optional, default: false)

Usage: <tool>publish_story</tool><args>{"story_id": "story-123", "section": "technology", "priority": "high"}</args>

Returns: PublishStoryResult with publication status
"""
    params_model: type[BaseModel] | None = PublishStoryParams
    task_executor: TaskExecutor

    # Private newspaper store instance
    _store: NewspaperFileStore = PrivateAttr(
        default_factory=NewspaperFileStore)

    # Private topic memory service instance
    _topic_memory: TopicMemoryService | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        """Initialize with topic memory service."""
        super().__init__(**data)
        try:
            config_service = ConfigService()
            self._topic_memory = TopicMemoryService(config_service)
        except Exception as e:
            logger.error(f"Failed to initialize topic memory service: {e}")
            self._topic_memory = None

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> PublishStoryResult:
        """Execute story publication."""
        if not isinstance(params, PublishStoryParams):
            return PublishStoryResult(
                reasoning="Invalid parameters provided",
                story_id="unknown",
                published=False,
                error="Invalid parameters type"
            )

        # Use provided story_id or generate one
        story_id = params.story_id or f"story-{params.story.created_at.timestamp()}"

        # Determine if this should be a cover story based on priority and breaking news
        is_cover_story = params.front_page or params.priority.lower() == "breaking"

        logger.info(
            "Publishing story to newspaper",
            title=params.story.title[:50],
            section=params.section,
            priority=params.priority,
            front_page=is_cover_story
        )

        try:
            # Map string values to enums
            try:
                section_enum = NewspaperSection(params.section.lower())
            except ValueError:
                section_enum = NewspaperSection.TECHNOLOGY

            try:
                priority_enum = StoryPriority(params.priority.lower())
            except ValueError:
                priority_enum = StoryPriority.MEDIUM

            # Always generate images using the image generation tool
            images_to_publish = await self._generate_and_save_story_images(params.story.title, params.story.field, story_id)

            # Create story object and publish directly using local store
            story = Story(
                id=story_id,
                title=params.story.title,
                field=params.story.field,
                status=StoryStatus.PUBLISHED,
                content=params.story.content,
                published_date=datetime.now().isoformat(),
                summary=params.story.summary,
                keywords=params.story.keywords,
                author="system",  # Would be actual reporter ID in full implementation
                section=section_enum,
                priority=priority_enum,
                front_page=is_cover_story,
                images=images_to_publish,
                sources=params.story.sources,
            )

            # Publish using local store
            self._store.publish(story)

            # Add topic to memory for future deduplication
            if self._topic_memory:
                topic = CoveredTopic(
                    title=params.story.title,
                    description=params.story.summary,
                    date_added=datetime.now().strftime("%Y-%m-%d")
                )
                success = self._topic_memory.add_topic(topic)
                if success:
                    logger.info(f"Added published story topic to memory: {params.story.title}")
                else:
                    logger.warning(f"Failed to add topic to memory: {params.story.title}")

            return PublishStoryResult(
                reasoning=f"Successfully published story to {params.section} section" +
                (" as cover story" if is_cover_story else "") +
                (" and added to topic memory" if self._topic_memory else ""),
                story_id=story_id,
                published=True
            )

        except Exception as e:
            logger.exception("Failed to publish story")
            return PublishStoryResult(
                reasoning="Error occurred while publishing story",
                story_id=story_id,
                published=False,
                error=str(e)
            )

    async def _generate_and_save_story_images(self, title: str, field: JournalistField, story_id: str) -> list[StoryImage]:
        """Generate images using the image generation tool and save to local storage.

        Args:
            title: The story title
            field: The reporter field
            story_id: The story ID for filename

        Returns:
            List of StoryImage objects with local paths
        """
        try:
            # Import image generation tool
            from agents.editor_agent.image_tool import ImageGenerationTool, ImageQuality, ImageSize, ImageStyle, ImageToolParams

            # Get LLM service from the task executor if available
            llm_service = getattr(
                self.task_executor, "llm_service", None) if self.task_executor else None

            if not llm_service:
                logger.warning(
                    "No LLM service available for image generation, returning empty list")
                return []

            # Create image generation tool
            image_tool = ImageGenerationTool(llm_service=llm_service)

            # Generate prompt based on story title and field
            prompt = self._create_image_prompt(title, field)

            # Generate image
            image_params = ImageToolParams(
                prompt=prompt,
                size=ImageSize.SQUARE_1024,
                quality=ImageQuality.STANDARD,
                style=ImageStyle.VIVID,
                n=1,
                story_id=story_id
            )

            result = await image_tool.execute(image_params)

            if result.success and result.images:
                generated_images: list[StoryImage] = []
                for i, img in enumerate(result.images):
                    # Check if this is a base64 data URL or regular URL
                    if img.url.startswith("data:image"):
                        # Base64 data URL - extract the base64 part and store it directly
                        base64_data = img.url.split(
                            ",", 1)[1] if "," in img.url else img.url
                        story_image = StoryImage(
                            url=img.url,  # Keep the full data URL for direct display
                            local_path=None,  # No local file needed
                            base64_data=base64_data,  # Store base64 for JSON
                            mime_type="image/png",
                            caption=f"AI-generated illustration for: {title}",
                            alt_text=f"Generated image depicting {title}",
                            is_generated=True,
                            source_type=ImageSourceType.GENERATED,
                            file_size_kb=None
                        )
                    else:
                        # Regular URL - try to download and save locally
                        local_path = self._download_and_save_image(
                            img.url, story_id, i)
                        story_image = StoryImage(
                            url=img.url,
                            local_path=local_path,
                            base64_data=None,
                            mime_type="image/png",
                            caption=f"AI-generated illustration for: {title}",
                            alt_text=f"Generated image depicting {title}",
                            is_generated=True,
                            source_type=ImageSourceType.GENERATED,
                            file_size_kb=None
                        )

                    generated_images.append(story_image)

                logger.info(
                    f"🖼️ Generated and saved {len(generated_images)} images for story {story_id}")
                return generated_images
            else:
                error_msg = result.error or "Unknown error - no error message provided"
                logger.error(
                    "Image generation failed",
                    success=result.success,
                    error=error_msg,
                    images_count=len(result.images) if result.images else 0,
                    model=getattr(result, "model", "unknown"),
                    prompt=getattr(result, "prompt", "unknown")[:100]
                )
                return []

        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return []

    def _create_image_prompt(self, title: str, field: JournalistField) -> str:
        """Create an image generation prompt based on story title and field.

        Args:
            title: Story title
            field: Reporter field

        Returns:
            Image generation prompt
        """
        field_styles = {
            JournalistField.TECHNOLOGY: "modern technology, digital innovation, sleek design",
            JournalistField.SCIENCE: "scientific research, laboratory setting, discovery theme",
            JournalistField.ECONOMICS: "business analytics, financial data, corporate environment"
        }

        style = field_styles.get(field, "professional news illustration")
        return f"Professional news illustration representing: {title}. Style: {style}, high quality, editorial style, clean composition"

    def _download_and_save_image(self, image_url: str, story_id: str, index: int = 0) -> str | None:
        """Download image from URL and save to local storage.

        Args:
            image_url: URL of the image to download
            story_id: Story identifier for filename
            index: Image index for multiple images

        Returns:
            Local file path where image was saved
        """
        try:
            # For now, just use a simple urllib approach to avoid adding dependencies
            import urllib.request
            import uuid
            from pathlib import Path

            # Create images directory if it doesn't exist
            images_dir = Path("data/images/generated")
            images_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            filename = f"{story_id}_{index}_{uuid.uuid4().hex[:8]}.png"
            local_path = images_dir / filename

            # Download and save image using urllib
            urllib.request.urlretrieve(image_url, local_path)

            logger.info(f"📁 Saved image to {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None
