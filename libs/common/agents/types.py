"""Shared types and enums for the agent framework."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ReporterField(StrEnum):
    """Fields of expertise for reporter agents."""
    ECONOMICS = "economics"
    TECHNOLOGY = "technology"
    SCIENCE = "science"


class StoryStatus(StrEnum):
    """Status of a story in the editorial workflow."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class EditorialDecisionType(StrEnum):
    """Types of editorial decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    HOLD = "hold"


class StoryPriority(StrEnum):
    """Priority levels for stories."""
    BREAKING = "breaking"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TechnologySubSection(StrEnum):
    """Sub-sections within Technology field."""
    AI_TOOLS = "ai_tools"
    TECH_TRENDS = "tech_trends"
    QUANTUM_COMPUTING = "quantum_computing"
    GENERAL_TECH = "general_tech"
    MAJOR_DEALS = "major_deals"
    GEN_AI_NEWS = "gen_ai_news"
    GEN_AI_IMAGE_EDITING = "gen_ai_image_editing"


class EconomicsSubSection(StrEnum):
    """Sub-sections within Economics field."""
    CRYPTO = "crypto"
    US_STOCK_MARKET = "us_stock_market"
    GENERAL_NEWS = "general_news"
    ISRAEL_ECONOMICS = "israel_economics"
    EXITS = "exits"
    UPCOMING_IPOS = "upcoming_ipos"
    MAJOR_TRANSACTIONS = "major_transactions"


class ScienceSubSection(StrEnum):
    """Sub-sections within Science field."""
    NEW_RESEARCH = "new_research"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    SPACE = "space"
    PHYSICS = "physics"




class NewspaperSection(StrEnum):
    """Sections of the newspaper."""
    FRONT_PAGE = "front_page"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    ECONOMICS = "economics"


class OverallQuality(StrEnum):
    """Overall quality ratings for editorial feedback."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class QualityThreshold(StrEnum):
    """Quality threshold levels for editor tasks."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentType(StrEnum):
    """Types of agents in the system."""
    REPORTER = "reporter"
    EDITOR = "editor"


class StoryReviewStatus(BaseModel):
    """Status of a story review."""
    approved: bool
    feedback: str = ""
    required_changes: list[str] = Field(default_factory=list)
    timestamp: datetime
    reasoning: str


class CycleStatus(StrEnum):
    """Status of a news cycle."""
    PLANNING = "planning"
    REPORTING = "reporting"
    EDITING = "editing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(StrEnum):
    """Types of tasks that can be assigned to reporter agents."""
    FIND_TOPICS = "find_topics"
    RESEARCH_TOPIC = "research_topic"
    WRITE_STORY = "write_story"


class EditorPhase(StrEnum):
    """Phases of editor orchestration workflow."""
    COLLECT_TOPICS = "collect_topics"      # Gather topic suggestions from reporters
    EVALUATE_TOPICS = "evaluate_topics"    # Evaluate and filter topics
    ASSIGN_TOPICS = "assign_topics"        # Assign topics to reporters
    COLLECT_STORIES = "collect_stories"    # Gather written stories
    REVIEW_STORIES = "review_stories"      # Review and request revisions
    PUBLISH_STORIES = "publish_stories"    # Final publishing


class EditorGoal(StrEnum):
    """Current goal of the editor in the news cycle."""
    COLLECT_TOPICS = "collect_topics"
    EVALUATE_TOPICS = "evaluate_topics"
    ASSIGN_TOPICS = "assign_topics"
    COLLECT_STORIES = "collect_stories"
    REVIEW_STORIES = "review_stories"
    PUBLISH_STORIES = "publish_stories"
    COMPLETE_CYCLE = "complete_cycle"


class EditorActionType(StrEnum):
    """Types of actions editor can take."""
    COLLECT_TOPICS = "collect_topics"
    ASSIGN_TOPICS = "assign_topics"
    COLLECT_STORY = "collect_story"
    REVIEW_STORY = "review_story"
    REQUEST_REVISION = "request_revision"
    PUBLISH_STORY = "publish_story"
    CREATE_REPORTER = "create_reporter"
    COMPLETE = "complete"


class RequiredAction(StrEnum):
    """Required actions when stories are rejected or blocked."""
    STORY_MUST_BE_REVISED_AND_RE_APPROVED = "story_must_be_revised_and_re_approved"
    INVESTIGATE_AND_RETRY = "investigate_and_retry"
    RETRY_OR_REASSIGN = "retry_or_reassign"
    CYCLE_COMPLETION_CHECK = "cycle_completion_check"


class FieldTopicRequest(BaseModel):
    """Request for news generation with field and optional sub-section."""
    field: ReporterField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None


class AgentMetricKey(StrEnum):
    """Keys for agent performance metrics in a cycle."""
    REPORTER_ECONOMICS = "reporter_economics"
    REPORTER_TECHNOLOGY = "reporter_technology"
    REPORTER_SCIENCE = "reporter_science"
    EDITOR_MAIN = "editor_main"

    @classmethod
    def from_reporter_field(cls, field: ReporterField) -> "AgentMetricKey":
        """Get the metric key for a reporter field."""
        field_to_key = {
            ReporterField.ECONOMICS: cls.REPORTER_ECONOMICS,
            ReporterField.TECHNOLOGY: cls.REPORTER_TECHNOLOGY,
            ReporterField.SCIENCE: cls.REPORTER_SCIENCE,
        }
        return field_to_key[field]
