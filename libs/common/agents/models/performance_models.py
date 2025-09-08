"""Performance-related models for agent workflows."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Import types from types module to avoid circular imports
from ..types import AgentMetricKey, AgentType


class AgentPerformance(BaseModel):
    """Performance metrics for an agent."""
    agent_id: str
    agent_type: AgentType
    tasks_completed: int = 0
    success_rate: float = 0.0
    average_task_time: float = 0.0  # in seconds
    quality_score: float = 0.0  # 0-100
    last_active: datetime = Field(default_factory=datetime.now)

    # Reporter-specific metrics
    stories_written: int = 0
    stories_published: int = 0
    average_word_count: float = 0.0

    # Editor-specific metrics
    stories_reviewed: int = 0
    stories_approved: int = 0
    stories_rejected: int = 0


# Typed default factory to avoid partially unknown dict types in Pylance

def _empty_agent_metrics() -> dict[AgentMetricKey, AgentPerformance]:
    return {}


class CyclePerformanceMetrics(BaseModel):
    """Performance metrics for a complete news cycle."""
    agent_metrics: dict[AgentMetricKey, AgentPerformance] = Field(default_factory=_empty_agent_metrics)
    cycle_duration_seconds: float = 0.0
    total_stories_written: int = 0
    total_stories_published: int = 0
    total_stories_rejected: int = 0
    average_story_quality: float = 0.0
    success_rate: float = 0.0  # published / written

    def add_agent_metrics(self, metric_key: AgentMetricKey, metrics: AgentPerformance) -> None:
        """Add metrics for a specific agent."""
        self.agent_metrics[metric_key] = metrics

    def calculate_cycle_metrics(self) -> None:
        """Calculate aggregate cycle metrics from agent metrics."""
        if not self.agent_metrics:
            return

        total_written = sum(m.stories_written for m in self.agent_metrics.values())
        total_published = sum(m.stories_published for m in self.agent_metrics.values())
        total_rejected = sum(m.stories_rejected for m in self.agent_metrics.values() if hasattr(m, "stories_rejected"))

        self.total_stories_written = total_written
        self.total_stories_published = total_published
        self.total_stories_rejected = total_rejected

        if total_written > 0:
            self.success_rate = total_published / total_written

        # Calculate average quality score across all agents
        quality_scores = [m.quality_score for m in self.agent_metrics.values() if m.quality_score > 0]
        if quality_scores:
            self.average_story_quality = sum(quality_scores) / len(quality_scores)
