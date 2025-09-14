"""Topic memory models for editorial content management."""

from datetime import datetime
from pydantic import BaseModel, Field


class CoveredTopic(BaseModel):
    """Represents a topic that has been covered."""
    title: str = Field(description="Title of the covered topic")
    description: str = Field(description="Brief description of the topic coverage")
    date_added: str = Field(description="Date when topic was added to memory (YYYY-MM-DD)")


class TopicMemory(BaseModel):
    """Container for all covered topics and memory metadata."""
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    topics: list[CoveredTopic] = Field(default_factory=list, description="List of covered topics")
    total_topics: int = Field(default=0, description="Total number of topics tracked")

    def model_post_init(self, __context) -> None:
        """Update total count after initialization."""
        self.total_topics = len(self.topics)

    def add_topic(self, topic: CoveredTopic) -> None:
        """Add a new topic to memory."""
        self.topics.append(topic)
        self.total_topics = len(self.topics)
        self.last_updated = datetime.now()

    def get_recent_topics(self, days: int = 30) -> list[CoveredTopic]:
        """Get topics added within the last N days."""
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        return [topic for topic in self.topics if topic.date_added >= cutoff_str]
