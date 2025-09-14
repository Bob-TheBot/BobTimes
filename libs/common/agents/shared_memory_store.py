"""Shared memory store for sharing agent memories during task execution."""

from datetime import datetime
from typing import Any

from agents.models.story_models import StorySource
from core.logging_service import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class SharedMemoryEntry(BaseModel):
    """A memory entry that can be shared across agents."""

    topic_name: str = Field(description="The topic this memory relates to")
    field: str = Field(description="The field/domain this memory belongs to")
    sources: list[StorySource] = Field(default_factory=lambda: [], description="All sources for this topic")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SharedMemoryStore:
    """In-memory store for sharing memories between agents during task execution."""

    def __init__(self, store_path: Any = None):
        """Initialize the memory store.

        Args:
            store_path: Ignored - kept for compatibility
        """
        # In-memory storage - no file persistence needed
        self._memories: dict[str, SharedMemoryEntry] = {}
        logger.info("Initialized in-memory SharedMemoryStore")
    
    def store_memory(self, topic_name: str, field: str, sources: list[StorySource]) -> None:
        """Store a memory entry with sources.

        Args:
            topic_name: The topic this memory relates to
            field: The field/domain this memory belongs to
            sources: List of StorySource objects for this topic
        """
        # Create or update memory entry in memory
        if topic_name in self._memories:
            # Update existing entry - merge sources
            memory = self._memories[topic_name]

            # Merge sources (avoid duplicates by URL)
            existing_urls = {source.url for source in memory.sources}
            new_sources = [source for source in sources if source.url not in existing_urls]
            memory.sources.extend(new_sources)
            memory.updated_at = datetime.now().isoformat()

            logger.info(f"Updated memory for topic: {topic_name} - added {len(new_sources)} new sources")
        else:
            # Create new entry
            memory = SharedMemoryEntry(
                topic_name=topic_name,
                field=field,
                sources=sources
            )
            self._memories[topic_name] = memory
            logger.info(f"Created new memory for topic: {topic_name} with {len(sources)} sources")

        logger.info(f"Stored memory for topic: {topic_name} in field: {field} - total sources: {len(memory.sources)}")
    
    def get_memory(self, topic_name: str) -> SharedMemoryEntry | None:
        """Retrieve a memory entry by topic name.

        Args:
            topic_name: The topic to retrieve memory for

        Returns:
            SharedMemoryEntry if found, None otherwise
        """
        return self._memories.get(topic_name)

    def get_sources_for_topic(self, topic_name: str) -> list[StorySource]:
        """Retrieve all sources for a specific topic.

        Args:
            topic_name: The topic to retrieve sources for

        Returns:
            List of StorySource objects for the topic
        """
        memory = self.get_memory(topic_name)
        return memory.sources if memory else []
    
    def get_memories_by_field(self, field: str) -> list[SharedMemoryEntry]:
        """Retrieve all memories for a specific field.

        Args:
            field: The field to retrieve memories for

        Returns:
            List of SharedMemoryEntry objects
        """
        memories: list[SharedMemoryEntry] = []

        for memory in self._memories.values():
            if memory.field == field:
                memories.append(memory)

        return memories
    
    def list_topics(self, field: str | None = None) -> list[str]:
        """List all topics in the memory store.

        Args:
            field: Optional field filter

        Returns:
            List of topic names
        """
        if field:
            return [
                topic_name for topic_name, memory in self._memories.items()
                if memory.field == field
            ]
        else:
            return list(self._memories.keys())
    
    def delete_memory(self, topic_name: str) -> bool:
        """Delete a memory entry.

        Args:
            topic_name: The topic to delete

        Returns:
            True if deleted, False if not found
        """
        if topic_name in self._memories:
            del self._memories[topic_name]
            logger.info(f"Deleted memory for topic: {topic_name}")
            return True

        return False
    
    def clear_field_memories(self, field: str) -> int:
        """Clear all memories for a specific field.

        Args:
            field: The field to clear memories for

        Returns:
            Number of memories deleted
        """
        topics_to_delete = [
            topic_name for topic_name, memory in self._memories.items()
            if memory.field == field
        ]

        for topic_name in topics_to_delete:
            del self._memories[topic_name]

        if topics_to_delete:
            logger.info(f"Cleared {len(topics_to_delete)} memories for field: {field}")

        return len(topics_to_delete)


# Global shared memory store instance
_shared_memory_store: SharedMemoryStore | None = None


def get_shared_memory_store() -> SharedMemoryStore:
    """Get the global shared memory store instance."""
    global _shared_memory_store
    if _shared_memory_store is None:
        _shared_memory_store = SharedMemoryStore()
    return _shared_memory_store
