"""Topic memory service for managing covered topics and preventing duplication."""

import json
from pathlib import Path
from typing import Optional
import re
import unicodedata


from core.config_service import ConfigService
from core.logging_service import get_logger

from .topic_memory_models import TopicMemory, CoveredTopic

logger = get_logger(__name__)


class TopicMemoryService:
    """Service for managing topic memory and preventing content duplication."""

    def __init__(self, config_service: ConfigService):
        """Initialize topic memory service."""
        self.config_service = config_service
        self.memory_file_path = Path("data/topics/covered_topics.json")
        self.memory: Optional[TopicMemory] = None

        # Ensure directory exists
        self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing memory
        self._load_memory()

        logger.info(f"Topic memory service initialized with {self.memory.total_topics} topics")

    def _load_memory(self) -> None:
        """Load topic memory from file."""
        try:
            if self.memory_file_path.exists():
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory = TopicMemory.model_validate(data)
                logger.info(f"Loaded {self.memory.total_topics} topics from memory file")
            else:
                self.memory = TopicMemory()
                logger.info("Created new topic memory (no existing file found)")
        except Exception as e:
            logger.error(f"Failed to load topic memory: {e}")
            self.memory = TopicMemory()

    def _save_memory(self) -> None:
        """Save topic memory to file."""
        try:
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved topic memory with {self.memory.total_topics} topics")
        except Exception as e:
            logger.error(f"Failed to save topic memory: {e}")

    def add_topic(self, topic: CoveredTopic) -> bool:
        """Add a new topic to memory and save to file."""
        try:
            self.memory.add_topic(topic)
            self._save_memory()
            logger.info(f"Added topic to memory: {topic.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to add topic to memory: {e}")
            return False

    def get_forbidden_topics(self, days_back: int = 30) -> list[CoveredTopic]:
        """Get list of topics that should be avoided."""
        return self.memory.get_recent_topics(days_back)

    def check_topic_similarity(self, proposed_topics: list[str]) -> dict[str, list[str]]:
        """Check if proposed topics are similar to existing ones using simple word matching."""
        similar_topics = {}

        for proposed in proposed_topics:
            similar = []
            proposed_lower = proposed.lower()

            for existing_topic in self.memory.topics:
                existing_lower = existing_topic.title.lower()

                # Simple similarity check using word overlap
-                if self._calculate_similarity(proposed_lower, existing_lower) > 0.7:
+                if self._calculate_similarity(proposed_lower, existing_lower) >= 0.6:
                    similar.append(existing_topic.title)

            similar_topics[proposed] = similar

        return similar_topics

    def _normalize(self, text: str) -> list[str]:
        """Normalize text for robust similarity: lowercase, NFKC, unify dashes, strip punctuation, remove stopwords."""
        import re
        import unicodedata

        s = unicodedata.normalize("NFKC", text.lower())
        # Unify various dash characters to '-'
        s = re.sub(r"[‐‑‒–—―]", "-", s)
        # Remove punctuation except dash and word chars
        s = re.sub(r"[^\w\s-]", " ", s)
        tokens = [t for t in s.split() if t not in {"the", "a", "an", "of", "for", "to", "and", "in", "on", "with", "by"}]
        return tokens

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings using Jaccard over normalized tokens."""
        words1 = set(self._normalize(text1))
        words2 = set(self._normalize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def dedupe_topics(self, topics: list[str], threshold: float = 0.6) -> list[str]:
        """Remove near-duplicates within the provided topic list using similarity threshold."""
        kept: list[str] = []
        for t in topics:
            if any(self._calculate_similarity(t, k) >= threshold for k in kept):
                continue
            kept.append(t)
        return kept

    def filter_against_memory(self, topics: list[str], days_back: int = 30, threshold: float = 0.6) -> list[str]:
        """Filter out topics that are similar to recently covered topics and dedupe within the batch."""
        recent_titles = [t.title for t in self.get_forbidden_topics(days_back)]
        allowed: list[str] = []
        for t in topics:
            # Exclude if similar to any recent memory topic
            if any(self._calculate_similarity(t, r) >= threshold for r in recent_titles):
                continue
            # Exclude if similar to something already kept (intra-batch dedupe)
            if any(self._calculate_similarity(t, k) >= threshold for k in allowed):
                continue
            allowed.append(t)
        return allowed

    def get_memory_stats(self) -> dict:
        """Get statistics about the topic memory."""
        return {
            "total_topics": self.memory.total_topics,
            "last_updated": self.memory.last_updated.isoformat(),
            "recent_topics_30_days": len(self.memory.get_recent_topics(30)),
            "recent_topics_7_days": len(self.memory.get_recent_topics(7))
        }

    def get_forbidden_topics_as_text(self, days_back: int = 30) -> str:
        """Get forbidden topics formatted as text for injection into reporter prompts."""
        forbidden_topics = self.get_forbidden_topics(days_back)

        if not forbidden_topics:
            return "No recent topics to avoid."

        text_lines = ["FORBIDDEN TOPICS (avoid these recent topics):"]
        for topic in forbidden_topics:
            text_lines.append(f"- {topic.title}: {topic.description}")

        return "\n".join(text_lines)
