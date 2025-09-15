# Lab 3: Content Improvement - Topic Deduplication & Editorial Memory

Welcome to Lab 3! In this lab, you'll implement an intelligent content improvement system that prevents topic repetition by giving the editor agent memory of current and past topics. The editor will maintain a forbidden topics list and ensure reporters avoid duplicate content.

## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Create a topic memory system for the editor agent
- ✅ Implement a forbidden topics tool for editorial control
- ✅ Build a persistent topic tracking file system
- ✅ Integrate topic filtering into the reporter workflow
- ✅ Test the complete deduplication system
- ✅ Understand how to maintain editorial consistency without databases

## 📋 Prerequisites

- ✅ Completed Lab 1 (Basic setup) and Lab 2 (YouTube integration)
- ✅ Working DevContainer or local development environment
- ✅ Understanding of the editor-reporter workflow
- ✅ Basic knowledge of file-based persistence

## What You Will Build (Goal)

In this lab, you will give the editor a lightweight, file‑backed memory of recently covered topics and use it to prevent duplicate coverage by injecting a FORBIDDEN TOPICS block into reporter task guidelines.

Success criteria:
- A JSON file at data/topics/covered_topics.json is created and updated
- Reporter tasks include a FORBIDDEN TOPICS block in guidelines
- After publishing a story, its topic is appended to memory
- A quick test prints forbidden topics and similarity results

You will implement:
- Topic memory models and a TopicMemoryService (file-based persistence)
- Editor-side helper that injects forbidden topics into ReporterTask.guidelines
- Small tests and a utility script to inspect/manage the memory

## 🚀 Step 1: Design the Topic Memory System

### 1.1 Topic Memory Architecture

The system will work as follows:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Editor Agent  │────│ Forbidden Topics │────│ Topic Memory    │
│                 │    │      Tool        │    │     File        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Reporter Agents │    │ Topic Filtering  │    │ data/topics/    │
│                 │    │   & Validation   │    │ covered_topics  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 1.2 Data Structure Design

**Topic Memory File Format:**
```json
{
  "last_updated": "2024-01-15T10:30:00Z",
  "topics": [
    {
      "title": "OpenAI GPT-5 Release",
      "description": "Coverage of OpenAI's latest language model release and its capabilities",
      "date_added": "2024-01-15"
    },
    {
      "title": "Climate Change Impact on Agriculture",
      "description": "Analysis of how rising temperatures affect crop yields globally",
      "date_added": "2024-01-14"
    }
  ]
}
```

**Key Design Choices:**
- Essential fields only: `title`, `description`, `date_added`
- Simple date-based filtering for recent topics
- Designed for prompt injection into reporter guidelines

### 1.3 Workflow Design

1. **Editor Initialization**: Load existing topic memory from file
2. **Topic Collection**: Editor gets forbidden topics list and injects it into reporter prompts
3. **Story Assignment**: Editor assigns unique topics to reporters with forbidden topics in prompt
4. **Story Publication**: Editor adds new topics to memory file after publication
5. **Memory Persistence**: Topic memory is saved to file for future sessions

**Key Features:**
- Forbidden topics are injected directly into reporter prompts
- Reporters receive forbidden topics as part of their task guidelines
- Simple prompt injection approach - no complex tool orchestration needed

## 🔧 Step 2: Create Topic Memory Models

### 2.1 Create Topic Memory Data Models

Create the file `libs/common/utils/topic_memory_models.py`:

```python
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
```

### 2.2 Create Topic Memory Service

Create the file `libs/common/utils/topic_memory_service.py`:

```python
"""Topic memory service for managing covered topics and preventing duplication."""

import json
from pathlib import Path
from typing import Optional

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
                if self._calculate_similarity(proposed_lower, existing_lower) > 0.7:
                    similar.append(existing_topic.title)

            similar_topics[proposed] = similar

        return similar_topics

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two text strings using word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

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
```

This is the foundation of our topic memory system. The key features are:
- Clean data structure with only essential fields
- All topics are treated equally for maximum simplicity
- Forbidden topics are injected into reporter prompts
- Focus on effective deduplication with minimal complexity

## 🔄 Next Steps Preview

In the remaining steps, we will:
- Update the editor agent to use topic memory for prompt injection
- Modify the reporter workflow to receive forbidden topics in prompts
- Test the complete deduplication system
- Create management utilities for topic memory

Ready to continue building the editorial intelligence system!

## 🛠️ Step 3: Update Editor Agent for Prompt Injection

### 3.1 Modify Editor Agent to Inject Forbidden Topics

Instead of creating a complex tool, the editor will simply inject forbidden topics into reporter prompts:

```python
"""Editor agent integration with topic memory."""

from datetime import datetime
from agents.models.task_models import ReporterTask
from agents.types import TaskType, ReporterField
from utils.topic_memory_service import TopicMemoryService
from utils.topic_memory_models import CoveredTopic


class EditorAgentWithMemory:
    """Editor agent enhanced with topic memory for deduplication."""

    def __init__(self, config_service):
        """Initialize editor with topic memory service."""
        self.topic_memory = TopicMemoryService(config_service)

    def create_reporter_task_with_forbidden_topics(
        self,
        task_type: TaskType,
        field: ReporterField,
        description: str,
        days_back: int = 30
    ) -> ReporterTask:
        """Create a reporter task with forbidden topics injected into guidelines."""

        # Get forbidden topics as formatted text
        forbidden_text = self.topic_memory.get_forbidden_topics_as_text(days_back)

        # Create enhanced guidelines with forbidden topics
        enhanced_guidelines = f"""
{description}

{forbidden_text}

IMPORTANT: Ensure your proposed topics are unique and not similar to the forbidden topics listed above.
Focus on fresh angles and new developments that haven't been covered recently.
"""

        return ReporterTask(
            name=task_type,
            field=field,
            description=description,
            guidelines=enhanced_guidelines.strip()
        )

    def add_published_story_to_memory(self, title: str, description: str) -> bool:
        """Add a published story to topic memory."""
        topic = CoveredTopic(
            title=title,
            description=description,
            date_added=datetime.now().strftime("%Y-%m-%d")
        )

        return self.topic_memory.add_topic(topic)


# Example usage in editor workflow:
"""
editor = EditorAgentWithMemory(config_service)

# When creating reporter tasks:
task = editor.create_reporter_task_with_forbidden_topics(
    task_type=TaskType.FIND_TOPICS,
    field=ReporterField.TECHNOLOGY,
    description="Find 3 trending topics in technology",
    days_back=30
)

# After publishing a story:
editor.add_published_story_to_memory(
    title="New AI Breakthrough at OpenAI",
    description="OpenAI announces major advancement in language models"
)
"""
```

#### 3.1b Minimal example (uses current ReporterTask)

```python
from core.config_service import ConfigService
from agents.models.task_models import ReporterTask
from agents.types import TaskType, ReporterField
from utils.topic_memory_service import TopicMemoryService

config = ConfigService()
### 3.3 Integrate Forbidden Topics in Current Editor Orchestration (no structural changes)

Inject the FORBIDDEN TOPICS text when creating reporter tasks inside the existing CollectTopicsTool. This requires no class/interface changes.

Example (excerpt to adapt in agents/editor_agent/editor_tools.py):
<augment_code_snippet path="libs/common/agents/editor_agent/editor_tools.py" mode="EXCERPT">
````python
config = ConfigService()
forbidden = TopicMemoryService(config).get_forbidden_topics_as_text(days_back=30)
base = "Focus on current, newsworthy topics with broad appeal"
guidelines = f"{base}\n\n{forbidden}\n\nIMPORTANT: Avoid duplicates."

task = ReporterTask(
    name=TaskType.FIND_TOPICS,
    field=request.field,
    sub_section=request.sub_section,
    description=f"Find {params.topics_per_field} trending topics in {request.field.value}{sub_section_desc}",
    guidelines=guidelines,
)
````
</augment_code_snippet>

This integrates topic memory into reporter task guidelines without changing the editor or tool interfaces.

memory = TopicMemoryService(config)
forbidden = memory.get_forbidden_topics_as_text(days_back=30)

task = ReporterTask(
    name=TaskType.FIND_TOPICS,
    field=ReporterField.TECHNOLOGY,
    description="Find 3 trending topics in technology",
    sub_section=None,
    guidelines=f"{forbidden}\n\nEnsure your topics are unique.",
    min_sources=1,
    target_word_count=0,
    require_images=False,
)
```


### 3.2 Update Editor Agent System Prompt

Update the editor agent's system prompt to include topic memory responsibilities:

```python
# In libs/common/agents/editor_agent/editor_prompt.py

EDITOR_SYSTEM_PROMPT = """
You are the Editor-in-Chief of BobTimes, an AI-powered newspaper. Your responsibilities include:

1. **Topic Collection & Deduplication**:
   - Before assigning topics to reporters, check recent topic memory
   - Inject forbidden topics directly into reporter task guidelines
   - Ensure no topic repetition within the last 30 days

2. **Story Assignment**:
   - Create reporter tasks with forbidden topics included in guidelines
   - Ensure diverse coverage across different fields
   - Provide clear instructions to avoid duplicate content

3. **Editorial Review**:
   - Review story drafts for quality and accuracy
   - Check that stories don't duplicate recent coverage
   - Provide feedback for improvements

4. **Publication Management**:
   - Publish approved stories
   - Add published topics to memory for future deduplication
   - Maintain editorial consistency across news cycles

**WORKFLOW:**
1. Before creating reporter tasks: Get forbidden topics from memory
2. When creating tasks: Inject forbidden topics into task guidelines
3. After publishing stories: Add new topics to memory

Always prioritize content uniqueness and editorial quality.
"""
```

## 🧪 Step 4: Test Topic Memory System

### 4.1 Quick Check: Print Forbidden Topics (inline)

Paste the following into a Python shell (e.g., uv run python) to quickly verify the topic memory service:

```python
"""Quick check for topic memory system (run inline)."""

import asyncio
from datetime import datetime

from core.config_service import ConfigService
from utils.topic_memory_service import TopicMemoryService
from utils.topic_memory_models import CoveredTopic


async def test_topic_memory():
    """Test the topic memory system."""
    print("🧠 Topic Memory System Test")
    print("=" * 40)

    config_service = ConfigService()
    memory_service = TopicMemoryService(config_service)

    # Test 1: Add some sample topics
    print("\n1️⃣ Adding sample topics...")

    sample_topics = [
        CoveredTopic(
            title="OpenAI GPT-5 Release",
            description="Coverage of OpenAI's latest language model and its capabilities",
            date_added="2024-01-15"
        ),
        CoveredTopic(
            title="Climate Change Impact on Agriculture",
            description="Analysis of rising temperatures affecting global crop yields",
            date_added="2024-01-14"
        ),
        CoveredTopic(
            title="Federal Reserve Interest Rate Decision",
            description="Analysis of the Fed's latest monetary policy changes",
            date_added="2024-01-13"
        )
    ]

    for topic in sample_topics:
        success = memory_service.add_topic(topic)
        if success:
            print(f"   ✅ Added: {topic.title}")
        else:
            print(f"   ❌ Failed to add: {topic.title}")

    # Test 2: Get forbidden topics
    print("\n2️⃣ Getting forbidden topics...")

    forbidden_topics = memory_service.get_forbidden_topics(days_back=30)
    print(f"   ✅ Found {len(forbidden_topics)} forbidden topics")
    for topic in forbidden_topics:
        print(f"      - {topic.title}")

    # Test 3: Test forbidden topics as text for prompt injection
    print("\n3️⃣ Getting forbidden topics as formatted text...")

    forbidden_text = memory_service.get_forbidden_topics_as_text(days_back=30)
    print("   📝 Formatted text for prompt injection:")
    print(f"   {forbidden_text}")

    # Test 4: Check similarity for proposed topics
    print("\n4️⃣ Checking similarity for proposed topics...")

    proposed_topics = [
        "OpenAI's New AI Model Launch",  # Should be similar to existing
        "Quantum Computing Breakthrough",  # Should be unique
        "Climate Effects on Farming"  # Should be similar to existing
    ]

    similar_topics = memory_service.check_topic_similarity(proposed_topics)
    print(f"   ✅ Similarity check completed")
    for proposed, similar in similar_topics.items():
        if similar:
            print(f"      ⚠️  '{proposed}' is similar to: {similar}")
        else:
            print(f"      ✅ '{proposed}' is unique")

    # Test 5: Get memory statistics
    print("\n5️⃣ Memory statistics...")
    stats = memory_service.get_memory_stats()
    print(f"   📊 Total topics: {stats['total_topics']}")
    print(f"   📊 Recent topics (7 days): {stats['recent_topics_7_days']}")
    print(f"   📊 Recent topics (30 days): {stats['recent_topics_30_days']}")

    print("\n🎉 Topic memory test completed!")


# In a Python shell call:
# import asyncio; asyncio.run(test_topic_memory())
```

### 4.2 Run the Quick Check

- Open a Python shell: uv run python
- Paste the snippet from 4.1
- Then run: import asyncio; asyncio.run(test_topic_memory())

## 🔗 Step 5: Create Complete Integration Example

### 5.1 Editor Workflow Integration Example (inline)

A complete example showing how the editor integrates with topic memory:

```python
"""Editor integration with topic memory (inline example)."""

import asyncio
from datetime import datetime

from core.config_service import ConfigService
from utils.topic_memory_service import TopicMemoryService
from utils.topic_memory_models import CoveredTopic
from agents.models.task_models import ReporterTask
from agents.types import TaskType, ReporterField


class EditorWithMemory:
    """Editor agent with topic memory integration."""

    def __init__(self, config_service):
        """Initialize editor with topic memory."""
        self.topic_memory = TopicMemoryService(config_service)

    def create_reporter_task_with_forbidden_topics(
        self,
        task_type: TaskType,
        field: ReporterField,
        description: str,
        days_back: int = 30
    ) -> ReporterTask:
        """Create reporter task with forbidden topics in guidelines."""

        # Get forbidden topics as formatted text
        forbidden_text = self.topic_memory.get_forbidden_topics_as_text(days_back)

        # Create enhanced guidelines
        enhanced_guidelines = f"""
{description}

{forbidden_text}

IMPORTANT: Ensure your proposed topics are unique and not similar to the forbidden topics listed above.
Focus on fresh angles and new developments that haven't been covered recently.
"""

        return ReporterTask(
            name=task_type,
            field=field,
            description=description,
            guidelines=enhanced_guidelines.strip()
        )

    def add_published_story_to_memory(self, title: str, description: str) -> bool:
        """Add published story to topic memory."""
        topic = CoveredTopic(
            title=title,
            description=description,
            date_added=datetime.now().strftime("%Y-%m-%d")
        )
        return self.topic_memory.add_topic(topic)


async def test_editor_integration():
    """Test complete editor integration with topic memory."""
    print("🗞️ Editor Integration with Topic Memory Test")
    print("=" * 60)

    config_service = ConfigService()
    editor = EditorWithMemory(config_service)

    # Step 1: Add some existing topics to memory
    print("\n1️⃣ Setting up existing topics in memory...")

    existing_topics = [
        CoveredTopic(
            title="Apple's New iPhone Features",
            description="Apple announces AI-powered camera improvements",
            date_added="2024-01-15"
        ),
        CoveredTopic(
            title="NASA Mars Mission Update",
            description="Latest findings from Mars rover exploration",
            date_added="2024-01-14"
        )
    ]

    for topic in existing_topics:
        success = editor.topic_memory.add_topic(topic)
        print(f"   {'✅' if success else '❌'} Added: {topic.title}")

    # Step 2: Create reporter task with forbidden topics
    print("\n2️⃣ Creating reporter task with forbidden topics...")

    task = editor.create_reporter_task_with_forbidden_topics(
        task_type=TaskType.FIND_TOPICS,
        field=ReporterField.TECHNOLOGY,
        description="Find 3 trending topics in technology",
        days_back=30
    )

    print(f"   📝 Created task for {task.field.value}")
    print(f"   📋 Task guidelines include forbidden topics:")
    print(f"   {task.guidelines[:200]}...")

    # Step 3: Simulate story publication
    print("\n3️⃣ Simulating story publication...")

    new_story = {
        "title": "Google's Quantum Computing Breakthrough",
        "description": "Google announces major advancement in quantum error correction"
    }

    success = editor.add_published_story_to_memory(
        title=new_story["title"],
        description=new_story["description"]
    )

    print(f"   {'✅' if success else '❌'} Published and added to memory: {new_story['title']}")

    # Step 4: Verify the new topic is now forbidden
    print("\n4️⃣ Verifying new topic is now in forbidden list...")

    forbidden_text = editor.topic_memory.get_forbidden_topics_as_text(days_back=1)
    if new_story["title"] in forbidden_text:
        print(f"   ✅ New topic is now in forbidden list")
    else:
        print(f"   ⚠️  New topic not found in forbidden list")

    # Step 5: Show memory statistics
    print("\n5️⃣ Final memory statistics...")
    stats = editor.topic_memory.get_memory_stats()
    print(f"   📊 Total topics: {stats['total_topics']}")
    print(f"   📊 Recent topics (7 days): {stats['recent_topics_7_days']}")
    print(f"   📊 Recent topics (30 days): {stats['recent_topics_30_days']}")

    print("\n🎉 Editor integration test completed!")


# In a Python shell call:
# import asyncio; asyncio.run(test_editor_integration())
```

### 5.2 Use Existing Task Models (no changes required)

You do not need to modify the task models. The current ReporterTask already has the fields you need, including guidelines for injecting forbidden topics.

Example (excerpt from agents/models/task_models.py):
```python
class ReporterTask(BaseModel):
    name: TaskType
    field: ReporterField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None
    description: str
    guidelines: str | None = None
    min_sources: int = 1
    target_word_count: int = 500
    require_images: bool = False
```

Tip: Use the guidelines field to pass the FORBIDDEN TOPICS block into reporter tasks without changing any model definitions.

## 🧪 Step 6: Complete Testing Suite

### 6.1 Quick Checks (no test files required)

Use the inline quick checks from Step 4 and Step 5 inside a Python shell to validate behavior. There are no test files in this repository.

### 6.2 What To Look For

- Added sample topics are persisted (you see them listed)
- Forbidden topics text shows a bullet list with titles and descriptions
- Similarity check flags clearly related topics
- Memory statistics report totals and recent counts

1️⃣ Adding sample topics...
   ✅ Added: OpenAI GPT-5 Release
   ✅ Added: Climate Change Impact on Agriculture
   ✅ Added: Federal Reserve Interest Rate Decision

2️⃣ Getting forbidden topics...
   ✅ Found 3 forbidden topics
      - OpenAI GPT-5 Release
      - Climate Change Impact on Agriculture
      - Federal Reserve Interest Rate Decision

3️⃣ Getting forbidden topics as formatted text...
   📝 Formatted text for prompt injection:
   FORBIDDEN TOPICS (avoid these recent topics):
   - OpenAI GPT-5 Release: Coverage of OpenAI's latest language model and its capabilities
   - Climate Change Impact on Agriculture: Analysis of rising temperatures affecting global crop yields
   - Federal Reserve Interest Rate Decision: Analysis of the Fed's latest monetary policy changes

4️⃣ Checking similarity for proposed topics...
   ✅ Similarity check completed
      ⚠️  'OpenAI's New AI Model Launch' is similar to: ['OpenAI GPT-5 Release']
      ✅ 'Quantum Computing Breakthrough' is unique
      ⚠️  'Climate Effects on Farming' is similar to: ['Climate Change Impact on Agriculture']

5️⃣ Memory statistics...
   📊 Total topics: 3
   📊 Recent topics (7 days): 3
   📊 Recent topics (30 days): 3

🎉 Topic memory test completed!
```

**Test 2 - Editor Integration:**
```
🗞️ Editor Integration with Topic Memory Test
============================================================

1️⃣ Setting up existing topics in memory...
   ✅ Added: Apple's New iPhone Features
   ✅ Added: NASA Mars Mission Update

2️⃣ Creating reporter task with forbidden topics...
   📝 Created task for technology
   📋 Task guidelines include forbidden topics:
   Find 3 trending topics in technology

   FORBIDDEN TOPICS (avoid these recent topics):
   - Apple's New iPhone Features: Apple announces AI-powered camera improvements
   - NASA Mars Mission Update: Latest findings from Mars rover exploration...

3️⃣ Simulating story publication...
   ✅ Published and added to memory: Google's Quantum Computing Breakthrough

4️⃣ Verifying new topic is now in forbidden list...
   ✅ New topic is now in forbidden list

5️⃣ Final memory statistics...
   📊 Total topics: 3
   📊 Recent topics (7 days): 3
   📊 Recent topics (30 days): 3

🎉 Editor integration test completed!
```

## 🔧 Step 7: Create Topic Memory Management Utilities

### 7.1 Create Topic Memory Manager Script (optional)

Create a utility script for managing the topic memory:

```python
# Create file: libs/common/utils/manage_topic_memory.py
"""Utility script for managing topic memory."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from core.config_service import ConfigService
from utils.topic_memory_service import TopicMemoryService


async def main():
    """Main topic memory management interface."""
    print("🧠 Topic Memory Manager")
    print("=" * 30)

    config_service = ConfigService()
    memory_service = TopicMemoryService(config_service)

    while True:
        print("\nAvailable commands:")
        print("1. View memory statistics")
        print("2. List recent topics")
        print("3. Search topics by keyword")
        print("4. Export memory to JSON")
        print("5. Clear old topics")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            await show_statistics(memory_service)
        elif choice == "2":
            await list_recent_topics(memory_service)
        elif choice == "3":
            await search_topics(memory_service)
        elif choice == "4":
            await export_memory(memory_service)
        elif choice == "5":
            await clear_old_topics(memory_service)
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


async def show_statistics(memory_service: TopicMemoryService):
    """Show memory statistics."""
    stats = memory_service.get_memory_stats()

    print("\n📊 Topic Memory Statistics:")
    print(f"   Total topics: {stats['total_topics']}")
    print(f"   Last updated: {stats['last_updated']}")
    print(f"   Recent topics (7 days): {stats['recent_topics_7_days']}")
    print(f"   Recent topics (30 days): {stats['recent_topics_30_days']}")


async def list_recent_topics(memory_service: TopicMemoryService):
    """List recent topics."""
    days = input("Enter number of days to look back (default 7): ").strip()
    days = int(days) if days.isdigit() else 7

    recent_topics = memory_service.memory.get_recent_topics(days)

    print(f"\n📋 Topics from last {days} days ({len(recent_topics)} found):")
    for i, topic in enumerate(recent_topics, 1):
        print(f"   {i}. {topic.title}")
        print(f"      Date: {topic.date_added}")
        print(f"      Description: {topic.description}")
        print()


async def search_topics(memory_service: TopicMemoryService):
    """Search topics by keyword."""
    keyword = input("Enter keyword to search: ").strip().lower()

    if not keyword:
        print("❌ Please enter a keyword.")
        return

    matching_topics = []
    for topic in memory_service.memory.topics:
        if (keyword in topic.title.lower() or
            keyword in topic.description.lower()):
            matching_topics.append(topic)

    print(f"\n🔍 Topics matching '{keyword}' ({len(matching_topics)} found):")
    for i, topic in enumerate(matching_topics, 1):
        print(f"   {i}. {topic.title}")
        print(f"      Date: {topic.date_added}")
        print(f"      Description: {topic.description}")
        print()


async def export_memory(memory_service: TopicMemoryService):
    """Export memory to JSON file."""
    filename = f"topic_memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(memory_service.memory.model_dump(), f, indent=2, default=str)

        print(f"✅ Memory exported to {filename}")
        print(f"📊 Exported {memory_service.memory.total_topics} topics")
    except Exception as e:
        print(f"❌ Export failed: {e}")


async def clear_old_topics(memory_service: TopicMemoryService):
    """Clear topics older than specified days."""
    days = input("Enter number of days to keep (topics older will be removed): ").strip()

    if not days.isdigit():
        print("❌ Please enter a valid number.")
        return

    days = int(days)
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    old_topics = [topic for topic in memory_service.memory.topics if topic.date_added < cutoff_date]

    if not old_topics:
        print(f"✅ No topics older than {days} days found.")
        return

    print(f"⚠️  Found {len(old_topics)} topics older than {days} days:")
    for topic in old_topics[:5]:  # Show first 5
        print(f"   - {topic.title} ({topic.date_added})")

    if len(old_topics) > 5:
        print(f"   ... and {len(old_topics) - 5} more")

    confirm = input(f"\nAre you sure you want to delete {len(old_topics)} topics? (y/N): ").strip().lower()

    if confirm == 'y':
        # Remove old topics
        memory_service.memory.topics = [topic for topic in memory_service.memory.topics if topic.date_added >= cutoff_date]
        memory_service.memory.total_topics = len(memory_service.memory.topics)
        memory_service.memory.last_updated = datetime.now()

        # Save updated memory
        memory_service._save_memory()

        print(f"✅ Removed {len(old_topics)} old topics")
        print(f"📊 {memory_service.memory.total_topics} topics remaining")
    else:
        print("❌ Operation cancelled")


if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 Step 8: Final Integration and Summary

### 8.1 Quick Checks Recap

- Run the inline quick checks from Step 4 and Step 5 inside a Python shell
- Optional: run the interactive manager if you created it

```bash
python libs/common/utils/manage_topic_memory.py
```

### 8.2 Design Focus

This approach focuses on the essential functionality:

1. Data structure focused on essentials:
   - Fields: `title`, `description`, `date_added`

2. Prompt injection (no extra tools):
   - Editor injects forbidden topics directly into `ReporterTask.guidelines`

3. Unified topic memory:
   - Single memory for all fields with date-based filtering and simple word-overlap similarity

4. Streamlined workflow:
   - Load forbidden topics → inject into guidelines → publish → add to memory

### 8.3 Verify File Structure

Check that the topic memory files are created correctly:

```bash
# Check the data directory structure
ls -la data/topics/

# View the topic memory file
cat data/topics/covered_topics.json
```

Expected file structure:
```
data/
└── topics/
    └── covered_topics.json
```

Expected JSON content:
```json
{
  "last_updated": "2024-01-15T10:30:00.123456",
  "topics": [
    {
      "title": "OpenAI GPT-5 Release",
      "description": "Coverage of OpenAI's latest language model and its capabilities",
      "date_added": "2024-01-15"
    },
    {
      "title": "Climate Change Impact on Agriculture",
      "description": "Analysis of rising temperatures affecting global crop yields",
      "date_added": "2024-01-14"
    }
  ],
  "total_topics": 2
}
```

## 🎯 Step 9: Integration with Full Newspaper Generation

### 9.1 Update Newspaper Generation Script

Modify your newspaper generation script to use the simplified topic memory system:

```python
# Update your existing newspaper generation script
# Add this to the beginning of the generation process

async def generate_newspaper_with_memory():
    """Generate newspaper with topic memory integration."""
    print("🗞️ Generating Newspaper with Topic Memory")
    print("=" * 50)

    config_service = ConfigService()
    editor_with_memory = EditorWithMemory(config_service)

    # Step 1: Create reporter tasks with forbidden topics
    fields = [ReporterField.TECHNOLOGY, ReporterField.SCIENCE, ReporterField.ECONOMICS]

    for field in fields:
        # Create task with forbidden topics injected into guidelines
        task = editor_with_memory.create_reporter_task_with_forbidden_topics(
            task_type=TaskType.FIND_TOPICS,
            field=field,
            description=f"Find 3 trending topics in {field.value}",
            days_back=30
        )

        print(f"📝 Created {field.value} task with forbidden topics in guidelines")

        # Your existing reporter execution logic here...
        # reporter = factory.create_reporter(field)
        # topics = await reporter.execute_task(task)

    # Step 2: After publishing each story, add to memory
    # Example for a published story:
    """
    success = editor_with_memory.add_published_story_to_memory(
        title=story.title,
        description=story.summary
    )

    if success:
        print(f"✅ Added story to memory: {story.title}")
    """

    print("✅ Newspaper generation with topic memory completed!")
```

## 🎉 Step 10: Lab Summary and Next Steps

### 10.1 What You've Built

Congratulations! You've successfully implemented a topic memory system that:

1. **Prevents Content Duplication**: Tracks covered topics and prevents repetition
2. **Simple Data Structure**: Uses only essential fields (title, description, date_added)
3. **Prompt Injection**: Injects forbidden topics directly into reporter guidelines
4. **File-Based Persistence**: Stores topic memory in JSON files without databases
5. **Basic Similarity Detection**: Uses word overlap to detect similar topics

### 10.2 Key Benefits of This Approach

- **Easier to Implement**: Fewer models, tools, and complex workflows
- **Easier to Maintain**: Simple data structure and straightforward logic
- **Easier to Debug**: Clear separation between memory storage and prompt injection
- **Easier to Extend**: Can add complexity later if needed

### 10.3 Files Created in This Lab

```
libs/common/utils/
├── topic_memory_models.py      # Data models
├── topic_memory_service.py     # Memory management service
└── manage_topic_memory.py      # Management utility (optional)

data/topics/
└── covered_topics.json         # Persistent topic storage
```

### 10.4 Next Steps

After completing this lab, you can:

1. **Integrate with Your Editor Agent**: Use the `EditorWithMemory` class
2. **Enhance Similarity Detection**: Add more sophisticated algorithms if needed
3. **Add Topic Categories**: Extend the model to include field categorization
4. **Implement Auto-Cleanup**: Add scheduled cleanup of old topics
5. **Add Analytics**: Track topic trends and coverage patterns

### 10.5 Testing Your Implementation

Run these commands to verify everything works:

```bash
# Interactive management (optional, if you created it)
python libs/common/utils/manage_topic_memory.py
```

### 10.6 Design Principles

1. Streamlined data structure (title, description, date_added)
2. Prompt injection approach via `ReporterTask.guidelines`
3. Unified topic memory with simple date filtering and similarity
4. Clean workflow: load → inject → publish → persist

These choices keep the lab focused, easy to implement, and easy to extend later.

🎯 **Lab 3 Complete!** You now have a working topic deduplication system that will prevent your AI newspaper from repeating content across news cycles while maintaining clean, maintainable code.
