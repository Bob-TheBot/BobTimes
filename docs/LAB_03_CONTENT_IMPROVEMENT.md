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

## 🚀 Step 1: Design the Topic Memory System

### 1.1 Topic Memory Architecture

The system will work as follows:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Editor Agent │────│ Forbidden Topics │────│ Topic Memory    │
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
      "field": "technology",
      "sub_section": "artificial_intelligence",
      "published_date": "2024-01-15",
      "story_id": "story-tech-001",
      "keywords": ["openai", "gpt-5", "ai", "language model"]
    },
    {
      "title": "Climate Change Impact on Agriculture",
      "description": "Analysis of how rising temperatures affect crop yields globally",
      "field": "science",
      "sub_section": "environmental_science",
      "published_date": "2024-01-14",
      "story_id": "story-sci-002",
      "keywords": ["climate change", "agriculture", "crops", "global warming"]
    }
  ]
}
```

### 1.3 Workflow Design

1. **Editor Initialization**: Load existing topic memory from file
2. **Topic Collection**: Reporters request topics, editor filters against forbidden list
3. **Story Assignment**: Editor assigns unique topics to reporters
4. **Story Publication**: Editor adds new topics to memory file
5. **Memory Persistence**: Topic memory is saved to file for future sessions

## 🔧 Step 2: Create Topic Memory Models

### 2.1 Create Topic Memory Data Models

Create the file `libs/common/utils/topic_memory_models.py`:

```python
"""Topic memory models for editorial content management."""

from datetime import datetime
from enum import StrEnum
from typing import Optional

from agents.types import ReporterField, TechnologySubSection, ScienceSubSection, EconomicsSubSection
from pydantic import BaseModel, Field


class TopicStatus(StrEnum):
    """Status of a covered topic."""
    PUBLISHED = "published"
    IN_PROGRESS = "in_progress"
    ARCHIVED = "archived"


class CoveredTopic(BaseModel):
    """Represents a topic that has been covered or is being covered."""
    title: str = Field(description="Title of the covered topic")
    description: str = Field(description="One-line description of the topic coverage")
    field: ReporterField = Field(description="News field (technology, science, economics)")
    sub_section: Optional[TechnologySubSection | ScienceSubSection | EconomicsSubSection] = Field(
        None, description="Sub-section within the field"
    )
    published_date: str = Field(description="Date when topic was published (YYYY-MM-DD)")
    story_id: str = Field(description="Unique identifier for the story")
    keywords: list[str] = Field(default_factory=list, description="Keywords associated with the topic")
    status: TopicStatus = Field(default=TopicStatus.PUBLISHED, description="Current status of the topic")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for duplicate detection")


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
    
    def get_topics_by_field(self, field: ReporterField) -> list[CoveredTopic]:
        """Get all topics for a specific field."""
        return [topic for topic in self.topics if topic.field == field]
    
    def get_recent_topics(self, days: int = 30) -> list[CoveredTopic]:
        """Get topics published within the last N days."""
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        return [topic for topic in self.topics if topic.published_date >= cutoff_str]


class ForbiddenTopicsParams(BaseModel):
    """Parameters for forbidden topics tool operations."""
    operation: str = Field(description="Operation: 'get_forbidden', 'add_topic', or 'check_similarity'")
    field: Optional[ReporterField] = Field(None, description="Field to filter topics (optional)")
    days_back: int = Field(default=30, description="Days to look back for recent topics")
    new_topic: Optional[CoveredTopic] = Field(None, description="New topic to add (for add_topic operation)")
    proposed_topics: list[str] = Field(default_factory=list, description="Topics to check for similarity")


class ForbiddenTopicsResult(BaseModel):
    """Result from forbidden topics tool execution."""
    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="Operation that was performed")
    forbidden_topics: list[str] = Field(default_factory=list, description="List of forbidden topic titles")
    forbidden_descriptions: list[str] = Field(default_factory=list, description="List of forbidden topic descriptions")
    similar_topics: dict[str, list[str]] = Field(default_factory=dict, description="Similar topics found for each proposed topic")
    topics_added: int = Field(default=0, description="Number of topics added to memory")
    total_topics_in_memory: int = Field(default=0, description="Total topics currently in memory")
    summary: str = Field(description="Summary of the operation results")
    error: Optional[str] = Field(None, description="Error message if operation failed")
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

from .topic_memory_models import TopicMemory, CoveredTopic, TopicStatus

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
    
    def get_forbidden_topics(self, field: Optional[str] = None, days_back: int = 30) -> list[CoveredTopic]:
        """Get list of topics that should be avoided."""
        if field:
            from agents.types import ReporterField
            field_enum = ReporterField(field)
            recent_topics = self.memory.get_recent_topics(days_back)
            return [topic for topic in recent_topics if topic.field == field_enum]
        else:
            return self.memory.get_recent_topics(days_back)
    
    def check_topic_similarity(self, proposed_topics: list[str]) -> dict[str, list[str]]:
        """Check if proposed topics are similar to existing ones."""
        similar_topics = {}
        
        for proposed in proposed_topics:
            similar = []
            proposed_lower = proposed.lower()
            
            for existing_topic in self.memory.topics:
                existing_lower = existing_topic.title.lower()
                
                # Simple similarity check - can be enhanced with more sophisticated algorithms
                if self._calculate_similarity(proposed_lower, existing_lower) > 0.7:
                    similar.append(existing_topic.title)
                
                # Also check against keywords
                for keyword in existing_topic.keywords:
                    if keyword.lower() in proposed_lower or proposed_lower in keyword.lower():
                        if existing_topic.title not in similar:
                            similar.append(existing_topic.title)
            
            similar_topics[proposed] = similar
        
        return similar_topics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two text strings."""
        # Simple word overlap similarity - can be enhanced with more sophisticated methods
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the topic memory."""
        from collections import Counter
        
        field_counts = Counter(topic.field.value for topic in self.memory.topics)
        status_counts = Counter(topic.status.value for topic in self.memory.topics)
        
        return {
            "total_topics": self.memory.total_topics,
            "last_updated": self.memory.last_updated.isoformat(),
            "topics_by_field": dict(field_counts),
            "topics_by_status": dict(status_counts),
            "recent_topics_30_days": len(self.memory.get_recent_topics(30)),
            "recent_topics_7_days": len(self.memory.get_recent_topics(7))
        }
```

This is the foundation of our topic memory system. In the next steps, we'll create the forbidden topics tool and integrate it with the editor agent workflow.

## 🔄 Next Steps Preview

In the remaining steps, we will:
- Create the ForbiddenTopicsTool for the editor agent
- Update the editor agent to use topic memory
- Modify the reporter workflow to respect forbidden topics
- Test the complete deduplication system
- Create management utilities for topic memory

Ready to continue building the editorial intelligence system!

## 🛠️ Step 3: Create Forbidden Topics Tool

### 3.1 Create the Forbidden Topics Tool

Create the file `libs/common/utils/forbidden_topics_tool.py`:

```python
"""Forbidden topics tool for editor agent to manage content deduplication."""

from datetime import datetime
from typing import Any

from agents.tools.base_tool import BaseTool
from agents.types import ReporterField
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel

from .topic_memory_models import ForbiddenTopicsParams, ForbiddenTopicsResult, CoveredTopic, TopicStatus
from .topic_memory_service import TopicMemoryService

logger = get_logger(__name__)


class ForbiddenTopicsTool(BaseTool):
    """Tool for managing forbidden topics and preventing content duplication."""

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize forbidden topics tool."""
        name = "forbidden_topics"
        description = f"""
Manage forbidden topics to prevent content duplication and maintain editorial consistency.

PARAMETER SCHEMA:
{ForbiddenTopicsParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
# Get forbidden topics for a specific field
{{"operation": "get_forbidden", "field": "technology", "days_back": 30}}

# Add a new published topic to memory
{{"operation": "add_topic", "new_topic": {{"title": "AI Breakthrough", "description": "Coverage of new AI model", "field": "technology", "published_date": "2024-01-15", "story_id": "story-001", "keywords": ["ai", "breakthrough"]}}}}

# Check if proposed topics are similar to existing ones
{{"operation": "check_similarity", "proposed_topics": ["New AI Development", "Machine Learning Advances"]}}

OPERATIONS:
- get_forbidden: Get list of topics to avoid (recent topics in specified field)
- add_topic: Add a newly published topic to the memory system
- check_similarity: Check if proposed topics are too similar to existing ones

USAGE GUIDELINES:
- Use get_forbidden before assigning topics to reporters
- Use add_topic after publishing stories to update memory
- Use check_similarity to validate topic uniqueness
- Set days_back to control how far back to look for forbidden topics
- Always provide field when getting forbidden topics for specific reporters

RETURNS:
- forbidden_topics: List of topic titles to avoid
- forbidden_descriptions: List of topic descriptions for context
- similar_topics: Dictionary mapping proposed topics to similar existing ones
- summary: Summary of operation results

This tool maintains editorial memory and prevents content duplication across news cycles.
"""
        super().__init__(name=name, description=description)
        self.params_model = ForbiddenTopicsParams
        self.config_service = config_service or ConfigService()

        # Initialize topic memory service
        try:
            self.topic_memory = TopicMemoryService(self.config_service)
        except Exception as e:
            logger.error(f"Failed to initialize topic memory service: {e}")
            self.topic_memory = None

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> ForbiddenTopicsResult:
        """Execute forbidden topics operation."""
        if not isinstance(params, ForbiddenTopicsParams):
            return ForbiddenTopicsResult(
                success=False,
                operation="unknown",
                error=f"Expected ForbiddenTopicsParams, got {type(params)}"
            )

        if not self.topic_memory:
            return ForbiddenTopicsResult(
                success=False,
                operation=params.operation,
                error="Topic memory service not available"
            )

        try:
            if params.operation == "get_forbidden":
                return await self._get_forbidden_topics(params)
            elif params.operation == "add_topic":
                return await self._add_topic(params)
            elif params.operation == "check_similarity":
                return await self._check_similarity(params)
            else:
                return ForbiddenTopicsResult(
                    success=False,
                    operation=params.operation,
                    error=f"Unknown operation: {params.operation}"
                )

        except Exception as e:
            logger.error(f"Forbidden topics tool execution failed: {e}")
            return ForbiddenTopicsResult(
                success=False,
                operation=params.operation,
                error=str(e)
            )

    async def _get_forbidden_topics(self, params: ForbiddenTopicsParams) -> ForbiddenTopicsResult:
        """Get list of forbidden topics for a field."""
        try:
            field_str = params.field.value if params.field else None
            forbidden_topics = self.topic_memory.get_forbidden_topics(field_str, params.days_back)

            titles = [topic.title for topic in forbidden_topics]
            descriptions = [f"{topic.title}: {topic.description}" for topic in forbidden_topics]

            field_desc = f" in {params.field.value}" if params.field else ""
            summary = f"Found {len(forbidden_topics)} forbidden topics{field_desc} from last {params.days_back} days"

            return ForbiddenTopicsResult(
                success=True,
                operation="get_forbidden",
                forbidden_topics=titles,
                forbidden_descriptions=descriptions,
                total_topics_in_memory=self.topic_memory.memory.total_topics,
                summary=summary
            )

        except Exception as e:
            logger.error(f"Failed to get forbidden topics: {e}")
            return ForbiddenTopicsResult(
                success=False,
                operation="get_forbidden",
                error=str(e)
            )

    async def _add_topic(self, params: ForbiddenTopicsParams) -> ForbiddenTopicsResult:
        """Add a new topic to the memory system."""
        try:
            if not params.new_topic:
                return ForbiddenTopicsResult(
                    success=False,
                    operation="add_topic",
                    error="No new_topic provided for add_topic operation"
                )

            success = self.topic_memory.add_topic(params.new_topic)

            if success:
                return ForbiddenTopicsResult(
                    success=True,
                    operation="add_topic",
                    topics_added=1,
                    total_topics_in_memory=self.topic_memory.memory.total_topics,
                    summary=f"Successfully added topic '{params.new_topic.title}' to memory"
                )
            else:
                return ForbiddenTopicsResult(
                    success=False,
                    operation="add_topic",
                    error="Failed to add topic to memory"
                )

        except Exception as e:
            logger.error(f"Failed to add topic: {e}")
            return ForbiddenTopicsResult(
                success=False,
                operation="add_topic",
                error=str(e)
            )

    async def _check_similarity(self, params: ForbiddenTopicsParams) -> ForbiddenTopicsResult:
        """Check similarity of proposed topics against existing ones."""
        try:
            if not params.proposed_topics:
                return ForbiddenTopicsResult(
                    success=False,
                    operation="check_similarity",
                    error="No proposed_topics provided for check_similarity operation"
                )

            similar_topics = self.topic_memory.check_topic_similarity(params.proposed_topics)

            total_similar = sum(len(similar) for similar in similar_topics.values())
            summary = f"Checked {len(params.proposed_topics)} proposed topics, found {total_similar} potential conflicts"

            return ForbiddenTopicsResult(
                success=True,
                operation="check_similarity",
                similar_topics=similar_topics,
                total_topics_in_memory=self.topic_memory.memory.total_topics,
                summary=summary
            )

        except Exception as e:
            logger.error(f"Failed to check similarity: {e}")
            return ForbiddenTopicsResult(
                success=False,
                operation="check_similarity",
                error=str(e)
            )
```

### 3.2 Register the Tool with Editor Agent

Update the editor agent to include the forbidden topics tool. In `libs/common/agents/editor_agent/editor_agent_main.py`, add:

```python
# Add import
from utils.forbidden_topics_tool import ForbiddenTopicsTool

# In the EditorAgent __init__ method, add the tool:
config = AgentConfig(
    system_prompt="Editor agent placeholder - will be updated after initialization",
    temperature=0.5,
    default_model_speed=ModelSpeed.SLOW,
    tools=[
        CollectTopicsTool(task_executor=task_service),
        AssignTopicsTool(task_executor=task_service),
        CollectStoryTool(task_executor=task_service),
        ReviewStoryTool(task_executor=task_service),
        PublishStoryTool(task_executor=task_service),
        ForbiddenTopicsTool(config_service)  # Add forbidden topics tool
    ]
)
```

## 🧪 Step 4: Test Topic Memory System

### 4.1 Create Topic Memory Test Script

Create a test script to verify the topic memory system works:

```python
# Create file: test_topic_memory.py
"""Test script for topic memory system."""

import asyncio
from datetime import datetime

from agents.types import ReporterField
from core.config_service import ConfigService
from utils.forbidden_topics_tool import ForbiddenTopicsTool, ForbiddenTopicsParams
from utils.topic_memory_models import CoveredTopic, TopicStatus


async def test_topic_memory():
    """Test the topic memory system."""
    print("🧠 Topic Memory System Test")
    print("=" * 40)

    config_service = ConfigService()
    forbidden_tool = ForbiddenTopicsTool(config_service)

    # Test 1: Add some sample topics
    print("\n1️⃣ Adding sample topics...")

    sample_topics = [
        CoveredTopic(
            title="OpenAI GPT-5 Release",
            description="Coverage of OpenAI's latest language model and its capabilities",
            field=ReporterField.TECHNOLOGY,
            published_date="2024-01-15",
            story_id="story-tech-001",
            keywords=["openai", "gpt-5", "ai", "language model"]
        ),
        CoveredTopic(
            title="Climate Change Impact on Agriculture",
            description="Analysis of rising temperatures affecting global crop yields",
            field=ReporterField.SCIENCE,
            published_date="2024-01-14",
            story_id="story-sci-001",
            keywords=["climate change", "agriculture", "crops"]
        ),
        CoveredTopic(
            title="Federal Reserve Interest Rate Decision",
            description="Analysis of the Fed's latest monetary policy changes",
            field=ReporterField.ECONOMICS,
            published_date="2024-01-13",
            story_id="story-econ-001",
            keywords=["federal reserve", "interest rates", "monetary policy"]
        )
    ]

    for topic in sample_topics:
        params = ForbiddenTopicsParams(
            operation="add_topic",
            new_topic=topic
        )
        result = await forbidden_tool.execute(params)

        if result.success:
            print(f"   ✅ Added: {topic.title}")
        else:
            print(f"   ❌ Failed to add: {topic.title} - {result.error}")

    # Test 2: Get forbidden topics for technology field
    print("\n2️⃣ Getting forbidden topics for technology field...")

    params = ForbiddenTopicsParams(
        operation="get_forbidden",
        field=ReporterField.TECHNOLOGY,
        days_back=30
    )
    result = await forbidden_tool.execute(params)

    if result.success:
        print(f"   ✅ Found {len(result.forbidden_topics)} forbidden topics")
        for topic in result.forbidden_topics:
            print(f"      - {topic}")
    else:
        print(f"   ❌ Failed: {result.error}")

    # Test 3: Check similarity for proposed topics
    print("\n3️⃣ Checking similarity for proposed topics...")

    proposed_topics = [
        "OpenAI's New AI Model Launch",  # Should be similar to existing
        "Quantum Computing Breakthrough",  # Should be unique
        "Climate Effects on Farming"  # Should be similar to existing
    ]

    params = ForbiddenTopicsParams(
        operation="check_similarity",
        proposed_topics=proposed_topics
    )
    result = await forbidden_tool.execute(params)

    if result.success:
        print(f"   ✅ Similarity check completed")
        for proposed, similar in result.similar_topics.items():
            if similar:
                print(f"      ⚠️  '{proposed}' is similar to: {similar}")
            else:
                print(f"      ✅ '{proposed}' is unique")
    else:
        print(f"   ❌ Failed: {result.error}")

    # Test 4: Get memory statistics
    print("\n4️⃣ Memory statistics...")
    if forbidden_tool.topic_memory:
        stats = forbidden_tool.topic_memory.get_memory_stats()
        print(f"   📊 Total topics: {stats['total_topics']}")
        print(f"   📊 Topics by field: {stats['topics_by_field']}")
        print(f"   📊 Recent topics (7 days): {stats['recent_topics_7_days']}")
        print(f"   📊 Recent topics (30 days): {stats['recent_topics_30_days']}")

    print("\n🎉 Topic memory test completed!")


if __name__ == "__main__":
    asyncio.run(test_topic_memory())
```

### 4.2 Run the Test

```bash
# In DevContainer terminal
python test_topic_memory.py
```

## 🔗 Step 5: Integrate with Editor Workflow

### 5.1 Update Editor System Prompt

The editor agent needs to be aware of its new responsibility. Update the editor's system prompt to include topic memory management:

```python
# In libs/common/agents/editor_agent/editor_prompt.py (or wherever editor prompts are defined)

EDITOR_SYSTEM_PROMPT = """
You are the Editor-in-Chief of BobTimes, an AI-powered newspaper. Your responsibilities include:

1. **Topic Collection & Deduplication**:
   - Use the forbidden_topics tool to get recent topics before assigning new ones
   - Ensure no topic repetition within the last 30 days
   - Check similarity of proposed topics against existing coverage

2. **Story Assignment**:
   - Assign unique, non-duplicate topics to reporter agents
   - Provide forbidden topics list to reporters to avoid duplication
   - Ensure diverse coverage across different fields

3. **Editorial Review**:
   - Review story drafts for quality and accuracy
   - Check that stories don't duplicate recent coverage
   - Provide feedback for improvements

4. **Publication Management**:
   - Publish approved stories
   - Add published topics to memory using forbidden_topics tool
   - Maintain editorial consistency across news cycles

**TOPIC MEMORY WORKFLOW:**
1. Before collecting topics: Use forbidden_topics tool with operation="get_forbidden"
2. When assigning topics: Include forbidden topics list in reporter instructions
3. After publishing stories: Use forbidden_topics tool with operation="add_topic"

**AVAILABLE TOOLS:**
- forbidden_topics: Manage topic memory and prevent duplication
- collect_topics: Get trending topics from reporters
- assign_topics: Assign specific topics to reporters
- collect_story: Get completed stories from reporters
- review_story: Review and provide feedback on stories
- publish_story: Publish approved stories

Always prioritize content uniqueness and editorial quality.
"""
```

### 5.2 Update Reporter Instructions

Modify how the editor communicates with reporters to include forbidden topics. Update the topic assignment process:

```python
# In libs/common/agents/editor_agent/editor_tools.py

# Update the AssignTopicsTool to include forbidden topics
class AssignTopicsParams(BaseModel):
    """Parameters for assigning topics to reporters."""
    field_requests: list[FieldTopicRequest]
    topics_per_field: int = Field(default=3, description="Number of topics per field")
    forbidden_topics: list[str] = Field(default_factory=list, description="Topics to avoid")
    forbidden_descriptions: list[str] = Field(default_factory=list, description="Descriptions of forbidden topics")

# Update the task creation to include forbidden topics
task = ReporterTask(
    name=TaskType.FIND_TOPICS,
    field=request.field,
    sub_section=request.sub_section,
    description=f"Find {params.topics_per_field} trending topics in {request.field.value}{sub_section_desc}",
    guidelines=f"""Focus on current, newsworthy topics with broad appeal.

FORBIDDEN TOPICS (avoid these recent topics):
{chr(10).join(f"- {topic}" for topic in params.forbidden_topics)}

FORBIDDEN TOPIC DESCRIPTIONS:
{chr(10).join(f"- {desc}" for desc in params.forbidden_descriptions)}

Ensure your proposed topics are unique and not similar to the forbidden topics listed above."""
)
```

### 5.3 Create Enhanced Editor Workflow

Create a script that demonstrates the complete workflow:

```python
# Create file: test_editor_workflow.py
"""Test the complete editor workflow with topic memory."""

import asyncio
from datetime import datetime

from agents.agent_factory import AgentFactory
from agents.models.task_models import ReporterTask
from agents.task_execution_service import TaskExecutionService
from agents.types import ReporterField, TaskType
from core.config_service import ConfigService
from utils.forbidden_topics_tool import ForbiddenTopicsTool, ForbiddenTopicsParams
from utils.topic_memory_models import CoveredTopic


async def test_complete_workflow():
    """Test the complete editor workflow with topic memory."""
    print("🗞️ Complete Editor Workflow Test")
    print("=" * 50)

    config_service = ConfigService()
    task_service = TaskExecutionService(config_service)
    factory = AgentFactory(config_service, task_service)

    # Create editor agent
    editor = factory.create_editor()
    forbidden_tool = ForbiddenTopicsTool(config_service)

    # Step 1: Check current forbidden topics
    print("\n1️⃣ Checking current forbidden topics...")

    params = ForbiddenTopicsParams(
        operation="get_forbidden",
        field=ReporterField.TECHNOLOGY,
        days_back=30
    )
    result = await forbidden_tool.execute(params)

    if result.success:
        print(f"   📋 Found {len(result.forbidden_topics)} forbidden topics:")
        for topic in result.forbidden_topics[:3]:  # Show first 3
            print(f"      - {topic}")
    else:
        print(f"   ❌ Failed to get forbidden topics: {result.error}")

    # Step 2: Simulate topic collection with forbidden topics awareness
    print("\n2️⃣ Collecting topics with forbidden topics awareness...")

    # Create a reporter task that includes forbidden topics
    task = ReporterTask(
        name=TaskType.FIND_TOPICS,
        field=ReporterField.TECHNOLOGY,
        description="Find 3 trending topics in technology",
        guidelines=f"""Focus on current, newsworthy topics with broad appeal.

FORBIDDEN TOPICS (avoid these recent topics):
{chr(10).join(f"- {topic}" for topic in result.forbidden_topics)}

Ensure your proposed topics are unique and not similar to the forbidden topics listed above."""
    )

    print(f"   📝 Created task with {len(result.forbidden_topics)} forbidden topics")

    # Step 3: Simulate story publication and memory update
    print("\n3️⃣ Simulating story publication...")

    # Simulate a new story being published
    new_topic = CoveredTopic(
        title="Quantum Computing Breakthrough at IBM",
        description="IBM announces major advancement in quantum error correction",
        field=ReporterField.TECHNOLOGY,
        published_date=datetime.now().strftime("%Y-%m-%d"),
        story_id=f"story-tech-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        keywords=["quantum computing", "ibm", "error correction", "breakthrough"]
    )

    # Add to memory
    add_params = ForbiddenTopicsParams(
        operation="add_topic",
        new_topic=new_topic
    )
    add_result = await forbidden_tool.execute(add_params)

    if add_result.success:
        print(f"   ✅ Added new topic to memory: {new_topic.title}")
        print(f"   📊 Total topics in memory: {add_result.total_topics_in_memory}")
    else:
        print(f"   ❌ Failed to add topic: {add_result.error}")

    # Step 4: Verify the topic is now forbidden
    print("\n4️⃣ Verifying topic is now in forbidden list...")

    verify_params = ForbiddenTopicsParams(
        operation="get_forbidden",
        field=ReporterField.TECHNOLOGY,
        days_back=1  # Just today
    )
    verify_result = await forbidden_tool.execute(verify_params)

    if verify_result.success:
        if new_topic.title in verify_result.forbidden_topics:
            print(f"   ✅ New topic is now in forbidden list")
        else:
            print(f"   ⚠️  New topic not found in forbidden list")
        print(f"   📋 Current forbidden topics: {len(verify_result.forbidden_topics)}")

    # Step 5: Test similarity detection
    print("\n5️⃣ Testing similarity detection...")

    similar_topics = [
        "IBM's Quantum Computing Advance",  # Should be similar
        "New Blockchain Technology",  # Should be unique
        "Quantum Error Correction Progress"  # Should be similar
    ]

    similarity_params = ForbiddenTopicsParams(
        operation="check_similarity",
        proposed_topics=similar_topics
    )
    similarity_result = await forbidden_tool.execute(similarity_params)

    if similarity_result.success:
        print(f"   🔍 Similarity check results:")
        for proposed, similar in similarity_result.similar_topics.items():
            if similar:
                print(f"      ⚠️  '{proposed}' conflicts with: {similar}")
            else:
                print(f"      ✅ '{proposed}' is unique")

    print("\n🎉 Complete workflow test finished!")
    print(f"📊 Final memory stats: {forbidden_tool.topic_memory.get_memory_stats()}")


if __name__ == "__main__":
    asyncio.run(test_complete_workflow())
```

## 🔧 Step 6: Create Topic Memory Management Utilities

### 6.1 Create Topic Memory Manager Script

Create a utility script for managing the topic memory:

```python
# Create file: manage_topic_memory.py
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
    print(f"   Topics by field: {stats['topics_by_field']}")
    print(f"   Topics by status: {stats['topics_by_status']}")


async def list_recent_topics(memory_service: TopicMemoryService):
    """List recent topics."""
    days = input("Enter number of days to look back (default 7): ").strip()
    days = int(days) if days.isdigit() else 7

    recent_topics = memory_service.memory.get_recent_topics(days)

    print(f"\n📋 Topics from last {days} days ({len(recent_topics)} found):")
    for i, topic in enumerate(recent_topics, 1):
        print(f"   {i}. {topic.title}")
        print(f"      Field: {topic.field.value}")
        print(f"      Date: {topic.published_date}")
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
            keyword in topic.description.lower() or
            any(keyword in kw.lower() for kw in topic.keywords)):
            matching_topics.append(topic)

    print(f"\n🔍 Topics matching '{keyword}' ({len(matching_topics)} found):")
    for i, topic in enumerate(matching_topics, 1):
        print(f"   {i}. {topic.title}")
        print(f"      Date: {topic.published_date}")
        print(f"      Keywords: {', '.join(topic.keywords)}")
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

    old_topics = [topic for topic in memory_service.memory.topics if topic.published_date < cutoff_date]

    if not old_topics:
        print(f"✅ No topics older than {days} days found.")
        return

    print(f"⚠️  Found {len(old_topics)} topics older than {days} days:")
    for topic in old_topics[:5]:  # Show first 5
        print(f"   - {topic.title} ({topic.published_date})")

    if len(old_topics) > 5:
        print(f"   ... and {len(old_topics) - 5} more")

    confirm = input(f"\nAre you sure you want to delete {len(old_topics)} topics? (y/N): ").strip().lower()

    if confirm == 'y':
        # Remove old topics
        memory_service.memory.topics = [topic for topic in memory_service.memory.topics if topic.published_date >= cutoff_date]
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

## 🧪 Step 7: Complete Integration Testing

### 7.1 Run All Tests

Execute the test scripts to verify everything works:

```bash
# Test 1: Basic topic memory functionality
python test_topic_memory.py

# Test 2: Complete editor workflow
python test_editor_workflow.py

# Test 3: Interactive memory management
python manage_topic_memory.py
```

### 7.2 Expected Test Results

**Test 1 - Topic Memory System:**
```
🧠 Topic Memory System Test
========================================

1️⃣ Adding sample topics...
   ✅ Added: OpenAI GPT-5 Release
   ✅ Added: Climate Change Impact on Agriculture
   ✅ Added: Federal Reserve Interest Rate Decision

2️⃣ Getting forbidden topics for technology field...
   ✅ Found 1 forbidden topics
      - OpenAI GPT-5 Release

3️⃣ Checking similarity for proposed topics...
   ✅ Similarity check completed
      ⚠️  'OpenAI's New AI Model Launch' is similar to: ['OpenAI GPT-5 Release']
      ✅ 'Quantum Computing Breakthrough' is unique
      ⚠️  'Climate Effects on Farming' is similar to: ['Climate Change Impact on Agriculture']

4️⃣ Memory statistics...
   📊 Total topics: 3
   📊 Topics by field: {'technology': 1, 'science': 1, 'economics': 1}
   📊 Recent topics (7 days): 3
   📊 Recent topics (30 days): 3

🎉 Topic memory test completed!
```

**Test 2 - Complete Workflow:**
```
🗞️ Complete Editor Workflow Test
==================================================

1️⃣ Checking current forbidden topics...
   📋 Found 1 forbidden topics:
      - OpenAI GPT-5 Release

2️⃣ Collecting topics with forbidden topics awareness...
   📝 Created task with 1 forbidden topics

3️⃣ Simulating story publication...
   ✅ Added new topic to memory: Quantum Computing Breakthrough at IBM
   📊 Total topics in memory: 4

4️⃣ Verifying topic is now in forbidden list...
   ✅ New topic is now in forbidden list
   📋 Current forbidden topics: 1

5️⃣ Testing similarity detection...
   🔍 Similarity check results:
      ⚠️  'IBM's Quantum Computing Advance' conflicts with: ['Quantum Computing Breakthrough at IBM']
      ✅ 'New Blockchain Technology' is unique
      ⚠️  'Quantum Error Correction Progress' conflicts with: ['Quantum Computing Breakthrough at IBM']

🎉 Complete workflow test finished!
```

### 7.3 Verify File Structure

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
      "field": "technology",
      "sub_section": null,
      "published_date": "2024-01-15",
      "story_id": "story-tech-001",
      "keywords": ["openai", "gpt-5", "ai", "language model"],
      "status": "published",
      "similarity_threshold": 0.8
    }
  ],
  "total_topics": 1
}
```

## 🎯 Step 8: Integration with Full Newspaper Generation

### 8.1 Update Newspaper Generation Script

Modify your newspaper generation script to use the topic memory system:

```python
# Update your existing newspaper generation script
# Add this to the beginning of the generation process

async def generate_newspaper_with_memory():
    """Generate newspaper with topic memory integration."""
    print("🗞️ Generating Newspaper with Topic Memory")
    print("=" * 50)

    config_service = ConfigService()
    task_service = TaskExecutionService(config_service)
    factory = AgentFactory(config_service, task_service)

    editor = factory.create_editor()
    forbidden_tool = ForbiddenTopicsTool(config_service)

    # Step 1: Get forbidden topics for each field
    fields = [ReporterField.TECHNOLOGY, ReporterField.SCIENCE, ReporterField.ECONOMICS]
    all_forbidden = {}

    for field in fields:
        params = ForbiddenTopicsParams(
            operation="get_forbidden",
            field=field,
            days_back=30
        )
        result = await forbidden_tool.execute(params)

        if result.success:
            all_forbidden[field] = {
                'topics': result.forbidden_topics,
                'descriptions': result.forbidden_descriptions
            }
            print(f"📋 {field.value}: {len(result.forbidden_topics)} forbidden topics")
        else:
            all_forbidden[field] = {'topics': [], 'descriptions': []}
            print(f"⚠️  {field.value}: Failed to get forbidden topics")

    # Step 2: Generate stories with forbidden topics awareness
    # (Your existing story generation logic here, but pass forbidden topics to reporters)

    # Step 3: After publishing each story, add to memory
    # Example for a published story:
    """
    new_topic = CoveredTopic(
        title=story.title,
        description=story.summary,
        field=story.field,
        published_date=datetime.now().strftime("%Y-%m-%d"),
        story_id=story.id,
        keywords=extract_keywords_from_story(story.content)
    )

    add_params = ForbiddenTopicsParams(
        operation="add_topic",
        new_topic=new_topic
    )
    await forbidden_tool.execute(add_params)
    """

    print("✅ Newspaper generation with topic memory completed!")
```

### 8.2 Create Complete Integration Example

```python
# Create file: complete_integration_example.py
"""Complete integration example showing topic memory in action."""

import asyncio
from datetime import datetime

from agents.types import ReporterField
from core.config_service import ConfigService
from utils.forbidden_topics_tool import ForbiddenTopicsTool, ForbiddenTopicsParams
from utils.topic_memory_models import CoveredTopic


async def simulate_news_cycle():
    """Simulate a complete news cycle with topic memory."""
    print("📰 Simulating Complete News Cycle with Topic Memory")
    print("=" * 60)

    config_service = ConfigService()
    forbidden_tool = ForbiddenTopicsTool(config_service)

    # Simulate Day 1: First news cycle
    print("\n📅 DAY 1: First News Cycle")
    print("-" * 30)

    day1_stories = [
        CoveredTopic(
            title="Apple Announces New iPhone Features",
            description="Apple reveals AI-powered camera and battery improvements",
            field=ReporterField.TECHNOLOGY,
            published_date="2024-01-15",
            story_id="story-tech-001",
            keywords=["apple", "iphone", "ai", "camera", "battery"]
        ),
        CoveredTopic(
            title="NASA Mars Mission Update",
            description="Latest findings from Mars rover exploration mission",
            field=ReporterField.SCIENCE,
            published_date="2024-01-15",
            story_id="story-sci-001",
            keywords=["nasa", "mars", "rover", "exploration", "space"]
        )
    ]

    # Publish Day 1 stories
    for story in day1_stories:
        params = ForbiddenTopicsParams(operation="add_topic", new_topic=story)
        result = await forbidden_tool.execute(params)
        print(f"   ✅ Published: {story.title}")

    # Simulate Day 2: Check forbidden topics before new cycle
    print("\n📅 DAY 2: New News Cycle with Memory Check")
    print("-" * 40)

    # Check what topics are now forbidden
    for field in [ReporterField.TECHNOLOGY, ReporterField.SCIENCE]:
        params = ForbiddenTopicsParams(
            operation="get_forbidden",
            field=field,
            days_back=30
        )
        result = await forbidden_tool.execute(params)

        if result.success and result.forbidden_topics:
            print(f"   🚫 {field.value} forbidden topics: {result.forbidden_topics}")
        else:
            print(f"   ✅ {field.value}: No forbidden topics")

    # Propose new topics and check for conflicts
    proposed_topics = [
        "Apple's New iPhone Camera Technology",  # Should conflict
        "Google's Latest AI Breakthrough",      # Should be unique
        "Mars Rover Discovers Water Evidence"   # Should conflict
    ]

    params = ForbiddenTopicsParams(
        operation="check_similarity",
        proposed_topics=proposed_topics
    )
    result = await forbidden_tool.execute(params)

    print(f"\n   🔍 Checking proposed topics for Day 2:")
    for proposed, similar in result.similar_topics.items():
        if similar:
            print(f"      ❌ REJECT: '{proposed}' (similar to: {similar})")
        else:
            print(f"      ✅ APPROVE: '{proposed}' (unique)")

    # Publish only unique topics
    unique_topics = [
        CoveredTopic(
            title="Google's Latest AI Breakthrough",
            description="Google announces new AI model with improved reasoning",
            field=ReporterField.TECHNOLOGY,
            published_date="2024-01-16",
            story_id="story-tech-002",
            keywords=["google", "ai", "breakthrough", "reasoning", "model"]
        )
    ]

    for story in unique_topics:
        params = ForbiddenTopicsParams(operation="add_topic", new_topic=story)
        result = await forbidden_tool.execute(params)
        print(f"   ✅ Published unique story: {story.title}")

    # Show final memory statistics
    print(f"\n📊 Final Memory Statistics:")
    if forbidden_tool.topic_memory:
        stats = forbidden_tool.topic_memory.get_memory_stats()
        print(f"   Total topics in memory: {stats['total_topics']}")
        print(f"   Topics by field: {stats['topics_by_field']}")
        print(f"   Recent topics (2 days): {len(forbidden_tool.topic_memory.memory.get_recent_topics(2))}")

    print("\n🎉 News cycle simulation completed!")
    print("✅ Topic duplication successfully prevented!")


if __name__ == "__main__":
    asyncio.run(simulate_news_cycle())
```

## 🏆 Step 9: Verification and Best Practices

### 9.1 Verification Checklist

Run through this checklist to ensure everything is working:

- [ ] ✅ Topic memory file is created in `data/topics/covered_topics.json`
- [ ] ✅ ForbiddenTopicsTool can add topics to memory
- [ ] ✅ ForbiddenTopicsTool can retrieve forbidden topics by field
- [ ] ✅ ForbiddenTopicsTool can detect similar topics
- [ ] ✅ Editor agent has access to forbidden topics tool
- [ ] ✅ Reporter tasks include forbidden topics in guidelines
- [ ] ✅ Memory persists between application restarts
- [ ] ✅ Topic similarity detection works correctly
- [ ] ✅ Memory management utilities function properly

### 9.2 Best Practices for Topic Memory

**Editorial Guidelines:**
1. **Regular Memory Maintenance**: Use the management utility weekly to clean old topics
2. **Similarity Threshold**: Adjust similarity thresholds based on your content needs
3. **Keyword Strategy**: Use comprehensive keywords for better duplicate detection
4. **Field-Specific Memory**: Maintain separate forbidden lists per field
5. **Memory Backup**: Regularly export memory for backup purposes

**Performance Considerations:**
1. **Memory Size**: Keep memory file under 10MB for optimal performance
2. **Cleanup Schedule**: Remove topics older than 90 days automatically
3. **Similarity Algorithm**: Consider upgrading to more sophisticated similarity detection
4. **Caching**: Implement caching for frequently accessed forbidden topics

**Integration Tips:**
1. **Error Handling**: Always handle memory service failures gracefully
2. **Fallback Strategy**: Have a fallback when topic memory is unavailable
3. **Logging**: Log all topic additions and similarity checks for debugging
4. **Testing**: Test with various topic combinations and edge cases

## 🎯 Lab 3 Summary

Congratulations! You've successfully implemented a comprehensive content improvement system with topic deduplication and editorial memory. Here's what you've accomplished:

### ✅ **Key Features Implemented:**

1. **Topic Memory System**: Persistent storage of covered topics with metadata
2. **Forbidden Topics Tool**: Editor tool for managing topic deduplication
3. **Similarity Detection**: Automatic detection of similar topics to prevent duplication
4. **Editorial Workflow Integration**: Seamless integration with editor-reporter workflow
5. **Memory Management**: Utilities for maintaining and managing topic memory
6. **Field-Specific Filtering**: Separate forbidden topics per news field
7. **Persistent Storage**: File-based storage without database dependencies

### 🔧 **Technical Achievements:**

- **Pydantic Models**: Type-safe data models for topic memory
- **Service Architecture**: Clean separation of concerns with TopicMemoryService
- **Tool Integration**: Proper integration with existing agent tool system
- **Error Handling**: Robust error handling and fallback mechanisms
- **Testing Suite**: Comprehensive tests for all functionality
- **Management Utilities**: Interactive tools for memory management

### 📈 **Editorial Benefits:**

- **Content Uniqueness**: Prevents topic repetition across news cycles
- **Editorial Consistency**: Maintains consistent coverage standards
- **Quality Control**: Improves overall content quality through deduplication
- **Efficiency**: Reduces wasted effort on duplicate content
- **Scalability**: Handles growing topic databases efficiently

Your BobTimes newspaper now has intelligent editorial memory that ensures fresh, unique content in every news cycle!

**🔄 Ready for Advanced Features?** Consider implementing:
- Advanced similarity algorithms (semantic similarity, embeddings)
- Topic trending analysis
- Automated topic suggestion based on gaps in coverage
- Integration with external news APIs for broader topic awareness

🎉 **Lab 3 Complete!** Your editorial system now has the intelligence to maintain content quality and uniqueness across all news cycles.
