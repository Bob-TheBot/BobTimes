# Lab 4: Researcher Agent - Information Discovery & Content Gathering

Welcome to Lab 4! In this lab, you'll create a new Researcher Agent that specializes in discovering topics and gathering comprehensive information. This creates a clear three-agent workflow: Researcher discovers and gathers, Editor decides and assigns, Reporter writes and publishes.

## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Create a dedicated Researcher Agent with specialized tools
- ✅ Implement topic discovery and content gathering workflows
- ✅ Integrate YouTube, DuckDuckGo, and web scraping capabilities
- ✅ Design a new three-agent editorial workflow
- ✅ Build research data models and content aggregation


## 📋 Prerequisites

- ✅ Completed Labs 1-3 (Basic setup, YouTube integration, Topic memory)
- ✅ Working DevContainer or local development environment
- ✅ Understanding of the existing editor-reporter workflow
- ✅ Familiarity with the tool architecture and agent system

## 🏗️ Step 1: Design the New Agent Architecture

### 1.1 Three-Agent Workflow Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Researcher Agent│    │  Editor Agent   │    │ Reporter Agent  │
│                 │    │                 │    │                 │
│ • Topic Discovery│───▶│ • Editorial     │───▶│ • Story Writing │
│ • Content Gather│    │   Decisions     │    │ • Content       │
│ • Source Aggreg │    │ • Topic Memory  │    │   Creation      │
│ • Data Research │    │ • Assignment    │    │ • Publishing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Research      │    │   Editorial     │    │   Published     │
│   Database      │    │   Decisions     │    │   Stories       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1.2 Researcher Agent Responsibilities

**Primary Functions:**
- **Topic Discovery**: Find trending topics across multiple sources
- **Content Gathering**: Collect detailed information about topics
- **Source Aggregation**: Compile sources from YouTube, web search, and scraping
- **Research Synthesis**: Provide comprehensive research packages to Editor

**Tools Available:**
- **YouTube Research Tool**: Channel monitoring and video analysis
- **DuckDuckGo Search Tool**: Web search for trending topics
- **Web Scraper Tool**: Deep content extraction from web sources
- **Research Aggregator**: Combine and synthesize research from multiple sources

### 1.3 Updated Workflow Process

1. **Research Phase**: Researcher Agent discovers topics and gathers information
2. **Editorial Phase**: Editor Agent reviews research, checks topic memory, makes decisions
3. **Assignment Phase**: Editor assigns selected topics with research data to Reporter
4. **Writing Phase**: Reporter Agent writes stories based on provided research
5. **Publication Phase**: Editor publishes stories and updates topic memory

## 🔧 Step 2: Create Researcher Agent Models

### 2.1 Research Data Models

Create the file `libs/common/agents/researcher_agent/research_models.py`:

```python
"""Research models for the Researcher Agent."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from agents.models.story_models import StorySource
from agents.types import ReporterField
from pydantic import BaseModel, Field


class ResearchSourceType(StrEnum):
    """Types of research sources."""
    YOUTUBE_VIDEO = "youtube_video"
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLE = "news_article"


class ResearchSource(BaseModel):
    """Extended source model for research data."""
    url: str = Field(description="Source URL")
    title: str = Field(description="Source title")
    summary: str = Field(description="Source summary or excerpt")
    source_type: ResearchSourceType = Field(description="Type of research source")
    content: Optional[str] = Field(None, description="Full content if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score (0-1)")
    accessed_at: datetime = Field(default_factory=datetime.now, description="When source was accessed")

    def to_story_source(self) -> StorySource:
        """Convert to StorySource for compatibility."""
        return StorySource(
            url=self.url,
            title=self.title,
            summary=self.summary,
            accessed_at=self.accessed_at
        )


class TopicResearch(BaseModel):
    """Comprehensive research data for a topic."""
    topic_title: str = Field(description="Main topic title")
    topic_description: str = Field(description="Detailed topic description")
    field: ReporterField = Field(description="News field category")
    keywords: list[str] = Field(default_factory=list, description="Related keywords")
    sources: list[ResearchSource] = Field(default_factory=list, description="Research sources")
    content_summary: str = Field(description="Summary of all gathered content")
    key_points: list[str] = Field(default_factory=list, description="Key points from research")
    trending_score: float = Field(default=0.0, description="Trending score (0-1)")
    research_depth: int = Field(default=0, description="Number of sources researched")
    research_timestamp: datetime = Field(default_factory=datetime.now, description="When research was conducted")

    def get_sources_by_type(self, source_type: ResearchSourceType) -> list[ResearchSource]:
        """Get sources filtered by type."""
        return [source for source in self.sources if source.source_type == source_type]

    def get_total_content_length(self) -> int:
        """Get total length of all content."""
        return sum(len(source.content or "") for source in self.sources)


class ResearchRequest(BaseModel):
    """Request for research on specific topics or fields."""
    fields: list[ReporterField] = Field(description="Fields to research")
    topics_per_field: int = Field(default=5, description="Number of topics to research per field")
    research_depth: int = Field(default=3, description="Number of sources per topic")
    include_youtube: bool = Field(default=True, description="Include YouTube research")
    include_web_search: bool = Field(default=True, description="Include web search")
    include_scraping: bool = Field(default=True, description="Include web scraping")
    max_content_length: int = Field(default=5000, description="Max content length per source")
    trending_only: bool = Field(default=True, description="Focus on trending topics only")


class ResearchResult(BaseModel):
    """Result from research operations."""
    success: bool = Field(description="Whether research was successful")
    topics_researched: list[TopicResearch] = Field(default_factory=list, description="Researched topics")
    total_sources: int = Field(default=0, description="Total sources gathered")
    research_summary: str = Field(description="Summary of research findings")
    fields_covered: list[ReporterField] = Field(default_factory=list, description="Fields that were researched")
    research_duration: float = Field(default=0.0, description="Research duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

    def get_topics_by_field(self, field: ReporterField) -> list[TopicResearch]:
        """Get topics filtered by field."""
        return [topic for topic in self.topics_researched if topic.field == field]

    def get_top_trending_topics(self, limit: int = 10) -> list[TopicResearch]:
        """Get top trending topics across all fields."""
        return sorted(self.topics_researched, key=lambda t: t.trending_score, reverse=True)[:limit]


class EditorialReviewRequest(BaseModel):
    """Request for editorial review of research."""
    research_result: ResearchResult = Field(description="Research result to review")
    forbidden_topics: list[str] = Field(default_factory=list, description="Topics to avoid")
    priority_fields: list[ReporterField] = Field(default_factory=list, description="Priority fields")
    max_topics_to_select: int = Field(default=10, description="Maximum topics to select")


class EditorialDecision(BaseModel):
    """Editorial decision on researched topics."""
    selected_topics: list[TopicResearch] = Field(default_factory=list, description="Topics selected for writing")
    rejected_topics: list[str] = Field(default_factory=list, description="Topics rejected with reasons")
    editorial_notes: dict[str, str] = Field(default_factory=dict, description="Notes for each selected topic")
    assignment_priority: dict[str, int] = Field(default_factory=dict, description="Priority ranking for assignments")
    decision_timestamp: datetime = Field(default_factory=datetime.now, description="When decision was made")


class StoryAssignment(BaseModel):
    """Assignment of researched topic to reporter."""
    topic_research: TopicResearch = Field(description="Research data for the topic")
    reporter_field: ReporterField = Field(description="Reporter field assignment")
    assignment_notes: str = Field(description="Special instructions for the reporter")
    priority_level: int = Field(default=1, description="Priority level (1-5)")
    deadline: Optional[datetime] = Field(None, description="Story deadline")
    required_word_count: int = Field(default=500, description="Target word count")
    editorial_guidelines: str = Field(description="Specific editorial guidelines")
```

This establishes the foundation for our research-focused data models. In the next steps, we'll create the Researcher Agent itself and its specialized tools.

## 🔄 Next Steps Preview

In the remaining steps, we will:
- Create the Researcher Agent with specialized research tools
- Implement multi-source research aggregation
- Update the Editor Agent to work with research data
- Modify the Reporter Agent to write from research

- Create research management utilities

Ready to build the intelligent research system!


To align with the finalized implementation:

- Reporter Agent toolset is restricted to memory-only operations: `fetch_from_memory` and `use_llm`.
- Reporter does not have research tools (no search, scrape, or YouTube).
- All memory access goes through our own `SharedMemoryStore`.
- Intelligent topic matching is built into `fetch_from_memory`.

### Shared Memory Access (project-native)

```python
# Access the global shared memory store
from libs.common.agents.shared_memory_store import get_shared_memory_store
store = get_shared_memory_store()
store.list_topics("technology")  # e.g., ["openai-o4o-launch", "gemini-2.5-updates", ...]
```

### Enhanced fetch_from_memory parameters

```python
# Pydantic params (conceptual snippet)
class FetchFromMemoryParams(BaseModel):
    field: str
    topic_key: str | None = None
    topic_query: str | None = None  # used for intelligent matching when key is unknown
```

### Usage examples

```python
# 1) Unknown exact key → use topic_query for intelligent matching
params = FetchFromMemoryParams(field="technology", topic_query="viral AI content creation tools")
result = await fetch_tool.execute(params)
# result.topic_key is the resolved memory key; result.content_summary contains the summary

# 2) Known exact key → pass topic_key directly
params = FetchFromMemoryParams(field="technology", topic_key="openai-o4o-launch")
result = await fetch_tool.execute(params)
```

### Guidelines

- Prefer `topic_key` when you know the exact memory key set by the Researcher/Editor.
- Use `topic_query` when only a human-friendly assignment string is available; the tool will pick the best match.
- On low-confidence matches, the tool surfaces suggestions; align future assignments to consistent memory keys to improve reliability.



## 🛠️ Step 3: Create Researcher Agent Tools

### 3.1 Multi-Source Research Tool

Create the file `libs/common/agents/researcher_agent/research_tools.py`:

```python
"""Research tools for the Researcher Agent."""

import asyncio
from datetime import datetime
from typing import Any

from agents.tools.base_tool import BaseTool
from core.config_service import ConfigService
from core.llm_service import LLMService, ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

from .research_models import (
    ResearchRequest, ResearchResult, TopicResearch, ResearchSource,
    ResearchSourceType, ReporterField
)

logger = get_logger(__name__)


class MultiSourceResearchTool(BaseTool):
    """Tool for conducting multi-source research across YouTube, web search, and scraping."""

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize multi-source research tool."""
        name = "multi_source_research"
        description = f"""
Conduct comprehensive research across multiple sources to discover trending topics and gather detailed information.

PARAMETER SCHEMA:
{ResearchRequest.model_json_schema()}

CORRECT USAGE EXAMPLES:
# Research trending topics across all fields
{{"fields": ["technology", "science", "economics"], "topics_per_field": 5, "research_depth": 3}}

# Deep research on technology topics only
{{"fields": ["technology"], "topics_per_field": 3, "research_depth": 5, "include_youtube": true, "include_web_search": true, "include_scraping": true}}

# Quick trending research without scraping
{{"fields": ["science"], "topics_per_field": 2, "research_depth": 2, "include_scraping": false}}

RESEARCH SOURCES:
- YouTube: Video content, transcripts, trending topics from channels
- Web Search: DuckDuckGo search results for trending topics
- Web Scraping: Deep content extraction from relevant websites

RESEARCH PROCESS:
1. Discover trending topics in specified fields
2. Gather content from multiple sources per topic
3. Analyze and score topic relevance and trending status
4. Compile comprehensive research packages
5. Provide content summaries and key points

RETURNS:
- topics_researched: List of TopicResearch objects with full data
- total_sources: Total number of sources gathered
- research_summary: Overview of research findings
- fields_covered: Fields that were successfully researched

This tool provides comprehensive research data for editorial decision-making.
"""
        super().__init__(name=name, description=description)
        self.params_model = ResearchRequest
        self.config_service = config_service or ConfigService()
        self.llm_service = LLMService(self.config_service)

        # Initialize source-specific tools
        self._init_source_tools()

    def _init_source_tools(self):
        """Initialize tools for different research sources."""
        try:
            # Import existing tools
            from utils.youtube_tool import YouTubeReporterTool
            from agents.reporter_agent.reporter_tools import ReporterSearchTool, ReporterScraperTool

            self.youtube_tool = YouTubeReporterTool(self.config_service)
            self.search_tool = ReporterSearchTool()
            self.scraper_tool = ReporterScraperTool()

            logger.info("Research tools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize research tools: {e}")
            self.youtube_tool = None
            self.search_tool = None
            self.scraper_tool = None

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> ResearchResult:
        """Execute multi-source research."""
        if not isinstance(params, ResearchRequest):
            return ResearchResult(
                success=False,
                research_summary="Invalid parameters",
                error=f"Expected ResearchRequest, got {type(params)}"
            )

        start_time = datetime.now()

        try:
            logger.info(f"Starting research for {len(params.fields)} fields")

            all_topics = []
            total_sources = 0
            fields_covered = []

            # Research each field
            for field in params.fields:
                try:
                    field_topics = await self._research_field(field, params)
                    all_topics.extend(field_topics)
                    total_sources += sum(len(topic.sources) for topic in field_topics)
                    fields_covered.append(field)

                    logger.info(f"Researched {len(field_topics)} topics for {field.value}")
                except Exception as e:
                    logger.error(f"Failed to research field {field.value}: {e}")
                    continue

            # Calculate research duration
            duration = (datetime.now() - start_time).total_seconds()

            # Generate research summary
            summary = await self._generate_research_summary(all_topics, total_sources, duration)

            return ResearchResult(
                success=True,
                topics_researched=all_topics,
                total_sources=total_sources,
                research_summary=summary,
                fields_covered=fields_covered,
                research_duration=duration
            )

        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            return ResearchResult(
                success=False,
                research_summary="Research failed",
                error=str(e)
            )

    async def _research_field(self, field: ReporterField, params: ResearchRequest) -> list[TopicResearch]:
        """Research topics for a specific field."""
        topics = []

        # Step 1: Discover trending topics
        trending_topics = await self._discover_trending_topics(field, params)

        # Step 2: Research each topic in depth
        for topic_title in trending_topics[:params.topics_per_field]:
            try:
                topic_research = await self._research_topic_in_depth(
                    topic_title, field, params
                )
                if topic_research:
                    topics.append(topic_research)
            except Exception as e:
                logger.error(f"Failed to research topic '{topic_title}': {e}")
                continue

        return topics

    async def _discover_trending_topics(self, field: ReporterField, params: ResearchRequest) -> list[str]:
        """Discover trending topics for a field using multiple sources."""
        trending_topics = []

        # YouTube topic discovery
        if params.include_youtube and self.youtube_tool:
            try:
                youtube_topics = await self._get_youtube_trending_topics(field)
                trending_topics.extend(youtube_topics)
            except Exception as e:
                logger.error(f"YouTube topic discovery failed: {e}")

        # Web search topic discovery
        if params.include_web_search and self.search_tool:
            try:
                search_topics = await self._get_search_trending_topics(field)
                trending_topics.extend(search_topics)
            except Exception as e:
                logger.error(f"Search topic discovery failed: {e}")

        # Remove duplicates and return top topics
        unique_topics = list(dict.fromkeys(trending_topics))  # Preserve order
        return unique_topics[:params.topics_per_field * 2]  # Get extra for filtering

    async def _get_youtube_trending_topics(self, field: ReporterField) -> list[str]:
        """Get trending topics from YouTube channels."""
        if not self.youtube_tool:
            return []

        # Get field-specific channel IDs (you'll need to configure these)
        channel_ids = self._get_field_channels(field)

        from utils.youtube_tool import YouTubeToolParams

        params = YouTubeToolParams(
            channel_ids=channel_ids,
            operation="topics",
            days_back=7,
            max_videos_per_channel=5
        )

        result = await self.youtube_tool.execute(params)

        if result.success:
            return result.topics_extracted
        else:
            logger.error(f"YouTube research failed: {result.error}")
            return []

    async def _get_search_trending_topics(self, field: ReporterField) -> list[str]:
        """Get trending topics from web search."""
        if not self.search_tool:
            return []

        # Create field-specific search queries
        search_queries = self._get_field_search_queries(field)
        topics = []

        from agents.reporter_agent.reporter_tools import ReporterSearchParams

        for query in search_queries:
            try:
                params = ReporterSearchParams(
                    query=query,
                    max_results=10
                )

                result = await self.search_tool.execute(params)

                if result.success and result.search_results:
                    # Extract topics from search result titles
                    for search_result in result.search_results:
                        topics.append(search_result.title)

            except Exception as e:
                logger.error(f"Search query '{query}' failed: {e}")
                continue

        return topics

    async def _research_topic_in_depth(self, topic_title: str, field: ReporterField, params: ResearchRequest) -> TopicResearch | None:
        """Research a specific topic in depth across multiple sources."""
        try:
            sources = []

            # YouTube research
            if params.include_youtube:
                youtube_sources = await self._research_topic_youtube(topic_title, field)
                sources.extend(youtube_sources)

            # Web search research
            if params.include_web_search:
                search_sources = await self._research_topic_search(topic_title, field)
                sources.extend(search_sources)

            # Web scraping research
            if params.include_scraping and len(sources) > 0:
                scraping_sources = await self._research_topic_scraping(sources[:3])  # Scrape top 3
                sources.extend(scraping_sources)

            # Limit sources based on research depth
            sources = sources[:params.research_depth]

            if not sources:
                return None

            # Generate topic research summary
            topic_research = await self._compile_topic_research(
                topic_title, field, sources, params
            )

            return topic_research

        except Exception as e:
            logger.error(f"In-depth research for '{topic_title}' failed: {e}")
            return None

    def _get_field_channels(self, field: ReporterField) -> list[str]:
        """Get YouTube channel IDs for a specific field."""
        # This should be configurable - for now, return sample channels
        channel_map = {
            ReporterField.TECHNOLOGY: [
                "UC_x5XG1OV2P6uZZ5FSM9Ttw",  # Google Developers
                "UCXuqSBlHAE6Xw-yeJA0Tunw",  # Linus Tech Tips
                "UC4QZ_LsYcvcq7qOsOhpAX4A"   # CodeBullet
            ],
            ReporterField.SCIENCE: [
                "UCsXVk37bltHxD1rDPwtNM8Q",  # Kurzgesagt
                "UC6nSFpj9HTCZ5t-N3Rm3-HA",  # Vsauce
                "UCHnyfMqiRRG1u-2MsSQLbXA"   # Veritasium
            ],
            ReporterField.ECONOMICS: [
                "UCZ4AMrDcNrfy3X6nsU8-rPg",  # Economics Explained
                "UCGy6uV7yqGWDeUWTZzT3ZEg"   # Ben Felix
            ]
        }

        return channel_map.get(field, [])

    def _get_field_search_queries(self, field: ReporterField) -> list[str]:
        """Get search queries for discovering trending topics in a field."""
        query_map = {
            ReporterField.TECHNOLOGY: [
                "latest technology news 2024",
                "AI breakthrough news",
                "tech industry updates",
                "software development trends"
            ],
            ReporterField.SCIENCE: [
                "scientific discoveries 2024",
                "research breakthrough news",
                "space exploration updates",
                "medical research news"
            ],
            ReporterField.ECONOMICS: [
                "economic news today",
                "market analysis 2024",
                "financial industry updates",
                "economic policy changes"
            ]
        }

        return query_map.get(field, [f"{field.value} news today"])
```

### 3.2 Research Compilation Methods

Continue the MultiSourceResearchTool with compilation methods:

```python
    async def _research_topic_youtube(self, topic: str, field: ReporterField) -> list[ResearchSource]:
        """Research a topic using YouTube sources."""
        if not self.youtube_tool:
            return []

        try:
            from utils.youtube_tool import YouTubeToolParams

            # Search for videos related to the topic
            channel_ids = self._get_field_channels(field)

            params = YouTubeToolParams(
                channel_ids=channel_ids,
                operation="topics",
                days_back=14,
                max_videos_per_channel=3
            )

            result = await self.youtube_tool.execute(params)

            if not result.success:
                return []

            sources = []
            for detail in result.detailed_results:
                if topic.lower() in detail.get("title", "").lower():
                    source = ResearchSource(
                        url=detail.get("url", ""),
                        title=detail.get("title", ""),
                        summary=f"YouTube video: {detail.get('description', '')[:200]}...",
                        source_type=ResearchSourceType.YOUTUBE_VIDEO,
                        metadata={
                            "channel": detail.get("channel", ""),
                            "published": detail.get("published", ""),
                            "views": detail.get("views", 0)
                        },
                        relevance_score=0.8  # High relevance for matching topics
                    )
                    sources.append(source)

            return sources

        except Exception as e:
            logger.error(f"YouTube research for '{topic}' failed: {e}")
            return []

    async def _research_topic_search(self, topic: str, field: ReporterField) -> list[ResearchSource]:
        """Research a topic using web search."""
        if not self.search_tool:
            return []

        try:
            from agents.reporter_agent.reporter_tools import ReporterSearchParams

            params = ReporterSearchParams(
                query=f"{topic} {field.value} news",
                max_results=5
            )

            result = await self.search_tool.execute(params)

            if not result.success:
                return []

            sources = []
            for search_result in result.search_results:
                source = ResearchSource(
                    url=search_result.url,
                    title=search_result.title,
                    summary=search_result.snippet,
                    source_type=ResearchSourceType.WEB_SEARCH,
                    metadata={
                        "search_query": params.query,
                        "rank": len(sources) + 1
                    },
                    relevance_score=0.7  # Good relevance for search results
                )
                sources.append(source)

            return sources

        except Exception as e:
            logger.error(f"Search research for '{topic}' failed: {e}")
            return []

    async def _research_topic_scraping(self, sources: list[ResearchSource]) -> list[ResearchSource]:
        """Research topics by scraping content from existing sources."""
        if not self.scraper_tool:
            return []

        scraped_sources = []

        for source in sources:
            try:
                from agents.reporter_agent.reporter_tools import ReporterScraperParams

                params = ReporterScraperParams(
                    url=source.url,
                    max_content_length=2000
                )

                result = await self.scraper_tool.execute(params)

                if result.success and result.scraped_content:
                    scraped_source = ResearchSource(
                        url=source.url,
                        title=f"Scraped: {source.title}",
                        summary=result.scraped_content[:300] + "...",
                        source_type=ResearchSourceType.WEB_SCRAPE,
                        content=result.scraped_content,
                        metadata={
                            "original_source_type": source.source_type.value,
                            "content_length": len(result.scraped_content)
                        },
                        relevance_score=0.9  # High relevance for scraped content
                    )
                    scraped_sources.append(scraped_source)

            except Exception as e:
                logger.error(f"Scraping failed for {source.url}: {e}")
                continue

        return scraped_sources

    async def _compile_topic_research(self, topic_title: str, field: ReporterField, sources: list[ResearchSource], params: ResearchRequest) -> TopicResearch:
        """Compile comprehensive research data for a topic."""
        try:
            # Generate topic description using LLM
            topic_description = await self._generate_topic_description(topic_title, sources)

            # Extract keywords
            keywords = await self._extract_keywords(topic_title, sources)

            # Generate content summary
            content_summary = await self._generate_content_summary(sources)

            # Extract key points
            key_points = await self._extract_key_points(sources)

            # Calculate trending score
            trending_score = self._calculate_trending_score(sources)

            return TopicResearch(
                topic_title=topic_title,
                topic_description=topic_description,
                field=field,
                keywords=keywords,
                sources=sources,
                content_summary=content_summary,
                key_points=key_points,
                trending_score=trending_score,
                research_depth=len(sources)
            )

        except Exception as e:
            logger.error(f"Failed to compile research for '{topic_title}': {e}")
            # Return basic research object
            return TopicResearch(
                topic_title=topic_title,
                topic_description=f"Research on {topic_title}",
                field=field,
                sources=sources,
                content_summary="Research compilation failed",
                key_points=[],
                trending_score=0.5,
                research_depth=len(sources)
            )

    async def _generate_topic_description(self, topic_title: str, sources: list[ResearchSource]) -> str:
        """Generate a comprehensive topic description using LLM."""
        try:
            source_summaries = "\n".join([f"- {source.summary}" for source in sources[:3]])

            prompt = f"""Based on the following research sources, write a comprehensive 2-3 sentence description of the topic "{topic_title}":

Research Sources:
{source_summaries}

Provide a clear, informative description that captures the essence of this topic and why it's newsworthy."""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=200
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate topic description: {e}")
            return f"Research and analysis of {topic_title}"

    async def _extract_keywords(self, topic_title: str, sources: list[ResearchSource]) -> list[str]:
        """Extract relevant keywords from sources."""
        try:
            # Simple keyword extraction - can be enhanced with NLP
            keywords = set()

            # Add words from topic title
            keywords.update(word.lower() for word in topic_title.split() if len(word) > 3)

            # Add words from source titles and summaries
            for source in sources:
                title_words = [word.lower() for word in source.title.split() if len(word) > 3]
                summary_words = [word.lower() for word in source.summary.split() if len(word) > 3]
                keywords.update(title_words[:3])  # Top 3 from title
                keywords.update(summary_words[:2])  # Top 2 from summary

            return list(keywords)[:10]  # Return top 10 keywords

        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return [topic_title.lower()]

    async def _generate_content_summary(self, sources: list[ResearchSource]) -> str:
        """Generate a summary of all content from sources."""
        try:
            all_content = []
            for source in sources:
                content = source.content or source.summary
                all_content.append(f"{source.title}: {content[:200]}...")

            combined_content = "\n\n".join(all_content)

            prompt = f"""Summarize the following research content in 3-4 sentences, focusing on the key information and main themes:

{combined_content}

Provide a comprehensive summary that captures the main points and significance."""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=300
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate content summary: {e}")
            return f"Summary of {len(sources)} research sources"

    async def _extract_key_points(self, sources: list[ResearchSource]) -> list[str]:
        """Extract key points from research sources."""
        try:
            source_content = []
            for source in sources[:3]:  # Use top 3 sources
                content = source.content or source.summary
                source_content.append(f"{source.title}: {content}")

            combined_content = "\n\n".join(source_content)

            prompt = f"""Extract 3-5 key points from the following research content. Each point should be a concise, factual statement:

{combined_content}

Format as a simple list of key points."""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=200
            )

            # Parse response into list
            key_points = [point.strip().lstrip('- ').lstrip('• ') for point in response.split('\n') if point.strip()]
            return key_points[:5]  # Return max 5 points

        except Exception as e:
            logger.error(f"Failed to extract key points: {e}")
            return ["Key research findings available"]

    def _calculate_trending_score(self, sources: list[ResearchSource]) -> float:
        """Calculate trending score based on source characteristics."""
        try:
            if not sources:
                return 0.0

            score = 0.0

            # Base score from number of sources
            score += min(len(sources) * 0.1, 0.5)

            # Score from source types (YouTube and recent content scores higher)
            for source in sources:
                if source.source_type == ResearchSourceType.YOUTUBE_VIDEO:
                    score += 0.2
                elif source.source_type == ResearchSourceType.WEB_SCRAPE:
                    score += 0.15
                else:
                    score += 0.1

            # Score from recency (sources from last 7 days get bonus)
            recent_sources = [s for s in sources if (datetime.now() - s.accessed_at).days <= 7]
            score += len(recent_sources) * 0.05

            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Failed to calculate trending score: {e}")
            return 0.5

    async def _generate_research_summary(self, topics: list[TopicResearch], total_sources: int, duration: float) -> str:
        """Generate overall research summary."""
        try:
            field_counts = {}
            for topic in topics:
                field_counts[topic.field.value] = field_counts.get(topic.field.value, 0) + 1

            top_trending = sorted(topics, key=lambda t: t.trending_score, reverse=True)[:3]

            summary = f"""Research completed in {duration:.1f} seconds.

Found {len(topics)} trending topics across {len(field_counts)} fields.
Gathered {total_sources} sources total.

Field breakdown: {', '.join(f'{field}: {count}' for field, count in field_counts.items())}

Top trending topics:
{chr(10).join(f'- {topic.topic_title} (score: {topic.trending_score:.2f})' for topic in top_trending)}"""

            return summary

        except Exception as e:
            logger.error(f"Failed to generate research summary: {e}")
            return f"Research completed: {len(topics)} topics, {total_sources} sources"
```

## 🤖 Step 4: Create the Researcher Agent

### 4.1 Create Researcher Agent Main Class

Create the file `libs/common/agents/researcher_agent/researcher_agent_main.py`:

```python
"""Main Researcher Agent implementation."""

from agents.agent_config import AgentConfig
from agents.base_agent import BaseAgent
from agents.types import AgentType
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger

from .research_tools import MultiSourceResearchTool

logger = get_logger(__name__)


class ResearcherAgent(BaseAgent):
    """Researcher Agent specialized in topic discovery and content gathering."""

    def __init__(self, config_service: ConfigService):
        """Initialize the Researcher Agent."""

        # Create agent configuration
        config = AgentConfig(
            system_prompt=self._get_system_prompt(),
            temperature=0.3,  # Lower temperature for more focused research
            default_model_speed=ModelSpeed.FAST,  # Fast for research tasks
            tools=[
                MultiSourceResearchTool(config_service)
            ]
        )

        super().__init__(
            agent_type=AgentType.RESEARCHER,
            config=config,
            config_service=config_service
        )

        logger.info("Researcher Agent initialized")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Researcher Agent."""
        return """
You are the Researcher Agent for BobTimes, an AI-powered newspaper. Your primary responsibility is discovering trending topics and gathering comprehensive research data for editorial decision-making.

## CORE RESPONSIBILITIES:

### 1. Topic Discovery
- Identify trending topics across technology, science, and economics
- Monitor multiple sources: YouTube channels, web search, news sites
- Focus on current, newsworthy topics with broad appeal
- Prioritize topics with high engagement and relevance

### 2. Content Gathering
- Collect detailed information from multiple sources per topic
- Gather YouTube video content, transcripts, and metadata
- Perform web searches for comprehensive coverage
- Scrape relevant websites for in-depth content
- Compile sources with proper attribution and metadata

### 3. Research Analysis
- Analyze topic trending scores and relevance
- Extract key points and insights from gathered content
- Generate comprehensive topic summaries
- Identify keywords and themes
- Assess newsworthiness and editorial value

### 4. Research Packaging
- Compile research into structured TopicResearch objects
- Provide content summaries and key points
- Include source attribution and metadata
- Calculate trending scores and relevance metrics
- Prepare research packages for editorial review

## RESEARCH WORKFLOW:

1. **Discovery Phase**: Use multi_source_research tool to find trending topics
2. **Gathering Phase**: Collect content from YouTube, search, and scraping
3. **Analysis Phase**: Analyze content for key insights and relevance
4. **Compilation Phase**: Package research into structured format
5. **Delivery Phase**: Provide comprehensive research to Editor Agent

## RESEARCH QUALITY STANDARDS:

- **Accuracy**: Verify information across multiple sources
- **Relevance**: Focus on topics with broad audience appeal
- **Timeliness**: Prioritize recent and trending content
- **Depth**: Gather sufficient detail for informed editorial decisions
- **Attribution**: Properly cite all sources and maintain metadata

## AVAILABLE TOOLS:

- **multi_source_research**: Comprehensive research across YouTube, web search, and scraping

## COMMUNICATION STYLE:

- Provide clear, structured research summaries
- Include quantitative metrics (trending scores, source counts)
- Highlight key insights and editorial opportunities
- Maintain objective, factual tone
- Focus on actionable research findings

Your research directly impacts editorial decisions and story quality. Prioritize thoroughness, accuracy, and editorial value in all research activities.
"""

    async def research_trending_topics(self, fields: list[str], topics_per_field: int = 5) -> dict:
        """Research trending topics across specified fields."""
        try:
            from .research_models import ResearchRequest, ReporterField

            # Convert string fields to enum
            field_enums = []
            for field_str in fields:
                try:
                    field_enums.append(ReporterField(field_str))
                except ValueError:
                    logger.warning(f"Invalid field: {field_str}")
                    continue

            if not field_enums:
                return {
                    "success": False,
                    "error": "No valid fields provided"
                }

            # Create research request
            request = ResearchRequest(
                fields=field_enums,
                topics_per_field=topics_per_field,
                research_depth=3,
                include_youtube=True,
                include_web_search=True,
                include_scraping=True
            )

            # Execute research
            result = await self.tools[0].execute(request)

            if result.success:
                return {
                    "success": True,
                    "topics_researched": [topic.model_dump() for topic in result.topics_researched],
                    "total_sources": result.total_sources,
                    "research_summary": result.research_summary,
                    "fields_covered": [field.value for field in result.fields_covered],
                    "research_duration": result.research_duration
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }

        except Exception as e:
            logger.error(f"Research trending topics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def deep_research_topic(self, topic_title: str, field: str) -> dict:
        """Conduct deep research on a specific topic."""
        try:
            from .research_models import ResearchRequest, ReporterField

            field_enum = ReporterField(field)

            # Create focused research request
            request = ResearchRequest(
                fields=[field_enum],
                topics_per_field=1,
                research_depth=5,  # Deep research
                include_youtube=True,
                include_web_search=True,
                include_scraping=True,
                trending_only=False  # Include all relevant content
            )

            # Execute research
            result = await self.tools[0].execute(request)

            if result.success and result.topics_researched:
                # Find the most relevant topic
                best_match = None
                best_score = 0

                for topic in result.topics_researched:
                    # Simple relevance scoring based on title similarity
                    score = self._calculate_title_similarity(topic_title, topic.topic_title)
                    if score > best_score:
                        best_score = score
                        best_match = topic

                if best_match:
                    return {
                        "success": True,
                        "topic_research": best_match.model_dump(),
                        "relevance_score": best_score
                    }

            return {
                "success": False,
                "error": "No relevant research found for the topic"
            }

        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
```

### 4.2 Create Researcher Agent Factory Integration

Update the agent factory to include the Researcher Agent. Add to `libs/common/agents/agent_factory.py`:

```python
# Add import
from agents.researcher_agent.researcher_agent_main import ResearcherAgent

# Add to AgentFactory class
def create_researcher(self) -> ResearcherAgent:
    """Create a Researcher Agent instance."""
    return ResearcherAgent(self.config_service)
```

### 4.3 Update Agent Types

Add the researcher agent type to `libs/common/agents/types.py`:

```python
class AgentType(StrEnum):
    """Types of agents in the system."""
    EDITOR = "editor"
    REPORTER = "reporter"
    RESEARCHER = "researcher"  # Add this line
```

## 🔄 Step 5: Update Editor Agent for Three-Agent Workflow

### 5.1 Create Editorial Research Review Tool

Create the file `libs/common/agents/editor_agent/editorial_research_tool.py`:

```python
"""Editorial research review tool for the Editor Agent."""

from datetime import datetime
from typing import Any

from agents.tools.base_tool import BaseTool
from core.config_service import ConfigService
from core.llm_service import LLMService, ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

from agents.researcher_agent.research_models import (
    EditorialReviewRequest, EditorialDecision, TopicResearch, StoryAssignment
)
from utils.forbidden_topics_tool import ForbiddenTopicsTool, ForbiddenTopicsParams

logger = get_logger(__name__)


class EditorialResearchReviewTool(BaseTool):
    """Tool for Editor Agent to review research and make editorial decisions."""

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize editorial research review tool."""
        name = "editorial_research_review"
        description = f"""
Review research findings from Researcher Agent and make editorial decisions about which topics to pursue.

PARAMETER SCHEMA:
{EditorialReviewRequest.model_json_schema()}

CORRECT USAGE EXAMPLES:
# Review research and select topics
{{"research_result": research_data, "forbidden_topics": ["topic1", "topic2"], "max_topics_to_select": 5}}

# Priority review for specific fields
{{"research_result": research_data, "priority_fields": ["technology"], "max_topics_to_select": 3}}

EDITORIAL PROCESS:
1. Review all researched topics and their quality
2. Check against forbidden topics from topic memory
3. Evaluate trending scores and newsworthiness
4. Consider field balance and editorial priorities
5. Select topics for story assignment
6. Generate editorial notes and assignment priorities

DECISION CRITERIA:
- Topic uniqueness (not in forbidden list)
- Research quality and depth
- Trending score and timeliness
- Editorial value and audience appeal
- Field balance and coverage diversity

RETURNS:
- selected_topics: Topics chosen for story development
- rejected_topics: Topics rejected with reasons
- editorial_notes: Specific guidance for each selected topic
- assignment_priority: Priority ranking for story assignments

This tool enables informed editorial decision-making based on comprehensive research data.
"""
        super().__init__(name=name, description=description)
        self.params_model = EditorialReviewRequest
        self.config_service = config_service or ConfigService()
        self.llm_service = LLMService(self.config_service)
        self.forbidden_tool = ForbiddenTopicsTool(self.config_service)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.SLOW) -> EditorialDecision:
        """Execute editorial research review."""
        if not isinstance(params, EditorialReviewRequest):
            return EditorialDecision(
                selected_topics=[],
                rejected_topics=[f"Invalid parameters: expected EditorialReviewRequest, got {type(params)}"]
            )

        try:
            logger.info(f"Reviewing {len(params.research_result.topics_researched)} researched topics")

            # Step 1: Filter out forbidden topics
            filtered_topics = await self._filter_forbidden_topics(
                params.research_result.topics_researched,
                params.forbidden_topics
            )

            # Step 2: Evaluate and rank topics
            evaluated_topics = await self._evaluate_topics(filtered_topics, params)

            # Step 3: Select top topics based on criteria
            selected_topics = await self._select_topics(evaluated_topics, params)

            # Step 4: Generate editorial notes
            editorial_notes = await self._generate_editorial_notes(selected_topics)

            # Step 5: Create assignment priorities
            assignment_priority = self._create_assignment_priorities(selected_topics)

            # Step 6: Generate rejection reasons
            rejected_topics = self._generate_rejection_reasons(
                params.research_result.topics_researched,
                selected_topics,
                params.forbidden_topics
            )

            return EditorialDecision(
                selected_topics=selected_topics,
                rejected_topics=rejected_topics,
                editorial_notes=editorial_notes,
                assignment_priority=assignment_priority
            )

        except Exception as e:
            logger.error(f"Editorial research review failed: {e}")
            return EditorialDecision(
                selected_topics=[],
                rejected_topics=[f"Review failed: {str(e)}"]
            )

    async def _filter_forbidden_topics(self, topics: list[TopicResearch], forbidden_topics: list[str]) -> list[TopicResearch]:
        """Filter out topics that are forbidden or too similar to recent coverage."""
        filtered_topics = []

        for topic in topics:
            # Check against explicit forbidden list
            if any(forbidden.lower() in topic.topic_title.lower() for forbidden in forbidden_topics):
                logger.info(f"Filtered out forbidden topic: {topic.topic_title}")
                continue

            # Check similarity against forbidden topics using the forbidden topics tool
            similarity_params = ForbiddenTopicsParams(
                operation="check_similarity",
                proposed_topics=[topic.topic_title]
            )

            similarity_result = await self.forbidden_tool.execute(similarity_params)

            if (similarity_result.success and
                similarity_result.similar_topics.get(topic.topic_title)):
                logger.info(f"Filtered out similar topic: {topic.topic_title}")
                continue

            filtered_topics.append(topic)

        logger.info(f"Filtered {len(topics)} topics down to {len(filtered_topics)} unique topics")
        return filtered_topics

    async def _evaluate_topics(self, topics: list[TopicResearch], params: EditorialReviewRequest) -> list[tuple[TopicResearch, float]]:
        """Evaluate and score topics based on editorial criteria."""
        evaluated_topics = []

        for topic in topics:
            score = 0.0

            # Base score from trending score
            score += topic.trending_score * 0.3

            # Score from research depth
            score += min(topic.research_depth / 5.0, 0.2)

            # Score from content quality
            if len(topic.content_summary) > 100:
                score += 0.15
            if len(topic.key_points) >= 3:
                score += 0.15

            # Priority field bonus
            if topic.field in params.priority_fields:
                score += 0.2

            # Recency bonus
            hours_old = (datetime.now() - topic.research_timestamp).total_seconds() / 3600
            if hours_old < 24:
                score += 0.1
            elif hours_old < 72:
                score += 0.05

            evaluated_topics.append((topic, score))

        # Sort by score descending
        evaluated_topics.sort(key=lambda x: x[1], reverse=True)

        return evaluated_topics

    async def _select_topics(self, evaluated_topics: list[tuple[TopicResearch, float]], params: EditorialReviewRequest) -> list[TopicResearch]:
        """Select top topics based on editorial criteria and field balance."""
        selected_topics = []
        field_counts = {}

        max_per_field = max(1, params.max_topics_to_select // 3)  # Balance across fields

        for topic, score in evaluated_topics:
            if len(selected_topics) >= params.max_topics_to_select:
                break

            field_count = field_counts.get(topic.field, 0)

            # Select if we haven't reached field limit or if it's a high-scoring topic
            if field_count < max_per_field or score > 0.8:
                selected_topics.append(topic)
                field_counts[topic.field] = field_count + 1
                logger.info(f"Selected topic: {topic.topic_title} (score: {score:.2f})")

        return selected_topics

    async def _generate_editorial_notes(self, topics: list[TopicResearch]) -> dict[str, str]:
        """Generate editorial notes for each selected topic."""
        editorial_notes = {}

        for topic in topics:
            try:
                prompt = f"""As an editor, provide specific editorial guidance for this story topic:

Topic: {topic.topic_title}
Field: {topic.field.value}
Description: {topic.topic_description}
Key Points: {', '.join(topic.key_points)}

Provide 2-3 sentences of editorial guidance covering:
- Story angle and focus
- Target audience considerations
- Key elements to emphasize

Keep it concise and actionable for the reporter."""

                response = await self.llm_service.generate_text(
                    prompt=prompt,
                    model_speed=ModelSpeed.FAST,
                    max_tokens=150
                )

                editorial_notes[topic.topic_title] = response.strip()

            except Exception as e:
                logger.error(f"Failed to generate editorial notes for {topic.topic_title}: {e}")
                editorial_notes[topic.topic_title] = "Focus on key findings and provide clear, engaging coverage for general audience."

        return editorial_notes

    def _create_assignment_priorities(self, topics: list[TopicResearch]) -> dict[str, int]:
        """Create priority rankings for story assignments."""
        assignment_priority = {}

        # Sort by trending score and assign priorities
        sorted_topics = sorted(topics, key=lambda t: t.trending_score, reverse=True)

        for i, topic in enumerate(sorted_topics):
            priority = min(i + 1, 5)  # Priority 1-5 (1 is highest)
            assignment_priority[topic.topic_title] = priority

        return assignment_priority

    def _generate_rejection_reasons(self, all_topics: list[TopicResearch], selected_topics: list[TopicResearch], forbidden_topics: list[str]) -> list[str]:
        """Generate reasons for rejected topics."""
        selected_titles = {topic.topic_title for topic in selected_topics}
        rejected_reasons = []

        for topic in all_topics:
            if topic.topic_title not in selected_titles:
                reason = f"{topic.topic_title}: "

                # Check why it was rejected
                if any(forbidden.lower() in topic.topic_title.lower() for forbidden in forbidden_topics):
                    reason += "Similar to recent coverage"
                elif topic.trending_score < 0.3:
                    reason += "Low trending score"
                elif topic.research_depth < 2:
                    reason += "Insufficient research depth"
                else:
                    reason += "Lower editorial priority"

                rejected_reasons.append(reason)

        return rejected_reasons
```

### 5.2 Update Editor Agent System Prompt

Update the Editor Agent to work with the new three-agent workflow. Modify the system prompt in `libs/common/agents/editor_agent/editor_prompt.py`:

```python
EDITOR_SYSTEM_PROMPT_THREE_AGENT = """
You are the Editor-in-Chief of BobTimes, an AI-powered newspaper operating with a three-agent workflow: Researcher, Editor, and Reporter.

## THREE-AGENT WORKFLOW:

### 1. Research Phase (Researcher Agent)
- Researcher Agent discovers trending topics across multiple sources
- Gathers comprehensive research data including YouTube, web search, and scraping
- Provides structured research packages with sources, summaries, and analysis

### 2. Editorial Phase (Your Role)
- Review research findings from Researcher Agent
- Check against topic memory to prevent duplication
- Make editorial decisions on which topics to pursue
- Assign selected topics with research data to Reporter Agents

### 3. Writing Phase (Reporter Agent)
- Reporter Agents write stories based on your assignments and research data
- Focus on story creation rather than research and topic discovery
- Use provided research as foundation for comprehensive stories

## YOUR EDITORIAL RESPONSIBILITIES:

### Research Review & Selection
- Use editorial_research_review tool to evaluate research findings
- Apply editorial judgment to select newsworthy topics
- Consider trending scores, research depth, and audience appeal
- Ensure field balance across technology, science, and economics

### Topic Memory Management
- Use forbidden_topics tool to check against recent coverage
- Prevent topic duplication within editorial timeframes
- Add published topics to memory for future reference
- Maintain editorial consistency across news cycles

### Story Assignment
- Assign selected topics to appropriate Reporter Agents
- Provide research data and editorial guidance
- Set priorities and deadlines for story development
- Include specific editorial notes and requirements

### Quality Control
- Review completed stories for accuracy and quality
- Ensure stories meet editorial standards
- Provide feedback for revisions when needed
- Approve stories for publication

## AVAILABLE TOOLS:

- **editorial_research_review**: Review and select topics from research findings
- **forbidden_topics**: Manage topic memory and prevent duplication
- **assign_topics**: Assign selected topics with research to reporters
- **collect_story**: Collect completed stories from reporters
- **review_story**: Review and provide feedback on stories
- **publish_story**: Publish approved stories

## EDITORIAL DECISION CRITERIA:

1. **Uniqueness**: Not covered recently (check topic memory)
2. **Newsworthiness**: High trending score and current relevance
3. **Research Quality**: Sufficient sources and depth
4. **Audience Appeal**: Broad interest and engagement potential
5. **Field Balance**: Diverse coverage across all fields
6. **Editorial Value**: Aligns with publication standards

## WORKFLOW PROCESS:

1. **Receive Research**: Get comprehensive research from Researcher Agent
2. **Review & Filter**: Use editorial_research_review to evaluate topics
3. **Check Memory**: Verify topics against forbidden_topics
4. **Make Decisions**: Select topics based on editorial criteria
5. **Assign Stories**: Provide research data and guidance to reporters
6. **Review & Publish**: Oversee story completion and publication
7. **Update Memory**: Add published topics to topic memory

Your role is crucial in maintaining editorial quality and ensuring the newspaper delivers unique, high-quality content based on comprehensive research.
"""
```



## 🔧 Step 7: Update Reporter Agent for Research-Based Writing

### 7.1 Create Research-Based Story Writing Tool

Create the file `libs/common/agents/reporter_agent/research_story_tool.py`:

```python
"""Research-based story writing tool for Reporter Agent."""

from datetime import datetime
from typing import Any

from agents.tools.base_tool import BaseTool
from core.config_service import ConfigService
from core.llm_service import LLMService, ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

from agents.researcher_agent.research_models import StoryAssignment, TopicResearch
from agents.models.story_models import StoryDraft, StorySource

logger = get_logger(__name__)


class ResearchStoryParams(BaseModel):
    """Parameters for research-based story writing."""
    story_assignment: StoryAssignment = Field(description="Story assignment with research data")
    writing_style: str = Field(default="informative", description="Writing style: informative, engaging, analytical")
    include_quotes: bool = Field(default=True, description="Include relevant quotes from sources")
    focus_angle: str = Field(default="general", description="Story focus angle")


class ResearchStoryResult(BaseModel):
    """Result from research-based story writing."""
    success: bool = Field(description="Whether story writing was successful")
    story_draft: StoryDraft | None = Field(None, description="Generated story draft")
    word_count: int = Field(default=0, description="Story word count")
    sources_used: int = Field(default=0, description="Number of sources incorporated")
    research_coverage: float = Field(default=0.0, description="Percentage of research data used")
    writing_time: float = Field(default=0.0, description="Time taken to write story")
    error: str | None = Field(None, description="Error message if failed")


class ResearchStoryTool(BaseTool):
    """Tool for writing stories based on comprehensive research data."""

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize research story tool."""
        name = "research_story_writer"
        description = f"""
Write comprehensive stories based on research data provided by the Editor Agent.

PARAMETER SCHEMA:
{ResearchStoryParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
# Write story from research assignment
{{"story_assignment": assignment_data, "writing_style": "engaging", "include_quotes": true}}

# Analytical story with specific focus
{{"story_assignment": assignment_data, "writing_style": "analytical", "focus_angle": "industry_impact"}}

WRITING PROCESS:
1. Analyze provided research data and sources
2. Extract key information and insights
3. Structure story with compelling narrative
4. Incorporate quotes and evidence from sources
5. Ensure comprehensive coverage of research findings
6. Meet word count and editorial requirements

STORY ELEMENTS:
- Compelling headline and lead paragraph
- Well-structured body with clear flow
- Integration of research findings and sources
- Proper attribution and source citations
- Engaging conclusion that ties themes together

WRITING STYLES:
- informative: Clear, factual reporting style
- engaging: More narrative and accessible approach
- analytical: Deep analysis with expert insights

RETURNS:
- story_draft: Complete story with headline, content, and sources
- word_count: Final word count
- sources_used: Number of research sources incorporated
- research_coverage: How much of the research was utilized

This tool creates high-quality stories based on comprehensive research rather than conducting new research.
"""
        super().__init__(name=name, description=description)
        self.params_model = ResearchStoryParams
        self.config_service = config_service or ConfigService()
        self.llm_service = LLMService(self.config_service)

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.SLOW) -> ResearchStoryResult:
        """Execute research-based story writing."""
        if not isinstance(params, ResearchStoryParams):
            return ResearchStoryResult(
                success=False,
                error=f"Expected ResearchStoryParams, got {type(params)}"
            )

        start_time = datetime.now()

        try:
            logger.info(f"Writing story: {params.story_assignment.topic_research.topic_title}")

            # Step 1: Analyze research data
            research_analysis = await self._analyze_research_data(params.story_assignment.topic_research)

            # Step 2: Create story structure
            story_structure = await self._create_story_structure(
                params.story_assignment, research_analysis, params
            )

            # Step 3: Write story content
            story_content = await self._write_story_content(
                story_structure, params.story_assignment, params
            )

            # Step 4: Generate headline
            headline = await self._generate_headline(
                params.story_assignment.topic_research, story_content
            )

            # Step 5: Convert research sources to story sources
            story_sources = self._convert_research_sources(
                params.story_assignment.topic_research.sources
            )

            # Step 6: Create story draft
            story_draft = StoryDraft(
                title=headline,
                content=story_content,
                field=params.story_assignment.reporter_field,
                sources=story_sources,
                summary=params.story_assignment.topic_research.topic_description,
                word_count=len(story_content.split()),
                created_at=datetime.now()
            )

            # Calculate metrics
            writing_time = (datetime.now() - start_time).total_seconds()
            sources_used = len([s for s in story_sources if s.url in story_content])
            research_coverage = self._calculate_research_coverage(
                params.story_assignment.topic_research, story_content
            )

            return ResearchStoryResult(
                success=True,
                story_draft=story_draft,
                word_count=story_draft.word_count,
                sources_used=sources_used,
                research_coverage=research_coverage,
                writing_time=writing_time
            )

        except Exception as e:
            logger.error(f"Research story writing failed: {e}")
            return ResearchStoryResult(
                success=False,
                error=str(e)
            )

    async def _analyze_research_data(self, topic_research: TopicResearch) -> dict[str, Any]:
        """Analyze research data to extract key themes and insights."""
        try:
            # Combine all source content
            all_content = []
            for source in topic_research.sources:
                content = source.content or source.summary
                all_content.append(f"{source.title}: {content}")

            combined_content = "\n\n".join(all_content)

            prompt = f"""Analyze this research data and provide key insights for story writing:

Topic: {topic_research.topic_title}
Description: {topic_research.topic_description}
Key Points: {', '.join(topic_research.key_points)}

Research Content:
{combined_content[:2000]}...

Provide analysis in this format:
MAIN_THEMES: [list 3-4 main themes]
KEY_INSIGHTS: [list 3-4 key insights]
STORY_ANGLES: [list 2-3 potential story angles]
IMPORTANT_FACTS: [list 4-5 important facts to include]"""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=400
            )

            # Parse response (simplified parsing)
            analysis = {
                "main_themes": [],
                "key_insights": [],
                "story_angles": [],
                "important_facts": []
            }

            current_section = None
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('MAIN_THEMES:'):
                    current_section = "main_themes"
                    continue
                elif line.startswith('KEY_INSIGHTS:'):
                    current_section = "key_insights"
                    continue
                elif line.startswith('STORY_ANGLES:'):
                    current_section = "story_angles"
                    continue
                elif line.startswith('IMPORTANT_FACTS:'):
                    current_section = "important_facts"
                    continue

                if current_section and line:
                    analysis[current_section].append(line.lstrip('- '))

            return analysis

        except Exception as e:
            logger.error(f"Research analysis failed: {e}")
            return {
                "main_themes": [topic_research.topic_title],
                "key_insights": topic_research.key_points,
                "story_angles": ["general coverage"],
                "important_facts": topic_research.key_points
            }

    async def _create_story_structure(self, assignment: StoryAssignment, analysis: dict, params: ResearchStoryParams) -> dict[str, str]:
        """Create story structure based on research and editorial guidelines."""
        try:
            prompt = f"""Create a story structure for this article:

Topic: {assignment.topic_research.topic_title}
Field: {assignment.reporter_field.value}
Word Count Target: {assignment.required_word_count}
Writing Style: {params.writing_style}
Editorial Guidelines: {assignment.editorial_guidelines}
Assignment Notes: {assignment.assignment_notes}

Main Themes: {', '.join(analysis['main_themes'])}
Key Insights: {', '.join(analysis['key_insights'])}

Create a structure with:
LEAD: [compelling opening paragraph outline]
BODY_SECTION_1: [first main section outline]
BODY_SECTION_2: [second main section outline]
BODY_SECTION_3: [third main section outline]
CONCLUSION: [conclusion outline]"""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=300
            )

            # Parse structure
            structure = {}
            current_section = None

            for line in response.split('\n'):
                line = line.strip()
                if ':' in line and line.split(':')[0].upper() in ['LEAD', 'BODY_SECTION_1', 'BODY_SECTION_2', 'BODY_SECTION_3', 'CONCLUSION']:
                    parts = line.split(':', 1)
                    current_section = parts[0].upper()
                    structure[current_section] = parts[1].strip() if len(parts) > 1 else ""
                elif current_section and line:
                    structure[current_section] += " " + line

            return structure

        except Exception as e:
            logger.error(f"Story structure creation failed: {e}")
            return {
                "LEAD": "Compelling opening about the topic",
                "BODY_SECTION_1": "Main findings and details",
                "BODY_SECTION_2": "Analysis and implications",
                "BODY_SECTION_3": "Expert perspectives and context",
                "CONCLUSION": "Summary and future outlook"
            }

    async def _write_story_content(self, structure: dict, assignment: StoryAssignment, params: ResearchStoryParams) -> str:
        """Write the complete story content based on structure and research."""
        try:
            # Prepare research context
            research_context = self._prepare_research_context(assignment.topic_research)

            prompt = f"""Write a complete {params.writing_style} news story based on this research and structure:

TOPIC: {assignment.topic_research.topic_title}
FIELD: {assignment.reporter_field.value}
TARGET WORDS: {assignment.required_word_count}
EDITORIAL NOTES: {assignment.assignment_notes}

STORY STRUCTURE:
{chr(10).join(f'{section}: {outline}' for section, outline in structure.items())}

RESEARCH DATA:
{research_context}

WRITING REQUIREMENTS:
- Write in {params.writing_style} style
- Include specific facts and details from research
- {"Include relevant quotes where appropriate" if params.include_quotes else "Focus on factual reporting without quotes"}
- Maintain journalistic standards and accuracy
- Create engaging, informative content
- Target approximately {assignment.required_word_count} words

Write the complete story now:"""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.SLOW,
                max_tokens=min(assignment.required_word_count * 2, 2000)
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Story content writing failed: {e}")
            return f"Story about {assignment.topic_research.topic_title} could not be generated due to technical issues."

    def _prepare_research_context(self, topic_research: TopicResearch) -> str:
        """Prepare research context for story writing."""
        context_parts = []

        # Add topic information
        context_parts.append(f"TOPIC DESCRIPTION: {topic_research.topic_description}")
        context_parts.append(f"KEY POINTS: {', '.join(topic_research.key_points)}")
        context_parts.append(f"KEYWORDS: {', '.join(topic_research.keywords)}")

        # Add source information
        context_parts.append("\nSOURCE INFORMATION:")
        for i, source in enumerate(topic_research.sources[:5], 1):  # Use top 5 sources
            content = source.content or source.summary
            context_parts.append(f"{i}. {source.title}")
            context_parts.append(f"   URL: {source.url}")
            context_parts.append(f"   Content: {content[:300]}...")
            context_parts.append("")

        return "\n".join(context_parts)

    async def _generate_headline(self, topic_research: TopicResearch, story_content: str) -> str:
        """Generate compelling headline for the story."""
        try:
            prompt = f"""Create a compelling, accurate headline for this news story:

Topic: {topic_research.topic_title}
Field: {topic_research.field.value}
Key Points: {', '.join(topic_research.key_points[:3])}

Story Content Preview:
{story_content[:500]}...

Create a headline that is:
- Accurate and informative
- Engaging and clickable
- 8-12 words long
- Captures the main news value

Headline:"""

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_speed=ModelSpeed.FAST,
                max_tokens=50
            )

            headline = response.strip().strip('"').strip("'")
            return headline if headline else topic_research.topic_title

        except Exception as e:
            logger.error(f"Headline generation failed: {e}")
            return topic_research.topic_title

    def _convert_research_sources(self, research_sources: list) -> list[StorySource]:
        """Convert research sources to story sources."""
        story_sources = []

        for research_source in research_sources:
            story_source = StorySource(
                url=research_source.url,
                title=research_source.title,
                summary=research_source.summary,
                accessed_at=research_source.accessed_at
            )
            story_sources.append(story_source)

        return story_sources

    def _calculate_research_coverage(self, topic_research: TopicResearch, story_content: str) -> float:
        """Calculate how much of the research data was used in the story."""
        try:
            story_lower = story_content.lower()

            # Check coverage of key points
            key_points_used = sum(1 for point in topic_research.key_points
                                if any(word.lower() in story_lower for word in point.split() if len(word) > 3))

            # Check coverage of keywords
            keywords_used = sum(1 for keyword in topic_research.keywords
                              if keyword.lower() in story_lower)

            # Check coverage of sources (by title words)
            sources_referenced = 0
            for source in topic_research.sources:
                title_words = [word for word in source.title.split() if len(word) > 3]
                if any(word.lower() in story_lower for word in title_words):
                    sources_referenced += 1

            # Calculate overall coverage
            total_elements = len(topic_research.key_points) + len(topic_research.keywords) + len(topic_research.sources)
            used_elements = key_points_used + keywords_used + sources_referenced

            return min(used_elements / total_elements if total_elements > 0 else 0.0, 1.0)

        except Exception as e:
            logger.error(f"Research coverage calculation failed: {e}")
            return 0.5
```

### 7.2 Update Reporter Agent System Prompt

Update the Reporter Agent to work with research-based assignments. Modify the system prompt:

```python
REPORTER_SYSTEM_PROMPT_RESEARCH_BASED = """
You are a Reporter Agent for BobTimes, specialized in writing high-quality stories based on comprehensive research data provided by the Editor Agent.

## RESEARCH-BASED WORKFLOW:

### Your Role in Three-Agent System:
1. **Researcher Agent**: Discovers topics and gathers comprehensive research
2. **Editor Agent**: Reviews research, makes editorial decisions, assigns stories
3. **Reporter Agent (You)**: Write stories based on provided research and editorial guidance

### Your Responsibilities:
- Write compelling stories using provided research data
- Incorporate multiple sources and perspectives
- Follow editorial guidelines and requirements
- Meet word count and style specifications
- Ensure accuracy and proper attribution

## STORY WRITING PROCESS:

### 1. Research Analysis
- Review all provided research sources thoroughly
- Identify key themes, insights, and story angles
- Extract important facts and supporting evidence
- Note source credibility and relevance

### 2. Story Structure
- Create compelling lead paragraph
- Develop logical story flow
- Incorporate research findings naturally
- Build toward strong conclusion

### 3. Content Creation
- Write in specified style (informative, engaging, analytical)
- Include relevant quotes and evidence from sources
- Maintain journalistic standards and accuracy
- Ensure proper source attribution

### 4. Quality Assurance
- Meet target word count requirements
- Verify facts against research sources
- Ensure comprehensive coverage of research
- Maintain editorial consistency

## AVAILABLE TOOLS:

- **research_story_writer**: Write complete stories based on research assignments

## WRITING STANDARDS:

### Accuracy & Attribution
- Use only information from provided research sources
- Properly attribute all facts and quotes
- Verify information across multiple sources when possible
- Maintain journalistic integrity

### Style & Engagement
- Write compelling headlines and leads
- Use clear, accessible language
- Create engaging narrative flow
- Balance information with readability

### Editorial Compliance
- Follow all editorial guidelines and notes
- Meet specified word count targets
- Adhere to publication standards
- Incorporate editorial feedback

## RESEARCH UTILIZATION:

- **Comprehensive Coverage**: Use research data thoroughly
- **Source Integration**: Incorporate multiple research sources
- **Fact Verification**: Cross-reference information when possible
- **Context Building**: Use research to provide proper context

Your stories should demonstrate deep understanding of the research while creating engaging, informative content for readers. Focus on storytelling that brings research to life rather than conducting new research.
"""
```


## 🎯 Step 9: Integration with Full Newspaper Generation

### 9.1 Workflow Outline (no full script)

- Phase 1 — Researcher: discover trending topics per field, aggregate multi-source research, save to SharedMemoryStore
- Phase 2 — Editor: review research results, filter against forbidden topics, select topics, add notes and priorities
- Phase 3 — Reporter (memory-only): retrieve research from memory using fetch_from_memory (topic_key or topic_query), write story with use_llm
- Phase 4 — Publication: publish stories and update topic memory

Note:
- Reporter uses only fetch_from_memory and use_llm (no search/scrape/YouTube)
- All memory access is via our SharedMemoryStore

### 9.2 Minimal code snippets (guidance only)

Create agents:
```python
from agents.agent_factory import AgentFactory
from agents.types import ReporterField
from core.config_service import ConfigService

config = ConfigService()
factory = AgentFactory(config)
researcher = factory.create_researcher()
editor = factory.create_editor()
reporter = factory.create_reporter(ReporterField.TECHNOLOGY)
```

Fetch research from memory with intelligent matching:
```python
from agents.reporter_agent.reporter_tools import FetchFromMemoryParams

# Get the fetch tool from the reporter's toolset
fetch_tool = next(t for t in reporter.tools if t.name == "fetch_from_memory")

# Use topic_query when the exact key is unknown; tool selects the best match
params = FetchFromMemoryParams(field="technology", topic_query="o4o multimodal launch")
result = await fetch_tool.execute(params)
# result.topic_key, result.content_summary
```

Draft with the LLM:
```python
from agents.reporter_agent.reporter_tools import UseLLMParams

use_llm = next(t for t in reporter.tools if t.name == "use_llm")
draft = await use_llm.execute(UseLLMParams(prompt="Write a 200-word news brief summarizing the research."))
```

## 🏆 Step 10: Lab Summary and Best Practices (Concise)

Architecture at a glance
- Researcher: multi-source research, topic discovery, saves results to SharedMemoryStore
- Editor: reviews research, filters against forbidden topics, selects topics, assigns stories
- Reporter: memory-only writing using fetch_from_memory + use_llm (no search/scrape/YouTube)

Key guidelines
- Use the project-native SharedMemoryStore
- Prefer exact topic_key; otherwise provide topic_query for intelligent matching
- Keep topic keys consistent across news cycles for reliable retrieval
- Use the forbidden topics tool to avoid duplication across cycles
- Keep reporter prompts focused on writing from memory; no research tools
- Use enums and Pydantic models for type safety across agents and tools

Performance and quality
- Parallelize safe research tasks to improve throughput
- Cache reusable research results where helpful
- Attribute sources with URLs and preserve provenance
- Use the shared logging service for clear, structured logs

Optional enhancements
- Embedding-based topic matching
- Multilingual research and summarization
- Automated fact-checking and source quality scoring
- Image workflow refinements (selection > generation)

🎉 Lab 4 complete: your system now runs a three-agent workflow with memory-first writing.
