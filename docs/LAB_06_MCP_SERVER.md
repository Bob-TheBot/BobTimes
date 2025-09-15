# Lab 6: MCP Server - Unified Tool Interface

Welcome to Lab 6! In this lab, you'll create a comprehensive MCP (Model Context Protocol) server that wraps all your search and editor tools into a single, unified interface. This will transform your agents to use only MCP tools instead of regular tools, creating a more standardized and interoperable system.

## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Create a unified MCP server using FastMCP
- ✅ Wrap DuckDuckGo search tools as MCP tools
- ✅ Convert YouTube search and transcription tools to MCP
- ✅ Transform web scraper tools into MCP tools
- ✅ Convert file/content creation tools to MCP tools
- ✅ Test the complete MCP server with all tools
- ✅ Update agents to use MCP tools exclusively
- ✅ Understand MCP architecture and benefits

## 📋 Prerequisites

- ✅ Completed Labs 1-5 (Basic setup through Researcher Agent)
- ✅ Working DevContainer or local development environment
- ✅ Understanding of the existing tool architecture
- ✅ Familiarity with the agent system and workflow
- ✅ Basic knowledge of MCP concepts

## 🏗️ Step 1: Design the MCP Server Architecture

### 1.1 MCP Server Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BobTimes MCP Server                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔍 Search Tools          📺 YouTube Tools      📄 Content Tools│
│  ├─ DuckDuckGo Text      ├─ Channel Search     ├─ Story Creation│
│  ├─ DuckDuckGo Images    ├─ Video Transcribe   ├─ File Writing  │
│  ├─ DuckDuckGo News      └─ Topic Extraction   ├─ Image Gen     │
│  └─ DuckDuckGo Videos                          └─ Content Save  │
│                                                                 │
│  🌐 Web Scraping         📝 Editor Tools       🗂️ File Tools    │
│  ├─ URL Scraping         ├─ Story Review       ├─ File Read     │
│  ├─ Content Extraction   ├─ Story Publishing   ├─ File Write    │
│  └─ Batch Scraping       └─ Content Editing    └─ File Delete   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Clients                                │
├─────────────────────────────────────────────────────────────────┤
│  📰 Editor Agent    📝 Reporter Agent    🔬 Researcher Agent    │
│  Uses MCP Tools     Uses MCP Tools       Uses MCP Tools        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Tool Categories to Convert

**Search & Discovery Tools:**
- DuckDuckGo text, image, news, video search
- YouTube channel search and video discovery
- Web content scraping and extraction

**Content Creation Tools:**
- Story writing and editing
- Image generation and management
- File creation and manipulation

**Editorial Tools:**
- Story review and publishing
- Content organization
- Source management

## 🔧 Step 2: Create the MCP Server Foundation

### 2.1 Create MCP Server Directory

First, create a new directory for your MCP server:

```bash
mkdir -p mcp_server
cd mcp_server
```

### 2.2 Create MCP Server Configuration

Create `mcp_server/pyproject.toml`:

```toml
[project]
name = "bobtimes-mcp-server"
version = "0.1.0"
description = "BobTimes MCP Server - Unified tool interface"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastmcp>=2.12.2",
    "common",
    "pydantic>=2.0.0",
    "httpx>=0.28.1",
    "playwright>=1.40.0",
    "duckduckgo-search>=6.0.0",
    "google-api-python-client>=2.0.0",
    "youtube-transcript-api>=0.6.0",
]

[tool.uv.sources]
common = { workspace = true }
```

### 2.3 Update Root Workspace

Add the MCP server to your root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["libs/*/", "backend", "mcp_demo", "mcp_server"]
```

## 🛠️ Step 3: Implement Search Tools as MCP Tools

### 3.1 Create the Main MCP Server

Create `mcp_server/main.py`:

```python
"""BobTimes MCP Server - Unified tool interface for all BobTimes tools."""

import asyncio
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import McpError
from pydantic import BaseModel, Field

# Import existing tools
from common import (
    DuckDuckGoSearchTool,
    DDGRegion,
    DDGSafeSearch,
    DDGTimeLimit,
    AsyncPlaywrightScraper,
    create_duckduckgo_search_tool,
    create_async_playwright_scraper,
)
from core.config_service import ConfigService
from core.logging_service import get_logger

logger = get_logger(__name__)

# Initialize MCP server
mcp = FastMCP("BobTimes Unified Tools")

# Initialize services
config_service = ConfigService()
ddg_tool = create_duckduckgo_search_tool()
scraper = create_async_playwright_scraper()


# ============== SEARCH TOOL MODELS ==============

class DuckDuckGoSearchParams(BaseModel):
    """Parameters for DuckDuckGo search operations."""
    query: str = Field(description="Search query")
    search_type: str = Field(default="text", description="Type: text, images, videos, news")
    max_results: int = Field(default=10, description="Maximum results to return")
    region: str = Field(default="wt-wt", description="Search region")
    safe_search: str = Field(default="moderate", description="Safe search level")
    time_limit: str | None = Field(default=None, description="Time limit for results")


class WebScrapingParams(BaseModel):
    """Parameters for web scraping operations."""
    url: str = Field(description="URL to scrape")
    output_path: str | None = Field(default=None, description="Optional file path to save content")


# ============== SEARCH MCP TOOLS ==============

@mcp.tool()
def duckduckgo_search(params: DuckDuckGoSearchParams) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo for text, images, videos, or news.
    
    Supports multiple search types:
    - text: Web search results
    - images: Image search results  
    - videos: Video search results
    - news: News article results
    
    Args:
        params: Search parameters including query, type, and filters
        
    Returns:
        Dictionary with search results and metadata
    """
    try:
        logger.info(f"DuckDuckGo search: {params.query} (type: {params.search_type})")
        
        # Convert string parameters to enums
        region = DDGRegion(params.region) if params.region else DDGRegion.no_region
        safe_search = DDGSafeSearch(params.safe_search) if params.safe_search else DDGSafeSearch.moderate
        time_limit = DDGTimeLimit(params.time_limit) if params.time_limit else None
        
        # Perform search based on type
        results = ddg_tool.search_with_filters(
            query=params.query,
            search_type=params.search_type,
            region=region,
            safe_search=safe_search,
            time_limit=time_limit,
            max_results=params.max_results
        )
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serialized_results.append(result.model_dump())
            else:
                serialized_results.append(dict(result))
        
        return {
            "success": True,
            "search_type": params.search_type,
            "query": params.query,
            "results_count": len(serialized_results),
            "results": serialized_results
        }
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": params.query,
            "search_type": params.search_type
        }


@mcp.tool()
async def web_scraper(params: WebScrapingParams) -> dict[str, Any]:
    """
    Scrape content from a web URL using Playwright for dynamic content support.
    
    Handles JavaScript-rendered pages and extracts clean text content.
    Optionally saves content to a file.
    
    Args:
        params: Scraping parameters including URL and optional output path
        
    Returns:
        Dictionary with scraped content and metadata
    """
    try:
        logger.info(f"Web scraping: {params.url}")
        
        # Scrape the URL
        result = await scraper.scrape_and_save(params.url, params.output_path)
        
        return {
            "success": result.success,
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "word_count": result.word_count,
            "scraped_at": result.scraped_at,
            "error": result.error_message,
            "saved_to": params.output_path if params.output_path and result.success else None
        }
        
    except Exception as e:
        logger.error(f"Web scraping failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": params.url
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
```

## 🎬 Step 4: Add YouTube Tools to MCP Server

### 4.1 Add YouTube Tool Models

Add to `mcp_server/main.py` after the existing models:

```python
# ============== YOUTUBE TOOL MODELS ==============

class YouTubeSearchParams(BaseModel):
    """Parameters for YouTube search operations."""
    channel_ids: list[str] = Field(description="List of YouTube channel IDs")
    max_videos_per_channel: int = Field(default=5, description="Max videos per channel")
    days_back: int = Field(default=7, description="Days to look back for videos")
    operation: str = Field(default="topics", description="Operation: topics or transcribe")
    specific_video_ids: list[str] = Field(default_factory=list, description="Specific video IDs for transcription")
```

### 4.2 Add YouTube MCP Tools

Add to `mcp_server/main.py` after the web scraper tool:

```python
# Import YouTube tools
from utils.youtube_tool import YouTubeReporterTool, YouTubeToolParams

# Initialize YouTube tool
youtube_tool = YouTubeReporterTool(config_service)

@mcp.tool()
async def youtube_search(params: YouTubeSearchParams) -> dict[str, Any]:
    """
    Search YouTube channels for videos and extract topics or transcripts.
    
    Supports two operations:
    - topics: Extract video titles as potential story topics
    - transcribe: Get video transcripts for content generation
    
    Args:
        params: YouTube search parameters
        
    Returns:
        Dictionary with video data, topics, or transcripts
    """
    try:
        logger.info(f"YouTube search: {len(params.channel_ids)} channels, operation: {params.operation}")
        
        # Convert to YouTube tool parameters
        youtube_params = YouTubeToolParams(
            channel_ids=params.channel_ids,
            max_videos_per_channel=params.max_videos_per_channel,
            days_back=params.days_back,
            operation=params.operation,
            specific_video_ids=params.specific_video_ids
        )
        
        # Execute YouTube tool
        result = await youtube_tool.execute(youtube_params)
        
        return {
            "success": result.success,
            "operation": result.operation,
            "videos_found": result.videos_found,
            "transcripts_obtained": result.transcripts_obtained,
            "topics": result.topics,
            "detailed_results": result.detailed_results,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"YouTube search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": params.operation
        }
```

## 📝 Step 5: Add File and Content Tools

### 5.1 Add File Tool Models

Continue adding to `mcp_server/main.py`:

```python
# ============== FILE TOOL MODELS ==============

class FileReadParams(BaseModel):
    """Parameters for file reading operations."""
    file_path: str = Field(description="Path to file to read")


class FileWriteParams(BaseModel):
    """Parameters for file writing operations."""
    file_path: str = Field(description="Path to file to write")
    content: str = Field(description="Content to write to file")
    create_dirs: bool = Field(default=True, description="Create parent directories if needed")


class FileDeleteParams(BaseModel):
    """Parameters for file deletion operations."""
    file_path: str = Field(description="Path to file to delete")
```

### 5.2 Add File MCP Tools

Add the file operation tools:

```python
# ============== FILE MCP TOOLS ==============

@mcp.tool()
def read_file(params: FileReadParams) -> dict[str, Any]:
    """
    Read content from a file.
    
    Args:
        params: File reading parameters
        
    Returns:
        Dictionary with file content and metadata
    """
    try:
        file_path = Path(params.file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {params.file_path}",
                "file_path": params.file_path
            }
        
        content = file_path.read_text(encoding="utf-8")
        file_size = file_path.stat().st_size
        
        logger.info(f"Read file: {params.file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "file_path": params.file_path,
            "content": content,
            "file_size": file_size,
            "word_count": len(content.split())
        }
        
    except Exception as e:
        logger.error(f"File read failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": params.file_path
        }


@mcp.tool()
def write_file(params: FileWriteParams) -> dict[str, Any]:
    """
    Write content to a file.
    
    Args:
        params: File writing parameters
        
    Returns:
        Dictionary with write operation results
    """
    try:
        file_path = Path(params.file_path)
        
        # Create parent directories if needed
        if params.create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        file_path.write_text(params.content, encoding="utf-8")
        file_size = file_path.stat().st_size
        
        logger.info(f"Wrote file: {params.file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "file_path": params.file_path,
            "file_size": file_size,
            "word_count": len(params.content.split())
        }
        
    except Exception as e:
        logger.error(f"File write failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": params.file_path
        }


@mcp.tool()
def delete_file(params: FileDeleteParams) -> dict[str, Any]:
    """
    Delete a file.
    
    Args:
        params: File deletion parameters
        
    Returns:
        Dictionary with deletion operation results
    """
    try:
        file_path = Path(params.file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {params.file_path}",
                "file_path": params.file_path
            }
        
        file_path.unlink()
        logger.info(f"Deleted file: {params.file_path}")
        
        return {
            "success": True,
            "file_path": params.file_path,
            "deleted": True
        }
        
    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": params.file_path
        }
```

## 🧪 Step 6: Test the MCP Server

### 6.1 Create Test Script

Create `mcp_server/test_server.py`:

```python
"""Test script for BobTimes MCP Server."""

import asyncio
import json
from pathlib import Path

from main import mcp, DuckDuckGoSearchParams, WebScrapingParams, YouTubeSearchParams


async def test_search_tools():
    """Test search functionality."""
    print("🔍 Testing DuckDuckGo Search...")
    
    # Test text search
    search_params = DuckDuckGoSearchParams(
        query="artificial intelligence news 2024",
        search_type="text",
        max_results=5
    )
    
    result = mcp.call_tool("duckduckgo_search", search_params.model_dump())
    print(f"Text search results: {result['results_count']} items")
    
    # Test news search
    news_params = DuckDuckGoSearchParams(
        query="technology breakthrough",
        search_type="news",
        max_results=3
    )
    
    result = mcp.call_tool("duckduckgo_search", news_params.model_dump())
    print(f"News search results: {result['results_count']} items")


async def test_scraping_tools():
    """Test web scraping functionality."""
    print("🌐 Testing Web Scraping...")
    
    scrape_params = WebScrapingParams(
        url="https://example.com",
        output_path="test_scraped_content.txt"
    )
    
    result = await mcp.call_tool("web_scraper", scrape_params.model_dump())
    print(f"Scraping result: {result['success']}, words: {result.get('word_count', 0)}")


async def test_file_tools():
    """Test file operations."""
    print("📁 Testing File Operations...")
    
    # Test file writing
    write_result = mcp.call_tool("write_file", {
        "file_path": "test_file.txt",
        "content": "This is a test file created by MCP server.",
        "create_dirs": True
    })
    print(f"File write: {write_result['success']}")
    
    # Test file reading
    read_result = mcp.call_tool("read_file", {
        "file_path": "test_file.txt"
    })
    print(f"File read: {read_result['success']}, content length: {len(read_result.get('content', ''))}")
    
    # Test file deletion
    delete_result = mcp.call_tool("delete_file", {
        "file_path": "test_file.txt"
    })
    print(f"File delete: {delete_result['success']}")


async def main():
    """Run all tests."""
    print("🧪 BobTimes MCP Server Test Suite")
    print("=" * 50)
    
    await test_search_tools()
    print()
    await test_scraping_tools()
    print()
    await test_file_tools()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 Run Tests

```bash
cd mcp_server
uv run python test_server.py
```

## 🚀 Step 7: Update Workspace and Run Server

### 7.1 Install Dependencies

```bash
# From project root
uv sync
```

### 7.2 Start MCP Server

```bash
cd mcp_server
uv run python main.py
```

The server will start and listen for MCP protocol messages via stdio.

## 📋 Step 8: Next Steps - Agent Integration

In the next phase, you'll:

1. **Update Agent Configurations**: Modify agents to use MCP tools instead of direct tool imports
2. **Create MCP Client Wrapper**: Build a client that connects agents to the MCP server
3. **Test Agent Workflows**: Verify that all existing workflows work with MCP tools
4. **Performance Optimization**: Optimize MCP communication for production use

## 🎉 Congratulations!

You've successfully created a unified MCP server that wraps all your BobTimes tools! This provides:

- **Standardized Interface**: All tools accessible via MCP protocol
- **Tool Isolation**: Tools run in separate process from agents
- **Interoperability**: MCP standard allows integration with other systems
- **Scalability**: Easy to add new tools or scale server independently

Your MCP server now provides a unified interface for:
- ✅ DuckDuckGo search (text, images, videos, news)
- ✅ Web scraping with Playwright
- ✅ YouTube search and transcription
- ✅ File operations (read, write, delete)
- ✅ Content creation and management

The foundation is set for converting your agents to use only MCP tools!

## 🎨 Step 9: Add Image Generation and Content Tools

### 9.1 Add Image Generation Models

Add to `mcp_server/main.py` after the file tool models:

```python
# ============== IMAGE TOOL MODELS ==============

class ImageGenerationParams(BaseModel):
    """Parameters for image generation operations."""
    prompt: str = Field(description="Text description of image to generate")
    size: str = Field(default="1024x1024", description="Image size")
    quality: str = Field(default="standard", description="Image quality")
    style: str = Field(default="vivid", description="Image style")
    n: int = Field(default=1, description="Number of images to generate")
    story_id: str | None = Field(default=None, description="Optional story ID for filename")


class StoryCreationParams(BaseModel):
    """Parameters for story creation operations."""
    title: str = Field(description="Story title")
    content: str = Field(description="Story content")
    field: str = Field(description="Story field/category")
    summary: str = Field(description="Story summary")
    keywords: list[str] = Field(default_factory=list, description="Story keywords")
    sources: list[dict] = Field(default_factory=list, description="Story sources")
```

### 9.2 Add Image and Content MCP Tools

Add after the file tools:

```python
# Import image and content tools
from agents.editor_agent.image_tool import ImageGenerationTool, ImageToolParams, ImageSize, ImageQuality, ImageStyle
from agents.editor_agent.editor_tools import NewspaperFileStore, Story, StoryStatus, ReporterField, NewspaperSection, StoryPriority
from core.llm_service import LLMService

# Initialize image tool and content store
llm_service = LLMService(config_service)
image_tool = ImageGenerationTool(llm_service=llm_service)
story_store = NewspaperFileStore()

@mcp.tool()
async def generate_image(params: ImageGenerationParams) -> dict[str, Any]:
    """
    Generate an image using AI based on a text prompt.

    Creates high-quality images suitable for news articles and content.
    Automatically saves images to local storage.

    Args:
        params: Image generation parameters

    Returns:
        Dictionary with generated image information
    """
    try:
        logger.info(f"Generating image: {params.prompt[:50]}...")

        # Convert to image tool parameters
        image_params = ImageToolParams(
            prompt=params.prompt,
            size=ImageSize(params.size),
            quality=ImageQuality(params.quality),
            style=ImageStyle(params.style),
            n=params.n,
            story_id=params.story_id
        )

        # Generate image
        result = await image_tool.execute(image_params)

        return {
            "success": result.success,
            "prompt": params.prompt,
            "images_generated": len(result.image_urls) if result.success else 0,
            "image_urls": result.image_urls if result.success else [],
            "local_paths": result.local_paths if result.success else [],
            "error": result.error
        }

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt": params.prompt
        }


@mcp.tool()
def create_story(params: StoryCreationParams) -> dict[str, Any]:
    """
    Create and save a news story to the newspaper system.

    Handles story creation with metadata, categorization, and storage.
    Automatically generates story ID and publication date.

    Args:
        params: Story creation parameters

    Returns:
        Dictionary with story creation results
    """
    try:
        import uuid
        from datetime import datetime

        logger.info(f"Creating story: {params.title}")

        # Generate story ID
        story_id = f"story-{uuid.uuid4().hex[:8]}"

        # Convert field to enum
        try:
            field_enum = ReporterField(params.field.lower())
        except ValueError:
            field_enum = ReporterField.TECHNOLOGY  # Default fallback

        # Create story object
        story = Story(
            id=story_id,
            title=params.title,
            field=field_enum,
            status=StoryStatus.PUBLISHED,
            content=params.content,
            published_date=datetime.now().isoformat(),
            summary=params.summary,
            keywords=params.keywords,
            author="mcp-agent",
            section=NewspaperSection.MAIN,
            priority=StoryPriority.NORMAL,
            front_page=False,
            sources=[],  # Convert sources if needed
        )

        # Publish story
        story_store.publish(story)

        return {
            "success": True,
            "story_id": story_id,
            "title": params.title,
            "field": params.field,
            "word_count": story.word_count,
            "published_date": story.published_date
        }

    except Exception as e:
        logger.error(f"Story creation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "title": params.title
        }
```

## 🔧 Step 10: Add MCP Resources and Prompts

### 10.1 Add Resource Support

Add after the tools:

```python
# ============== MCP RESOURCES ==============

@mcp.resource("file:///data/newspaper.json", name="newspaper_data", description="Current newspaper stories and content")
async def get_newspaper_data() -> str:
    """Get current newspaper data including all published stories."""
    try:
        newspaper_path = Path("libs/common/data/newspaper.json")
        if not newspaper_path.exists():
            return "No newspaper data available"

        content = newspaper_path.read_text(encoding="utf-8")
        return content

    except Exception as e:
        logger.error(f"Failed to read newspaper data: {e}")
        return f"Error reading newspaper data: {str(e)}"


@mcp.resource("file:///logs/mcp_server.log")
async def get_server_logs() -> str:
    """Get MCP server logs for debugging and monitoring."""
    try:
        log_path = Path("logs/mcp_server.log")
        if not log_path.exists():
            return "No server logs available"

        # Return last 1000 lines to avoid overwhelming output
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-1000:])

    except Exception as e:
        logger.error(f"Failed to read server logs: {e}")
        return f"Error reading server logs: {str(e)}"
```

### 10.2 Add MCP Prompts

Add prompt support:

```python
# ============== MCP PROMPTS ==============

@mcp.prompt(
    name="story_research_prompt",
    description="Generate a comprehensive research prompt for story development",
    tags={"research", "story", "journalism"}
)
def story_research_prompt(topic: str, field: str = "technology") -> str:
    """Generate a research prompt for comprehensive story development.

    Args:
        topic: The story topic to research
        field: The field/category for the story

    Returns:
        A structured research prompt
    """
    return f"""
Research and develop a comprehensive news story on the following topic:

**Topic**: {topic}
**Field**: {field}

Please conduct thorough research using the available tools and create a well-structured news article that includes:

1. **Background Research**:
   - Use DuckDuckGo search to find recent news and developments
   - Search for expert opinions and industry analysis
   - Look for relevant statistics and data points

2. **Source Verification**:
   - Scrape key websites for detailed information
   - Cross-reference information from multiple sources
   - Identify authoritative sources and expert quotes

3. **Content Development**:
   - Write a compelling headline and summary
   - Develop the story with proper journalistic structure
   - Include relevant quotes and data points
   - Ensure factual accuracy and balanced reporting

4. **Visual Content**:
   - Generate appropriate images to accompany the story
   - Consider infographics or charts if relevant

5. **Publication Preparation**:
   - Format the story for publication
   - Add appropriate keywords and metadata
   - Save the final story to the newspaper system

Focus on creating high-quality, engaging journalism that informs and educates readers about {topic} in the {field} field.
"""


@mcp.prompt(
    name="content_analysis_prompt",
    description="Analyze content for quality, accuracy, and editorial standards",
    tags={"analysis", "editorial", "quality"}
)
def content_analysis_prompt(content: str, content_type: str = "news_story") -> str:
    """Generate a content analysis prompt for editorial review.

    Args:
        content: The content to analyze
        content_type: Type of content being analyzed

    Returns:
        A structured analysis prompt
    """
    return f"""
Please analyze the following {content_type} for editorial quality and standards:

**Content to Analyze**:
{content[:500]}{"..." if len(content) > 500 else ""}

**Analysis Framework**:

1. **Accuracy & Factual Verification**:
   - Check for factual claims that need verification
   - Identify any potential misinformation or bias
   - Assess source credibility and attribution

2. **Editorial Standards**:
   - Evaluate writing quality and clarity
   - Check grammar, spelling, and style consistency
   - Assess headline effectiveness and accuracy

3. **Journalistic Integrity**:
   - Review balance and fairness in reporting
   - Check for proper attribution and quotes
   - Evaluate ethical considerations

4. **Audience Engagement**:
   - Assess readability and accessibility
   - Evaluate story structure and flow
   - Consider visual elements and formatting

5. **Recommendations**:
   - Suggest specific improvements
   - Identify areas needing additional research
   - Recommend publication readiness or revision needs

Provide a comprehensive editorial assessment with specific, actionable feedback.
"""
```

## 🧪 Step 11: Enhanced Testing

### 11.1 Update Test Script

Update `mcp_server/test_server.py` to include new tools:

```python
async def test_image_generation():
    """Test image generation functionality."""
    print("🎨 Testing Image Generation...")

    image_params = {
        "prompt": "A modern newsroom with journalists working on computers",
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid",
        "n": 1,
        "story_id": "test-story-001"
    }

    result = await mcp.call_tool("generate_image", image_params)
    print(f"Image generation: {result['success']}, images: {result.get('images_generated', 0)}")


async def test_story_creation():
    """Test story creation functionality."""
    print("📰 Testing Story Creation...")

    story_params = {
        "title": "MCP Server Integration Success",
        "content": "The BobTimes newsroom has successfully integrated MCP server technology, enabling unified tool access across all editorial systems. This breakthrough allows for more efficient content creation and improved workflow automation.",
        "field": "technology",
        "summary": "BobTimes integrates MCP server for improved editorial workflow",
        "keywords": ["MCP", "technology", "newsroom", "automation"],
        "sources": []
    }

    result = mcp.call_tool("create_story", story_params)
    print(f"Story creation: {result['success']}, story ID: {result.get('story_id', 'N/A')}")


async def test_resources():
    """Test MCP resources."""
    print("📚 Testing MCP Resources...")

    # Test newspaper data resource
    newspaper_data = await mcp.get_resource("file:///data/newspaper.json")
    print(f"Newspaper data available: {len(newspaper_data) > 0}")


# Add to main function:
await test_image_generation()
print()
await test_story_creation()
print()
await test_resources()
```

## 🚀 Step 12: Production Configuration

### 12.1 Create Server Configuration

Create `mcp_server/config.py`:

```python
"""Configuration for BobTimes MCP Server."""

from pathlib import Path
from pydantic import BaseModel


class MCPServerConfig(BaseModel):
    """Configuration for MCP server."""

    # Server settings
    transport: str = "stdio"  # stdio, http, or sse
    host: str = "localhost"
    port: int = 8002

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/mcp_server.log"

    # Tool settings
    max_search_results: int = 20
    max_file_size_mb: int = 10
    allowed_file_extensions: list[str] = [".txt", ".md", ".json", ".csv", ".html"]

    # Paths
    data_dir: Path = Path("data")
    images_dir: Path = Path("data/images")
    logs_dir: Path = Path("logs")

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
```

### 12.2 Create Production Startup Script

Create `mcp_server/start_server.py`:

```python
"""Production startup script for BobTimes MCP Server."""

import sys
import logging
from pathlib import Path

from config import MCPServerConfig
from main import mcp


def setup_logging(config: MCPServerConfig):
    """Setup logging configuration."""
    config.logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Start the MCP server with production configuration."""
    config = MCPServerConfig()
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting BobTimes MCP Server...")
    logger.info(f"Transport: {config.transport}")
    logger.info(f"Host: {config.host}:{config.port}")

    try:
        if config.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(
                transport=config.transport,
                host=config.host,
                port=config.port
            )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## 📋 Step 13: Documentation and Usage

### 13.1 Create MCP Server README

Create `mcp_server/README.md`:

```markdown
# BobTimes MCP Server

Unified MCP server providing all BobTimes tools through the Model Context Protocol.

## Features

- **Search Tools**: DuckDuckGo web, image, video, news search
- **Web Scraping**: Playwright-based content extraction
- **YouTube Tools**: Channel search and video transcription
- **File Operations**: Read, write, delete files
- **Image Generation**: AI-powered image creation
- **Story Management**: Create and publish news stories
- **Resources**: Access to newspaper data and logs
- **Prompts**: Research and analysis prompt templates

## Quick Start

```bash
# Install dependencies
uv sync

# Start server (stdio mode)
uv run python main.py

# Start server (production mode)
uv run python start_server.py

# Run tests
uv run python test_server.py
```

## Tool Reference

### Search Tools

- `duckduckgo_search`: Web search with multiple types
- `web_scraper`: Extract content from URLs
- `youtube_search`: Search channels and transcribe videos

### Content Tools

- `generate_image`: Create AI images
- `create_story`: Publish news stories

### File Tools

- `read_file`: Read file contents
- `write_file`: Write content to files
- `delete_file`: Remove files

## Configuration

See `config.py` for server configuration options.

## Integration

This server is designed to be used by BobTimes agents through MCP client connections.
```

## 🎉 Final Steps and Validation

### 13.1 Complete Installation

```bash
# From project root
uv sync

# Test the complete setup
cd mcp_server
uv run python test_server.py
```

### 13.2 Verify All Tools Work

The test should show successful results for:
- ✅ DuckDuckGo search (text and news)
- ✅ Web scraping functionality
- ✅ File operations (read, write, delete)
- ✅ Image generation
- ✅ Story creation
- ✅ Resource access

## 🏆 Achievement Unlocked!

You've successfully created a comprehensive MCP server that unifies all BobTimes tools! This achievement includes:

**🔧 Technical Implementation:**
- Complete MCP server with FastMCP
- 8+ tool categories converted to MCP
- Resource and prompt support
- Production-ready configuration

**🛠️ Tool Coverage:**
- Search: DuckDuckGo, YouTube, Web scraping
- Content: Story creation, Image generation
- Files: Read, write, delete operations
- Editorial: Content management and publishing

**📈 Benefits Achieved:**
- Unified tool interface via MCP protocol
- Standardized agent-tool communication
- Scalable and maintainable architecture
- Foundation for agent MCP integration

Your newsroom is now powered by a modern, protocol-based tool system that sets the stage for the next evolution: converting agents to use only MCP tools!
