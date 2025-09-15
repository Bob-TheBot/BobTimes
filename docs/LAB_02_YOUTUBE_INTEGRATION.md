# Lab 2: Adding YouTube Data Source with Transcription

Welcome to Lab 2! In this lab, you'll extend the BobTimes system by adding YouTube as a new data source and implementing a sophisticated memory-first architecture. You'll learn how to create custom tools, integrate external APIs, and build intelligent content retrieval systems.


## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Create a YouTube tool that accepts user-provided channel lists
- ✅ Extract recent video titles for topic discovery
- ✅ Implement selective video transcription for content generation
- ✅ Integrate the tool into the reporter agent workflow
- ✅ Test the complete workflow: channels → topics → transcripts → content
- ✅ Understand how to extend the system with new data sources
- ✅: Implement SharedMemoryStore for cross-agent communication
- ✅: Create fetch_from_memory tool for intelligent content retrieval
- ✅: Build memory-first reporter workflow with smart topic matching
- ✅ **NEW**: Implement unified tool results architecture

## 📋 Prerequisites

- ✅ Completed Lab 1 (Basic setup and configuration)
- ✅ Working DevContainer or local development environment
- ✅ YouTube Data API v3 key (free from Google Cloud Console)
- ✅ Understanding that transcripts come from YouTube's auto-generated captions
- ✅ Basic understanding of the tool architecture

## 🧠 Memory-First Architecture Overview

This lab introduces a sophisticated **memory-first architecture** that revolutionizes how the BobTimes system handles content discovery and retrieval:

### 🔄 The Memory-First Workflow

1. **Topic Discovery Phase** (Editor → Reporter)
   - Editor assigns reporters to find trending topics in their field
   - Reporters use search, YouTube, and other tools to discover topics
   - All discovered content is stored in **SharedMemoryStore** with topic names as keys

2. **Topic Assignment Phase** (Editor → Reporter)
   - Editor assigns specific topics to reporters for story writing
   - Topic names might be slightly different from memory keys (e.g., "AI Tools" vs "Best AI Tools To Create Viral Content")

3. **Smart Content Retrieval Phase** (Reporter)
   - Reporter sees all available memory topics for their field
   - Reporter intelligently matches their assigned topic to the best memory key
   - Reporter uses **fetch_from_memory** tool to retrieve pre-validated content
   - If no memory match, reporter falls back to fresh search/scrape

### 🎯 Key Benefits

- **🚀 Performance**: Instant content retrieval from memory vs. slow API calls
- **💰 Cost Efficiency**: Reuse validated content instead of repeated API calls
- **🎯 Accuracy**: Pre-validated content ensures high-quality sources
- **🧠 Intelligence**: Smart topic matching handles naming variations
- **🔄 Fallback**: Graceful degradation to fresh search when needed

## 🎯 How this integration works (current implementation)

- Transcripts are fetched via youtube-transcript-api and embedded directly into each YouTubeVideo object as a transcript field.
- The YouTube reporter tool exposes a single operation: topics. It discovers recent videos from configured channels and only returns videos that have transcripts. The UnifiedToolResult includes topics_extracted, sources with transcript content, and metadata.
- Channels can be specified as full YouTube URLs or channel IDs. @username URLs are supported and will be resolved to channel IDs using the YouTube Data API (this may consume a small amount of quota).

## 🚀 Step 1: Setup YouTube Data API

### 1.1 Get YouTube Data API Key (FREE)

**💰 Cost Information:**
- YouTube Data API v3 is **FREE** up to 10,000 quota units per day
- Each video search costs ~100 units, each video details request costs ~1 unit
- This lab's usage will be well within the free tier (typically 50-100 videos per day)
- **No billing account required** for the free tier

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Create a new project or select existing one (free)

2. **Enable YouTube Data API v3**
   ```bash
   # In Google Cloud Console:
   # 1. Go to "APIs & Services" > "Library"
   # 2. Search for "YouTube Data API v3"
   # 3. Click "Enable" (no billing required for free tier)
   ```

3. **Create API Credentials**
   ```bash
   # In Google Cloud Console:
   # 1. Go to "APIs & Services" > "Credentials"
   # 2. Click "Create Credentials" > "API Key"
   # 3. Copy the generated API key
   # 4. (Recommended) Restrict the key to YouTube Data API v3 for security
   ```

**📊 Quota Usage Estimate for This Lab:**
- Search 3 channels × 5 videos each = ~300 quota units
- Get video details for 15 videos = ~15 quota units
- **Total per run: ~315 units** (well within 10,000 daily limit)
- You can run the lab **30+ times per day** within the free tier

**🆓 Transcript Access:**
- YouTube transcript fetching is **completely FREE** and has no quotas
- Uses the `youtube-transcript-api` Python library (no Google API key needed for transcripts)
- Only the video metadata requires the YouTube Data API key

**🔄 Alternative: No-API Approach (Optional)**
If you prefer not to use Google APIs, you can modify the lab to:
- Use hardcoded video IDs instead of channel searches
- Focus only on transcript extraction (completely free)
- Skip video metadata and channel information
- This approach requires **zero API keys** and has **zero costs**

### 1.2 Configure API Keys and Settings

Add your API key to the secrets file and channel configuration to the environment file:

```yaml
# In libs/common/secrets.yaml
llm_providers:
  # ... existing providers ...

# YouTube Data Source Configuration (API Key only)
youtube:
  api_key: "your_youtube_data_api_key_here"
```

### 1.3 Update Environment Configuration

```bash
# In libs/.env.development
# Add YouTube configuration
YOUTUBE_ENABLED=true
YOUTUBE_MAX_VIDEOS_PER_CHANNEL=1
YOUTUBE_DAYS_BACK=7
YOUTUBE_CONCURRENT_REQUESTS=3
YOUTUBE_REQUEST_DELAY_MS=100

# YouTube Channel Configuration (comma-separated channel URLs or IDs)
# You can use full channel URLs (preferred for UX) or direct channel IDs.
# Note: @username URLs will be resolved to channel IDs via the YouTube API and may consume a small amount of quota.

# Technology channels (examples)
YOUTUBE_CHANNELS_TECHNOLOGY="https://www.youtube.com/@LinusTechTips,https://www.youtube.com/@MKBHD,https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw"

# Science channels (examples)
YOUTUBE_CHANNELS_SCIENCE="https://www.youtube.com/@Kurzgesagt,https://www.youtube.com/@veritasium,https://www.youtube.com/channel/UCHnyfMqiRRG1u-2MsSQLbXA"

# Economics channels (examples)
YOUTUBE_CHANNELS_ECONOMICS="https://www.youtube.com/@economics-explained,https://www.youtube.com/@PatrickBoyleOnFinance"

# Sports channels (optional examples)
YOUTUBE_CHANNELS_SPORTS="https://www.youtube.com/@ESPN,https://www.youtube.com/@NBA"
```

## 🛠️ Step 2: Create YouTube Tool Infrastructure

### 2.1 Install Required Dependencies

```bash
# In DevContainer terminal
uv add youtube-transcript-api
uv add google-api-python-client
# Note: We only use YouTube's built-in transcription API, no additional models needed
```

### 2.2 Create YouTube Data Models

We use typed Pydantic models (no Any types) and embed transcripts directly into video objects. Key structures (excerpt):

<augment_code_snippet path="libs/common/utils/youtube_models.py" mode="EXCERPT">
````python
class YouTubeVideo(BaseModel):
    url: str = Field(description="Full YouTube URL")
    thumbnail_url: str | None = Field(None, description="Video thumbnail URL")
    transcript: "VideoTranscript | None" = Field(None, description="Video transcript if available")
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_models.py" mode="EXCERPT">
````python
class YouTubeSearchResult(BaseModel):
    success: bool
    videos: list[YouTubeVideo] = Field(default_factory=lambda: [])
````
</augment_code_snippet>

Guidelines:
- Use YouTubeField (StrEnum) for fields (technology/science/economics/sports)
- Prefer enums over strings in service/tool APIs
- Keep models small, focused, and fully typed
## 🔧 Step 3: Implement YouTube Tool

### 3.1 Create YouTube Service

Create the file `libs/common/utils/youtube_service.py`:

<augment_code_snippet path="libs/common/utils/youtube_service.py" mode="EXCERPT">
````python
def get_channels_for_field(
    self,
    field: YouTubeField,
    subsection: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None
) -> list[str]:
    ...
````
</augment_code_snippet>

- Environment variable priority:
  1) YOUTUBE_CHANNELS_{FIELD}_{SUBSECTION} (if subsection provided)
  2) YOUTUBE_CHANNELS_{FIELD} (fallback)
- Accept both YouTube URLs and channel IDs; @usernames will be resolved to channel IDs (may use minimal API quota)

<augment_code_snippet path="libs/common/utils/youtube_service.py" mode="EXCERPT">
````python
# Attach transcripts directly to videos when requested
if params.include_transcripts:
    for video in videos:
        transcript = await self._get_video_transcript(
            video.video_id, params.transcription_method
        )
        video.transcript = transcript
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_service.py" mode="EXCERPT">
````python
def _extract_channel_id(self, channel_input: str) -> str | None:
    # Supports /channel/UC..., /@username, /c/..., /user/...
    ...
````
</augment_code_snippet>

This is the foundation of our YouTube integration. Now let's create the reporter tool and integrate it into the agent workflow.

## 🧠 Step 4: Implement Memory-First Architecture

Before creating the YouTube tool, let's implement the memory-first architecture that will revolutionize content retrieval.

### 4.1 Create SharedMemoryStore

The SharedMemoryStore is the heart of our memory-first architecture. Create the file `libs/common/agents/shared_memory_store.py`:

Workshop-style overview (snippets, not full code):

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
class SharedMemoryEntry(BaseModel):
    topic_name: str
    field: str
    sources: list[StorySource]
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
class SharedMemoryStore:
    def store_memory(self, topic_name: str, field: str, sources: list[StorySource]) -> None:
        ...  # merge by normalized topic
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
# Global instance accessor
_shared_memory_store: SharedMemoryStore | None = None

def get_shared_memory_store() -> SharedMemoryStore:
    global _shared_memory_store
    _shared_memory_store = _shared_memory_store or SharedMemoryStore()
    return _shared_memory_store
````
</augment_code_snippet>

Guidelines:
- Normalize topic names consistently for keys
- Append sources on updates; do not overwrite existing ones
- Keep memory in-process (no persistence) for this lab

### 4.2 Create fetch_from_memory Tool

Now create the intelligent memory retrieval tool. Add this to `libs/common/agents/reporter_agent/reporter_tools.py`:

Workshop-style overview (snippets, not full code):

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
class FetchFromMemoryParams(BaseModel):
    topic_key: str  # exact key from memory
    field: str      # technology/economics/science
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
class FetchFromMemoryTool(BaseTool):
    name = "fetch_from_memory"
    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> FetchFromMemoryResult:
        ...
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
# Register tool
self.tools["fetch_from_memory"] = FetchFromMemoryTool()
````
</augment_code_snippet>

Guidelines:
- Agents should choose an exact topic key from available memory topics in their context
- Use this tool before search/scrape when a matching topic exists
- The result includes a content summary and source count

### 4.3 Update Reporter Prompt for Memory-First Workflow

Update the reporter prompt to show available memory topics and encourage using fetch_from_memory. Modify `libs/common/agents/reporter_agent/reporter_prompt.py`:

Workshop-style overview (snippet, not full code):

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_prompt.py" mode="EXCERPT">
````python
# Show available memory topics for this field
memory_store = get_shared_memory_store()
available = memory_store.list_topics(field=task.field.value)

# Encourage using fetch_from_memory first when a close match exists
instructions.extend([
    "🧠 MEMORY-FIRST: If your topic matches a memory topic, use fetch_from_memory",
])
````
</augment_code_snippet>

### 4.4 Update Reporter Executor for Memory Integration

The reporter executor already handles `fetch_from_memory` tool results automatically. The key change is to **remove automatic memory injection** and let agents choose when to use the tool.

**Important**: Remove any automatic memory loading from `reporter_executor.py`:

Workshop-style overview (snippet):

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_executor.py" mode="EXCERPT">
````python
# Do NOT auto-inject memory into state during WRITE_STORY
# Let the agent decide to call fetch_from_memory when appropriate
````
</augment_code_snippet>

**The memory integration works through the tool system**:
1. Agent sees available memory topics in prompt
2. Agent chooses to use `fetch_from_memory` tool with exact topic key
3. Tool handler automatically injects retrieved sources into state
4. Agent uses injected sources for story writing

## 🔧 Step 6: Create YouTube Reporter Tool

### 4.1 Create YouTube Reporter Tool

Create the file `libs/common/utils/youtube_tool.py`:

Workshop-style overview (snippets, not full code):

<augment_code_snippet path="libs/common/utils/youtube_tool.py" mode="EXCERPT">
````python
class YouTubeToolParams(BaseModel):
    field: YouTubeField | None = None
    subsection: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None
    operation: str = "topics"  # Only 'topics'
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_tool.py" mode="EXCERPT">
````python
async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
    ...  # returns UnifiedToolResult
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_tool.py" mode="EXCERPT">
````python
# Merge field/subsection channels with user input
if params.field is not None:
    ids = self.youtube_service.get_channels_for_field(params.field, params.subsection)
    channel_ids.extend(ids)
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_tool.py" mode="EXCERPT">
````python
# Use only videos with transcripts
videos = [v for v in result.videos if v.transcript is not None]
topics = [_normalize_topic_name(v.title) for v in videos]
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/utils/youtube_tool.py" mode="EXCERPT">
````python
# Return unified, typed result
return UnifiedToolResult(
    success=True,
    operation="topics",
    topics_extracted=topics,
    sources=sources,
    metadata={"channels_searched": len(channel_ids)}
)
````
</augment_code_snippet>

Usage examples:
- {"field": "technology", "subsection": "ai_tools", "operation": "topics", "days_back": 7}
- {"channel_ids": ["https://www.youtube.com/@3Blue1Brown"], "operation": "topics"}

Guidelines:
- Provide channel URLs or IDs (prefer URLs for readability)
- Use subsection for targeted channel sets
- Only "topics" is supported; transcripts are embedded in video objects and used automatically
- Keep days_back reasonable (1–30) for performance
### 4.2 Register YouTube Tool with Reporter Registry

Now we need to add the YouTube tool to the reporter tool registry. Update the file `libs/common/agents/reporter_agent/reporter_tools.py`:

Workshop-style overview (snippets, not full code):

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
from utils.youtube_tool import YouTubeReporterTool  # import

self.tools["youtube_search"] = YouTubeReporterTool(self.config_service)
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_executor.py" mode="EXCERPT">
````python
# Unified handling (YouTube uses UnifiedToolResult)
if tool_result.success and hasattr(result, "sources") and hasattr(result, "topic_source_mapping"):
    await self._handle_unified_tool_result(result, state)
````
</augment_code_snippet>

## 🔧 Step 7: Memory-First Architecture Implementation

The memory-first architecture is now fully implemented and integrated into the system. The key components work together to provide intelligent content retrieval and cross-agent communication.







## 🔧 Step 8: YouTube Integration Implementation

The YouTube integration is now fully implemented and ready for use in the news generation workflow.



## 🔧 Step 6: Configure Field-Specific and Subsection-Specific Channels

### 6.1 Channel Configuration Hierarchy

The YouTube integration supports two levels of channel configuration:

1. **Field-Level Configuration** (fallback): `YOUTUBE_CHANNELS_{FIELD}`
2. **Subsection-Level Configuration** (priority): `YOUTUBE_CHANNELS_{FIELD}_{SUBSECTION}`

When a reporter agent requests YouTube content for a specific subsection, the system will:
1. First look for subsection-specific channels (e.g., `YOUTUBE_CHANNELS_TECHNOLOGY_AI_TOOLS`)
2. Fall back to field-level channels if no subsection-specific channels are configured
3. Return empty list if neither is configured

### 6.2 Update Channel Configuration

Add more comprehensive channel lists to your `.env.development` file:

```bash
# Enhanced YouTube channel configuration in libs/.env.development
# Technology channels (comma-separated)
YOUTUBE_CHANNELS_TECHNOLOGY="UC_x5XG1OV2P6uZZ5FSM9Ttw,UCXuqSBlHAE6Xw-yeJA0Tunw,UC4QZ_LsYcvcq7qOsOhpAX4A,UCld68syR8Wi-GY_n4CaoJGA,UCVls1GmFKf6WlTraIb_IaJg"

# Science channels (comma-separated)
YOUTUBE_CHANNELS_SCIENCE="UCsXVk37bltHxD1rDPwtNM8Q,UC6nSFpj9HTCZ5t-N3Rm3-HA,UCHnyfMqiRRG1u-2MsSQLbXA,UC7_gcs09iThXybpVgjHZ_7g,UCtYLUTtgS3k1Fg4y5tAhLbw"

# Economics channels (comma-separated)
YOUTUBE_CHANNELS_ECONOMICS="UCfM3zsQsOnfWNUppiycmBuw,UC0p5jTq6Xx_DosDFxVXnWaQ,UCZ4AMrDcNrfy3X6nsU8-rPg,UC-uWLJbmXH1KbCPHmtG-3MQ,UCcunJy13-KFJNd_DkqkUeEg"

# Sports channels (comma-separated, optional)
YOUTUBE_CHANNELS_SPORTS="UCqFMzb-4AUf6WAIbhOJ5P8w,UCWWbZ8z9GwvbR_7NUjNYJdw"

# Channel name mapping for reference (comments only):
# UC_x5XG1OV2P6uZZ5FSM9Ttw = Google Developers
# UCXuqSBlHAE6Xw-yeJA0Tunw = Linus Tech Tips
# UC4QZ_LsYcvcq7qOsOhpAX4A = CodeBullet
# UCld68syR8Wi-GY_n4CaoJGA = Brodie Robertson (Linux/Tech)
# UCVls1GmFKf6WlTraIb_IaJg = DistroTube
# UCsXVk37bltHxD1rDPwtNM8Q = Kurzgesagt – In a Nutshell
# UC6nSFpj9HTCZ5t-N3Rm3-HA = Vsauce
# UCHnyfMqiRRG1u-2MsSQLbXA = Veritasium
# UC7_gcs09iThXybpVgjHZ_7g = PBS Space Time
# UCtYLUTtgS3k1Fg4y5tAhLbw = Statquest
# UCfM3zsQsOnfWNUppiycmBuw = Economics Explained
# UC0p5jTq6Xx_DosDFxVXnWaQ = Ben Felix
# UCZ4AMrDcNrfy3X6nsU8-rPg = Economics in Many Lessons
# UC-uWLJbmXH1KbCPHmtG-3MQ = Marginal Revolution University
# UCcunJy13-KFJNd_DkqkUeEg = CrashCourse Economics
```

### 6.2 Test Different Fields

```bash
# Test each field separately
python -c "
import asyncio
from utils.youtube_tool import YouTubeReporterTool, YouTubeToolParams
from core.config_service import ConfigService

async def test_field(field_name):
    tool = YouTubeReporterTool(ConfigService())
    params = YouTubeToolParams(field=field_name, days_back=7, max_videos_per_channel=1)
    result = await tool.execute(params)
    print(f'{field_name.upper()}: {result.videos_found} videos, {result.transcripts_obtained} transcripts')

async def main():
    for field in ['technology', 'science', 'economics']:
        await test_field(field)

asyncio.run(main())
"
```

## 📰 Step 7: Integrate with News Generation

### 7.1 Test YouTube Tool in Reporter Agent

Create a test script to see how the YouTube tool works within the reporter agent workflow:

```python
# Create file: test_youtube_reporter.py
"""Test YouTube integration with reporter agent."""

import asyncio
from agents.agent_factory import AgentFactory
from agents.models.task_models import ReporterTask
from agents.task_execution_service import TaskExecutionService
from agents.types import ReporterField, TaskType
from core.config_service import ConfigService


async def test_youtube_in_reporter():
    """Test YouTube tool integration with reporter agent."""
    config_service = ConfigService()
    task_service = TaskExecutionService(config_service)
    factory = AgentFactory(config_service, task_service)

    # Create a technology reporter
    reporter = factory.create_reporter(ReporterField.TECHNOLOGY)

    # Create a task that should use YouTube data
    task = ReporterTask(
        name=TaskType.WRITE_STORY,
        field=ReporterField.TECHNOLOGY,
        description="Write a story about recent developments in AI and machine learning",
        guidelines="Use YouTube videos and transcripts to find the latest discussions and announcements"
    )

    print("🤖 Starting reporter task with YouTube integration...")
    result = await task_service.execute_reporter_task(task)

    if result.success:
        print("✅ Task completed successfully!")
        print(f"📄 Story Title: {result.story.title}")
        print(f"📝 Story Length: {len(result.story.content)} characters")
        print(f"🔗 Sources: {len(result.story.sources)} sources")

        # Check if YouTube was used
        youtube_sources = [s for s in result.story.sources if 'youtube.com' in s.url]
        if youtube_sources:
            print(f"🎥 YouTube sources used: {len(youtube_sources)}")
            for source in youtube_sources[:2]:  # Show first 2
                print(f"   - {source.title}: {source.url}")
        else:
            print("⚠️  No YouTube sources were used")
    else:
        print(f"❌ Task failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_youtube_in_reporter())
```

### 7.2 Generate Full Newspaper with YouTube Integration

```bash
# Generate a newspaper that should now include YouTube sources
python generate_newpaper.py

# Check the output for YouTube sources
grep -r "youtube.com" data/newspapers/ || echo "No YouTube sources found"
```

## 🔧 Step 8: Advanced Configuration and Optimization

### 8.1 Add Caching Configuration

Update your environment configuration to include caching:

```bash
# In libs/.env.development
# YouTube caching settings (cache API responses to avoid rate limits)
YOUTUBE_CACHE_ENABLED=true
YOUTUBE_CACHE_DURATION_HOURS=6  # How long to cache video/channel data before refreshing
YOUTUBE_CACHE_MAX_VIDEOS=100    # Maximum number of videos to cache per field

# Performance settings
YOUTUBE_CONCURRENT_REQUESTS=3   # Max concurrent API requests
YOUTUBE_REQUEST_DELAY_MS=100    # Delay between requests to avoid rate limiting
```

### 8.2 Create Channel Management Script

Create a utility script to manage and validate YouTube channels:

Workshop-style overview (snippets, not full script):

<augment_code_snippet mode="EXCERPT">
````python
# manage_youtube_channels.py (optional utility)
config = ConfigService()
youtube = YouTubeService(config)
for field in [YouTubeField.TECHNOLOGY, YouTubeField.SCIENCE, YouTubeField.ECONOMICS]:
    ids = youtube.get_channels_for_field(field)
    print(field.value, len(ids), "channels")
````
</augment_code_snippet>

Guidelines:
- Use YouTubeService.get_channels_for_field(field, subsection) to inspect configured channels
- For quick spot checks, fetch 1–2 recent videos per channel and print titles
- Keep this as an optional local utility; do not commit API keys or outputs

### 9.2 Run Complete Test Suite

```bash
# Validate and check all configured channels
python manage_youtube_channels.py
```

## 🎯 Step 9: Implementation Verification

### 9.1 Complete Integration Status

The YouTube integration has been fully implemented and verified to work correctly with all system components.

### 9.2 Integration Verification

All components have been successfully integrated and are working correctly together.

## 🎉 Step 10: Lab Completion and Next Steps

### 10.1 Verify Lab Completion

Check that you have successfully:

- YouTube tool is visible in ReporterToolRegistry and available to reporter agents
- YouTube API key is set in libs/common/secrets.yaml (youtube.api_key)
- Channel lists are defined in libs/.env.development using URLs or IDs
- Running the workflow produces topics and sources that include YouTube transcripts

### 10.2 Understanding What You Built

You have successfully:

1. **🔧 Created a YouTube Data Source**
   - Integrated YouTube Data API v3
   - Implemented video search and metadata extraction
   - Added transcript fetching capabilities

2. **🛠️ Built a Reporter Tool**
   - Created `YouTubeReporterTool` following the BaseTool pattern
   - Implemented structured parameters and results
   - Added field-specific channel configuration

3. **🔗 Integrated with Agent Workflow**
   - Registered the tool with `ReporterToolRegistry`
   - Made it available to all reporter agents
   - Enabled automatic usage in story research

4. **⚙️ Configured Multi-Field Support**
   - Set up technology, science, and economics channels
   - Implemented custom channel override capability
   - Added comprehensive error handling

### 10.3 Key Implementation Details

**🔧 Architecture Decisions:**
- **Direct Channel IDs**: Using channel IDs instead of usernames to minimize API calls
- **Separate Libraries**: `youtube-transcript-api` for transcripts (quota-free) + `google-api-python-client` for metadata
- **Pydantic Models**: Full type safety with `YouTubeVideo`, `VideoTranscript`, `YouTubeToolParams`, etc.
- **Source Tracking**: Automatic `StorySource` creation for news attribution

**🎯 Key Learning Points:**
- **Tool Architecture**: How to extend the system with new data sources
- **API Integration**: Working with external APIs and handling authentication
- **Configuration Management**: Using the ConfigService for secrets and settings
- **Agent Integration**: How tools are discovered and used by agents
- **Error Handling**: Robust error handling for external service dependencies

### 10.4 Next Steps and Extensions

**Potential Enhancements:**
1. **Enhanced Transcript Processing**: Add transcript cleaning and summarization
2. **Video Analysis**: Add video thumbnail analysis using vision models
3. **Trending Detection**: Implement trending topic detection across channels
4. **Content Filtering**: Add content quality and relevance filtering
5. **Caching Layer**: Implement Redis caching for better performance
6. **Real-time Updates**: Add webhook support for real-time video notifications
7. **Multi-language Support**: Handle transcripts in different languages

**Other Data Sources to Add:**
- Twitter/X API integration
- Reddit API for community discussions
- RSS feed aggregation
- News API integration
- Podcast transcript analysis

Note: This lab does not expose a separate "transcribe" operation. Transcripts are embedded directly in YouTubeVideo objects by the service and used automatically by the topics flow. Use channel URLs/IDs and the topics operation for discovery; transcripts are attached under the hood.

## Summary and next steps

This lab added a YouTube data source that the reporter can use for topic discovery. Key pieces now in place:

- YouTube Data API integration via YouTubeService
- Transcripts embedded into YouTubeVideo and used automatically by the topics flow
- Field and optional subsection channel configuration from environment
- YouTubeReporterTool returning UnifiedToolResult and registered in the reporter tool registry
- Shared memory integration for cross-agent topic and source reuse
- Source attribution wired into story outputs

## What you covered

- Added a YouTube data source and tool following the project patterns
- Configured channels by field and optional subsection via environment
- Embedded transcripts and used them automatically in the topics flow
- Handled basic errors and rate limits in a pragmatic way

Next steps (optional):
- Expand and maintain your channel lists per field/subsection
- Tune days_back and max_videos_per_channel for your workflow
- Review logs for missing transcripts or channel resolution issues and adjust thresholds

### How sources appear in stories

- YouTube videos included by the reporter tool show up under the story's sources with URL, title, and a brief summary. No extra wiring is required; the tool returns StorySource objects that the pipeline preserves.

## 🔄 PART 2: UNIFIED ARCHITECTURE & MEMORY MANAGEMENT

**⚠️ WORKSHOP PARTICIPANTS: This is the NEW section you need to implement!**

The following sections represent significant architectural improvements that you'll implement to create a unified, memory-based topic management system.

### 🎯 Architecture Overview

The new architecture introduces:

1. **UnifiedToolResult**: All tools (Search, YouTube, Scraper) return the same standardized result format
2. **SharedMemoryStore**: In-memory storage for topics with validated content
3. **Content Validation**: Ensures topics have sufficient content before storage
4. **Content Enhancement**: Automatic fallback to scraper when search results lack content
5. **Memory-Based Editor**: Editor selects only from topics with validated content

### 📋 Implementation Roadmap

You'll implement these changes in the following order:

1. ✅ **Create UnifiedToolResult Class**
2. ✅ **Implement SharedMemoryStore**
3. ✅ **Update All Tools to Return UnifiedToolResult**
4. ✅ **Add Content Validation & Enhancement**
5. ✅ **Create Memory-Based Editor Tools**
6. ✅ **Update Reporter Executor for Unified Handling**

---

## 🔧 Step 1: Create UnifiedToolResult Class

### 1.1 Understanding the Problem

Currently, each tool returns different result formats:
- **SearchTool** → `SearchToolResult`
- **YouTubeTool** → `YouTubeToolResult`
- **ScraperTool** → `ScraperToolResult`

This creates complexity in the reporter executor that needs tool-specific handling.

### 1.2 Create the Unified Result Class

**📁 File: `libs/common/agents/tools/base_tool.py`**

Add this class to the base_tool module to avoid circular imports:

```python
class UnifiedToolResult(BaseModel):
    """Unified result class for all reporter tools."""
    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="Type of operation performed (search, youtube, scrape)")
    query: str | None = Field(default=None, description="Original query/URL used")

    # Core data that all tools provide
    sources: list[Any] = Field(default_factory=list, description="Sources found/scraped")
    topics_extracted: list[str] = Field(default_factory=list, description="Topics extracted from results")
    topic_source_mapping: dict[str, Any] = Field(default_factory=dict, description="Mapping of topics to their source data")

    # Optional metadata (tool-specific)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Tool-specific metadata")

    # Results and error handling
    summary: str | None = Field(default=None, description="Summary of operation results")
    error: str | None = Field(default=None, description="Error message if operation failed")
```

### 1.3 Key Design Decisions

- **`sources`**: All tools provide `StorySource` objects for consistent handling
- **`topics_extracted`**: Normalized topic names for consistent storage
- **`topic_source_mapping`**: Maps topics to their source data for memory storage
- **`metadata`**: Tool-specific data (e.g., YouTube video metadata)
- **`operation`**: Identifies which tool generated the result

---

## 🧠 Step 2: Implement SharedMemoryStore

### 2.1 Understanding Memory Management

The SharedMemoryStore provides:
- **Cross-agent communication**: Topics stored by reporters, accessed by editors
- **Content validation**: Only topics with sufficient content are stored
- **In-memory storage**: No file persistence needed (runtime only)
- **Field-based organization**: Topics organized by field (technology, science, economics)

### 2.2 Create SharedMemoryStore

**📁 File: `libs/common/agents/shared_memory_store.py`**

Workshop-style overview (snippets, not full code):

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
class SharedMemoryEntry(BaseModel):
    topic_name: str
    field: str
    sources: list[StorySource]
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
class SharedMemoryStore:
    def store_memory(self, topic_name: str, field: str, sources: list[StorySource]) -> None:
        ...  # merge/append sources, update timestamps
````
</augment_code_snippet>

<augment_code_snippet path="libs/common/agents/shared_memory_store.py" mode="EXCERPT">
````python
# Global accessor
_shared_memory_store: SharedMemoryStore | None = None

def get_shared_memory_store() -> SharedMemoryStore:
    global _shared_memory_store
    _shared_memory_store = _shared_memory_store or SharedMemoryStore()
    return _shared_memory_store
````
</augment_code_snippet>

Guidelines:
- Normalize topic names consistently
- Avoid duplicate sources by URL when merging
- Keep store in-memory for this lab

### 2.3 Implementation Tasks

**🔨 Your Tasks:**
1. Implement `store_memory()` method with duplicate prevention
2. Implement `get_memory()` and `get_memories_by_field()` methods
3. Add proper logging for memory operations
4. Handle memory updates (merge sources when topic already exists)

---

## 🔄 Step 3: Update Tools to Return UnifiedToolResult

### 3.1 Add Topic Normalization

First, add a normalization function to ensure consistent topic naming:

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
def _normalize_topic_name(topic: str) -> str:
    return topic.strip().title()
````
</augment_code_snippet>

Guideline: keep a single normalization helper and reuse it across tools.

Add this function to all tool files that extract topics.

### 3.2 Update SearchTool

**📁 File: `libs/common/agents/reporter_agent/reporter_tools.py`**

**Key Changes:**
1. Change return type from `SearchToolResult` to `UnifiedToolResult`
2. Add content validation to topic extraction
3. Create topic-source mapping with content quality flags

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_tools.py" mode="EXCERPT">
````python
def _extract_topics_from_search_results(results: list[Any], query: str) -> tuple[list[str], dict[str, Any]]:
    topics, mapping = [], {}
    for r in results:
        if getattr(r, "title", None):
            name = _normalize_topic_name(r.title)
            mapping[name] = {"source": StorySource(url=getattr(r, "url", "")), "query": query}
            topics.append(name)
    return topics, mapping
````
</augment_code_snippet>

Guidelines:
- Normalize titles to derive topics; keep mapping minimal
- Mark items needing scrape when content is too short; enrich later in executor

**Update the execute method:**

```python
async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
    # ... existing search logic ...

    # Extract topics from results for SharedMemoryStore
    topics_extracted, topic_source_mapping = _extract_topics_from_search_results(results, params.query)

    # Convert to story sources for unified interface
    story_sources = ReporterToolRegistry.convert_search_results_to_sources(cleaned_results)

    return UnifiedToolResult(
        success=True,
        operation="search",
        query=params.query,
        sources=story_sources,
        topics_extracted=topics_extracted,
        topic_source_mapping=topic_source_mapping,
        metadata={
            "search_type": params.search_type.value,
            "results_count": len(cleaned_results),
            "time_limit": params.time_limit
        },
        summary=f"Found {len(cleaned_results)} search results for '{params.query}'",
        error=None
    )
```

### 3.3 Update YouTubeTool

**📁 File: `libs/common/utils/youtube_tool.py`**

**Key Changes:**
1. Import `UnifiedToolResult` from `agents.tools.base_tool`
2. Update return type and all return statements
3. Add topic normalization
4. Move tool-specific data to metadata field

```python
from agents.tools.base_tool import BaseTool, UnifiedToolResult

async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
    # ... existing logic ...

    # Extract topics only from videos with transcripts - normalize for consistency
    topics = [_normalize_topic_name(video.title) for video in videos_with_transcripts]

    # ... create sources and topic mapping ...

    return UnifiedToolResult(
        success=True,
        operation="youtube",
        query=f"channels: {', '.join(params.channel_ids)}",
        sources=sources,
        topics_extracted=topics,
        topic_source_mapping=topic_source_mapping,
        metadata={
            "operation": "topics",
            "videos_found": len(videos_with_transcripts),
            "channels_searched": len(params.channel_ids),
            "video_metadata": video_metadata,
            "detailed_results": [...]  # Move detailed results to metadata
        },
        summary=summary,
        error=None
    )
```

### 3.4 Update ScraperTool

**📁 File: `libs/common/agents/reporter_agent/reporter_tools.py`**

Similar updates for the scraper tool to return `UnifiedToolResult`.

**🔨 Your Tasks:**
1. Update all three tools to return `UnifiedToolResult`
2. Add topic normalization to YouTube and Search tools
3. Ensure all tools create proper topic-source mappings
4. Move tool-specific data to the metadata field

---

## ✅ Step 4: Add Content Validation & Enhancement

### 4.1 Understanding Content Enhancement

The problem: Search results often only contain snippets, not full content. We need to:
1. **Validate content quality** when storing topics
2. **Use scraper as fallback** when search content is insufficient
3. **Filter out topics** that don't have enough content even after enhancement

### 4.2 Update Reporter Executor

**📁 File: `libs/common/agents/reporter_agent/reporter_executor.py`**

**Replace tool-specific handling with unified handling:**

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_executor.py" mode="EXCERPT">
````python
# Unified handling (avoid tool-specific branches)
if tool_result.success and getattr(result, "sources", None):
    await self._handle_unified_tool_result(result, state)
````
</augment_code_snippet>

Guideline: remove tool-specific branches in favor of a single unified path.

**Create the unified handler:**

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_executor.py" mode="EXCERPT">
````python
async def _handle_unified_tool_result(self, result: Any, state: ReporterState) -> None:
    # Optionally enhance search results, then save via ReporterStateManager
    ReporterStateManager.save_sources_with_topics(
        state=state,
        sources=result.sources,
        topic_source_mapping=getattr(result, "topic_source_mapping", {})
    )
````
</augment_code_snippet>

Guidelines:
- Enhance search results with scrape only when needed
- Keep handler tool-agnostic; rely on UnifiedToolResult fields

### 4.3 Implement Content Enhancement

<augment_code_snippet path="libs/common/agents/reporter_agent/reporter_executor.py" mode="EXCERPT">
````python
async def _enhance_search_content(self, sources: list[Any], mapping: dict[str, Any], state: ReporterState):
    # Skeleton: scrape when needed; keep only sufficiently detailed items
    return sources, mapping
````
</augment_code_snippet>

Guidelines:
- Scrape only when content length is below a threshold
- Replace source if scraped content is strictly better; otherwise keep original

**🔨 Your Tasks:**
1. Implement the unified tool result handler
2. Create content enhancement logic with scraper fallback
3. Add content validation and filtering
4. Update ReporterStateManager to use SharedMemoryStore

---

## 📝 Step 5: Create Memory-Based Editor Tools

### 5.1 Understanding the New Editor Flow

**OLD Flow:**
1. Editor asks reporters to find topics
2. Reporters return topics (may lack content)
3. Editor assigns topics for writing
4. Reporters may fail due to insufficient content

**NEW Flow:**
1. Editor asks reporters to find topics
2. Reporters store validated topics in SharedMemoryStore
3. Editor selects topics FROM MEMORY (guaranteed to have content)
4. Editor assigns selected topics for writing
5. Reporters retrieve content from memory and write stories

### 5.2 Create SelectTopicsFromMemoryTool

**📁 File: `libs/common/agents/editor_agent/editor_tools.py`**

```python
class MemoryTopicInfo(BaseModel):
    """Information about a topic stored in memory."""
    topic_name: str = Field(description="The topic name")
    summary: str = Field(description="Summary of the topic content")
    sources_count: int = Field(description="Number of sources available for this topic")
    content_length: int = Field(description="Total content length available")

class SelectTopicsFromMemoryParams(BaseModel):
    """Parameters for selecting topics from SharedMemoryStore."""
    field: ReporterField = Field(description="Field to select topics from")
    max_topics: int = Field(default=5, description="Maximum number of topics to select")

class SelectTopicsFromMemoryResult(BaseModel):
    """Result of selecting topics from memory."""
    available_topics: list[MemoryTopicInfo] = Field(description="All available topics in memory for this field")
    selected_topics: list[str] = Field(description="Topics selected by the editor")
    reasoning: str = Field(description="Editor's reasoning for topic selection")
    success: bool = Field(description="Whether the selection was successful")

class SelectTopicsFromMemoryTool(BaseTool):
    """Select topics from SharedMemoryStore for story assignment."""

    name: str = "select_topics_from_memory"
    description: str = """
Select topics from SharedMemoryStore that have validated content for story writing.
Only topics with sufficient content are available for selection.

Parameters:
- field: ReporterField (technology/economics/science) to select topics from
- max_topics: Maximum number of topics to select (default: 5)

Returns: List of available topics with summaries and your selected topics
"""

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> SelectTopicsFromMemoryResult:
        """Select topics from SharedMemoryStore for assignment."""
        # TODO: Implement topic selection from memory
        # 1. Get SharedMemoryStore instance
        # 2. Get all topics for the specified field
        # 3. Create MemoryTopicInfo objects with summaries
        # 4. Return available topics and selected topics
        pass
```

### 5.3 Update AssignTopicsTool

**Update the parameters to work with memory-selected topics:**

```python
class AssignTopicsParams(BaseModel):
    """Parameters for creating topic assignments from memory topics."""
    field: ReporterField  # Field to assign topics from
    selected_topics: list[str]  # Topics selected from memory to assign
    priority: StoryPriority = StoryPriority.MEDIUM  # Priority level for assignments
```

**Update the execute method:**

```python
async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> AssignTopicsResult:
    """Create assignments from memory-selected topics for spawning reporters."""
    # Use the selected topics from memory
    if not params.selected_topics:
        return AssignTopicsResult(
            reasoning=f"No topics selected for field {params.field.value}",
            assignments=[],
            success=False
        )

    topics_to_assign = params.selected_topics
    # ... rest of assignment logic ...
```

**🔨 Your Tasks:**
1. Implement `SelectTopicsFromMemoryTool` with proper memory access
2. Update `AssignTopicsParams` to use selected topics instead of available topics
3. Create meaningful topic summaries from source content
4. Add proper error handling for empty memory

---

## 🔄 Step 6: Update Reporter for Memory-Based Writing

### 6.1 Understanding Tool-Based Memory Retrieval

When a reporter receives a `WRITE_STORY` task, it should:
1. **See available memory topics** in the prompt for its field
2. **Choose to use fetch_from_memory tool** if a relevant topic exists
3. **Receive memory sources** automatically injected after tool usage
4. **Fall back to research tools** if no relevant memory exists

### 6.2 Reporter Prompt Shows Available Topics

**📁 File: `libs/common/agents/reporter_agent/reporter_prompt.py`**

**The prompt builder shows available memory topics:**

```python
def _build_write_story_instructions(self, state: ReporterState) -> list[str]:
    """Build instructions for WRITE_STORY tasks with memory-first approach."""
    instructions = []

    # Show available memory topics for this field
    from agents.shared_memory_store import get_shared_memory_store
    memory_store = get_shared_memory_store()
    available_topics = memory_store.list_topics(field=state.current_task.field.value)

    if available_topics:
        instructions.extend([
            "🧠 MEMORY-FIRST APPROACH:",
            f"Available topics in memory for {state.current_task.field.value}:",
            *[f"  - '{topic}'" for topic in available_topics],
            "",
            "If your assigned topic matches any memory topic, use fetch_from_memory tool first.",
            "This will give you pre-validated, high-quality sources for your story.",
            ""
        ])

    instructions.extend([
        f"📝 WRITE STORY: {state.current_task.topic}",
        f"Target: {state.current_task.target_word_count} words",
        "Strategy: Use fetch_from_memory if available, otherwise research fresh content"
    ])

    return instructions
```

### 6.3 Agent Decision-Making Process

**The agent now makes intelligent decisions:**

1. **Sees Available Topics**: Prompt shows all memory topics for the field
2. **Smart Matching**: Agent matches assigned topic to best memory key
3. **Tool Usage**: Agent calls `fetch_from_memory` with exact topic key
4. **Automatic Injection**: Tool handler injects sources into state
5. **Story Writing**: Agent uses injected sources for content

**Example Agent Reasoning:**
```
Agent sees:
- Assigned topic: "Meta's New AI-Powered Brand Tools"
- Available memory: ["Meta Has New Tools For Brand And Performance Goals, With A Focus On Ai (Of Course)"]
- Decision: Use fetch_from_memory with the available key
- Result: Gets pre-validated sources automatically injected
```

**Benefits of Tool-Based Approach:**
- **Intelligent Matching**: LLM handles topic name variations
- **Selective Retrieval**: Only fetches relevant content when needed
- **Performance**: No unnecessary memory loading
- **Flexibility**: Agent can choose research vs memory based on context

**🔨 Your Tasks:**
1. Implement memory source injection for WRITE_STORY tasks
2. Update prompts to instruct LLM behavior based on memory availability
3. Add proper logging for memory operations
4. Ensure topic normalization consistency between storage and retrieval

---

## 🎯 Summary & Next Steps

### ✅ What You've Implemented

1. **UnifiedToolResult**: Standardized result format across all tools
2. **SharedMemoryStore**: In-memory topic storage with content validation
3. **Content Enhancement**: Automatic scraper fallback for insufficient search content
4. **Memory-Based Editor**: Topic selection only from validated memory content
5. **Topic Normalization**: Consistent naming across all components

### 🔄 Architecture Benefits

- **Simplified Tool Handling**: Single unified handler instead of tool-specific logic
- **Content Guarantee**: Editor only sees topics with validated content
- **Automatic Enhancement**: Poor search results automatically enhanced via scraping
- **Cross-Agent Communication**: Topics stored by reporters, accessed by editors
- **Consistent Naming**: Topic normalization prevents lookup failures

### 🚀 Implementation Ready

The unified architecture is now complete and ready for production use with all components working together seamlessly.

### 🎓 Key Learning Outcomes

- **Unified Interfaces**: How to create consistent APIs across different tools
- **Memory Management**: Cross-agent data sharing patterns
- **Content Validation**: Ensuring data quality in agent workflows
- **Fallback Mechanisms**: Automatic content enhancement strategies
- **Editor Workflows**: Memory-based decision making for content selection

You now have a solid, unified architecture with memory-based topic management in place. It’s ready for iterative improvement and real-world usage.
✅ **Editorial Oversight**: Editors can review source quality and relevance
✅ **Compliance**: Proper attribution meets journalistic standards

### Integration with Existing Workflow

The YouTube tool seamlessly integrates with the existing source tracking system:

1. **Search Tool** → Web search results as sources
2. **Scraper Tool** → Scraped web pages as sources
3. **YouTube Tool** → Video content and transcripts as sources
4. **All Sources** → Automatically tracked in story metadata

**🔄 Ready for Lab 3?** The next lab will focus on advanced agent customization and workflow optimization!

---

## Final notes

- This lab focused on adding a YouTube data source and wiring it into the reporter flow using UnifiedToolResult and SharedMemoryStore.
- Transcripts are embedded in YouTubeVideo by the service and used automatically in the topics operation; there is no separate transcribe mode.
- Keep conclusions practical: expand channel lists as needed, monitor logs, and iterate on thresholds for performance and content quality.
