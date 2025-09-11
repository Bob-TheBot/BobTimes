# BobTimes AI Newspaper Workshop Guide

Welcome to the BobTimes AI Newspaper Generation System! This guide will help you quickly understand the application structure, workflow, and how to get started.

## 🏗️ Application Architecture

### Project Structure
```
bobtimes/
├── backend/                    # FastAPI backend server
│   ├── app/                   # API endpoints and services
│   └── run_news_cycle.py      # Alternative cycle runner
├── client/                    # React frontend (Vite + Tailwind)
├── libs/common/               # Shared libraries and core services
│   ├── agents/               # AI agent implementations
│   ├── core/                 # Core services (config, LLM, logging)
│   └── utils/                # Utility tools (scraping, search)
├── generate_newpaper.py       # Main newspaper generation script
└── justfile                   # Development commands
```

### Core Components

1. **Agents** (`libs/common/agents/`)
   - **EditorAgent**: Orchestrates news cycles, manages workflow
   - **ReporterAgent**: Researches topics and writes stories
   - **AgentFactory**: Creates and manages agent instances

2. **Core Services** (`libs/common/core/`)
   - **ConfigService**: Manages configuration and secrets
   - **LLMService**: Handles AI model interactions
   - **LoggingService**: Centralized logging

3. **Tools** (`libs/common/utils/`)
   - **DuckDuckGoSearchTool**: Web search functionality
   - **PlaywrightScraper**: Web content extraction
   - **ImageGenerationTool**: AI image creation

## 🔄 News Generation Workflow

### Complete Flow Overview
```
1. Initialize Services → 2. Create Editor Agent → 3. Start News Cycle
                                    ↓
8. Publish Stories ← 7. Review Stories ← 6. Collect Stories
                                    ↓
                     4. Collect Topics → 5. Assign Topics
```

### Detailed Workflow Steps

1. **Initialization**
   - Load configuration from `.env` files and `secrets.yaml`
   - Create LLM service with specified provider
   - Initialize newspaper service with editor agent

2. **Topic Collection**
   - Editor spawns reporter agents for each field (Technology, Science, Economics)
   - Reporters research trending topics using search tools
   - Topics are collected and evaluated by the editor

3. **Topic Assignment**
   - Editor selects best topics from collected suggestions
   - Topics are assigned to specific reporters for story development

4. **Story Writing**
   - Reporters conduct deep research using web scraping
   - Stories are written with proper journalistic structure
   - Images are generated or sourced for each story

5. **Editorial Review**
   - Editor reviews each story for quality and accuracy
   - Stories can be approved, rejected, or sent back for revision
   - Feedback is provided for improvements

6. **Publication**
   - Approved stories are published to the newspaper
   - Stories are organized by section and priority
   - Final newspaper content is generated

## 🤖 Agent Creation and Management

### Agent Types

**EditorAgent**
- **Role**: Orchestrates the entire news cycle
- **Responsibilities**: Topic evaluation, story review, publication decisions
- **Tools**: CollectTopicsTool, ReviewStoryTool, PublishStoryTool
- **Model**: Uses SLOW model for careful editorial decisions

**ReporterAgent**
- **Role**: Researches and writes news stories
- **Specialization**: Field-specific (Technology, Science, Economics)
- **Tools**: SearchTool, ScraperTool, ImageGenerationTool
- **Model**: Uses SLOW model for thorough research

### Agent Creation Process
```python
# Via AgentFactory
factory = AgentFactory(config_service, task_service)

# Create specialized reporter
reporter = factory.create_reporter(
    field=ReporterField.TECHNOLOGY,
    sub_section=TechnologySubSection.AI_TOOLS
)

# Create editor
editor = factory.create_editor()
```

## 🛠️ Tools Structure and Usage

### Tool Architecture
All tools inherit from `BaseTool` and implement:
- **Structured Parameters**: Pydantic models for type safety
- **Async Execution**: Non-blocking operations
- **LLM Integration**: Optional LLM service for AI-powered tools

### Available Tools

**Search Tools**
- `DuckDuckGoSearchTool`: Web search with text, image, news, video results
- Supports filtering by region, time, safety settings

**Content Tools**
- `AsyncPlaywrightScraper`: Advanced web scraping with JavaScript support
- Handles dynamic content, interactive elements, and content extraction

**Image Tools**
- `ImageGenerationTool`: AI-powered image creation
- Supports multiple providers (OpenAI DALL-E, Google Imagen)
- Automatic image saving and processing

### Tool Usage Example
```python
# In reporter agent
search_tool = DuckDuckGoSearchTool()
search_params = TextSearchParams(
    query="latest AI developments 2024",
    max_results=10,
    region=DDGRegion.US
)
results = await search_tool.execute(search_params)
```

## 📊 Data Sources and Processing

### External Data Sources
1. **Web Search**: DuckDuckGo API for current information
2. **Web Scraping**: Playwright for detailed content extraction
3. **Image Generation**: AI models for visual content
4. **News APIs**: Real-time news data integration

### Data Flow
```
External Sources → Tools → Agents → Processing → Storage → API → Frontend
```

### Content Processing
- **Text Cleaning**: Remove ads, navigation, extract main content
- **Image Processing**: Resize, optimize, convert to base64
- **Content Validation**: Check quality, relevance, accuracy
- **Structured Storage**: Pydantic models for type safety

## 🔧 Configuration and Secrets

### Environment Configuration
**Location**: `libs/.env.development` (or `.env.production`)
```bash
# Application settings
APP_ENV=development
PROJECT_NAME=Bob Times
HOST=0.0.0.0
PORT=9200

# LLM Configuration
LLM_PROVIDER=aws_bedrock  # or openai, anthropic, gemini
IMAGE_GEN_PROVIDER=openai

# Logging
LOG_LEVEL=INFO
LOG_JSON_FORMAT=False
```

### Secrets Management
**Location**: `libs/common/secrets.yaml` (copy from `secrets.example.yaml`)
```yaml
# LLM Provider API Keys
llm_providers:
  openai:
    api_key: "sk-your_openai_api_key_here"
  anthropic:
    api_key: "sk-ant-your_anthropic_api_key_here"
  gemini:
    api_key: "your_gemini_api_key_here"
  aws_bedrock:
    api_key: "your_aws_bedrock_bearer_token_here"
```

### Configuration Access
```python
# In your code
config_service = ConfigService()
api_key = config_service.get("llm_providers.openai.api_key")
log_level = config_service.get("LOG_LEVEL", "INFO")
```

## 🚀 Running the Application

### Prerequisites

**🐳 DevContainer Setup (Recommended for Workshop):**
1. **VS Code** with Dev Containers extension installed
2. **Docker Desktop** running on your machine
3. **API Keys** for LLM providers

**🖥️ Local Development Setup (Alternative):**
1. **Python 3.13+** with `uv` package manager
2. **Node.js 18+** for frontend
3. **API Keys** for LLM providers

### Setup Steps

#### Option A: DevContainer Setup (Recommended)

1. **Open in DevContainer**
   ```bash
   # In VS Code, open the project folder
   # Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
   # Type "Dev Containers: Reopen in Container"
   # Wait for container to build and setup to complete
   ```

2. **Verify DevContainer Setup**
   ```bash
   # The setup script runs automatically, but you can verify:
   uv --version     # Should show uv package manager
   just --version   # Should show just command runner
   node --version   # Should show Node.js LTS
   python --version # Should show Python 3.11+
   ```

#### Option B: Local Development Setup

1. **Install Dependencies**
   ```bash
   # Install Python dependencies
   just sync  # or: uv sync

   # Install frontend dependencies
   cd client && npm install
   ```

2. **Configure Secrets (Both DevContainer and Local)**
   ```bash
   # Copy example secrets file
   cp libs/secrets.example.yaml libs/common/secrets.yaml

   # Edit with your API keys
   nano libs/common/secrets.yaml
   # OR in VS Code: code libs/common/secrets.yaml
   ```

3. **DevContainer-Specific Notes**

   **🐳 If using DevContainer:**
   - All dependencies are automatically installed during container setup
   - Ports are automatically forwarded (3000, 9200, 8000, 51273)
   - Use `aws-env` alias to load AWS credentials from secrets.yaml
   - Development tools (just, uv, docker) are pre-installed
   - Zsh shell with useful aliases is configured

   **Verify DevContainer Setup:**
   ```bash
   # Check essential tools are available
   uv --version && just --version && python --version

   # Check you're in the container
   echo $DEVCONTAINER  # Should show container info

   # Check port forwarding in VS Code "Ports" tab
   ```

4. **Run the Application**
   ```bash
   # Option 1: Run both backend and frontend
   just run
   
   # Option 2: Run separately
   just run-backend  # Backend on port 9200
   just run-client   # Frontend on port 3000
   
   # Option 3: Generate newspaper directly
   python generate_newpaper.py
   ```

### Development Commands
```bash
# Code quality
just lint          # Check code style
just format        # Format code
just pyright       # Type checking

# Testing
python generate_newpaper.py  # Generate full newspaper
```

## 🎯 Key Features for Workshop

### 1. **Modular Agent System**
- Easy to extend with new agent types
- Field-specific specialization
- Tool-based architecture

### 2. **Flexible LLM Integration**
- Support for multiple providers
- Speed-based model selection (FAST/SLOW)
- Centralized configuration

### 3. **Real-time Content Generation**
- Live web scraping and search
- AI-powered image generation
- Dynamic topic discovery

### 4. **Professional Workflow**
- Editorial review process
- Quality control mechanisms
- Structured content organization

## 🔍 Exploring the Code

### Start Here
1. **`generate_newpaper.py`** - Main entry point
2. **`libs/common/agents/editor_agent/`** - Workflow orchestration
3. **`libs/common/agents/reporter_agent/`** - Content creation
4. **`backend/app/services/newspaper_service.py`** - API integration

### Key Models
- **`agents/types.py`** - Enums and type definitions
- **`agents/models/`** - Data structures for stories, tasks, cycles
- **`core/llm_service.py`** - LLM provider configurations

## 📋 Model Definitions and Data Structures

### Core Data Models

**Story Models** (`agents/models/story_models.py`)
```python
class StoryDraft(BaseModel):
    title: str
    content: str
    summary: str
    field: ReporterField
    sources: list[StorySource]
    suggested_images: list[StoryImage]
    created_at: datetime

class PublishedStory(BaseModel):
    story_id: str
    title: str
    content: str
    author: str
    section: NewspaperSection
    priority: StoryPriority
    published_at: datetime
```

**Task Models** (`agents/models/task_models.py`)
```python
class ReporterTask(BaseModel):
    name: TaskType  # FIND_TOPICS, WRITE_STORY
    field: ReporterField
    sub_section: Optional[SubSection]
    description: str
    topic: Optional[str]  # For story writing tasks
```

**Cycle Models** (`agents/models/cycle_models.py`)
```python
class NewsCycle(BaseModel):
    cycle_id: str
    cycle_status: CycleStatus
    published_stories: list[PublishedStory]
    submissions: list[StorySubmission]
    editorial_decisions: list[EditorialDecision]
    start_time: datetime
```

### Defining New Models

**Best Practices**:
1. **Use Pydantic BaseModel** for all data structures
2. **Include type hints** for all fields
3. **Add field descriptions** using `Field(description="...")`
4. **Use enums** instead of string literals
5. **Include validation** where appropriate

**Example: Custom Model**
```python
from pydantic import BaseModel, Field
from datetime import datetime
from agents.types import ReporterField

class CustomStoryMetrics(BaseModel):
    """Metrics for story performance analysis."""
    story_id: str = Field(description="Unique story identifier")
    field: ReporterField = Field(description="Story field/category")
    word_count: int = Field(gt=0, description="Number of words in story")
    research_time: float = Field(ge=0, description="Time spent researching (minutes)")
    quality_score: float = Field(ge=0, le=10, description="Editorial quality rating")
    created_at: datetime = Field(default_factory=datetime.now)
```

## 🔧 Advanced Configuration

### LLM Provider Setup

**AWS Bedrock** (Recommended)
```yaml
# In secrets.yaml
llm_providers:
  aws_bedrock:
    api_key: "your_bearer_token_here"
    region_name: "us-west-2"
```

**OpenAI**
```yaml
llm_providers:
  openai:
    api_key: "sk-your_key_here"
    organization: "org-your_org_id"  # Optional
```

**Environment Variables**
```bash
# In .env.development
LLM_PROVIDER=aws_bedrock
IMAGE_GEN_PROVIDER=openai

# Model preferences
LLM_FAST_MODEL=claude-sonnet-4
LLM_SLOW_MODEL=claude-opus-4
```

### Custom Tool Development

**Creating a New Tool**
```python
from agents.tools.base_tool import BaseTool
from pydantic import BaseModel
from core.llm_service import ModelSpeed

class CustomToolParams(BaseModel):
    query: str
    max_results: int = 10

class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "Description of what this tool does"
    params_model = CustomToolParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST):
        # Validate params
        if not isinstance(params, CustomToolParams):
            raise ValueError("Invalid parameters")

        # Tool logic here
        result = await self._perform_custom_operation(params.query)

        return {
            "success": True,
            "data": result,
            "query": params.query
        }
```

### Extending Agent Capabilities

**Adding New Reporter Fields**
```python
# In agents/types.py
class ReporterField(StrEnum):
    ECONOMICS = "economics"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    SPORTS = "sports"  # New field
    POLITICS = "politics"  # New field

# Create corresponding sub-sections
class SportsSubSection(StrEnum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    OLYMPICS = "olympics"
```

**Custom Agent Specialization**
```python
class SportsReporterAgent(ReporterAgent):
    def __init__(self, sub_section: SportsSubSection = None, **kwargs):
        super().__init__(
            field=ReporterField.SPORTS,
            sub_section=sub_section,
            **kwargs
        )

        # Add sports-specific tools
        self.tool_registry.add_tool(SportsDataTool())
        self.tool_registry.add_tool(ScoreTrackerTool())
```

## 🐛 Troubleshooting Common Issues

### Configuration Problems
```bash
# Check environment loading
python -c "from core.config_service import ConfigService; c=ConfigService(); print(c.get_environment())"

# Verify secrets loading
python -c "from core.config_service import ConfigService; c=ConfigService(); print('API key loaded:', bool(c.get('llm_providers.openai.api_key')))"
```

### LLM Service Issues
```bash
# Test LLM connectivity
python -c "
from core.config_service import ConfigService
from app.dependencies import get_default_llm_service
config = ConfigService()
llm = get_default_llm_service(config)
print('LLM service created successfully')
"
```

### Agent Execution Problems
- **Check iteration limits**: Agents have max_iterations (default: 15)
- **Verify tool availability**: Ensure all required tools are registered
- **Monitor token usage**: Large contexts may hit model limits
- **Review error logs**: Check `logs/` directory for detailed errors

### Performance Optimization
- **Use FAST models** for simple tasks (topic collection, basic research)
- **Use SLOW models** for complex tasks (story writing, editorial review)
- **Implement caching** for repeated searches
- **Batch operations** where possible

## 📚 Learning Resources

### Code Exploration Path
1. **Start Simple**: Run `generate_newpaper.py` and observe the flow
2. **Understand Agents**: Study `EditorAgent` and `ReporterAgent` classes
3. **Explore Tools**: Look at search and scraping implementations
4. **Modify Workflow**: Try changing story requirements or fields
5. **Add Features**: Implement new tools or agent capabilities

### Key Files to Study
- `libs/common/agents/agent.py` - Base agent framework
- `libs/common/agents/editor_agent/editor_executor.py` - Workflow logic
- `libs/common/agents/reporter_agent/reporter_executor.py` - Content creation
- `libs/common/core/llm_service.py` - AI model integration
- `backend/app/services/newspaper_service.py` - API service layer

