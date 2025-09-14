# Lab 1: Getting Started with BobTimes AI Newspaper

Welcome to your first hands-on lab! In this lab, you'll set up the BobTimes system, configure your AI models, and generate your first AI newspaper. This lab is designed for complete newcomers to the repository.

## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Set up the development environment
- ✅ Configure LLM providers and models (fast/slow)
- ✅ Set up image generation with fallback options
- ✅ Generate your first minimal newspaper
- ✅ Understand the basic workflow and data models
- ✅ Make agents aware of the current date

## 📋 Prerequisites

**🐳 DevContainer Setup (Recommended for Workshop):**
- **VS Code** with Dev Containers extension installed
- **Docker Desktop** running on your machine
- **API Keys** for at least one LLM provider
- **Basic terminal/command line knowledge**

**🖥️ Local Development Setup (Alternative):**
- **Python 3.13+** installed
- **Node.js 18+** for frontend (optional for this lab)
- **API Keys** for at least one LLM provider
- **Basic terminal/command line knowledge**

## 🚀 Step 1: Environment Setup

### 1.1 DevContainer Setup (Recommended)

**If you're in the workshop, use this method:**

```bash
# 1. Open VS Code in the project directory
# 2. Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
# 3. Type "Dev Containers: Reopen in Container"
# 4. Select the option and wait for container to build
# 5. The setup will run automatically (takes 2-3 minutes)
```

**Verify DevContainer Setup:**
```bash
# Check that everything is installed
uv --version      # Should show uv package manager
just --version    # Should show just command runner
node --version    # Should show Node.js LTS
python --version  # Should show Python 3.11+
docker --version  # Should show Docker

# Check project dependencies are installed
ls -la            # Should see all project files
which python      # Should point to container Python
```

### 1.2 Local Development Setup (Alternative)

**If you're NOT using DevContainer:**

```bash
# Navigate to the project directory (if not already there)
cd bobtimes

# Install Python dependencies using uv
just sync
# OR manually: uv sync

# Verify installation
python --version  # Should be 3.13+
uv --version      # Should show uv version
```

### 1.2 Verify Project Structure

```bash
# Check key directories exist
ls -la
# You should see: backend/, client/, libs/, generate_newpaper.py, justfile
```

### 1.4 DevContainer-Specific Notes

**🐳 If you're using DevContainer:**

The DevContainer automatically:
- ✅ Installs all Python dependencies via `uv sync`
- ✅ Installs Node.js dependencies via `npm install`
- ✅ Sets up proper networking with `host.docker.internal`
- ✅ Forwards ports: 3000 (frontend), 9200 (backend), 8000, 51273
- ✅ Configures zsh shell with useful aliases
- ✅ Installs development tools (just, docker, aws-cli)

**Important DevContainer Commands:**
```bash
# Check if you're in DevContainer
echo $DEVCONTAINER  # Should show "true" or container info

# Use these aliases (pre-configured in DevContainer)
aws-env             # Load AWS credentials from secrets.yaml
c                   # Clear terminal (alias for clear)
reload              # Reload zsh configuration

# Access host services (if needed)
ping host.docker.internal  # Should work from container
```

**Port Access:**
- Backend API: `http://localhost:9200` (auto-forwarded)
- Frontend: `http://localhost:3000` (auto-forwarded)
- All ports are automatically forwarded to your host machine

## 🔧 Step 2: Configuration Setup

### 2.1 Create Your Secrets File

```bash
# Copy the example secrets file
cp libs/secrets.example.yaml libs/common/secrets.yaml

# Edit the secrets file with your API keys
nano libs/common/secrets.yaml
# OR use your preferred editor: code libs/common/secrets.yaml
```

### 2.2 Configure LLM Providers

Choose **ONE** of the following providers based on what API keys you have:

#### Option A: OpenAI (Recommended for beginners)
```yaml
# In libs/common/secrets.yaml
llm_providers:
  openai:
    api_key: "sk-your_actual_openai_api_key_here"
    organization: ""  # Optional
```

#### Option B: Google Gemini (Free tier available)
```yaml
# In libs/common/secrets.yaml
llm_providers:
  gemini:
    api_key: "your_actual_gemini_api_key_here"
```

#### Option C: Anthropic Claude
```yaml
# In libs/common/secrets.yaml
llm_providers:
  anthropic:
    api_key: "sk-ant-your_actual_anthropic_api_key_here"
```

#### Option D: AWS Bedrock (Advanced)
```yaml
# In libs/common/secrets.yaml
llm_providers:
  aws_bedrock:
    api_key: "your_aws_bedrock_bearer_token_here"
```

### 2.3 Set Environment Variables

Create or edit the environment file:

```bash
# Copy example environment file
cp libs/.env.example libs/.env.development

# Edit environment settings
nano libs/.env.development
```

**Configure based on your chosen provider:**

#### For OpenAI:
```bash
# In libs/.env.development
LLM_PROVIDER=openai
IMAGE_GEN_PROVIDER=openai

# Model selection (these are the actual model names)
LLM_FAST_MODEL=gpt-5-mini      # Fast, cheaper model
LLM_SLOW_MODEL=gpt-5           # Slow, higher quality model
```

#### For Google Gemini:
```bash
# In libs/.env.development
LLM_PROVIDER=gemini
IMAGE_GEN_PROVIDER=google

# Model selection
LLM_FAST_MODEL=gemini/gemini-2.0-flash-exp
LLM_SLOW_MODEL=gemini/gemini-2.5-pro-preview-03-25
```

#### For Anthropic:
```bash
# In libs/.env.development
LLM_PROVIDER=anthropic
IMAGE_GEN_PROVIDER=google  # Anthropic doesn't do images, use Google

# Model selection
LLM_FAST_MODEL=claude-sonnet-4-20250514
LLM_SLOW_MODEL=claude-opus-4-1-20250805
```

#### For AWS Bedrock:
```bash
# In libs/.env.development
LLM_PROVIDER=aws_bedrock
IMAGE_GEN_PROVIDER=openai  # Bedrock doesn't do images, use OpenAI

# Model selection
LLM_FAST_MODEL=anthropic.claude-3-5-haiku-20241022-v1:0
LLM_SLOW_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
```

## 🖼️ Step 3: Image Generation Setup

### 3.1 Primary Option: OpenAI DALL-E

If you have OpenAI API access:
```yaml
# In libs/common/secrets.yaml (add to existing openai section)
llm_providers:
  openai:
    api_key: "sk-your_openai_api_key_here"
```

```bash
# In libs/.env.development
IMAGE_GEN_PROVIDER=openai
```

### 3.2 Fallback Option: Google Imagen (Free Tier)

If OpenAI is not available, use Google's free tier:

1. **Get Google API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Copy the key

2. **Configure Google:**
```yaml
# In libs/common/secrets.yaml
llm_providers:
  gemini:
    api_key: "your_google_ai_studio_api_key_here"
```

```bash
# In libs/.env.development
IMAGE_GEN_PROVIDER=google
```

## 🧪 Step 4: Test Your Configuration

### 4.1 Verify Configuration Loading

```bash
# Test configuration service
python -c "
from core.config_service import ConfigService
config = ConfigService()
print('Environment:', config.get_environment())
print('LLM Provider:', config.get('LLM_PROVIDER'))
print('Image Provider:', config.get('IMAGE_GEN_PROVIDER'))
print('API Key loaded:', bool(config.get('llm_providers.openai.api_key')))
"
```

### 4.2 Test LLM Service

```bash
# Test LLM connectivity
python -c "
from core.config_service import ConfigService
from backend.app.dependencies import get_default_llm_service
config = ConfigService()
llm = get_default_llm_service(config)
print('✅ LLM service created successfully')
print('Default model:', llm.default_model)
"
```

## 📰 Step 5: Generate Your First Newspaper

### 5.1 Simple Single-Topic Generation

Let's start with generating a single story to test everything works:

```bash
# Generate a single technology story
python -c "
import asyncio
from generate_newpaper import generate_single_issue
from agents.types import ReporterField, TechnologySubSection

# Generate one story about AI tools
exit_code = asyncio.run(
    generate_single_issue(
        field=ReporterField.TECHNOLOGY,
        sub_section=TechnologySubSection.AI_TOOLS
    )
)
print(f'Generation completed with exit code: {exit_code}')
"
```

### 5.2 Full Minimal Newspaper

Generate a complete newspaper with minimal sections:

```bash
# Run the main newspaper generator
python generate_newpaper.py
```

This will generate stories for:
- **Technology** (AI tools, software, hardware)
- **Science** (research, biology, space)
- **Economics** (markets, business, finance)

## 📊 Step 6: Understanding the Output

### 6.1 Check Generated Content

```bash
# Look for generated files
ls -la data/
ls -la data/images/

# Check logs
ls -la logs/
tail -f logs/app.log  # View recent logs
```

### 6.2 Examine the Data Models

The system uses these key models:

**Story Draft** (during creation):
```python
class StoryDraft(BaseModel):
    title: str
    content: str
    summary: str
    field: ReporterField  # TECHNOLOGY, SCIENCE, ECONOMICS
    sources: list[StorySource]
    suggested_images: list[StoryImage]
```

**Published Story** (final output):
```python
class PublishedStory(BaseModel):
    story_id: str
    title: str
    content: str
    author: str
    section: NewspaperSection
    priority: StoryPriority  # BREAKING, HIGH, MEDIUM, LOW
    published_at: datetime
```

## 🔍 Step 7: Understanding Fast vs Slow Models

### 7.1 Model Usage Patterns

The system automatically selects models based on task complexity:

**FAST Models** (cheaper, quicker):
- Topic collection and evaluation
- Basic research tasks
- Simple content processing
- Used by: `ModelSpeed.FAST`

**SLOW Models** (expensive, higher quality):
- Story writing and content creation
- Editorial review and decision making
- Complex analysis tasks
- Used by: `ModelSpeed.SLOW`

### 7.2 Model Configuration Examples

```python
# In agent configuration
config = AgentConfig(
    default_model_speed=ModelSpeed.SLOW,  # For reporters (quality writing)
    temperature=0.7
)

# In editor configuration  
config = AgentConfig(
    default_model_speed=ModelSpeed.SLOW,  # For editors (careful decisions)
    temperature=0.5
)
```

## ✅ Step 8: Verify Success

### 8.1 Check for Successful Generation

Look for these indicators:
- ✅ No error messages in terminal
- ✅ Generated images in `data/images/`
- ✅ Log files show successful completion
- ✅ Stories contain realistic content

### 8.2 Common Success Patterns

```bash
# Successful output should show:
# 🎯 GENERATION COMPLETE!
# 📊 Success rate: 3/3 (100.0%)
```

## 🐛 Troubleshooting Common Issues

### DevContainer-Specific Issues

### Issue 1: DevContainer Won't Start
```
Error: Failed to start container
```
**Solution:**
```bash
# Check Docker is running
docker --version

# Rebuild container from scratch
# In VS Code: Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### Issue 2: Dependencies Not Installed
```
Error: uv: command not found
```
**Solution:** The setup script may have failed. Run manually:
```bash
# In DevContainer terminal
bash scripts/setup-devcontainer.sh
```

### Issue 3: Port Forwarding Issues
```
Error: Cannot connect to localhost:9200
```
**Solution:** Check VS Code port forwarding:
```bash
# In VS Code, check "Ports" tab (next to Terminal)
# Ports 3000, 9200, 8000, 51273 should be forwarded
# If not, add them manually
```

### General Issues

### Issue 4: API Key Not Found
```
Error: OpenAI API key not found in configuration
```
**Solution:** Double-check your `secrets.yaml` file has the correct API key format.

### Issue 5: Model Not Found
```
Error: Model 'gpt-5' not found
```
**Solution:** Update your model names in `.env.development` to match available models.

### Issue 6: Image Generation Fails
```
Error: Provider does not support image generation
```
**Solution:** Set `IMAGE_GEN_PROVIDER=google` if OpenAI is not available.

### Issue 7: Permission Errors
```
Error: Permission denied writing to data/
```
**Solution:** Create directories manually:
```bash
mkdir -p data/images/generated
mkdir -p data/images/downloaded
chmod 755 data/
```

### Issue 8: DevContainer Python Path Issues
```
Error: Module not found
```
**Solution:** Ensure you're using the container Python:
```bash
# Check Python path
which python  # Should be /usr/local/python/current/bin/python
echo $PATH    # Should include /usr/local/python/current/bin

# If not, reload shell
source ~/.zshrc
# OR restart VS Code terminal
```

## 🐳 DevContainer Quick Verification

**If you're using DevContainer, run this verification script:**

```bash
# Create and run verification script
cat > verify_devcontainer.sh << 'EOF'
#!/bin/bash
echo "🔍 DevContainer Verification Script"
echo "=================================="

# Check if we're in DevContainer
if [ -n "$DEVCONTAINER" ] || [ -f /.dockerenv ]; then
    echo "✅ Running in DevContainer"
else
    echo "❌ Not in DevContainer (this is OK if running locally)"
fi

# Check essential tools
echo -n "uv: "; uv --version 2>/dev/null && echo "✅" || echo "❌"
echo -n "just: "; just --version 2>/dev/null && echo "✅" || echo "❌"
echo -n "python: "; python --version 2>/dev/null && echo "✅" || echo "❌"
echo -n "node: "; node --version 2>/dev/null && echo "✅" || echo "❌"

# Check project structure
echo -n "Project files: "
if [ -f "generate_newpaper.py" ] && [ -d "libs" ] && [ -f "justfile" ]; then
    echo "✅"
else
    echo "❌"
fi

# Check configuration files
echo -n "Secrets file: "
if [ -f "libs/common/secrets.yaml" ]; then
    echo "✅"
else
    echo "❌ (needs to be created)"
fi

echo -n "Environment file: "
if [ -f "libs/.env.development" ]; then
    echo "✅"
else
    echo "❌ (needs to be created)"
fi

echo "=================================="
echo "Verification complete!"
EOF

chmod +x verify_devcontainer.sh
./verify_devcontainer.sh
```

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the BobTimes development environment (DevContainer or local)
- ✅ Configured LLM providers with fast/slow model selection
- ✅ Set up image generation with fallback options
- ✅ Generated your first AI newspaper
- ✅ Understood the basic data models and workflow

## 🔄 Next Steps

- **Lab 2:** Customizing Agent Behavior and Tools
- **Lab 3:** Adding New Reporter Fields and Sections
- **Lab 4:** Building Custom Tools and Integrations
- **Lab 5:** Frontend Integration and API Usage

## 📚 Quick Reference

**Key Files:**
- `libs/common/secrets.yaml` - API keys and secrets
- `libs/.env.development` - Environment configuration
- `generate_newpaper.py` - Main generation script
- `libs/common/core/llm_service.py` - LLM provider configurations

**Key Commands:**
- `just sync` - Install dependencies
- `python generate_newpaper.py` - Generate newspaper
- `just run-backend` - Start API server
- `just lint` - Check code quality

## 🔬 Step 9: Understanding the Models in Detail

### 9.1 Available Models by Provider

**OpenAI Models:**
```python
class OpenAIModel(StrEnum):
    FAST = "gpt-5-mini"      # Fast, cost-effective
    SLOW = "gpt-5"           # High quality, expensive

class OpenAIImageModel(StrEnum):
    GPT_IMAGE_1 = "gpt-image-1"  # Latest image model
```

**Google Gemini Models:**
```python
class GeminiModel(StrEnum):
    FAST = "gemini/gemini-2.0-flash-exp"           # Fast responses
    SLOW = "gemini/gemini-2.5-pro-preview-03-25"   # High quality

class GoogleImageModel(StrEnum):
    IMAGEN_4_0 = "gemini/imagen-4.0-generate-preview-06-06"
```

**Anthropic Models:**
```python
class AnthropicModel(StrEnum):
    FAST = "claude-sonnet-4-20250514"    # Balanced speed/quality
    SLOW = "claude-opus-4-1-20250805"    # Highest quality
```

**AWS Bedrock Models:**
```python
class AWSBedrockModel(StrEnum):
    FAST = "anthropic.claude-3-5-haiku-20241022-v1:0"
    SLOW = "us.anthropic.claude-sonnet-4-20250514-v1:0"
```

### 9.2 Model Selection Strategy

The system automatically chooses models based on task type:

**Reporter Tasks (SLOW models):**
- Topic research and analysis
- Story writing and content creation
- Source verification and fact-checking

**Editor Tasks (SLOW models):**
- Story review and quality assessment
- Editorial decision making
- Content organization and prioritization

**Quick Tasks (FAST models):**
- Simple searches and data retrieval
- Basic content processing
- Status updates and notifications

## 🎛️ Step 10: Advanced Configuration Options

### 10.1 Custom Model Configuration

You can override default models in your environment:

```bash
# In libs/.env.development
LLM_PROVIDER=openai

# Override specific models
LLM_FAST_MODEL=gpt-5-mini
LLM_SLOW_MODEL=gpt-5

# Token limits (optional)
LLM_FAST_MODEL_MAX_TOKENS=8192
LLM_SLOW_MODEL_MAX_TOKENS=20000

# Temperature settings
LLM_TEMPERATURE=0.7
```

### 10.2 Image Generation Configuration

```bash
# Primary image provider
IMAGE_GEN_PROVIDER=openai

# Image generation settings
IMAGE_SIZE=1024x1024
IMAGE_QUALITY=high
IMAGE_STYLE=vivid
```

### 10.3 Logging Configuration

```bash
# Logging settings
LOG_LEVEL=INFO
LOG_JSON_FORMAT=False
LOG_CONSOLE_OUTPUT=True
LOG_FILE_OUTPUT=True
LOG_FILE_PATH=logs/app.log
```

## 🧪 Step 11: Testing Individual Components

### 11.1 Test LLM Generation

```python
# Create a test script: test_llm.py
import asyncio
from core.config_service import ConfigService
from backend.app.dependencies import get_default_llm_service
from pydantic import BaseModel

class TestResponse(BaseModel):
    message: str
    confidence: float

async def test_llm():
    config = ConfigService()
    llm = get_default_llm_service(config)

    response = await llm.generate(
        prompt="Generate a brief news headline about AI",
        response_type=TestResponse
    )

    print(f"Generated: {response.message}")
    print(f"Confidence: {response.confidence}")

# Run: python -c "import asyncio; from test_llm import test_llm; asyncio.run(test_llm())"
```

### 11.2 Test Image Generation

```python
# Test image generation
import asyncio
from core.config_service import ConfigService
from backend.app.dependencies import get_image_generation_service

async def test_image():
    config = ConfigService()
    image_service = get_image_generation_service(config)

    response = image_service.generate_image(
        prompt="A modern newspaper on a desk",
        size="1024x1024",
        n=1
    )

    print(f"Generated {len(response.data)} images")
    for img in response.data:
        print(f"Image URL: {img.url}")

# Run: python -c "import asyncio; from test_image import test_image; asyncio.run(test_image())"
```

## 📈 Step 12: Monitoring and Debugging

### 12.1 Enable Verbose Logging

```bash
# Set detailed logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python generate_newpaper.py
```

### 12.2 Check System Status

```python
# Create system_check.py
from core.config_service import ConfigService
from backend.app.dependencies import get_default_llm_service

def system_check():
    print("🔍 BobTimes System Check")
    print("=" * 40)

    # Check configuration
    config = ConfigService()
    print(f"✅ Environment: {config.get_environment()}")
    print(f"✅ LLM Provider: {config.get('LLM_PROVIDER')}")
    print(f"✅ Image Provider: {config.get('IMAGE_GEN_PROVIDER')}")

    # Check API keys
    providers = ['openai', 'anthropic', 'gemini', 'aws_bedrock']
    for provider in providers:
        key = config.get(f'llm_providers.{provider}.api_key')
        status = "✅ Configured" if key else "❌ Missing"
        print(f"{status} {provider.upper()} API Key")

    # Test LLM service
    try:
        llm = get_default_llm_service(config)
        print(f"✅ LLM Service: {llm.default_model}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")

    print("=" * 40)
    print("System check complete!")

if __name__ == "__main__":
    system_check()
```

Run: `python system_check.py`

## 🎯 Step 13: Minimal Newspaper Structure

### 13.1 Understanding the Output Structure

When you generate a newspaper, you get:

```python
class NewspaperContent(BaseModel):
    title: str = "Bob Times"
    tagline: str = "AI-Generated News That Matters"
    stories: list[PublishedStory]
    metadata: dict[str, Any]
```

### 13.2 Story Organization

Stories are organized by:
- **Section**: Technology, Science, Economics
- **Priority**: Breaking, High, Medium, Low
- **Publication Time**: Most recent first

### 13.3 Minimal Sections Configuration

For this lab, we use these minimal sections:

```python
# Default field requests
field_requests = [
    FieldTopicRequest(field=ReporterField.TECHNOLOGY),
    FieldTopicRequest(field=ReporterField.SCIENCE),
    FieldTopicRequest(field=ReporterField.ECONOMICS)
]
```

Each field generates 1 story by default, giving you a 3-story newspaper.

## 🗓️ Step 14: Making Agents Date-Aware

### 14.1 The Challenge

By default, AI agents don't know the current date, which can lead to outdated or irrelevant news stories. Your task is to ensure all agents (reporters and editors) are aware of the current date and use it appropriately in their work.

### 14.2 Understanding the Problem

Test the current behavior:

```python
# Test current date awareness
python -c "
import asyncio
from agents.reporter_agent import ReporterAgent
from agents.types import ReporterField, TechnologySubSection
from core.config_service import ConfigService

async def test_date_awareness():
    config = ConfigService()
    agent = ReporterAgent(config=config)

    # Ask agent about current date
    response = await agent.execute(
        task='What is today\'s date? Include it in a brief tech news summary.',
        field=ReporterField.TECHNOLOGY,
        sub_section=TechnologySubSection.AI_TOOLS
    )
    print('Agent response:', response.content if hasattr(response, 'content') else response)

asyncio.run(test_date_awareness())
"
```

**Expected Issue:** The agent likely won't know the current date or will provide an incorrect/outdated date.

### 14.3 Exploration Hints

Look at these key areas in the codebase:

**Agent System Prompts:**
```bash
# Find where system prompts are defined
find . -name "*.py" -exec grep -l "system.*prompt\|SystemMessage" {} \;
```

**Agent Configuration:**
```bash
# Look for agent initialization and configuration
find . -name "*agent*.py" -type f
```

**Date/Time Utilities:**
```bash
# Check if there are existing date utilities
find . -name "*.py" -exec grep -l "datetime\|date\|time" {} \;
```

### 14.4 Implementation Direction

You'll need to modify the agent system to include current date information. Consider these approaches:

**Approach 1: System Prompt Enhancement**
```python
# Example system prompt modification (find the actual location)
system_prompt = f"""
You are a professional news reporter for Bob Times.
Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

When writing stories, always consider:
- Is this information current as of {datetime.now().strftime('%B %d, %Y')}?
- Are there recent developments since this date?
- Include temporal context in your reporting
"""
```

**Approach 2: Context Injection**
```python
# Example context addition to agent calls
context = {
    "current_date": datetime.now().isoformat(),
    "formatted_date": datetime.now().strftime('%B %d, %Y'),
    "day_of_week": datetime.now().strftime('%A')
}
```

**Approach 3: Tool Enhancement**
```python
# Example tool that provides date information
class DateTool:
    def get_current_date(self) -> dict:
        now = datetime.now()
        return {
            "iso_date": now.isoformat(),
            "formatted": now.strftime('%B %d, %Y'),
            "day": now.strftime('%A'),
            "time_utc": now.strftime('%H:%M:%S UTC')
        }
```

### 14.5 Testing Your Implementation

After implementing your solution, test it:

```python
# Test date-aware story generation
python -c "
import asyncio
from generate_newpaper import generate_single_issue
from agents.types import ReporterField, TechnologySubSection

async def test_date_aware_story():
    print('Generating date-aware story...')
    exit_code = await generate_single_issue(
        field=ReporterField.TECHNOLOGY,
        sub_section=TechnologySubSection.AI_TOOLS
    )
    print(f'Generation completed: {exit_code}')

asyncio.run(test_date_aware_story())
"
```

**Look for these improvements in the generated story:**
- [ ] Current date mentioned or referenced
- [ ] Temporal context (e.g., "this week", "recently", "as of today")
- [ ] Recent developments prioritized
- [ ] Time-sensitive language used appropriately

### 14.6 Verification Questions

After implementation, your agents should be able to answer:

```python
# Create a verification script
test_questions = [
    "What is today's date?",
    "What day of the week is it?",
    "Write a news headline that includes today's date",
    "Is this information current as of today?"
]

# Test each question with your modified agents
```

### 14.7 Advanced Considerations

Once basic date awareness works, consider:

**Time Zones:**
- Should agents work in UTC or local time?
- How to handle global news across time zones?

**Date Formatting:**
- Consistent date formats across all stories
- User-friendly vs. machine-readable formats

**Temporal Context:**
- "Breaking news" vs. "developing story" language
- Age-appropriate references ("yesterday", "this morning")

### 14.8 Common Pitfalls to Avoid

- **Hardcoding dates** - Always use dynamic date generation
- **Timezone confusion** - Be explicit about timezone usage
- **Inconsistent formatting** - Standardize date formats across agents
- **Performance impact** - Don't call datetime.now() excessively

### 14.9 Success Criteria

Your implementation is successful when:
- [ ] ✅ Agents consistently know the current date
- [ ] ✅ Stories include appropriate temporal context
- [ ] ✅ Date information is accurate and formatted consistently
- [ ] ✅ No performance degradation in story generation
- [ ] ✅ Both reporter and editor agents are date-aware

**Hint:** Look for where agent prompts are constructed and where agent context is established. The solution might involve modifying system prompts, adding context variables, or creating date-aware tools.

## 🏁 Final Verification Checklist

Before moving to the next lab, ensure:

- [ ] ✅ Environment loads without errors
- [ ] ✅ API keys are properly configured
- [ ] ✅ LLM service initializes successfully
- [ ] ✅ Image generation works (or fallback is configured)
- [ ] ✅ Generated at least one complete story
- [ ] ✅ Images are saved to `data/images/`
- [ ] ✅ Logs show successful completion
- [ ] ✅ No critical errors in output
- [ ] ✅ **Agents are aware of current date and use it appropriately**

## 🚀 Ready for Lab 2?

Congratulations! You've successfully completed Lab 1. You now have:

1. **Working Development Environment** ✅
2. **Configured LLM Providers** ✅
3. **Image Generation Setup** ✅
4. **Generated Your First Newspaper** ✅
5. **Understanding of Core Models** ✅

**Next up:** Lab 2 will teach you how to customize agent behavior, add new tools, and modify the newspaper generation workflow.

Ready for the next lab? Let's dive deeper into customizing the system!
