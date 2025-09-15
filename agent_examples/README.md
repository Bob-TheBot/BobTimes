# Strands Agents SDK Examples

This directory contains comprehensive examples demonstrating how to use the Strands Agents SDK to build intelligent AI agents. Each example showcases different capabilities and patterns for agent development.

## Prerequisites

Before running these examples, ensure you have:

1. **Python 3.13+** installed
2. **Install dependencies from repo root**:
   ```bash
   # From the bobtimes repository root directory
   uv sync
   ```
3. **Model provider configured** (AWS Bedrock by default)
4. **Additional dependencies** for specific examples (see individual requirements below)

## Examples Overview

### 1. Weather Forecaster Agent (`weather_agent.py`)

**Features:**
- Tool Used: `http_request`
- API: National Weather Service API (no key required)
- Complexity: Beginner
- Agent Type: Single Agent
- Interaction: Command Line Interface

**Description:**
A weather forecasting agent that uses the National Weather Service API to provide weather information for US locations. The agent demonstrates the multi-step API workflow pattern where it first gets location information and then retrieves forecast data.

**Key Capabilities:**
- Get current weather conditions
- Provide multi-day forecasts
- Handle location queries (coordinates, zip codes, city names)
- Format weather data in user-friendly language

**Usage:**
```bash
python weather_agent.py
```

**Example Queries:**
- "What's the weather like in Seattle, Washington?"
- "Get the weather forecast for latitude 40.7128, longitude -74.0060"
- "Will it rain this weekend in Miami, Florida?"

### 2. MCP Agent (`mcp_agent.py`)

**Features:**
- Tool Used: Model Context Protocol (MCP) servers
- APIs: Multiple MCP servers (filesystem, web search, calculator, database)
- Complexity: Intermediate
- Agent Type: Single Agent with multiple tool integrations
- Interaction: Async command line interface

**Description:**
Demonstrates how to integrate multiple MCP (Model Context Protocol) servers to create a versatile agent with diverse capabilities including file operations, web search, calculations, and database operations.

**Key Capabilities:**
- File system operations (read, write, create, delete)
- Web search using Brave Search API
- Mathematical calculations
- SQLite database operations
- Combined multi-tool workflows

**Prerequisites:**
- Node.js installed (for MCP servers)
- Optional: `BRAVE_SEARCH_API_KEY` environment variable for web search

**Usage:**
```bash
python mcp_agent.py
```

**Example Queries:**
- "Create a file called 'hello.txt' with some content and read it back"
- "Search for the latest AI news and summarize the top results"
- "Calculate compound interest on $10,000 at 5% for 10 years"

### 3. Memory Agent (`memory_agent.py`)

**Features:**
- Tools Used: `mem0_memory`, `use_llm`
- Storage: mem0.ai for persistent memory management
- Complexity: Intermediate
- Agent Type: Single Agent with Memory Management
- Interaction: Command Line Interface
- Key Focus: Memory Operations & Contextual Responses

**Description:**
A memory-enhanced agent that leverages mem0.ai to maintain context across conversations and provide personalized responses. It demonstrates how to store, retrieve, and utilize memories to create more intelligent and contextual AI interactions.

**Key Capabilities:**
- Persistent memory storage across sessions
- Semantic memory retrieval based on relevance
- Natural language response generation using retrieved memories
- Command classification for different memory operations
- User-specific memory isolation

**Memory Operations Workflow:**
1. **Command Classification**: Determines if input is for storing, listing, or retrieving memories
2. **Memory Storage**: Stores new information when users share details
3. **Memory Retrieval**: Finds semantically relevant memories for queries
4. **Response Generation**: Uses LLM to create natural responses from memories

**Usage:**
```bash
python memory_agent.py
```

**Memory Commands:**
- Storage: "Remember that I like pizza" or "Note that I work at Google"
- Listing: "Show me all my memories" or "List my memories"
- Retrieval: Ask any question and the agent uses relevant memories

**Example Interactions:**
- "Remember that I prefer window seats on flights"
- "What do you know about my travel preferences?"
- "Show me all my memories"

### 4. Multi-Agent System (`multi_agent.py`)

**Features:**
- Tools Used: calculator, python_repl, shell, http_request, editor, file operations
- Agent Structure: Multi-Agent Architecture with Tool-Agent Pattern
- Complexity: Intermediate
- Interaction: Command Line Interface
- Key Technique: Dynamic Query Routing

**Description:**
A Teacher's Assistant system that demonstrates multi-agent architecture using Strands Agents, where specialized agents work together under the coordination of a central orchestrator. The system uses natural language routing to direct queries to the most appropriate specialized agent based on subject matter expertise.

**Architecture:**
- **Teacher's Assistant (Orchestrator)**: Central coordinator that routes queries to specialists
- **Math Assistant**: Handles mathematical calculations and concepts using calculator tool
- **English Assistant**: Processes grammar and language comprehension with editor tools
- **Language Assistant**: Manages translations using HTTP request tool for external APIs
- **Computer Science Assistant**: Handles programming with Python REPL, shell, and file tools
- **General Assistant**: Processes queries outside specialized domains

**Key Features:**
- **Tool-Agent Pattern**: Specialized agents wrapped as tools using @tool decorator
- **Natural Language Routing**: Automatic query classification and routing
- **Specialized System Prompts**: Domain-specific prompts for each agent
- **Clean Output**: Orchestrator suppresses intermediate output for better user experience

**Usage:**
```bash
python multi_agent.py
```

**Example Interactions:**
- "Solve the quadratic equation x^2 + 5x + 6 = 0" → Routes to Math Assistant
- "Write a Python function to check if a string is a palindrome" → Routes to CS Assistant
- "Translate 'Hello, how are you?' to Spanish" → Routes to Language Assistant

## Running the Examples

### Quick Start

1. **Navigate to the bobtimes repository root**
2. **Install all dependencies**:
   ```bash
   uv sync
   ```
3. **Configure your model provider** (AWS Bedrock credentials by default)
4. **Run any example from the strands_examples directory**:
   ```bash
   cd strands_examples
   python weather_agent.py
   python mcp_agent.py
   python memory_agent.py
   python multi_agent.py
   ```

## Installation

All dependencies are managed through the main workspace. Simply run:

```bash
# From the bobtimes repository root
uv sync
```

This will install:
- Core Strands Agents SDK (`strands-agents`, `strands-agents-tools`)
- All other project dependencies
- Development tools and utilities

### Additional Requirements for MCP Agent
The MCP agent requires Node.js and MCP servers:

```bash
# Install Node.js MCP servers (requires Node.js)
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-sqlite
```

### Model Provider Configuration

All examples use AWS Bedrock by default with Claude 3.5 Sonnet. To use different providers:

```python
from strands.models import OpenAIModel, AnthropicModel

# OpenAI
model = OpenAIModel(model_id="gpt-4", api_key="your-key")

# Anthropic
model = AnthropicModel(model_id="claude-3-sonnet-20240229", api_key="your-key")

# Update agent creation
agent = Agent(model=model, tools=[...])
```

## Key Concepts Demonstrated

### 1. Tool Integration
- Using built-in tools (`http_request`)
- Creating custom tools with `@tool` decorator
- Integrating MCP servers
- Tool composition and chaining

### 2. Agent Patterns
- Single-purpose specialized agents
- Multi-agent coordination
- Memory-enhanced agents
- Workflow orchestration

### 3. System Prompts
- Role definition and capabilities
- Behavioral guidelines
- Output formatting instructions
- Error handling guidance

### 4. Async Operations
- Asynchronous agent execution
- Concurrent tool usage
- Multi-agent coordination
- Resource management

## Extending the Examples

### Adding New Tools
```python
from strands import tool

@tool
def my_custom_tool(param: str) -> str:
    """Tool description for the LLM."""
    # Your tool logic here
    return result

agent = Agent(tools=[my_custom_tool])
```

### Creating New Agent Types
```python
specialized_agent = Agent(
    model=your_model,
    tools=[relevant_tools],
    system_prompt="Your specialized role and capabilities..."
)
```

### Implementing New Memory Patterns
```python
# Custom memory storage
class CustomMemoryManager:
    def store(self, key, value):
        # Your storage logic
        pass
    
    def retrieve(self, query):
        # Your retrieval logic
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed
2. **Model Provider Issues**: Check your credentials and model access
3. **MCP Server Issues**: Verify Node.js installation and MCP server availability
4. **Memory Issues**: Check ChromaDB installation and file permissions

### Getting Help

- Check the [Strands Agents Documentation](https://strandsagents.com/latest/documentation/)
- Review the [GitHub Repository](https://github.com/strands-agents/sdk-python)
- Look at the system prompts in each example for guidance on agent behavior

## Next Steps

After exploring these examples, consider:

1. **Combining patterns** from different examples
2. **Creating domain-specific agents** for your use case
3. **Building production workflows** with error handling and monitoring
4. **Exploring advanced features** like streaming, evaluation, and deployment

Each example is designed to be educational and extensible. Use them as starting points for your own agent development projects!
