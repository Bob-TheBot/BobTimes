# Customer Support MCP Server & Client

A complete MCP (Model Context Protocol) implementation featuring a FastMCP server for customer support operations and an intelligent client with Claude integration.

## Features

### Server Features
- **Tools**: Get formatted support tickets for customers with database integration
- **Resources**: Access application logs and customer-specific logs (static and dynamic)
- **Prompts**: Generate comprehensive customer issue summaries with customizable parameters

### Client Features
- **Interactive Chat Interface**: Natural language queries with tool integration
- **Resource Access**: Direct access to logs and data via `@` syntax
- **Prompt Integration**: Execute server prompts with `#` or `prompt:` syntax
- **Claude Integration**: Run prompts directly with Claude and tool access
- **Multi-syntax Support**: Flexible command interface for different use cases

## Installation

1. Install dependencies:
```bash
uv add "mcp[cli]"
```

## Running the Server

### HTTP Mode (Recommended)

Run the server with HTTP transport for web-based access:

```bash
# Method 1: Direct Python execution
python main.py

# Method 2: Using FastMCP CLI
fastmcp run main.py

# Method 3: Using uv with FastMCP
uv run fastmcp run main.py
```

**Server will be available at:** `http://0.0.0.0:8001/mcp`

## Running the Client

The client (`client.py`) is an interactive MCP client that connects to the server and provides multiple ways to interact with the available tools, resources, and prompts.

### Basic Usage

```bash
python client.py http://localhost:8001/mcp
```

### Client Architecture

The client is built using:
- **MCP SDK**: Uses the official MCP Python SDK for protocol communication
- **Anthropic Claude**: Integrated with Claude for natural language processing and tool execution
- **Streamable HTTP Transport**: Connects to the server via HTTP for reliable communication
- **Async Architecture**: Built with asyncio for efficient concurrent operations

### Client Commands

Once connected, the client supports multiple command syntaxes:

#### Resource Access (@ syntax)
```bash
@logs                    # Get application logs
@app                     # Alternative for application logs
@customer_ACM001         # Get customer-specific logs
@list                    # List all available resources
```

#### Prompt Access (# and prompt: syntax)
```bash
#prompts                                                    # List all available prompts
#customer_issue_summary                                     # Get prompt with defaults
#customer_issue_summary customer_id=ACM001 timeframe=7days # Get prompt with arguments

# Alternative syntax
prompt:customer_issue_summary customer_id=ACM001 timeframe=24hours
```

#### Natural Language Queries
```bash
Which issues did customer ACM001 have in the last 300 days?
Get support tickets for customer ACM001
Show me the application logs
```

### Example Session

```bash
$ python client.py http://localhost:8001/mcp

🚀 Simple MCP Client Started! (Remote HTTP Connection)
🌐 Connected to your remote MCP server
💡 Commands:
  @logs / @app             - Application logs (static)
  @customer_ACM001         - Customer logs (dynamic!)
  @list                    - List all available resources
  #prompts                 - List all available prompts
  #prompt_name             - Get specific prompt
  prompt:name arg=value    - Get prompt with arguments (+ Claude option)
  help                     - Show this help
  quit                     - Exit
  Or just ask questions naturally!

💬 Query: prompt:customer_issue_summary customer_id=ACM001 timeframe=24hours

📝 Getting prompt: customer_issue_summary
   Arguments: {'customer_id': 'ACM001', 'timeframe': '24hours'}

📋 Prompt: customer_issue_summary
============================================================
Please create a comprehensive customer issue summary for customer ACM001 covering the last 24hours.

Analyze the following data sources:
1. Customer support tickets and their resolution status
2. Application logs showing system interactions and errors
3. Customer-specific activity logs
4. Any patterns in issues or system behavior

[... full prompt content ...]
============================================================

🤖 Would you like to run this prompt with Claude? (y/n): y

🚀 Running prompt with Claude...
🔄 Sending to Claude with tools...
🔧 Using tool: get_support_tickets

🤖 Claude's Response:
============================================================
# Customer Issue Summary: ACM001 (Acme Corporation)

## Customer Overview
- **Customer ID**: ACM001
- **Company**: Acme Corporation
- **Primary Contacts**:
  - Sarah Johnson (CTO) - sarah.johnson@acme.com
  - Robert Davis (IT Director) - robert.davis@acme.com

[... comprehensive analysis with actual data ...]
============================================================
```

### How the Client Works

The client operates through several key components:

1. **Connection Management**: Establishes HTTP connection to the MCP server using streamable HTTP transport
2. **Capability Discovery**: Automatically discovers and registers available tools, resources, and prompts
3. **Command Parsing**: Interprets user input and routes to appropriate handlers (@, #, prompt:, natural language)
4. **Tool Integration**: When Claude needs data, it automatically calls MCP tools and incorporates results
5. **Prompt Enhancement**: Prompts can trigger Claude with full tool access for comprehensive analysis

**Configuration options in `main.py`:**
```python
# Customize host and port
mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)

# Available transports: "streamable-http", "sse", "stdio"
```

### STDIO Mode

For command-line integration and MCP clients that use standard input/output:

```bash
# Method 1: Direct Python execution with stdio
python -c "
import main
main.mcp.run(transport='stdio')
"

# Method 2: Modify main.py temporarily to use stdio transport
# Change the last line in main.py to:
# mcp.run(transport="stdio")
# Then run:
python main.py

# Method 3: Using MCP CLI (legacy)
uv run mcp run -t stdio main.py
```

**Note:** In STDIO mode, the server communicates via standard input/output streams and doesn't use HTTP.

## Available Endpoints & Capabilities

### Resources
- **Application Logs**: `file:///logs/app.log` - TechNova application logs
- **Customer Logs**: `file:///logs/customer_{customer_id}.log` - Customer-specific logs (dynamic template)

### Tools
- **get_support_tickets**: Retrieve formatted support tickets for a customer with database integration
  - Parameters: `customer_id` (required), `timeframe` (optional, default: "30days")
  - Returns: Formatted customer information with all support tickets

### Prompts
- **customer_issue_summary**: Generate comprehensive customer issue analysis prompts
  - Parameters: `customer_id` (required), `timeframe` (optional, default: "24hours")
  - Returns: Structured prompt for creating detailed customer issue summaries

## Client Requirements

- Python 3.8+
- Required packages (install via `pip install`):
  - `anthropic` - For Claude API integration
  - `mcp` - MCP SDK for protocol communication
  - `python-dotenv` - For environment variable management

**Environment Setup:**
Create a `.env` file with your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Testing & Usage Examples

### Quick Start
1. **Start the server:**
   ```bash
   python main.py
   ```

2. **In another terminal, start the client:**
   ```bash
   python client.py http://localhost:8001/mcp
   ```

3. **Try these commands:**
   ```bash
   # List available prompts
   #prompts

   # Get customer data with Claude analysis
   prompt:customer_issue_summary customer_id=ACM001 timeframe=24hours

   # Direct tool usage
   Which issues did customer ACM001 have in the last 300 days?

   # Access logs directly
   @customer_ACM001
   ```

### Advanced HTTP Mode Testing
```python
from fastmcp.client import Client, StreamableHttpTransport

transport = StreamableHttpTransport("http://localhost:8001/mcp")
client = Client(transport)

async with client:
    # List resources
    resources = await client.list_resources()

    # Read application logs
    app_logs = await client.read_resource("file:///logs/app.log")

    # Get customer issue summary prompt
    prompt = await client.get_prompt("customer_issue_summary", {
        "customer_id": "ACM001",
        "timeframe": "24hours"
    })
```

### STDIO Mode Testing
When running in STDIO mode, the server accepts JSON-RPC messages via stdin:

```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "resources/list"}' | python main.py
```

## Sample Data

The server includes sample log files for testing:
- `logs/app.log` - Application system logs
- `logs/customer_ACM001.log` - Sample customer logs for ACM001

## Configuration

### Server Configuration
- **Database**: SQLite database with customer and support ticket data (`data/customers.db`)
- **Logs Directory**: `./logs/` - Contains application and customer log files
- **Transport Options**:
  - `streamable-http` - Full HTTP server with streaming support (recommended)
  - `sse` - Server-Sent Events transport
  - `stdio` - Standard input/output for CLI integration

### Client Configuration
- **Environment Variables**: Store in `.env` file
  - `ANTHROPIC_API_KEY` - Required for Claude integration
- **Connection**: Automatically discovers server capabilities on connection
- **Model**: Uses Claude 3.5 Haiku for fast responses (configurable in client.py)

## Architecture Overview

```
┌─────────────────┐    HTTP/MCP     ┌─────────────────┐
│                 │◄───────────────►│                 │
│  MCP Client     │                 │   MCP Server    │
│  (client.py)    │                 │   (main.py)     │
│                 │                 │                 │
│ ┌─────────────┐ │                 │ ┌─────────────┐ │
│ │   Claude    │ │                 │ │  FastMCP    │ │
│ │ Integration │ │                 │ │   Server    │ │
│ └─────────────┘ │                 │ └─────────────┘ │
│                 │                 │                 │
│ ┌─────────────┐ │                 │ ┌─────────────┐ │
│ │ Interactive │ │                 │ │  SQLite DB  │ │
│ │    Shell    │ │                 │ │   + Logs    │ │
│ └─────────────┘ │                 │ └─────────────┘ │
└─────────────────┘                 └─────────────────┘
```

The client provides a bridge between human operators and the MCP server, with Claude acting as an intelligent intermediary that can understand natural language requests and execute the appropriate MCP tools to gather and analyze data.