#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Agent Example using Strands Agents SDK

This example demonstrates how to create an agent that uses MCP servers to:
1. Access file system operations
2. Perform web searches
3. Use calculator functions
4. Integrate multiple MCP servers seamlessly

Prerequisites:
- pip install strands-agents strands-agents-tools
- Node.js installed (for MCP servers)
- Set up your model provider (AWS Bedrock by default)
"""

import os

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

def create_filesystem_mcp_client():
    """Create MCP client for filesystem operations."""
    return MCPClient(
        transport_callable=lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        ))
    )

def create_brave_search_mcp_client():
    """Create MCP client for web search operations."""
    return MCPClient(
        transport_callable=lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"]
        ))
    )

def create_calculator_mcp_client():
    """Create MCP client for mathematical calculations."""
    return MCPClient(
        transport_callable=lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"]
        ))
    )

def create_sqlite_mcp_client():
    """Create MCP client for SQLite database operations."""
    return MCPClient(
        transport_callable=lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-sqlite", "/tmp/example.db"]
        ))
    )

def create_mcp_agent():
    """Create an agent with MCP tools using the correct pattern."""

    print("🚀 Setting up MCP servers...")

    # Create MCP clients
    filesystem_client = create_filesystem_mcp_client()
    brave_search_client = create_brave_search_mcp_client()
    calculator_client = create_calculator_mcp_client()
    sqlite_client = create_sqlite_mcp_client()

    # Collect all tools from MCP servers
    all_tools = []

    # Collect tools from MCP clients
    mcp_clients = [
        ("filesystem", filesystem_client),
        ("search", brave_search_client),
        ("calculator", calculator_client),
        ("database", sqlite_client)
    ]

    for name, client in mcp_clients:
        try:
            # Add the MCP client directly as a tool
            all_tools.append(client)
            print(f"✅ Added {name} MCP client")
        except Exception as e:
            print(f"⚠️  Could not add {name} client: {e}")

    if not all_tools:
        print("❌ No MCP tools were loaded. Please check your Node.js installation and MCP servers.")
        return None

    print(f"🔧 Total MCP clients loaded: {len(all_tools)}")

    # Configure the model
    model = BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.3,
        streaming=True
    )

    # Create agent with all MCP tools
    agent = Agent(
        model=model,
        tools=all_tools,
        system_prompt="""You are an AI assistant with access to multiple MCP (Model Context Protocol) servers.

        Your capabilities include:
        1. **File Operations**: Read, write, and manage files using the filesystem server
        2. **Web Search**: Search the internet using Brave Search for current information
        3. **Calculations**: Perform mathematical calculations and computations
        4. **Database Operations**: Query and manage SQLite databases

        When helping users:
        - Use the appropriate tools based on their requests
        - Explain what you're doing when using tools
        - Provide clear, helpful responses
        - If you need to search for information, use the web search capability
        - For file operations, be careful and ask for confirmation before making changes
        - Show your work for calculations and explain the results

        You have access to multiple specialized servers, so you can handle a wide variety of tasks!"""
    )

    return agent

def run_examples(agent: Agent):
    """Run example interactions with the MCP agent."""

    examples = [
        {
            "description": "File Operations Example",
            "query": "Create a file called 'hello.txt' with the content 'Hello from MCP Agent!' and then read it back to confirm."
        },
        {
            "description": "Web Search Example",
            "query": "Search for the latest news about artificial intelligence and summarize the top 3 results."
        },
        {
            "description": "Calculator Example",
            "query": "Calculate the compound interest on $10,000 invested at 5% annual interest for 10 years, compounded monthly."
        },
        {
            "description": "Database Example",
            "query": "Create a simple SQLite table for storing book information (title, author, year) and insert a few sample books."
        },
        {
            "description": "Combined Operations Example",
            "query": "Search for information about Python programming, save the summary to a file, and then calculate how many words are in the summary."
        }
    ]

    print("🔧 MCP Agent Examples")
    print("=" * 60)

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print("-" * 40)
        print(f"Query: {example['query']}")
        print("\n🤖 Agent Response:")

        try:
            response = agent(example['query'])
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60)
def interactive_mode(agent:Agent):
    """Run the agent in interactive mode."""

    print("\n🔧 Interactive MCP Agent (type 'quit' to exit)")
    print("=" * 60)
    print("Available capabilities:")
    print("• File system operations (read, write, create, delete)")
    print("• Web search (current information)")
    print("• Mathematical calculations")
    print("• SQLite database operations")
    print("• Combined multi-tool workflows")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🔧 Ask the MCP agent: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\n🤖 MCP Agent:")
            response = agent(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the MCP agent example."""

    # Check for required environment variables
    if not os.getenv("BRAVE_API_KEY"):
        print("⚠️  Note: Set BRAVE_API_KEY environment variable for web search functionality")
        print("   You can get a free API key from: https://brave.com/search/api/")

    try:
        # Create the MCP agent
        agent = create_mcp_agent()

        if not agent:
            print("❌ Failed to create agent")
            return

        print("✅ MCP Agent ready!")

        # Choose mode
        print("\nChoose a mode:")
        print("1. Run examples")
        print("2. Interactive mode")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            run_examples(agent)
        elif choice == "2":
            interactive_mode(agent)
        else:
            print("Invalid choice. Running examples...")
            run_examples(agent)

    except Exception as e:
        print(f"❌ Error setting up MCP agent: {e}")
        print("\n💡 Troubleshooting tips:")
        print("• Make sure Node.js is installed")
        print("• Check that MCP server packages can be installed")
        print("• Verify your model provider credentials")

if __name__ == "__main__":
    main()
