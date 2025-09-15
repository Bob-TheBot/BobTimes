#!/usr/bin/env python3
"""
Simple Working MCP Client with Dynamic Resources
Based on your chatbot example - just works!
Now connects to remote server via HTTP!

Usage: python client.py <server_url>
Example: python client.py http://localhost:8001/mcp
"""

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession
# HTTP transport for remote connection
from mcp.client.streamable_http import streamablehttp_client
from contextlib import AsyncExitStack
import asyncio
import sys

load_dotenv()

class SimpleMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools = []
        self.available_prompts = []
        self.sessions = {}  # Maps resource URIs and tool names to sessions

    async def connect_to_remote_server(self, server_url):
        """Connect to remote MCP server via HTTP"""
        try:
            print(f"🌐 Connecting to remote server: {server_url}")
            
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(server_url)
            )
            read, write, _ = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            await session.initialize()
            
            # Register tools
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                print(f"✅ Connected to server: {len(response.tools)} tools")
                
                # Show available tools
                if response.tools:
                    print("🔧 Available tools:")
                    for tool in response.tools:
                        print(f"  • {tool.name}: {tool.description}")
                        
            except Exception as e:
                print(f"⚠️  Could not load tools: {e}")
            
            # Register static resources  
            try:
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
                    print(f"✅ Loaded {len(resources_response.resources)} resources")
                    
                    # Show sample resources
                    sample_resources = resources_response.resources[:3]
                    if sample_resources:
                        print("📁 Sample resources:")
                        for resource in sample_resources:
                            print(f"  • {resource.uri}")
                else:
                    print("📁 No static resources found")
            except Exception as e:
                print(f"⚠️  Could not load resources: {e}")

            # Register prompts
            try:
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments if hasattr(prompt, 'arguments') else []
                        })
                        # Store session for each prompt
                        self.sessions[f"prompt:{prompt.name}"] = session
                    print(f"✅ Loaded {len(prompts_response.prompts)} prompts")

                    # Show available prompts
                    if prompts_response.prompts:
                        print("📝 Available prompts:")
                        for prompt in prompts_response.prompts:
                            print(f"  • {prompt.name}: {prompt.description}")
                else:
                    print("📝 No prompts found")
            except Exception as e:
                print(f"⚠️  Could not load prompts: {e}")

            # Store a session for dynamic resources
            self.sessions["server:remote"] = session
            
        except Exception as e:
            print(f"❌ Error connecting to server: {e}")
            raise

    async def get_resource(self, resource_uri):
        """Get resource - handles dynamic URIs like your chatbot example"""
        session = self.sessions.get(resource_uri)
        
        # Fallback for dynamic resources - find any session that can handle this type
        if not session:
            if resource_uri.startswith("file:///logs/"):
                # Find any session that can handle file resources
                for uri, sess in self.sessions.items():
                    if uri.startswith("file:///") or "technova" in uri or uri.startswith("server:"):
                        session = sess
                        break
        
        if not session:
            print(f"❌ Resource '{resource_uri}' not found.")
            return
        
        try:
            print(f"📖 Reading: {resource_uri}")
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\n📄 Resource: {resource_uri}")
                print("="*60)
                print(result.contents[0].text)
                print("="*60)
            else:
                print("❌ No content available.")
        except Exception as e:
            print(f"❌ Error: {e}")

    async def get_prompt(self, prompt_name, arguments=None):
        """Get and render a prompt with optional arguments"""
        session = self.sessions.get(f"prompt:{prompt_name}")

        if not session:
            print(f"❌ Prompt '{prompt_name}' not found.")
            return

        try:
            print(f"📝 Getting prompt: {prompt_name}")
            if arguments:
                print(f"   Arguments: {arguments}")

            result = await session.get_prompt(name=prompt_name, arguments=arguments or {})
            if result and result.messages:
                print(f"\n📋 Prompt: {prompt_name}")
                print("="*60)

                # Collect the prompt content
                prompt_content = ""
                for message in result.messages:
                    if hasattr(message, 'content'):
                        if hasattr(message.content, 'text'):
                            content = message.content.text
                            print(content)
                            prompt_content += content + "\n"
                        else:
                            content = str(message.content)
                            print(content)
                            prompt_content += content + "\n"
                    else:
                        content = str(message)
                        print(content)
                        prompt_content += content + "\n"

                print("="*60)

                # Ask if user wants to run this prompt with Claude
                try:
                    response = input("\n🤖 Would you like to run this prompt with Claude? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        print("\n🚀 Running prompt with Claude...")
                        await self.run_prompt_with_claude(prompt_content.strip())
                    else:
                        print("👍 Prompt displayed only.")
                except KeyboardInterrupt:
                    print("\n👍 Prompt displayed only.")
            else:
                print("❌ No prompt content available.")
        except Exception as e:
            print(f"❌ Error getting prompt: {e}")

    async def run_prompt_with_claude(self, prompt_content):
        """Run the prompt content with Claude and display the response"""
        try:
            print("🔄 Sending to Claude with tools...")

            # Use the same process_query logic but with the prompt content
            messages = [{'role': 'user', 'content': prompt_content}]

            while True:
                response = self.anthropic.messages.create(
                    max_tokens=4000,
                    model='claude-3-5-haiku-20241022',
                    tools=self.available_tools,  # Include tools!
                    messages=messages
                )

                print("\n🤖 Claude's Response:")
                print("="*60)

                assistant_content = []
                has_tool_use = False

                for content in response.content:
                    if content.type == 'text':
                        print(content.text)
                        assistant_content.append(content)
                    elif content.type == 'tool_use':
                        has_tool_use = True
                        assistant_content.append(content)

                        # Get session and call tool
                        session = self.sessions.get(content.name)
                        if not session:
                            print(f"❌ Tool '{content.name}' not found.")
                            break

                        print(f"\n🔧 Using tool: {content.name}")
                        tool_result = await session.call_tool(content.name, arguments=content.input)

                        # Add assistant message with tool use
                        messages.append({'role': 'assistant', 'content': assistant_content})

                        # Add tool result
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": tool_result.content
                                }
                            ]
                        })

                if not has_tool_use:
                    print("="*60)
                    break

        except Exception as e:
            print(f"❌ Error running prompt with Claude: {e}")

    async def list_prompts(self):
        """List all available prompts"""
        print("\n📝 AVAILABLE PROMPTS:")
        print("="*60)

        if not self.available_prompts:
            print("No prompts available")
            return

        for prompt in self.available_prompts:
            print(f"📋 {prompt['name']}")
            print(f"   Description: {prompt['description']}")
            if prompt.get('arguments'):
                try:
                    arg_names = []
                    for arg in prompt['arguments']:
                        if hasattr(arg, 'name'):
                            arg_names.append(arg.name)
                        elif isinstance(arg, dict) and 'name' in arg:
                            arg_names.append(arg['name'])
                        else:
                            arg_names.append(str(arg))
                    print(f"   Arguments: {arg_names}")
                except Exception as e:
                    print(f"   Arguments: {len(prompt['arguments'])} parameters")
            print()

        print("💡 Usage:")
        print("  #prompt_name                         - Get prompt with default arguments")
        print("  #prompt_name arg1=value1             - Get prompt with specific arguments")
        print("  prompt:prompt_name arg1=value1       - Alternative syntax with arguments")
        print("\n💡 Example:")
        print("  prompt:customer_issue_summary customer_id=ACM001 timeframe=24hours")
        print("\n🤖 After displaying a prompt, you'll be asked if you want to run it with Claude!")
        print("="*60)

    async def list_resources(self):
        """List all available resources (static and dynamic patterns)"""
        print("\n📚 AVAILABLE RESOURCES:")
        print("="*60)
        
        # Show static resources
        static_resources = []
        for uri, session in self.sessions.items():
            if uri.startswith("file://") or uri.startswith("http://") or uri.startswith("https://"):
                static_resources.append(uri)
        
        if static_resources:
            print("📄 Static Resources:")
            for uri in sorted(static_resources):
                print(f"  • {uri}")
        
        # Show dynamic resource patterns
        print("\n🔄 Dynamic Resource Patterns:")
        print("  • file:///logs/customer_{customer_id}.log")
        print("    Examples: @customer_ACM001, @customer_GLX002, @customer_UMB003")
        
        # Show prompts
        if self.available_prompts:
            print("\n📝 Available Prompts:")
            for prompt in self.available_prompts:
                print(f"  • #{prompt['name']}: {prompt['description']}")

        print("\n💡 Quick Access:")
        print("  • @logs, @app, @application  → Application logs")
        print("  • @customer_<ID>             → Customer logs")
        print("  • @list                      → Show this list")
        print("  • #prompts                   → List all prompts")
        print("  • #prompt_name               → Get specific prompt")
        print("  • prompt:name arg=value      → Get prompt with arguments")
        print("="*60)

    async def process_query(self, query):
        """Process query with tools"""
        if not self.available_tools:
            print("❌ No tools available")
            return
            
        messages = [{'role': 'user', 'content': query}]
        
        while True:
            response = self.anthropic.messages.create(
                max_tokens = 2024,
                model = 'claude-3-5-haiku-20241022', 
                tools = self.available_tools,
                messages = messages # type: ignore
            )
            
            assistant_content = []
            has_tool_use = False
            
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    assistant_content.append(content)
                    messages.append({'role': 'assistant', 'content': assistant_content})
                    
                    # Get session and call tool
                    session = self.sessions.get(content.name)
                    if not session:
                        print(f"❌ Tool '{content.name}' not found.")
                        break
                        
                    print(f"🔧 Using tool: {content.name}")
                    result = await session.call_tool(content.name, arguments=content.input)
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })
            
            if not has_tool_use:
                break

    async def chat_loop(self):
        print("\n🚀 Simple MCP Client Started! (Remote HTTP Connection)")
        print("🌐 Connected to your remote MCP server")
        print("💡 Commands:")
        print("  @logs / @app             - Application logs (static)")
        print("  @customer_ACM001         - Customer logs (dynamic!)")
        print("  @list                    - List all available resources")
        print("  #prompts                 - List all available prompts")
        print("  #prompt_name             - Get specific prompt")
        print("  prompt:name arg=value    - Get prompt with arguments (+ Claude option)")
        print("  help                     - Show this help")
        print("  quit                     - Exit")
        print("  Or just ask questions naturally!")
        
        while True:
            try:
                query = input("\n💬 Query: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                    
                if query.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  @logs / @app / @application  - Get application logs (static)")
                    print("  @customer_<ID>               - Get customer logs (e.g., @customer_ACM001)")
                    print("  @list                        - List all available resources")
                    print("  #prompts                     - List all available prompts")
                    print("  #prompt_name                 - Get specific prompt")
                    print("  prompt:name arg=value        - Get prompt with arguments (+ Claude option)")
                    print("  Natural language             - Ask anything and tools will be used automatically")
                    print("\n🔧 Available Tools:")
                    for tool in self.available_tools:
                        print(f"  • {tool['name']}: {tool['description']}")
                    print("\n📝 Available Prompts:")
                    for prompt in self.available_prompts:
                        print(f"  • {prompt['name']}: {prompt['description']}")
                    continue
                
                # Handle @ syntax for resources (like your chatbot!)
                if query.startswith('@'):
                    resource_name = query[1:]
                    
                    # Special commands
                    if resource_name == "list":
                        await self.list_resources()
                        continue
                    
                    # Map common names to URIs
                    if resource_name in ["logs", "app", "application"]:
                        resource_uri = "file:///logs/app.log"
                    elif resource_name.startswith("customer_"):
                        # Dynamic resource - construct URI directly!
                        customer_id = resource_name.replace("customer_", "")
                        resource_uri = f"file:///logs/customer_{customer_id}.log"
                    else:
                        resource_uri = resource_name
                    
                    await self.get_resource(resource_uri)
                    continue

                # Handle # syntax for prompts
                if query.startswith('#'):
                    prompt_query = query[1:]

                    # Special commands
                    if prompt_query == "prompts":
                        await self.list_prompts()
                        continue

                    # Parse prompt name and arguments
                    if ' ' in prompt_query:
                        prompt_name, args_str = prompt_query.split(' ', 1)
                        # Parse simple key=value arguments
                        arguments = {}
                        for arg in args_str.split():
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                arguments[key] = value
                    else:
                        prompt_name = prompt_query
                        arguments = {}

                    await self.get_prompt(prompt_name, arguments)
                    continue

                # Handle prompt: syntax for prompts (alternative syntax)
                if query.startswith('prompt:'):
                    prompt_query = query[7:]  # Remove 'prompt:' prefix

                    # Parse prompt name and arguments
                    if ' ' in prompt_query:
                        prompt_name, args_str = prompt_query.split(' ', 1)
                        # Parse simple key=value arguments
                        arguments = {}
                        for arg in args_str.split():
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                arguments[key] = value
                    else:
                        prompt_name = prompt_query
                        arguments = {}

                    await self.get_prompt(prompt_name, arguments)
                    continue

                # Process as natural language
                await self.process_query(query)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python client.py <server_url>")
        print()
        print("Examples:")
        print("  python client.py https://technova-mcp-server-324351717986.us-central1.run.app/mcp/")
        print("  python client.py https://your-server.com/mcp/")
        print()
        print("💡 Make sure the URL ends with /mcp/ (with trailing slash)")
        sys.exit(1)
    
    server_url = sys.argv[1]
    
    # Validate URL format
    if not server_url.startswith(('http://', 'https://')):
        print("❌ Error: Server URL must start with http:// or https://")
        sys.exit(1)
    
    client = SimpleMCPClient()
    try:
        print(f"🔌 Connecting to remote MCP server...")
        await client.connect_to_remote_server(server_url)
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("💡 Make sure:")
        print("  • Server URL is correct and accessible")
        print("  • Server supports MCP Streamable HTTP protocol")
        print("  • URL ends with /mcp/ (with trailing slash)")
        sys.exit(1)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())