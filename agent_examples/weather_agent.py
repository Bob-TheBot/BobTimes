#!/usr/bin/env python3
"""
Weather Forecaster Agent Example using Strands Agents SDK

This example demonstrates how to create a weather forecasting agent using the National Weather Service API.
The agent can provide weather forecasts for locations in the United States without requiring an API key.

Features:
- Tool Used: http_request
- API: National Weather Service API (no key required)
- Complexity: Beginner
- Agent Type: Single Agent
- Interaction: Command Line Interface

Prerequisites:
- pip install strands-agents strands-agents-tools
- Set up your model provider (AWS Bedrock by default)
"""

from strands import Agent
from strands_tools import http_request
from strands.models import BedrockModel

# Define a weather-focused system prompt
WEATHER_SYSTEM_PROMPT = """You are a weather assistant with HTTP capabilities. You can:

1. Make HTTP requests to the National Weather Service API
2. Process and display weather forecast data
3. Provide weather information for locations in the United States

When retrieving weather information:
1. First get the coordinates or grid information using https://api.weather.gov/points/{latitude},{longitude}
2. Then use the returned forecast URL to get the actual forecast
3. For locations, you can also try https://api.weather.gov/points/{zipcode} if you have a zip code

API Workflow:
- Step 1: Get location info from https://api.weather.gov/points/{lat},{lon}
- Step 2: Extract the forecast URL from properties.forecast
- Step 3: Make a second request to get the actual forecast data

When displaying responses:
- Format weather data in a human-readable way
- Highlight important information like temperature, precipitation, and alerts
- Handle errors appropriately (API only covers US locations)
- Convert technical terms to user-friendly language
- Show both current conditions and upcoming forecast periods

For locations outside the US, politely explain that the National Weather Service only covers US locations.

Always explain the weather conditions clearly and provide context for the forecast.
"""

def create_weather_agent():
    """Create and configure the weather agent."""

    # Configure the model (using Bedrock by default)
    model = BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.3,
        streaming=True
    )

    # Create an agent with HTTP capabilities
    agent = Agent(
        model=model,
        system_prompt=WEATHER_SYSTEM_PROMPT,
        tools=[http_request],  # Explicitly enable http_request tool
    )

    return agent

def demonstrate_weather_agent():
    """Demonstrate the weather agent with example queries."""

    # Create the weather agent
    agent = create_weather_agent()

    # Example interactions that work with National Weather Service API
    examples = [
        {
            "description": "Basic weather query for a major US city",
            "query": "What's the weather like in Seattle, Washington?"
        },
        {
            "description": "Forecast request with specific coordinates",
            "query": "Get the weather forecast for latitude 40.7128, longitude -74.0060 (New York City)"
        },
        {
            "description": "Weather query using zip code",
            "query": "What's the weather forecast for zip code 90210?"
        },
        {
            "description": "Multi-day forecast request",
            "query": "Will it rain this weekend in Miami, Florida?"
        },
        {
            "description": "Temperature comparison query",
            "query": "Compare the temperature between Denver and Phoenix this week"
        }
    ]

    print("🌤️  Weather Forecaster Agent Example")
    print("=" * 60)
    print("Using National Weather Service API (US locations only)")
    print("=" * 60)

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print("-" * 40)
        print(f"Query: {example['query']}")
        print("\n🤖 Weather Agent Response:")

        try:
            response = agent(example['query'])
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60)

def interactive_weather_agent():
    """Run the weather agent in interactive mode."""

    # Create the weather agent
    agent = create_weather_agent()

    print("\n🌤️  Interactive Weather Agent (type 'quit' to exit)")
    print("=" * 60)
    print("📍 Note: This agent uses the National Weather Service API")
    print("   and only provides weather data for US locations.")
    print("\nExample queries:")
    print("• 'Weather in Seattle'")
    print("• 'Forecast for 90210'")
    print("• 'Will it rain in Miami tomorrow?'")
    print("• 'Temperature for 40.7128,-74.0060'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🌤️  Ask about US weather: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\n🤖 Weather Agent:")
            response = agent(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the weather agent example."""
    print("Choose mode:")
    print("1. Run demonstration with example queries")
    print("2. Interactive mode")

    try:
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            demonstrate_weather_agent()
        elif choice == "2":
            interactive_weather_agent()
        else:
            print("Invalid choice. Running demonstration mode...")
            demonstrate_weather_agent()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
