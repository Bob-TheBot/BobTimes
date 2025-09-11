#!/usr/bin/env python3
"""
Teacher's Assistant - Strands Multi-Agent Architecture Example

This example demonstrates how to implement a multi-agent architecture using Strands Agents,
where specialized agents work together under the coordination of a central orchestrator.
The system uses natural language routing to direct queries to the most appropriate
specialized agent based on subject matter expertise.

Features:
- Tools Used: calculator, python_repl, shell, http_request, editor, file operations
- Agent Structure: Multi-Agent Architecture
- Complexity: Intermediate
- Interaction: Command Line Interface
- Key Technique: Dynamic Query Routing

Prerequisites:
- pip install strands-agents strands-agents-tools
- Set up your model provider (AWS Bedrock by default)
"""

from strands import Agent, tool
from strands_tools import calculator, python_repl, shell, http_request, editor, file_read, file_write

# System prompts for specialized agents
TEACHER_SYSTEM_PROMPT = """You are a Teacher's Assistant that helps students with various academic subjects.
You coordinate with specialized teaching assistants to provide the best possible help.

Your role is to:
1. Analyze student queries to understand what subject area they need help with
2. Route queries to the most appropriate specialized assistant
3. Ensure students get accurate, helpful responses

Available specialized assistants:
- Math Assistant: For mathematical calculations, equations, and concepts
- English Assistant: For grammar, writing, and language comprehension
- Language Assistant: For translations and foreign language queries
- Computer Science Assistant: For programming, coding, and technical concepts
- General Assistant: For general knowledge questions outside specialized domains

Always route queries to the most appropriate specialist. If unsure, use the General Assistant."""

MATH_ASSISTANT_SYSTEM_PROMPT = """You are a Math Assistant specializing in mathematical calculations,
problems, and concepts. You have access to advanced mathematical tools.

Your capabilities:
- Solve equations and mathematical problems step by step
- Explain mathematical concepts clearly
- Perform calculations using advanced mathematical functions
- Show work and reasoning for all solutions
- Handle algebra, calculus, statistics, and other mathematical domains

Always show your work and explain concepts clearly for educational purposes."""

ENGLISH_ASSISTANT_SYSTEM_PROMPT = """You are an English Assistant specializing in grammar,
writing, and language comprehension.

Your capabilities:
- Help with grammar and syntax questions
- Assist with writing and composition
- Explain language rules and concepts
- Provide feedback on writing style and clarity
- Help with reading comprehension

Focus on clear explanations and educational guidance."""

LANGUAGE_ASSISTANT_SYSTEM_PROMPT = """You are a Language Assistant specializing in translations
and foreign language queries.

Your capabilities:
- Translate text between different languages
- Explain language differences and cultural context
- Help with pronunciation and language learning
- Provide grammar explanations for foreign languages

Use external resources when needed for accurate translations."""

COMPUTER_SCIENCE_ASSISTANT_SYSTEM_PROMPT = """You are a Computer Science Assistant specializing in
programming, coding, and technical concepts.

Your capabilities:
- Write, edit, and debug code in various programming languages
- Explain programming concepts and algorithms
- Help with software development questions
- Execute and test code snippets
- Provide technical guidance and best practices

Always provide working code examples and clear explanations."""

GENERAL_ASSISTANT_SYSTEM_PROMPT = """You are a General Assistant that handles queries outside
of specialized academic domains.

Your capabilities:
- Answer general knowledge questions
- Provide information on various topics
- Help with research and fact-finding
- Offer guidance on non-specialized subjects

Provide accurate, helpful information while acknowledging when topics require specialized expertise."""
    
# Specialized agent tools using the Tool-Agent Pattern
@tool
def math_assistant(query: str) -> str:
    """
    Process and respond to math-related queries using a specialized math agent.

    Args:
        query: The mathematical question or problem to solve

    Returns:
        Detailed mathematical solution with step-by-step explanation
    """
    # Format the query for the math agent with clear instructions
    formatted_query = f"Please solve the following mathematical problem, showing all steps and explaining concepts clearly: {query}"

    try:
        print("🔢 Routed to Math Assistant")
        # Create the math agent with calculator capability
        math_agent = Agent(
            system_prompt=MATH_ASSISTANT_SYSTEM_PROMPT,
            tools=[calculator],
        )
        response = math_agent(formatted_query)
        return str(response)

    except Exception as e:
        return f"Error processing your mathematical query: {str(e)}"

@tool
def english_assistant(query: str) -> str:
    """
    Process and respond to English language, grammar, and writing queries.

    Args:
        query: The English language question or writing request

    Returns:
        Detailed explanation or assistance with English language concepts
    """
    formatted_query = f"Please help with this English language question, providing clear explanations and examples: {query}"

    try:
        print("📝 Routed to English Assistant")
        english_agent = Agent(
            system_prompt=ENGLISH_ASSISTANT_SYSTEM_PROMPT,
            tools=[editor, file_read, file_write],
        )
        response = english_agent(formatted_query)
        return str(response)

    except Exception as e:
        return f"Error processing your English language query: {str(e)}"

@tool
def language_assistant(query: str) -> str:
    """
    Process and respond to translation and foreign language queries.

    Args:
        query: The translation or foreign language question

    Returns:
        Translation or foreign language assistance
    """
    formatted_query = f"Please help with this language or translation request: {query}"

    try:
        print("🌍 Routed to Language Assistant")
        language_agent = Agent(
            system_prompt=LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
            tools=[http_request],
        )
        response = language_agent(formatted_query)
        return str(response)

    except Exception as e:
        return f"Error processing your language query: {str(e)}"

@tool
def computer_science_assistant(query: str) -> str:
    """
    Process and respond to programming and computer science queries.

    Args:
        query: The programming or computer science question

    Returns:
        Code examples, explanations, and technical guidance
    """
    formatted_query = f"Please help with this programming or computer science question, providing working code examples when appropriate: {query}"

    try:
        print("💻 Routed to Computer Science Assistant")
        cs_agent = Agent(
            system_prompt=COMPUTER_SCIENCE_ASSISTANT_SYSTEM_PROMPT,
            tools=[python_repl, shell, editor, file_read, file_write],
        )
        response = cs_agent(formatted_query)
        return str(response)

    except Exception as e:
        return f"Error processing your computer science query: {str(e)}"

@tool
def general_assistant(query: str) -> str:
    """
    Process and respond to general knowledge queries outside specialized domains.

    Args:
        query: The general knowledge question

    Returns:
        General information and guidance
    """
    formatted_query = f"Please help with this general question: {query}"

    try:
        print("🎓 Routed to General Assistant")
        general_agent = Agent(
            system_prompt=GENERAL_ASSISTANT_SYSTEM_PROMPT,
            tools=[],
        )
        response = general_agent(formatted_query)
        return str(response)

    except Exception as e:
        return f"Error processing your general query: {str(e)}"
    
def create_teacher_assistant():
    """Create the Teacher's Assistant (orchestrator) with all specialized agents as tools."""

    # Create the teacher assistant with all specialized agents as tools
    teacher_agent = Agent(
        system_prompt=TEACHER_SYSTEM_PROMPT,
        callback_handler=None,  # Suppress intermediate output for cleaner experience
        tools=[math_assistant, language_assistant, english_assistant,
               computer_science_assistant, general_assistant],
    )

    return teacher_agent

def demonstrate_multi_agent_system():
    """Demonstrate the multi-agent system with example queries."""

    teacher = create_teacher_assistant()

    print("🎓 Teacher's Assistant - Multi-Agent System Demonstration")
    print("=" * 70)

    # Example queries that showcase different specialized agents
    examples = [
        {
            "category": "Mathematics",
            "query": "Solve the quadratic equation x^2 + 5x + 6 = 0",
            "expected_agent": "Math Assistant"
        },
        {
            "category": "Computer Science",
            "query": "Write a Python function to check if a string is a palindrome",
            "expected_agent": "Computer Science Assistant"
        },
        {
            "category": "Language Translation",
            "query": "Translate 'Hello, how are you?' to Spanish",
            "expected_agent": "Language Assistant"
        },
        {
            "category": "English Grammar",
            "query": "Explain the difference between 'affect' and 'effect' with examples",
            "expected_agent": "English Assistant"
        },
        {
            "category": "General Knowledge",
            "query": "What are the main causes of climate change?",
            "expected_agent": "General Assistant"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['category']} Query")
        print("-" * 50)
        print(f"Query: {example['query']}")
        print(f"Expected Route: {example['expected_agent']}")
        print("\n🎓 Teacher's Assistant Response:")

        try:
            response = teacher(example['query'])
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 70)

def interactive_teacher_assistant():
    """Run the Teacher's Assistant in interactive mode."""

    teacher = create_teacher_assistant()

    print("🎓 Interactive Teacher's Assistant")
    print("=" * 60)
    print("I can help you with various academic subjects by routing your")
    print("questions to specialized teaching assistants:")
    print()
    print("📚 Available Specialties:")
    print("• 🔢 Mathematics - equations, calculations, math concepts")
    print("• 📝 English - grammar, writing, language comprehension")
    print("• 🌍 Languages - translations, foreign language help")
    print("• 💻 Computer Science - programming, coding, technical concepts")
    print("• 🎓 General Knowledge - other academic topics")
    print()
    print("Type 'demo' for demonstration examples, 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🎓 Student: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Keep learning!")
                break

            if user_input.lower() == 'demo':
                demonstrate_multi_agent_system()
                continue

            if not user_input:
                continue

            print("\n🎓 Teacher's Assistant:")
            response = teacher(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye! Keep learning!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the Teacher's Assistant example."""
    print("🎓 Teacher's Assistant - Multi-Agent Architecture")
    print("=" * 60)
    print("This example demonstrates a multi-agent system where specialized")
    print("agents work together under a central orchestrator using natural")
    print("language routing to direct queries to the most appropriate specialist.")
    print("=" * 60)
    print()
    print("Choose mode:")
    print("1. Run demonstration with example queries")
    print("2. Interactive mode")

    try:
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            demonstrate_multi_agent_system()
        elif choice == "2":
            interactive_teacher_assistant()
        else:
            print("Invalid choice. Running demonstration mode...")
            demonstrate_multi_agent_system()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
