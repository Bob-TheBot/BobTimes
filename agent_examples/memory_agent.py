#!/usr/bin/env python3
"""
# 🧠 Personal Agent with FAISS Memory

A specialized Strands agent that personalizes answers based on websearch and local FAISS memory.

## What This Example Shows

This example demonstrates:
- Creating a specialized Strands agent with local FAISS memory capabilities
- Storing information across conversations using vector embeddings
- Retrieving relevant memories based on semantic similarity
- Using memory to create more personalized and contextual AI interactions
- Web search integration for up-to-date information

## Key Memory Operations

- **store**: Save important information for later retrieval using FAISS vector store
- **retrieve**: Access relevant memories based on semantic similarity
- **list**: View all stored memories
- **search**: Find information on the web using DuckDuckGo

## Usage Examples

Storing memories: `Remember that I prefer tea over coffee`
Retrieving memories: `What do you know about my preferences?`
Listing all memories: `Show me everything you remember about me`
Web search: `Search for the best hiking trails in Japan`
"""

import os
import json
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException
from strands import Agent, tool # type: ignore
from strands_tools import http_request

# FAISS Memory System
class FAISSMemory:
    """Local FAISS-based memory system for storing and retrieving memories."""

    def __init__(self, user_id: str, memory_dir: str = "memory_data"):
        self.user_id = user_id
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, f"{user_id}_memories.json")
        self.index_file = os.path.join(memory_dir, f"{user_id}_index.faiss")

        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)

        # Initialize or load existing memories
        self.memories = []
        self.index = None
        self._load_memories()

    def _load_memories(self):
        """Load existing memories and FAISS index."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memories = json.load(f)

        if os.path.exists(self.index_file) and self.memories:
            self.index = faiss.read_index(self.index_file)
        else:
            # Create new FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity

    def _save_memories(self):
        """Save memories and FAISS index to disk."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

        if self.index and self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_file)

    def store(self, content: str) -> str:
        """Store a new memory."""
        # Create memory entry
        memory = {
            "id": len(self.memories),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id
        }

        # Generate embedding
        embedding = self.encoder.encode([content])
        embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity

        # Add to FAISS index
        self.index.add(embedding.astype('float32'))

        # Add to memories list
        self.memories.append(memory)

        # Save to disk
        self._save_memories()

        return f"✅ Memory stored: {content}"

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant memories based on query."""
        if not self.memories or self.index.ntotal == 0:
            return "No memories found."

        # Generate query embedding
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.memories)))

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.memories) and score > 0.3:  # Similarity threshold
                memory = self.memories[idx]
                results.append(f"• {memory['content']} (similarity: {score:.2f})")

        if results:
            return "📚 Relevant memories:\n" + "\n".join(results)
        else:
            return "No relevant memories found for your query."

    def list_all(self) -> str:
        """List all stored memories."""
        if not self.memories:
            return "No memories stored yet."

        result = f"📋 All memories for {self.user_id}:\n"
        for i, memory in enumerate(self.memories, 1):
            timestamp = datetime.fromisoformat(memory['timestamp']).strftime("%Y-%m-%d %H:%M")
            result += f"{i}. {memory['content']} (stored: {timestamp})\n"

        return result

# User identifier
USER_ID = "demo_user"  # In the real app, this would be set based on user authentication.

# Initialize FAISS memory system
memory_system = FAISSMemory(USER_ID)

# System prompt
SYSTEM_PROMPT = """You are a helpful personal assistant that provides personalized responses based on user history and web search.

Capabilities:
- Store information using the faiss_memory tool (action="store")
- Retrieve memories using the faiss_memory tool (action="retrieve")
- List all memories using the faiss_memory tool (action="list")
- Search the web using the websearch tool

Key Rules:
- Be conversational and natural
- Always retrieve relevant memories before responding to questions
- Store new user information and preferences automatically
- Use web search for current information and facts
- Provide personalized responses based on stored memories
- Share only relevant information
"""


@tool
def faiss_memory(action: str, content: str = "", query: str = "") -> str:
    """Manage memories using local FAISS vector store.

    Args:
        action (str): Action to perform - "store", "retrieve", or "list"
        content (str): Content to store (required for "store" action)
        query (str): Query to search for (required for "retrieve" action)

    Returns:
        str: Result of the memory operation.
    """
    if action == "store":
        if not content:
            return "❌ Error: Content is required for storing memories."
        return memory_system.store(content)

    elif action == "retrieve":
        if not query:
            return "❌ Error: Query is required for retrieving memories."
        return memory_system.retrieve(query)

    elif action == "list":
        return memory_system.list_all()

    else:
        return f"❌ Error: Unknown action '{action}'. Use 'store', 'retrieve', or 'list'."


@tool
def websearch(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    """Search the web for updated information.

    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc.
        max_results (int): The maximum number of results to return.

    Returns:
        str: Search results formatted as text.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        if results:
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                url = result.get('href', 'No URL')
                formatted_results.append(f"{i}. **{title}**\n   {body}\n   URL: {url}\n")
            return "\n".join(formatted_results)
        else:
            return "No search results found."
    except RatelimitException:
        return "Rate limit reached. Please try again later."
    except DDGSException as e:
        return f"Search error: {e}"


# Initialize agent
memory_agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[faiss_memory, websearch, http_request],
)

if __name__ == "__main__":
    """Run the personal agent interactive session."""
    print("\n🧠 Personal Agent with FAISS Memory 🧠\n")
    print("This agent uses local FAISS memory and websearch capabilities.")
    print("You can ask me to remember things, retrieve memories, or search the web.")
    print("Commands: 'exit' to quit, 'memory' to list all memories\n")

    # Initialize with a welcome memory
    initial_memory = f"User {USER_ID} started a new conversation session."
    memory_system.store(initial_memory)
    print(f"✅ Session initialized for user: {USER_ID}")

    # Interactive loop
    while True:
        try:
            print("\n" + "="*60)
            user_input = input("💬 You: ").strip()

            if user_input.lower() == "exit":
                print("\n👋 Goodbye! Your memories have been saved.")
                break

            elif user_input.lower() == "memory":
                print("\n" + memory_system.list_all())
                continue

            elif user_input:
                print("\n🤖 Assistant:")
                response = memory_agent(user_input)
                print(response)

        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted. Your memories have been saved.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try a different request.")