"""Reporter-specific tools for memory retrieval and drafting (no web research)."""
from typing import Any
from difflib import SequenceMatcher

from agents.tools.base_tool import BaseTool
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# ============================================================================
# MEMORY FETCH TOOL (with intelligent matching)
# ============================================================================

class FetchFromMemoryParams(BaseModel):
    """Parameters for fetching content from SharedMemoryStore.
    Provide either an exact topic_key OR a topic_query for intelligent matching.
    """
    field: str = Field(description="Field to search within (technology/economics/science)")
    topic_key: str | None = Field(default=None, description="Exact topic key from memory (if known)")
    topic_query: str | None = Field(default=None, description="Free-text topic query to match against stored topics")


class FetchFromMemoryResult(BaseModel):
    """Result of fetching content from memory."""
    success: bool = Field(description="Whether the fetch was successful")
    topic_key: str = Field(description="The topic key that was fetched")
    sources_count: int = Field(default=0, description="Number of sources retrieved")
    content_summary: str = Field(default="", description="Summary of retrieved content")
    error: str | None = Field(default=None, description="Error message if fetch failed")


class FetchFromMemoryTool(BaseTool):
    """Fetch content from SharedMemoryStore for a specific topic key.

    This tool allows reporters to retrieve validated content that was previously
    stored in memory during topic discovery phase.
    """

    name: str = "fetch_from_memory"
    description: str = """
Fetch content from SharedMemoryStore by exact topic key or via intelligent matching.
Use this when you have a topic assignment and need to retrieve the research content stored by the researcher.

Parameters:
- field: Field to search within (technology/economics/science)
- topic_key (optional): Exact topic key from memory (use when you know the exact key)
- topic_query (optional): Free-text query (use when you don't know the exact key)

Behavior:
- If topic_key is provided and exists in the given field → fetch that memory
- Else, the tool will similarity-match topic_query against stored topic keys in the field and select the best match

Examples:
<tool>fetch_from_memory</tool><args>{"field": "technology", "topic_key": "Best AI Tools To Create Viral Content"}</args>
<tool>fetch_from_memory</tool><args>{"field": "technology", "topic_query": "viral AI content creation tools"}</args>

Returns: A summary of sources and the resolved topic_key
"""
    params_model: type[BaseModel] | None = FetchFromMemoryParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> FetchFromMemoryResult:
        """Fetch content from SharedMemoryStore with intelligent topic matching."""
        if not isinstance(params, FetchFromMemoryParams):
            return FetchFromMemoryResult(success=False, topic_key="", error="Invalid parameters provided")

        try:
            from agents.shared_memory_store import get_shared_memory_store
            memory_store = get_shared_memory_store()

            selected_key: str | None = None
            memory_entry = None

            # Try exact key if provided
            if params.topic_key:
                entry = memory_store.get_memory(params.topic_key)
                if entry and entry.field == params.field:
                    selected_key = params.topic_key
                    memory_entry = entry

            # Decide on query for matching if exact wasn't resolved
            query = params.topic_query or params.topic_key

            if selected_key is None:
                candidates = memory_store.list_topics(field=params.field)
                if not candidates:
                    return FetchFromMemoryResult(success=False, topic_key="", error=f"No topics found for field '{params.field}'")

                if not query:
                    return FetchFromMemoryResult(success=False, topic_key="", error=f"Provide 'topic_key' or 'topic_query'. Available topics: {candidates}")

                def norm(s: str) -> str:
                    return " ".join(s.lower().strip().split())

                nq = norm(query)
                best_key = None
                best_score = 0.0
                for cand in candidates:
                    nc = norm(cand)
                    ratio = SequenceMatcher(a=nq, b=nc).ratio()
                    contains = 1.0 if (nq in nc or nc in nq) else 0.0
                    qs = set(nq.split())
                    cs = set(nc.split())
                    jacc = (len(qs & cs) / len(qs | cs)) if (qs or cs) else 0.0
                    score = 0.5 * ratio + 0.4 * jacc + 0.1 * contains
                    if score > best_score:
                        best_score = score
                        best_key = cand

                if best_key is None or best_score < 0.5:
                    suggestions = candidates[:5]
                    return FetchFromMemoryResult(success=False, topic_key="", error=f"No good match for '{query}'. Top suggestions: {suggestions}")

                selected_key = best_key
                memory_entry = memory_store.get_memory(selected_key)

            if not memory_entry:
                return FetchFromMemoryResult(success=False, topic_key="", error="Matched topic could not be loaded from memory")

            # Create content summary
            content_parts: list[str] = []
            for source in memory_entry.sources:
                if getattr(source, "content", None):
                    content_parts.append(f"Source: {source.title}\nContent: {source.content[:200]}...")
                elif getattr(source, "summary", None):
                    content_parts.append(f"Source: {source.title}\nSummary: {source.summary}")

            content_summary = "\n\n".join(content_parts)

            logger.info(
                f"📚 [FETCH-MEMORY] Retrieved content for topic: {selected_key}",
                sources_count=len(memory_entry.sources),
                field=params.field,
                content_length=len(content_summary)
            )

            return FetchFromMemoryResult(
                success=True,
                topic_key=selected_key,
                sources_count=len(memory_entry.sources),
                content_summary=content_summary,
                error=None,
            )

        except Exception as e:
            logger.exception("Failed to fetch from memory")
            return FetchFromMemoryResult(success=False, topic_key="", error=str(e))



# ============================================================================
# USE LLM (generic completion for drafting text with memory context)
# ============================================================================
class UseLLMParams(BaseModel):
    prompt: str = Field(description="Prompt to send to the LLM")
    temperature: float | None = Field(default=None, description="Optional temperature override")


class UseLLMResult(BaseModel):
    success: bool
    content: str | None = None
    error: str | None = None


class UseLLMTool(BaseTool):
    """Call the configured LLM for free-form text generation.

    Reporters can use this after retrieving memory to compose drafts or outlines.
    """

    name: str = "use_llm"
    description: str = """
Generate text using the configured LLM service.

Parameters:
- prompt: string prompt to send
- temperature: optional float to control creativity

Example:
<tool>use_llm</tool><args>{"prompt": "Write a 200-word summary about ..."}</args>
"""
    params_model: type[BaseModel] | None = UseLLMParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.SLOW) -> UseLLMResult:
        if not isinstance(params, UseLLMParams):
            return UseLLMResult(success=False, error="Invalid parameters provided")
        try:
            llm = self.get_llm_service()
            if llm is None:
                return UseLLMResult(success=False, error="LLM service not available")

            class TextResponse(BaseModel):
                content: str

            resp = await llm.generate(
                prompt=params.prompt,
                response_type=TextResponse,
                model_speed=model_speed,
                temperature=params.temperature
            )
            return UseLLMResult(success=True, content=resp.content)
        except Exception as e:
            logger.exception("use_llm tool failed")
            return UseLLMResult(success=False, error=str(e))

class ReporterToolRegistry:
    """Registry for reporter tools with automatic schema generation."""

    def __init__(self, config_service: ConfigService | None = None) -> None:
        """Initialize the tool registry with reporter-specific tools."""
        self.config_service = config_service or ConfigService()

        # Import YouTube tool here to avoid circular imports

        self.tools = {
            "fetch_from_memory": FetchFromMemoryTool(),
            "use_llm": UseLLMTool(),
        }

    def get_all_tools(self) -> list[BaseTool]:
        """Get all reporter tools as a list."""
        return list(self.tools.values())

    def get_tool_by_name(self, name: str) -> BaseTool | None:
        """Get a specific tool by name."""
        return self.tools.get(name)

    def get_tool_descriptions(self) -> dict[str, str]:
        """Get tool names and descriptions for prompt generation."""
        return {name: tool.description for name, tool in self.tools.items()}

    def get_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get tool parameter schemas for prompt generation."""
        schemas: dict[str, dict[str, Any]] = {}
        for name, tool in self.tools.items():
            if tool.params_model:
                schemas[name] = tool.params_model.model_json_schema()
        return schemas

    def format_tools_for_prompt(self) -> str:
        """Format all tools for inclusion in system prompt."""
        tool_sections = []

        for name, tool in self.tools.items():
            section = f"## {name.upper()}\n**Tool name:** `{name}`\n\n{tool.description.strip()}"
            tool_sections.append(section)

        return "\n\n".join(tool_sections)

