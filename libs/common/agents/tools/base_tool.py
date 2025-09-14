"""Base tool interface for agent tools."""

import json
from abc import ABC, abstractmethod
from typing import Any, cast

from core.llm_service import ModelSpeed
from pydantic import BaseModel, Field, PrivateAttr


class UnifiedToolResult(BaseModel):
    """Unified result class for all reporter tools."""
    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="Type of operation performed (search, youtube, scrape)")
    query: str | None = Field(default=None, description="Original query/URL used")

    # Core data that all tools provide
    sources: list[Any] = Field(default_factory=list, description="Sources found/scraped")
    topics_extracted: list[str] = Field(default_factory=list, description="Topics extracted from results")
    topic_source_mapping: dict[str, Any] = Field(default_factory=dict, description="Mapping of topics to their source data")

    # Optional metadata (tool-specific)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Tool-specific metadata")

    # Results and error handling
    summary: str | None = Field(default=None, description="Summary of operation results")
    error: str | None = Field(default=None, description="Error message if operation failed")


class BaseTool(ABC, BaseModel):
    """Abstract base class for all agent tools."""

    name: str
    description: str

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    # Optional LLM service for tools that need it
    _llm_service: Any = PrivateAttr(default=None)

    # Each concrete tool must set this to its params model type
    params_model: type[BaseModel] | None = None

    def set_llm_service(self, llm_service: Any) -> None:
        """Set the LLM service for tools that need it."""
        self._llm_service = llm_service

    def get_llm_service(self) -> Any:
        """Get the LLM service if available."""
        return self._llm_service

    @abstractmethod
    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> Any:
        """Execute the tool with structured params.

        Args:
            params: Pydantic params model instance for this tool
            model_speed: Model speed preference (FAST or SLOW), defaults to FAST

        Returns:
            Tool execution result
        """

    # Note: JSON schema export removed. Tools should include usage guidance in `description`.

    # ------------------------------------------------------------------
    # Simple usage help for prompting
    # ------------------------------------------------------------------
    def get_usage(self) -> str:
        """Return a concise usage guide string for this tool.

        The default implementation introspects the tool's Pydantic params_model
        (if provided) to produce a minimal, human-readable guide with examples.
        Tools can override this for custom instructions.
        """
        # Header
        lines: list[str] = [f"- {self.name}: {self.description}"]

        # If the tool defines structured params, derive a lightweight spec
        if self.params_model is not None and hasattr(self.params_model, "model_json_schema"):
            schema: dict[str, Any] = self.params_model.model_json_schema()
            props_any: dict[str, Any] = schema.get("properties", {}) or {}
            props: dict[str, dict[str, Any]] = cast("dict[str, dict[str, Any]]", props_any)
            required: list[str] = schema.get("required", []) or []

            if props:
                lines.append("  Params:")
                for pname, pinfo in props.items():
                    ptype: str = str(pinfo.get("type") or "object")
                    enum_values: list[Any] | None = cast("list[Any] | None", pinfo.get("enum"))  # may be list[str|int|...]
                    default: Any = pinfo.get("default")
                    is_required = pname in required
                    desc: str | None = cast("str | None", pinfo.get("description"))

                    details: list[str] = []
                    details.append(ptype)
                    if enum_values:
                        details.append(f"one of {enum_values}")
                    if default is not None:
                        details.append(f"default {default}")
                    if is_required:
                        details.append("required")
                    summary = ", ".join(details)

                    if desc:
                        lines.append(f"    - {pname}: {desc} ({summary})")
                    else:
                        lines.append(f"    - {pname}: ({summary})")

                # Build a tiny example args object
                example: dict[str, Any] = {}
                for pname, pinfo in props.items():
                    if "default" in pinfo:
                        example[pname] = pinfo["default"]
                        continue
                    # Prefer a sensible example for required fields
                    if pname in required:
                        example[pname] = self._example_for_type(pname, pinfo)
                if example:
                    lines.append("  Example call:")
                    lines.append(f"    <tool>{self.name}</tool><args>{json.dumps(example)}</args>")
        else:
            # No structured params; generic example
            lines.append("  Params: none or free-form JSON")
            lines.append(f"  Example call: <tool>{self.name}</tool><args>{{}}</args>")

        return "\n".join(lines)

    def _example_for_type(self, pname: str, pinfo: dict[str, Any]) -> Any:
        """Heuristic example value for a JSON schema-like property."""
        # Enum takes precedence
        enum_values: list[Any] | None = cast("list[Any] | None", pinfo.get("enum"))
        if enum_values and len(enum_values) > 0:
            return enum_values[0]

        ptype: str | None = cast("str | None", pinfo.get("type"))
        if ptype == "string":
            if "url" in pname:
                return "https://example.com"
            if "query" in pname or pname == "q":
                return "latest news"
            if "id" in pname:
                return "abc123"
            return "example"
        if ptype == "integer":
            return 1
        if ptype == "number":
            return 1.0
        if ptype == "boolean":
            return False
        if ptype == "array":
            items_val: Any = pinfo.get("items")
            if isinstance(items_val, dict):
                items: dict[str, Any] = items_val
            else:
                items = {}
            ex_item: Any = self._example_for_type("item", items) if items else "example"
            return [ex_item]
        if ptype == "object":
            return {}
        # Fallback
        return "example"

    def validate_arguments(self, **kwargs: Any) -> dict[str, Any]:
        """Validate tool arguments before execution.

        Args:
            **kwargs: Arguments to validate

        Returns:
            Validated arguments

        Raises:
            ValueError: If arguments are invalid
        """
        # Default implementation - can be overridden by subclasses
        return kwargs
