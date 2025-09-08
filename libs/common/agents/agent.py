"""Unified agent framework with types, configuration, models, and base implementation."""

import json
import re
from collections.abc import Sequence
from typing import Any, TypeVar

import litellm
from core.config_service import ConfigService
from core.llm_service import BaseProviderConfig, LLMProvider, LLMService, ModelSpeed, ProviderFactory
from core.logging_service import get_logger
from pydantic import BaseModel, Field, ValidationError

from .models import (
    AgentPerformance,
    CyclePerformanceMetrics,
    EditorialDecision,
    EditorialFeedback,
    EditorTask,
    NewsCycle,
    PublishedStory,
    ReporterTask,
    StoryDraft,
    StoryImage,
    StorySource,
    StorySubmission,
)
from .models.story_models import AgentResponse, ResearchResult, TopicList
from .tools.base_tool import BaseTool

# Import all types from the types module
from .types import (
    AgentMetricKey,
    AgentType,
    CycleStatus,
    EditorialDecisionType,
    NewspaperSection,
    OverallQuality,
    QualityThreshold,
    StoryPriority,
    StoryStatus,
    TaskType,
)

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


# ============================================================================
# CONFIGURATION MODELS (from agent_config.py)
# ============================================================================

class ToolCall(BaseModel):
    """Represents a tool call request from an agent."""
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""
    name: str
    result: Any


class AgentMessage(BaseModel):
    """Message in agent conversation history."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None


class AgentState(BaseModel):
    """Current state of an agent execution."""
    messages: list[AgentMessage]
    iteration: int
    task: str


def _default_tools() -> list[BaseTool]:
    return []


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    system_prompt: str
    provider: BaseProviderConfig | None = None
    tools: Sequence[BaseTool] = Field(default_factory=_default_tools)
    max_iterations: int = 10
    temperature: float = 0.7
    default_model_speed: ModelSpeed = ModelSpeed.FAST


# ============================================================================
# MODEL IMPORTS (from agent_models.py)
# ============================================================================

# Re-export all models from the models package for backward compatibility

# ============================================================================
# BASE AGENT IMPLEMENTATION (from base_agent.py)
# ============================================================================


class BaseAgent:
    """Base class for all agents with tool execution capabilities."""

    def __init__(self, config: AgentConfig, config_service: ConfigService):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration including system prompt, tools, provider, and settings
            config_service: Configuration service instance
        """
        self.config = config
        self.config_service = config_service

        # Resolve or create provider configuration using enum-based provider selection
        provider = config.provider if config.provider is not None else self._create_provider_from_env(
            config_service)

        # Create LLM service using the resolved provider
        self.llm_service = LLMService(
            config_service=config_service,
            provider=provider
        )

        self.state = AgentState(
            messages=[],
            iteration=0,
            task=""
        )

        # Inject shared LLMService into tools that accept it (e.g., ImageGenerationTool)
        for tool in config.tools:
            if hasattr(tool, "set_llm_service"):
                try:
                    tool.set_llm_service(self.llm_service)
                except Exception:
                    # Best-effort; tools without this method will ignore it
                    pass
        self.tools = {tool.name: tool for tool in config.tools}

    def _create_provider_from_env(self, config_service: ConfigService) -> BaseProviderConfig:
        """Create a provider configuration from environment using enum-based selection.

        Uses LLM_PROVIDER env/config (enum names) and reads API keys from ConfigService.
        Supports synonyms (google->gemini, bedrock->aws_bedrock, azure->azure_openai).
        """
        raw = str(config_service.get("LLM_PROVIDER", "")).lower()
        if not raw:
            raise ValueError(
                "No LLM provider configured. Please set LLM_PROVIDER in environment.")

        # Normalize synonyms to enum values
        alias_map = {
            "google": LLMProvider.GEMINI.value,
            "gemini": LLMProvider.GEMINI.value,
            "bedrock": LLMProvider.AWS_BEDROCK.value,
            "aws_bedrock": LLMProvider.AWS_BEDROCK.value,
            "azure": LLMProvider.AZURE_OPENAI.value,
            "azure_openai": LLMProvider.AZURE_OPENAI.value,
            "openai": LLMProvider.OPENAI.value,
            "anthropic": LLMProvider.ANTHROPIC.value,
        }
        normalized = alias_map.get(raw, raw)

        try:
            provider_enum = LLMProvider(normalized)
        except ValueError as e:
            raise ValueError(f"Unsupported LLM provider: {raw}") from e

        # Construct provider via ProviderFactory
        if provider_enum is LLMProvider.OPENAI:
            api_key = config_service.get("llm_providers.openai.api_key")
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration.")
            return ProviderFactory.create_openai(api_key=api_key)

        if provider_enum is LLMProvider.ANTHROPIC:
            api_key = config_service.get("llm_providers.anthropic.api_key")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found in configuration.")
            return ProviderFactory.create_anthropic(api_key=api_key)

        if provider_enum is LLMProvider.GEMINI:
            api_key = config_service.get("llm_providers.gemini.api_key")
            if not api_key:
                raise ValueError("Gemini API key not found in configuration.")
            return ProviderFactory.create_gemini(
                api_key=api_key,
                project_id=config_service.get("GOOGLE_PROJECT_ID")
            )

        if provider_enum is LLMProvider.AWS_BEDROCK:
            api_key = config_service.get("llm_providers.aws_bedrock.api_key")
            if not api_key:
                raise ValueError(
                    "AWS Bedrock API key not found in configuration.")
            return ProviderFactory.create_bedrock(
                api_key=api_key,
                region_name=config_service.get("AWS_REGION", "us-west-2")
            )

        if provider_enum is LLMProvider.AZURE_OPENAI:
            api_key = config_service.get("AZURE_OPENAI_API_KEY")
            api_base = config_service.get(
                "AZURE_OPENAI_ENDPOINT") or config_service.get("AZURE_OPENAI_BASE")
            if not api_key or not api_base:
                raise ValueError(
                    "Azure OpenAI API key and base URL not found in configuration.")
            return ProviderFactory.create_azure(
                api_key=api_key,
                api_base=api_base,
                api_version=config_service.get(
                    "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                deployment_name=config_service.get("AZURE_OPENAI_DEPLOYMENT")
            )

        # Should not reach here due to enum validation
        raise ValueError(f"Unsupported LLM provider: {raw}")

    async def execute(self, task: str, output_model: type[T], model_speed: ModelSpeed = ModelSpeed.FAST) -> T:
        """Execute agent task with tool calls until completion.

        Args:
            task: The task to complete
            output_model: Pydantic model class for validating output

        Returns:
            Validated output matching the output_model schema

        Raises:
            ValueError: If max iterations reached or validation fails
        """
        self.state.task = task
        self.state.messages = [
            AgentMessage(
                role="system",
                content=self.config.system_prompt
            ),
            AgentMessage(
                role="user",
                content=task
            )
        ]

        while self.state.iteration < self.config.max_iterations:
            self.state.iteration += 1

            logger.info(
                "🔄 Agent execution iteration started",
                iteration=self.state.iteration,
                max_iterations=self.config.max_iterations,
                task_preview=task[:100] + "..." if len(task) > 100 else task,
                agent_type=self.__class__.__name__
            )

            # Generate response from LLM
            response = await self._generate_response(model_speed=model_speed)

            # Check if agent wants to use tools
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                logger.info(
                    "🔧 Tool calls detected in agent response",
                    tool_count=len(tool_calls),
                    tools=", ".join([call.name for call in tool_calls]),
                    iteration=self.state.iteration
                )

                # Execute tools and add results to messages
                tool_results = await self._execute_tools(tool_calls)
                self.state.messages.append(
                    AgentMessage(
                        role="assistant",
                        content=response,
                        tool_calls=tool_calls
                    )
                )
                self.state.messages.append(
                    AgentMessage(
                        role="tool",
                        content=json.dumps(tool_results),
                        tool_results=tool_results
                    )
                )
            else:
                # No tool calls, try to parse final output
                self.state.messages.append(
                    AgentMessage(
                        role="assistant",
                        content=response
                    )
                )

                try:
                    # Attempt to parse response as final output
                    final_output = self._parse_output(response, output_model)
                    logger.info(
                        "🎯 Agent task completed successfully",
                        iterations=self.state.iteration,
                        output_type=output_model.__name__,
                        agent_type=self.__class__.__name__,
                        final_output_summary=str(final_output)[
                            :200] + "..." if len(str(final_output)) > 200 else str(final_output)
                    )
                    return final_output
                except (ValidationError, json.JSONDecodeError) as e:
                    # If parsing fails, continue iteration
                    logger.debug(
                        "Output parsing failed, continuing",
                        error=str(e)
                    )
                    self.state.messages.append(
                        AgentMessage(
                            role="user",
                            content=f"Please provide a valid response matching the required output format: {output_model.model_json_schema()}"
                        )
                    )

        raise ValueError(
            f"Agent failed to complete task within {self.config.max_iterations} iterations")

    async def _generate_response(self, model_speed: ModelSpeed | None = None) -> str:
        """Generate response from LLM based on current state.

        Returns:
            LLM response string
        """
        # Convert messages to format expected by LLM service
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.state.messages
        ]

        # Tool descriptions should be included in the system prompt from the start
        # No need to inject them here - agents should include tools in their system prompts

        # Use LLM service with appropriate model speed
        # For simple responses, use fast model; for complex tasks, use slow model
        if not model_speed:
            model_speed = self._determine_model_speed(messages)

        # Create a simple response model for text generation
        class TextResponse(BaseModel):
            content: str

        try:
            # Use async LLM service generate method with full message history
            response_obj = await self.llm_service.generate(
                messages=messages,
                response_type=TextResponse,
                model_speed=model_speed,
                temperature=self.config.temperature
            )
            return response_obj.content
        except Exception as e:
            logger.warning(
                f"LLM service failed, falling back to direct litellm: {e}")
            # Fallback to direct litellm async call
            response = await litellm.acompletion(
                model=self.llm_service.default_model,
                messages=messages,
                temperature=self.config.temperature
            )

        # Extract content from response using defensive approach
        try:
            # Try to get content using getattr with defaults
            choices = getattr(response, "choices", [])
            if choices:
                choice = choices[0]
                message = getattr(choice, "message", None)
                if message:
                    content = getattr(message, "content", None)
                    if content:
                        return content
                # Try alternative structure
                text = getattr(choice, "text", None)
                if text:
                    return text

            # Fallback: try to convert response to string
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return ""

    def _parse_tool_calls(self, response: str) -> list[ToolCall] | None:
        """Parse tool calls from LLM response.

        Args:
            response: LLM response that may contain tool calls

        Returns:
            List of tool calls or None if no tools requested
        """
        # Look for tool call patterns in response
        # Format: <tool>tool_name</tool><args>{json_args}</args>

        tool_pattern = r"<tool>(.*?)</tool><args>(.*?)</args>"
        matches = re.findall(tool_pattern, response, re.DOTALL)

        if not matches:
            return None

        tool_calls: list[ToolCall] = []
        for tool_name, args_str in matches:
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        name=tool_name.strip(),
                        arguments=args
                    )
                )
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool arguments",
                    tool=tool_name,
                    args=args_str[:100]
                )

        return tool_calls if tool_calls else None

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tools to execute

        Returns:
            List of tool execution results
        """
        results: list[ToolResult] = []

        for call in tool_calls:
            if call.name not in self.tools:
                results.append(
                    ToolResult(
                        name=call.name,
                        result={"error": f"Tool '{call.name}' not found"}
                    )
                )
                continue

            try:
                tool = self.tools[call.name]

                logger.info(
                    f"🛠️  Executing tool: {call.name}",
                    tool_name=call.name,
                    parameters=str(call.arguments)[
                        :200] + "..." if len(str(call.arguments)) > 200 else str(call.arguments)
                )

                # Determine appropriate model speed for this tool
                model_speed = self._determine_tool_model_speed(call.name)

                # Instantiate the tool's params model if provided
                params = call.arguments
                if hasattr(tool, "params_model") and tool.params_model is not None:
                    try:
                        params_obj = tool.params_model(**params)
                    except Exception as e:
                        logger.error(
                            f"❌ Tool parameter validation failed: {call.name}",
                            tool_name=call.name,
                            error=str(e)
                        )
                        results.append(ToolResult(name=call.name, result={
                                       "error": f"Invalid params: {e}"}))
                        continue
                    result = await tool.execute(params_obj, model_speed=model_speed)
                else:
                    # Fallback for tools not yet migrated: pass through kwargs
                    result = await tool.execute(**params)

                # Log successful execution with result summary
                result_summary = str(
                    result)[:150] + "..." if len(str(result)) > 150 else str(result)
                logger.info(
                    f"✅ Tool executed successfully: {call.name}",
                    tool_name=call.name,
                    result_summary=result_summary,
                    result_type=type(result).__name__
                )

                results.append(
                    ToolResult(
                        name=call.name,
                        result=result
                    )
                )
            except Exception as e:
                logger.exception("Tool execution failed")
                results.append(
                    ToolResult(
                        name=call.name,
                        result={"error": str(e)}
                    )
                )

        return results

    def _parse_output(self, response: str, output_model: type[T]) -> T:
        """Parse and validate final output.

        Args:
            response: LLM response to parse
            output_model: Pydantic model for validation

        Returns:
            Validated output instance

        Raises:
            ValidationError: If output doesn't match schema
        """
        # Try to extract JSON from response

        # Look for JSON block in response
        json_pattern = r"```json\n?(.*?)\n?```"
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
            else:
                # Last resort: assume entire response is JSON
                json_str = response

        data = json.loads(json_str)
        return output_model.model_validate(data)

    # Deprecated: tool schemas have been removed in favor of plain-text usage in descriptions.

    def _determine_model_speed(self, messages: list[dict[str, str]]) -> ModelSpeed:
        """Determine appropriate model speed based on task complexity.

        Args:
            messages: Conversation messages to analyze

        Returns:
            ModelSpeed enum value (FAST or SLOW)
        """
        # Use slow model for complex tasks that require reasoning
        if self.tools:
            # If tools are available, likely a complex task
            return ModelSpeed.SLOW

        # Check message content for complexity indicators
        content = " ".join(msg.get("content", "") for msg in messages)
        content_lower = content.lower()

        # Complex task indicators
        complex_indicators = [
            "analyze", "research", "investigate", "compare", "evaluate",
            "write a story", "editorial decision", "review", "critique",
            "detailed", "comprehensive", "thorough", "in-depth"
        ]

        if any(indicator in content_lower for indicator in complex_indicators):
            return ModelSpeed.SLOW

        # Simple tasks use fast model
        return self.config.default_model_speed

    def _determine_tool_model_speed(self, tool_name: str) -> ModelSpeed:
        """Determine appropriate model speed for a specific tool.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            ModelSpeed enum value (FAST or SLOW)
        """
        # Editorial decision-making and evaluation tools use SLOW model
        evaluation_tools = {
            "review_story",
            "evaluate_story",
            "editorial_decision",
            "quality_check",
            "fact_check"
        }

        # Topic selection and story writing tools use FAST model
        content_generation_tools = {
            "collect_topics",
            "assign_topics",
            "collect_story",
            "write_story",
            "generate_content",
            "search",
            "scrape",
            "generate_image",
            "publish_story"
        }

        if tool_name in evaluation_tools:
            return ModelSpeed.SLOW
        elif tool_name in content_generation_tools:
            return ModelSpeed.FAST
        else:
            # Default to agent's configured model speed
            return self.config.default_model_speed

    # ============================================================================
    # SHARED AGENT METHODS FOR REPORTER AND EDITOR
    # ============================================================================

    async def execute_single_tool_call(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call and return the result.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool execution result
        """
        if tool_call.name not in self.tools:
            logger.error(f"Tool not found: {tool_call.name}")
            return {"error": f"Tool '{tool_call.name}' not found"}

        tool = self.tools[tool_call.name]

        try:
            # Determine appropriate model speed for this tool
            model_speed = self._determine_tool_model_speed(tool_call.name)

            # Execute the tool with parameters
            if hasattr(tool, "params_model") and tool.params_model:
                # Validate parameters with the tool's model
                params = tool.params_model(**tool_call.arguments)
                result = await tool.execute(params, model_speed=model_speed)
            else:
                # Legacy tool without params model
                result = await tool.execute(**tool_call.arguments)

            logger.debug(f"Tool '{tool_call.name}' executed successfully")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_call.name}': {e}")
            return {"error": str(e)}

    def format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in system prompt.
        
        Returns:
            Formatted string of tool descriptions
        """
        tool_sections = []

        for tool_name, tool in self.tools.items():
            section = f"## {tool_name.upper()}\n{tool.description.strip()}"
            tool_sections.append(section)

        return "\n\n".join(tool_sections)

    def get_tool_schemas_for_prompt(self) -> str:
        """Get tool parameter schemas for prompt generation.
        
        Returns:
            Formatted string of tool schemas
        """
        schema_sections = []

        for tool_name, tool in self.tools.items():
            if hasattr(tool, "params_model") and tool.params_model:
                schema = tool.params_model.model_json_schema()
                section = f"### {tool_name} Parameters:\n{schema}"
                schema_sections.append(section)

        return "\n\n".join(schema_sections)

    async def generate_structured_response(
        self,
        prompt: str,
        response_type: type[T],
        model_speed: ModelSpeed | None = None,
        temperature: float | None = None
    ) -> T:
        """Generate a structured response using the LLM service.
        
        Args:
            prompt: The prompt to send to the LLM
            response_type: Pydantic model type for the expected response
            model_speed: Optional model speed override
            temperature: Optional temperature override
            
        Returns:
            Structured response of the specified type
        """
        effective_model_speed = model_speed or self.config.default_model_speed
        effective_temperature = temperature or self.config.temperature

        response = await self.llm_service.generate(
            prompt=prompt,
            response_type=response_type,
            model_speed=effective_model_speed,
            temperature=effective_temperature
        )

        return response

    def create_base_system_prompt(self, agent_type: str, role_description: str) -> str:
        """Create a base system prompt with common structure.
        
        Args:
            agent_type: Type of agent (e.g., "Reporter", "Editor")
            role_description: Description of the agent's role
            
        Returns:
            Base system prompt string
        """
        tools_section = self.format_tools_for_prompt()

        base_prompt = f"""You are an experienced {agent_type.lower()} for a major newspaper.

# ROLE
{role_description}

# AVAILABLE TOOLS
{tools_section}

# CRITICAL RULES
- Always respond with valid JSON matching the required schema
- Use tools when you need additional information
- Maintain journalistic standards and integrity
- Provide reasoning for your decisions and actions
"""
        return base_prompt


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Types and enums (excluding ReporterField which should be imported from agents.types)
    "StoryStatus",
    "EditorialDecisionType",
    "StoryPriority",
    "NewspaperSection",
    "OverallQuality",
    "QualityThreshold",
    "AgentType",
    "CycleStatus",
    "AgentMetricKey",
    "TaskType",
    # Configuration models
    "ToolCall",
    "ToolResult",
    "AgentMessage",
    "AgentState",
    "BaseTool",
    "AgentConfig",
    # Base agent
    "BaseAgent",
    # Model imports
    "AgentPerformance",
    "CyclePerformanceMetrics",
    "EditorialDecision",
    "EditorialFeedback",
    "EditorTask",
    "NewsCycle",
    "PublishedStory",
    "ReporterTask",
    "StoryDraft",
    "StoryImage",
    "StorySource",
    "StorySubmission",
    "TopicList",
    "ResearchResult",
    "AgentResponse",
]
