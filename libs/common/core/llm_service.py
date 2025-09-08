"""Unified LLM Service - Complete LLM implementation with types, providers, and service.

This module provides a unified interface for multiple LLM providers using LiteLLM's
built-in features for configuration, validation, and connection management.
Supports OpenAI, Anthropic, Gemini, AWS Bedrock, and Azure OpenAI.

Combines:
- LLM types and models (from llm_types.py)
- Provider configurations (from llm_providers.py) 
- LLM service implementation (from llm_service.py)
"""

import json
import os
import time
from enum import StrEnum
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, Field

from .config_service import ConfigService
from .logging_service import get_logger

logger = get_logger(__name__)

# ============================================================================
# TYPES AND MODELS
# ============================================================================

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    """Represents a message in an LLM conversation."""
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")


class LLMUsage(BaseModel):
    """Represents token usage information from an LLM response."""
    prompt_tokens: int = Field(0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the completion")
    total_tokens: int = Field(0, description="Total number of tokens used")


class LLMChoice(BaseModel):
    """Represents a choice in an LLM response."""
    message: LLMMessage = Field(..., description="The message content")
    finish_reason: str | None = Field(None, description="Reason the generation finished")
    index: int = Field(0, description="Index of this choice")


class LLMResponse(BaseModel):
    """Represents a complete response from an LLM."""
    choices: list[LLMChoice] = Field(..., description="List of response choices")
    usage: LLMUsage | None = Field(None, description="Token usage information")
    model: str = Field("", description="Model used for generation")
    id: str | None = Field(None, description="Unique identifier for the response")


class ImageGenerationRequest(BaseModel):
    """Represents a request for image generation."""
    prompt: str = Field(..., description="The text prompt for image generation")
    model: str = Field(..., description="The model to use for image generation")
    size: str | None = Field(None, description="Image size (e.g., '1024x1024')")
    quality: str | None = Field(None, description="Image quality (e.g., 'standard', 'hd')")
    style: str | None = Field(None, description="Image style (e.g., 'vivid', 'natural')")
    n: int = Field(1, description="Number of images to generate")
    response_format: str = Field("url", description="Response format ('url' or 'b64_json')")


class ImageGenerationData(BaseModel):
    """Represents a single generated image data."""
    url: str | None = Field(None, description="URL of the generated image")
    b64_json: str | None = Field(None, description="Base64 encoded image data")
    revised_prompt: str | None = Field(None, description="Revised prompt used for generation")


class ImageGenerationResponse(BaseModel):
    """Represents a response from image generation."""
    created: int = Field(..., description="Unix timestamp of creation")
    data: list[ImageGenerationData] = Field(..., description="List of generated images")
    usage: LLMUsage | None = Field(None, description="Token usage information")
    model: str = Field("", description="Model used for generation")
    id: str | None = Field(None, description="Unique identifier for the response")


class ModelSpeed(StrEnum):
    """Model speed options for generation."""
    FAST = "fast"
    SLOW = "slow"


# ============================================================================
# PROVIDER ENUMS AND CONFIGURATIONS
# ============================================================================

class LLMProvider(StrEnum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AWS_BEDROCK = "aws_bedrock"
    AZURE_OPENAI = "azure_openai"


class OpenAIModel(StrEnum):
    """Available OpenAI models."""
    FAST = "gpt-5-mini"
    SLOW = "gpt-5"


class OpenAIImageModel(StrEnum):
    """Available OpenAI image generation models."""
    GPT_IMAGE_1 = "gpt-image-1"   # Latest image model


class AnthropicModel(StrEnum):
    """Available Anthropic models."""
    FAST = "claude-sonnet-4-20250514"
    SLOW = "claude-opus-4-1-20250805"


class GeminiModel(StrEnum):
    """Available Google models (for Google AI Studio API)."""
    FAST = "gemini/gemini-2.0-flash-exp"
    SLOW = "gemini/gemini-2.5-pro-preview-03-25"


class GoogleImageModel(StrEnum):
    """Available Google image generation models."""
    IMAGEN_4_0 = "gemini/imagen-4.0-generate-preview-06-06"    # Latest image model


class AWSBedrockModel(StrEnum):
    """Available AWS Bedrock models (using the same Anthropic models)."""
    FAST = "anthropic.claude-3-5-haiku-20241022-v1:0"    # Fast
    SLOW = "us.anthropic.claude-sonnet-4-20250514-v1:0"    # Slow


class AzureModel(StrEnum):
    """Available Azure OpenAI models."""
    SLOW = "azure/gpt-4"
    FAST = "azure/gpt-4-turbo"


class BaseProviderConfig(BaseModel):
    """Base configuration for all LLM providers."""
    api_key: str = Field(description="API key for the provider")
    default_model: str = Field(description="Default model to use")
    fast_model: str = Field(description="Fast/cheap model for quick responses")
    slow_model: str = Field(description="Slow/expensive model for quality responses")
    fast_model_max_tokens: int | None = Field(default=None, description="Maximum tokens for fast model")
    slow_model_max_tokens: int | None = Field(default=None, description="Maximum tokens for slow model")
    max_tokens: int | None = Field(default=20000, description="Default maximum tokens for completion")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")

    class Config:
        validate_assignment = True


class OpenAIConfig(BaseProviderConfig):
    """OpenAI provider configuration."""
    api_key: str = Field(description="OpenAI API key")
    default_model: str = Field(default=OpenAIModel.FAST, description="Default OpenAI model")
    fast_model: str = Field(default=OpenAIModel.FAST, description="Fast OpenAI model")
    slow_model: str = Field(default=OpenAIModel.SLOW, description="Quality OpenAI model")
    organization_id: str | None = Field(default=None, description="OpenAI organization ID")
    base_url: str | None = Field(default=None, description="Custom base URL for OpenAI-compatible APIs")
    temperature: float = Field(default=1, ge=0.0, le=2.0, description="Temperature for generation")


class AnthropicConfig(BaseProviderConfig):
    """Anthropic provider configuration."""
    api_key: str = Field(description="Anthropic API key")
    default_model: str = Field(default=AnthropicModel.FAST, description="Default Anthropic model")
    fast_model: str = Field(default=AnthropicModel.FAST, description="Fast Anthropic model")
    slow_model: str = Field(default=AnthropicModel.SLOW, description="Quality Anthropic model")
    anthropic_version: str = Field(default="2023-06-01", description="Anthropic API version")


class GeminiConfig(BaseProviderConfig):
    """Google Gemini provider configuration."""
    api_key: str = Field(description="Google API key")
    default_model: str = Field(default=GeminiModel.FAST, description="Default Gemini model")
    fast_model: str = Field(default=GeminiModel.FAST, description="Fast Gemini model")
    slow_model: str = Field(default=GeminiModel.SLOW, description="Quality Gemini model")
    project_id: str | None = Field(default=None, description="Google Cloud project ID")
    location: str = Field(default="us-central1", description="Google Cloud location")


class AWSBedrockConfig(BaseProviderConfig):
    """AWS Bedrock provider configuration."""
    api_key: str = Field(description="AWS Bedrock Bearer Token")
    default_model: str = Field(default=AWSBedrockModel.FAST, description="Default Bedrock model")
    fast_model: str = Field(default=AWSBedrockModel.FAST, description="Fast Bedrock model")
    slow_model: str = Field(default=AWSBedrockModel.SLOW, description="Quality Bedrock model")
    fast_model_max_tokens: int | None = Field(default=8192, description="Maximum tokens for slow model")
    region_name: str = Field(default="us-west-2", description="AWS region")


class AzureOpenAIConfig(BaseProviderConfig):
    """Azure OpenAI provider configuration."""
    api_key: str = Field(description="Azure API key")
    default_model: str = Field(default=AzureModel.FAST, description="Default Azure model")
    fast_model: str = Field(default=AzureModel.FAST, description="Fast Azure model")
    slow_model: str = Field(default=AzureModel.SLOW, description="Quality Azure model")
    api_base: str | None = Field(default=None, description="Azure endpoint URL")
    api_version: str = Field(default="2024-02-15-preview", description="Azure API version")
    deployment_name: str | None = Field(default=None, description="Azure deployment name")


# ============================================================================
# PROVIDER FACTORY
# ============================================================================

class ProviderFactory:
    """Factory for creating provider configurations."""

    @staticmethod
    def create_openai(
        api_key: str | None = None,
        default_model: str | None = None,
        **kwargs: Any
    ) -> OpenAIConfig:
        """Create OpenAI configuration."""
        if not api_key:
            raise ValueError("api_key is required for OpenAI configuration")

        config_dict: dict[str, Any] = {"api_key": api_key}
        if default_model:
            config_dict["default_model"] = default_model
            config_dict["fast_model"] = default_model

        # Convert string values to proper types
        for key, value in kwargs.items():
            if key in ["max_tokens", "timeout", "max_retries"] and isinstance(value, str):
                config_dict[key] = int(value)
            elif key == "temperature" and isinstance(value, str):
                config_dict[key] = float(value)
            elif key == "extra_params" and isinstance(value, str):
                continue  # Skip string values for dict fields
            else:
                config_dict[key] = value

        return OpenAIConfig(**config_dict)

    @staticmethod
    def create_anthropic(
        api_key: str | None = None,
        default_model: str | None = None,
        **kwargs: Any
    ) -> AnthropicConfig:
        """Create Anthropic configuration."""
        if not api_key:
            raise ValueError("api_key is required for Anthropic configuration")

        config_dict: dict[str, Any] = {"api_key": api_key}
        if default_model:
            config_dict["default_model"] = default_model
            config_dict["fast_model"] = default_model

        # Convert string values to proper types
        for key, value in kwargs.items():
            if key in ["max_tokens", "timeout", "max_retries"] and isinstance(value, str):
                config_dict[key] = int(value)
            elif key == "temperature" and isinstance(value, str):
                config_dict[key] = float(value)
            elif key == "extra_params" and isinstance(value, str):
                continue  # Skip string values for dict fields
            else:
                config_dict[key] = value

        return AnthropicConfig(**config_dict)

    @staticmethod
    def create_gemini(
        api_key: str,
        default_model: str | None = None,
        **kwargs: Any
    ) -> GeminiConfig:
        """Create Gemini configuration."""
        config_dict: dict[str, Any] = {"api_key": api_key}
        if default_model:
            config_dict["default_model"] = default_model
            config_dict["fast_model"] = default_model

        # Convert string values to proper types
        for key, value in kwargs.items():
            if key in ["max_tokens", "timeout", "max_retries"] and isinstance(value, str):
                config_dict[key] = int(value)
            elif key == "temperature" and isinstance(value, str):
                config_dict[key] = float(value)
            elif key == "extra_params" and isinstance(value, str):
                continue  # Skip string values for dict fields
            else:
                config_dict[key] = value

        return GeminiConfig(**config_dict)

    @staticmethod
    def create_bedrock(
        api_key: str,
        region_name: str = "us-west-2",
        default_model: str | None = None,
        **kwargs: Any
    ) -> AWSBedrockConfig:
        """Create AWS Bedrock configuration."""
        config_dict: dict[str, Any] = {"api_key": api_key, "region_name": region_name}
        if default_model:
            config_dict["default_model"] = default_model
            config_dict["fast_model"] = default_model

        # Convert string values to proper types
        for key, value in kwargs.items():
            if key in ["max_tokens", "timeout", "max_retries"] and isinstance(value, str):
                config_dict[key] = int(value)
            elif key == "temperature" and isinstance(value, str):
                config_dict[key] = float(value)
            elif key == "extra_params" and isinstance(value, str):
                continue  # Skip string values for dict fields
            else:
                config_dict[key] = value

        return AWSBedrockConfig(**config_dict)

    @staticmethod
    def create_azure(
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str | None = None,
        **kwargs: Any
    ) -> AzureOpenAIConfig:
        """Create Azure OpenAI configuration."""
        if not api_key:
            raise ValueError("api_key is required for Azure OpenAI configuration")
        if not api_base:
            raise ValueError("api_base is required for Azure OpenAI configuration")

        config_dict: dict[str, Any] = {"api_key": api_key, "api_base": api_base}
        if default_model:
            config_dict["default_model"] = default_model
            config_dict["fast_model"] = default_model

        # Convert string values to proper types
        for key, value in kwargs.items():
            if key in ["max_tokens", "timeout", "max_retries"] and isinstance(value, str):
                config_dict[key] = int(value)
            elif key == "temperature" and isinstance(value, str):
                config_dict[key] = float(value)
            elif key == "extra_params" and isinstance(value, str):
                continue  # Skip string values for dict fields
            else:
                config_dict[key] = value

        return AzureOpenAIConfig(**config_dict)


# ============================================================================
# LLM SERVICE IMPLEMENTATION
# ============================================================================

class LLMService:
    """Simplified LLM service using LiteLLM's built-in features.

    Provides a unified interface for multiple LLM providers with automatic
    configuration, validation, and fallback support.
    """

    def __init__(
        self,
        config_service: ConfigService,
        provider: BaseProviderConfig
    ):
        """Initialize LLM service with dependency injection.

        Args:
            config_service: Configuration service instance (injected)
            provider: Provider configuration to use (injected)
        """
        self.config_service = config_service
        self.provider = provider

        # Ensure API key is populated from ConfigService secrets if missing
        self._populate_secrets_from_config()

        # Use provider's default model
        self.default_model = provider.default_model

        # Prepare auth params for litellm (avoid leaking secrets via environment)
        self._auth_params = self._build_auth_params()

        # Configure any provider-specific runtime requirements (no secret reads)
        self._configure_provider_runtime()

        # Setup environment variables for LiteLLM
        self._setup_environment()

        # Configure LiteLLM non-secret settings
        self._configure_litellm()

        logger.info(
            f"Initialized LLM service with provider: {provider.__class__.__name__}, model: {self.default_model}"
        )

    def _setup_environment(self) -> None:
        """Setup environment variables for LiteLLM using provider configuration."""
        if not self.provider or not self.provider.api_key:
            return

        # Set up environment variables based on provider type using match statement
        # LiteLLM expects these environment variables to be set
        match self.provider:
            case OpenAIConfig():
                os.environ["OPENAI_API_KEY"] = self.provider.api_key
                if self.provider.organization_id:
                    os.environ["OPENAI_ORGANIZATION"] = self.provider.organization_id
                if self.provider.base_url:
                    os.environ["OPENAI_API_BASE"] = self.provider.base_url

            case AnthropicConfig():
                os.environ["ANTHROPIC_API_KEY"] = self.provider.api_key

            case GeminiConfig():
                os.environ["GOOGLE_API_KEY"] = self.provider.api_key
                os.environ["GEMINI_API_KEY"] = self.provider.api_key
                if self.provider.project_id:
                    os.environ["GOOGLE_PROJECT_ID"] = self.provider.project_id

            case AWSBedrockConfig():
                # AWS Bedrock uses bearer token
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.provider.api_key
                if self.provider.region_name:
                    os.environ["AWS_REGION_NAME"] = self.provider.region_name

            case AzureOpenAIConfig():
                os.environ["AZURE_API_KEY"] = self.provider.api_key
                if self.provider.api_base:
                    os.environ["AZURE_API_BASE"] = self.provider.api_base
                if self.provider.api_version:
                    os.environ["AZURE_API_VERSION"] = self.provider.api_version
                if self.provider.deployment_name:
                    os.environ["AZURE_DEPLOYMENT_NAME"] = self.provider.deployment_name

            case _:
                logger.warning(f"Unknown provider type: {type(self.provider).__name__}")

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings using provider configuration (non-secret)."""
        # Configure basic settings
        litellm.drop_params = True  # Drop unsupported params

        # Set timeouts and retries via litellm global config if supported; fall back to env
        if self.provider:
            if hasattr(self.provider, "timeout") and self.provider.timeout:
                try:
                    litellm.request_timeout = int(self.provider.timeout)  # type: ignore[attr-defined]
                except Exception:
                    os.environ["LITELLM_REQUEST_TIMEOUT"] = str(self.provider.timeout)
            if hasattr(self.provider, "max_retries") and self.provider.max_retries:
                try:
                    litellm.num_retries = int(self.provider.max_retries)
                except Exception:
                    os.environ["LITELLM_NUM_RETRIES"] = str(self.provider.max_retries)

        logger.debug("Configured LiteLLM settings")

    def _populate_secrets_from_config(self) -> None:
        """Populate provider.api_key from ConfigService if not already set.

        This reads from secrets.yaml via ConfigService; no environment access.
        """
        if not self.provider or getattr(self.provider, "api_key", None):
            return

        # Try to resolve based on provider type
        if isinstance(self.provider, OpenAIConfig):
            api_key = self.config_service.get("llm_providers.openai.api_key")
            if api_key:
                self.provider.api_key = str(api_key)
        elif isinstance(self.provider, AnthropicConfig):
            api_key = self.config_service.get("llm_providers.anthropic.api_key")
            if api_key:
                self.provider.api_key = str(api_key)
        elif isinstance(self.provider, GeminiConfig):
            api_key = self.config_service.get("llm_providers.gemini.api_key")
            if api_key:
                self.provider.api_key = str(api_key)
        elif isinstance(self.provider, AWSBedrockConfig):
            api_key = self.config_service.get("llm_providers.aws_bedrock.api_key")
            if api_key:
                self.provider.api_key = str(api_key)
        elif isinstance(self.provider, AzureOpenAIConfig):
            api_key = self.config_service.get("AZURE_OPENAI_API_KEY")
            if api_key:
                self.provider.api_key = str(api_key)

    def _build_auth_params(self) -> dict[str, Any]:
        """Build provider-specific auth params to pass directly to LiteLLM calls.

        Avoids setting or reading environment variables for secrets.
        """
        params: dict[str, Any] = {}
        p = self.provider
        if not p:
            return params

        if isinstance(p, OpenAIConfig):
            params["api_key"] = p.api_key
            if p.base_url:
                params["api_base"] = p.base_url
            if p.organization_id:
                params["organization"] = p.organization_id
        elif isinstance(p, AnthropicConfig):
            params["api_key"] = p.api_key
        elif isinstance(p, GeminiConfig):
            params["api_key"] = p.api_key
        elif isinstance(p, AWSBedrockConfig):
            # AWS Bedrock uses environment variables for authentication
            # Don't pass bearer token directly to litellm completion call
            # Set region for bedrock
            params["aws_region_name"] = p.region_name
        elif isinstance(p, AzureOpenAIConfig):
            params["api_key"] = p.api_key
            if p.api_base:
                params["api_base"] = p.api_base
            if p.api_version:
                params["api_version"] = p.api_version
            if p.deployment_name:
                params["azure_deployment"] = p.deployment_name
        return params

    def _configure_provider_runtime(self) -> None:
        """Configure any provider-specific non-secret runtime settings.

        Keeps secrets out of environment variables.
        """
        # Currently no non-secret runtime setup needed.
        return

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        response_type: type[T] | None = None,
        model_speed: ModelSpeed | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        **kwargs: Any
    ) -> T:
        """Generate structured output using the configured LLM with retry logic and backoff.

        Args:
            prompt: The input prompt for completion
            response_type: Pydantic model class for response validation
            model_speed: Model speed (fast/slow), uses default if None
            max_retries: Maximum number of retry attempts (default: 5)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 60.0)
            backoff_factor: Multiplier for exponential backoff (default: 2.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            Parsed and validated response as instance of response_type

        Raises:
            ValueError: If all retry attempts fail or response cannot be parsed
        """
        if response_type is None:
            raise ValueError("response_type is required")

        # Determine which model to use
        model = self._get_model_for_speed(model_speed)

        # Get provider settings with model-specific max_tokens
        max_tokens = self._get_max_tokens_for_model(model_speed)
        temperature = getattr(self.provider, "temperature", None) if self.provider else None

        # Prepare messages
        if messages is not None:
            # Use provided messages and enhance the last user message with schema
            final_messages = messages.copy()
            if final_messages and final_messages[-1]["role"] == "user":
                # Enhance the last user message with structured output instructions
                last_content = final_messages[-1]["content"]
                enhanced_content = self._create_structured_prompt(last_content, response_type)
                final_messages[-1]["content"] = enhanced_content
            else:
                raise ValueError("Messages must end with a user message")
        elif prompt is not None:
            # Create single user message from prompt
            enhanced_prompt = self._create_structured_prompt(prompt, response_type)
            final_messages = [{"role": "user", "content": enhanced_prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Merge provider extra params with kwargs and auth params
        provider_params = getattr(self.provider, "extra_params", {}) if self.provider else {}
        params = {**self._auth_params, **provider_params, **kwargs}

        # Remove parameters from params if they exist to avoid duplicate parameter errors
        # We'll pass them explicitly below
        params.pop("temperature", None)
        params.pop("max_tokens", None)

        last_error = None
        delay = base_delay

        for attempt in range(max_retries):
            content = ""  # Initialize content for error handling
            try:
                logger.debug(f"Generation attempt {attempt + 1}/{max_retries}")

                # Call LiteLLM async
                response = await litellm.acompletion(
                    model=model,
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **params
                )

                # Extract response content
                content = self._extract_response_content(response)
                if not content:
                    raise ValueError("Empty response from LLM")

                # Parse and validate response
                parsed_response = self._parse_structured_response(content, response_type)
                logger.info(f"Successfully generated structured response on attempt {attempt + 1}")
                return parsed_response

            except Exception as e:
                last_error = e

                # Create detailed error message for validation failures
                error_details = str(e)
                if "validation error" in error_details.lower():
                    # Extract validation details for better logging
                    try:
                        from pydantic import ValidationError
                        if isinstance(e, ValidationError):
                            validation_details = []
                            for error in e.errors():
                                field_path = " -> ".join(str(loc) for loc in error["loc"])
                                validation_details.append(f"Field '{field_path}': {error['msg']} (type: {error['type']})")
                            error_details = f"Validation failed: {'; '.join(validation_details)}"
                    except Exception:
                        # If we can't parse validation details, use the original error
                        pass

                schema_fields = list(response_type.model_json_schema().get("properties", {}).keys())
                logger.warning(
                    f"Generation attempt {attempt + 1} failed: {error_details}",
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    model=model,
                    response_type=response_type.__name__,
                    content_preview=content[:200] + "..." if content and len(content) > 200 else content or "No content",
                    expected_schema_fields=", ".join(schema_fields)
                )

                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    # Add error feedback to the conversation for next attempt
                    error_message = f"The previous response was invalid: {error_details}. Please provide a valid JSON response matching the required schema."
                    final_messages.append({"role": "assistant", "content": content or "Invalid response"})
                    final_messages.append({"role": "user", "content": error_message})

                    # Exponential backoff with jitter
                    sleep_time = min(delay, max_delay)
                    logger.debug(f"Waiting {sleep_time:.2f} seconds before retry...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor

        # All retries failed
        error_msg = f"Failed to generate valid response after {max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _get_model_for_speed(self, model_speed: ModelSpeed | None) -> str:
        """Get the appropriate model based on speed preference."""
        if not model_speed:
            return self.default_model

        if model_speed == ModelSpeed.FAST:
            return getattr(self.provider, "fast_model", self.default_model)
        elif model_speed == ModelSpeed.SLOW:
            return getattr(self.provider, "slow_model", self.default_model)

        return self.default_model

    def _get_max_tokens_for_model(self, model_speed: ModelSpeed | None) -> int | None:
        """Get the appropriate max_tokens based on model speed.
        
        Returns model-specific max_tokens if configured, otherwise falls back to default.
        """
        if not self.provider:
            return None
            
        # If no model speed specified, use default max_tokens
        if not model_speed:
            return getattr(self.provider, "max_tokens", None)
        
        # Check for model-specific max_tokens first, then fall back to default
        if model_speed == ModelSpeed.FAST:
            fast_max_tokens = getattr(self.provider, "fast_model_max_tokens", None)
            return fast_max_tokens if fast_max_tokens is not None else getattr(self.provider, "max_tokens", None)
        else:  # ModelSpeed.SLOW or any other value
            slow_max_tokens = getattr(self.provider, "slow_model_max_tokens", None)
            return slow_max_tokens if slow_max_tokens is not None else getattr(self.provider, "max_tokens", None)

    def _create_structured_prompt(self, prompt: str, response_type: type[T]) -> str:
        """Create an enhanced prompt with JSON schema requirements."""
        schema = response_type.model_json_schema()

        enhanced_prompt = f"""{prompt}

CRITICAL: You must respond with valid JSON that matches this exact schema:

{json.dumps(schema, indent=2)}

REQUIRED FIELDS: All fields marked as required in the schema MUST be included in your response.
FIELD TYPES: Ensure all field values match the expected types (string, boolean, array, object, etc.).
VALIDATION: Your response will be validated against this schema - missing required fields will cause failure.

Your response must be valid JSON only, with no additional text, explanations, or markdown formatting."""

        return enhanced_prompt

    def _extract_response_content(self, response: Any) -> str:
        """Extract content from LiteLLM response."""
        choices = getattr(response, "choices", None)
        if not choices or len(choices) == 0:
            raise ValueError("No choices in LLM response")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if not message:
            raise ValueError("No message in LLM response choice")

        content = getattr(message, "content", None)
        if not content:
            raise ValueError("No content in LLM response message")

        return content.strip()

    def _parse_structured_response(self, content: str, response_type: type[T]) -> T:
        """Parse and validate JSON response against Pydantic model."""
        try:
            # Try to extract JSON from the response if it's wrapped in markdown
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            data = json.loads(json_str)

            # Validate and create Pydantic model instance
            return response_type(**data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")  # noqa: B904
        except Exception as e:
            raise ValueError(f"Failed to validate response against {response_type.__name__}: {e}")  # noqa: B904

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "high",
        n: int = 1,
        **kwargs: Any
    ) -> ImageGenerationResponse:
        """Generate images using the configured image generation provider.

        Args:
            prompt: The image generation prompt
            size: Image size (default: "1024x1024")
            quality: Image quality (default: "high")
            n: Number of images to generate (default: 1)
            **kwargs: Additional provider-specific parameters

        Returns:
            Image generation response with URLs or base64 data

        Raises:
            ValueError: If the provider does not support image generation
        """
        # Use IMAGE_GEN_PROVIDER instead of current provider for image generation
        image_provider = self.config_service.get("IMAGE_GEN_PROVIDER", "").lower()
        if image_provider:
            model, auth_params = self._get_image_provider_config(image_provider)
        else:
            # Check if current provider supports image generation
            if not self.supports_image_generation():
                raise ValueError(f"Provider {self.provider.__class__.__name__} does not support image generation and no IMAGE_GEN_PROVIDER configured")
            # Use current provider
            model = self._get_image_model_for_provider()
            auth_params = self._auth_params

        try:
            logger.info(f"Generating image with model: {model}")

            # Merge auth params and kwargs and call LiteLLM for image generation
            call_params = {**auth_params, **kwargs}
            response: Any = litellm.image_generation(  # type: ignore
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                **call_params
            )

            # Convert LiteLLM response to our ImageGenerationResponse format
            return self._convert_image_response(response, model)

        except Exception as e:
            logger.error(f"Error generating image with model {model}: {e}")
            raise

    def _get_image_model_for_provider(self) -> str:
        """Get the appropriate image model based on the provider type."""
        match self.provider:
            case OpenAIConfig():
                return OpenAIImageModel.GPT_IMAGE_1.value
            case GeminiConfig():
                return GoogleImageModel.IMAGEN_4_0.value
            case _:
                raise ValueError(f"Provider {self.provider.__class__.__name__} does not support image generation")

    def _convert_image_response(self, litellm_response: Any, model: str) -> ImageGenerationResponse:
        """Convert LiteLLM image response to our ImageGenerationResponse format."""
        try:
            # Extract data from LiteLLM response
            data = []
            if hasattr(litellm_response, "data") and litellm_response.data:
                for item in litellm_response.data:
                    data.append(ImageGenerationData(
                        url=getattr(item, "url", None),
                        b64_json=getattr(item, "b64_json", None),
                        revised_prompt=getattr(item, "revised_prompt", None)
                    ))

            # Skip usage field for now to avoid type conversion issues
            # LiteLLM usage objects may not match our LLMUsage type exactly

            return ImageGenerationResponse(
                created=getattr(litellm_response, "created", int(time.time())),
                data=data,
                usage=None,  # Skip usage for now to avoid type issues
                model=model,
                id=getattr(litellm_response, "id", None)
            )

        except Exception as e:
            logger.error(f"Error converting image response: {e}")
            # Return a basic response structure if conversion fails
            return ImageGenerationResponse(
                created=int(time.time()),
                data=[],  # Empty list as fallback
                usage=None,
                model=model,
                id=None
            )

    def supports_image_generation(self) -> bool:
        """Check if the current provider supports image generation."""
        # Only OpenAI and Gemini providers support image generation
        return isinstance(self.provider, (OpenAIConfig, GeminiConfig))

    def _get_image_provider_config(self, image_provider: str) -> tuple[str, dict[str, Any]]:
        """Get model and auth params for the configured image generation provider.
        
        Args:
            image_provider: The image generation provider name (from IMAGE_GEN_PROVIDER)
            
        Returns:
            Tuple of (model_name, auth_params)
            
        Raises:
            ValueError: If the image provider is not supported or not configured
        """
        if image_provider == "openai":
            api_key = self.config_service.get("llm_providers.openai.api_key")
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration.")
            
            auth_params = {"api_key": api_key}
            organization_id = self.config_service.get("llm_providers.openai.organization")
            base_url = self.config_service.get("llm_providers.openai.base_url")
            
            if organization_id:
                auth_params["organization"] = organization_id
            if base_url:
                auth_params["api_base"] = base_url
                
            return OpenAIImageModel.GPT_IMAGE_1.value, auth_params
            
        elif image_provider in {"gemini", "google"}:
            api_key = self.config_service.get("llm_providers.gemini.api_key")
            if not api_key:
                raise ValueError("Gemini API key not found in configuration.")
                
            auth_params = {"api_key": api_key}
            project_id = self.config_service.get("GOOGLE_PROJECT_ID")
            
            if project_id:
                auth_params["project_id"] = project_id
                
            return GoogleImageModel.IMAGEN_4_0.value, auth_params
            
        else:
            raise ValueError(f"Unsupported image generation provider: {image_provider}")
