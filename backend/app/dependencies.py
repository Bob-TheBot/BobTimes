from collections.abc import Generator

from core.config_service import ConfigService
from core.llm_service import LLMService
from fastapi import Depends
from sqlalchemy.orm import Session

from shared_db.db import SessionLocal

from .services.newspaper_service import NewspaperService


def get_db() -> Generator[Session]:
    """Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_config_service() -> ConfigService:
    """Dependency for configuration service."""
    return ConfigService()


def get_default_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for default LLM service."""
    from core.llm_service import ProviderFactory

    # Get the provider name from config
    llm_provider = config_service.get("LLM_PROVIDER", "").lower()
    if not llm_provider:
        raise ValueError("No LLM provider configured. Please set LLM_PROVIDER in environment.")

    # Create provider configuration based on the provider name
    if llm_provider == "openai":
        api_key = config_service.get("llm_providers.openai.api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in configuration.")
        provider = ProviderFactory.create_openai(
            api_key=api_key,
            organization_id=config_service.get("llm_providers.openai.organization"),
            base_url=config_service.get("llm_providers.openai.base_url")
        )
    elif llm_provider == "anthropic":
        api_key = config_service.get("llm_providers.anthropic.api_key")
        if not api_key:
            raise ValueError("Anthropic API key not found in configuration.")
        provider = ProviderFactory.create_anthropic(api_key=api_key)
    elif llm_provider in {"gemini", "google"}:
        api_key = config_service.get("llm_providers.gemini.api_key")
        if not api_key:
            raise ValueError("Gemini API key not found in configuration.")
        provider = ProviderFactory.create_gemini(
            api_key=api_key,
            project_id=config_service.get("GOOGLE_PROJECT_ID")
        )
    elif llm_provider in {"bedrock", "aws_bedrock"}:
        api_key = config_service.get("llm_providers.aws_bedrock.api_key")
        if not api_key:
            raise ValueError("AWS Bedrock API key not found in configuration.")
        provider = ProviderFactory.create_bedrock(
            api_key=api_key,
            region_name=config_service.get("AWS_REGION", "us-west-2")
        )
    elif llm_provider in {"azure", "azure_openai"}:
        api_key = config_service.get("AZURE_OPENAI_API_KEY")
        api_base = config_service.get("AZURE_OPENAI_ENDPOINT") or config_service.get("AZURE_OPENAI_BASE")
        if not api_key or not api_base:
            raise ValueError("Azure OpenAI API key and base URL not found in configuration.")
        provider = ProviderFactory.create_azure(
            api_key=api_key,
            api_base=api_base,
            api_version=config_service.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            deployment_name=config_service.get("AZURE_OPENAI_DEPLOYMENT")
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_openai_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for OpenAI-specific LLM service."""
    from core.llm_service import ProviderFactory

    api_key = config_service.get("llm_providers.openai.api_key")
    if not api_key:
        raise ValueError("OpenAI API key not found in configuration.")

    provider = ProviderFactory.create_openai(
        api_key=api_key,
        organization_id=config_service.get("llm_providers.openai.organization"),
        base_url=config_service.get("llm_providers.openai.base_url")
    )

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_anthropic_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for Anthropic-specific LLM service."""
    from core.llm_service import ProviderFactory

    api_key = config_service.get("llm_providers.anthropic.api_key")
    if not api_key:
        raise ValueError("Anthropic API key not found in configuration.")

    provider = ProviderFactory.create_anthropic(api_key=api_key)

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_gemini_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for Gemini-specific LLM service."""
    from core.llm_service import ProviderFactory

    api_key = config_service.get("llm_providers.gemini.api_key")
    if not api_key:
        raise ValueError("Gemini API key not found in configuration.")

    provider = ProviderFactory.create_gemini(
        api_key=api_key,
        project_id=config_service.get("GOOGLE_PROJECT_ID")
    )

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_aws_bedrock_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for AWS Bedrock-specific LLM service."""
    from core.llm_service import ProviderFactory

    api_key = config_service.get("llm_providers.aws_bedrock.api_key")
    if not api_key:
        raise ValueError("AWS Bedrock API key not found in configuration.")

    provider = ProviderFactory.create_bedrock(
        api_key=api_key,
        region_name=config_service.get("AWS_REGION", "us-west-2")
    )

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_azure_openai_llm_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for Azure OpenAI-specific LLM service."""
    from core.llm_service import ProviderFactory

    api_key = config_service.get("AZURE_OPENAI_API_KEY")
    api_base = config_service.get("AZURE_OPENAI_ENDPOINT") or config_service.get("AZURE_OPENAI_BASE")
    if not api_key or not api_base:
        raise ValueError("Azure OpenAI API key and base URL not found in configuration.")

    provider = ProviderFactory.create_azure(
        api_key=api_key,
        api_base=api_base,
        api_version=config_service.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=config_service.get("AZURE_OPENAI_DEPLOYMENT")
    )

    return LLMService(
        config_service=config_service,
        provider=provider
    )


def get_fast_llm_service(
    default_service: LLMService = Depends(get_default_llm_service)
) -> LLMService:
    """Dependency for fast LLM service (uses default service with fast model preference)."""
    # The LLMService can use ModelSpeed.FAST when generating
    return default_service


def get_slow_llm_service(
    default_service: LLMService = Depends(get_default_llm_service)
) -> LLMService:
    """Dependency for slow LLM service (uses default service with slow model preference)."""
    # The LLMService can use ModelSpeed.SLOW when generating
    return default_service


def get_image_generation_service(
    config_service: ConfigService = Depends(get_config_service)
) -> LLMService:
    """Dependency for image generation service."""
    from core.llm_service import ProviderFactory

    # Get the image provider name from config
    image_provider = config_service.get("IMAGE_GEN_PROVIDER", "").lower()
    if not image_provider:
        raise ValueError("No image generation provider configured. Please set IMAGE_GEN_PROVIDER in environment.")

    # Create provider configuration for image generation
    if image_provider == "openai":
        api_key = config_service.get("llm_providers.openai.api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in configuration.")
        provider = ProviderFactory.create_openai(api_key=api_key)
    elif image_provider in {"gemini", "google"}:
        api_key = config_service.get("llm_providers.gemini.api_key")
        if not api_key:
            raise ValueError("Gemini API key not found in configuration.")
        provider = ProviderFactory.create_gemini(api_key=api_key)
    else:
        raise ValueError(f"Unsupported image generation provider: {image_provider}")

    return LLMService(
        config_service=config_service,
        provider=provider
    )


# Singleton instance of NewspaperService
_newspaper_service = None


def get_newspaper_service(
    config_service: ConfigService = Depends(get_config_service),
    image_service: LLMService = Depends(get_image_generation_service)
) -> NewspaperService:
    """Dependency for newspaper service (singleton)."""
    global _newspaper_service
    if _newspaper_service is None:
        _newspaper_service = NewspaperService(
            config_service=config_service,
            llm_service=image_service
        )
    return _newspaper_service


# Backward compatibility alias
def get_agent_service(
    config_service: ConfigService = Depends(get_config_service)
) -> NewspaperService:
    """Dependency for agent service (singleton) - backward compatibility alias."""
    return get_newspaper_service(config_service)
