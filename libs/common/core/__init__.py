# Simplified core module - removed complex factory patterns

from .config_service import ConfigService, config_service
from .llm_service import LLMService, ModelSpeed

__all__ = [
    "ConfigService",
    "LLMService",
    "ModelSpeed",
    "config_service",
]
