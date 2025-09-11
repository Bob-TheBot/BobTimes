"""Simplified Configuration Service for managing application settings and secrets.

This module provides a centralized configuration service that loads settings
from environment variables and secrets files, with support for different
environments (development, production, testing).

The ConfigService focuses only on basic configuration management.
LLM provider configurations are now handled by the LLMService directly
to avoid circular dependencies.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigService:
    """Simplified configuration service that loads from environment variables and secrets.
    No complex Pydantic models or provider-specific configurations needed.
    LLM provider management has been moved to LLMService to avoid circular dependencies.
    """

    def __init__(self) -> None:
        """Initialize the configuration service."""
        self._env = os.getenv("APP_ENV", "development")
        self._secrets: dict[str, Any] = {}

        self._load_environment()
        self._load_secrets()

    def _load_environment(self) -> None:
        """Load environment variables from .env files."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        env_file = base_dir / f".env.{self._env}"

        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(f"Environment file not found: {env_file}")

    def _load_secrets(self) -> None:
        """Load secrets from secrets.yaml file."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        secrets_file = base_dir / "secrets.yaml"

        if secrets_file.exists():
            try:
                with open(secrets_file) as f:
                    self._secrets = yaml.safe_load(f) or {}
                logger.info("Loaded secrets from secrets.yaml")
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
                self._secrets = {}
        else:
            logger.info("No secrets.yaml file found")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'database.host')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        # First try environment variables
        env_value = os.getenv(key.upper().replace(".", "_"))
        if env_value is not None:
            return env_value

        # Then try secrets with dot notation
        try:
            value = self._secrets
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    # Environment helpers

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self._env.lower() == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self._env.lower() == "production"

    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self._env.lower() in ("test", "testing")

    def get_environment(self) -> str:
        """Get the current environment name."""
        return self._env


# Create a singleton instance
config_service = ConfigService()


# Determine the environment file path for Settings
def get_env_file_path() -> str:
    """Get the path to the environment file for Pydantic Settings."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    env = os.getenv("APP_ENV", "development")
    return str(base_dir / f".env.{env}")


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Type-safe application settings loaded from .env files.

    Only fields declared here are parsed; all other keys in the .env are ignored
    to avoid ValidationError: extra inputs are not permitted.
    """

    # Core app settings used by the backend
    PROJECT_NAME: str = "Bob Times"
    API_V1_STR: str = "/api/v1"
    # Comma-separated list of allowed CORS origins
    CORS_ORIGINS: str = "http://localhost:51273,http://localhost:3000"

    # Pydantic-settings v2 configuration
    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra keys present in the .env file
    )

    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        """Return CORS origins as a list for FastAPI middleware."""
        return [o.strip() for o in str(self.CORS_ORIGINS).split(",") if o.strip()]


# Create settings instance
settings = Settings()
