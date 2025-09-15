"""Centralized logging service for the application.
Provides structured JSON logging with configurable output destinations and log levels.
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Any

from core.config_service import ConfigService
from loguru import logger as loguru_logger
from pydantic import BaseModel


class LogLevel(str, Enum):
    """Log levels supported by the logging service."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogConfig(BaseModel):
    """Configuration for the logging service."""

    level: LogLevel = LogLevel.INFO
    json_format: bool = True
    console_output: bool = True
    file_output: bool = False
    log_file_path: str | None = None
    rotation: str = "20 MB"  # Size at which to rotate log files
    retention: str = "1 week"  # How long to keep log files
    compression: str = "zip"  # Compression format for rotated logs


class LoggingService:
    """Centralized logging service for the application.
    Provides structured JSON logging with configurable output destinations and log levels.
    """

    def __init__(self, config: LogConfig | None = None) -> None:
        """Initialize the logging service with the given configuration.
        If no configuration is provided, it will be loaded from the config service.
        """
        self.config = config or self._load_config_from_service()
        self._configure_loguru()

    def _load_config_from_service(self) -> LogConfig:
        """Load logging configuration from the config service."""
        config_service = ConfigService()
        return LogConfig(
            level=LogLevel(config_service.get("log_level", "INFO").upper()),
            json_format=config_service.get("log_json_format", True),
            console_output=config_service.get("log_console_output", True),
            file_output=config_service.get("log_file_output", False),
            log_file_path=config_service.get("log_file_path"),
            rotation=config_service.get("log_rotation", "20 MB"),
            retention=config_service.get("log_retention", "1 week"),
            compression=config_service.get("log_compression", "zip"),
        )

    def _configure_loguru(self) -> None:
        """Configure loguru with the current settings."""
        # Remove default handlers
        loguru_logger.remove()

        # Define the log format based on configuration
        if self.config.json_format:
            # For JSON format, we'll use serialize=True and a custom function to format the JSON
            log_format = "{message}"
        else:
            # For human-readable format, we need to include extra fields in the message itself
            # We'll use a custom format that shows extra fields
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

        # Add console handler if enabled
        if self.config.console_output:
            loguru_logger.add(
                sys.stdout,
                format=log_format,
                level=self.config.level.value,
                serialize=self.config.json_format,
                backtrace=True,
                diagnose=True,
            )

        # Add file handler if enabled
        if self.config.file_output and self.config.log_file_path:
            log_path = Path(self.config.log_file_path)

            # Create directory if it doesn't exist
            # Handle case where 'logs' might exist as a file instead of directory
            if log_path.parent.exists() and not log_path.parent.is_dir():
                # Remove the file and create directory
                log_path.parent.unlink()
            log_path.parent.mkdir(parents=True, exist_ok=True)

            loguru_logger.add(
                str(log_path),
                format=log_format,
                level=self.config.level.value,
                serialize=self.config.json_format,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                backtrace=True,
                diagnose=True,
            )

    def _serialize_record(self, record: Any) -> dict[str, object]:
        """Serialize a log record for JSON output.

        This is used by loguru's serialize parameter to convert the record to a dict
        that can be serialized to JSON.
        """
        # Basic log data
        serialized: dict[str, object] = {
            "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "process_id": record["process"].id,
            "thread_id": record["thread"].id,
        }

        # Add exception info if available
        if record["exception"] is not None:
            serialized["exception"] = {
                "type": record["exception"].type,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }

        # Add all extra fields at the top level
        if record["extra"]:
            for key, value in record["extra"].items():
                # Don't overwrite standard fields
                if key not in serialized:
                    serialized[key] = value

        return serialized

    def get_logger(self, name: str) -> "Logger":
        """Get a logger for the given name.

        Args:
            name: The name of the logger, typically the module name.

        Returns:
            A Logger instance configured with the current settings.
        """
        return Logger(name, self.config)

    def update_config(self, config: LogConfig) -> None:
        """Update the logging configuration.

        Args:
            config: The new configuration to apply.
        """
        self.config = config
        self._configure_loguru()


class Logger:
    """Logger class that wraps loguru logger with additional functionality."""

    def __init__(self, name: str, config: LogConfig) -> None:
        """Initialize a logger with the given name and configuration.

        Args:
            name: The name of the logger, typically the module name.
            config: The logging configuration to use.
        """
        self.name = name
        self.config = config
        self.logger = loguru_logger.bind(name=name)

    def _format_message_with_extras(self, message: str, **kwargs: str | int | bool) -> str:
        """Format message with extra fields for human-readable output."""
        if not kwargs or self.config.json_format:
            return message

        extra_parts = []
        for key, value in kwargs.items():
            extra_parts.append(f"{key}={value}")

        if extra_parts:
            return f"{message} | {', '.join(extra_parts)}"
        return message

    def _add_agent_context(self, **kwargs: str | int | bool) -> dict[str, str | int | bool]:
        """Add agent context to log kwargs if available."""
        import inspect

        # Try to find agent context from the call stack
        frame = inspect.currentframe()
        agent_context: dict[str, str | int | bool] = {}

        try:
            # Look through the call stack for agent instances
            while frame:
                frame = frame.f_back
                if not frame:
                    break

                # Check if we're in an agent method
                frame_locals = frame.f_locals
                if 'self' in frame_locals:
                    obj = frame_locals['self']

                    # Check for reporter agent
                    if hasattr(obj, 'reporter_id') and hasattr(obj, 'field'):
                        agent_context['agent_id'] = str(obj.reporter_id)
                        agent_context['agent_type'] = 'REPORTER'
                        agent_context['agent_field'] = str(obj.field.value)
                        break
                    # Check for editor agent
                    elif hasattr(obj, 'editor_id'):
                        agent_context['agent_id'] = str(obj.editor_id)
                        agent_context['agent_type'] = 'EDITOR'
                        break
                    # Check for base agent
                    elif hasattr(obj, '__class__') and 'Agent' in obj.__class__.__name__:
                        agent_context['agent_type'] = str(obj.__class__.__name__.replace('Agent', '').upper())
                        if hasattr(obj, 'reporter_id'):
                            agent_context['agent_id'] = str(obj.reporter_id)
                        elif hasattr(obj, 'editor_id'):
                            agent_context['agent_id'] = str(obj.editor_id)
                        break
        except Exception:
            # If anything goes wrong with stack inspection, just continue
            pass
        finally:
            del frame

        # Merge agent context with provided kwargs (kwargs take precedence)
        result: dict[str, str | int | bool] = {**agent_context, **kwargs}
        return result

    def debug(self, message: str, **kwargs: str | int | bool) -> None:
        """Log a debug message."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).debug(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).debug(formatted_message)

    def info(self, message: str, **kwargs: str | int | bool) -> None:
        """Log an info message."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).info(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).info(formatted_message)

    def warning(self, message: str, **kwargs: str | int | bool) -> None:
        """Log a warning message."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).warning(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).warning(formatted_message)

    def error(self, message: str, **kwargs: str | int | bool) -> None:
        """Log an error message."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).error(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).error(formatted_message)

    def critical(self, message: str, **kwargs: str | int | bool) -> None:
        """Log a critical message."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).critical(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).critical(formatted_message)

    def exception(self, message: str, **kwargs: str | int | bool) -> None:
        """Log an exception message with traceback."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).exception(message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).exception(formatted_message)

    def log(self, level: str | int, message: str, **kwargs: str | int | bool) -> None:
        """Log a message with the specified level."""
        enhanced_kwargs = self._add_agent_context(**kwargs)
        if self.config.json_format:
            # Use depth=1 to capture the caller's frame instead of this method
            self.logger.opt(depth=1).log(level, message, **enhanced_kwargs)
        else:
            formatted_message = self._format_message_with_extras(message, **enhanced_kwargs)
            self.logger.opt(depth=1).log(level, formatted_message)

    def bind(self, **kwargs: str | int | bool) -> "Logger":
        """Bind contextual information to the logger.

        Args:
            **kwargs: Key-value pairs to bind to the logger.

        Returns:
            A new Logger instance with the bound context.
        """
        new_logger = Logger(self.name, self.config)
        # Bind the kwargs to the logger
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger


# Create a singleton instance of the logging service
logging_service = LoggingService()


def get_logger(name: str) -> Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, typically the module name.

    Returns:
        A Logger instance configured with the current settings.
    """
    return logging_service.get_logger(name)
