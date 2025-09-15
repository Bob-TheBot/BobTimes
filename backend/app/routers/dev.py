"""Development-only router for testing and debugging.
Only available in development mode.
"""

from typing import Any

from core.config_service import ConfigService
from core.logging_service import get_logger
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = get_logger(__name__)
config_service = ConfigService()

# Only create router if in development mode
dev_router = APIRouter() if config_service.is_development() else None


class HelloResponse(BaseModel):
    """Response schema for hello world endpoint"""

    message: str
    environment: str
    timestamp: str


def check_development_mode() -> None:
    """Dependency to ensure endpoint is only available in development"""
    if not config_service.is_development():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Development endpoints not available in production",
        )


if dev_router is not None:

    @dev_router.get("/hello", response_model=HelloResponse)
    async def hello_world() -> HelloResponse:
        """Simple hello world endpoint for development testing.
        Only available in development mode.
        """
        from datetime import datetime

        logger.info("Hello world endpoint called")

        return HelloResponse(
            message="Hello World from Development API!",
            environment="development",
            timestamp=datetime.now().isoformat(),
        )

    @dev_router.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint for development.
        Only available in development mode.
        """
        logger.info("Health check endpoint called")

        return {
            "status": "healthy",
            "message": "Development API is running",
            "environment": "development",
        }

else:
    # Create empty router for production
    dev_router = APIRouter()

    @dev_router.get("/", response_model=None)
    async def dev_not_available():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Development endpoints not available in production",
        )
