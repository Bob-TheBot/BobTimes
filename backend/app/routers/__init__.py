# backend/app/routers/__init__.py

from ..api.news import router as news_router
from core.config_service import ConfigService
from fastapi import APIRouter

from .content import router as content_router
from .dev import dev_router

config_service = ConfigService()

router = APIRouter()


# Health check endpoint
@router.get("/api/v1/health")
async def health_check():
    """Health check endpoint for monitoring and testing"""
    import os

    # Debug print to see what environment is loaded
    env_info = { 
        "APP_ENV": os.getenv("APP_ENV"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "config_service_env": config_service.get_environment(),
        "config_service_testing": config_service.is_testing(),
    }
    print(f"[HEALTH DEBUG] Environment info: {env_info}")

    return {
        "status": "healthy",
        "service": "backend",
        "environment": config_service.get_environment(),
        "is_testing": config_service.is_testing(),
    }


# Include route definitions
router.include_router(dev_router, prefix="/api/v1/dev", tags=["development"])
router.include_router(content_router, prefix="/api/v1", tags=["content"])

# Import and include news API routes (for frontend queries only)

router.include_router(news_router, prefix="/api/v1", tags=["news"])
