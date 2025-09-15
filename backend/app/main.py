from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from core.config_service import settings
from core.logging_service import get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.routers import router as api_router

# Configure logging
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, Any]:
    # Startup logic
    logger.info("Starting application database setup")
    
    yield

    # Shutdown logic
    logger.info("Application shutting down")


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A FastAPI backend for apartment management and customer service chatbot",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Support Team",
        "url": "http://example.com/contact/",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# 1. Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# 2. CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,  # This is a property that returns a list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Welcome to the FastAPI application!"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    from core.config_service import ConfigService

    config_service = ConfigService()

    # Get host and port from configuration
    host = config_service.get("host", "0.0.0.0")
    port = config_service.get("port", 9200)

    logger.info(
        "Starting application server",
        host=host,
        port=port,
        environment=config_service.get_environment(),
    )

    uvicorn.run(app, host=host, port=port)
