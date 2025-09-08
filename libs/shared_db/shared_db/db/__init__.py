# Shared database configuration
import os

from core.config_service import ConfigService
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Initialize config service
config_service = ConfigService()
DATABASE_URL = config_service.get_database_url()

# For testing with multiple workers, use in-memory SQLite with shared cache
if config_service.is_testing() and DATABASE_URL.startswith("sqlite"):
    # Check if we should use in-memory database for parallel testing
    if os.getenv("PLAYWRIGHT_WORKERS", "1") != "1" or os.getenv("USE_MEMORY_DB", "false").lower() == "true":
        DATABASE_URL = "sqlite:///:memory:?cache=shared"
        # Enable WAL mode for better concurrency
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
            },
            pool_pre_ping=True,
            echo=False
        )
    else:
        # Use file-based SQLite for single worker testing
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            echo=False
        )
else:
    # Production/development database
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
