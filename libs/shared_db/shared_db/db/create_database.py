import sys

import psycopg2
from core.config_service import ConfigService
from core.logging_service import get_logger
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Get logger for this module
logger = get_logger(__name__)


def create_database() -> bool | None:
    """Create the database if it doesn't exist."""
    # Get database connection parameters from config
    config_service = ConfigService()
    db_url = config_service.get_database_url()

    # Parse the database URL to get connection parameters
    # Format: postgresql://username:password@host:port/database
    db_url_parts = db_url.replace("postgresql://", "").split("/")
    connection_string = db_url_parts[0]
    db_name = db_url_parts[1] if len(db_url_parts) > 1 else "mydatabase"

    # Parse connection string to get user, password, host, port
    user_pass, host_port = connection_string.split("@")
    user, password = user_pass.split(":") if ":" in user_pass else (user_pass, "")
    host, port = host_port.split(":") if ":" in host_port else (host_port, 5432)

    db_params = {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
    }

    logger.info(
        "Database creation parameters",
        database_name=db_name,
        host=host,
        port=port,
    )

    try:
        # Connect to PostgreSQL server (to postgres database)
        logger.info(
            "Connecting to PostgreSQL server",
            host=db_params["host"],
            port=db_params["port"],
        )
        conn = psycopg2.connect(**db_params, database="postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (db_name,),
        )
        exists = cursor.fetchone()

        if not exists:
            logger.info("Creating database", database_name=db_name)
            # Use identifier quoting for safety
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(
                "Database created successfully",
                database_name=db_name,
                status="created",
            )
        else:
            logger.info(
                "Database already exists",
                database_name=db_name,
                status="exists",
            )

        cursor.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        logger.exception(
            "PostgreSQL error during database creation",
            error=str(e),
            error_code=e.pgcode if hasattr(e, "pgcode") else None,
        )
        return False
    except Exception as e:
        logger.exception("Unexpected error during database creation", error=str(e))
        return False


if __name__ == "__main__":
    # This allows the script to be run directly
    success = create_database()
    sys.exit(0 if success else 1)
