"""Middleware for logging HTTP requests and responses."""

import time
import uuid
from collections.abc import Awaitable, Callable

from core.logging_service import get_logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Get logger for this module
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.
    Logs request details, response status, and timing information.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Create a logger with request ID context
        req_logger = logger.bind(request_id=request_id)

        # Extract request details
        client_host = request.client.host if request.client else "unknown"
        request_path = request.url.path
        request_method = request.method

        # Log the request
        req_logger.info(
            f"Request started: {request_method} {request_path}",
            event="request_started",
            client_ip=client_host,
            method=request_method,
            path=request_path,
            query_params=str(request.query_params),
        )

        # Record start time
        start_time = time.time()

        try:
            # Process the request
            response: Response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log the response
            req_logger.info(
                f"Request completed: {request_method} {request_path} - {response.status_code}",
                event="request_completed",
                method=request_method,
                path=request_path,
                status_code=response.status_code,
                process_time_ms=int(round(process_time * 1000, 2)),
            )

            return response
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log the error
            req_logger.error(
                f"Request failed: {request_method} {request_path}",
                event="request_failed",
                method=request_method,
                path=request_path,
                error=str(e),
                process_time_ms=int(round(process_time * 1000, 2)),
                exc_info=True,
            )

            # Re-raise the exception
            raise
