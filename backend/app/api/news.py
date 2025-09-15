"""API endpoints for news content queries (frontend to backend)."""

import base64

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from utils.image_utils import ensure_image_directories

from ..dependencies import get_newspaper_service
from ..models.newspaper_models import NewspaperContentResponse
from ..services.newspaper_service import NewspaperService

router = APIRouter(prefix="/news", tags=["news"])


class ImageResponse(BaseModel):
    """Response model for image data."""
    image_data: str  # base64 encoded image
    mime_type: str
    filename: str


@router.get("/content", response_model=NewspaperContentResponse)
async def get_newspaper_content(
    newspaper_service: NewspaperService = Depends(get_newspaper_service)
) -> NewspaperContentResponse:
    """Get current newspaper content for the frontend.

    Args:
        newspaper_service: Newspaper service dependency

    Returns:
        Current newspaper content
    """
    try:
        content = await newspaper_service.get_newspaper_content()
        return NewspaperContentResponse.from_newspaper_content(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{image_path:path}", response_model=ImageResponse)
async def get_image_base64(image_path: str) -> ImageResponse:
    """Get an image file as base64 encoded data.

    Args:
        image_path: Path to the image file (relative to data/images/)

    Returns:
        Base64 encoded image data with metadata
    """
    try:
        # Get image directories to determine base path
        image_dirs = ensure_image_directories()
        base_image_dir = image_dirs["downloaded"].parent  # This gives us /workspaces/bobtimes/data/images

        # Construct full path to image
        full_path = base_image_dir / image_path

        # Security check: ensure path is within images directory
        if not str(full_path.resolve()).startswith(str(base_image_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid image path")

        # Check if file exists
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")

        # Read and encode image
        with open(full_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Determine MIME type based on file extension
        extension = full_path.suffix.lower()
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        mime_type = mime_type_map.get(extension, "image/jpeg")

        return ImageResponse(
            image_data=image_data,
            mime_type=mime_type,
            filename=full_path.name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")
