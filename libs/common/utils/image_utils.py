"""Image utilities for downloading, validating, and storing images.

Provides shared functionality for handling images from search results
and AI generation, including local storage and size validation.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import requests
from core.logging_service import get_logger

logger = get_logger(__name__)


def ensure_image_directories() -> dict[str, Path]:
    """Create and return paths for image storage directories.
    
    Returns:
        Dictionary with 'generated' and 'downloaded' Path objects
    """
    base_dir = Path("/workspaces/bobtimes/data/images")
    generated_dir = base_dir / "generated"
    downloaded_dir = base_dir / "downloaded"
    
    # Create directories if they don't exist
    generated_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        "Ensured image directories exist",
        generated=str(generated_dir),
        downloaded=str(downloaded_dir)
    )
    
    return {
        "generated": generated_dir,
        "downloaded": downloaded_dir,
    }


def create_image_filename(source: str = "search", extension: str = ".jpg", story_id: str | None = None) -> str:
    """Generate a unique filename for an image.
    
    Args:
        source: Source type ('search' or 'generated')
        extension: File extension (default .jpg)
        story_id: Optional story ID to include in filename for better tracking
    
    Returns:
        Unique filename with timestamp and hash, optionally including story ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add a short hash for uniqueness
    base_string = f"{timestamp}{source}{story_id or ''}"
    unique_hash = hashlib.md5(base_string.encode()).hexdigest()[:8]
    
    if story_id:
        # Include story ID in filename for easier tracking
        filename = f"{source}_{story_id}_{timestamp}_{unique_hash}{extension}"
    else:
        filename = f"{source}_{timestamp}_{unique_hash}{extension}"
    return filename


def download_and_save_image(url: str, save_dir: Path, timeout: int = 30, story_id: str | None = None) -> tuple[Path | None, int]:
    """Download an image from URL and save it locally.
    
    Args:
        url: Image URL to download
        save_dir: Directory to save the image
        timeout: Request timeout in seconds
        story_id: Optional story ID to include in filename
    
    Returns:
        Tuple of (local_path, file_size_bytes) or (None, 0) if failed
    """
    try:
        logger.info(f"Downloading image from URL: {url[:100]}...")
        
        # Set headers to appear as a regular browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Determine file extension from content type
        content_type = response.headers.get("Content-Type", "")
        extension = ".jpg"  # default
        if "png" in content_type:
            extension = ".png"
        elif "webp" in content_type:
            extension = ".webp"
        elif "gif" in content_type:
            extension = ".gif"
        elif "jpeg" in content_type or "jpg" in content_type:
            extension = ".jpg"
        
        # Generate filename based on source type
        source_type = "downloaded" if "downloaded" in str(save_dir) else "generated"
        filename = create_image_filename(source=source_type, extension=extension, story_id=story_id)
        file_path = save_dir / filename
        
        # Download and save the image
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        logger.info(
            "Image downloaded successfully",
            path=str(file_path),
            size_kb=file_size // 1024,
            url=url[:100]
        )
        
        return file_path, file_size
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download image from {url[:100]}: {e}")
        return None, 0
    except Exception as e:
        logger.exception(f"Unexpected error downloading image from {url[:100]}: {e}")
        return None, 0


def validate_image_size(file_path: Path, min_size_kb: int = 200) -> bool:
    """Check if an image file meets the minimum size requirement.
    
    Args:
        file_path: Path to the image file
        min_size_kb: Minimum size in kilobytes (default 200 KB)
    
    Returns:
        True if image meets size requirement, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"Image file does not exist: {file_path}")
            return False
        
        file_size_kb = file_path.stat().st_size / 1024
        is_valid = file_size_kb >= min_size_kb
        
        logger.debug(
            "Image size validation",
            path=str(file_path),
            size_kb=int(file_size_kb),
            min_size_kb=min_size_kb,
            is_valid=is_valid
        )
        
        return is_valid
        
    except Exception as e:
        logger.exception(f"Error validating image size for {file_path}: {e}")
        return False


def download_image_from_url(url: str, source_type: str = "search", story_id: str | None = None) -> tuple[str | None, int]:
    """Download an image and store it in the appropriate directory.
    
    This is a convenience function that combines directory creation,
    downloading, and validation.
    
    Args:
        url: Image URL to download
        source_type: Either 'search' or 'generated'
        story_id: Optional story ID to include in filename
    
    Returns:
        Tuple of (local_path_string, file_size_kb) or (None, 0) if failed
    """
    try:
        # Ensure directories exist
        dirs = ensure_image_directories()
        
        # Determine save directory based on source type
        if source_type == "generated":
            save_dir = dirs["generated"]
        else:
            save_dir = dirs["downloaded"]
        
        # Download the image
        file_path, file_size = download_and_save_image(url, save_dir, story_id=story_id)
        
        if file_path and file_size > 0:
            # Validate size (200 KB minimum)
            if validate_image_size(file_path, min_size_kb=200):
                return str(file_path), file_size // 1024
            else:
                logger.info(f"Image too small, removing: {file_path}")
                file_path.unlink()  # Delete small image
                return None, 0
        
        return None, 0
        
    except Exception as e:
        logger.exception(f"Error in download_image_from_url: {e}")
        return None, 0