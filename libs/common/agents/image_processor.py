"""Post-agent image processing for story drafts.

This module handles image selection and generation after the reporter agent
completes its task, ensuring the editor gets a complete story with images.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any

from agents.editor_agent.image_tool import ImageGenerationTool, ImageToolParams
from agents.models.search_models import SearchResultSummary
from agents.models.story_models import ImageSourceType, StoryDraft, StoryImage
from core.llm_service import LLMService
from core.logging_service import get_logger

logger = get_logger(__name__)


def convert_image_to_base64(image_path: str | Path) -> tuple[str, str] | None:
    """Convert a local image file to base64 data URL format.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        Tuple of (base64_data, mime_type) or None if conversion fails
    """
    try:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.warning(f"Image file not found: {image_path}")
            return None

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            logger.warning(f"Invalid or unknown image type: {mime_type}")
            return None

        # Read and encode image
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")

        logger.debug(f"Converted image to base64: {path.name}, size: {len(base64_data)} chars")
        return base64_data, mime_type

    except Exception as e:
        logger.exception(f"Failed to convert image to base64: {image_path}, error: {e}")
        return None


async def process_story_images(
    story_draft: StoryDraft,
    search_results: list[SearchResultSummary],
    llm_service: LLMService
) -> None:
    """Process images for a story draft after agent completion.
    
    This function:
    1. Checks if story already has images
    2. Looks for large images (>200KB) from search results
    3. Selects the largest available image
    4. Generates an image if no suitable images found
    5. Adds the selected/generated image to the story
    
    Args:
        story_draft: The story draft to process images for
        search_results: List of search results from the agent's execution
        llm_service: LLM service for image generation if needed
    """
    # Check if story already has images
    if story_draft.suggested_images and len(story_draft.suggested_images) > 0:
        logger.info("Story already has images, skipping image processing")
        return

    # Find the best available image from search results
    best_image_result = None
    largest_size = 0

    for search_result in search_results:
        # Get the largest image from this search result
        largest_image = search_result.get_largest_image()
        if largest_image and largest_image.image_size_kb:
            if largest_image.image_size_kb > largest_size:
                largest_size = largest_image.image_size_kb
                best_image_result = largest_image

    if best_image_result and best_image_result.image_local_path:
        # Use the best image from search results
        base64_result = convert_image_to_base64(best_image_result.image_local_path)

        story_image = StoryImage(
            url=best_image_result.image_url or "",
            local_path=best_image_result.image_local_path,
            base64_data=base64_result[0] if base64_result else None,
            mime_type=base64_result[1] if base64_result else None,
            caption=f"Image related to: {story_draft.title}",
            is_generated=False,
            source_type=ImageSourceType.SEARCH,
            file_size_kb=best_image_result.image_size_kb,
        )
        story_draft.suggested_images.append(story_image)

        logger.info(
            "Added search result image to story with base64 data",
            local_path=story_image.local_path or "unknown",
            size_kb=story_image.file_size_kb or 0,
            has_base64=story_image.base64_data is not None,
            mime_type=story_image.mime_type or "unknown",
            title=story_draft.title[:50]
        )
    else:
        # No suitable image found, generate one
        logger.info("No suitable image from search results, generating image for story")

        # Create a prompt based on the story title and content
        image_prompt = f"News illustration for: {story_draft.title}"
        if len(image_prompt) > 200:
            image_prompt = image_prompt[:200]

        # Use the image generation tool
        image_tool = ImageGenerationTool(llm_service)
        image_params = ImageToolParams(
            prompt=image_prompt,
            n=1
        )

        try:
            result = await image_tool.execute(image_params)
            if result.success and result.images:
                generated_image = result.images[0]

                # Convert generated image to base64 if local path exists
                base64_result = None
                if generated_image.local_path:
                    base64_result = convert_image_to_base64(generated_image.local_path)

                story_image = StoryImage(
                    url=generated_image.url,
                    local_path=generated_image.local_path,
                    base64_data=base64_result[0] if base64_result else None,
                    mime_type=base64_result[1] if base64_result else None,
                    caption=f"Illustration for {story_draft.title}",
                    is_generated=True,
                    source_type=ImageSourceType.GENERATED,
                )
                story_draft.suggested_images.append(story_image)
                logger.info(
                    "Generated and added image to story with base64 data",
                    local_path=generated_image.local_path or "None",
                    url=generated_image.url[:100],
                    has_base64=story_image.base64_data is not None,
                    mime_type=story_image.mime_type or "unknown",
                    title=story_draft.title[:50]
                )
            else:
                logger.warning("Failed to generate image for story")
        except Exception as e:
            logger.exception(f"Error generating image for story: {e}")


def extract_search_results_from_messages(messages: list[dict[str, Any]]) -> list[SearchResultSummary]:
    """Extract SearchResultSummary objects from agent message history.
    
    Args:
        messages: Message history from agent execution
        
    Returns:
        List of SearchResultSummary objects found in messages
    """
    search_results:list[SearchResultSummary] = []

    # This is a simplified implementation - in a real scenario, we would need
    # to properly reconstruct SearchResultSummary objects from the message content
    # For now, we'll return an empty list and rely on the image generation fallback

    logger.debug(f"Processed {len(messages)} messages, found {len(search_results)} search results")
    return search_results
