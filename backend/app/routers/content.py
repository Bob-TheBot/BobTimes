"""Content API endpoints.
Handles requests for stories, images, and content organization.
"""


from core.logging_service import get_logger
from fastapi import APIRouter, HTTPException, Query, status

from app.models.content_models import ContentFilters, ContentResponse, ContentSection, MainPageContent, PaginationParams, SectionPageContent, StoryModel
from app.services.content_service import ContentService

logger = get_logger(__name__)

router = APIRouter(prefix="/content", tags=["content"])


def get_content_service() -> ContentService:
    """Dependency to get ContentService instance."""
    return ContentService()


@router.get("/main-page", response_model=MainPageContent)
async def get_main_page_content() -> MainPageContent:
    """Get content for the main page including breaking news, featured stories,
    recent stories, and section highlights.
    """
    logger.info("API request: Get main page content")
    try:
        content_service = get_content_service()
        content = content_service.get_main_page_content()
        logger.info("Successfully retrieved main page content")
        return content
    except Exception as e:
        logger.error(f"Error retrieving main page content: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve main page content"
        )


@router.get("/sections/{section}", response_model=SectionPageContent)
async def get_section_content(
    section: ContentSection,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
) -> SectionPageContent:
    """Get content for a specific section page including featured stories,
    recent stories, and paginated list of all stories.
    """
    logger.info(f"API request: Get section content for {section.value}")
    try:
        content_service = get_content_service()
        pagination = PaginationParams(page=page, page_size=page_size)
        content = content_service.get_section_page_content(section, pagination)
        logger.info(f"Successfully retrieved content for section: {section.value}")
        return content
    except Exception as e:
        logger.error(f"Error retrieving section content for {section.value}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve content for section: {section.value}"
        )


@router.get("/stories/{slug}", response_model=StoryModel)
async def get_story_by_slug(slug: str) -> StoryModel:
    """Get a specific story by its slug.
    This also increments the view count for the story.
    """
    logger.info(f"API request: Get story by slug: {slug}")
    try:
        content_service = get_content_service()
        story = content_service.get_story_by_slug(slug)
        if not story:
            logger.warning(f"Story not found: {slug}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Story not found: {slug}"
            )

        logger.info(f"Successfully retrieved story: {slug}")
        return story
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving story {slug}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve story"
        )


@router.get("/stories", response_model=list[StoryModel])
async def search_stories(
    section: ContentSection | None = Query(None, description="Filter by section"),
    search: str | None = Query(None, description="Search in title, summary, and content"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
) -> list[StoryModel]:
    """Search and filter stories with pagination.
    """
    logger.info("API request: Search stories")
    try:
        content_service = get_content_service()
        filters = ContentFilters(
            section=section,
            search=search,
            tags=tags
        )
        pagination = PaginationParams(page=page, page_size=page_size)

        stories, total_count = content_service.search_stories(filters, pagination)
        logger.info(f"Successfully retrieved {len(stories)} stories (total: {total_count})")
        return stories
    except Exception as e:
        logger.error(f"Error searching stories: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search stories"
        )


@router.get("/popular", response_model=list[StoryModel])
async def get_popular_stories(
    section: ContentSection | None = Query(None, description="Filter by section"),
    limit: int = Query(10, ge=1, le=50, description="Number of stories to return")
) -> list[StoryModel]:
    """Get popular stories based on view count.
    """
    logger.info("API request: Get popular stories")
    try:
        content_service = get_content_service()
        stories = content_service.get_popular_stories(section=section, limit=limit)
        logger.info(f"Successfully retrieved {len(stories)} popular stories")
        return stories
    except Exception as e:
        logger.error(f"Error retrieving popular stories: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve popular stories"
        )


# Health check endpoint for content service
@router.get("/health", response_model=ContentResponse)
async def content_health_check() -> ContentResponse:
    """Health check endpoint for content service.
    """
    return ContentResponse(
        success=True,
        message="Content service is healthy",
        data={"service": "content", "status": "operational"}
    )
