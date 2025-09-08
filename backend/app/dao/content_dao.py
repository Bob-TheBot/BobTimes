"""Data Access Objects for content management.
Handles database operations for stories, images, and authors.
"""

from uuid import UUID

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session, selectinload

from app.models.content_models import ContentFilters, ContentSection, ContentStatus, PaginationParams, Priority
from shared_db.dao.base_dao import BaseDAO
from shared_db.models.content import Author, ContentMetrics, Image, Story


class AuthorDAO(BaseDAO[Author]):
    """Data Access Object for Author operations."""

    def __init__(self, db: Session):
        super().__init__(db, Author)

    def get_by_email(self, email: str) -> Author | None:
        """Get author by email address."""
        return self.db.query(Author).filter(Author.email == email).first()

    def get_active_authors(self) -> list[Author]:
        """Get all active authors."""
        return self.db.query(Author).filter(Author.is_active == True).all()

    def get_authors_with_stories(self) -> list[Author]:
        """Get authors who have published stories."""
        return (
            self.db.query(Author)
            .join(Story)
            .filter(Story.status == ContentStatus.PUBLISHED)
            .distinct()
            .all()
        )


class ImageDAO(BaseDAO[Image]):
    """Data Access Object for Image operations."""

    def __init__(self, db: Session):
        super().__init__(db, Image)

    def get_by_filename(self, filename: str) -> Image | None:
        """Get image by filename."""
        return self.db.query(Image).filter(Image.filename == filename).first()

    def get_recent_images(self, limit: int = 20) -> list[Image]:
        """Get recently uploaded images."""
        return (
            self.db.query(Image)
            .order_by(desc(Image.created_at))
            .limit(limit)
            .all()
        )

    def get_images_by_size_range(self, min_width: int, max_width: int) -> list[Image]:
        """Get images within a specific width range."""
        return (
            self.db.query(Image)
            .filter(and_(Image.width >= min_width, Image.width <= max_width))
            .all()
        )


class StoryDAO(BaseDAO[Story]):
    """Data Access Object for Story operations."""

    def __init__(self, db: Session):
        super().__init__(db, Story)

    def get_by_slug(self, slug: str) -> Story | None:
        """Get story by slug."""
        return (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(Story.slug == slug)
            .first()
        )

    def get_published_stories(
        self,
        section: ContentSection | None = None,
        limit: int | None = None,
        offset: int | None = None
    ) -> list[Story]:
        """Get published stories, optionally filtered by section."""
        query = (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(Story.status == ContentStatus.PUBLISHED)
            .order_by(desc(Story.published_at))
        )

        if section:
            query = query.filter(Story.section == section.value)

        if offset:
            query = query.offset(offset)

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_featured_stories(self, limit: int = 5) -> list[Story]:
        """Get featured stories."""
        return (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(Story.status == ContentStatus.FEATURED)
            .order_by(desc(Story.published_at))
            .limit(limit)
            .all()
        )

    def get_breaking_news(self, limit: int = 10) -> list[Story]:
        """Get breaking news stories (high priority, recent)."""
        return (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(
                and_(
                    Story.status == ContentStatus.PUBLISHED,
                    Story.priority == Priority.URGENT
                )
            )
            .order_by(desc(Story.published_at))
            .limit(limit)
            .all()
        )

    def get_stories_by_section(
        self,
        section: ContentSection,
        pagination: PaginationParams
    ) -> tuple[list[Story], int]:
        """Get stories by section with pagination."""
        base_query = (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(
                and_(
                    Story.section == section.value,
                    Story.status == ContentStatus.PUBLISHED
                )
            )
        )

        total_count = base_query.count()

        stories = (
            base_query
            .order_by(desc(Story.published_at))
            .offset((pagination.page - 1) * pagination.page_size)
            .limit(pagination.page_size)
            .all()
        )

        return stories, total_count

    def search_stories(
        self,
        filters: ContentFilters,
        pagination: PaginationParams
    ) -> tuple[list[Story], int]:
        """Search stories with filters and pagination."""
        query = (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
        )

        # Apply filters
        conditions = []

        if filters.section:
            conditions.append(Story.section == filters.section.value)

        if filters.status:
            conditions.append(Story.status == filters.status.value)

        if filters.priority:
            conditions.append(Story.priority == filters.priority.value)

        if filters.author_id:
            conditions.append(Story.author_id == filters.author_id)

        if filters.tags:
            # Stories that have any of the specified tags
            conditions.append(Story.tags.overlap(filters.tags))

        if filters.search:
            # Search in title, summary, and content
            search_term = f"%{filters.search}%"
            conditions.append(
                or_(
                    Story.title.ilike(search_term),
                    Story.summary.ilike(search_term),
                    Story.content.ilike(search_term)
                )
            )

        if filters.date_from:
            conditions.append(Story.published_at >= filters.date_from)

        if filters.date_to:
            conditions.append(Story.published_at <= filters.date_to)

        if conditions:
            query = query.filter(and_(*conditions))

        total_count = query.count()

        stories = (
            query
            .order_by(desc(Story.published_at))
            .offset((pagination.page - 1) * pagination.page_size)
            .limit(pagination.page_size)
            .all()
        )

        return stories, total_count

    def increment_view_count(self, story_id: UUID) -> bool:
        """Increment the view count for a story."""
        try:
            self.db.query(Story).filter(Story.id == story_id).update(
                {Story.view_count: Story.view_count + 1}
            )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False

    def get_popular_stories(self, section: ContentSection | None = None, limit: int = 10) -> list[Story]:
        """Get popular stories based on view count."""
        query = (
            self.db.query(Story)
            .options(
                selectinload(Story.author).selectinload(Author.avatar_image),
                selectinload(Story.featured_image),
                selectinload(Story.gallery_images)
            )
            .filter(Story.status == ContentStatus.PUBLISHED)
            .order_by(desc(Story.view_count))
        )

        if section:
            query = query.filter(Story.section == section.value)

        return query.limit(limit).all()


class ContentMetricsDAO(BaseDAO[ContentMetrics]):
    """Data Access Object for ContentMetrics operations."""

    def __init__(self, db: Session):
        super().__init__(db, ContentMetrics)

    def record_metric(self, story_id: UUID, metric_type: str, value: int = 1) -> ContentMetrics:
        """Record a metric for a story."""
        metric = ContentMetrics(
            story_id=story_id,
            metric_type=metric_type,
            metric_value=value
        )
        return self.create(metric)

    def get_story_metrics(self, story_id: UUID) -> list[ContentMetrics]:
        """Get all metrics for a specific story."""
        return (
            self.db.query(ContentMetrics)
            .filter(ContentMetrics.story_id == story_id)
            .order_by(desc(ContentMetrics.recorded_at))
            .all()
        )
