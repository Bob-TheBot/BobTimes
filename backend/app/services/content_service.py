"""Content management service.
Business logic for handling stories, images, and content organization.
Uses hardcoded data for simplicity.
"""

import uuid
from datetime import UTC, datetime, timedelta

from core.logging_service import get_logger

from app.models.content_models import (
    AuthorModel,
    ContentFilters,
    ContentSection,
    ContentStatus,
    ImageModel,
    MainPageContent,
    PaginationParams,
    Priority,
    SectionPageContent,
    SectionSummary,
    StoryModel,
)

logger = get_logger(__name__)


class ContentService:
    """Service for content management operations with hardcoded data."""

    def __init__(self):
        self.authors = self._create_mock_authors()
        self.images = self._create_mock_images()
        self.stories = self._create_mock_stories()

    def _create_mock_authors(self) -> list[AuthorModel]:
        """Create mock authors data."""
        now = datetime.now(UTC)
        return [
            AuthorModel(
                id=uuid.uuid4(),
                name="Sarah Chen",
                email="sarah.chen@bobtimes.com",
                bio="Senior Technology Reporter with 10+ years covering Silicon Valley innovations and emerging tech trends.",
                avatar_image=None,
                specialties=["AI", "Startups", "Cybersecurity", "Innovation"],
                created_at=now - timedelta(days=365),
                updated_at=now
            ),
            AuthorModel(
                id=uuid.uuid4(),
                name="Marcus Johnson",
                email="marcus.johnson@bobtimes.com",
                bio="Economics correspondent specializing in global markets, financial policy, and economic analysis.",
                avatar_image=None,
                specialties=["Markets", "Policy", "Cryptocurrency", "Global Economy"],
                created_at=now - timedelta(days=300),
                updated_at=now
            ),
            AuthorModel(
                id=uuid.uuid4(),
                name="Elena Rodriguez",
                email="elena.rodriguez@bobtimes.com",
                bio="Sports journalist covering professional athletics, Olympic games, and competitive sports.",
                avatar_image=None,
                specialties=["Olympics", "Professional Sports", "Athletic Performance"],
                created_at=now - timedelta(days=200),
                updated_at=now
            ),
            AuthorModel(
                id=uuid.uuid4(),
                name="David Kim",
                email="david.kim@bobtimes.com",
                bio="Political correspondent covering government affairs, policy developments, and democratic processes.",
                avatar_image=None,
                specialties=["Politics", "Government", "Policy Analysis", "Elections"],
                created_at=now - timedelta(days=400),
                updated_at=now
            )
        ]

    def _create_mock_images(self) -> list[ImageModel]:
        """Create mock images data."""
        now = datetime.now(UTC)
        return [
            ImageModel(
                id=uuid.uuid4(),
                filename="tech-ai-breakthrough.jpg",
                original_filename="ai-breakthrough-2024.jpg",
                url="https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=400&fit=crop",
                alt_text="AI technology breakthrough visualization",
                caption="Revolutionary AI system demonstrating human-level reasoning capabilities",
                width=800,
                height=400,
                file_size=245760,
                mime_type="image/jpeg",
                created_at=now - timedelta(hours=2),
                updated_at=now - timedelta(hours=2)
            ),
            ImageModel(
                id=uuid.uuid4(),
                filename="sports-championship.jpg",
                original_filename="championship-finals-2024.jpg",
                url="https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800&h=400&fit=crop",
                alt_text="Championship sports competition",
                caption="Intense championship finals competition",
                width=800,
                height=400,
                file_size=312450,
                mime_type="image/jpeg",
                created_at=now - timedelta(hours=4),
                updated_at=now - timedelta(hours=4)
            ),
            ImageModel(
                id=uuid.uuid4(),
                filename="economics-market.jpg",
                original_filename="market-analysis-2024.jpg",
                url="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&h=400&fit=crop",
                alt_text="Economic market analysis visualization",
                caption="Global market trends and economic indicators",
                width=800,
                height=400,
                file_size=198340,
                mime_type="image/jpeg",
                created_at=now - timedelta(hours=6),
                updated_at=now - timedelta(hours=6)
            )
        ]

    def _create_mock_stories(self) -> list[StoryModel]:
        """Create mock stories data."""
        now = datetime.now(UTC)

        stories = [
            # Breaking News Stories
            StoryModel(
                id=uuid.uuid4(),
                title="Global Climate Summit Reaches Historic Agreement on Carbon Reduction",
                slug="global-climate-summit-historic-agreement",
                summary="World leaders commit to ambitious new targets for carbon emissions, marking a turning point in international climate action.",
                content="""In a historic moment for global environmental policy, representatives from 195 countries have reached a comprehensive agreement on carbon reduction targets that experts are calling the most ambitious climate accord since the Paris Agreement.

The agreement establishes binding commitments for developed nations to reduce carbon emissions by 50% by 2030, with developing countries committing to a 30% reduction over the same period. The accord also includes provisions for $500 billion in climate financing to support renewable energy transitions in emerging economies.

"This agreement represents a fundamental shift in how the international community approaches climate change," said the summit's lead negotiator. "We're moving from voluntary commitments to binding obligations with real consequences for non-compliance."

The accord also establishes new international standards for renewable energy deployment and includes provisions for technology sharing between developed and developing nations.

Environmental groups have praised the agreement as a crucial step forward, while acknowledging that implementation will be the true test of its effectiveness.""",
                section=ContentSection.NEWS,
                status=ContentStatus.PUBLISHED,
                priority=Priority.URGENT,
                author=self.authors[3],  # David Kim
                featured_image=self.images[2],
                gallery_images=[],
                tags=["Climate Change", "Politics", "Environment", "International"],
                read_time_minutes=6,
                view_count=15420,
                published_at=now - timedelta(hours=2),
                created_at=now - timedelta(hours=3),
                updated_at=now - timedelta(hours=2)
            ),

            # Technology Stories
            StoryModel(
                id=uuid.uuid4(),
                title="Revolutionary AI Breakthrough: New Model Achieves Human-Level Reasoning",
                slug="ai-breakthrough-human-level-reasoning",
                summary="Scientists at leading tech companies announce a major breakthrough in artificial intelligence that could reshape how we interact with technology.",
                content="""In a groundbreaking development that could fundamentally change the landscape of artificial intelligence, researchers have unveiled a new AI model that demonstrates unprecedented reasoning capabilities, matching human-level performance across a wide range of cognitive tasks.

The breakthrough, achieved through a novel combination of transformer architecture and neuromorphic computing principles, represents a significant leap forward in machine learning capabilities. The system can now perform complex logical reasoning, understand context with remarkable accuracy, and even demonstrate creative problem-solving abilities.

"This is the closest we've come to achieving artificial general intelligence," explained the lead researcher. "The model doesn't just process information—it truly understands and reasons about it in ways that mirror human cognition."

The implications for industries ranging from healthcare to education are profound. Early applications show the system can assist in medical diagnosis, provide personalized tutoring, and even contribute to scientific research by identifying patterns that human researchers might miss.

However, the development also raises important questions about AI safety and the future of human-AI collaboration that the research community is actively addressing.""",
                section=ContentSection.TECHNOLOGY,
                status=ContentStatus.FEATURED,
                priority=Priority.HIGH,
                author=self.authors[0],  # Sarah Chen
                featured_image=self.images[0],
                gallery_images=[],
                tags=["AI", "Machine Learning", "Technology", "Innovation"],
                read_time_minutes=8,
                view_count=23150,
                published_at=now - timedelta(hours=4),
                created_at=now - timedelta(hours=5),
                updated_at=now - timedelta(hours=4)
            ),

            # Sports Stories
            StoryModel(
                id=uuid.uuid4(),
                title="Championship Finals Set as Top Teams Advance Through Playoffs",
                slug="championship-finals-playoff-advancement",
                summary="After intense playoff competition, the final matchups have been determined for this year's championship games across major sports leagues.",
                content="""The playoff season has concluded with thrilling matchups that have set the stage for what promises to be an exceptional championship series across multiple professional sports leagues.

In basketball, two powerhouse teams have emerged from their respective conferences after hard-fought playoff battles. Both teams demonstrated exceptional teamwork and strategic execution throughout the postseason.

The football championship game features teams with contrasting styles - one known for their explosive offensive capabilities, while the other has built their success on a dominant defensive unit.

"These matchups represent the best of professional sports," commented a sports analyst. "The level of competition and athleticism we've seen throughout the playoffs has been remarkable."

Baseball's championship series will showcase teams from different leagues, each bringing unique strengths and compelling storylines to the final stage of competition.

Ticket demand has reached unprecedented levels, with fans eager to witness what could be historic performances from elite athletes at the peak of their careers.""",
                section=ContentSection.SPORTS,
                status=ContentStatus.PUBLISHED,
                priority=Priority.HIGH,
                author=self.authors[2],  # Elena Rodriguez
                featured_image=self.images[1],
                gallery_images=[],
                tags=["Championships", "Playoffs", "Professional Sports", "Competition"],
                read_time_minutes=4,
                view_count=18750,
                published_at=now - timedelta(hours=6),
                created_at=now - timedelta(hours=7),
                updated_at=now - timedelta(hours=6)
            )
        ]

        return stories

    def get_main_page_content(self) -> MainPageContent:
        """Get content for the main page."""
        logger.info("Fetching main page content")

        # Get breaking news (urgent priority stories)
        breaking_news = [story for story in self.stories if story.priority == Priority.URGENT][:5]

        # Get featured stories
        featured_stories = [story for story in self.stories if story.status == ContentStatus.FEATURED][:6]

        # Get recent stories (all published stories, sorted by date)
        recent_stories = sorted(
            [story for story in self.stories if story.status == ContentStatus.PUBLISHED],
            key=lambda x: x.published_at or x.created_at,
            reverse=True
        )[:12]

        # Get section highlights (stories per section)
        section_highlights = {}
        for section in ContentSection:
            section_stories = [story for story in self.stories if story.section == section][:3]
            if section_stories:
                section_highlights[section] = section_stories

        # Get section summaries
        sections_summary = []
        for section in ContentSection:
            section_stories = [story for story in self.stories if story.section == section]
            latest_story = max(section_stories, key=lambda x: x.published_at or x.created_at) if section_stories else None

            sections_summary.append(SectionSummary(
                section=section,
                name=section.value.title(),
                description=self._get_section_description(section),
                story_count=len(section_stories),
                latest_update=latest_story.published_at if latest_story else None
            ))

        return MainPageContent(
            breaking_news=breaking_news,
            featured_stories=featured_stories,
            recent_stories=recent_stories,
            section_highlights=section_highlights,
            sections_summary=sections_summary,
            last_updated=datetime.now(UTC)
        )

    def get_section_page_content(
        self,
        section: ContentSection,
        pagination: PaginationParams
    ) -> SectionPageContent:
        """Get content for a specific section page."""
        logger.info(f"Fetching content for section: {section.value}")

        # Get all stories for this section
        section_stories = [story for story in self.stories if story.section == section]

        # Sort by publication date
        section_stories = sorted(
            section_stories,
            key=lambda x: x.published_at or x.created_at,
            reverse=True
        )

        # Get featured stories for this section (first 3)
        featured_stories = section_stories[:3]

        # Get recent stories for this section (first 6)
        recent_stories = section_stories[:6]

        # Apply pagination to all stories
        start_idx = (pagination.page - 1) * pagination.page_size
        end_idx = start_idx + pagination.page_size
        all_stories = section_stories[start_idx:end_idx]
        total_count = len(section_stories)

        # Create section info
        latest_story = section_stories[0] if section_stories else None
        section_info = SectionSummary(
            section=section,
            name=section.value.title(),
            description=self._get_section_description(section),
            story_count=total_count,
            latest_update=latest_story.published_at if latest_story else None
        )

        return SectionPageContent(
            section=section,
            section_info=section_info,
            featured_stories=featured_stories,
            recent_stories=recent_stories,
            all_stories=all_stories,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            last_updated=datetime.now(UTC)
        )

    def get_story_by_slug(self, slug: str) -> StoryModel | None:
        """Get a story by its slug."""
        for story in self.stories:
            if story.slug == slug:
                # Increment view count (simulate)
                story.view_count += 1
                return story
        return None

    def search_stories(
        self,
        filters: ContentFilters,
        pagination: PaginationParams
    ) -> tuple[list[StoryModel], int]:
        """Search stories with filters."""
        filtered_stories = self.stories.copy()

        # Apply filters
        if filters.section:
            filtered_stories = [s for s in filtered_stories if s.section == filters.section]

        if filters.search:
            search_term = filters.search.lower()
            filtered_stories = [
                s for s in filtered_stories
                if search_term in s.title.lower() or
                   search_term in s.summary.lower() or
                   search_term in s.content.lower()
            ]

        if filters.tags:
            filtered_stories = [
                s for s in filtered_stories
                if any(tag in s.tags for tag in filters.tags)
            ]

        # Sort by publication date
        filtered_stories = sorted(
            filtered_stories,
            key=lambda x: x.published_at or x.created_at,
            reverse=True
        )

        total_count = len(filtered_stories)

        # Apply pagination
        start_idx = (pagination.page - 1) * pagination.page_size
        end_idx = start_idx + pagination.page_size
        paginated_stories = filtered_stories[start_idx:end_idx]

        return paginated_stories, total_count

    def _get_section_description(self, section: ContentSection) -> str:
        """Get description for a content section."""
        descriptions = {
            ContentSection.NEWS: "Breaking news and current events coverage",
            ContentSection.SPORTS: "Athletic competition and sports industry reporting",
            ContentSection.TECHNOLOGY: "Technology trends and innovation analysis",
            ContentSection.ECONOMICS: "Economic analysis and financial market coverage",
            ContentSection.POLITICS: "Political developments and government affairs"
        }
        return descriptions.get(section, "News coverage and analysis")

    def get_popular_stories(self, section: ContentSection | None = None, limit: int = 10) -> list[StoryModel]:
        """Get popular stories based on view count."""
        stories = self.stories.copy()

        if section:
            stories = [s for s in stories if s.section == section]

        # Sort by view count
        stories = sorted(stories, key=lambda x: x.view_count, reverse=True)

        return stories[:limit]
