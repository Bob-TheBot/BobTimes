"""Search result models for cleaner message storage and image handling."""

from datetime import datetime

from pydantic import BaseModel, Field


class CleanedSearchResult(BaseModel):
    """Cleaned search result with essential fields and image handling."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL to the full content")
    content: str = Field(..., description="Full content/body text (not truncated)")
    image_url: str | None = Field(None, description="Image URL from search result")
    image_local_path: str | None = Field(None, description="Local path after download")
    image_size_kb: int | None = Field(None, description="Image file size in KB")
    source: str | None = Field(None, description="Source of the content")
    date: str | None = Field(None, description="Publication date if available")

    def has_valid_image(self) -> bool:
        """Check if this result has a valid image URL."""
        return self.image_url is not None

    def is_large_image(self) -> bool:
        """Check if this result has image metadata indicating large size (>200KB)."""
        return self.image_size_kb is not None and self.image_size_kb >= 200


class SearchResultSummary(BaseModel):
    """Summary of search results for efficient message storage."""
    query: str = Field(..., description="Original search query")
    search_type: str = Field(..., description="Type of search (text/news)")
    result_count: int = Field(..., description="Number of results")
    results: list[CleanedSearchResult] = Field(default_factory=lambda: [])
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_message_content(self) -> str:
        """Convert to clean string representation for LLM context.
        
        Returns:
            Formatted string with search results
        """
        lines = [
            f"Search Results for: {self.query}",
            f"Type: {self.search_type} | Count: {self.result_count}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "-" * 50,
        ]

        for i, result in enumerate(self.results, 1):
            lines.append(f"\n[{i}] {result.title}")
            lines.append(f"URL: {result.url}")
            if result.source:
                lines.append(f"Source: {result.source}")
            if result.date:
                lines.append(f"Date: {result.date}")
            if result.image_local_path:
                lines.append("Image: Available (stored locally)")
            elif result.image_url:
                lines.append("Image: Available (not downloaded)")
            lines.append(f"Content: {result.content}")
            lines.append("-" * 30)

        return "\n".join(lines)

    def get_valid_images(self) -> list[CleanedSearchResult]:
        """Get all results that have valid image URLs.
        
        Returns:
            List of results with valid image URLs
        """
        return [r for r in self.results if r.has_valid_image()]

    def get_large_images(self) -> list[CleanedSearchResult]:
        """Get all results that have large images (>200KB).

        Returns:
            List of results with large images
        """
        return [r for r in self.results if r.is_large_image()]

    def get_largest_image(self) -> CleanedSearchResult | None:
        """Get the result with the largest available image.

        Returns:
            Result with largest image or None if no images
        """
        large_images = self.get_large_images()
        if large_images:
            # Sort by size descending and return the largest
            return max(large_images, key=lambda r: r.image_size_kb or 0)

        return None

    def get_best_image(self) -> CleanedSearchResult | None:
        """Get the result with the best available image.

        Prioritizes largest images first, then any images with URLs.

        Returns:
            Result with best image or None if no images
        """
        # First try to get the largest image
        largest_image = self.get_largest_image()
        if largest_image:
            return largest_image

        # Fall back to any valid images
        valid_images = self.get_valid_images()
        if valid_images:
            return valid_images[0]

        # Fall back to results with image URLs
        results_with_urls = [r for r in self.results if r.image_url]
        if results_with_urls:
            return results_with_urls[0]

        return None
