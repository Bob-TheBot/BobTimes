"""Example demonstrating the DuckDuckGo search tool functionality.

This example shows how to use the DuckDuckGoSearchTool for various types of searches:
- Text search with different regions and filters
- Image search with size, color, and license filters
- Video search with resolution and duration filters
- News search with time limits
- Comprehensive search across all categories
"""

import json
from pathlib import Path
from typing import Any

from utils.duckduckgo_search_tool import (
    DDGImageColor,
    DDGImageLicense,
    DDGImageSize,
    DDGImageType,
    DDGRegion,
    DDGSafeSearch,
    DDGTimeLimit,
    DDGVideoDuration,
    DDGVideoResolution,
    create_duckduckgo_search_tool,
)


def test_text_search():
    """Test basic text search functionality."""
    print("🔍 Testing Text Search")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # Basic text search
    results = search_tool.search_text(
        query="artificial intelligence trends 2024",
        region=DDGRegion.united_states,
        safe_search=DDGSafeSearch.moderate,
        max_results=5,
    )

    print(f"Found {len(results)} text results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.href}")
        print(f"   Summary: {result.body[:100]}...")

    return results


def test_image_search():
    """Test image search with filters."""
    print("\n🖼️  Testing Image Search")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # Image search with filters
    results = search_tool.search_images(
        query="sustainable technology",
        region=DDGRegion.united_states,
        safe_search=DDGSafeSearch.moderate,
        size=DDGImageSize.large,
        color=DDGImageColor.color,
        type_image=DDGImageType.photo,
        license_image=DDGImageLicense.share,
        max_results=3,
    )

    print(f"Found {len(results)} image results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   Image URL: {result.image}")
        print(f"   Page URL: {result.url}")
        print(f"   Size: {result.width}x{result.height}")
        print(f"   Source: {result.source}")

    return results


def test_video_search():
    """Test video search with filters."""
    print("\n🎥 Testing Video Search")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # Video search with filters
    results = search_tool.search_videos(
        query="machine learning tutorial",
        region=DDGRegion.united_states,
        safe_search=DDGSafeSearch.moderate,
        time_limit=DDGTimeLimit.month,
        resolution=DDGVideoResolution.high,
        duration=DDGVideoDuration.medium,
        max_results=3,
    )

    print(f"Found {len(results)} video results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   Video URL: {result.content}")
        print(f"   Duration: {result.duration}")
        print(f"   Publisher: {result.publisher}")
        print(f"   Views: {result.statistics.viewCount}")
        print(f"   Description: {result.description[:100]}...")

    return results


def test_news_search():
    """Test news search with time limits."""
    print("\n📰 Testing News Search")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # News search with time limit
    results = search_tool.search_news(
        query="climate change technology",
        region=DDGRegion.united_states,
        safe_search=DDGSafeSearch.moderate,
        time_limit=DDGTimeLimit.week,
        max_results=5,
    )

    print(f"Found {len(results)} news results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Source: {result.source}")
        print(f"   Date: {result.date}")
        print(f"   Summary: {result.body[:100]}...")
        if result.image:
            print(f"   Image: {result.image}")

    return results


def test_comprehensive_search():
    """Test comprehensive search across all categories."""
    print("\n🌐 Testing Comprehensive Search")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # Comprehensive search
    response = search_tool.comprehensive_search(
        query="renewable energy innovations",
        region=DDGRegion.united_states,
        safe_search=DDGSafeSearch.moderate,
        time_limit=DDGTimeLimit.month,
        max_results_per_category=3,
        include_text=True,
        include_images=True,
        include_videos=True,
        include_news=True,
    )

    print(f"Comprehensive search results for: '{response.query}'")
    print(f"Region: {response.region}")
    print(f"Safe Search: {response.safe_search}")
    print(f"Total Results: {response.total_results}")
    print(f"Timestamp: {response.timestamp}")

    print(f"\n📝 Text Results: {len(response.text_results)}")
    for i, result in enumerate(response.text_results, 1):
        print(f"  {i}. {result.title}")

    print(f"\n🖼️  Image Results: {len(response.image_results)}")
    for i, result in enumerate(response.image_results, 1):
        print(f"  {i}. {result.title} ({result.width}x{result.height})")

    print(f"\n🎥 Video Results: {len(response.video_results)}")
    for i, result in enumerate(response.video_results, 1):
        print(f"  {i}. {result.title} ({result.duration})")

    print(f"\n📰 News Results: {len(response.news_results)}")
    for i, result in enumerate(response.news_results, 1):
        print(f"  {i}. {result.title} - {result.source}")

    return response


def test_search_with_filters():
    """Test the flexible search_with_filters method."""
    print("\n⚙️  Testing Search with Filters")
    print("=" * 50)

    search_tool = create_duckduckgo_search_tool()

    # Test different search types with the unified method
    search_configs: list[dict[str, Any]] = [
        {
            "search_type": "text",
            "query": "python programming best practices",
            "max_results": 3,
        },
        {
            "search_type": "images",
            "query": "python logo",
            "max_results": 2,
            "size": DDGImageSize.medium,
            "type_image": DDGImageType.transparent,
        },
        {
            "search_type": "news",
            "query": "python programming language",
            "max_results": 2,
            "time_limit": DDGTimeLimit.week,
        },
    ]

    for config in search_configs:
        search_type = str(config.pop("search_type"))
        query = str(config["query"])

        print(f"\n🔍 {search_type.title()} search for: '{query}'")
        results = search_tool.search_with_filters(search_type=search_type, **config)
        print(f"   Found {len(results)} results")

        if results:
            first_result = results[0]
            if hasattr(first_result, "title"):
                print(f"   First result: {first_result.title}")


def save_results_to_file(results: Any, filename: str) -> None:
    """Save search results to a JSON file for inspection."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / filename

    # Convert Pydantic models to dictionaries for JSON serialization
    data: Any
    if hasattr(results, "model_dump"):
        # Single SearchResponse object
        data = results.model_dump()
    elif isinstance(results, list) and results:
        # List of search results
        data = [result.model_dump() for result in results]  # type: ignore[attr-defined]
    else:
        data = results  # type: ignore[assignment]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Results saved to: {output_file}")


def main():
    """Demonstrate all DuckDuckGo search tool functionality."""
    print("🚀 DuckDuckGo Search Tool Demo")
    print("=" * 50)

    try:
        # Test individual search types
        text_results = test_text_search()
        image_results = test_image_search()
        video_results = test_video_search()
        news_results = test_news_search()

        # Test comprehensive search
        comprehensive_results = test_comprehensive_search()

        # Test flexible search method
        test_search_with_filters()

        # Save results to files for inspection
        save_results_to_file(text_results, "text_search_results.json")
        save_results_to_file(image_results, "image_search_results.json")
        save_results_to_file(video_results, "video_search_results.json")
        save_results_to_file(news_results, "news_search_results.json")
        save_results_to_file(comprehensive_results, "comprehensive_search_results.json")

        print("\n🎉 All tests completed successfully!")
        print("\nThe DuckDuckGoSearchTool is ready for LLM integration!")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install the ddgs package:")
        print("   uv add ddgs")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    main()
