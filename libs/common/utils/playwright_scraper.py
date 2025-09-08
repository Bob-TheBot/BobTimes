"""Simple Playwright Web Scraper

A clean, focused web scraper that uses only Playwright to extract content
from web pages, especially JavaScript-heavy sites.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment
from core.logging_service import get_logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from playwright.async_api import Page

try:
    from playwright.async_api import async_playwright
    _playwright_available = True
except ImportError:
    _playwright_available = False
    async_playwright = None


logger = get_logger(__name__)


class ScrapedContent(BaseModel):
    """Model representing scraped web content."""
    url: str = Field(description="The original URL that was scraped")
    title: str = Field(description="The page title")
    content: str = Field(description="The cleaned text content")
    word_count: int = Field(description="Number of words in the content")
    scraped_at: str = Field(description="ISO timestamp when content was scraped")
    success: bool = Field(description="Whether scraping was successful")
    error_message: str | None = Field(default=None, description="Error message if scraping failed")


class AsyncPlaywrightScraper:
    """An async web scraper using Playwright for JavaScript-heavy sites.
    """

    def __init__(
        self,
        timeout: int = 60,
        wait_time: int = 2000,
        headless: bool = True,
        handle_interactive_content: bool = True,
        max_content_buttons: int = 5,
    ):
        """Initialize the Playwright scraper.

        Args:
            timeout: Page load timeout in seconds
            wait_time: Time to wait for content to load in milliseconds
            headless: Whether to run browser in headless mode
            handle_interactive_content: Whether to click content-revealing buttons
            max_content_buttons: Maximum number of content buttons to click
        """
        if not _playwright_available:
            raise ImportError("Playwright is not installed. Please run: uv add playwright && playwright install chromium")

        self.timeout = timeout
        self.wait_time = wait_time
        self.headless = headless
        self.handle_interactive_content = handle_interactive_content
        self.max_content_buttons = max_content_buttons

        # User agents to rotate through
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]

    def _clean_text(self, html: str) -> str:
        """Clean HTML content and extract readable text.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        unwanted_tags = ["script", "style", "nav", "header", "footer", "aside",
                        "menu", "sidebar", "form", "button", "input", "select",
                        "textarea", "noscript", "iframe"]
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove elements with unwanted classes/ids (ads, social, etc.)
        unwanted_patterns = ["menu", "nav", "sidebar", "footer", "header", "ad",
                           "advertisement", "social", "share", "comment", "related",
                           "recommend", "popup", "modal"]

        # Use CSS selectors for more reliable element removal
        for pattern in unwanted_patterns:
            # Remove by class containing pattern
            for element in soup.select(f'[class*="{pattern}"]'):
                element.decompose()

            # Remove by id containing pattern
            for element in soup.select(f'[id*="{pattern}"]'):
                element.decompose()

        # Get text and clean it
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    async def _handle_cookie_popups(self, page: "Page") -> None:
        """Handle cookie consent popups and other overlays that might block content.

        Args:
            page: Playwright page object
        """
        logger.debug("Attempting to handle cookie popups and overlays...")

        # First try specific cookie framework handlers
        if await self._handle_specific_cookie_frameworks(page):
            return  # Successfully handled with specific framework

        # Cookie consent button selectors (Accept/Allow buttons)
        cookie_accept_selectors = [
            # Generic accept buttons
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("Accept all")',
            'button:has-text("Allow All")',
            'button:has-text("Allow all")',
            'button:has-text("I Accept")',
            'button:has-text("I Agree")',
            'button:has-text("Agree")',
            'button:has-text("OK")',
            'button:has-text("Got it")',
            'button:has-text("Continue")',

            # Common data attributes and IDs
            '[data-testid*="accept"]',
            '[data-testid*="consent"]',
            '[data-testid*="cookie"]',
            '[id*="accept"]',
            '[id*="consent"]',
            '[id*="cookie"]',

            # Class-based selectors
            '.accept-cookies',
            '.cookie-accept',
            '.consent-accept',
            '.gdpr-accept',
            '.privacy-accept',

            # Specific cookie banner frameworks
            '#onetrust-accept-btn-handler',  # OneTrust
            '.optanon-allow-all',  # OneTrust
            '#cookieChoiceDismiss',  # Google Cookie Choice
            '.cc-allow',  # Cookie Consent
            '.cc-dismiss',  # Cookie Consent
            '.cookie-banner-accept',
            '.gdpr-banner-accept',

            # Language variations
            'button:has-text("Accepter")',  # French
            'button:has-text("Aceptar")',   # Spanish
            'button:has-text("Akzeptieren")',  # German
            'button:has-text("Accetta")',   # Italian
            'button:has-text("Aceitar")',   # Portuguese
            'button:has-text("同意")',        # Chinese
            'button:has-text("同意する")',     # Japanese
            'button:has-text("동의")',        # Korean
            'button:has-text("Принять")',   # Russian
            'button:has-text("قبول")',       # Arabic
            'button:has-text("אישור")',      # Hebrew
        ]

        # Generic close button selectors
        close_selectors = [
            '[data-testid="close"]',
            '[data-testid*="close"]',
            '.close',
            '.modal-close',
            '.popup-close',
            '.overlay-close',
            '[aria-label="Close"]',
            '[aria-label="close"]',
            '[aria-label="סגור"]',  # Hebrew "Close"
            'button:has-text("Close")',
            'button:has-text("close")',
            'button:has-text("סגור")',
            'button:has-text("×")',
            'button:has-text("✕")',
            '[title="Close"]',
            '[title="close"]',
            '.fa-times',  # Font Awesome close icon
            '.fa-close',  # Font Awesome close icon
        ]

        # Try cookie accept buttons first (preferred approach)
        for selector in cookie_accept_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=2000)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(1000)  # Wait for popup to disappear
                    logger.debug(f"Accepted cookies with selector: {selector}")
                    return  # Exit after first successful click
            except Exception:
                continue

        # If no cookie accept buttons found, try generic close buttons
        for selector in close_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=1000)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(500)
                    logger.debug(f"Closed popup with selector: {selector}")
                    # Don't return here - there might be multiple popups
            except Exception:
                continue

        # Try to dismiss any remaining overlays by pressing Escape
        try:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
            logger.debug("Pressed Escape to dismiss overlays")
        except Exception:
            pass

        logger.debug("Finished handling cookie popups and overlays")

    async def _handle_specific_cookie_frameworks(self, page: "Page") -> bool:
        """Handle specific cookie consent frameworks that need special treatment.

        Args:
            page: Playwright page object

        Returns:
            True if a framework was detected and handled, False otherwise
        """
        # OneTrust framework
        try:
            onetrust_banner = await page.wait_for_selector('#onetrust-banner-sdk', timeout=2000)
            if onetrust_banner and await onetrust_banner.is_visible():
                # Try the accept all button first
                accept_btn = await page.query_selector('#onetrust-accept-btn-handler')
                if accept_btn and await accept_btn.is_visible():
                    await accept_btn.click()
                    await page.wait_for_timeout(1000)
                    logger.debug("Handled OneTrust cookie banner")
                    return True
        except Exception:
            pass

        # Cookiebot framework
        try:
            cookiebot_banner = await page.wait_for_selector('#CybotCookiebotDialog', timeout=2000)
            if cookiebot_banner and await cookiebot_banner.is_visible():
                accept_btn = await page.query_selector('#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
                if accept_btn and await accept_btn.is_visible():
                    await accept_btn.click()
                    await page.wait_for_timeout(1000)
                    logger.debug("Handled Cookiebot cookie banner")
                    return True
        except Exception:
            pass

        # Google Cookie Consent
        try:
            google_consent = await page.wait_for_selector('.VfPpkd-LgbsSe[jsname="tWT92d"]', timeout=2000)
            if google_consent and await google_consent.is_visible():
                await google_consent.click()
                await page.wait_for_timeout(1000)
                logger.debug("Handled Google cookie consent")
                return True
        except Exception:
            pass

        return False

    async def _handle_content_revealing_buttons(self, page: "Page") -> None:
        """Handle buttons that reveal additional content when clicked.

        This method looks for common patterns of buttons that expand content,
        such as "Continue reading", "Show more", "Read full article", etc.

        Args:
            page: Playwright page object
        """
        logger.debug("Attempting to handle content-revealing buttons...")

        # Content-revealing button selectors
        content_buttons = [
            # Generic content expansion buttons
            'button:has-text("Continue reading")',
            'button:has-text("Continue Reading")',
            'button:has-text("Read more")',
            'button:has-text("Read More")',
            'button:has-text("Show more")',
            'button:has-text("Show More")',
            'button:has-text("View more")',
            'button:has-text("View More")',
            'button:has-text("See more")',
            'button:has-text("See More")',
            'button:has-text("Load more")',
            'button:has-text("Load More")',
            'button:has-text("Expand")',
            'button:has-text("Full article")',
            'button:has-text("Full Article")',
            'button:has-text("Read full article")',
            'button:has-text("Read Full Article")',
            'button:has-text("View full story")',
            'button:has-text("View Full Story")',
            'button:has-text("Complete article")',
            'button:has-text("Complete Article")',

            # Link-based content expansion
            'a:has-text("Continue reading")',
            'a:has-text("Continue Reading")',
            'a:has-text("Read more")',
            'a:has-text("Read More")',
            'a:has-text("Show more")',
            'a:has-text("Show More")',
            'a:has-text("Full article")',
            'a:has-text("Full Article")',
            'a:has-text("Read full article")',
            'a:has-text("Read Full Article")',

            # Common data attributes and classes
            '[data-testid*="read-more"]',
            '[data-testid*="continue"]',
            '[data-testid*="expand"]',
            '[data-testid*="show-more"]',
            '[data-testid*="full-article"]',
            '[id*="read-more"]',
            '[id*="continue"]',
            '[id*="expand"]',
            '[id*="show-more"]',
            '[id*="full-article"]',
            '.read-more',
            '.continue-reading',
            '.show-more',
            '.expand-content',
            '.full-article',
            '.view-more',

            # Language variations
            'button:has-text("Lire la suite")',      # French
            'button:has-text("Leer más")',           # Spanish
            'button:has-text("Weiterlesen")',        # German
            'button:has-text("Leggi di più")',       # Italian
            'button:has-text("Ler mais")',           # Portuguese
            'button:has-text("続きを読む")',            # Japanese
            'button:has-text("더 보기")',              # Korean
            'button:has-text("Читать далее")',       # Russian
            'button:has-text("اقرأ المزيد")',         # Arabic
            'button:has-text("קרא עוד")',             # Hebrew
        ]

        # Try each selector to find and click content-revealing buttons
        buttons_clicked = 0
        max_buttons = self.max_content_buttons  # Use configured limit

        for selector in content_buttons:
            if buttons_clicked >= max_buttons:
                break

            try:
                # Look for the button with a short timeout
                element = await page.wait_for_selector(selector, timeout=1000)
                if element and await element.is_visible():
                    # Check if the element is actually clickable
                    if await element.is_enabled():
                        logger.debug(f"Found content button with selector: {selector}")

                        # Scroll the element into view
                        await element.scroll_into_view_if_needed()
                        await page.wait_for_timeout(500)

                        # Click the button
                        await element.click()
                        buttons_clicked += 1

                        # Wait for content to load after clicking
                        await page.wait_for_timeout(2000)

                        logger.debug(f"Clicked content button: {selector}")

                        # Check if more content appeared by comparing page height
                        # This is a simple heuristic to detect if content was revealed
                        try:
                            await page.wait_for_function(
                                "() => document.readyState === 'complete'",
                                timeout=3000
                            )
                        except Exception:
                            pass  # Continue even if this check fails

            except Exception:
                continue  # Try next selector if this one fails

        if buttons_clicked > 0:
            logger.debug(f"Successfully clicked {buttons_clicked} content-revealing buttons")
            # Give extra time for all content to load
            await page.wait_for_timeout(2000)
        else:
            logger.debug("No content-revealing buttons found or clicked")

    async def _detect_truncated_content(self, page: "Page") -> bool:
        """Detect if content appears to be truncated or has expandable sections.

        Args:
            page: Playwright page object

        Returns:
            True if truncated content is detected, False otherwise
        """
        try:
            # Look for common indicators of truncated content
            truncation_indicators = [
                # Text-based indicators
                ':has-text("...")',
                ':has-text("…")',
                ':has-text("Read more")',
                ':has-text("Continue reading")',
                ':has-text("Show more")',
                ':has-text("[truncated]")',
                ':has-text("(more)")',

                # Class-based indicators
                '.truncated',
                '.excerpt',
                '.summary',
                '.preview',
                '.teaser',
                '.collapsed',

                # Common truncation patterns
                '[class*="truncat"]',
                '[class*="excerpt"]',
                '[class*="preview"]',
                '[class*="teaser"]',
                '[class*="collapsed"]',
            ]

            for indicator in truncation_indicators:
                try:
                    element = await page.wait_for_selector(indicator, timeout=500)
                    if element and await element.is_visible():
                        logger.debug(f"Detected truncated content with indicator: {indicator}")
                        return True
                except Exception:
                    continue

            return False

        except Exception as e:
            logger.debug(f"Error detecting truncated content: {e}")
            return False

    async def _try_additional_expansion_strategies(self, page: "Page") -> None:
        """Try additional strategies to expand content when standard buttons don't work.

        Args:
            page: Playwright page object
        """
        logger.debug("Trying additional content expansion strategies...")

        try:
            # Strategy 1: Try clicking on elements that might be expandable
            expandable_selectors = [
                '.expandable',
                '.collapsible',
                '.toggle',
                '[data-toggle]',
                '[data-expand]',
                '[data-collapse="false"]',
                '.article-body .more',
                '.content .more',
                '.text .more',
            ]

            for selector in expandable_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=1000)
                    if element and await element.is_visible() and await element.is_enabled():
                        await element.click()
                        await page.wait_for_timeout(1000)
                        logger.debug(f"Clicked expandable element: {selector}")
                        break  # Only try one expansion per strategy
                except Exception:
                    continue

            # Strategy 2: Try scrolling to trigger lazy loading
            try:
                # Scroll to bottom to trigger any lazy loading
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)

                # Scroll back to top
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(1000)

                logger.debug("Performed scroll-based content loading")
            except Exception:
                pass

            # Strategy 3: Try pressing common keyboard shortcuts
            try:
                # Some sites respond to spacebar or enter to expand content
                await page.keyboard.press("Space")
                await page.wait_for_timeout(500)
                logger.debug("Tried spacebar for content expansion")
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Additional expansion strategies failed: {e}")

    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape content from a URL using Playwright.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object with the results
        """
        logger.info(f"Starting to scrape URL with Playwright: {url}")

        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")

            if async_playwright is None:
                raise ImportError("Playwright is not available")

            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor"
                    ]
                )

                # Create context with realistic settings
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                    timezone_id="America/New_York",
                )

                # Create page
                page = await context.new_page()
                page.set_default_timeout(self.timeout * 1000)

                # Navigate to the page
                logger.info(f"Navigating to: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)

                # Wait for content to load
                logger.info(f"Waiting {self.wait_time}ms for content to load...")
                await page.wait_for_timeout(self.wait_time)

                # Handle cookie popups and other overlays
                await self._handle_cookie_popups(page)

                # Handle content-revealing buttons (e.g., "Continue reading", "Show more")
                if self.handle_interactive_content:
                    # First check if there's potentially truncated content
                    has_truncated = await self._detect_truncated_content(page)
                    if has_truncated:
                        logger.debug("Detected potentially truncated content, attempting to expand...")

                    await self._handle_content_revealing_buttons(page)

                    # If we detected truncated content but didn't click any buttons,
                    # try some additional strategies
                    if has_truncated:
                        await self._try_additional_expansion_strategies(page)

                # Get the page content
                html_content = await page.content()
                title = await page.title()

                await browser.close()

                # Clean the content
                logger.info("Cleaning extracted content...")
                cleaned_content = self._clean_text(html_content)
                word_count = len(cleaned_content.split())

                logger.info(f"Successfully scraped {word_count:,} words from {url}")

                return ScrapedContent(
                    url=url,
                    title=title,
                    content=cleaned_content,
                    word_count=word_count,
                    scraped_at=datetime.now().isoformat(),
                    success=True
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to scrape {url}: {error_msg}")

            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                scraped_at=datetime.now().isoformat(),
                success=False,
                error_message=error_msg
            )

    async def scrape_and_save(self, url: str, output_path: str | None = None) -> ScrapedContent:
        """Scrape content from URL and save to file.
        
        Args:
            url: URL to scrape
            output_path: Path to save the content. If None, content is not saved.
            
        Returns:
            ScrapedContent object with the results
        """
        result = await self.scrape_url(url)

        if output_path and result.success:
            try:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Save as text file with metadata header
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"URL: {result.url}\n")
                    f.write(f"Title: {result.title}\n")
                    f.write(f"Scraped at: {result.scraped_at}\n")
                    f.write(f"Word count: {result.word_count}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(result.content)

                logger.info(f"Saved scraped content to: {output_file}")

            except Exception as e:
                logger.error(f"Failed to save content to {output_path}: {e}")

        return result


def create_async_playwright_scraper(
    timeout: int = 60,
    wait_time: int = 2000,
    headless: bool = True,
    handle_interactive_content: bool = True,
    max_content_buttons: int = 5,
) -> AsyncPlaywrightScraper:
    """Create an async Playwright scraper instance.

    Args:
        timeout: Page load timeout in seconds
        wait_time: Time to wait for content to load in milliseconds
        headless: Whether to run browser in headless mode
        handle_interactive_content: Whether to click content-revealing buttons
        max_content_buttons: Maximum number of content buttons to click

    Returns:
        Configured AsyncPlaywrightScraper instance
    """
    return AsyncPlaywrightScraper(
        timeout=timeout,
        wait_time=wait_time,
        headless=headless,
        handle_interactive_content=handle_interactive_content,
        max_content_buttons=max_content_buttons,
    )
