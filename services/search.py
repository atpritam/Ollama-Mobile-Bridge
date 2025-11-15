"""
Web search service using Brave Search API with content scraping.
"""
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
import httpx

from config import Config
from utils.constants import SearchType
from utils.html_parser import HTMLParser
from utils.logger import app_logger

if TYPE_CHECKING:
    from routes.chat import ChatContext


class SearchService:
    """Service for performing web searches and scraping content."""

    def __init__(self, context: 'ChatContext'):
        """
        Initialize SearchService with context.

        Args:
            context: ChatContext containing request-scoped data
        """
        self.context = context

    def _get_search_params(self, search_type: str, query: str) -> Dict[str, Any]:
        """
        Build search parameters optimized for specific search types.

        Args:
            search_type: Type of search (google, wikipedia, reddit, etc.)
            query: Search query string

        Returns:
            Dictionary of search parameters for Brave API
        """
        # Base parameters
        params: Dict[str, Any] = {
            "q": query,
            "count": Config.DEFAULT_SEARCH_RESULTS_COUNT,
        }

        if search_type == SearchType.WIKIPEDIA:
            # For Wikipedia: fewer high-quality results
            params["count"] = 3
            params["search_lang"] = "en"

        elif search_type == SearchType.REDDIT:
            # For Reddit: more results to capture discussions, prefer recent content
            params["count"] = 8
            params["freshness"] = "pw"

        elif search_type == SearchType.GOOGLE:
            # For general search:
            params["count"] = Config.DEFAULT_SEARCH_RESULTS_COUNT

        app_logger.debug(f"Search params for {search_type}: {params}")
        return params

    def _get_result_count_for_display(self, search_type: str) -> int:
        """Get the number of search results to display based on search type."""
        if search_type == SearchType.WIKIPEDIA:
            return 3
        elif search_type == SearchType.REDDIT:
            return 6
        else:
            return 5

    async def perform_search(self, search_type: str, query: str) -> Tuple[str, Optional[str]]:
        """
        Perform web search using Brave Search API with type-specific optimizations.

        Args:
            search_type: Type of search (google, wikipedia, reddit, weather)
            query: Search query string

        Returns:
            Tuple of (search_results, source_url)
        """
        if not Config.BRAVE_SEARCH_API_KEY:
            return f"Search query: '{query}'\nSearch API key not configured. Cannot perform search.", None

        try:
            async with httpx.AsyncClient(timeout=Config.SEARCH_TIMEOUT, follow_redirects=True) as client:
                # type-specific search parameters
                params = self._get_search_params(search_type, query)

                # Get search results from Brave
                response = await client.get(
                    Config.BRAVE_SEARCH_URL,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": Config.BRAVE_SEARCH_API_KEY
                    },
                    params=params
                )

                if response.status_code == 200:
                    data = response.json()
                    return await self._process_search_results(data, client, search_type)
                elif response.status_code == 401:
                    return f"Search query: '{query}'\nInvalid API key. Please check your BRAVE_SEARCH_API_KEY.", None
                elif response.status_code == 429:
                    return f"Search query: '{query}'\nAPI rate limit exceeded. Please try again later.", None
                else:
                    return f"Search query: '{query}'\nSearch API error (status {response.status_code}).", None

        except Exception as e:
            app_logger.error(f"Search failed: {str(e)}")
            return f"Search query: '{query}'\nSearch failed: {str(e)}", None

    async def _process_search_results(self, data: dict, client: httpx.AsyncClient, search_type: str) -> Tuple[str, Optional[str]]:
        """
        Process search results and scrape content with type-specific handling.

        Args:
            data: Search API response data
            client: HTTP client for scraping
            search_type: Type of search to optimize result processing

        Returns:
            Tuple of (processed_results, source_url)
        """
        results = []
        source_url = None

        # Infobox/Featured snippet
        if data.get("infobox"):
            infobox = data["infobox"]
            if infobox.get("description"):
                results.append(f"Quick Answer: {infobox['description']}")

        # Scrape content from top result(s)
        if data.get("web") and data["web"].get("results"):
            web_results = data["web"]["results"]

            # For Reddit: scrape multiple threads
            if search_type == SearchType.REDDIT:
                reddit_results = [r for r in web_results[:8] if "reddit.com" in r.get("url", "")][:3]

                scraped_count = 0
                for idx, result in enumerate(reddit_results):
                    result_url = result.get("url")
                    if not result_url:
                        continue

                    if source_url is None:
                        source_url = result_url

                    scraped_content = await self._scrape_page(
                        client, result_url, result.get('title', f'Reddit thread {idx+1}'), search_type
                    )

                    if scraped_content:
                        results.append(scraped_content)
                        scraped_count += 1

                app_logger.info(f"Scraped {scraped_count} Reddit threads for diverse opinions")

            else:
                target_result = self._select_best_result(web_results, search_type)

                if target_result:
                    target_url = target_result.get("url")
                    source_url = target_url

                    if target_url:
                        scraped_content = await self._scrape_page(
                            client, target_url, target_result.get('title', 'top result'), search_type
                        )
                        if scraped_content:
                            results.append(scraped_content)

        if data.get("web") and data["web"].get("results"):
            display_count = self._get_result_count_for_display(search_type)
            web_results = self._format_web_results(data["web"]["results"], display_count)
            if web_results:
                results.append("Additional Search Results:\n" + "\n\n".join(web_results))

        if results:
            combined_results = "\n\n".join(results)
            app_logger.info(f"Returning {len(combined_results)} chars total to LLM ({len(results)} content sections)")
            return combined_results, source_url
        else:
            return "No results found.", None

    def _select_best_result(self, results: List[dict], search_type: str) -> Optional[dict]:
        """
        Select the best result to scrape based on search type.

        Args:
            results: List of search results
            search_type: Type of search

        Returns:
            Best result to scrape, or None
        """
        if not results:
            return None

        # For Wikipedia searches, prioritize wikipedia.org URLs
        if search_type == SearchType.WIKIPEDIA:
            for result in results[:3]:
                url = result.get("url", "")
                if "wikipedia.org" in url:
                    app_logger.info(f"Selected Wikipedia URL: {url}")
                    return result

        # For Reddit searches, prioritize reddit.com URLs
        elif search_type == SearchType.REDDIT:
            for result in results[:5]:
                url = result.get("url", "")
                if "reddit.com" in url:
                    app_logger.info(f"Selected Reddit URL: {url}")
                    return result

        return results[0]

    async def _scrape_page(self, client: httpx.AsyncClient, url: str, title: str, search_type: str) -> Optional[str]:
        """
        Scrape content from a webpage with type-specific handling.
        Falls back to Jina Reader for JavaScript-heavy sites.

        Args:
            client: HTTP client for making requests
            url: URL to scrape
            title: Title of the page
            search_type: Type of search to optimize content extraction

        Returns:
            Scraped content or None if scraping fails
        """
        try:
            app_logger.info(f"Scraping content from: {url} (type: {search_type})")
            page_response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=Config.WEB_SCRAPING_TIMEOUT
            )

            if page_response.status_code == 200:
                if "wikipedia.org" in url:
                    search_type = "Wikipedia"
                elif "reddit.com" in url:
                    search_type = "Reddit"
                extraction_type = search_type.capitalize()
                max_length = self._get_max_content_length(search_type)

                content = HTMLParser.extract_text(page_response.text, max_length, url=url)

                # javaScript-rendered page - try Jina Reader
                if content and len(content) < 200:
                    jina_content = await self._scrape_with_jina(client, url, max_length)
                    if jina_content and len(jina_content) > len(content):
                        content = jina_content

                if content:
                    app_logger.info(f"Successfully scraped {len(content)} chars (search_type: {search_type})")

                    result = f"=== Content from: {title} ===\n"
                    result += f"Source: {url}\n"
                    result += content

                    return result
        except Exception as scrape_error:
            app_logger.warning(f"Scraping failed: {scrape_error}")

        return None

    async def _scrape_with_jina(self, client: httpx.AsyncClient, url: str, max_length: int) -> Optional[str]:
        """
        Scrape content using Jina Reader API for JavaScript-rendered pages.

        Args:
            client: HTTP client for making requests
            url: URL to scrape
            max_length: Maximum content length

        Returns:
            Scraped content or None if scraping fails
        """
        try:
            jina_url = f"https://r.jina.ai/{url}"

            jina_response = await client.get(
                jina_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/plain"
                },
                timeout=10.0
            )

            if jina_response.status_code == 200:
                content = jina_response.text.strip()

                # Limit to max_length
                if len(content) > max_length:
                    content = content[:max_length]
                    last_period = content.rfind('.')
                    if last_period > max_length * 0.7:
                        content = content[:last_period + 1]

                return content
            else:
                app_logger.warning(f"Jina Reader failed with status {jina_response.status_code}")
                return None

        except Exception as e:
            app_logger.warning(f"Jina Reader error: {e}")
            return None

    def _get_max_content_length(self, search_type: str) -> int:
        """
        Get maximum content length based on search type and model size from context.

        Args:
            search_type: Type of search

        Returns:
            Maximum content length in characters
        """
        # Access model name directly from context - no parameter chaining needed!
        return Config.get_max_html_text_length(self.context.model_name)

    def _format_web_results(self, results: List[dict], display_count: int = 5) -> List[str]:
        """
        Format web search results into readable text.

        Args:
            results: List of search result dictionaries
            display_count: Number of results to display

        Returns:
            List of formatted result strings
        """
        web_results = []
        for idx, result in enumerate(results[:display_count], 1):
            title = result.get("title", "")
            description = result.get("description", "")
            url = result.get("url", "")

            if title and description:
                result_text = f"{idx}. {title}\n   {description}"
                if url:
                    result_text += f"\n   URL: {url}"
                web_results.append(result_text)

        return web_results