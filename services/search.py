"""
Web search service using Brave Search API with content scraping.
"""
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING, LiteralString, Coroutine
import asyncio
import httpx

from config import Config
from utils.constants import SearchType
from utils.html_parser import HTMLParser
from utils.logger import app_logger
from utils.http_client import HTTPClientManager
from utils.cache import get_search_cache

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

    async def perform_search(self, search_type: str, query: str) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Perform web search using Brave Search API with type-specific optimizations.
        Checks cache first to avoid redundant searches.

        Args:
            search_type: Type of search (google, wikipedia, reddit, weather)
            query: Search query string

        Returns:
            Tuple of (search_results, source_url, search_id)
            search_id is used to reference this search in conversation history
        """
        if not Config.BRAVE_SEARCH_API_KEY:
            return f"Search query: '{query}'\nSearch API key not configured. Cannot perform search.", None, None

        # Check cache first
        cache = get_search_cache()
        cached = cache.get(search_type, query)
        if cached:
            results, source_url, metadata = cached
            search_id = metadata["search_id"]
            return results, source_url, search_id

        try:
            client = HTTPClientManager.get_search_client()

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
                return await self._process_search_results(data, search_type, query)
            elif response.status_code == 401:
                return f"Search query: '{query}'\nInvalid API key. Please check your BRAVE_SEARCH_API_KEY.", None, None
            elif response.status_code == 429:
                return f"Search query: '{query}'\nAPI rate limit exceeded. Please try again later.", None, None
            else:
                return f"Search query: '{query}'\nSearch API error (status {response.status_code}).", None, None

        except httpx.TimeoutException as e:
            app_logger.error(f"Search timed out: {str(e)}")
            return f"Search query: '{query}'\nSearch timed out. Please try again.", None, None
        except httpx.RequestError as e:
            app_logger.error(f"Search request failed: {str(e)}")
            return f"Search query: '{query}'\nSearch request failed: {str(e)}", None, None
        except Exception as e:
            app_logger.error(f"Unexpected search error: {str(e)}")
            return f"Search query: '{query}'\nUnexpected error during search.", None, None

    async def _process_search_results(self, data: dict, search_type: str, query: str) -> tuple[
                                                                                             LiteralString, LiteralString | None | Any, int] | \
                                                                                         tuple[str, None, None]:
        """
        Process search results and scrape content with type-specific handling.
        Caches results for future use.

        Args:
            data: Search API response data
            search_type: Type of search to optimize result processing
            query: Original search query (for caching)

        Returns:
            Tuple of (processed_results, source_url, search_id)
        """
        client = HTTPClientManager.get_general_client()

        scraped_contents_dict = {}  # URL -> content mapping
        source_url = None
        summaries_text = None

        # Infobox/Featured snippet
        infobox_content = None
        if data.get("infobox"):
            infobox = data["infobox"]
            if infobox.get("description"):
                infobox_content = f"Quick Answer: {infobox['description']}"

        # Scrape content from top result(s)
        if data.get("web") and data["web"].get("results"):
            web_results = data["web"]["results"]

            # Dynamic multi-page scraping based on search type and model size
            scrape_count_target = self._get_scrape_count(search_type)
            max_total_length = self._get_max_content_length(search_type)
            max_length_per_page = max_total_length // scrape_count_target if scrape_count_target > 0 else max_total_length

            if search_type == SearchType.REDDIT:
                reddit_results = [r for r in web_results[:8] if "reddit.com" in r.get("url", "")][:scrape_count_target]

                # Fallback: if no Reddit URLs found, use top general results
                if not reddit_results:
                    reddit_results = web_results[:scrape_count_target]

                # Get URLs to check/scrape
                urls_to_check = [r.get("url") for r in reddit_results if r.get("url")]

                # Check cache for these URLs
                cache = get_search_cache()
                cached_url_contents = cache.get_cached_urls(urls_to_check, search_type)
                scraped_contents_dict.update(cached_url_contents)

                # Filter out cached URLs
                urls_to_scrape = [url for url in urls_to_check if url not in cached_url_contents]
                url_to_result = {r.get("url"): r for r in reddit_results if r.get("url")}

                scrape_tasks = []
                task_urls = []
                for url in urls_to_scrape:
                    result = url_to_result.get(url)
                    if not result:
                        continue

                    if source_url is None:
                        source_url = url

                    task = self._scrape_page(
                        client, url, result.get('title', 'Reddit thread'), search_type, max_length_per_page
                    )
                    scrape_tasks.append(task)
                    task_urls.append(url)

                if scrape_tasks:
                    scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)

                    scraped_count = 0
                    for idx, content in enumerate(scraped_contents):
                        if isinstance(content, Exception):
                            app_logger.warning(f"Reddit scraping failed for {task_urls[idx]}: {content}")
                        elif content:
                            scraped_contents_dict[task_urls[idx]] = content
                            scraped_count += 1
                        else:
                            app_logger.warning(f"Reddit scraping returned empty for {task_urls[idx]}")

                    app_logger.info(
                        f"Scraped {scraped_count}/{len(task_urls)} Reddit thread"
                        + ("s in parallel" if scraped_count > 1 else "")
                    )

                total_threads = len(scraped_contents_dict)
                app_logger.info(f"Total Reddit content: {total_threads} threads ({len(cached_url_contents)} from cache, {len(task_urls)} scraped)")

            elif search_type == SearchType.WIKIPEDIA:
                wiki_results = [r for r in web_results[:5] if "wikipedia.org" in r.get("url", "")][:scrape_count_target]

                # Fallback: if no Wikipedia URLs found, use top general results
                if not wiki_results:
                    wiki_results = web_results[:scrape_count_target]

                # Get URLs to check/scrape
                urls_to_check = [r.get("url") for r in wiki_results if r.get("url")]

                # Check cache for these URLs
                cache = get_search_cache()
                cached_url_contents = cache.get_cached_urls(urls_to_check, search_type)
                scraped_contents_dict.update(cached_url_contents)

                # Filter out cached URLs
                urls_to_scrape = [url for url in urls_to_check if url not in cached_url_contents]
                url_to_result = {r.get("url"): r for r in wiki_results if r.get("url")}

                scrape_tasks = []
                task_urls = []
                for url in urls_to_scrape:
                    result = url_to_result.get(url)
                    if not result:
                        continue

                    if source_url is None:
                        source_url = url

                    task = self._scrape_page(
                        client, url, result.get('title', 'Wikipedia article'), search_type, max_length_per_page
                    )
                    scrape_tasks.append(task)
                    task_urls.append(url)

                if scrape_tasks:
                    scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)

                    scraped_count = 0
                    for idx, content in enumerate(scraped_contents):
                        if content and not isinstance(content, Exception):
                            scraped_contents_dict[task_urls[idx]] = content
                            scraped_count += 1

                    app_logger.info(
                        f"Scraped {scraped_count}/{len(task_urls)} Wikipedia articles"
                        + (" in parallel" if scraped_count > 1 else "")
                    )

                total_articles = len(scraped_contents_dict)
                app_logger.info(f"Total Wikipedia content: {total_articles} articles ({len(cached_url_contents)} from cache, {len(task_urls)} scraped)")

            else:
                # Google: Parallel scraping with smart result selection and fallback
                max_total_attempts = min(len(web_results), scrape_count_target * 3)
                scraped_count = 0
                urls_tried = 0
                batch_size = scrape_count_target

                # Get cache once before the loop
                cache = get_search_cache()

                while scraped_count < scrape_count_target and urls_tried < max_total_attempts:
                    batch_start = urls_tried
                    batch_end = min(urls_tried + batch_size, max_total_attempts)
                    batch_results = web_results[batch_start:batch_end]

                    if not batch_results:
                        break

                    # Get URLs for this batch
                    batch_urls = [r.get("url") for r in batch_results if r.get("url")]

                    # Check cache for batch URLs
                    cached_url_contents = cache.get_cached_urls(batch_urls, search_type)
                    scraped_contents_dict.update(cached_url_contents)
                    scraped_count += len(cached_url_contents)

                    scrape_tasks = []
                    task_urls = []
                    for result in batch_results:
                        result_url = result.get("url")
                        if not result_url or result_url in cached_url_contents:
                            continue

                        if scraped_count >= scrape_count_target:
                            break

                        if source_url is None:
                            source_url = result_url

                        task = self._scrape_page(
                            client, result_url, result.get('title', 'Search result'), search_type, max_length_per_page
                        )
                        scrape_tasks.append(task)
                        task_urls.append(result_url)

                    if not scrape_tasks:
                        urls_tried += len(batch_results)
                        if scraped_count >= scrape_count_target:
                            break
                        continue

                    # Scrape batch in parallel
                    scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                    urls_tried += len(task_urls)

                    # Collect successful results from batch
                    for idx, content in enumerate(scraped_contents):
                        if scraped_count >= scrape_count_target:
                            break

                        if isinstance(content, Exception):
                            app_logger.warning(f"Scraping failed for {task_urls[idx]}: {content}")
                        elif content:
                            scraped_contents_dict[task_urls[idx]] = content
                            scraped_count += 1
                        else:
                            app_logger.warning(f"Scraping returned empty for {task_urls[idx]}")

                    # If we got enough results, stop trying
                    if scraped_count >= scrape_count_target:
                        break

                total_pages = len(scraped_contents_dict)
                app_logger.info(
                    f"Scraped {scraped_count}/{scrape_count_target} Google page"
                    + ("s in parallel" if scraped_count > 1 else "")
                    + f" (tried {urls_tried} URLs, {total_pages} total pages)"
                )

        # Additional search results (Brave summaries)
        if data.get("web") and data["web"].get("results"):
            display_count = self._get_result_count_for_display(search_type)
            web_results = self._format_web_results(data["web"]["results"], display_count)
            if web_results:
                dedicated_content = "\n\n".join(scraped_contents_dict.values())
                dedicated_chars = len(dedicated_content)
                max_length = self._get_max_content_length(search_type)
                remaining_budget = max_length - dedicated_chars
                summary_limit = max(remaining_budget, Config.MIN_SUMMARY_CHARS)

                summaries_text = "Additional Search Results:\n" + "\n\n".join(web_results)
                if len(summaries_text) > summary_limit:
                    summaries_text = summaries_text[:summary_limit]
                    last_period = summaries_text.rfind('.')
                    if last_period > summary_limit * 0.8:
                        summaries_text = summaries_text[:last_period + 1]

                app_logger.info(f"Appending additional summaries {len(summaries_text)} chars")

        # Combine all content and save to cache
        if scraped_contents_dict or infobox_content or summaries_text:
            if infobox_content:
                scraped_contents_dict["_infobox"] = infobox_content

            # Build results for return
            results_list = list(scraped_contents_dict.values())
            if summaries_text:
                results_list.append(summaries_text)

            combined_results = "\n\n".join(results_list)
            final_source = ", ".join([url for url in scraped_contents_dict.keys() if not url.startswith("_")]) if scraped_contents_dict else source_url

            num_sections = len(scraped_contents_dict) + (1 if summaries_text else 0)
            app_logger.info(f"Injecting {len(combined_results)} chars total to LLM ({num_sections} content sections)")

            # Cache the results
            cache = get_search_cache()
            search_id = cache.set(search_type, query, scraped_contents_dict, summaries_text)

            return combined_results, final_source, search_id
        else:
            return "No results found.", None, None

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

    async def _scrape_wikipedia_api(self, client: httpx.AsyncClient, url: str, title: str, max_length: int) -> Optional[str]:
        """
        Fetch Wikipedia content using Wikipedia API.

        Args:
            client: HTTP client for making requests
            url: Wikipedia URL
            title: Article title
            max_length: Maximum content length

        Returns:
            Wikipedia content or None if fetch fails
        """
        try:
            import re
            from urllib.parse import unquote

            match = re.search(r'wikipedia\.org/wiki/(.+)$', url)
            if not match:
                return None

            article_title = unquote(match.group(1))
            app_logger.info(f"Fetching Wikipedia article: {url}")

            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "titles": article_title,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "plain"
            }

            response = await client.get(
                api_url,
                params=params,
                headers={
                    "User-Agent": "OllamaMobileBridge/1.0 (https://github.com/atpritam/Ollama-Mobile-Bridge; pritam@gmail.com) httpx/0.27.0"
                },
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        app_logger.warning(f"Wikipedia article not found: {article_title}")
                        return None

                    extract = page_data.get("extract", "")
                    if extract:
                        if len(extract) > max_length:
                            extract = extract[:max_length]
                            last_period = extract.rfind('.')
                            if last_period > max_length * 0.7:
                                extract = extract[:last_period + 1]

                        app_logger.info(f"Successfully fetched {len(extract)} chars from Wikipedia")

                        result = f"=== Content from: {title} ===\n"
                        result += f"Source: {url}\n"
                        result += extract

                        return result
            else:
                app_logger.warning(f"Wikipedia API returned status {response.status_code}")
                return None

        except Exception as e:
            app_logger.warning(f"Wikipedia API error for {url}: {e}")
            return None

    async def _scrape_page(self, client: httpx.AsyncClient, url: str, title: str, search_type: str, max_length: Optional[int] = None) -> Optional[str]:
        """
        Scrape content from a webpage with type-specific handling.
        Falls back to Jina Reader for JavaScript-heavy sites.

        Args:
            client: HTTP client for making requests
            url: URL to scrape
            title: Title of the page
            search_type: Type of search to optimize content extraction
            max_length: Maximum content length

        Returns:
            Scraped content or None if scraping fails
        """
        try:
            # URL Security Validation
            Config.validate_url(url)

            # Wikipedia API
            if "wikipedia.org" in url:
                if max_length is None:
                    max_length = self._get_max_content_length(SearchType.WIKIPEDIA)

                wiki_content = await self._scrape_wikipedia_api(client, url, title, max_length)
                if wiki_content:
                    return wiki_content

                # If Wikipedia API fails, try Jina Reader
                app_logger.info(f"Wikipedia API failed, falling back to Jina Reader")
                jina_content = await self._scrape_with_jina(client, url, max_length)
                if jina_content:
                    result = f"=== Content from: {title} ===\n"
                    result += f"Source: {url}\n"
                    result += jina_content
                    app_logger.info(f"Successfully fetched {len(jina_content)} chars")
                    return result

            app_logger.info(f"Scraping content from: {url} (type: {search_type})")

            async with client.stream(
                'GET', url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=Config.WEB_SCRAPING_TIMEOUT,
                follow_redirects=True
            ) as page_response:

                if page_response.status_code != 200:
                    app_logger.warning(f"Failed to fetch {url}: status {page_response.status_code}")
                    return None

                # Validate content-type before downloading
                content_type = page_response.headers.get('content-type', '').lower().split(';')[0].strip()
                if content_type and not any(allowed in content_type for allowed in Config.ALLOWED_CONTENT_TYPES):
                    app_logger.warning(f"Skipping {url}: unsupported content-type '{content_type}'")
                    return None

                size = 0
                chunks = []
                async for chunk in page_response.aiter_bytes():
                    size += len(chunk)
                    if size > Config.MAX_RESPONSE_SIZE:
                        app_logger.warning(f"Response from {url} exceeds size limit ({size} bytes)")
                        return None
                    chunks.append(chunk)

                html_content = b''.join(chunks).decode('utf-8', errors='ignore')

            if html_content:
                if "wikipedia.org" in url:
                    search_type = "Wikipedia"
                elif "reddit.com" in url:
                    search_type = "Reddit"

                if max_length is None:
                    max_length = self._get_max_content_length(search_type)

                content = HTMLParser.extract_text(html_content, max_length, url=url)

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
        except ValueError as e:
            app_logger.warning(f"Scraping validation failed for {url}: {e}")
        except httpx.TimeoutException:
            app_logger.warning(f"Scraping timed out for {url}")
        except httpx.RequestError as e:
            app_logger.warning(f"Scraping request failed for {url}: {e}")
        except UnicodeDecodeError as e:
            app_logger.warning(f"Failed to decode content from {url}: {e}")
        except Exception as e:
            app_logger.warning(f"Unexpected scraping error for {url}: {e}")

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
        """Get maximum content length based on search type and model size from context."""
        return Config.get_max_html_text_length(self.context.model_name)

    def _get_scrape_count(self, search_type: str) -> int:
        """Get number of pages to scrape based on search type and model size."""
        return Config.get_scrape_count(self.context.model_name, search_type)

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