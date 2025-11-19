"""
Simple in-memory cache with TTL support for search results and scraped content.
Uses similarity-based lookup to improve cache hit rate for semantically similar queries.
"""
import time
import hashlib
import re
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from config import Config
from utils.logger import app_logger
from utils.text_similarity import TextSimilarity


@dataclass
class CacheEntry:
    """Cache entry with structured content per URL and summaries."""
    scraped_contents: Dict[str, str]  # URL -> content mapping
    summaries: Optional[str]  # Additional search results/summaries
    expires_at: float  # Applies to all contents in this entry
    metadata: dict  # search_type, query, timestamp, sources


class SearchCache:
    """
    Lightweight in-memory cache for search results with automatic TTL.
    """

    # TTL configurations (in seconds)
    TTL = {
        "weather": 30 * 60,
        "google": 15 * 60 * 60,
        "reddit": 8 * 60 * 60,
        "wikipedia": 5 * 24 * 60 * 60,
        "default": 2 * 60 * 60
    }

    def __init__(self, max_size: int = 500):
        """
        Initialize search cache.

        Args:
            max_size: Maximum number of cached searches (default 500 = ~8MB RAM)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._search_counter = 0
        self._similarity_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._simhash_distance = Config.CACHE_SIMHASH_DISTANCE
        self._use_synonyms = Config.CACHE_USE_SYNONYMS
        self._max_synonyms = Config.CACHE_MAX_SYNONYMS

    @staticmethod
    def _clean_query(query: str) -> str:
        """Remove site: prefixes from query for consistent caching."""
        cleaned = re.sub(r'site:\S+\s*', '', query, flags=re.IGNORECASE)
        return cleaned.strip()

    def _make_key(self, search_type: str, query: str) -> str:
        """Create cache key from search type and cleaned query."""
        clean_query = self._clean_query(query)
        query_hash = hashlib.md5(clean_query.lower().strip().encode()).hexdigest()[:12]
        return f"{search_type}:{query_hash}"

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if v.expires_at < now]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            app_logger.debug(f"Cache: evicted {len(expired_keys)} expired entries")

    def _evict_lru(self) -> None:
        """Evict oldest entries if cache is full."""
        if len(self._cache) >= self._max_size:
            # Remove 10% oldest entries
            to_remove = max(1, self._max_size // 10)
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1].expires_at)
            for key, _ in sorted_items[:to_remove]:
                del self._cache[key]
            app_logger.debug(f"Cache: LRU evicted {to_remove} entries")

    def get(self, search_type: str, query: str) -> Optional[Tuple[str, Optional[str], dict]]:
        """
        Get cached search result using exact match first, then similarity-based fuzzy matching.

        Args:
            search_type: Type of search (google, reddit, etc.)
            query: Search query

        Returns:
            Tuple of (results, source_url, metadata) or None if not cached/expired
        """
        self._evict_expired()
        query = self._clean_query(query)
        # Step 1: Try exact match
        key = self._make_key(search_type, query)
        if key in self._cache:
            entry = self._cache[key]
            if entry.expires_at >= time.time():
                app_logger.info(f"Cache HIT (exact): {search_type} | '{query[:50]}'")
                combined_results, source_urls = self._combine_entry_contents(entry)
                return combined_results, source_urls, entry.metadata

            del self._cache[key]

        # Step 2: Try similarity-based fuzzy matching within same search type
        similar_match = self._find_similar_cached_query(search_type, query)

        if similar_match:
            cached_query, similarity_score, entry = similar_match
            app_logger.info(
                f"Cache HIT (fuzzy): {search_type} | similarity={similarity_score:.3f} | "
                f"original='{cached_query[:40]}' | new='{query[:40]}'"
            )
            combined_results, source_urls = self._combine_entry_contents(entry)
            return combined_results, source_urls, entry.metadata

        # Step 3: No match found
        app_logger.debug(f"Cache MISS: {search_type} | '{query[:50]}'")
        return None

    def _combine_entry_contents(self, entry: CacheEntry) -> Tuple[str, Optional[str]]:
        """Combine all scraped contents and summaries from a cache entry."""
        results = []

        # Add all scraped contents
        for url, content in entry.scraped_contents.items():
            if content:
                results.append(content)

        # Add summaries if present
        if entry.summaries:
            results.append(entry.summaries)

        combined_results = "\n\n".join(results) if results else ""
        source_urls = ", ".join(entry.scraped_contents.keys()) if entry.scraped_contents else None

        return combined_results, source_urls

    def _find_similar_cached_query(
        self, search_type: str, query: str
    ) -> Optional[Tuple[str, float, CacheEntry]]:
        """
        Find a cached query similar to the new query within the same search type.

        Args:
            search_type: Type of search to match within
            query: New search query

        Returns:
            Tuple of (cached_query, similarity_score, cache_entry) or None
        """
        now = time.time()
        type_queries: Dict[str, CacheEntry] = {}
        for cache_key, entry in self._cache.items():
            if (entry.expires_at >= now and
                entry.metadata.get("search_type") == search_type):
                cached_query = entry.metadata.get("query")
                if cached_query:
                    type_queries[cached_query] = entry

        if not type_queries:
            return None

        # Find similar queries using hybrid similarity + simhash + synonym expansion
        similar = TextSimilarity.find_similar_queries(
            new_query=query,
            cached_queries=list(type_queries.keys()),
            threshold=self._similarity_threshold,
            use_simhash=True,
            simhash_threshold=self._simhash_distance,
            use_synonyms=self._use_synonyms,
            max_synonyms=self._max_synonyms
        )

        # Return best match if found
        if similar:
            best_query, best_score = similar[0]
            return best_query, best_score, type_queries[best_query]

        return None

    def get_cached_urls(self, urls: list[str], search_type: str) -> Dict[str, str]:
        """
        Check which URLs have cached content and return them.

        Args:
            urls: List of URLs to check
            search_type: Type of search (for TTL matching)

        Returns:
            Dict mapping cached URLs to their content
        """
        self._evict_expired()
        now = time.time()
        cached_urls = {}

        # Search through all cache entries for matching URLs
        for entry in self._cache.values():
            if entry.expires_at < now:
                continue

            # Check if this entry has any of the requested URLs
            for url in urls:
                if url in entry.scraped_contents:
                    cached_urls[url] = entry.scraped_contents[url]
                    if len(cached_urls) == len(urls):
                        return cached_urls

        if cached_urls:
            app_logger.info(
                f"Cache HIT (URL): Found {len(cached_urls)}/{len(urls)} URLs in cache"
            )

        return cached_urls

    def set(
        self,
        search_type: str,
        query: str,
        scraped_contents: Optional[Dict[str, str]] = None,
        summaries: Optional[str] = None
    ) -> int:
        """
        Cache search result with structured content per URL.

        Args:
            search_type: Type of search
            query: Search query
            scraped_contents: Dict mapping URLs to their scraped content
            summaries: Additional search result summaries

        Returns:
            Search ID for reference in conversation history
        """
        self._evict_lru()
        query = self._clean_query(query)
        key = self._make_key(search_type, query)
        ttl = self.TTL.get(search_type, self.TTL["default"])
        expires_at = time.time() + ttl
        self._search_counter += 1
        search_id = self._search_counter

        # Extract source domains
        sources = []
        if scraped_contents:
            from urllib.parse import urlparse
            for url in scraped_contents.keys():
                sources.append(urlparse(url).netloc)

        metadata = {
            "search_id": search_id,
            "search_type": search_type,
            "query": query,
            "timestamp": time.time(),
            "ttl_hours": ttl / 3600,
            "sources": sources
        }

        self._cache[key] = CacheEntry(
            scraped_contents=scraped_contents or {},
            summaries=summaries,
            expires_at=expires_at,
            metadata=metadata
        )

        url_count = len(scraped_contents) if scraped_contents else 0
        app_logger.info(
            f"Cache SET: {search_id} | {search_type} | '{query[:50]}' | "
            f"{url_count} URLs (TTL: {ttl/60:.0f}min)"
        )
        return search_id

    def get_by_id(self, search_id: str) -> Optional[Tuple[str, Optional[str], dict]]:
        """Get cached search by search ID."""
        for entry in self._cache.values():
            if entry.metadata["search_id"] == search_id and entry.expires_at >= time.time():
                combined_results, source_urls = self._combine_entry_contents(entry)
                app_logger.info(f"Cache recall: {search_id}")
                return combined_results, source_urls, entry.metadata
        return None

    def get_recent_searches(self, limit: int = 5) -> list[dict]:
        """Get metadata of recent searches."""
        valid_entries = [
            entry.metadata for entry in self._cache.values()
            if entry.expires_at >= time.time()
        ]
        # Sort by timestamp, newest first
        valid_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return valid_entries[:limit]

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        count = len(self._cache)
        self._cache.clear()
        app_logger.info(f"Cache cleared: {count} entries removed")

# Global cache instance
_search_cache = SearchCache(max_size=500)


def get_search_cache() -> SearchCache:
    """Get the global search cache instance."""
    return _search_cache