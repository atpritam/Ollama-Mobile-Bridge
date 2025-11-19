"""
Persistent cache with TTL support for search results and scraped content.
Uses SQLite for persistence and similarity-based lookup to improve cache hit rate.
"""
import time
import hashlib
import re
import sqlite3
import json
import threading
from pathlib import Path
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
    simhash: int  # Pre-computed simhash for fast similarity matching


class SearchCache:
    """
    Persistent SQLite-backed cache for search results with automatic TTL.
    """

    # TTL configurations (in seconds)
    TTL = {
        "weather": 30 * 60,
        "google": 15 * 60 * 60,
        "reddit": 8 * 60 * 60,
        "wikipedia": 5 * 24 * 60 * 60,
        "default": 2 * 60 * 60
    }

    def __init__(self, max_size: int = 500, db_path: Optional[str] = None):
        """
        Initialize search cache with SQLite persistence.

        Args:
            max_size: Maximum number of cached searches (default 500)
            db_path: Path to SQLite database file (default: ./data/cache.db)
        """
        self._max_size = max_size
        self._similarity_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._simhash_distance = Config.CACHE_SIMHASH_DISTANCE
        self._use_synonyms = Config.CACHE_USE_SYNONYMS
        self._max_synonyms = Config.CACHE_MAX_SYNONYMS

        # Setup SQLite database
        if db_path is None:
            db_dir = Path("data")
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "cache.db")

        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

        # Clean up expired entries on startup
        self._evict_expired()

        app_logger.info(f"Cache initialized with SQLite: {self._db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # WAL mode for better concurrent read/write performance
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                search_type TEXT NOT NULL,
                query TEXT NOT NULL,
                scraped_contents TEXT NOT NULL,
                summaries TEXT,
                expires_at REAL NOT NULL,
                metadata TEXT NOT NULL,
                simhash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        # indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_type_expires
            ON cache_entries(search_type, expires_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at
            ON cache_entries(expires_at)
        """)

        conn.commit()

        # Get max search_id from existing entries
        cursor.execute("SELECT MAX(json_extract(metadata, '$.search_id')) FROM cache_entries")
        result = cursor.fetchone()
        self._search_counter = int(result[0]) if result[0] else 0

    @staticmethod
    def _serialize_entry(entry: CacheEntry) -> Tuple[str, str, str]:
        """Serialize cache entry for database storage."""
        scraped_json = json.dumps(entry.scraped_contents)
        metadata_json = json.dumps(entry.metadata)
        return scraped_json, metadata_json, entry.summaries

    @staticmethod
    def _deserialize_entry(row: sqlite3.Row) -> CacheEntry:
        """Deserialize cache entry from database row."""
        return CacheEntry(
            scraped_contents=json.loads(row['scraped_contents']),
            summaries=row['summaries'],
            expires_at=row['expires_at'],
            metadata=json.loads(row['metadata']),
            simhash=int(row['simhash'])
        )

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
        """Remove expired entries from database."""
        now = time.time()
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?", (now,))
        count = cursor.fetchone()[0]

        if count > 0:
            cursor.execute("DELETE FROM cache_entries WHERE expires_at < ?", (now,))
            conn.commit()
            app_logger.debug(f"Cache: evicted {count} expired entries")

    def _evict_lru(self) -> None:
        """Evict oldest entries if cache is full."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        current_count = cursor.fetchone()[0]

        if current_count >= self._max_size:
            # Remove 10% oldest entries
            to_remove = max(1, self._max_size // 10)
            cursor.execute("""
                DELETE FROM cache_entries
                WHERE cache_key IN (
                    SELECT cache_key FROM cache_entries
                    ORDER BY created_at ASC
                    LIMIT ?
                )
            """, (to_remove,))
            conn.commit()
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
        query = self._clean_query(query)
        conn = self._get_conn()
        cursor = conn.cursor()
        now = time.time()

        # Step 1: Try exact match
        key = self._make_key(search_type, query)
        cursor.execute("""
            SELECT * FROM cache_entries
            WHERE cache_key = ? AND expires_at >= ?
        """, (key, now))

        row = cursor.fetchone()
        if row:
            entry = self._deserialize_entry(row)
            app_logger.info(f"Cache HIT (exact): {search_type} | '{query[:50]}'")
            combined_results, source_urls = self._combine_entry_contents(entry)
            return combined_results, source_urls, entry.metadata

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

    @staticmethod
    def _combine_entry_contents(entry: CacheEntry) -> Tuple[str, Optional[str]]:
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
        Uses pre-computed simhashes from cache entries for fast filtering.

        Args:
            search_type: Type of search to match within
            query: New search query

        Returns:
            Tuple of (cached_query, similarity_score, cache_entry) or None
        """
        now = time.time()
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get all valid entries for this search type
        cursor.execute("""
            SELECT * FROM cache_entries
            WHERE search_type = ? AND expires_at >= ?
        """, (search_type, now))

        rows = cursor.fetchall()
        if not rows:
            return None

        type_queries: Dict[str, CacheEntry] = {}
        for row in rows:
            entry = self._deserialize_entry(row)
            cached_query = entry.metadata.get("query")
            if cached_query:
                type_queries[cached_query] = entry

        if not type_queries:
            return None

        # Build map of cached queries to their pre-computed simhashes
        cached_simhashes = {
            query: entry.simhash
            for query, entry in type_queries.items()
        }

        # Find similar queries using hybrid similarity + simhash + synonym expansion
        similar = TextSimilarity.find_similar_queries(
            new_query=query,
            cached_queries=list(type_queries.keys()),
            threshold=self._similarity_threshold,
            use_simhash=True,
            simhash_threshold=self._simhash_distance,
            use_synonyms=self._use_synonyms,
            max_synonyms=self._max_synonyms,
            cached_simhashes=cached_simhashes
        )

        # Return best match if found
        if similar:
            best_query, best_score = similar[0]
            return best_query, best_score, type_queries[best_query]

        return None

    def get_cached_urls(self, urls: list[str], search_type: str = "") -> Dict[str, str]:  # noqa: ARG002
        """
        Check which URLs have cached content and return them.

        Args:
            urls: List of URLs to check
            search_type: Reserved for future use (currently unused)

        Returns:
            Dict mapping cached URLs to their content
        """
        _ = search_type  # Keep parameter for API compatibility
        now = time.time()
        conn = self._get_conn()
        cursor = conn.cursor()
        cached_urls = {}

        # Get all valid cache entries
        cursor.execute("""
            SELECT * FROM cache_entries
            WHERE expires_at >= ?
        """, (now,))

        rows = cursor.fetchall()

        # Search through entries for matching URLs
        for row in rows:
            entry = self._deserialize_entry(row)

            # Check if this entry has any of the requested URLs
            for url in urls:
                if url in entry.scraped_contents:
                    cached_urls[url] = entry.scraped_contents[url]
                    if len(cached_urls) == len(urls):
                        if cached_urls:
                            app_logger.info(
                                f"Cache HIT (URL): Found {len(cached_urls)}/{len(urls)} URLs in cache"
                            )
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
        created_at = time.time()
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

        # Pre-compute simhash for fast similarity matching
        normalized_query = TextSimilarity.normalize_query(query)
        query_simhash = TextSimilarity.simhash(normalized_query)

        # Serialize data for database
        scraped_json = json.dumps(scraped_contents or {})
        metadata_json = json.dumps(metadata)

        # Insert or replace in database
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO cache_entries
            (cache_key, search_type, query, scraped_contents, summaries, expires_at, metadata, simhash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (key, search_type, query, scraped_json, summaries, expires_at, metadata_json, str(query_simhash), created_at))

        conn.commit()

        url_count = len(scraped_contents) if scraped_contents else 0
        app_logger.info(
            f"Cache SET: {search_id} | {search_type} | '{query[:50]}' | "
            f"{url_count} URLs (TTL: {ttl/60:.0f}min)"
        )
        return search_id

    def get_by_id(self, search_id: str) -> Optional[Tuple[str, Optional[str], dict]]:
        """Get cached search by search ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            SELECT * FROM cache_entries
            WHERE expires_at >= ?
        """, (now,))

        rows = cursor.fetchall()
        for row in rows:
            entry = self._deserialize_entry(row)
            if entry.metadata.get("search_id") == search_id:
                combined_results, source_urls = self._combine_entry_contents(entry)
                app_logger.info(f"Cache recall: {search_id}")
                return combined_results, source_urls, entry.metadata

        return None

    def get_recent_searches(self, limit: int = 5) -> list[dict]:
        """Get metadata of recent searches."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            SELECT metadata FROM cache_entries
            WHERE expires_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (now, limit))

        rows = cursor.fetchall()
        return [json.loads(row['metadata']) for row in rows]

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM cache_entries")
        conn.commit()

        app_logger.info(f"Cache cleared: {count} entries removed")

# Global cache instance
_search_cache = SearchCache(max_size=500)


def get_search_cache() -> SearchCache:
    """Get the global search cache instance."""
    return _search_cache