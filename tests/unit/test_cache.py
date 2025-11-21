import pytest
import asyncio
from utils.cache import SearchCache, CacheEntry

@pytest.fixture
def memory_cache(monkeypatch):
    """Provides a SearchCache instance using in-memory dictionary for testing."""
    _cache = {}
    
    def mock_get(search_type, query):
        entry = _cache.get(query)
        if entry:
            return entry.scraped_contents, list(entry.scraped_contents.keys())[0] if entry.scraped_contents else None, entry.metadata
        return None
    
    _search_counter = 0

    def mock_set(self, search_type, query, scraped_contents=None, summaries=None):
        nonlocal _search_counter
        _search_counter += 1
        search_id = _search_counter
        entry = CacheEntry(
            scraped_contents=scraped_contents or {},
            summaries=summaries,
            expires_at=float('inf'),
            metadata={"search_id": search_id, "search_type": search_type, "query": query},
            simhash=0
        )
        _cache[query] = entry
        return search_id
    
    def mock_get_by_id(item_id):
        if item_id:
            for entry in _cache.values():
                if isinstance(entry, CacheEntry) and entry.metadata.get("search_id") == item_id:
                    results = []
                    for url, content in entry.scraped_contents.items():
                        if content:
                            results.append(content)
                    if entry.summaries:
                        results.append(entry.summaries)
                    combined_results = "\n\n".join(results) if results else ""
                    source_urls = ", ".join(entry.scraped_contents.keys()) if entry.scraped_contents else None
                    return combined_results, source_urls, entry.metadata
        return None

    def mock_get_cached_urls(urls: list[str], search_type: str = ""):
        cached_urls = {}
        for entry in _cache.values():
            if isinstance(entry, CacheEntry):
                for url in urls:
                    if url in entry.scraped_contents:
                        cached_urls[url] = entry.scraped_contents[url]
        return cached_urls

    monkeypatch.setattr(SearchCache, "get", lambda self, search_type, query: mock_get(search_type, query))
    monkeypatch.setattr(SearchCache, "set", mock_set)
    monkeypatch.setattr(SearchCache, "get_by_id", lambda self, item_id: mock_get_by_id(item_id))
    monkeypatch.setattr(SearchCache, "get_cached_urls", lambda self, urls: mock_get_cached_urls(urls))
    
    return SearchCache(), _cache


@pytest.mark.anyio
async def test_cache_concurrent_writes(memory_cache):
    """
    Verify that SearchCache handles concurrent write attempts gracefully,
    ensuring data integrity and avoiding race conditions.
    """
    memory_cache_instance, internal_cache_dict = memory_cache
    num_concurrent_writes = 10
    write_data = [
        (f"query_{i}", {f"url_{i}": f"result_{i}"}, f"url_{i}", i) for i in range(num_concurrent_writes)
    ]

    async def write_to_cache(query_str, scraped_contents_dict, url, search_id):
        return memory_cache_instance.set(
            "default", # search_type
            query_str,
            scraped_contents=scraped_contents_dict,
            summaries=f"summary for {query_str}",
        )

    tasks = [
        write_to_cache(
            query_str=d[0],
            scraped_contents_dict=d[1],
            url=d[2],
            search_id=d[3]
        ) for d in write_data
    ]

    await asyncio.gather(*tasks)

    assert len(internal_cache_dict) == num_concurrent_writes
    for query, scraped_contents, url, search_id in write_data:
        retrieved_data = memory_cache_instance.get("default", query)
        assert retrieved_data is not None, f"Query '{query}' not found in cache after concurrent writes."
        entry_scraped_contents, entry_url, entry_metadata = retrieved_data
        assert entry_scraped_contents is not None
        assert entry_metadata is not None
        assert scraped_contents[url] in entry_scraped_contents.values()

    write_data_with_error = [
        (f"query_{i + num_concurrent_writes}", {f"url_{i + num_concurrent_writes}": f"result_{i + num_concurrent_writes}"}, f"url_{i + num_concurrent_writes}", i)
        for i in range(num_concurrent_writes)
    ]

    async def write_to_cache_with_error(query_str, scraped_contents_dict, url, search_id):
        if search_id == 5:
            raise ValueError("Simulated error")
        return memory_cache_instance.set(
            "default", # search_type
            query_str,
            scraped_contents=scraped_contents_dict,
            summaries=f"summary for {query_str}",
        )

    tasks_with_error = [
        write_to_cache_with_error(
            query_str=d[0],
            scraped_contents_dict=d[1],
            url=d[2],
            search_id=d[3]
        ) for d in write_data_with_error
    ]

    results = await asyncio.gather(*tasks_with_error, return_exceptions=True)
    
    error_count = sum(1 for r in results if isinstance(r, ValueError))
    assert error_count == 1

    assert len(internal_cache_dict) == num_concurrent_writes + (num_concurrent_writes - error_count)
