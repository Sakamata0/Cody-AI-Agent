"""
Response caching for Bedrock API calls.

Caches LLM responses for identical queries to avoid redundant API calls
and reduce token costs. Uses a simple in-memory LRU cache.
"""

import hashlib
from functools import lru_cache
from collections import OrderedDict
from threading import Lock


class ResponseCache:
    """
    LRU cache for agent responses.

    Caches responses based on the query + chat history hash.
    Avoids repeated Bedrock calls for identical questions within the same session.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, history_length: int) -> str:
        """Create a cache key from query and history context."""
        raw = f"{query}|{history_length}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, history_length: int) -> dict | None:
        """Look up a cached response. Returns None on miss."""
        key = self._make_key(query, history_length)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used).
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, query: str, history_length: int, response: dict) -> None:
        """Store a response in the cache."""
        key = self._make_key(query, history_length)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove oldest.
            self.cache[key] = response

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_entries": len(self.cache),
        }

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


# Global cache instance.
response_cache = ResponseCache(max_size=50)
