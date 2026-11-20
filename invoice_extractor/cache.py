"""Disk caching for raw Gemini responses and token usage."""

import json

from invoice_extractor.config import CACHE_DIR

_cache_enabled: bool = True


def is_enabled() -> bool:
    """Check if disk caching is currently enabled."""
    return _cache_enabled


def enable() -> None:
    """Enable disk caching."""
    global _cache_enabled
    _cache_enabled = True


def disable() -> None:
    """Disable disk caching."""
    global _cache_enabled
    _cache_enabled = False


def get(key: str) -> dict | None:
    """Retrieve cached response dict by key if caching is enabled."""
    if not _cache_enabled:
        return None
    cache_path = CACHE_DIR / f"{key}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def put(key: str, value: dict) -> None:
    """Persist raw response dict to disk under key."""
    if not _cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{key}.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
    except OSError:
        pass
