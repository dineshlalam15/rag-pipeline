import time

ONE_YEAR_SECONDS = 365 * 24 * 60 * 60


class InMemoryTTLCache:
    """A local in-memory fallback mechanism when Redis is unavailable or disabled."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        """Return cached string value if present and not expired."""
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        """Store string value with a monotonic expiration timestamp."""
        self._store[key] = (time.monotonic() + ttl_seconds, value)

    def delete(self, key: str) -> None:
        """Remove a key from the in-memory store if present."""
        self._store.pop(key, None)

    def incr(self, key: str) -> int:
        """Increase a stored counter by 1 (defaults to starting at 1 -> 2)."""
        current = self.get(key)
        if current is None:
            current_number = 1
        else:
            current_number = int(current)

        next_number = current_number + 1
        self.setex(key, ONE_YEAR_SECONDS, str(next_number))
        return next_number
