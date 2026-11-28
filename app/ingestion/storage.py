"""Local filesystem storage for uploaded documents."""

from pathlib import Path


class LocalDocumentStorage:
    """Stores and retrieves uploaded documents on the local filesystem."""

    def __init__(self, base_dir: str | Path = "./data/uploads") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_resolve(self, target: str | Path) -> Path:
        candidate = Path(target)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.base_dir / candidate).resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise ValueError(f"Path '{target}' escapes storage root '{self.base_dir}'.")
        return resolved

    def save_document(self, relative_path: str, content: bytes) -> Path:
        """Persist raw document bytes under base_dir and return the resolved file path."""
        destination = self._safe_resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    def read_document(self, file_path: str | Path) -> bytes:
        """Read raw bytes from a stored local document."""
        target = self._safe_resolve(file_path)
        return target.read_bytes()

    def delete_document(self, file_path: str | Path) -> None:
        """Delete a stored local document if it exists."""
        target = self._safe_resolve(file_path)
        if target.exists():
            target.unlink()
