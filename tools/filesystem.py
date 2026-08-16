"""Workspace-constrained filesystem primitives and reviewable patches."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilePatch:
    """A proposed full-file replacement and its unified diff."""

    path: Path
    before: str
    after: str
    diff: str


class WorkspaceFilesystem:
    """Read and write text files only beneath a configured project root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a user path and reject directory traversal outside the root."""
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path is outside workspace: {path}") from exc
        return candidate

    def read_file(self, path: str | Path, max_characters: int = 200_000) -> str:
        """Read a UTF-8 file with a context-size bound."""
        target = self.resolve(path)
        if target.stat().st_size > max_characters:
            raise ValueError(f"file exceeds {max_characters} character context limit: {path}")
        return target.read_text(encoding="utf-8")

    def list_files(self, path: str | Path = ".", max_files: int = 250) -> list[str]:
        """List non-hidden workspace files relative to root."""
        directory = self.resolve(path)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        files: list[str] = []
        ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
        for item in sorted(directory.rglob("*")):
            if any(part in ignored for part in item.parts) or not item.is_file():
                continue
            files.append(str(item.relative_to(self.root)))
            if len(files) >= max_files:
                break
        return files

    def propose_write(self, path: str | Path, content: str) -> FilePatch:
        """Construct a unified diff without modifying the filesystem."""
        target = self.resolve(path)
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{target.relative_to(self.root)}",
                tofile=f"b/{target.relative_to(self.root)}",
            )
        )
        return FilePatch(target, before, content, diff)

    def apply_patch(self, patch: FilePatch) -> None:
        """Apply exactly a previously reviewed patch, failing on concurrent changes."""
        current = patch.path.read_text(encoding="utf-8") if patch.path.exists() else ""
        if current != patch.before:
            raise RuntimeError("file changed after proposal; generate and review a new diff")
        patch.path.parent.mkdir(parents=True, exist_ok=True)
        patch.path.write_text(patch.after, encoding="utf-8")

    def rollback(self, patch: FilePatch) -> None:
        """Restore the file content that existed before a successfully applied patch."""
        current = patch.path.read_text(encoding="utf-8") if patch.path.exists() else ""
        if current != patch.after:
            raise RuntimeError("file changed after application; refusing unsafe rollback")
        if patch.before:
            patch.path.write_text(patch.before, encoding="utf-8")
        else:
            patch.path.unlink(missing_ok=True)

