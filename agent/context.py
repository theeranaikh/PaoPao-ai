"""Targeted project-context selection for coding and repair tasks."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from tools.filesystem import WorkspaceFilesystem


@dataclass
class ContextFile:
    """A selected file plus bounded source content."""

    path: str
    content: str


class ProjectContext:
    """Find related local source files without loading an entire repository."""

    def __init__(self, filesystem: WorkspaceFilesystem, max_files: int = 8, max_characters: int = 80_000) -> None:
        self.filesystem = filesystem
        self.max_files = max_files
        self.max_characters = max_characters

    def _imports(self, source: str) -> set[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def select(self, target: str, error_text: str = "") -> list[ContextFile]:
        """Select the target, imported sibling modules, matching tests, then error matches."""
        target_path = self.filesystem.resolve(target)
        target_relative = str(target_path.relative_to(self.filesystem.root))
        target_source = self.filesystem.read_file(target_relative)
        imports = self._imports(target_source)
        keywords = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", error_text))
        selected = [target_relative]
        candidates = self.filesystem.list_files()
        target_stem = target_path.stem
        ranked: list[tuple[int, str]] = []
        for candidate in candidates:
            if candidate == target_relative or not candidate.endswith(".py"):
                continue
            path = Path(candidate)
            score = 0
            if path.stem in imports:
                score += 10
            if "test" in path.parts or path.name.startswith("test_"):
                score += 4
            if target_stem in path.stem:
                score += 4
            if any(keyword in path.stem for keyword in keywords):
                score += 2
            if score:
                ranked.append((score, candidate))
        selected.extend(candidate for _, candidate in sorted(ranked, key=lambda item: (-item[0], item[1])))
        context: list[ContextFile] = []
        used = 0
        for candidate in selected[: self.max_files]:
            try:
                content = self.filesystem.read_file(candidate, max_characters=self.max_characters - used)
            except (OSError, ValueError):
                continue
            used += len(content)
            context.append(ContextFile(candidate, content))
            if used >= self.max_characters:
                break
        return context

