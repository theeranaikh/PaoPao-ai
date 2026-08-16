"""Python-file execution through the PaoPao command sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

from .terminal import CommandResult, SandboxConfig, execute_command


def run_python(
    file_path: str | Path, cwd: str | Path = ".", workspace_root: str | Path | None = None,
    config: SandboxConfig = SandboxConfig(),
) -> CommandResult:
    """Run one project Python file under the configured execution limits."""
    root = Path(workspace_root or cwd).resolve()
    requested = Path(file_path)
    target = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Python file is outside workspace: {file_path}") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return execute_command([sys.executable, str(target)], root, root, config)
