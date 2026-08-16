"""pytest/unittest discovery through the constrained terminal runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .terminal import CommandResult, SandboxConfig, execute_command


def run_tests(
    project_path: str | Path = ".", config: SandboxConfig = SandboxConfig(timeout_seconds=120, cpu_seconds=90),
) -> CommandResult:
    """Run pytest when present, otherwise Python's standard unittest discovery."""
    root = Path(project_path).resolve()
    if importlib.util.find_spec("pytest"):
        command = [sys.executable, "-m", "pytest", "-q"]
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-v"]
    return execute_command(command, cwd=root, workspace_root=root, config=config)
