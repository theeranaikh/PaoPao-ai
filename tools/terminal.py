"""Constrained command execution for a local coding-agent workspace."""

from __future__ import annotations

import os
import re
import resource
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SandboxConfig:
    """Limits applied to each command execution."""

    timeout_seconds: int = 30
    cpu_seconds: int = 20
    memory_mb: int = 2_048
    max_output_characters: int = 100_000
    allow_network: bool = False


@dataclass
class CommandResult:
    """Captured command outcome suitable for agent context."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    blocked: bool = False
    reason: str | None = None


class CommandSafetyError(ValueError):
    """Raised when a command violates the terminal safety policy."""


_BLOCKED_PATTERNS = (
    r"(^|/)rm$",
    r"\b(shutdown|reboot|halt|poweroff|mkfs|fdisk|parted|mount|umount|chown|sudo|su)\b",
    r"\bdd\b",
    r"\b(curl|wget|nc|ncat|ssh|scp|rsync|git)\b",
    r"(^|/)pip(?:3)?(?:\s|$)",
    r"\bapt(?:-get)?\b",
)
_NETWORK_PATTERNS = (r"\b(curl|wget|nc|ncat|ssh|scp|rsync|git)\b",)
_SHELL_CONTROL = {"|", "||", "&&", ";", ">", ">>", "<", "<<", "`"}


def _workspace(path: str | Path) -> Path:
    return Path(path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_command(command: str | Sequence[str], allow_network: bool = False) -> list[str]:
    """Parse a command without a shell and reject destructive/network-capable programs."""
    arguments = shlex.split(command) if isinstance(command, str) else list(command)
    if not arguments:
        raise CommandSafetyError("command is empty")
    joined = " ".join(arguments)
    if any(token in _SHELL_CONTROL for token in arguments) or "$(" in joined or "`" in joined:
        raise CommandSafetyError("shell composition is not allowed; pass one executable and its arguments")
    executable = Path(arguments[0]).name
    if executable == "rm":
        raise CommandSafetyError("blocked by command safety policy: rm")
    candidate = f"{arguments[0]} {joined}"
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, candidate):
            if allow_network and pattern in _NETWORK_PATTERNS:
                continue
            raise CommandSafetyError(f"blocked by command safety policy: {executable}")
    return arguments


def _limit_resources(config: SandboxConfig) -> None:
    """Child-only resource limits for POSIX systems."""
    resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_seconds, config.cpu_seconds + 1))
    memory = config.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


def execute_command(
    command: str | Sequence[str],
    cwd: str | Path = ".",
    workspace_root: str | Path | None = None,
    config: SandboxConfig = SandboxConfig(),
) -> CommandResult:
    """Execute one safe local command with time, CPU, RAM, and output limits.

    A shell is never invoked. Network utilities are blocked by default; this is a
    command-policy sandbox, not a replacement for an OS/container network namespace.
    Run PaoPao in a container/VM for adversarial untrusted source code.
    """
    root = _workspace(workspace_root or cwd)
    working_directory = _workspace(cwd)
    if not _is_relative_to(working_directory, root):
        raise CommandSafetyError("working directory is outside the workspace root")
    try:
        arguments = validate_command(command, config.allow_network)
    except CommandSafetyError as exc:
        return CommandResult([], 126, "", str(exc), blocked=True, reason=str(exc))
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "*",
        "http_proxy": "",
        "https_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
    }
    try:
        completed = subprocess.run(
            arguments,
            cwd=working_directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.timeout_seconds,
            check=False,
            preexec_fn=lambda: _limit_resources(config) if os.name == "posix" else None,
        )
        stdout = completed.stdout[: config.max_output_characters]
        stderr = completed.stderr[: config.max_output_characters]
        return CommandResult(arguments, completed.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "")
        stderr = (exc.stderr or "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CommandResult(arguments, 124, stdout[: config.max_output_characters], stderr[: config.max_output_characters], timed_out=True, reason="timeout")
    except OSError as exc:
        return CommandResult(arguments, 127, "", str(exc), reason="execution failure")
