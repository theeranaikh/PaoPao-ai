from __future__ import annotations

import sys

import pytest

from agent import PaoPaoAgent
from tools.filesystem import WorkspaceFilesystem
from tools.terminal import SandboxConfig, execute_command


def test_terminal_executes_safe_command(tmp_path):
    result = execute_command(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        workspace_root=tmp_path,
        config=SandboxConfig(timeout_seconds=5, cpu_seconds=3, memory_mb=512),
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_terminal_blocks_destructive_command(tmp_path):
    result = execute_command(["rm", "-rf", "target"], cwd=tmp_path, workspace_root=tmp_path)
    assert result.blocked
    assert result.returncode == 126


def test_python_runner_rejects_files_outside_workspace(tmp_path):
    from tools.python_runner import run_python

    with pytest.raises(PermissionError):
        run_python("/etc/passwd", cwd=tmp_path, workspace_root=tmp_path)


def test_filesystem_patch_requires_explicit_apply_and_rolls_back(tmp_path):
    source = tmp_path / "main.py"
    source.write_text("answer = 1\n", encoding="utf-8")
    filesystem = WorkspaceFilesystem(tmp_path)
    patch = filesystem.propose_write("main.py", "answer = 2\n")
    assert "-answer = 1" in patch.diff
    assert source.read_text(encoding="utf-8") == "answer = 1\n"
    filesystem.apply_patch(patch)
    assert source.read_text(encoding="utf-8") == "answer = 2\n"
    filesystem.rollback(patch)
    assert source.read_text(encoding="utf-8") == "answer = 1\n"


def test_agent_without_model_only_diagnoses_and_never_writes(tmp_path):
    source = tmp_path / "broken.py"
    original = "def broken(:\n    pass\n"
    source.write_text(original, encoding="utf-8")
    agent = PaoPaoAgent(tmp_path)
    proposal = agent.propose_fix("broken.py")
    assert proposal.patch is None
    assert "SyntaxError" in proposal.diagnosis
    assert source.read_text(encoding="utf-8") == original
