"""A narrow, auditable tool controller for PaoPao agent actions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.filesystem import WorkspaceFilesystem
from tools.python_runner import run_python
from tools.terminal import SandboxConfig, execute_command
from tools.test_runner import run_tests


class ToolController:
    """Dispatch allowlisted tool calls and require approval before writes."""

    TOOL_SCHEMAS = (
        {"name": "read_file", "arguments": {"path": "workspace-relative text path"}},
        {"name": "list_files", "arguments": {"path": "optional workspace-relative directory"}},
        {"name": "execute_command", "arguments": {"command": "single shell-free command"}},
        {"name": "run_python", "arguments": {"file": "workspace-relative Python file"}},
        {"name": "run_tests", "arguments": {}},
        {"name": "write_file", "arguments": {"path": "text path", "content": "complete replacement"}},
        {"name": "edit_file", "arguments": {"path": "text path", "changes": "[{old, new}] or content"}},
    )

    def __init__(self, root: str | Path, sandbox: SandboxConfig = SandboxConfig()) -> None:
        self.filesystem = WorkspaceFilesystem(root)
        self.root = self.filesystem.root
        self.sandbox = sandbox

    def available_tools(self) -> tuple[dict[str, object], ...]:
        """Return the narrow tool contract a tool-calling model may choose from."""
        return self.TOOL_SCHEMAS

    def execute(self, name: str, arguments: dict[str, Any], approved: bool = False) -> dict[str, Any]:
        """Execute a named tool; write operations return a diff until approved."""
        if name == "read_file":
            return {"content": self.filesystem.read_file(arguments["path"])}
        if name == "list_files":
            return {"files": self.filesystem.list_files(arguments.get("path", "."))}
        if name == "execute_command":
            result = execute_command(arguments["command"], self.root, self.root, self.sandbox)
            return asdict(result)
        if name == "run_python":
            result = run_python(arguments["file"], self.root, self.root, self.sandbox)
            return asdict(result)
        if name == "run_tests":
            result = run_tests(self.root, self.sandbox)
            return asdict(result)
        if name in {"write_file", "edit_file"}:
            path = arguments["path"]
            content = arguments.get("content")
            if name == "edit_file" and content is None:
                changes = arguments.get("changes")
                if not isinstance(changes, list):
                    raise ValueError("edit_file requires `content` or a list of {old, new} changes")
                content = self.filesystem.read_file(path)
                for change in changes:
                    if not isinstance(change, dict) or not isinstance(change.get("old"), str) or not isinstance(change.get("new"), str):
                        raise ValueError("each edit change must have string `old` and `new` fields")
                    old, new = change["old"], change["new"]
                    if content.count(old) != 1:
                        raise ValueError("each edit `old` value must match exactly one location")
                    content = content.replace(old, new, 1)
            if content is None:
                raise ValueError("write_file/edit_file requires complete replacement `content`")
            if not isinstance(content, str):
                raise ValueError("file content must be text")
            patch = self.filesystem.propose_write(path, content)
            response: dict[str, Any] = {"diff": patch.diff, "requires_confirmation": True}
            if approved:
                self.filesystem.apply_patch(patch)
                response["applied"] = True
            return response
        raise ValueError(f"unknown or disallowed tool: {name}")
