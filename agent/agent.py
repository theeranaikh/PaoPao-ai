"""Review-first automatic bug-fixing workflow for trained PaoPao models."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tools.filesystem import FilePatch
from tools.python_runner import run_python
from tools.terminal import CommandResult, SandboxConfig
from tools.test_runner import run_tests

from .context import ProjectContext
from .controller import ToolController

Generator = Callable[[str], str]


@dataclass
class RepairProposal:
    """A reviewable repair proposal, generated before any write is performed."""

    patch: FilePatch | None
    diagnosis: str
    execution: CommandResult | None
    model_used: bool

    @property
    def diff(self) -> str:
        return self.patch.diff if self.patch else ""


@dataclass
class RepairResult:
    """Final outcome after applying approved repair attempts."""

    applied: bool
    passed: bool
    attempts: int
    message: str
    last_result: CommandResult | None


class PaoPaoAgent:
    """Coordinates context, a local generation function, testing, and safe file updates."""

    def __init__(
        self,
        root: str | Path = ".",
        generator: Generator | None = None,
        sandbox: SandboxConfig = SandboxConfig(),
        retry_limit: int = 2,
    ) -> None:
        if retry_limit < 0:
            raise ValueError("retry_limit cannot be negative")
        self.controller = ToolController(root, sandbox)
        self.context = ProjectContext(self.controller.filesystem)
        self.generator = generator
        self.sandbox = sandbox
        self.retry_limit = retry_limit

    def explain(self, path: str) -> str:
        """Return a local static summary, augmented by the model when one is configured."""
        source = self.controller.filesystem.read_file(path)
        try:
            tree = ast.parse(source)
            functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
            summary = f"{path}: {len(source.splitlines())} lines; functions={functions or 'none'}; classes={classes or 'none'}."
        except SyntaxError as exc:
            summary = f"{path}: cannot parse Python syntax: {exc.msg} at line {exc.lineno}."
        if self.generator:
            prompt = f"<|task:explain|>\n<|code|>\n{source}\n"
            return self.generator(prompt).strip() or summary
        return summary

    @staticmethod
    def _extract_python(response: str) -> str | None:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() + "\n" if match else None

    def _diagnose(self, path: str) -> tuple[str, CommandResult | None]:
        source = self.controller.filesystem.read_file(path)
        try:
            compile(source, path, "exec")
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}", None
        result = run_python(path, self.controller.root, self.controller.root, self.sandbox)
        if result.returncode == 0:
            return "The file executes successfully. Run the test suite for behavioral failures.", result
        details = (result.stderr or result.stdout).strip()
        return details[-12_000:] or f"Process exited with status {result.returncode}", result

    def propose_fix(self, path: str) -> RepairProposal:
        """Inspect and create a diff proposal. This method never modifies a file."""
        diagnosis, execution = self._diagnose(path)
        if not self.generator:
            return RepairProposal(None, diagnosis, execution, False)
        context = self.context.select(path, diagnosis)
        rendered_context = "\n\n".join(f"# FILE: {item.path}\n{item.content}" for item in context)
        prompt = (
            "<|task:bugfix|>\nRepair the target file. Return ONLY its complete corrected contents in a "
            "```python fenced block. Preserve unrelated behavior.\n"
            f"TARGET: {path}\n<|error|>\n{diagnosis}\n<|code|>\n{rendered_context}\n"
        )
        candidate = self._extract_python(self.generator(prompt))
        if not candidate:
            return RepairProposal(None, "Model response did not contain a complete Python fenced block. " + diagnosis, execution, True)
        patch = self.controller.filesystem.propose_write(path, candidate)
        if not patch.diff:
            return RepairProposal(None, "Model proposed no changes. " + diagnosis, execution, True)
        return RepairProposal(patch, diagnosis, execution, True)

    def apply_and_verify(self, proposal: RepairProposal, run_test_suite: bool = True) -> RepairResult:
        """Apply an explicitly reviewed proposal, run it, then roll back failed attempts."""
        if proposal.patch is None:
            return RepairResult(False, False, 0, "No applicable patch was proposed.", proposal.execution)
        self.controller.filesystem.apply_patch(proposal.patch)
        target = str(proposal.patch.path.relative_to(self.controller.root))
        result = run_python(target, self.controller.root, self.controller.root, self.sandbox)
        if result.returncode == 0 and run_test_suite:
            result = run_tests(self.controller.root, self.sandbox)
        if result.returncode == 0:
            return RepairResult(True, True, 1, "Patch applied and verification passed.", result)
        self.controller.filesystem.rollback(proposal.patch)
        return RepairResult(True, False, 1, "Verification failed; patch was rolled back.", result)

    def fix_after_approval(self, path: str, initial: RepairProposal, run_test_suite: bool = True) -> RepairResult:
        """Apply a reviewed proposal and retry regenerated repairs up to ``retry_limit``.

        Calling this method is the caller's confirmation for every attempt in this
        bounded repair session. Failed patches are rolled back before another model
        proposal is generated from the new execution error.
        """
        proposal = initial
        last_result: CommandResult | None = proposal.execution
        for attempt in range(self.retry_limit + 1):
            result = self.apply_and_verify(proposal, run_test_suite)
            last_result = result.last_result
            if result.passed:
                result.attempts = attempt + 1
                return result
            if attempt == self.retry_limit or not self.generator:
                return RepairResult(
                    result.applied,
                    False,
                    attempt + 1,
                    f"Repair did not verify after {attempt + 1} attempt(s); failed changes were rolled back.",
                    last_result,
                )
            proposal = self.propose_fix(path)
            if proposal.patch is None:
                return RepairResult(
                    False,
                    False,
                    attempt + 1,
                    "A retry proposal could not be generated; no unreviewed changes were applied.",
                    last_result,
                )
        return RepairResult(False, False, self.retry_limit + 1, "Retry limit reached.", last_result)
