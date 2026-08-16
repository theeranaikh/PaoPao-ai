"""Human-facing CLI for generation, analysis, testing, and review-first fixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from agent import PaoPaoAgent
from tools.python_runner import run_python
from tools.test_runner import run_tests


def _generator_from_args(args: argparse.Namespace) -> Callable[[str], str] | None:
    if not args.checkpoint:
        return None
    from inference.generate import generate, load_model

    loaded = load_model(args.checkpoint, args.tokenizer, args.device)

    def call(prompt: str) -> str:
        return generate(
            loaded, prompt, max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p
        )

    return call


def _print_result(stdout: str, stderr: str) -> None:
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the package script and smoke tests."""
    parser = argparse.ArgumentParser(prog="paopao", description="PaoPao from-scratch code model and local coding agent")
    parser.add_argument("--checkpoint", help="Local trained PaoPao checkpoint for chat/generate/fix")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default=".", help="Workspace root for terminal-agent commands")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    commands = parser.add_subparsers(dest="command")
    generate_parser = commands.add_parser("generate", help="Generate a response from a checkpoint")
    generate_parser.add_argument("prompt")
    chat_parser = commands.add_parser("chat", help="Interactive local checkpoint conversation")
    chat_parser.add_argument("prompt", nargs="?")
    run_parser = commands.add_parser("run", help="Run a Python file in the command sandbox")
    run_parser.add_argument("file")
    explain_parser = commands.add_parser("explain", help="Explain a Python file")
    explain_parser.add_argument("file")
    fix_parser = commands.add_parser("fix", help="Propose a reviewable repair for a Python file")
    fix_parser.add_argument("file")
    fix_parser.add_argument("--apply", action="store_true", help="Confirm and apply the displayed repair session")
    fix_parser.add_argument("--retries", type=int, default=2)
    test_parser = commands.add_parser("test", help="Run the project pytest or unittest suite")
    test_parser.add_argument("path", nargs="?", default=".")
    return parser


def _require_generator(generator: Callable[[str], str] | None) -> Callable[[str], str]:
    if generator is None:
        raise SystemExit("This command requires --checkpoint and a matching --tokenizer.")
    return generator


def _chat(generator: Callable[[str], str], initial_prompt: str | None) -> int:
    if initial_prompt:
        print(generator(initial_prompt))
        return 0
    if not sys.stdin.isatty():
        raise SystemExit("`paopao chat` needs a prompt on non-interactive input.")
    while True:
        try:
            prompt = input("paopao> ").strip()
        except EOFError:
            print()
            return 0
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt:
            print(generator(prompt))


def _normalize_direct_prompt(arguments: list[str]) -> list[str]:
    """Insert ``generate`` for a bare prompt while preserving global option order."""
    commands = {"chat", "generate", "run", "fix", "explain", "test", "-h", "--help"}
    if any(argument in commands for argument in arguments):
        return arguments
    options_with_values = {
        "--checkpoint", "--tokenizer", "--device", "--root", "--max-tokens", "--temperature", "--top-k", "--top-p"
    }
    index = 0
    while index < len(arguments):
        if arguments[index] in options_with_values:
            index += 2
            continue
        if arguments[index].startswith("-"):
            return arguments
        return [*arguments[:index], "generate", *arguments[index:]]
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    # Support both `paopao "..."` and `paopao --checkpoint path "..."`.
    raw_args = _normalize_direct_prompt(raw_args)
    args = parser.parse_args(raw_args)
    if not args.command:
        parser.print_help()
        return 2
    root = Path(args.root).resolve()
    if args.command in {"generate", "chat"}:
        generator = _require_generator(_generator_from_args(args))
        return _chat(generator, getattr(args, "prompt", None)) if args.command == "chat" else _chat(generator, args.prompt)
    if args.command == "run":
        result = run_python(args.file, root, root)
        _print_result(result.stdout, result.stderr)
        return result.returncode
    if args.command == "test":
        result = run_tests(Path(args.path).resolve())
        _print_result(result.stdout, result.stderr)
        return result.returncode
    generator = _generator_from_args(args)
    agent = PaoPaoAgent(root, generator=generator, retry_limit=getattr(args, "retries", 2))
    if args.command == "explain":
        print(agent.explain(args.file))
        return 0
    if args.command == "fix":
        proposal = agent.propose_fix(args.file)
        if proposal.diagnosis:
            print("Diagnosis:\n" + proposal.diagnosis)
        if not proposal.patch:
            print("\nNo patch proposed. Supply --checkpoint to use a trained local PaoPao model.")
            return 1
        print("\nProposed diff:\n" + proposal.diff)
        apply = args.apply
        if not apply and sys.stdin.isatty():
            apply = input("Apply this repair session? [y/N] ").strip().lower() in {"y", "yes"}
        if not apply:
            print("Patch not applied.")
            return 0
        result = agent.fix_after_approval(args.file, proposal)
        print(result.message)
        if result.last_result:
            _print_result(result.last_result.stdout, result.last_result.stderr)
        return 0 if result.passed else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
