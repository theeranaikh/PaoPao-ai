"""Conservative cleaning and validation for Python training examples."""

from __future__ import annotations

import ast
import hashlib
import re


def clean_code(code: str, max_characters: int = 200_000) -> str | None:
    """Normalize line endings and reject empty, binary, or extremely large samples."""
    if not isinstance(code, str) or "\x00" in code:
        return None
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    if len(code) < 8 or len(code) > max_characters:
        return None
    return code


def is_likely_python(code: str) -> bool:
    """Check syntax when possible while retaining intentionally buggy repair inputs."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return bool(re.search(r"\b(def|class|import|for|if|return)\b", code))


def content_hash(text: str) -> str:
    """Stable hash used for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

