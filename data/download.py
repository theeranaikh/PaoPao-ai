"""Explicit dataset downloading utilities; downloads are never implicit in training."""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)
CODESEARCHNET_PYTHON_URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip"


def download_file(url: str, output: str | Path, timeout: int = 60) -> Path:
    """Download a user-selected URL to a local location with a clear failure mode."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "PaoPao dataset downloader/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, output.open("wb") as destination:
            shutil.copyfileobj(response, destination)
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url}: {exc}") from exc
    LOGGER.info("Downloaded %s to %s", url, output)
    return output


def extract_zip(archive: str | Path, output_dir: str | Path) -> Path:
    """Extract a ZIP archive while rejecting path traversal entries."""
    archive, output = Path(archive), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (output / member.filename).resolve()
            try:
                destination.relative_to(output.resolve())
            except ValueError as exc:
                raise RuntimeError(f"unsafe archive member: {member.filename}") from exc
        bundle.extractall(output)
    LOGGER.info("Extracted %s to %s", archive, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a source dataset explicitly.")
    parser.add_argument("--url", default=CODESEARCHNET_PYTHON_URL)
    parser.add_argument("--output", default="data/raw/codesearchnet-python.zip")
    parser.add_argument("--extract-dir", help="Optionally extract the downloaded ZIP into this directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    archive = download_file(args.url, args.output)
    if args.extract_dir:
        extract_zip(archive, args.extract_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
