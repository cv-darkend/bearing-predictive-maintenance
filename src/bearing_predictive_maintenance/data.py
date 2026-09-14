"""Download the project dataset through the official Kaggle CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

DATASET_SLUG = "vinayak123tyagi/bearing-dataset"
DEFAULT_DESTINATION = Path("data/raw")


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded safely."""


def build_download_command(executable: str, destination: Path, *, force: bool = False) -> list[str]:
    """Build the Kaggle CLI argument vector without shell interpolation."""
    command = [
        executable,
        "datasets",
        "download",
        DATASET_SLUG,
        "--path",
        str(destination),
        "--unzip",
    ]
    if force:
        command.append("--force")
    return command


def download_dataset(destination: Path = DEFAULT_DESTINATION, *, force: bool = False) -> Path:
    """Download and extract the configured dataset into ``destination``."""
    executable = shutil.which("kaggle")
    if executable is None:
        raise DatasetDownloadError(
            "Kaggle CLI is unavailable. Run `uv sync` and invoke this command with "
            "`uv run bearing-download`."
        )

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = build_download_command(executable, destination, force=force)

    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as error:
        raise DatasetDownloadError(
            "Kaggle could not download the dataset. Verify authentication and dataset access, "
            "then retry."
        ) from error

    return destination


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Download and extract Kaggle dataset {DATASET_SLUG}."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Extraction directory (default: {DEFAULT_DESTINATION}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh Kaggle download when files already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset download command."""
    args = build_parser().parse_args(argv)
    try:
        destination = download_dataset(args.destination, force=args.force)
    except DatasetDownloadError as error:
        raise SystemExit(str(error)) from error
    print(f"Dataset available at {destination}")
