"""Expose batch simulation and independent evidence verification to workflow stages."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from npa.clients.storage import StorageClient
from .evidence import verify_evidence


def build_parser() -> argparse.ArgumentParser:
    """Construct the module command parser used by workflow argv validation.

    Args:
        None.
    Returns:
        Parser for simulate and verify commands with local or S3 artifact paths.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser(
        "simulate", help="Run one complete six-carton batch."
    )
    simulate.add_argument("--output-path", required=True)
    simulate.add_argument("--run-id", required=True)
    verify = commands.add_parser(
        "verify", help="Validate saved measurements and decoded PNGs."
    )
    verify.add_argument("--input-path", required=True)
    verify.add_argument("--output-path", required=True)
    return parser


def _publish(directory: Path, destination: str) -> None:
    if destination.startswith("s3://"):
        StorageClient.from_environment().upload_directory(str(directory), destination)
        return
    target = Path(destination)
    if target.exists():
        raise FileExistsError(
            "Output directory already exists; choose a fresh run path"
        )
    shutil.copytree(directory, target)


def _simulate(arguments, directory: Path) -> None:
    from .runtime import simulate

    try:
        application = simulate(directory, arguments.run_id)
        report = verify_evidence(directory)
        (directory / "verification.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    except Exception as error:
        (directory / "failure.json").write_text(
            json.dumps({"error_type": type(error).__name__}) + "\n"
        )
        _publish(directory, arguments.output_path)
        raise
    _publish(directory, arguments.output_path)
    if not report["all_passed"]:
        raise RuntimeError(
            "Warehouse acceptance checks failed; retained verification.json"
        )
    # Kit close can exit the process, so it follows validation and publication.
    application.close()


def _verify(arguments, directory: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="npa-warehouse-input-") as temporary:
        source = Path(temporary)
        if arguments.input_path.startswith("s3://"):
            StorageClient.from_environment().download_directory(
                arguments.input_path, str(source)
            )
        else:
            source = Path(arguments.input_path)
        report = verify_evidence(source)
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    _publish(directory, arguments.output_path)
    if not report["all_passed"]:
        raise RuntimeError("Warehouse evidence failed acceptance checks")


def main(argv: list[str] | None = None) -> None:
    """Execute a stage, preserving failure evidence and nonzero errors.

    Args:
        argv: Command arguments; defaults to the process command line.
    Returns:
        None.
    Raises:
        RuntimeError: Physical or image acceptance checks fail.
        Exception: Runtime, evidence, or storage access fails.
    """
    arguments = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="npa-antioch-warehouse-") as temporary:
        directory = Path(temporary)
        if arguments.command == "simulate":
            _simulate(arguments, directory)
        else:
            _verify(arguments, directory)
