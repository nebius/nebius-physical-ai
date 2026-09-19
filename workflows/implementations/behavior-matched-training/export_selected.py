"""Export only the predeclared holdout-selected EMA checkpoint with byte identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_selected(
    checkpoint_root: Path, selection_path: Path, output_root: Path
) -> dict:
    """Copy the selected inference params and assets into a sealed export tree."""
    selection = json.loads(selection_path.read_text())
    if selection.get("schema") != "npa.behavior.rlc-holdout-selection.v1":
        raise ValueError("selection receipt schema differs")
    step = int(selection["selected_step"])
    source = checkpoint_root / str(step)
    if not (source / "params").is_dir() or not (source / "assets").is_dir():
        raise FileNotFoundError(f"selected checkpoint {step} is incomplete")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    shutil.copytree(source / "params", output_root / "params")
    shutil.copytree(source / "assets", output_root / "assets")
    files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(output_root))] = {
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
    receipt = {
        "schema": "npa.behavior.rlc-selected-export.v1",
        "selected_step": step,
        "selection_receipt_sha256": _digest(selection_path),
        "files": files,
        "status": "holdout_selected_not_rollout_evaluated",
    }
    (output_root / "export-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    """Parse arguments and export the selected checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export_selected(args.checkpoint_root, args.selection, args.output_root)


if __name__ == "__main__":
    main()
