"""Seal stage artifacts and carry allowlisted lineage between policy workers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from npa.workflows.lerobot_transfer_data import materialize, publish
from .contracts import digest
from .public_vla_data import write_json

__all__ = ["materialize", "publish", "inherit", "record", "read"]


def read(root: Path, name: str) -> dict:
    """Read one finite JSON evidence document.

    Args:
        root: Verified stage directory.
        name: Relative document name.
    Returns:
        JSON evidence.
    Raises:
        ValueError: JSON is nonfinite or not an object.
        OSError: The document is unavailable.
    """
    value = json.loads((root / name).read_text(), parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("stage evidence must be an object")
    return value


def _invalid_constant(value):
    raise ValueError("nonfinite stage evidence")


def inherit(source: Path, target: Path) -> None:
    """Carry recipe, corpus, quality measurements and measured stage history.

    Args:
        source: Verified parent stage.
        target: New output directory.
    Returns:
        None.
    Raises:
        OSError: Evidence cannot be copied.
    """
    for name in ("recipe.json", "corpus.json", "quality.json", "split.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    for name in ("history", "previews"):
        if (source / name).is_dir():
            shutil.copytree(source / name, target / name, dirs_exist_ok=True)


def record(output: Path, stage: str, measurements: dict) -> None:
    """Bind measurements to the immutable recipe and corpus selection.

    Args:
        output: Stage directory containing inherited provenance.
        stage: Public stage or attempt name.
        measurements: Component-produced values.
    Returns:
        None.
    Raises:
        ValueError: Provenance is incomplete or nonfinite.
        OSError: Evidence cannot be written.
    """
    value = measurements | {
        "schema": "npa.policy-public.stage.v1",
        "stage": stage,
        "recipe_sha256": digest(read(output, "recipe.json")),
    }
    if (output / "corpus.json").exists():
        value["corpus_sha256"] = digest(read(output, "corpus.json"))
    write_json(output / "stage.json", value)
    write_json(output / "history" / f"{stage}.json", value)


def require_identity(parent: Path, recipe: dict, corpus: dict) -> None:
    """Reject a checkpoint from a different recipe or curated data selection.

    Args:
        parent: Verified candidate or gate stage.
        recipe: Current immutable recipe.
        corpus: Current curated split.
    Returns:
        None.
    Raises:
        ValueError: Lineage differs.
    """
    stage = read(parent, "stage.json")
    if stage["recipe_sha256"] != digest(recipe) or stage["corpus_sha256"] != digest(
        corpus
    ):
        raise ValueError("candidate recipe or curated corpus identity changed")
