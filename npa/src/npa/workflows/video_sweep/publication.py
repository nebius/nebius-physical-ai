"""Publish only reviewed, tracked video bytes and an explicit next-run inventory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from npa.workflows.video_sweep.artifacts import (
    digest,
    download,
    read_json,
    upload,
    write_json,
)
from npa.workflows.video_sweep.execution import reviewed


def publish(args) -> None:
    """Copy accepted clips after verifying exact review and lineage coverage.

    Args:
        args: Parsed stage arguments.
    Returns:
        None.
    Raises:
        ValueError: No clips pass, lineage is incomplete, or media changed.
    """
    plan, report = reviewed(args)
    receipt = read_json(args.root_uri + "/lineage.json")
    _verify_lineage(receipt, report)
    accepted = [row for row in report["items"] if row["accepted"]]
    if not accepted:
        raise ValueError(
            "No candidate passed; review is retained, dataset is not published"
        )
    dataset = _copy_accepted(args.root_uri, accepted)
    write_json(
        args.root_uri + "/dataset/manifest.json",
        {
            "schema": "npa.video_sweep.dataset.v1",
            "run_id": plan["run_id"],
            "review_sha256": digest(report),
            "lineage_sha256": digest(receipt),
            "clips": dataset,
        },
    )
    write_json(
        args.root_uri + "/dataset/next-sources.json",
        {
            "schema": "npa.video_sweep.sources.v1",
            "parent_run_id": plan["run_id"],
            "clips": [row["uri"] for row in dataset],
        },
    )


def _verify_lineage(receipt: dict, report: dict) -> None:
    expected = {row["id"] for row in report["items"]}
    tracked = receipt.get("items", [])
    if (
        receipt.get("review_sha256") != digest(report)
        or len(tracked) != len(expected)
        or {row["id"] for row in tracked} != expected
    ):
        raise ValueError("Lineage does not cover the complete review")


def _copy_accepted(root: str, accepted: list[dict]) -> list[dict]:
    dataset = []
    with tempfile.TemporaryDirectory() as directory:
        for row in accepted:
            path = Path(directory) / "clip.mp4"
            download(row["uri"], path, row["sha256"])
            target = root + "/dataset/clips/" + row["id"] + ".mp4"
            upload(path, target)
            dataset.append({"id": row["id"], "uri": target, "sha256": row["sha256"]})
    return dataset
