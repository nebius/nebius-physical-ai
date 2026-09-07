"""Independently decode a downloaded Ray Train result and its factual Rerun data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def inspect(directory: Path) -> dict:
    """Verify all exported bytes and every decoded optimizer metric against its journal."""
    from rerun.recording import load_recording
    from artifacts import EXPORT_FILES, file_sha256
    from train import validate_journal, validate_recipe

    expected_files = EXPORT_FILES - {"SHA256SUMS"}
    if {path.name for path in directory.iterdir()} != EXPORT_FILES:
        raise ValueError("Export contains unexpected or missing files")
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("Only regular exported artifact files may be inspected")
    lines = (directory / "SHA256SUMS").read_text().splitlines()
    entries = [line.split("  ", 1) for line in lines]
    if len(entries) != len(expected_files) or {name for _, name in entries} != expected_files:
        raise ValueError("Checksum manifest does not cover exactly the exported artifacts")
    hashes = {}
    for expected, name in entries:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Exported artifact is missing or is a symlink")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
        hashes[name] = actual
    report = json.loads((directory / "result.json").read_text())
    journal = json.loads((directory / "metrics.json").read_text())
    for field, name in (("checkpoint_sha256", "state.pt"), ("journal_sha256", "metrics.json"), ("rrd_sha256", "metrics.rrd")):
        if report[field] != hashes[name]:
            raise ValueError(f"Report artifact binding differs: {field}")
    validate_recipe(report["recipe"])
    validate_journal(journal, report["recipe"])
    steps = list(range(1, report["recipe"]["steps"] + 1))
    if [row["optimizer_step"] for row in journal] != steps:
        raise ValueError("Journal steps are incomplete or duplicated")
    recording = load_recording(directory / "metrics.rrd")
    if recording.application_id() != "npa-ray-train-synthetic" or recording.recording_id() != report["run_name"]:
        raise ValueError("Recording identity mismatch")
    observed = {}
    provenance = []
    for chunk in recording.chunks():
        if chunk.is_static and str(chunk.entity_path) == "/provenance/run":
            provenance.extend(chunk.to_record_batch().column("TextDocument:text").to_pylist())
        if chunk.is_static or not str(chunk.entity_path).startswith(("/metrics/", "/health/", "/checkpoint/")):
            continue
        batch = chunk.to_record_batch()
        timestamps = batch.column("optimizer_step").to_pylist()
        values = batch.column("Scalars:scalars").to_pylist()
        observed.setdefault(str(chunk.entity_path), []).extend(zip(timestamps, values, strict=True))
    embedded_report = {key: value for key, value in report.items() if key != "rrd_sha256"}
    if len(provenance) != 1 or len(provenance[0]) != 1 or json.loads(provenance[0][0]) != embedded_report:
        raise ValueError("Recording provenance differs from result report")
    for key in ("loss", "gradient_norm", "parameter_delta", "learning_rate", "samples_per_second"):
        rows = sorted(observed.get(f"/metrics/{key}", []))
        if [step for step, _ in rows] != steps:
            raise ValueError(f"Decoded {key} timeline differs from journal")
        if any(len(value) != 1 or not math.isclose(value[0], source[key], rel_tol=1e-12)
               for (_, value), source in zip(rows, journal, strict=True)):
            raise ValueError(f"Decoded {key} values differ from journal")
    rank_rows = sorted(observed.get("/health/cuda_ranks", []))
    if rank_rows != [(step, [report["recipe"]["workers"]]) for step in steps]:
        raise ValueError("Decoded CUDA rank timeline differs")
    checkpoint_steps = [step for step in steps if step % report["recipe"]["checkpoint_interval"] == 0 or step == steps[-1]]
    if sorted(observed.get("/checkpoint/materialized", [])) != [(step, [1]) for step in checkpoint_steps]:
        raise ValueError("Decoded checkpoint timeline differs")
    return {"artifacts_verified": len(entries), "optimizer_steps_decoded": len(steps),
            "metric_entities_decoded": 5, "checkpoint_events_decoded": len(checkpoint_steps)}


def main() -> None:
    """Inspect a downloaded export without contacting Ray, CUDA or object storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    transfer = parser.add_mutually_exclusive_group()
    transfer.add_argument("--download", help="Unsigned s3:// bucket/prefix/run/exports to retrieve first")
    transfer.add_argument("--publish", help="Retry an existing verified export to its original S3 destination")
    args = parser.parse_args()
    if args.download:
        from artifacts import download, storage

        filesystem, source = storage(args.download)
        download(filesystem, source, args.directory)
    result = inspect(args.directory)
    if args.publish:
        from artifacts import publish, storage

        filesystem, destination = storage(args.publish)
        publish(filesystem, destination, args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
