"""Join complete-episode prefix shards into the frozen canonical episode order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from panel_data import TASKS, load_split


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_record(record: dict) -> tuple[int, int]:
    if record.get("schema") != "npa.behavior.rlc-prefix-record.v2":
        raise ValueError("prefix shard schema differs")
    key = record.get("task_id"), record.get("episode_index")
    length = record.get("episode_length")
    if any(type(value) is not int for value in (*key, length)) or length < 1:
        raise ValueError("prefix shard episode identity or length is invalid")
    expected_frames = list(range(0, length, 20))
    if record.get("frames") != expected_frames:
        raise ValueError("prefix shard has missing, duplicate, or unordered frames")
    for field in ("auxiliary_raw_logits", "served_valid_logits", "sample_identities"):
        if not isinstance(record.get(field), list) or len(record[field]) != len(
            expected_frames
        ):
            raise ValueError(f"prefix shard {field} does not cover every frame")
    return key


def _read_shard(path: Path, expected: list[tuple[int, int]]) -> list[dict]:
    records = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("prefix shard record must be an object")
            _validate_record(record)
            records.append(record)
    observed = [(row["task_id"], row["episode_index"]) for row in records]
    if observed != expected:
        raise ValueError("prefix shard episode membership or order differs")
    return records


def merge(args: argparse.Namespace) -> dict:
    """Validate shard membership and write one complete canonical prefix stream.

    Args:
        args: Frozen split, ordered shard paths, and new output/receipt paths.

    Returns:
        Receipt binding every input and the merged output by SHA-256.

    Raises:
        FileExistsError: An output or receipt already exists.
        ValueError: A shard is incomplete, duplicated, reordered, or from another split.
    """
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("prefix merge outputs must be new")
    split = load_split(args.episode_split)
    keys = [
        (task, episode)
        for task in TASKS
        for episode in split["tasks"][str(task)][args.split]
    ]
    records, inputs = {}, []
    count = len(args.shard_input)
    if count < 1 or len(set(args.shard_input)) != count:
        raise ValueError("at least one distinct shard input is required")
    for index, path in enumerate(args.shard_input):
        rows = _read_shard(path, keys[index::count])
        records.update(((row["task_id"], row["episode_index"]), row) for row in rows)
        inputs.append(
            {
                "index": index,
                "sha256": _digest(path),
                "episodes": len(rows),
                "frames": sum(len(row["frames"]) for row in rows),
            }
        )
    with args.output.open("x") as stream:
        for key in keys:
            stream.write(
                json.dumps(records[key], sort_keys=True, separators=(",", ":")) + "\n"
            )
    receipt = {
        "schema": "npa.behavior.rlc-prefix-merge.v1",
        "split": args.split,
        "episode_split_sha256": _digest(args.episode_split),
        "shards": inputs,
        "output_sha256": _digest(args.output),
        "episodes": len(keys),
        "frames": sum(row["frames"] for row in inputs),
        "status": "merged",
    }
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    """Parse the explicit ordered shards and merge them after validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-split", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "holdout"), required=True)
    parser.add_argument("--shard-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
