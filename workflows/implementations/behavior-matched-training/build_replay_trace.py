"""Build a canonical native-filter replay trace from frozen-parent prefix outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_conditioning import replay_episode, write_trace


def build(
    source: Path, output: Path, identities: Path, split: str
) -> dict[str, object]:
    """Validate chronological episode records and build one immutable trace."""

    identity_values = json.loads(identities.read_text())
    records = [json.loads(line) for line in source.read_text().splitlines() if line]
    rows = []
    previous = None
    for record in records:
        if record.get("schema") != "npa.behavior.rlc-prefix-record.v2":
            raise ValueError("prefix record schema differs")
        key = (int(record["task_id"]), int(record["episode_index"]))
        if previous is not None and key <= previous:
            raise ValueError(
                "prefix records must be strictly ordered by task and episode"
            )
        previous = key
        episode_length = int(record["episode_length"])
        frames = list(range(0, episode_length, 20))
        if record["frames"] != frames:
            raise ValueError(f"prefix record frame cadence differs: {key}")
        rows.extend(
            replay_episode(
                task_id=key[0],
                episode_index=key[1],
                episode_length=episode_length,
                logits=record["auxiliary_raw_logits"],
                served_logits=record["served_valid_logits"],
                sample_digests=[
                    (item["observation_sha256"], item["action_sha256"])
                    for item in record["sample_identities"]
                ],
            )
        )
    return write_trace(output, rows, split=split, identities=identity_values)


def main() -> None:
    """Parse arguments and build the canonical replay trace."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-records", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.prefix_records, args.output, args.identities, args.split)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
