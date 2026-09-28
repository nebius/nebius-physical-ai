"""Freeze one matched training arm against exact source, dataset, trace, and parent identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_conditioning import load_trace, sha256

PARENT_CHECKPOINT = "9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea"
DATASET_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"


def freeze(args: argparse.Namespace) -> dict:
    """Validate and write a complete arm-specific input manifest."""

    trace_identities = json.loads(args.trace_identities.read_text())
    trace = load_trace(
        args.training_trace, split="training", identities=trace_identities
    )
    files = {
        "config.json": args.config,
        "episode-split.json": args.episode_split,
        "source-manifest.json": args.source_manifest,
        "dataset-validation.json": args.dataset_validation,
        "training-trace.jsonl": args.training_trace,
    }
    manifest = {
        "schema": "npa.behavior.rlc-matched-stage-inputs.v1",
        "dataset_revision": DATASET_REVISION,
        "parent_checkpoint_archive_sha256": PARENT_CHECKPOINT,
        "candidate_config_sha256": sha256(args.config),
        "trace_identities": trace_identities,
        "training_trace_rows": len(trace),
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        },
        "adapter_files": json.loads(args.adapter_files.read_text()),
        "development_or_reporting_instances_used": False,
        "holdout_optimizer_access": False,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    """Parse arguments and freeze the arm input manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "episode-split",
        "source-manifest",
        "dataset-validation",
        "training-trace",
        "trace-identities",
        "adapter-files",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
