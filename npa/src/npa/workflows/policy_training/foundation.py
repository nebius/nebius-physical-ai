"""Prepare reviewable foundation-training event workflows and Soperator batch scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from .distributed import slurm_script
from .events import event_workflow


def main() -> None:
    """Materialize private launch files without submitting jobs or changing a cluster.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: The input recipe or event is invalid.
        SystemExit: Required command arguments are absent.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("event", "slurm"))
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--template-path", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input_path.read_text())
    if args.command == "slurm":
        rendered = slurm_script(payload)
    else:
        if args.template_path is None:
            parser.error("event requires --template-path")
        source = yaml.safe_load(args.template_path.read_text())
        rendered = yaml.safe_dump(event_workflow(source, payload), sort_keys=False)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(rendered)


if __name__ == "__main__":
    main()
