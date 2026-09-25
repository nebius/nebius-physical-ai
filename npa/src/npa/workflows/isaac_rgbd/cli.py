"""Expose stateless rig capture, validation, and input-fixture adapters to workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .contract import _json_bytes
from .fixture import write_fixture
from .transport import _acquire, _publish, _s3_uri, validate_s3


def build_parser():
    """Build the module parser used by the workflow toolRef argv guardrail.

    Args:
        None.

    Returns:
        ArgumentParser for stateless capture, validate, and fixture operations.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Calibrated Isaac RGB-D sensor-rig adapters"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("capture", "validate"):
        command = commands.add_parser(operation)
        command.add_argument("--input-path", required=True)
        command.add_argument("--output-path", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--output-path", required=True)
    reference = commands.add_parser("prepare-reference")
    reference.add_argument("--output-path", required=True)
    return parser


def _capture(args, temporary):
    from .runtime import capture_local

    _s3_uri(args.output_path)
    root, output = temporary / "input", temporary / "output"
    request, scope = _acquire(args.input_path, root)
    output.mkdir()

    def publish(directory, manifest):
        _publish(directory, manifest, args.output_path)
        print(
            json.dumps({"published": True, "frames": len(manifest["frames"])}),
            flush=True,
        )

    capture_local(root, request, output, scope=scope, publish=publish)


def main(argv=None):
    """Run one stage, preserving nonzero exceptions across Kit teardown.

    Args:
        argv: Optional argument list; defaults to process arguments.

    Returns:
        Zero after a completed operation; fixture creation does not render data.

    Raises:
        ValueError: Invalid inputs or failed data validation.
        RuntimeError: Isaac runtime requirements are unmet.
        Exception: Renderer or storage failures propagate to the workflow runtime.
    """
    args = build_parser().parse_args(argv)
    if args.operation == "fixture":
        write_fixture(args.output_path)
        print("Created procedural input bundle; no capture has been executed.")
        return 0
    with tempfile.TemporaryDirectory(prefix="npa-rgbd-") as directory:
        root = Path(directory)
        if args.operation == "capture":
            _capture(args, root)
        elif args.operation == "prepare-reference":
            from .reference import prepare_reference

            request = prepare_reference(args.output_path, root)
            print(
                json.dumps({"prepared": True, "rig_poses": len(request["trajectory"])})
            )
        else:
            report = validate_s3(args.input_path, args.output_path, root)
            print(_json_bytes(report).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
