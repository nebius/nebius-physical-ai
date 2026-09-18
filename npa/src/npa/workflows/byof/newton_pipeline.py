"""Newton physics engine workbench workflow stages.

Mirrors the ``openpi_pipeline.py`` module pattern: stage functions backed by
an argparse entry point (``main``), a module-level error type, and JSON
stage artifacts written to URIs.

The heavy Newton imports stay lazy so the CLI surface can be imported on
machines without the physics engine installed.  The three stages are
currently stubs: they validate their arguments, record a stub manifest at
``output_uri`` for plumbing, and then raise :class:`NewtonPipelineError`
with a clear "not yet implemented" message (tracking issue
nebius/nebius-physical-ai#499).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

ISSUE_REF = "nebius/nebius-physical-ai#499"

SOURCE_REPOSITORY = "https://github.com/newton-physics/newton"
SOURCE_LICENSE = "Apache-2.0"
DEFAULT_CONFIG_NAME = "newton_double_pendulum"
STUB_SCHEMA_TRAIN = "npa.workbench.newton.train-teacher.stub.v1"
STUB_SCHEMA_DEMOS = "npa.workbench.newton.generate-demos.stub.v1"
STUB_SCHEMA_EVAL = "npa.workbench.newton.eval.stub.v1"

_NOT_IMPLEMENTED = (
    "not yet implemented: Newton workbench pipeline stages are stubs in this "
    f"release. Tracking issue: {ISSUE_REF}."
)


class NewtonPipelineError(RuntimeError):
    """Raised when a Newton pipeline invariant is not met."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def newton_version() -> str:
    """Return the installed Newton package version (lazy import)."""
    import newton

    return str(getattr(newton, "__version__", "unknown"))


def _write_bytes_uri(
    uri: str, payload: bytes, *, content_type: str = "application/octet-stream"
) -> None:
    del content_type  # file writes only; kept for interface parity with openpi_pipeline
    parsed = urlparse(uri)
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json_uri(uri: str, payload: Mapping[str, Any]) -> None:
    _write_bytes_uri(
        uri, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _stage_manifest(
    *,
    stage: str,
    schema: str,
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": schema,
        "stage": stage,
        "status": "not_implemented",
        "issue": ISSUE_REF,
        "inputs": dict(inputs),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "license": SOURCE_LICENSE,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def train_teacher(
    *,
    dataset_uri: str,
    output_uri: str,
    config_name: str = DEFAULT_CONFIG_NAME,
    train_steps: int = 100,
    seed: int | None = None,
) -> Mapping[str, Any]:
    """Validate arguments, record a stub manifest, then raise (stage not implemented)."""
    if not dataset_uri:
        raise NewtonPipelineError("dataset_uri is required")
    if not output_uri:
        raise NewtonPipelineError("output_uri is required")
    if train_steps < 1:
        raise NewtonPipelineError("train_steps must be >= 1")
    manifest = _stage_manifest(
        stage="train-teacher",
        schema=STUB_SCHEMA_TRAIN,
        inputs={
            "dataset_uri": dataset_uri,
            "config_name": config_name,
            "train_steps": train_steps,
            "seed": seed,
        },
    )
    _write_json_uri(output_uri, manifest)
    raise NewtonPipelineError(_NOT_IMPLEMENTED)


def generate_demos(
    *,
    checkpoint_uri: str,
    output_uri: str,
    num_demos: int = 10,
    seed: int | None = None,
) -> Mapping[str, Any]:
    """Validate arguments, record a stub manifest, then raise (stage not implemented)."""
    if not checkpoint_uri:
        raise NewtonPipelineError("checkpoint_uri is required")
    if not output_uri:
        raise NewtonPipelineError("output_uri is required")
    if num_demos < 1:
        raise NewtonPipelineError("num_demos must be >= 1")
    manifest = _stage_manifest(
        stage="generate-demos",
        schema=STUB_SCHEMA_DEMOS,
        inputs={
            "checkpoint_uri": checkpoint_uri,
            "num_demos": num_demos,
            "seed": seed,
        },
    )
    _write_json_uri(output_uri, manifest)
    raise NewtonPipelineError(_NOT_IMPLEMENTED)


def eval(
    *,
    checkpoint_uri: str,
    dataset_uri: str,
    output_uri: str,
    num_episodes: int = 5,
    seed: int | None = None,
) -> Mapping[str, Any]:
    """Validate arguments, record a stub manifest, then raise (stage not implemented)."""
    if not checkpoint_uri:
        raise NewtonPipelineError("checkpoint_uri is required")
    if not dataset_uri:
        raise NewtonPipelineError("dataset_uri is required")
    if not output_uri:
        raise NewtonPipelineError("output_uri is required")
    if num_episodes < 1:
        raise NewtonPipelineError("num_episodes must be >= 1")
    manifest = _stage_manifest(
        stage="eval",
        schema=STUB_SCHEMA_EVAL,
        inputs={
            "checkpoint_uri": checkpoint_uri,
            "dataset_uri": dataset_uri,
            "num_episodes": num_episodes,
            "seed": seed,
        },
    )
    _write_json_uri(output_uri, manifest)
    raise NewtonPipelineError(_NOT_IMPLEMENTED)


def _run_stage(args: argparse.Namespace) -> int:
    dispatch = {
        "train-teacher": lambda: train_teacher(
            dataset_uri=args.dataset_uri,
            output_uri=args.output_uri,
            config_name=args.config_name,
            train_steps=args.train_steps,
            seed=args.seed,
        ),
        "generate-demos": lambda: generate_demos(
            checkpoint_uri=args.checkpoint_uri,
            output_uri=args.output_uri,
            num_demos=args.num_demos,
            seed=args.seed,
        ),
        "eval": lambda: eval(
            checkpoint_uri=args.checkpoint_uri,
            dataset_uri=args.dataset_uri,
            output_uri=args.output_uri,
            num_episodes=args.num_episodes,
            seed=args.seed,
        ),
    }
    dispatch[args.stage]()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newton_pipeline",
        description="Newton physics workbench pipeline stages (stubs).",
    )
    commands = parser.add_subparsers(dest="stage", required=True)

    train = commands.add_parser("train-teacher")
    train.add_argument("--dataset-uri", required=True)
    train.add_argument("--output-uri", required=True)
    train.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    train.add_argument("--train-steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=None)

    demos = commands.add_parser("generate-demos")
    demos.add_argument("--checkpoint-uri", required=True)
    demos.add_argument("--output-uri", required=True)
    demos.add_argument("--num-demos", type=int, default=10)
    demos.add_argument("--seed", type=int, default=None)

    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--checkpoint-uri", required=True)
    evaluate.add_argument("--dataset-uri", required=True)
    evaluate.add_argument("--output-uri", required=True)
    evaluate.add_argument("--num-episodes", type=int, default=5)
    evaluate.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
