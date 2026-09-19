"""MolmoAct VLA fine-tune / serve / eval workflow.

Companion to ``npa.cli.molmoact``: programmatic entry points for the BYOF
orchestrator, mirroring the ``openpi_pipeline`` module layout.  These helpers
validate a run configuration and return a run manifest dict; they do not
launch trainers or servers on their own.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_ID = "allenai/MolmoAct-7B-O-0812"
SOURCE_REPOSITORY = "https://huggingface.co/allenai/MolmoAct-7B-O-0812"
SOURCE_LICENSE = "Apache-2.0"


class MolmoActPipelineError(RuntimeError):
    """Raised when a MolmoAct workflow run configuration is invalid."""


def _require_model_id(model_id: str) -> str:
    model_id = (model_id or "").strip()
    if not model_id:
        raise MolmoActPipelineError("model_id must not be empty")
    if "/" not in model_id:
        raise MolmoActPipelineError(
            f"model_id {model_id!r} does not look like a HF repo id (expected org/name)"
        )
    return model_id


def _require_s3_uri(uri: str, name: str) -> str:
    uri = (uri or "").strip()
    if not uri:
        raise MolmoActPipelineError(f"{name} is required")
    if not uri.startswith("s3://"):
        raise MolmoActPipelineError(f"{name} must be an s3:// URI, got {uri!r}")
    return uri


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise MolmoActPipelineError(f"{name} must be positive, got {value}")
    return value


def finetune(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a MolmoAct fine-tune configuration; return its run manifest."""
    config = dict(config)
    model_id = _require_model_id(str(config.get("model_id", DEFAULT_MODEL_ID)))
    dataset_uri = _require_s3_uri(str(config.get("dataset_uri", "")), "dataset_uri")
    output_s3_uri = _require_s3_uri(
        str(config.get("output_s3_uri", "")), "output_s3_uri"
    )
    max_steps = _positive(int(config.get("max_steps", 5000)), "max_steps")
    batch_size = _positive(int(config.get("batch_size", 32)), "batch_size")
    num_gpus = _positive(int(config.get("num_gpus", 1)), "num_gpus")
    return {
        "command": "finetune",
        "status": "validated",
        "model_id": model_id,
        "dataset_uri": dataset_uri,
        "output_s3_uri": output_s3_uri,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": float(config.get("learning_rate", 1e-5)),
        "num_gpus": num_gpus,
        "run_name": str(config.get("run_name", "molmoact-finetune")),
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
    }


def serve(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a MolmoAct serve configuration; return its run manifest."""
    config = dict(config)
    model_id = _require_model_id(str(config.get("model_id", DEFAULT_MODEL_ID)))
    port = int(config.get("port", 8000))
    if not 1 <= port <= 65535:
        raise MolmoActPipelineError(f"port must be in [1, 65535], got {port}")
    device = str(config.get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise MolmoActPipelineError(
            f"unsupported device {device!r} (expected cuda or cpu)"
        )
    return {
        "command": "serve",
        "status": "validated",
        "model_id": model_id,
        "checkpoint": str(config.get("checkpoint", model_id)),
        "host": str(config.get("host", "0.0.0.0")),
        "port": port,
        "device": device,
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
    }


def eval(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a MolmoAct eval configuration; return its run manifest."""
    config = dict(config)
    model_id = _require_model_id(str(config.get("model_id", DEFAULT_MODEL_ID)))
    dataset_uri = _require_s3_uri(str(config.get("dataset_uri", "")), "dataset_uri")
    output_s3_uri = _require_s3_uri(
        str(config.get("output_s3_uri", "")), "output_s3_uri"
    )
    num_episodes = _positive(int(config.get("num_episodes", 50)), "num_episodes")
    return {
        "command": "eval",
        "status": "validated",
        "model_id": model_id,
        "checkpoint": str(config.get("checkpoint", model_id)),
        "dataset_uri": dataset_uri,
        "num_episodes": num_episodes,
        "output_s3_uri": output_s3_uri,
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
    }


_COMMANDS = {"finetune": finetune, "serve": serve, "eval": eval}


def run(command: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a named MolmoAct pipeline command with the given configuration."""
    handler = _COMMANDS.get(command)
    if handler is None:
        raise MolmoActPipelineError(
            f"unknown command {command!r}; expected one of {sorted(_COMMANDS)}"
        )
    return handler(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("finetune", help="Validate a fine-tune configuration.")
    sub.add_parser("serve", help="Validate a serve configuration.")
    sub.add_parser("eval", help="Validate an eval configuration.")
    for subparser in (sub.choices["finetune"], sub.choices["eval"]):
        subparser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        subparser.add_argument("--dataset-uri", required=True)
        subparser.add_argument("--output-s3-uri", required=True)
    for subparser in (sub.choices["serve"], sub.choices["eval"]):
        subparser.add_argument("--checkpoint", default="")
    sub.choices["finetune"].add_argument("--max-steps", type=int, default=5000)
    sub.choices["finetune"].add_argument("--num-gpus", type=int, default=1)
    sub.choices["serve"].add_argument("--port", type=int, default=8000)
    sub.choices["serve"].add_argument("--device", default="cuda")
    sub.choices["eval"].add_argument("--num-episodes", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = {k: v for k, v in vars(args).items() if k != "command"}
    manifest = run(args.command, config)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
