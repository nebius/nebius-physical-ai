"""OpenVLA / OpenVLA-OFT workflow stages (train / serve / eval).

Mirrors the shape of ``npa.workflows.byof.openpi_pipeline``: train/serve/eval
entry points plus a ``python -m npa.workflows.byof.openvla_pipeline`` argparse
interface. This module never vendors weights — it builds and runs the upstream
OpenVLA scripts (``vla-scripts/finetune.py`` for the OFT LoRA recipe and
``vla-scripts/deploy.py`` for serving) from
https://github.com/openvla/openvla (MIT licensed), or validates inputs and
prints the execution plan in ``--dry-run`` mode.

Runnable: ``python -m npa.workflows.byof.openvla_pipeline <train|serve|eval|check> ...``
"""

from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

SOURCE_REPOSITORY = "https://github.com/openvla/openvla"
SOURCE_LICENSE = "MIT"
DEFAULT_MODEL_ID = "openvla/openvla-7b"
HF_MODEL_API = "https://huggingface.co/api/models/openvla/openvla-7b"

# Upstream entry points inside an https://github.com/openvla/openvla checkout.
UPSTREAM_FINETUNE_SCRIPT = "vla-scripts/finetune.py"
UPSTREAM_DEPLOY_SCRIPT = "vla-scripts/deploy.py"

# OpenVLA-OFT LoRA recipe defaults, taken from the upstream FinetuneConfig.
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_STEPS = 200_000
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_DROPOUT = 0.0


class OpenVLAPipelineError(RuntimeError):
    """Raised when an OpenVLA workflow invariant is not met."""


def _require_non_empty(value: str, name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise OpenVLAPipelineError(f"{name} must not be empty")
    return cleaned


_HF_REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def check_model_accessible(
    model_id: str = DEFAULT_MODEL_ID, *, timeout: float = 15.0
) -> bool:
    """Return True when the HuggingFace model repo resolves over the public API."""
    cleaned = _require_non_empty(model_id, "model_id")
    if not _HF_REPO_ID.match(cleaned):
        raise OpenVLAPipelineError(
            f"model_id {cleaned!r} is not a valid HuggingFace repo id (expected owner/name)"
        )
    url = f"https://huggingface.co/api/models/{cleaned}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "npa-openvla-pipeline"}
    )
    try:
        # URL is confined to the HuggingFace API host: the repo id is validated
        # against _HF_REPO_ID above so it cannot alter the scheme or host.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("id") == model_id and payload.get("private") is False
    except Exception:
        return False


def require_model_accessible(model_id: str = DEFAULT_MODEL_ID) -> None:
    """Fail closed when the base OpenVLA weights are not publicly reachable."""
    if not check_model_accessible(model_id):
        raise OpenVLAPipelineError(
            f"model repo {model_id!r} is not publicly accessible via the "
            "HuggingFace API; check the id or network access"
        )


@dataclass
class TrainConfig:
    """OpenVLA-OFT LoRA fine-tuning configuration."""

    model_id: str = DEFAULT_MODEL_ID
    dataset_uri: str = ""
    dataset_name: str = "finetune"
    output_dir: str = "runs/openvla-oft"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = DEFAULT_MAX_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    image_aug: bool = True
    seed: int = 7
    dry_run: bool = True
    openvla_checkout: str = ""
    extra_args: Sequence[str] = field(default_factory=list)

    def validate(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.dataset_uri, "dataset_uri")
        _require_non_empty(self.output_dir, "output_dir")
        for name in ("batch_size", "max_steps", "lora_rank"):
            if getattr(self, name) <= 0:
                raise OpenVLAPipelineError(f"{name} must be > 0")
        if self.learning_rate <= 0:
            raise OpenVLAPipelineError("learning_rate must be > 0")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise OpenVLAPipelineError("lora_dropout must be in [0, 1)")

    def to_upstream_argv(self) -> list[str]:
        """Build the upstream ``vla-scripts/finetune.py`` (draccus) argv."""
        self.validate()
        script = Path(self.openvla_checkout or ".") / UPSTREAM_FINETUNE_SCRIPT
        argv = [
            sys.executable,
            str(script),
            "--vla_path",
            self.model_id,
            "--data_root_dir",
            self.dataset_uri,
            "--dataset_name",
            self.dataset_name,
            "--run_root_dir",
            self.output_dir,
            "--batch_size",
            str(self.batch_size),
            "--max_steps",
            str(self.max_steps),
            "--learning_rate",
            repr(self.learning_rate),
            "--lora_rank",
            str(self.lora_rank),
            "--lora_dropout",
            repr(self.lora_dropout),
            "--seed",
            str(self.seed),
        ]
        argv.append("--image_aug" if self.image_aug else "--no_image_aug")
        argv.extend(self.extra_args)
        return argv


@dataclass
class ServeConfig:
    """OpenVLA checkpoint serving configuration."""

    checkpoint: str = DEFAULT_MODEL_ID
    host: str = "127.0.0.1"
    port: int = 8000
    dry_run: bool = True
    openvla_checkout: str = ""

    def validate(self) -> None:
        _require_non_empty(self.checkpoint, "checkpoint")
        _require_non_empty(self.host, "host")
        if not 1 <= self.port <= 65535:
            raise OpenVLAPipelineError("port must be in [1, 65535]")

    def to_upstream_argv(self) -> list[str]:
        """Build the upstream ``vla-scripts/deploy.py`` (draccus) argv."""
        self.validate()
        script = Path(self.openvla_checkout or ".") / UPSTREAM_DEPLOY_SCRIPT
        return [
            sys.executable,
            str(script),
            "--openvla_path",
            self.checkpoint,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]


@dataclass
class EvalConfig:
    """OpenVLA checkpoint evaluation configuration."""

    checkpoint: str = ""
    dataset_uri: str = ""
    num_episodes: int = 10
    output_uri: str = ""
    seed: int = 7
    dry_run: bool = True

    def validate(self) -> None:
        _require_non_empty(self.checkpoint, "checkpoint")
        _require_non_empty(self.dataset_uri, "dataset_uri")
        if self.num_episodes <= 0:
            raise OpenVLAPipelineError("num_episodes must be > 0")

    def plan(self) -> dict[str, object]:
        """Return the evaluation plan as JSON-serializable metadata."""
        self.validate()
        return {
            "checkpoint": self.checkpoint,
            "dataset_uri": self.dataset_uri,
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "output_uri": self.output_uri or None,
            "upstream": f"{SOURCE_REPOSITORY} ({SOURCE_LICENSE})",
        }


def _run(argv: Sequence[str]) -> int:
    env = dict(os.environ)
    print(f"[openvla] exec: {' '.join(argv)}", flush=True)
    proc = subprocess.run(list(argv), env=env)
    return proc.returncode


def train(cfg: TrainConfig) -> int:
    """Run OFT LoRA fine-tuning (or print the plan in dry-run mode)."""
    cfg.validate()
    argv = cfg.to_upstream_argv()
    print("[openvla] train plan")
    print(f"  model:   {cfg.model_id}")
    print(f"  dataset: {cfg.dataset_uri} (name: {cfg.dataset_name})")
    print(f"  output:  {cfg.output_dir}")
    print(f"  upstream argv: {' '.join(argv)}")
    if cfg.dry_run:
        print("[openvla] dry-run: training not started")
        return 0
    return _run(argv)


def serve(cfg: ServeConfig) -> int:
    """Serve a checkpoint (or print the plan in dry-run mode)."""
    cfg.validate()
    argv = cfg.to_upstream_argv()
    print("[openvla] serve plan")
    print(f"  checkpoint: {cfg.checkpoint}")
    print(f"  endpoint:   http://{cfg.host}:{cfg.port}")
    print(f"  upstream argv: {' '.join(argv)}")
    if cfg.dry_run:
        print("[openvla] dry-run: server not started")
        return 0
    return _run(argv)


def evaluate(cfg: EvalConfig) -> int:
    """Evaluate a checkpoint (or print the plan in dry-run mode)."""
    plan = cfg.plan()
    print("[openvla] eval plan")
    print(json.dumps(plan, indent=2, sort_keys=True))
    if cfg.dry_run:
        print("[openvla] dry-run: evaluation not started")
        return 0
    # Rollout execution requires a live serve endpoint; the plan above is the
    # contract a GPU workbench consumes.
    print(
        "[openvla] stub: rollout execution runs on a GPU workbench against "
        "`openvla_pipeline serve`."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openvla_pipeline",
        description="OpenVLA / OpenVLA-OFT workflow stages.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_p = commands.add_parser("train", help="OFT LoRA fine-tuning.")
    train_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    train_p.add_argument("--dataset-uri", required=True)
    train_p.add_argument("--dataset-name", default="finetune")
    train_p.add_argument("--output-dir", default="runs/openvla-oft")
    train_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    train_p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    train_p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    train_p.add_argument("--image-aug/--no-image-aug", dest="image_aug", default=True)
    train_p.add_argument("--seed", type=int, default=7)
    train_p.add_argument("--openvla-checkout", default="")
    train_p.add_argument("--dry-run/--no-dry-run", dest="dry_run", default=True)

    serve_p = commands.add_parser("serve", help="Serve a checkpoint over HTTP.")
    serve_p.add_argument("--checkpoint", default=DEFAULT_MODEL_ID)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--openvla-checkout", default="")
    serve_p.add_argument("--dry-run/--no-dry-run", dest="dry_run", default=True)

    eval_p = commands.add_parser("eval", help="Evaluate a checkpoint.")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--dataset-uri", required=True)
    eval_p.add_argument("--num-episodes", type=int, default=10)
    eval_p.add_argument("--output-uri", default="")
    eval_p.add_argument("--seed", type=int, default=7)
    eval_p.add_argument("--dry-run/--no-dry-run", dest="dry_run", default=True)

    check_p = commands.add_parser(
        "check", help="Verify the base OpenVLA model repo is publicly accessible."
    )
    check_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return train(
            TrainConfig(
                model_id=args.model_id,
                dataset_uri=args.dataset_uri,
                dataset_name=args.dataset_name,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_dropout=args.lora_dropout,
                image_aug=args.image_aug,
                seed=args.seed,
                dry_run=args.dry_run,
                openvla_checkout=args.openvla_checkout,
            )
        )
    if args.command == "serve":
        return serve(
            ServeConfig(
                checkpoint=args.checkpoint,
                host=args.host,
                port=args.port,
                dry_run=args.dry_run,
                openvla_checkout=args.openvla_checkout,
            )
        )
    if args.command == "eval":
        return evaluate(
            EvalConfig(
                checkpoint=args.checkpoint,
                dataset_uri=args.dataset_uri,
                num_episodes=args.num_episodes,
                output_uri=args.output_uri,
                seed=args.seed,
                dry_run=args.dry_run,
            )
        )
    if args.command == "check":
        ok = check_model_accessible(args.model_id)
        print(f"[openvla] model {args.model_id!r} accessible: {ok}")
        return 0 if ok else 1
    raise OpenVLAPipelineError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
