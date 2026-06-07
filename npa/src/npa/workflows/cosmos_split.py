"""Cosmos2 transfer and Cosmos3 reason workflow contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse


@dataclass(frozen=True)
class Cosmos2TransferConfig:
    """Runtime contract for a Cosmos2 transfer augmentation stage."""

    input_uri: str
    output_uri: str
    assets_uri: str = ""
    scene_spec_uri: str = ""
    image: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class Cosmos3ReasonConfig:
    """Runtime contract for a Cosmos3 reasoning stage."""

    input_uri: str
    output_uri: str
    model: str = "npa-cosmos3-reason"
    image: str = ""
    prompt: str = ""
    run_id: str = ""


def build_cosmos2_transfer_manifest(config: Cosmos2TransferConfig) -> dict[str, Any]:
    """Return the standalone Cosmos2 transfer manifest used by YAML, CLI, and SDK."""

    _require_uri(config.input_uri, "input_uri")
    _require_uri(config.output_uri, "output_uri")
    return {
        "schema": "npa.cosmos2.transfer.v1",
        "stage": "cosmos2-transfer",
        "run_id": config.run_id,
        "input_uri": config.input_uri,
        "output_uri": config.output_uri,
        "assets_uri": config.assets_uri,
        "scene_spec_uri": config.scene_spec_uri,
        "image": config.image,
        "status": "contract_ready",
    }


def build_cosmos3_reason_manifest(config: Cosmos3ReasonConfig) -> dict[str, Any]:
    """Return the standalone Cosmos3 reason manifest used by YAML, CLI, and SDK."""

    _require_uri(config.input_uri, "input_uri")
    _require_uri(config.output_uri, "output_uri")
    return {
        "schema": "npa.cosmos3.reason.v1",
        "stage": "cosmos3-reason",
        "run_id": config.run_id,
        "input_uri": config.input_uri,
        "output_uri": config.output_uri,
        "model": config.model,
        "image": config.image,
        "prompt": config.prompt,
        "status": "contract_ready",
    }


def write_manifest(payload: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Write a manifest to a local JSON file and return the payload with path metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "written_path": str(path)}


def _require_uri(value: str, name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must not be empty")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for raw workflow YAMLs."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    transfer = sub.add_parser("cosmos2-transfer")
    transfer.add_argument("--input-uri", required=True)
    transfer.add_argument("--output-uri", required=True)
    transfer.add_argument("--assets-uri", default="")
    transfer.add_argument("--scene-spec-uri", default="")
    transfer.add_argument("--image", default="")
    transfer.add_argument("--run-id", default="")
    transfer.add_argument("--output-json", default="")
    reason = sub.add_parser("cosmos3-reason")
    reason.add_argument("--input-uri", required=True)
    reason.add_argument("--output-uri", required=True)
    reason.add_argument("--model", default="npa-cosmos3-reason")
    reason.add_argument("--image", default="")
    reason.add_argument("--prompt", default="")
    reason.add_argument("--run-id", default="")
    reason.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    if args.command == "cosmos2-transfer":
        payload = build_cosmos2_transfer_manifest(
            Cosmos2TransferConfig(
                input_uri=args.input_uri,
                output_uri=args.output_uri,
                assets_uri=args.assets_uri,
                scene_spec_uri=args.scene_spec_uri,
                image=args.image,
                run_id=args.run_id,
            )
        )
    else:
        payload = build_cosmos3_reason_manifest(
            Cosmos3ReasonConfig(
                input_uri=args.input_uri,
                output_uri=args.output_uri,
                model=args.model,
                image=args.image,
                prompt=args.prompt,
                run_id=args.run_id,
            )
        )
    if args.output_json:
        payload = write_manifest(payload, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
