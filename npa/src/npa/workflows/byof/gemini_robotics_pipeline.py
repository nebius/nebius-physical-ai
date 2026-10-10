"""Gemini Robotics hosted planning and evaluation workflow stages.

The Gemini API is closed-weight, so these CPU stages use the default NPA image.
Their public workflow handoff is deliberately narrow: every input and receipt is
an exact S3 object, and a receipt is published with a conditional create. A
caller must supply the API base URL and model id explicitly because neither
validated default model id, live-verified 2026-10-10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from npa.cli.path_contract import (
    PathContractError,
    validate_read_path,
    validate_write_path,
)
from npa.clients.gemini_robotics import (
    DEFAULT_MODEL_ID,
    EVAL_RECEIPT_SCHEMA,
    PLAN_RECEIPT_SCHEMA,
    EvalResult,
    GeminiRoboticsClient,
    GeminiRoboticsError,
    PlanResult,
    resolve_config,
)
from npa.clients.storage import StorageClient, StorageError


class GeminiRoboticsPipelineError(RuntimeError):
    """Raised when a Gemini workflow stage cannot safely run or publish."""


class ArtifactStorage(Protocol):
    """The small storage boundary needed by Gemini workflow stages."""

    def put_bytes_conditional(
        self,
        payload: bytes,
        bucket_uri: str,
        *,
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Create an object only when it does not already exist."""

    def read_bytes_with_etag(self, bucket_uri: str) -> tuple[bytes, str] | None:
        """Read one exact object and its storage version."""


@dataclass
class GeminiRoboticsPipelineConfig:
    """Inputs for one hosted Gemini stage.

    ``output_path`` is an exact, previously unused ``s3://`` receipt URI. Image
    paths are internal, materialized files used only after a caller has validated
    and downloaded its public S3 inputs.
    """

    task: str
    output_path: str
    images: list[str] = field(default_factory=list)
    model: str = ""

    def require_model(self) -> str:
        """Return the model id, defaulting to the live-validated endpoint model."""

        return self.model.strip() or DEFAULT_MODEL_ID

    def require_output_path(self) -> str:
        """Validate the durable receipt target before the hosted request."""

        try:
            output_path = validate_write_path(
                self.output_path,
                tool="gemini-robotics",
                required=True,
            )
        except PathContractError as exc:
            raise GeminiRoboticsPipelineError(str(exc)) from exc
        if output_path.endswith("/"):
            raise GeminiRoboticsPipelineError(
                "gemini-robotics --output-path must name one receipt object, not an S3 prefix."
            )
        return output_path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _storage_or_default(storage: ArtifactStorage | None) -> ArtifactStorage:
    if storage is not None:
        return storage
    try:
        return StorageClient.from_environment()
    except StorageError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc


def _client_or_default(client: GeminiRoboticsClient | None) -> GeminiRoboticsClient:
    if client is not None:
        return client
    try:
        return GeminiRoboticsClient(resolve_config())
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc


def _write_receipt(
    storage: ArtifactStorage, output_path: str, payload: Mapping[str, Any]
) -> tuple[str, str]:
    raw = _receipt_bytes(payload)
    try:
        etag = storage.put_bytes_conditional(
            raw,
            output_path,
            if_none_match=True,
            content_type="application/json",
        )
    except StorageError as exc:
        raise GeminiRoboticsPipelineError(
            f"could not conditionally publish receipt at {output_path!r}: {exc}"
        ) from exc
    return _sha256_bytes(raw), etag


def read_eval_input(
    input_path: str, storage: ArtifactStorage | None = None
) -> tuple[dict[str, Any], str, str, str]:
    """Load the exact S3 JSON object containing ``plan_text`` and ``rubric``."""

    try:
        input_path = validate_read_path(
            input_path,
            tool="gemini-robotics eval",
            option="--input-path",
            allow_hf=False,
        )
    except PathContractError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc
    if input_path.endswith("/"):
        raise GeminiRoboticsPipelineError(
            "gemini-robotics eval --input-path must name one JSON object, not an S3 prefix."
        )
    active_storage = _storage_or_default(storage)
    try:
        source = active_storage.read_bytes_with_etag(input_path)
    except StorageError as exc:
        raise GeminiRoboticsPipelineError(
            f"could not read eval input at {input_path!r}: {exc}"
        ) from exc
    if source is None:
        raise GeminiRoboticsPipelineError(
            f"gemini-robotics eval input does not exist: {input_path!r}"
        )
    raw, etag = source
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise GeminiRoboticsPipelineError(
            "gemini-robotics eval input must be a JSON object with non-empty "
            "plan_text and rubric strings."
        ) from exc
    if not isinstance(payload, dict):
        raise GeminiRoboticsPipelineError(
            "gemini-robotics eval input must be a JSON object with non-empty "
            "plan_text and rubric strings."
        )
    plan_text = payload.get("plan_text")
    rubric = payload.get("rubric")
    if not isinstance(plan_text, str) or not plan_text.strip():
        raise GeminiRoboticsPipelineError(
            "gemini-robotics eval input requires a non-empty plan_text string."
        )
    if not isinstance(rubric, str) or not rubric.strip():
        raise GeminiRoboticsPipelineError(
            "gemini-robotics eval input requires a non-empty rubric string."
        )
    return payload, input_path, etag, _sha256_bytes(raw)


def run_er_planning_stage(
    config: GeminiRoboticsPipelineConfig,
    client: GeminiRoboticsClient | None = None,
    storage: ArtifactStorage | None = None,
) -> dict[str, Any]:
    """Run ER planning and conditionally publish its durable receipt."""

    model = config.require_model()
    output_path = config.require_output_path()
    active = _client_or_default(client)
    active_storage = _storage_or_default(storage)
    try:
        result: PlanResult = active.plan(
            task=config.task,
            images=[Path(path) for path in config.images],
            model=model,
        )
    except (GeminiRoboticsError, OSError) as exc:
        raise GeminiRoboticsPipelineError(f"ER planning stage failed: {exc}") from exc
    payload: dict[str, Any] = {
        "schema": PLAN_RECEIPT_SCHEMA,
        "task": config.task,
        "model": result.model,
        "plan_text": result.text,
        "model_function_calls": result.model_function_calls,
        "finish_reason": result.finish_reason,
        "created_at": _utc_now(),
        "advisory_only": True,
    }
    digest, etag = _write_receipt(active_storage, output_path, payload)
    payload["artifact_path"] = output_path
    payload["artifact_sha256"] = digest
    payload["artifact_etag"] = etag
    return payload


def run_eval_stage(
    config: GeminiRoboticsPipelineConfig,
    plan_receipt: Mapping[str, Any],
    rubric: str,
    client: GeminiRoboticsClient | None = None,
    storage: ArtifactStorage | None = None,
    *,
    input_path: str = "",
    input_etag: str = "",
    input_sha256: str = "",
) -> dict[str, Any]:
    """Evaluate a durable plan input and conditionally publish the receipt."""

    model = config.require_model()
    output_path = config.require_output_path()
    plan_text = plan_receipt.get("plan_text")
    if not isinstance(plan_text, str) or not plan_text.strip():
        raise GeminiRoboticsPipelineError("Eval stage requires non-empty plan_text.")
    if not rubric.strip():
        raise GeminiRoboticsPipelineError("Eval stage requires a non-empty rubric.")
    if any((input_path, input_etag, input_sha256)) and not all(
        (input_path, input_etag, input_sha256)
    ):
        raise GeminiRoboticsPipelineError(
            "Eval stage input provenance must include path, ETag, and SHA-256 together."
        )
    active = _client_or_default(client)
    active_storage = _storage_or_default(storage)
    try:
        result: EvalResult = active.eval_plan(
            plan_text=plan_text,
            rubric=rubric,
            model=model,
        )
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(f"Eval stage failed: {exc}") from exc
    payload: dict[str, Any] = {
        "schema": EVAL_RECEIPT_SCHEMA,
        "model": result.model,
        "scores": result.scores,
        "summary": result.summary,
        "created_at": _utc_now(),
        "advisory_only": True,
    }
    if input_path:
        payload["input_path"] = input_path
        payload["input_etag"] = input_etag
        payload["input_sha256"] = input_sha256
    digest, etag = _write_receipt(active_storage, output_path, payload)
    payload["artifact_path"] = output_path
    payload["artifact_sha256"] = digest
    payload["artifact_etag"] = etag
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow-stage parser with exact S3 handoff flags."""

    parser = argparse.ArgumentParser(
        prog="gemini_robotics_pipeline",
        description="Gemini Robotics hosted workflow stages (live-validated API adapter).",
    )
    parser.add_argument(
        "--api-base-url",
        default="",
        help="Gemini API base URL (default: live-validated endpoint).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Gemini API key (falls back to GOOGLE_API_KEY).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan", help="Run ER planning and publish one receipt.")
    plan_p.add_argument("--task", required=True)
    plan_p.add_argument(
        "--model", default="", help="Gemini model id (default: live-validated)."
    )
    plan_p.add_argument("--output-path", required=True, help="Exact S3 receipt URI.")

    eval_p = sub.add_parser("eval", help="Evaluate one plan/rubric input object.")
    eval_p.add_argument("--input-path", required=True, help="Exact S3 JSON input URI.")
    eval_p.add_argument(
        "--model", default="", help="Gemini model id (default: live-validated)."
    )
    eval_p.add_argument("--output-path", required=True, help="Exact S3 receipt URI.")
    return parser


def _configured_client(args: argparse.Namespace) -> GeminiRoboticsClient:
    try:
        return GeminiRoboticsClient(
            resolve_config(api_key=args.api_key or None, base_url=args.api_base_url)
        )
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run one hosted stage with prevalidated durable handoff paths."""

    args = build_parser().parse_args(argv)
    if args.command == "plan":
        config = GeminiRoboticsPipelineConfig(
            task=args.task,
            output_path=args.output_path,
            model=args.model,
        )
        config.require_model()
        config.require_output_path()
        client = _configured_client(args)
        storage = _storage_or_default(None)
        receipt = run_er_planning_stage(config, client, storage)
    elif args.command == "eval":
        config = GeminiRoboticsPipelineConfig(
            task=args.input_path,
            output_path=args.output_path,
            model=args.model,
        )
        config.require_model()
        config.require_output_path()
        client = _configured_client(args)
        storage = _storage_or_default(None)
        plan_receipt, input_path, input_etag, input_sha256 = read_eval_input(
            args.input_path, storage
        )
        receipt = run_eval_stage(
            config,
            plan_receipt,
            str(plan_receipt["rubric"]),
            client,
            storage,
            input_path=input_path,
            input_etag=input_etag,
            input_sha256=input_sha256,
        )
    else:  # pragma: no cover - argparse restricts commands
        raise GeminiRoboticsPipelineError(f"unknown command {args.command!r}")
    print(
        json.dumps(
            {
                "artifact_path": receipt["artifact_path"],
                "artifact_sha256": receipt["artifact_sha256"],
                "artifact_etag": receipt["artifact_etag"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
