"""Create the durable prompt artifact for the agent deployment-review workflow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

PROMPT_SCHEMA = "npa.token_factory.prompts.v1"

DEPLOYMENT_REVIEW_PROMPTS = (
    (
        "delivery-plan",
        "Propose a concrete Nebius Physical AI delivery plan for a Kubernetes "
        "workflow, covering its execution target, durable artifact path, and "
        "operator confirmation boundary.",
    ),
    (
        "readiness-risks",
        "Identify the highest-priority prerequisites and operational risks before "
        "running a Nebius Physical AI workflow on Kubernetes. Give actionable "
        "checks and mitigations.",
    ),
    (
        "validation-checklist",
        "Write an evidence-driven validation checklist for a completed Nebius "
        "Physical AI workflow: durable artifacts, stage status, logs, and a "
        "safe handoff to an operator.",
    ),
)


def deployment_review_prompt_records() -> list[dict[str, str]]:
    """Return the concrete prompt records consumed by Token Factory.

    Args:
        None.

    Returns:
        JSON-compatible prompt records with stable IDs and non-empty prompts.

    Raises:
        None.
    """

    return [{"id": prompt_id, "prompt": prompt} for prompt_id, prompt in DEPLOYMENT_REVIEW_PROMPTS]


def render_deployment_review_prompts() -> str:
    """Serialize deployment-review prompts as Token Factory JSONL.

    Args:
        None.

    Returns:
        Newline-terminated JSONL payload accepted by the generate tool.

    Raises:
        None.
    """

    return "".join(
        json.dumps(record, sort_keys=True) + "\n"
        for record in deployment_review_prompt_records()
    )


def write_deployment_review_prompts(
    output_uri: str, *, storage_client: Any | None = None
) -> dict[str, Any]:
    """Write the deployment-review prompt JSONL to a durable or local destination.

    Args:
        output_uri: Exact local path or S3 URI for the prompt JSONL artifact.
        storage_client: Optional storage client used by tests or remote execution.

    Returns:
        Artifact metadata including the schema, record count, and written URI.

    Raises:
        ValueError: If ``output_uri`` is empty.
    """

    target = output_uri.strip()
    if not target:
        raise ValueError("--output-uri is required")
    payload = render_deployment_review_prompts()
    written = _write_prompt_payload(payload, target, storage_client=storage_client)
    return {"schema": PROMPT_SCHEMA, "prompt_count": len(DEPLOYMENT_REVIEW_PROMPTS), "output_uri": written}


def _write_prompt_payload(
    payload: str, target: str, *, storage_client: Any | None
) -> str:
    if not target.startswith("s3://"):
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        return str(destination)
    from npa.clients.storage import StorageClient

    storage = storage_client or StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-deployment-prompts-") as temporary:
        source = Path(temporary) / "prompts.jsonl"
        source.write_text(payload, encoding="utf-8")
        return str(storage.upload_file(str(source), target))


def build_parser() -> argparse.ArgumentParser:
    """Build the module CLI parser.

    Args:
        None.

    Returns:
        Parser for the durable prompt-preparation stage.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the prompt-preparation stage.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit status.

    Raises:
        None.
    """

    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(write_deployment_review_prompts(args.output_uri), sort_keys=True))
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPLOYMENT_REVIEW_PROMPTS",
    "PROMPT_SCHEMA",
    "deployment_review_prompt_records",
    "render_deployment_review_prompts",
    "write_deployment_review_prompts",
]
