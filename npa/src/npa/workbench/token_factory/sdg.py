"""Build reviewed synthetic instruction datasets with automatic open-weight routing."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from npa.clients.credentials import load_credentials
from npa.clients.token_factory import TokenFactoryClient
from npa.workbench.token_factory import (
    TokenFactoryToolError,
    _materialized_input,
    _write_text,
)
from .sdg_protocol import (
    FAST_MODEL,
    REASONING_MODEL,
    PROMPT_REVISION,
    generate_candidate,
)

MODEL_CARDS = {
    FAST_MODEL: "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    REASONING_MODEL: "https://huggingface.co/MiniMaxAI/MiniMax-M3",
}


class SdgRequest(BaseModel):
    """Configure the hosted text SDG pipeline without imposing a record limit.

    Args:
        input_path: JSONL seed file, local for SDK development or an S3 URI.
        output_path: Artifact directory or S3 prefix for one run.
        context_path: Optional shared UTF-8 reference file, local or S3.
        router: Hosted open-weight router by default, or optional TypeSafe Jev.
        dry_run: Validate inputs without inference or output writes.
    Returns:
        Validated pipeline request.
    Raises:
        ValueError: Required paths or router selection are invalid.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    input_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    context_path: str = ""
    router: Literal["token_factory", "jev"] = "token_factory"
    dry_run: bool = False


class _Seed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


def _digest(body):
    return hashlib.sha256(body.encode()).hexdigest()


def _read_text(path):
    with _materialized_input(path) as local:
        return local.read_text(encoding="utf-8")


def _seeds(body):
    result, seen = [], set()
    for number, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        try:
            seed = _Seed.model_validate_json(line).model_dump()
        except ValueError:
            raise TokenFactoryToolError(
                f"Invalid id/prompt seed at line {number}"
            ) from None
        if seed["id"] in seen:
            raise TokenFactoryToolError(f"Duplicate seed id at line {number}")
        seen.add(seed["id"])
        result.append(seed)
    if not result:
        raise TokenFactoryToolError("Seed input must contain at least one JSONL record")
    return result


def _jev_key(router):
    if router != "jev":
        return ""
    key = os.environ.get("TYPESAFE_API_KEY", "") or load_credentials().tokens.get(
        "TYPESAFE_API_KEY", ""
    )
    if not key.strip():
        raise TokenFactoryToolError("The jev router requires TYPESAFE_API_KEY")
    return key.strip()


def _deduplicate(records):
    seen = set()
    for record in records:
        if record["status"] != "accepted":
            continue
        canonical = [
            " ".join(record["pair"][key].casefold().split())
            for key in ("instruction", "response")
        ]
        digest = _digest(json.dumps(canonical))
        if digest in seen:
            record.update(status="rejected", reason="duplicate_instruction_answer")
        else:
            seen.add(digest)


def _usage_by_stage(records):
    groups = defaultdict(list)
    for record in records:
        for call in record["calls"]:
            groups[call["stage"], call["requested_model"]].append(call)
    result = []
    for (stage, model), calls in sorted(groups.items()):
        item = {"stage": stage, "model": model, "request_count": len(calls)}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
        ):
            values = [call["usage"].get(key) for call in calls]
            item[key] = (
                sum(values) if all(type(value) is int for value in values) else None
            )
        item["cache_counters_reported"] = sum(
            "cached_tokens" in call["usage"] for call in calls
        )
        result.append(item)
    return result


def _summary(request, records, seed_body, context):
    counts = Counter(record["status"] for record in records)
    return {
        "schema": "npa.token_factory.sdg.v1",
        "prompt_revision": PROMPT_REVISION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed"
        if counts["error"] or not counts["accepted"]
        else "completed",
        "router": request.router,
        "router_location": "workbench",
        "seed_count": len(records),
        "accepted_count": counts["accepted"],
        "rejected_count": counts["rejected"],
        "error_count": counts["error"],
        "input_sha256": _digest(seed_body),
        "context_sha256": _digest(context),
        "generation_model_counts": dict(
            Counter(
                record["served_model"] for record in records if record["served_model"]
            )
        ),
        "routing_status_counts": dict(
            Counter(record["routing"]["status"] for record in records)
        ),
        "usage_by_stage": _usage_by_stage(records),
        "model_cards": MODEL_CARDS,
        "application_response_cache": False,
        "jev_usage": [record["routing"].get("usage", {}) for record in records]
        if request.router == "jev"
        else [],
    }


def _jsonl(records):
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _training_rows(records):
    return [
        {
            "id": record["id"],
            "messages": [
                {"role": "user", "content": record["pair"]["instruction"]},
                {"role": "assistant", "content": record["pair"]["response"]},
            ],
        }
        for record in records
        if record["status"] == "accepted"
    ]


def _write_artifacts(output_path, records, report):
    bodies = {
        "dataset.jsonl": _jsonl(_training_rows(records)),
        "rejected.jsonl": _jsonl(
            [record for record in records if record["status"] != "accepted"]
        ),
        "provenance.jsonl": _jsonl(records),
    }
    report["artifacts"] = {
        name: {"sha256": _digest(body)} for name, body in bodies.items()
    }
    # The manifest is written last so consumers can verify the completed artifact set.
    bodies["report.json"] = json.dumps(report, indent=2, sort_keys=True) + "\n"
    for filename, body in bodies.items():
        _write_text(
            body,
            result_uri=output_path.rstrip("/") + "/" + filename,
            filename=filename,
            storage_client=None,
        )


def run_sdg(request: SdgRequest, *, client: TokenFactoryClient | None = None) -> dict:
    """Route seeds to open-weight generators, review candidates, and export JSONL.

    Args:
        request: Input, output, reference context, and router configuration.
        client: Injectable Token Factory client; otherwise use private NPA credentials.
    Returns:
        Manifest with accepted/rejected/error counts, hashes, model routes, and usage.
    Raises:
        TokenFactoryToolError: Seeds, credentials, or account model availability are invalid.
        TokenFactoryError: Account-scoped model discovery fails.
        OSError: Input reading or artifact publication fails.
    """
    seed_body = _read_text(request.input_path)
    seeds = _seeds(seed_body)
    context = _read_text(request.context_path) if request.context_path else ""
    if request.dry_run:
        return {
            "status": "planned",
            "seed_count": len(seeds),
            "router": request.router,
            "models": list(MODEL_CARDS),
            "inference_performed": False,
        }
    key = _jev_key(request.router)
    active = client or TokenFactoryClient()
    if not set(MODEL_CARDS) <= set(active.list_models()):
        raise TokenFactoryToolError(
            "Both SDG open-weight models must be available to this Token Factory key"
        )
    records = [
        generate_candidate(active, seed, context, router=request.router, jev_key=key)
        for seed in seeds
    ]
    _deduplicate(records)
    report = _summary(request, records, seed_body, context)
    _write_artifacts(request.output_path, records, report)
    return report
