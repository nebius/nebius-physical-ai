"""Executable Token Factory triage stage: run artifacts in, a written report out.

`tokenfactory-train-triage.yaml` did this in ~45 lines of inline bash and python: list the
run's textual artifacts on S3, download them, call the pure helpers in
:mod:`npa.workflows.token_factory_combos`, write ``prompts.jsonl`` and ``system_prompt.txt`` to
a temp directory, then invoke ``npa workbench token-factory generate`` with
``--system-prompt "$(cat …)"``.

That shell substitution is precisely what a `toolRef` argv cannot express, so the stage moves
here as one command. The pure helpers stay pure: `token_factory_combos` documents that it holds
no network, storage or Token Factory calls, and this module is where those live.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workflows.token_factory_combos import (
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    render_triage_prompts_jsonl,
    summarize_run_artifacts,
    triage_prompt_record,
    triage_prompts_uri,
    triage_report_uri,
)

#: Suffixes worth sending to a text model. A checkpoint or a video tells it nothing, and the
#: template used exactly this set.
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"})

DEFAULT_MAX_TOKENS = 900


class TriageError(RuntimeError):
    """Raised when the triage stage cannot produce a report."""


def download_textual_artifacts(artifacts_uri: str, dest: Path, *, storage_client: Any = None) -> list[str]:
    """Download every textual artifact under ``artifacts_uri`` into ``dest``.

    Returns the relative paths fetched, so a caller can tell "no artifacts" from "no text
    artifacts" — the first is a broken upstream stage, the second only a quiet one.
    """

    if not artifacts_uri.startswith("s3://"):
        source = Path(artifacts_uri)
        if not source.is_dir():
            raise TriageError(f"artifacts path is not a directory: {artifacts_uri}")
        fetched: list[str] = []
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                target = dest / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                fetched.append(str(path.relative_to(source)))
        return fetched

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    parsed = urlparse(artifacts_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    fetched = []
    paginator = client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or ():
            key = str(obj["Key"])
            if Path(key).suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = key[len(prefix) :].lstrip("/")
            if not relative:
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.s3.download_file(bucket, key, str(target))
            fetched.append(relative)
    return fetched


def run_triage(
    *,
    artifacts_uri: str,
    triage_uri: str,
    job_name: str,
    model: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str = DEFAULT_TRIAGE_SYSTEM_PROMPT,
    storage_client: Any = None,
    generate: Any = None,
) -> dict[str, Any]:
    """Digest a run's artifacts, then have a hosted text model write the triage report.

    ``generate`` is injectable so this is unit-testable without Token Factory; by default it is
    the real ``npa.workbench.token_factory.generate_text``.
    """

    if not artifacts_uri.strip():
        raise TriageError("--artifacts-uri is required")
    if not triage_uri.strip():
        raise TriageError("--triage-uri is required")

    with tempfile.TemporaryDirectory(prefix="npa-triage-") as tmp:
        work = Path(tmp)
        artifacts = work / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        fetched = download_textual_artifacts(
            artifacts_uri, artifacts, storage_client=storage_client
        )
        if not fetched:
            raise TriageError(
                f"no textual artifacts under {artifacts_uri}; the producing stage wrote nothing "
                f"a text model can read (looked for {sorted(TEXT_SUFFIXES)})"
            )
        digest = summarize_run_artifacts(artifacts)
        record = triage_prompt_record(
            job_name=job_name, output_uri=artifacts_uri, artifact_digest=digest
        )
        prompts_path = work / "prompts.jsonl"
        prompts_path.write_text(render_triage_prompts_jsonl([record]), encoding="utf-8")

        if generate is None:
            from npa.workbench.token_factory import generate_text as generate

        kwargs: dict[str, Any] = {
            "input_path": str(prompts_path),
            "output_path": triage_report_uri(triage_uri),
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
        }
        if model.strip():
            kwargs["model"] = model
        result = generate(**kwargs)

        # Publish the prompt next to the report: without it the report is unreviewable.
        prompts_written = _publish(
            prompts_path, triage_prompts_uri(triage_uri), storage_client=storage_client
        )

    generations = getattr(result, "generations", []) or []
    report_uri = getattr(result, "result_uri", triage_report_uri(triage_uri))
    written = _write_generations(generations, report_uri, storage_client=storage_client)
    return {
        "schema": "npa.token_factory.triage.v1",
        "status": "completed",
        "job_name": job_name,
        "artifacts_uri": artifacts_uri,
        "artifact_count": len(fetched),
        "model": getattr(result, "model", model),
        "prompt_count": getattr(result, "prompt_count", len(generations)),
        "prompts_uri": prompts_written,
        "report_uri": written,
    }


def _publish(local: Path, target_uri: str, *, storage_client: Any = None) -> str:
    if not target_uri.startswith("s3://"):
        destination = Path(target_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(local.read_text(encoding="utf-8"), encoding="utf-8")
        return str(destination)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    return client.upload_file(str(local), target_uri)


def _write_generations(generations: Any, report_uri: str, *, storage_client: Any = None) -> str:
    from dataclasses import asdict, is_dataclass

    from npa.workbench.token_factory import write_generations

    rows = [asdict(item) if is_dataclass(item) else dict(item) for item in generations]
    return write_generations(rows, result_uri=report_uri, storage_client=storage_client)


def build_parser() -> argparse.ArgumentParser:
    """Return this module's CLI parser (checked by the module-argv guardrail)."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Digest run artifacts and write a triage report.")
    run.add_argument("--artifacts-uri", required=True)
    run.add_argument("--triage-uri", required=True)
    run.add_argument("--job-name", default="tokenfactory-train-triage")
    run.add_argument("--model", default="")
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_triage(
            artifacts_uri=args.artifacts_uri,
            triage_uri=args.triage_uri,
            job_name=args.job_name,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except TriageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
