#!/usr/bin/env python3
"""Train on a Nebius serverless GPU, then triage the run with Token Factory.

This is a *combo* workflow: it uses **real Nebius cloud compute** (a LeRobot
serverless GPU Job via ``npa workbench lerobot train --runtime serverless``) and
**hosted Token Factory inference** (a text model reads the run's artifacts and
writes a triage + next-steps report). The GPU half produces the artifacts; the
zero-GPU hosted half explains them.

Stages:
  1. Submit a LeRobot serverless GPU Job (``--smoke`` by default) and wait for it
     to finish. The Job uploads its run artifacts (configs, logs, metrics) to S3.
  2. Download those textual artifacts, build a triage prompt, and call Token
     Factory ``generate`` to write a human-readable report next to the run.

Examples:
  # No-infrastructure preview of what would run.
  python npa/scripts/run_tokenfactory_train_triage.py --render-only

  # Full live run: serverless GPU smoke train + Token Factory triage.
  NEBIUS_TOKEN_FACTORY_KEY=... python npa/scripts/run_tokenfactory_train_triage.py

  # Skip the GPU stage and only triage an existing run prefix (cheap iteration).
  NEBIUS_TOKEN_FACTORY_KEY=... python npa/scripts/run_tokenfactory_train_triage.py \
      --from-output-path s3://my-bucket/lerobot-serverless-test/<ts>/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from npa.clients.token_factory import DEFAULT_TEXT_MODEL
from npa.workflows.token_factory_combos import (
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    build_triage_prompt,
    default_triage_run_id,
    join_uri,
    render_triage_prompts_jsonl,
    summarize_run_artifacts,
    triage_job_name,
    triage_report_uri,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_plan(args)
    if args.render_only:
        print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
        return 0
    try:
        return _run(args, plan)
    except TriageRunError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 1


class TriageRunError(RuntimeError):
    """Raised when the serverless or Token Factory stage fails."""


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the run plan without touching infrastructure (render-only safe)."""

    run_id = args.run_id or default_triage_run_id()
    job_name = args.job_name or triage_job_name(run_id)
    plan: dict[str, Any] = {
        "run_id": run_id,
        "compute": "nebius-serverless-gpu",
        "hosted_inference": "nebius-token-factory",
        "triage_model": args.model or DEFAULT_TEXT_MODEL,
        "skip_train": bool(args.from_output_path),
    }
    if args.from_output_path:
        plan["artifacts_uri"] = args.from_output_path
        plan["triage_root"] = args.triage_root or join_uri(args.from_output_path, "triage")
    else:
        plan["train_command"] = _train_command(args, job_name)
        plan["job_name"] = job_name
        plan["triage_root_template"] = "<run output-path>/triage/ (resolved after the Job completes)"
        if args.triage_root:
            plan["triage_root"] = args.triage_root
    return plan


def _train_command(args: argparse.Namespace, job_name: str) -> list[str]:
    cmd = [
        "workbench",
        "lerobot",
        "train",
        "--runtime",
        "serverless",
        "--policy-type",
        args.policy_type,
        "--dataset",
        args.dataset,
        "--job-name",
        job_name,
        "--gpu-type",
        args.gpu_type,
        "--wait-timeout",
        str(args.wait_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--output",
        "json",
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.steps:
        cmd += ["--steps", str(args.steps)]
    if args.project_id:
        cmd += ["--project-id", args.project_id]
    if args.image:
        cmd += ["--image", args.image]
    if args.output_path:
        cmd += ["--output-path", args.output_path]
    return cmd


def _hydrate_credentials() -> None:
    """Export ~/.npa/credentials.yaml into the environment for S3 + subprocess.

    The CLI does this on startup, but this runner is invoked as a plain script,
    so the serverless subprocess and StorageClient need the credentials exported
    here. Existing environment values win.
    """

    try:
        import os

        from npa.clients.credentials import load_credentials, shared_credential_env

        for key, value in shared_credential_env(load_credentials()).items():
            if value:
                # This workflow chains cloud + hosted stages; prefer canonical
                # credentials from ~/.npa/credentials.yaml over inherited shell env.
                os.environ[key] = value
    except Exception:  # noqa: BLE001 - best-effort; live calls surface real errors.
        pass


def _run(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    _hydrate_credentials()
    summary: dict[str, Any] = {"status": "running", "plan": plan}

    if args.from_output_path:
        artifacts_uri = args.from_output_path
        triage_root = plan["triage_root"]
    else:
        train_result = _submit_serverless_train(plan["train_command"])
        summary["train"] = train_result
        if train_result.get("status") not in {"succeeded", "success"}:
            summary["status"] = "failed"
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 1
        artifacts_uri = train_result.get("output_path") or ""
        if not artifacts_uri:
            raise TriageRunError("serverless train did not report an output_path to triage")
        triage_root = args.triage_root or join_uri(artifacts_uri, "triage")

    summary["artifacts_uri"] = artifacts_uri
    summary["triage_root"] = triage_root

    triage = _triage_artifacts(
        artifacts_uri=artifacts_uri,
        triage_root=triage_root,
        job_name=plan.get("job_name", plan["run_id"]),
        model=plan["triage_model"],
        max_tokens=args.max_tokens,
        extra_context=args.notes,
    )
    summary["triage"] = triage
    summary["status"] = "completed"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# ``npa.cli.main`` has no ``__main__`` guard; invoke the Typer app entrypoint
# directly so this runner uses the same npa package it was imported from.
_NPA_CLI_BOOTSTRAP = "from npa.cli.main import app_entry; app_entry()"


def _submit_serverless_train(train_command: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-c", _NPA_CLI_BOOTSTRAP, *train_command]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise TriageRunError(
            "serverless lerobot train failed "
            f"(exit {proc.returncode}): {proc.stderr.strip()[-800:] or proc.stdout.strip()[-800:]}"
        )
    return _parse_last_json(proc.stdout) or {"status": "succeeded", "raw_stdout": proc.stdout[-400:]}


def _triage_artifacts(
    *,
    artifacts_uri: str,
    triage_root: str,
    job_name: str,
    model: str,
    max_tokens: int,
    extra_context: str,
) -> dict[str, Any]:
    from npa.workbench.token_factory import generate_text

    with tempfile.TemporaryDirectory(prefix="npa-tf-triage-") as tmp:
        tmp_path = Path(tmp)
        local_artifacts = _materialize(artifacts_uri, tmp_path / "artifacts")
        digest = summarize_run_artifacts(local_artifacts)
        prompt = build_triage_prompt(
            job_name=job_name,
            output_uri=artifacts_uri,
            artifact_digest=digest,
            extra_context=extra_context,
        )
        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(
            render_triage_prompts_jsonl([{"id": f"triage-{triage_job_name(job_name)}", "prompt": prompt}]),
            encoding="utf-8",
        )
        result = generate_text(
            input_path=str(prompts_file),
            output_path=triage_root,
            model=model,
            system_prompt=DEFAULT_TRIAGE_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        report_text = result.generations[0].completion if result.generations else ""

    return {
        "status": result.status,
        "model": result.model,
        "report_uri": triage_report_uri(triage_root)
        if not triage_root.endswith((".json", ".jsonl"))
        else result.result_uri,
        "report_preview": report_text[:600],
    }


def _materialize(uri: str, dest: Path) -> Path:
    if not uri.startswith("s3://"):
        return Path(uri)
    from npa.clients.storage import StorageClient

    dest.mkdir(parents=True, exist_ok=True)
    local = StorageClient.from_environment().download_path(uri, str(dest))
    return Path(local)


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="", help="Run ID; defaults to a timestamped value.")
    parser.add_argument("--policy-type", default="act", help="LeRobot policy type for the GPU train Job.")
    parser.add_argument("--dataset", default="lerobot/pusht", help="Public HF dataset repo ID for the train Job.")
    parser.add_argument("--job-name", default="", help="Serverless Job name; derived from --run-id if omitted.")
    parser.add_argument("--gpu-type", default="h200", help="Serverless GPU type (h200, b300, l40s, ...).")
    parser.add_argument("--project-id", default="", help="Nebius project ID override (auto-resolved if omitted).")
    parser.add_argument("--image", default="", help="Container image override for the serverless Job.")
    parser.add_argument("--output-path", default="", help="S3 URI for train artifacts (auto-derived if omitted).")
    parser.add_argument("--steps", type=int, default=0, help="Training steps (0 = tool default; --smoke overrides).")
    parser.add_argument("--smoke", action="store_true", default=True, help="Use smoke training settings (default).")
    parser.add_argument("--no-smoke", dest="smoke", action="store_false", help="Disable smoke settings (real run).")
    parser.add_argument("--model", default="", help=f"Token Factory triage model (default {DEFAULT_TEXT_MODEL}).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for the triage report.")
    parser.add_argument("--notes", default="", help="Optional operator context added to the triage prompt.")
    parser.add_argument("--triage-root", default="", help="S3 URI for the triage report (default: <artifacts>/triage/).")
    parser.add_argument(
        "--from-output-path",
        default="",
        help="Skip the GPU Job and triage an existing artifacts S3 prefix instead.",
    )
    parser.add_argument("--wait-timeout", type=int, default=3600, help="Max seconds to wait for the Job.")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between Job status checks.")
    parser.add_argument("--render-only", action="store_true", help="Print the plan and exit (no infrastructure).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
