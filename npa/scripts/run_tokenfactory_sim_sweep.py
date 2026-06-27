#!/usr/bin/env python3
"""Design a sweep with Token Factory, run it on Nebius GPUs, then rank it.

This is a *combo* workflow that exercises all three layers in one run:

  1. **Token Factory (hosted, zero-GPU)** designs the sweep: a text model writes
     a per-variant hypothesis for the experiment grid.
  2. **Nebius AI Cloud (serverless GPU)** runs the sweep: one LeRobot serverless
     GPU Job per variant (``--smoke`` by default), each writing artifacts to S3.
  3. **Token Factory (hosted, zero-GPU)** ranks the sweep: a text model reads the
     completed runs' real artifacts and produces a best-to-worst ranking with a
     promoted winner.

The actual hyper-parameters launched on GPUs come from a deterministic grid
(:func:`npa.workflows.token_factory_combos.sweep_variants`), so the GPU stage
never depends on parsing free-form model output. The model contributes the
experiment *rationale* (stage 1) and the *judgement* (stage 3).

Examples:
  # No-infrastructure preview of the full plan (design prompt + grid + commands).
  python npa/scripts/run_tokenfactory_sim_sweep.py --render-only --num-variants 2

  # Full live run: design -> N serverless GPU smoke trains -> ranking.
  NEBIUS_TOKEN_FACTORY_KEY=... python npa/scripts/run_tokenfactory_sim_sweep.py \
      --project-id project-xxxxxxxx \
      --bucket s3://your-bucket/tf-sim-sweep \
      --num-variants 2

  # Cheap iteration: skip the GPU stage and only rank existing run prefixes.
  NEBIUS_TOKEN_FACTORY_KEY=... python npa/scripts/run_tokenfactory_sim_sweep.py \
      --rank-existing s3://your-bucket/runA/,s3://your-bucket/runB/
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
    DEFAULT_SWEEP_DESIGN_SYSTEM_PROMPT,
    DEFAULT_SWEEP_RANKING_SYSTEM_PROMPT,
    build_ranking_prompt,
    build_sweep_design_prompt,
    default_sweep_run_id,
    join_uri,
    render_triage_prompts_jsonl,
    summarize_run_artifacts,
    sweep_variant_output_uri,
    sweep_variants,
    triage_job_name,
)


class SweepRunError(RuntimeError):
    """Raised when a serverless or Token Factory stage fails."""


# ``npa.cli.main`` has no ``__main__`` guard; invoke the Typer app entrypoint
# directly so this runner uses the same npa package it was imported from.
_NPA_CLI_BOOTSTRAP = "from npa.cli.main import app_entry; app_entry()"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_plan(args)
    if args.render_only:
        print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
        return 0
    try:
        return _run(args, plan)
    except SweepRunError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 1


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the sweep plan without touching infrastructure (render-only safe)."""

    run_id = args.run_id or default_sweep_run_id()
    plan: dict[str, Any] = {
        "run_id": run_id,
        "compute": "nebius-serverless-gpu",
        "hosted_inference": "nebius-token-factory",
        "design_model": args.model or DEFAULT_TEXT_MODEL,
        "ranking_model": args.model or DEFAULT_TEXT_MODEL,
    }
    if args.rank_existing:
        plan["mode"] = "rank-existing"
        plan["variant_uris"] = _split_csv(args.rank_existing)
        return plan

    plan["mode"] = "full-sweep"
    sweep_root = join_uri(args.bucket, run_id) if args.bucket else f"<bucket>/{run_id}"
    variants = sweep_variants(args.num_variants)
    plan["sweep_root"] = sweep_root
    plan["variants"] = [
        {
            **variant,
            "output_uri": sweep_variant_output_uri(sweep_root, variant["id"]),
            "train_command": _train_command(args, variant, sweep_root, run_id),
        }
        for variant in variants
    ]
    plan["design_prompt"] = build_sweep_design_prompt(
        objective=args.objective,
        dataset=args.dataset,
        policy_type=args.policy_type,
        variants=variants,
    )
    return plan


def _train_command(
    args: argparse.Namespace, variant: dict[str, Any], sweep_root: str, run_id: str
) -> list[str]:
    # Derive the per-variant Job name from the *resolved* run_id (which is
    # timestamped when --run-id is omitted), not the raw arg. Otherwise two
    # sweeps launched without --run-id submit colliding serverless Job names
    # even though their S3 output prefixes differ.
    job_name = triage_job_name(f"{run_id}-{variant['id']}")
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
        "--steps",
        str(variant["steps"]),
        "--output-path",
        sweep_variant_output_uri(sweep_root, variant["id"]) + "/",
        "--wait-timeout",
        str(args.wait_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--output",
        "json",
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.project_id:
        cmd += ["--project-id", args.project_id]
    if args.image:
        cmd += ["--image", args.image]
    return cmd


def _run(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    _hydrate_credentials()
    summary: dict[str, Any] = {"status": "running", "plan_mode": plan["mode"], "run_id": plan["run_id"]}

    if plan["mode"] == "rank-existing":
        completed = _label_existing_runs(plan["variant_uris"])
        default_rank_base = plan["variant_uris"][0]
    else:
        summary["design"] = _design_sweep(plan, model=plan["design_model"], max_tokens=args.max_tokens)
        completed = _launch_variants(plan["variants"])
        summary["variants"] = completed
        default_rank_base = plan["sweep_root"]

    ranking = _rank_runs(
        objective=args.objective,
        runs=completed,
        rank_root=args.rank_root or join_uri(default_rank_base, "ranking"),
        model=plan["ranking_model"],
        max_tokens=args.max_tokens,
    )
    summary["ranking"] = ranking
    summary["status"] = "completed"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _design_sweep(plan: dict[str, Any], *, model: str, max_tokens: int) -> dict[str, Any]:
    from npa.workbench.token_factory import generate_text

    with tempfile.TemporaryDirectory(prefix="npa-tf-sweep-design-") as tmp:
        prompts_file = Path(tmp) / "prompts.jsonl"
        prompts_file.write_text(
            render_triage_prompts_jsonl([{"id": "sweep-design", "prompt": plan["design_prompt"]}]),
            encoding="utf-8",
        )
        result = generate_text(
            input_path=str(prompts_file),
            output_path=join_uri(plan["sweep_root"], "design"),
            model=model,
            system_prompt=DEFAULT_SWEEP_DESIGN_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    text = result.generations[0].completion if result.generations else ""
    return {"status": result.status, "model": result.model, "design_preview": text[:600]}


def _launch_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for variant in variants:
        train_result = _submit_serverless_train(variant["train_command"])
        if train_result.get("status") not in {"succeeded", "success"}:
            raise SweepRunError(f"variant {variant['id']} train failed: {train_result}")
        completed.append(
            {
                "id": variant["id"],
                "steps": variant["steps"],
                "uri": train_result.get("output_path") or variant["output_uri"],
                "status": train_result.get("status"),
            }
        )
    return completed


def _rank_runs(
    *,
    objective: str,
    runs: list[dict[str, Any]],
    rank_root: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    from npa.workbench.token_factory import generate_text

    with tempfile.TemporaryDirectory(prefix="npa-tf-sweep-rank-") as tmp:
        tmp_path = Path(tmp)
        digested: list[dict[str, str]] = []
        for run in runs:
            local = _materialize(run["uri"], tmp_path / run["id"])
            digested.append({"id": run["id"], "uri": run["uri"], "digest": summarize_run_artifacts(local)})
        prompt = build_ranking_prompt(objective=objective, runs=digested)
        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(
            render_triage_prompts_jsonl([{"id": "sweep-ranking", "prompt": prompt}]),
            encoding="utf-8",
        )
        result = generate_text(
            input_path=str(prompts_file),
            output_path=rank_root,
            model=model,
            system_prompt=DEFAULT_SWEEP_RANKING_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    text = result.generations[0].completion if result.generations else ""
    return {
        "status": result.status,
        "model": result.model,
        "report_uri": result.result_uri,
        "ranked_variants": [run["id"] for run in runs],
        "report_preview": text[:800],
    }


def _submit_serverless_train(train_command: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-c", _NPA_CLI_BOOTSTRAP, *train_command]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SweepRunError(
            "serverless lerobot train failed "
            f"(exit {proc.returncode}): {proc.stderr.strip()[-800:] or proc.stdout.strip()[-800:]}"
        )
    return _parse_last_json(proc.stdout) or {"status": "succeeded", "raw_stdout": proc.stdout[-400:]}


def _materialize(uri: str, dest: Path) -> Path:
    if not uri.startswith("s3://"):
        return Path(uri)
    from npa.clients.storage import StorageClient

    dest.mkdir(parents=True, exist_ok=True)
    local = StorageClient.from_environment().download_path(uri, str(dest))
    return Path(local)


def _hydrate_credentials() -> None:
    """Export ~/.npa/credentials.yaml into the environment for S3 + subprocess.

    The CLI does this on startup, but this runner is invoked as a plain script,
    so the serverless subprocess and StorageClient need the credentials exported
    here. Existing environment values win.
    """

    try:
        import os

        from npa.clients.credentials import apply_shared_credential_env, load_credentials

        apply_shared_credential_env(os.environ, load_credentials())
    except Exception:  # noqa: BLE001 - best-effort; live calls surface real errors.
        pass


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


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _uri_label(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1] or uri


def _label_existing_runs(uris: list[str]) -> list[dict[str, str]]:
    """Assign a unique, human-readable id to each existing run prefix.

    Last path segments often collide (e.g. ``.../pretrained_model``), so any
    label shared by more than one prefix is suffixed with a positional index.
    """

    labels = [_uri_label(uri) for uri in uris]
    runs: list[dict[str, str]] = []
    for index, (uri, label) in enumerate(zip(uris, labels)):
        unique = label if labels.count(label) == 1 else f"{label}-{index}"
        runs.append({"id": unique, "uri": uri})
    return runs


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default="", help="Run ID; defaults to a timestamped value.")
    parser.add_argument("--objective", default="Maximize PushT success while keeping training stable.", help="Sweep objective fed to the design + ranking models.")
    parser.add_argument("--num-variants", type=int, default=2, help="Number of GPU variants to launch (clamped to the seed grid).")
    parser.add_argument("--policy-type", default="act", help="LeRobot policy type for each GPU train Job.")
    parser.add_argument("--dataset", default="lerobot/pusht", help="Public HF dataset repo ID for the train Jobs.")
    parser.add_argument("--gpu-type", default="h200", help="Serverless GPU type (h200, b300, l40s, ...).")
    parser.add_argument("--smoke", action="store_true", default=True, help="Use smoke training settings (default).")
    parser.add_argument("--no-smoke", dest="smoke", action="store_false", help="Disable smoke settings (real run).")
    parser.add_argument("--project-id", default="", help="Nebius project ID override (auto-resolved if omitted).")
    parser.add_argument("--image", default="", help="Container image override for the serverless Jobs.")
    parser.add_argument("--bucket", default="", help="S3 prefix for sweep artifacts, e.g. s3://bucket/tf-sim-sweep.")
    parser.add_argument("--rank-root", default="", help="S3 URI for the ranking report (default: <sweep-root>/ranking/).")
    parser.add_argument("--model", default="", help=f"Token Factory text model for design + ranking (default {DEFAULT_TEXT_MODEL}).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for design + ranking generations.")
    parser.add_argument(
        "--rank-existing",
        default="",
        help="Comma-separated existing artifact prefixes to rank (skips design + GPU stages).",
    )
    parser.add_argument("--wait-timeout", type=int, default=3600, help="Max seconds to wait for each Job.")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between Job status checks.")
    parser.add_argument("--render-only", action="store_true", help="Print the plan and exit (no infrastructure).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
