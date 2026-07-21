"""``npa workbench health`` — preflight diagnostics for workbench workflows.

Runs the recurring cold-start blockers as explicit PASS/WARN/FAIL/SKIP checks so
a customer hits them as a clear preflight instead of a mid-pipeline failure.
"""

from __future__ import annotations

import json as json_module
import os
import shutil
from pathlib import Path
from typing import Optional

import typer

from npa.clients.credentials import load_credentials
from npa.clients.huggingface import validate_hf_access
from npa.clients.kube import run_kubectl
from npa.clients.storage import StorageClient
from npa.guardrails.skypilot import inspect_image_exists
from npa.workflows.credential_preflight import (
    CREDENTIAL_CHECKS,
    CredentialProbes,
    run_credential_preflight,
)
from npa.workflows.sim2real_health import (
    ALL_CHECKS,
    DoctorProbes,
    FAIL,
    KubeResult,
    PASS,
    SKIP,
    WARN,
    has_failure,
    run_preflight,
)
from npa.workflows.sim2real_loop import build_config_from_env

app = typer.Typer(
    name="health",
    help="Preflight health checks for workbench workflows.",
    no_args_is_help=True,
)

_STATUS_ICON = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}


def _repo_root() -> Path:
    override = os.environ.get("NPA_REPO_ROOT")
    if override:
        return Path(override)
    # Walk up to the repo root that contains the workflow tree.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "npa" / "workflows" / "workbench").is_dir():
            return parent
    return Path.cwd()


def _image_inspector(image: str) -> bool | None:
    try:
        return inspect_image_exists(image)
    except RuntimeError:
        return None
    except Exception:  # noqa: BLE001 - any inspection error means "not verified pullable"
        return False


def _kube_runner_factory(context: str, kubeconfig: str):
    if not (os.environ.get("NPA_KUBECTL_BIN") or shutil.which("kubectl")):
        return None

    def _run(args: list[str]) -> KubeResult:
        # run_kubectl self-heals from a stale ambient NEBIUS_IAM_TOKEN that would
        # otherwise shadow the kubeconfig exec plugin and fail every call.
        result = run_kubectl(args, context=context, kubeconfig=kubeconfig, timeout=30)
        return KubeResult(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    return _run


def _emit_results(results, *, output_json: bool) -> None:
    """Render a list of CheckResult objects as text or JSON."""

    if output_json:
        payload = {
            "checks": [result.as_dict() for result in results],
            "ok": not has_failure(results),
        }
        typer.echo(json_module.dumps(payload, indent=2, sort_keys=True))
        return
    for result in results:
        typer.echo(
            f"[{_STATUS_ICON.get(result.status, result.status)}] "
            f"{result.name}: {result.summary}"
        )
        for detail in result.details:
            typer.echo(f"        - {detail}")
        if result.remedy and result.status in {FAIL, WARN, SKIP}:
            typer.echo(f"        fix: {result.remedy}")
    counts = {status: 0 for status in (PASS, WARN, FAIL, SKIP)}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    typer.echo(
        f"summary: {counts[PASS]} pass, {counts[WARN]} warn, "
        f"{counts[FAIL]} fail, {counts[SKIP]} skip"
    )


def _token_factory_verifier() -> list[str]:
    """Live Token Factory auth probe: resolve the key and list models."""

    from npa.clients.token_factory import TokenFactoryClient, resolve_config

    config = resolve_config(require_api_key=True)
    return TokenFactoryClient(config=config).list_models()


@app.command("preflight")
def preflight_command(
    checks: str = typer.Option(
        ",".join(CREDENTIAL_CHECKS),
        "--checks",
        help=(
            "Comma-separated checks to run, or 'all'. "
            f"Choices: all, {', '.join(CREDENTIAL_CHECKS)}."
        ),
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Skip live network probes (HF/S3/Token Factory); only check presence.",
    ),
    warn_only: bool = typer.Option(
        False, "--warn-only", help="Exit 0 even when a check fails."
    ),
    output_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Validate HF, NGC, S3, and Token Factory credentials before a deploy or GPU job.

    A single PASS/WARN/FAIL/SKIP report over the credentials nearly every
    workbench tool needs, so cold-start credential gaps surface here instead of
    mid-run. Exits non-zero on any FAIL unless ``--warn-only`` is passed.
    """

    selected = [item.strip() for item in checks.split(",") if item.strip()]
    if "all" in selected:
        selected = list(CREDENTIAL_CHECKS)
    unknown = [item for item in selected if item not in CREDENTIAL_CHECKS]
    if unknown:
        raise typer.BadParameter(
            f"unknown check(s): {', '.join(unknown)}. Choices: {', '.join(CREDENTIAL_CHECKS)}."
        )

    credentials = load_credentials()
    if offline:
        probes = CredentialProbes()
    else:
        probes = CredentialProbes(
            hf_validator=validate_hf_access,
            s3_client_factory=lambda: StorageClient.from_environment(),
            token_factory_verifier=_token_factory_verifier,
        )

    results = run_credential_preflight(credentials, probes=probes, checks=selected)
    _emit_results(results, output_json=output_json)

    if has_failure(results) and not warn_only:
        raise typer.Exit(code=1)


@app.command("sim2real", hidden=True)
def sim2real_command(
    run_id: str = typer.Option("sim2real-doctor", "--run-id", help="Run id for the probed config."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket for artifact upload."),
    s3_prefix: Optional[str] = typer.Option(None, "--s3-prefix", help="S3 prefix parent for this run."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="Non-default S3-compatible endpoint."),
    trigger_dataset_uri: str = typer.Option("", "--trigger-dataset-uri", help="Trigger dataset path."),
    trigger_dataset_id: str = typer.Option("", "--trigger-dataset-id", help="Source dataset id."),
    assets_uri: str = typer.Option("", "--assets-uri", help="BYO simulation asset source path."),
    scene_spec_uri: str = typer.Option("", "--scene-spec-uri", help="BYO SceneSpec path."),
    augment_image: str = typer.Option("", "--augment-image", help="BYO augmentation image."),
    policy_image: str = typer.Option("", "--policy-image", help="BYO policy image."),
    trainer_image: str = typer.Option("", "--trainer-image", help="BYO VLM-RL trainer image."),
    vlm_image: str = typer.Option("", "--vlm-image", help="BYO VLM image."),
    eval_image: str = typer.Option("", "--eval-image", help="BYO held-out eval image."),
    vlm_model: str = typer.Option("", "--vlm-model", help="VLM model id/name."),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Held-out success threshold."),
    inner_iterations: Optional[int] = typer.Option(None, "--inner-iterations", help="Inner-loop cap."),
    outer_iterations: Optional[int] = typer.Option(None, "--outer-iterations", help="Outer-loop cap."),
    loop_of_loops_iterations: Optional[int] = typer.Option(
        None, "--loop-of-loops-iterations", help="Loop-of-loops cap."
    ),
    rollout_count: Optional[int] = typer.Option(None, "--rollout-count", help="Train rollout count."),
    steps_per_rollout: Optional[int] = typer.Option(None, "--steps-per-rollout", help="Steps per rollout."),
    heldout_env_count: Optional[int] = typer.Option(None, "--heldout-env-count", help="Held-out env count."),
    k8s_namespace: str = typer.Option("", "--k8s-namespace", help="Namespace for sibling Jobs."),
    k8s_context: str = typer.Option("", "--k8s-context", help="Kube context to pin the check to."),
    k8s_kubeconfig: str = typer.Option("", "--k8s-kubeconfig", help="Explicit kubeconfig path."),
    checks: str = typer.Option(
        ",".join(ALL_CHECKS),
        "--checks",
        help=(
            "Comma-separated checks to run, or 'all'. "
            f"Choices: all, {', '.join(ALL_CHECKS)}."
        ),
    ),
    warn_only: bool = typer.Option(
        False, "--warn-only", help="Exit 0 even when a check fails."
    ),
    output_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Validate a sim2real config and check the recurring blockers up front.

    Deprecated: use ``npa workbench workflow submit`` on the sim2real runbook for
    preflight and ``npa workbench workflow status <run-id>`` for live progress.
    """

    overrides: dict[str, object] = {
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_endpoint": s3_endpoint,
        "trigger_dataset_uri": trigger_dataset_uri,
        "trigger_dataset_id": trigger_dataset_id,
        "assets_uri": assets_uri,
        "scene_spec_uri": scene_spec_uri,
        "augment_image": augment_image,
        "policy_image": policy_image,
        "trainer_image": trainer_image,
        "vlm_image": vlm_image,
        "eval_image": eval_image,
        "vlm_model": vlm_model,
        "k8s_namespace": k8s_namespace,
        "k8s_context": k8s_context,
        "k8s_kubeconfig": k8s_kubeconfig,
    }
    if s3_prefix is not None:
        overrides["s3_prefix"] = s3_prefix
    for key, value in (
        ("threshold", threshold),
        ("inner_iterations", inner_iterations),
        ("outer_iterations", outer_iterations),
        ("loop_of_loops_iterations", loop_of_loops_iterations),
        ("rollout_count", rollout_count),
        ("steps_per_rollout", steps_per_rollout),
        ("heldout_env_count", heldout_env_count),
    ):
        if value is not None:
            overrides[key] = value

    config = build_config_from_env(**overrides)
    credentials = load_credentials()

    selected = [item.strip() for item in checks.split(",") if item.strip()]
    # 'all' is the documented shorthand (operator runbooks and the 10-min demo
    # script use `--checks all`) — expand it to the full check set.
    if "all" in selected:
        selected = list(ALL_CHECKS)
    unknown = [item for item in selected if item not in ALL_CHECKS]
    if unknown:
        raise typer.BadParameter(
            f"unknown check(s): {', '.join(unknown)}. Choices: {', '.join(ALL_CHECKS)}."
        )

    probes = DoctorProbes(
        s3_client_factory=lambda: StorageClient.from_environment(
            endpoint_url=config.s3_endpoint
        ),
        image_inspector=_image_inspector,
        credentials=credentials,
        kube_runner=_kube_runner_factory(config.k8s_context, config.k8s_kubeconfig),
    )

    results = run_preflight(
        config, repo_root=_repo_root(), probes=probes, checks=selected
    )

    if output_json:
        payload = {
            "run_id": config.run_id,
            "checks": [result.as_dict() for result in results],
            "ok": not has_failure(results),
        }
        typer.echo(json_module.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            typer.echo(f"[{_STATUS_ICON.get(result.status, result.status)}] {result.name}: {result.summary}")
            for detail in result.details:
                typer.echo(f"        - {detail}")
            if result.remedy and result.status in {FAIL, WARN, SKIP}:
                typer.echo(f"        fix: {result.remedy}")
        counts = {status: 0 for status in (PASS, WARN, FAIL, SKIP)}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        typer.echo(
            f"summary: {counts[PASS]} pass, {counts[WARN]} warn, "
            f"{counts[FAIL]} fail, {counts[SKIP]} skip"
        )

    if has_failure(results) and not warn_only:
        raise typer.Exit(code=1)
