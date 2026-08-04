"""Typer CLI for `npa workbench detection-training`."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import typer

from npa.clients.config import resolve_container_registry
from npa.clients.credentials import load_credentials
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY, container_image_for_tool
from npa.workbench.detection_training.artifacts import (
    EVAL_METRICS_FILENAME,
    DetectionTrainingArtifactError,
    assert_eval_metrics,
    discover_checkpoint_uri,
    eval_result_uri_for,
)
from npa.workbench.detection_training.schemas import (
    DEFAULT_LANCE_URI,
    DEFAULT_PORT,
    DEFAULT_TOKEN_ENV,
    CheckpointS3Settings,
    EvalRequest,
    TrainRequest,
    WandbSettings,
)

app = typer.Typer(
    name="detection-training",
    help="Train Faster R-CNN detectors from LanceDB materialized views.",
    no_args_is_help=True,
)

DEFAULT_IMAGE = container_image_for_tool("detection-training", registry=DEFAULT_CONTAINER_REGISTRY)
DEFAULT_NAME = "npa-detection-training"
DEFAULT_NAMESPACE = "default"
#: `--gpu-type` shorthand -> the `node.kubernetes.io/instance-type` label to select on.
#: RTX PRO 6000 is here because it is what the workbench's own GPU cluster runs: without it a
#: deploy defaulted to an l40s selector no node carries, the pod stayed Unschedulable, and the
#: only symptom was `rollout status` timing out with nothing said about node labels
#: (EVIDENCE.md §R46). `--node-selector-value` still overrides for anything not listed.
GPU_NODE_SELECTORS = {
    "h100": "gpu-h100-sxm",
    "l40s": "gpu-l40s-d",
    "rtx6000": "gpu-rtx6000",
    "rtxpro6000": "gpu-rtx6000",
}


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text if text is not None else "\n".join(f"{key}: {value}" for key, value in payload.items()))


def deploy_cmd(
    project: str = typer.Option("", "--project", "-p", help="Project alias used to resolve container_registry."),
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help=(
            "NPA cluster profile whose cached kubeconfig to use. Empty (the default) uses the "
            "ambient kubeconfig, i.e. the cluster `kubectl` is already pointed at."
        ),
    ),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    image: str = typer.Option("", "--image", help=f"Container image to deploy. Defaults to {DEFAULT_IMAGE}."),
    name: str = typer.Option(DEFAULT_NAME, "--name", help="Kubernetes deployment/service name."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Service port."),
    input_path: str = typer.Option(DEFAULT_LANCE_URI, "--input-path", help="Default LanceDB input URI."),
    output_path: str = typer.Option("", "--output-path", help="Default S3 output URI."),
    gpu_type: str = typer.Option("h100", "--gpu-type", help="GPU type: h100 or l40s."),
    node_selector_key: str = typer.Option("node.kubernetes.io/instance-type", "--node-selector-key", help="GPU node selector label key."),
    node_selector_value: str = typer.Option("", "--node-selector-value", help="GPU node selector label value override."),
    image_pull_secret: str = typer.Option("npa-nebius-registry", "--image-pull-secret", help="Kubernetes imagePullSecret name for private registries."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    auth_mode: str = typer.Option("token", "--auth-mode", help="Auth mode: none or token. Defaults to token (secure)."),
    insecure_no_auth: bool = typer.Option(
        False,
        "--insecure-no-auth",
        help="Explicitly deploy without token auth (overrides --auth-mode to none). Not recommended.",
    ),
    destroy: bool = typer.Option(False, "--destroy", help="Delete the Kubernetes service, deployment, and secret."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print Kubernetes manifest without applying it."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy the detection-training service to an NPA Workbench Kubernetes cluster."""
    if port < 1024 or port > 65535:
        fail("--port must be between 1024 and 65535")
    if insecure_no_auth:
        auth_mode = "none"
    if auth_mode not in {"none", "token"}:
        fail("--auth-mode must be none or token")
    if not output_path and not destroy:
        fail("--output-path is required")
    resolved_kubeconfig = _resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig)
    if destroy:
        _kubectl(["delete", "service", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "deployment", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "secret", f"{name}-env", "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        emit({"status": "deleted", "name": name, "namespace": namespace}, output=output)
        return

    selector_value = node_selector_value.strip() or GPU_NODE_SELECTORS.get(gpu_type.strip().lower())
    if not selector_value:
        fail(
            "--gpu-type must be one of "
            + ", ".join(sorted(GPU_NODE_SELECTORS))
            + " unless --node-selector-value is provided"
        )
    resolved_image = image.strip() or container_image_for_tool(
        "detection-training",
        registry=resolve_container_registry(project or None),
    )
    manifest = _kubernetes_manifest(
        image=resolved_image,
        name=name,
        namespace=namespace,
        port=port,
        input_path=input_path,
        output_path=output_path,
        node_selector_key=node_selector_key,
        node_selector_value=selector_value,
        image_pull_secret=image_pull_secret,
        auth_mode=auth_mode,
        token_env=token_env,
    )
    if dry_run:
        typer.echo(json.dumps(_redact_manifest(manifest), indent=2, sort_keys=True))
        return
    if auth_mode == "none":
        typer.echo(
            "Warning: --auth-mode none deploys detection-training without token auth. The service "
            "drives GPU training and carries S3 credentials, and any pod in the cluster can reach it. "
            "Use --auth-mode token with DETECTION_TRAINING_TOKEN set.",
            err=True,
        )
    if image_pull_secret:
        _ensure_image_pull_secret(
            image=resolved_image,
            secret_name=image_pull_secret,
            namespace=namespace,
            kubeconfig=resolved_kubeconfig,
        )
    _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), kubeconfig=resolved_kubeconfig)
    _kubectl(["rollout", "status", f"deployment/{name}", "-n", namespace, "--timeout=900s"], kubeconfig=resolved_kubeconfig)
    endpoint = f"http://{name}.{namespace}.svc.cluster.local:{port}"
    emit(
        {
            "status": "deployed",
            "name": name,
            "namespace": namespace,
            "image": resolved_image,
            "endpoint": endpoint,
            "node_selector": {node_selector_key: selector_value},
        },
        output=output,
        text=f"Detection-training service deployed: {endpoint}",
    )


#: Statuses `/status` reports when a run is over.
TRAINING_DONE = "completed"
TRAINING_FAILED = "failed"


def parse_label_map(raw: str) -> dict[str, int] | None:
    """Parse ``--label-map`` from JSON or ``name=index`` pairs.

    The BDD100K pipeline template passed a full category map in its request body
    (``BDD100K_LABEL_MAP``), but ``label_map`` had no CLI flag and is not an accepted
    ``--override`` key, so the field was unreachable from the CLI — and therefore from any
    npa.workflow spec. Both spellings are accepted because the template's value is JSON
    while ``a=0,b=1`` is friendlier to type.
    """

    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            fail(f"--label-map is not valid JSON: {exc}")
            return None  # pragma: no cover - fail() raises
        if not isinstance(parsed, dict):
            fail("--label-map JSON must be an object of category -> index")
        pairs = list(parsed.items())
    else:
        pairs = []
        for chunk in text.split(","):
            name, sep, index = chunk.partition("=")
            if not sep or not name.strip():
                fail(f"--label-map entry must be name=index, got {chunk!r}")
            pairs.append((name.strip(), index.strip()))

    label_map: dict[str, int] = {}
    for name, index in pairs:
        try:
            label_map[str(name)] = int(index)
        except (TypeError, ValueError):
            fail(f"--label-map index for {name!r} must be an integer, got {index!r}")
    return label_map


def wait_for_training_run(
    run_id: str,
    *,
    endpoint: str,
    token_env: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Poll ``/status`` until the run completes, mirroring the retired template's loop.

    The BDD100K pipeline's train task POSTed ``/train`` and then polled ``/status`` in bash
    until ``completed``, failing on ``failed`` or timeout and finally asserting that every
    epoch ran and a checkpoint pattern was produced. Without that wait, ``train`` returns
    while training is still running and the next stage evaluates a checkpoint that does not
    exist yet — so the wait has to live in the tool, where a spec can reach it.
    """

    import time

    if not run_id:
        fail("service did not return a run_id to wait for")
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        status_payload = request_json(
            "GET",
            endpoint,
            "/status",
            params={"run_id": run_id},
            token_env=token_env,
            timeout=30.0,
        )
        status = str(status_payload.get("status") or "").strip().lower()
        if status == TRAINING_DONE:
            _assert_training_run_is_complete(status_payload)
            return status_payload
        if status == TRAINING_FAILED:
            typer.echo(json.dumps(status_payload, indent=2, sort_keys=True), err=True)
            fail(f"detection-training run {run_id} failed")
        if time.monotonic() >= deadline:
            typer.echo(json.dumps(status_payload, indent=2, sort_keys=True), err=True)
            fail(f"detection-training run {run_id} did not complete within {timeout_seconds:g}s")
        time.sleep(max(poll_seconds, 0.0))


def _assert_training_run_is_complete(payload: dict[str, Any]) -> None:
    """The template's closing `jq -e` assertion: all epochs ran and a checkpoint exists."""

    completed = payload.get("epochs_completed")
    total = payload.get("total_epochs")
    if completed != total:
        fail(f"training reported completed after {completed}/{total} epochs")
    if not str(payload.get("checkpoint_uri_pattern") or "").strip():
        fail("training completed without a checkpoint_uri_pattern")


def train_cmd(
    view: str = typer.Option(..., "--view", help="Lance materialized view name."),
    output_uri: str = typer.Option("", "--output-uri", "--output-path", help="S3/local output URI."),
    data_path: str = typer.Option("", "--data-path", help="Custom LanceDB training data URI."),
    lance_uri: str = typer.Option(DEFAULT_LANCE_URI, "--lance-uri", "--input-path", help="Compatibility alias for --data-path."),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Generic override as KEY=VALUE. Supported keys map to detection-training request fields.",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb/--no-wandb", help="Enable W&B logging for the training run."),
    wandb_project: str = typer.Option("", "--wandb-project", help="W&B project name."),
    wandb_run_name: str = typer.Option("", "--wandb-run-name", help="W&B run name."),
    wandb_mode: str = typer.Option("offline", "--wandb-mode", help="W&B mode such as online, offline, or disabled."),
    checkpoint_s3_uri: str = typer.Option("", "--checkpoint-s3-uri", help="S3 URI for checkpoint upload."),
    checkpoint_s3_endpoint_url: str = typer.Option("", "--checkpoint-s3-endpoint-url", help="S3-compatible endpoint URL."),
    checkpoint_s3_access_key_id: str = typer.Option("", "--checkpoint-s3-access-key-id", help="S3 access key ID."),
    checkpoint_s3_secret_access_key: str = typer.Option("", "--checkpoint-s3-secret-access-key", help="S3 secret access key."),
    num_classes: int = typer.Option(10, "--num-classes", help="Detector class count."),
    label_map: str = typer.Option(
        "",
        "--label-map",
        help='Category-to-index map as JSON ({"person":0,...}) or "person=0,rider=1".',
    ),
    epochs: int = typer.Option(10, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size."),
    learning_rate: float = typer.Option(0.005, "--learning-rate", help="SGD learning rate."),
    validation_filter_sql: str = typer.Option("", "--validation-filter-sql", help="Optional validation filter SQL."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help="Poll /status until the run completes, and fail if it does not.",
    ),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds", help="Interval between --wait polls."),
    timeout_seconds: float = typer.Option(
        21600.0, "--timeout-seconds", help="Give up waiting after this many seconds."
    ),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Start a detection-training run."""
    checkpoint_s3 = CheckpointS3Settings(
        uri=checkpoint_s3_uri,
        endpoint_url=checkpoint_s3_endpoint_url,
        aws_access_key_id=checkpoint_s3_access_key_id,
        aws_secret_access_key=checkpoint_s3_secret_access_key,
    )
    effective_output_uri = output_uri or checkpoint_s3.uri
    if not effective_output_uri:
        fail("--output-uri or --checkpoint-s3-uri is required")
    payload = TrainRequest(
        view=view,
        lance_uri=data_path or lance_uri,
        data_path=data_path,
        output_uri=effective_output_uri,
        overrides=override,
        wandb=WandbSettings(
            enabled=wandb_enabled,
            project=wandb_project,
            run_name=wandb_run_name,
            mode=wandb_mode,
        ),
        checkpoint_s3=checkpoint_s3,
        num_classes=num_classes,
        label_map=parse_label_map(label_map),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_filter_sql=validation_filter_sql or None,
    ).model_dump(mode="json")
    if service:
        resolved_endpoint = resolve_endpoint(endpoint)
        result = request_json("POST", resolved_endpoint, "/train", payload=payload, token_env=token_env, timeout=60.0)
        if wait:
            result = wait_for_training_run(
                str(result.get("run_id") or ""),
                endpoint=resolved_endpoint,
                token_env=token_env,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
    else:
        from npa.sdk.workbench.detection_training import train

        result = train(**payload).model_dump(mode="json")
    emit(result, output=output, text=f"run_id: {result.get('run_id')}\nstatus: {result.get('status')}")


def eval_cmd(
    checkpoint_uri: str = typer.Option(
        "",
        "--checkpoint-uri",
        help=(
            "Checkpoint S3/local URI. With --discover-checkpoint this is the training "
            "OUTPUT prefix to search instead."
        ),
    ),
    eval_view: str = typer.Option(..., "--eval-view", help="Lance materialized view to evaluate."),
    output_uri: str = typer.Option(..., "--output-uri", "--output-path", help="S3/local output URI."),
    lance_uri: str = typer.Option(DEFAULT_LANCE_URI, "--lance-uri", "--input-path", help="LanceDB URI."),
    discover_checkpoint: bool = typer.Option(
        False,
        "--discover-checkpoint/--no-discover-checkpoint",
        help=(
            "Resolve the checkpoint from the last completed /runs entry under "
            "--checkpoint-uri, substituting the trained epoch count."
        ),
    ),
    label_map: str = typer.Option(
        "",
        "--label-map",
        help=(
            "Category name -> class index, as JSON or name=index pairs. Required whenever the "
            "dataset stores string categories, which BDD100K does. `train` (the vehicle) is a "
            "real category name, and without a map it reaches int() and raises "
            "\"invalid literal for int() with base 10: 'train'\"."
        ),
    ),
    write_canonical_metrics: bool = typer.Option(
        False,
        "--write-canonical-metrics/--no-write-canonical-metrics",
        help="Publish the eval response to <output-uri>/metrics.json.",
    ),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Evaluate a detection-training checkpoint."""
    if not checkpoint_uri.strip():
        fail("--checkpoint-uri is required (with --discover-checkpoint it is the search prefix)")
    resolved_endpoint = resolve_endpoint(endpoint) if service else ""
    if discover_checkpoint:
        if not service:
            fail("--discover-checkpoint queries /runs, so it requires --service")
        checkpoint_uri = resolve_checkpoint_from_runs(
            checkpoint_uri, endpoint=resolved_endpoint, token_env=token_env
        )
    payload = EvalRequest(
        checkpoint_uri=checkpoint_uri,
        eval_view=eval_view,
        lance_uri=lance_uri,
        output_uri=output_uri,
        # Eval must read labels the same way training wrote them; EvalRequest has carried this
        # field all along with no CLI flag to fill it (EVIDENCE.md §R46).
        label_map=parse_label_map(label_map),
    ).model_dump(mode="json")
    if service:
        result = request_json("POST", resolved_endpoint, "/eval", payload=payload, token_env=token_env, timeout=900.0)
        # The template closed with a `jq -e` numeric check: a service can answer 200 with a
        # null mAP and the stage would otherwise report success on an unusable report.
        try:
            assert_eval_metrics(result)
        except DetectionTrainingArtifactError as exc:
            typer.echo(json.dumps(result, indent=2, sort_keys=True), err=True)
            fail(str(exc))
    else:
        from npa.sdk.workbench.detection_training import eval as sdk_eval

        result = sdk_eval(**payload).model_dump(mode="json")
    if write_canonical_metrics:
        result["metrics_uri"] = write_eval_metrics(result, output_uri=output_uri)
    emit(result, output=output, text=f"mAP: {result.get('mAP')}\neval_run_id: {result.get('eval_run_id')}")


def resolve_checkpoint_from_runs(output_uri: str, *, endpoint: str, token_env: str) -> str:
    """Ask ``/runs`` for the checkpoint the last completed run under ``output_uri`` wrote."""

    runs_payload = request_json("GET", endpoint, "/runs", token_env=token_env, timeout=60.0)
    runs = runs_payload.get("runs")
    if not isinstance(runs, list):
        fail(f"/runs did not return a runs list: {runs_payload!r}")
    try:
        return discover_checkpoint_uri(runs, output_uri=output_uri)
    except DetectionTrainingArtifactError as exc:
        typer.echo(json.dumps(runs_payload, indent=2, sort_keys=True), err=True)
        fail(str(exc))
        raise  # pragma: no cover - fail() raises


def write_eval_metrics(payload: dict[str, Any], *, output_uri: str) -> str:
    """Publish the eval response as the canonical ``metrics.json`` for this output prefix."""

    import tempfile

    target = eval_result_uri_for(output_uri)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.startswith("s3://"):
        from npa.clients.storage import StorageClient

        with tempfile.TemporaryDirectory(prefix="npa-detection-eval-") as tmp:
            local = Path(tmp) / EVAL_METRICS_FILENAME
            local.write_text(body, encoding="utf-8")
            return StorageClient.from_environment().upload_file(str(local), target)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Training run ID."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Fetch training run status."""
    if service:
        result = request_json(
            "GET",
            resolve_endpoint(endpoint),
            "/status",
            params={"run_id": run_id},
            token_env=token_env,
            timeout=30.0,
        )
    else:
        from npa.sdk.workbench.detection_training import status

        result = status(run_id=run_id).model_dump(mode="json")
    emit(result, output=output, text=f"status: {result.get('status')}\nepochs_completed: {result.get('epochs_completed')}")


def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show detection-training runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.detection_training.service import system_info_payload

        result = system_info_payload()
    emit(result, output=output)


def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help=(
            "NPA cluster profile whose cached kubeconfig to use. Empty (the default) uses the "
            "ambient kubeconfig, i.e. the cluster `kubectl` is already pointed at."
        ),
    ),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace for local listing."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List service-managed runs or Kubernetes resources."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/runs", token_env=token_env, timeout=30.0)
        emit(result, output=output, text="\n".join(run["run_id"] for run in result.get("runs", [])) or "No runs found.")
        return
    stdout = _kubectl(
        [
            "get",
            "deploy,svc",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=npa-detection-training",
            "-o",
            "json",
        ],
        capture=True,
        kubeconfig=_resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig),
    )
    data = json.loads(stdout or "{}")
    names = [item.get("metadata", {}).get("name", "") for item in data.get("items", [])]
    result = {"namespace": namespace, "resources": names, "count": len(names)}
    emit(result, output=output, text="\n".join(names) or "No detection-training resources found.")


def resolve_endpoint(endpoint: str) -> str:
    resolved = endpoint.strip() or os.environ.get("NPA_DETECTION_TRAINING_ENDPOINT", "")
    if not resolved:
        fail("--endpoint is required")
    if not resolved.startswith(("http://", "https://")):
        fail("--endpoint must be an http:// or https:// URL")
    return resolved.rstrip("/")


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{endpoint}{path}",
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fail(f"Detection-training request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach detection-training endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("Detection-training endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("Detection-training endpoint returned an unexpected response")
    return data


def _kubernetes_manifest(
    *,
    image: str,
    name: str,
    namespace: str,
    port: int,
    input_path: str,
    output_path: str,
    node_selector_key: str,
    node_selector_value: str,
    image_pull_secret: str,
    auth_mode: str,
    token_env: str,
) -> dict[str, Any]:
    env = _service_env(input_path=input_path, output_path=output_path, auth_mode=auth_mode, token_env=token_env, port=port)
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": f"{name}-env", "namespace": namespace},
                "type": "Opaque",
                "data": {key: base64.b64encode(value.encode("utf-8")).decode("ascii") for key, value in env.items()},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "labels": {"app.kubernetes.io/name": "npa-detection-training", "app.kubernetes.io/instance": name},
                },
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
                    "template": {
                        "metadata": {"labels": {"app.kubernetes.io/name": "npa-detection-training", "app.kubernetes.io/instance": name}},
                        "spec": {
                            "nodeSelector": {node_selector_key: node_selector_value},
                            **({"imagePullSecrets": [{"name": image_pull_secret}]} if image_pull_secret else {}),
                            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                            "containers": [
                                {
                                    "name": "service",
                                    "image": image,
                                    "imagePullPolicy": "Always",
                                    "ports": [{"containerPort": port, "name": "http"}],
                                    "envFrom": [{"secretRef": {"name": f"{name}-env"}}],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                        "requests": {"nvidia.com/gpu": "1"},
                                    },
                                    "readinessProbe": {"httpGet": {"path": "/health", "port": "http"}, "initialDelaySeconds": 10, "periodSeconds": 10},
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                        "seccompProfile": {"type": "RuntimeDefault"},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "selector": {"app.kubernetes.io/instance": name},
                    "ports": [{"name": "http", "port": port, "targetPort": "http"}],
                },
            },
        ],
    }


def _service_env(*, input_path: str, output_path: str, auth_mode: str, token_env: str, port: int) -> dict[str, str]:
    creds = load_credentials()
    env = {
        "DETECTION_TRAINING_AUTH_MODE": auth_mode,
        "DETECTION_TRAINING_PORT": str(port),
        "NPA_INPUT_PATH": input_path,
        "NPA_OUTPUT_PATH": output_path,
        "AWS_REGION": os.environ.get("AWS_REGION", "auto"),
    }
    if auth_mode == "token":
        token = os.environ.get(token_env, "")
        if not token:
            fail(f"{token_env} is required when --auth-mode token")
        env["DETECTION_TRAINING_TOKEN"] = token
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or creds.s3_endpoint
    if access_key:
        env["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = secret_key
    if endpoint:
        env["AWS_ENDPOINT_URL"] = endpoint
        env["AWS_ENDPOINT_URL_S3"] = endpoint
        env["NEBIUS_S3_ENDPOINT"] = endpoint
    return {key: value for key, value in env.items() if value}


def _ensure_image_pull_secret(*, image: str, secret_name: str, namespace: str, kubeconfig: str) -> None:
    """Put a usable pull secret in the namespace, minting one rather than copying a stale file.

    `~/.docker/config.json` holds whatever token the operator last logged in with, and Nebius IAM
    tokens expire — so a deploy that copies it can leave a Deployment whose kubelet gets
    `401 Unauthorized` on its next restart. Minting is what the LanceDB deploy learned to do
    (EVIDENCE.md §R41); doing it the same way here means one answer to the same question.
    """

    registry = _image_registry(image)
    if not registry:
        return
    from npa.workbench.service_kubernetes import ServiceKubernetesError, ensure_registry_secret

    try:
        ensure_registry_secret(secret_name, namespace, registry)
        return
    except ServiceKubernetesError:
        # No mintable IAM identity here; fall back to whatever the operator is logged in as.
        pass
    docker_config = _docker_auth_config(registry)
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": base64.b64encode(json.dumps(docker_config).encode("utf-8")).decode("ascii"),
        },
    }
    _kubectl(["apply", "-f", "-"], stdin=json.dumps(payload), kubeconfig=kubeconfig)


def _image_registry(image: str) -> str:
    if "/" not in image:
        return ""
    first = image.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return ""


def _docker_auth_config(registry: str) -> dict[str, Any]:
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():
        fail(f"Cannot create image pull secret for {registry}: {config_path} does not exist")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"Cannot parse {config_path}: {exc}")
    entry = _docker_auth_entry(config, registry)
    if not entry:
        helper = _docker_credential_helper(config, registry)
        if not helper:
            fail(f"Cannot find Docker auth or credential helper for {registry}")
        entry = _docker_auth_from_helper(helper, registry)
    return {"auths": {registry: entry}}


def _docker_auth_entry(config: dict[str, Any], registry: str) -> dict[str, str] | None:
    auths = config.get("auths", {})
    if not isinstance(auths, dict):
        return None
    for candidate in (registry, f"https://{registry}", f"http://{registry}"):
        raw = auths.get(candidate)
        if isinstance(raw, dict) and (raw.get("auth") or raw.get("identitytoken")):
            return {key: value for key, value in raw.items() if isinstance(value, str)}
    return None


def _docker_credential_helper(config: dict[str, Any], registry: str) -> str:
    helpers = config.get("credHelpers", {})
    if isinstance(helpers, dict):
        for candidate in (registry, f"https://{registry}", f"http://{registry}"):
            helper = helpers.get(candidate)
            if isinstance(helper, str) and helper:
                return helper
    store = config.get("credsStore")
    return store if isinstance(store, str) else ""


def _docker_auth_from_helper(helper: str, registry: str) -> dict[str, str]:
    executable = f"docker-credential-{helper}"
    if shutil.which(executable) is None:
        fail(f"Docker credential helper {executable} is not installed")
    try:
        result = subprocess.run(
            [executable, "get"],
            input=registry,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"Docker credential helper {executable} cannot read credentials for {registry}: {detail}")
    except json.JSONDecodeError as exc:
        fail(f"Docker credential helper {executable} returned invalid JSON: {exc}")
    username = str(payload.get("Username") or payload.get("username") or "")
    secret = str(payload.get("Secret") or payload.get("secret") or "")
    if not username or not secret:
        fail(f"Docker credential helper {executable} returned incomplete credentials for {registry}")
    auth = base64.b64encode(f"{username}:{secret}".encode("utf-8")).decode("ascii")
    return {"username": username, "password": secret, "auth": auth}


def _kubectl(
    args: list[str],
    *,
    stdin: str | None = None,
    dry_run: bool = False,
    capture: bool = False,
    kubeconfig: str = "",
) -> str:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(args)
    if dry_run:
        typer.echo(" ".join(cmd))
        return ""
    try:
        result = subprocess.run(cmd, input=stdin, text=True, capture_output=True, check=True)
    except FileNotFoundError:
        fail("kubectl is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"kubectl command failed: {detail}")
    if not capture and result.stdout.strip():
        typer.echo(result.stdout.strip())
    return result.stdout


def _resolve_kubeconfig(*, cluster_name: str, kubeconfig: str) -> str:
    """Which kubeconfig to talk to, in order of explicitness.

    `--cluster-name` used to DEFAULT to a specific profile, so every deploy quietly targeted
    whichever cluster that cached kubeconfig pointed at — not the one the operator's `kubectl`
    was on. Live, that produced the least helpful failure available: `kubectl apply` reported
    "deployment configured", `rollout status` timed out, and the deployment was in no namespace
    of the cluster being inspected, because it had been created on a different one
    (EVIDENCE.md §R46).
    """

    if kubeconfig.strip():
        return kubeconfig.strip()
    if not cluster_name.strip():
        return ""
    path = Path.home() / ".npa" / "clusters" / cluster_name.strip() / "kubeconfig"
    return str(path) if path.exists() else ""


def _redact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(manifest))
    for item in redacted.get("items", []):
        if item.get("kind") == "Secret":
            item["data"] = {key: "<redacted>" for key in item.get("data", {})}
    return redacted


app.command("deploy")(deploy_cmd)
app.command("train")(train_cmd)
app.command("eval")(eval_cmd)
app.command("status")(status_cmd)
app.command("system-info")(system_info_cmd)
app.command("list")(list_cmd)
