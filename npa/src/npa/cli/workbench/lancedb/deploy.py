"""Deploy command for the LanceDB Workbench tool."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import typer

from npa.clients.credentials import storage_endpoint_url, storage_endpoint_warning

from .helpers import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CONTAINER_NAME,
    DEFAULT_PORT,
    DEFAULT_TOKEN_ENV,
    LanceDBRuntime,
    OutputFormat,
    container_image,
    emit,
    fail,
    storage_env,
    validate_endpoint,
    validate_port,
    validate_storage_path,
)


_SECRET_ENV_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "ACCESS_KEY")


def _is_secret_env(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def _auth_mode_for(runtime: LanceDBRuntime, auth_mode: str) -> str:
    value = auth_mode.strip().lower()
    if value == "auto":
        return "none" if runtime == LanceDBRuntime.container else "token"
    if value not in {"none", "token"}:
        fail("--auth-mode must be one of auto, none, or token")
    if value == "none" and runtime in {LanceDBRuntime.vm, LanceDBRuntime.byovm}:
        typer.echo("Warning: --auth-mode none exposes the LanceDB wrapper without token auth.", err=True)
    return value


def _container_name(name: str, port: int) -> str:
    return name.strip() or f"{DEFAULT_CONTAINER_NAME}-{port}"


def _run_container(
    *,
    image: str,
    name: str,
    port: int,
    storage_path: str,
    auth_mode: str,
    token_env: str,
    storage_endpoint: str,
    detach: bool,
    replace: bool,
    dry_run: bool,
) -> str:
    env = {
        **storage_env(),
        "LANCEDB_STORAGE_PATH": storage_path,
        "LANCEDB_PORT": str(port),
        "LANCEDB_AUTH_MODE": auth_mode,
        "LANCEDB_TOKEN": os.environ.get(token_env, ""),
    }
    if storage_endpoint:
        endpoint_url = storage_endpoint_url(storage_endpoint)
        env["AWS_ENDPOINT_URL"] = endpoint_url
        env["NEBIUS_S3_ENDPOINT"] = endpoint_url
    if auth_mode == "token" and not env["LANCEDB_TOKEN"]:
        fail(f"{token_env} is required when --auth-mode token")

    if replace:
        rm_cmd = ["docker", "rm", "-f", name]
        if dry_run:
            typer.echo(" ".join(rm_cmd))
        else:
            subprocess.run(rm_cmd, check=False, capture_output=True, text=True)

    cmd = [
        "docker",
        "run",
        "--rm" if not detach else "-d",
        "--name",
        name,
        "-p",
        f"{port}:{port}",
    ]
    redacted_cmd = list(cmd)
    for key, value in env.items():
        if value:
            cmd.extend(["-e", f"{key}={value}"])
            redacted_cmd.extend(["-e", f"{key}={'<redacted>' if _is_secret_env(key) else value}"])
    cmd.append(image)
    redacted_cmd.append(image)

    if dry_run:
        typer.echo(" ".join(redacted_cmd))
        return "<dry-run>"

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        fail("Docker is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        fail(f"LanceDB container deploy failed: {stderr}")
    return result.stdout.strip()


def _cloud_payload(
    *,
    endpoint: str,
    api_key_env: str,
    database: str,
    cloud_region: str,
) -> dict[str, Any]:
    if not endpoint:
        fail("--endpoint is required for --runtime cloud")
    if not os.environ.get(api_key_env):
        fail(f"{api_key_env} is required for --runtime cloud")
    if not database:
        fail("--database is required for --runtime cloud")
    if not cloud_region:
        fail("--cloud-region is required for --runtime cloud")
    return {
        "runtime": "cloud",
        "endpoint": validate_endpoint(endpoint),
        "database": database,
        "cloud_region": cloud_region,
        "api_key_env": api_key_env,
        "status": "configured",
    }


def deploy_cmd(
    runtime: LanceDBRuntime = typer.Option(LanceDBRuntime.vm, "--runtime", help="Runtime: vm, container, byovm, or cloud."),
    storage_path: str = typer.Option("", "--storage-path", help="S3 URI or absolute local path for LanceDB data."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="LanceDB wrapper port."),
    auth_mode: str = typer.Option("auto", "--auth-mode", help="Auth mode: auto, none, or token."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    endpoint: str = typer.Option("", "--endpoint", help="Existing endpoint or LanceDB Cloud URL."),
    api_key_env: str = typer.Option(DEFAULT_API_KEY_ENV, "--api-key-env", help="Environment variable containing LanceDB Cloud API key."),
    database: str = typer.Option("", "--database", help="LanceDB Cloud database name."),
    cloud_region: str = typer.Option("", "--cloud-region", help="LanceDB Cloud region."),
    cpu_platform: str = typer.Option("", "--cpu-platform", help="CPU VM selector for managed VM deploy."),
    cpu_preset: str = typer.Option("4vcpu-16gb", "--cpu-preset", help="CPU VM preset."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Boot disk size in GiB."),
    data_disk_size: int | None = typer.Option(None, "--data-disk-size", help="Optional data disk size in GiB."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("", "--region", help="Nebius region."),
    tf_dir: str = typer.Option("", "--tf-dir", help="Terraform directory override."),
    tf_var: list[str] = typer.Option([], "--tf-var", "-v", help="Extra Terraform variable key=value."),
    storage_endpoint: str = typer.Option(
        "",
        "--storage-endpoint",
        help=(
            "Nebius S3-compatible endpoint override, for example "
            "storage.eu-north1.nebius.cloud. Also settable with NPA_STORAGE_ENDPOINT."
        ),
    ),
    skip_infra: bool = typer.Option(False, "--skip-infra", help="Skip infrastructure provisioning."),
    skip_app: bool = typer.Option(False, "--skip-app", help="Skip application deploy."),
    destroy: bool = typer.Option(False, "--destroy", help="Tear down or unregister the service."),
    replace: bool = typer.Option(False, "--replace", help="Replace existing container or instance."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without running them."),
    default: bool = typer.Option(False, "--default", help="Save endpoint as default config."),
    image: str = typer.Option("", "--image", help="Container image reference."),
    container_name: str = typer.Option("", "--container-name", help="Local container name override."),
    detach: bool = typer.Option(True, "--detach/--no-detach", help="Run local container in the background."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy or register a LanceDB service."""
    validate_port(port)
    storage_endpoint_override = storage_endpoint.strip() or os.environ.get("NPA_STORAGE_ENDPOINT", "").strip()
    endpoint_warning = storage_endpoint_warning(
        storage_endpoint_override
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
        or os.environ.get("AWS_ENDPOINT_URL", "")
    )
    if endpoint_warning:
        typer.echo(endpoint_warning)
    # TODO: infer the storage endpoint from the selected Nebius region once the
    # VM/BYOVM LanceDB deploy path is backed by shared Workbench registration.
    if runtime == LanceDBRuntime.serverless:
        fail("LanceDB deploy does not support --runtime serverless; use container, vm, byovm, or cloud.")

    if runtime == LanceDBRuntime.cloud:
        payload = _cloud_payload(
            endpoint=endpoint,
            api_key_env=api_key_env,
            database=database,
            cloud_region=cloud_region,
        )
        if default:
            payload["default_requested"] = True
            payload["note"] = "Saving defaults is deferred to the Workbench parent registration follow-up."
        emit(payload, output=output, text=f"LanceDB Cloud endpoint configured: {payload['endpoint']}")
        return

    resolved_storage = validate_storage_path(storage_path)
    resolved_auth = _auth_mode_for(runtime, auth_mode)
    image_ref = container_image(image)

    if destroy and runtime == LanceDBRuntime.container:
        name = _container_name(container_name, port)
        cmd = ["docker", "rm", "-f", name]
        if dry_run:
            typer.echo(" ".join(cmd))
        else:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        emit({"runtime": "container", "container": name, "status": "removed"}, output=output)
        return

    if runtime == LanceDBRuntime.container:
        name = _container_name(container_name, port)
        container_id = _run_container(
            image=image_ref,
            name=name,
            port=port,
            storage_path=resolved_storage,
            auth_mode=resolved_auth,
            token_env=token_env,
            storage_endpoint=storage_endpoint_override,
            detach=detach,
            replace=replace,
            dry_run=dry_run,
        )
        payload = {
            "runtime": "container",
            "endpoint": f"http://localhost:{port}",
            "container": name,
            "container_id": container_id,
            "image": image_ref,
            "storage_path": resolved_storage,
            "auth_mode": resolved_auth,
            "status": "running" if not dry_run else "dry-run",
        }
        emit(payload, output=output, text=f"LanceDB container running at {payload['endpoint']}")
        return

    if skip_app and not skip_infra:
        payload = {
            "runtime": runtime.value,
            "status": "infra-only",
            "storage_path": resolved_storage,
            "cpu_platform": cpu_platform,
            "cpu_preset": cpu_preset,
            "disk_size": disk_size,
            "data_disk_size": data_disk_size,
            "project_id": project_id,
            "tenant_id": tenant_id,
            "region": region,
            "tf_dir": tf_dir,
            "tf_var": tf_var,
        }
        emit(payload, output=output, text="LanceDB VM infrastructure plan accepted.")
        return

    if runtime in {LanceDBRuntime.vm, LanceDBRuntime.byovm}:
        payload = {
            "runtime": runtime.value,
            "status": "blocked",
            "storage_path": resolved_storage,
            "image": image_ref,
            "reason": (
                "LanceDB VM/BYOVM app deploy is not available from this command yet: "
                "the wrapper must be registered through the shared Workbench deploy "
                "path. Use --runtime container for a local server, --runtime cloud for "
                "LanceDB Cloud, or pass --dry-run to preview the plan."
            ),
            "skip_infra": skip_infra,
            "skip_app": skip_app,
            "dry_run": dry_run,
        }
        if dry_run:
            emit(payload, output=output, text="LanceDB VM/BYOVM deploy plan accepted.")
            return
        fail(payload["reason"])
