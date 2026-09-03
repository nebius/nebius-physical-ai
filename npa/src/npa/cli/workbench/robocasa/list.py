"""List RoboCasa runs or Kubernetes resources."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from npa.workbench.robocasa.schemas import DEFAULT_TOKEN_ENV

from npa.cli.workbench.robocasa.helpers import OutputFormat, emit, fail, request_json, resolve_endpoint

DEFAULT_NAMESPACE = "default"


def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="RoboCasa service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help="NPA cluster profile whose cached kubeconfig to use. Empty (the default) uses the ambient kubeconfig.",
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
            "app.kubernetes.io/name=npa-robocasa",
            "-o",
            "json",
        ],
        capture=True,
        kubeconfig=_resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig),
    )
    data = json.loads(stdout or "{}")
    names = [item.get("metadata", {}).get("name", "") for item in data.get("items", [])]
    result = {"namespace": namespace, "resources": names, "count": len(names)}
    emit(result, output=output, text="\n".join(names) or "No robocasa resources found.")


def _kubectl(
    args: list[str],
    *,
    capture: bool = False,
    kubeconfig: str = "",
) -> str:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(args)
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except FileNotFoundError:
        fail("kubectl is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"kubectl command failed: {detail}")
    if not capture and result.stdout.strip():
        print(result.stdout.strip())
    return result.stdout


def _resolve_kubeconfig(*, cluster_name: str, kubeconfig: str) -> str:
    if kubeconfig.strip():
        return kubeconfig.strip()
    if not cluster_name.strip():
        return ""
    path = Path.home() / ".npa" / "clusters" / cluster_name.strip() / "kubeconfig"
    return str(path) if path.exists() else ""
