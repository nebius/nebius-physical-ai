"""Refresh Nebius Container Registry pull credentials before K8s sim2real jobs."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from typing import Any


def mint_nebius_registry_token(*, nebius_cli: str = "nebius") -> str:
    """Return a short-lived IAM token for ``cr.*.nebius.cloud`` pulls.

    Prefers a ``NEBIUS_IAM_TOKEN`` from the environment (in-pod contexts often
    have the token injected but not the ``nebius`` CLI), falling back to
    ``nebius iam get-access-token`` when the env var is unset. The CLI fallback
    runs under ``nebius_cli_env()``, which drops any ambient token so a stale one
    cannot shadow the active profile and mint a pull secret for the wrong
    principal (401 on image pulls).
    """

    from npa.clients.nebius import nebius_cli_env

    env_token = os.environ.get("NEBIUS_IAM_TOKEN", "").strip()
    if env_token:
        return env_token
    try:
        result = subprocess.run(
            [nebius_cli, "iam", "get-access-token"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            # A stale ambient NEBIUS_IAM_TOKEN would be minted into the pull
            # secret and 401 on image pulls; use the active profile instead.
            env=nebius_cli_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not mint Nebius registry token with `nebius iam get-access-token`"
        ) from exc
    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(
            "Could not mint Nebius registry token with `nebius iam get-access-token`: "
            + detail
        )
    return token


def _registry_server_from_image(image: str) -> str:
    ref = image.removeprefix("docker:").strip()
    if "/" not in ref:
        return ""
    host = ref.split("/", 1)[0]
    if "." in host or ":" in host or host == "localhost":
        return host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return ""


def docker_config_json(*, registry_server: str, token: str) -> dict[str, Any]:
    username = "iam"
    auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    return {
        "auths": {
            registry_server: {
                "username": username,
                "password": token,
                "auth": auth,
            }
        }
    }


def ensure_nebius_registry_pull_secret(
    *,
    registry_server: str,
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
    nebius_cli: str = "nebius",
) -> None:
    """Apply a fresh docker-registry secret so orchestrator pulls do not 401."""

    server = registry_server.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not server.startswith("cr.") or ".nebius.cloud" not in server:
        return
    token = mint_nebius_registry_token(nebius_cli=nebius_cli)
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": base64.b64encode(
                json.dumps(docker_config_json(registry_server=server, token=token)).encode(
                    "utf-8"
                )
            ).decode("ascii")
        },
    }
    cmd = ["kubectl"]
    if k8s_context:
        cmd.extend(["--context", k8s_context])
    cmd.extend(["-n", namespace, "apply", "-f", "-"])
    env = dict(os.environ)
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # In-pod orchestrators frequently have no kubectl on PATH, which raises
        # FileNotFoundError. Callers treat this refresh as best-effort and catch
        # RuntimeError, so keep every expected failure inside that contract
        # instead of letting an OSError escape and kill the run.
        raise RuntimeError(
            f"failed to apply registry pull secret {secret_name}: {exc}"
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"failed to apply registry pull secret {secret_name}: {detail}")


def ensure_registry_pull_secret_for_images(
    *images: str,
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
) -> None:
    for image in images:
        server = _registry_server_from_image(image)
        if server:
            ensure_nebius_registry_pull_secret(
                registry_server=server,
                secret_name=secret_name,
                namespace=namespace,
                kubeconfig=kubeconfig,
                k8s_context=k8s_context,
            )
            return
