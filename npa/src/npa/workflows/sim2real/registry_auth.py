"""Refresh Nebius Container Registry pull credentials before K8s sim2real jobs."""

from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from npa.clients.nebius_auth import mint_nebius_iam_token, strip_ambient_token_env


def mint_nebius_registry_token(*, nebius_cli: str = "nebius") -> str:
    """Return a short-lived IAM token for ``cr.*.nebius.cloud`` pulls.

    Delegates to the canonical :func:`npa.clients.nebius_auth.mint_nebius_iam_token`,
    which performs a fresh profile-scoped exchange first (ambient token stripped,
    so a stale/wrong-identity ``NEBIUS_IAM_TOKEN`` can't be re-embedded into the
    very pull secret this refresh exists to fix — the ``403`` / ``ErrImagePull``
    failure), and only falls back to an injected ``NEBIUS_IAM_TOKEN`` when the
    ``nebius`` CLI is unavailable/fails — the in-pod case (token injected, no CLI
    on PATH). Raises ``NebiusTokenError`` (a ``RuntimeError``) if no token can be
    obtained, which best-effort callers catch.
    """

    return mint_nebius_iam_token(nebius_cli=nebius_cli)


def _registry_server_from_image(image: str) -> str:
    ref = image.removeprefix("docker:").strip()
    if "/" not in ref:
        return ""
    host = ref.split("/", 1)[0]
    if "." in host or ":" in host or host == "localhost":
        return host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return ""


def docker_config_json(
    *, registry_servers: Sequence[str], token: str, username: str = "iam"
) -> dict[str, Any]:
    """Return a dockerconfigjson whose ``auths`` covers every given registry host.

    A pull secret holds exactly one dockerconfigjson and an API patch replaces
    it wholesale, so every host a run pulls from has to be in the same document. One
    IAM token authenticates all of them, since the token is identity-scoped rather
    than host-scoped.
    """

    auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    return {
        "auths": {
            server: {"username": username, "password": token, "auth": auth}
            for server in dict.fromkeys(server for server in registry_servers if server)
        }
    }


def _normalize_registry_server(value: str) -> str:
    server = value.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not server.startswith("cr.") or ".nebius.cloud" not in server:
        return ""
    return server


def _docker_helper_credential(
    registry_server: str,
    *,
    env: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Resolve the configured credential Docker uses for ``registry_server``.

    Operator VMs commonly configure ``docker-credential-nebius-agent-sa`` in
    ``~/.docker/config.json``. Copying that file into a Kubernetes pull secret
    only copies an empty ``auths`` placeholder; kubelet cannot execute Docker's
    credential helper and the subsequent private-image pull fails with 403.
    Materialize the helper result into the pull secret instead. Explicit
    ``docker login`` auth entries are already materialized credentials and are
    returned directly. In-cluster orchestrators with neither form fall through
    to the existing CLI/injected-token path.
    """

    process_env = dict(os.environ if env is None else env)
    docker_config_dir = (process_env.get("DOCKER_CONFIG") or "").strip()
    config_path = (
        Path(docker_config_dir) / "config.json"
        if docker_config_dir
        else Path.home() / ".docker" / "config.json"
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    helper_suffix = str(
        (config.get("credHelpers") or {}).get(registry_server) or ""
    ).strip()
    if not helper_suffix:
        # Docker configs written by an explicit `docker login` commonly carry
        # the usable credential directly under `auths`.  Kubelet cannot read
        # the operator's config file, so materialize this form into the Secret
        # exactly as we do for a credential helper.  Never fall through to a
        # newly minted token merely because the config uses the standard direct
        # representation: doing so can overwrite a proven project-scoped pull
        # credential with an identity that the target registry rejects.
        entry = (config.get("auths") or {}).get(registry_server) or {}
        encoded = str(entry.get("auth") or "").strip()
        if not encoded:
            return None
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
        username, separator, secret = decoded.partition(":")
        if not separator or not username or not secret:
            return None
        return username, secret
    try:
        result = subprocess.run(
            [f"docker-credential-{helper_suffix}", "get"],
            input=f"{registry_server}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            env=strip_ambient_token_env(process_env),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        credential = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    username = str(credential.get("Username") or "").strip()
    secret = str(credential.get("Secret") or "").strip()
    if not username or not secret:
        return None
    return username, secret


def ensure_nebius_registry_pull_secret(
    *,
    registry_server: str = "",
    registry_servers: Sequence[str] = (),
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
    nebius_cli: str = "nebius",
    username: str = "iam",
    token: str = "",
) -> None:
    """Apply a fresh docker-registry secret so orchestrator pulls do not 401.

    Pass every host a run pulls from in one call. Applying them one at a time would
    leave only the host from the last call, because each apply replaces the secret,
    and pods pulling from the others would 401 — the failure this refresh exists to
    prevent.
    """

    servers = [
        normalized
        for normalized in (
            _normalize_registry_server(value)
            for value in (registry_server, *registry_servers)
        )
        if normalized
    ]
    if not servers:
        return
    credentials: dict[str, tuple[str, str]] = {}
    fallback_token = token
    for server in servers:
        credential = (username, token) if token else _docker_helper_credential(server)
        if credential is None:
            if not fallback_token:
                fallback_token = mint_nebius_registry_token(nebius_cli=nebius_cli)
            credential = ("iam", fallback_token)
        credentials[server] = credential
    auths: dict[str, Any] = {}
    for server, (username, token) in credentials.items():
        auths.update(
            docker_config_json(
                registry_servers=[server], token=token, username=username
            )["auths"]
        )
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": base64.b64encode(
                json.dumps({"auths": auths}).encode("utf-8")
            ).decode("ascii")
        },
    }
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    try:
        client = KubernetesJobClient.from_environment(
            namespace=namespace,
            kubeconfig=kubeconfig,
            context=k8s_context,
        )
        client.apply_secret(payload)
    except Exception as exc:
        raise RuntimeError(
            f"failed to apply registry pull secret {secret_name}: {exc}"
        ) from exc


def ensure_registry_pull_secret_for_images(
    *images: str,
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
) -> None:
    servers = [server for server in map(_registry_server_from_image, images) if server]
    if not servers:
        return
    ensure_nebius_registry_pull_secret(
        registry_servers=servers,
        secret_name=secret_name,
        namespace=namespace,
        kubeconfig=kubeconfig,
        k8s_context=k8s_context,
    )
