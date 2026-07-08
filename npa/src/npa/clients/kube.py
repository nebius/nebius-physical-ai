"""Host-side ``kubectl`` invocation that is robust to a stale ambient IAM token.

Nebius managed-Kubernetes kubeconfigs authenticate through an exec credential
plugin (``nebius mk8s ... cluster get-token``). When a stale/expired
``NEBIUS_IAM_TOKEN`` is exported in the ambient environment (a common trap on
long-lived operator VMs and inherited ``tmux`` server envs), the ``nebius`` CLI
honors that env token instead of minting a fresh one, and every ``kubectl`` call
fails with ``Unauthenticated`` / exit code 7.

``run_kubectl`` runs the command as-is first, and — only when it fails with that
specific auth signature *and* an ambient IAM token is present — retries once with
the token stripped from the subprocess environment so the exec plugin (or CLI
profile) re-authenticates. This is safe: a token-only environment with a valid
token succeeds on the first try (no retry), and a token-only environment with an
invalid token fails either way, so stripping on retry never regresses it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

# Ambient env vars that carry a Nebius IAM token. Either can shadow the
# kubeconfig exec plugin when stale.
IAM_TOKEN_ENV_KEYS: tuple[str, ...] = ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN")

# Substrings that mark a kubectl/exec-plugin failure caused by a bad IAM token.
_STALE_TOKEN_SIGNATURES: tuple[str, ...] = (
    "nebius_iam_token",
    "expired token",
    "unauthenticated",
    "invalid token",
    "get-token",
    "getting credentials: exec",
    "failed with exit code 7",
)


@dataclass(frozen=True)
class KubectlResult:
    """Result of a ``kubectl`` invocation, plus whether the token was stripped."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    retried_without_iam_token: bool = False


def ambient_iam_token_present(env: Mapping[str, str] | None = None) -> bool:
    """Return True when a Nebius IAM token is set in *env* (defaults to os.environ)."""

    source = os.environ if env is None else env
    return any(str(source.get(key, "")).strip() for key in IAM_TOKEN_ENV_KEYS)


def strip_iam_token_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* with the ambient IAM token keys removed."""

    return {k: v for k, v in env.items() if k not in IAM_TOKEN_ENV_KEYS}


def looks_like_stale_iam_token_error(text: str) -> bool:
    """Return True when kubectl output matches a stale-IAM-token failure."""

    lowered = str(text).lower()
    return any(sig in lowered for sig in _STALE_TOKEN_SIGNATURES)


def run_kubectl(
    args: Sequence[str],
    *,
    context: str = "",
    kubeconfig: str = "",
    timeout: float = 30.0,
    binary: str | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> KubectlResult:
    """Run ``kubectl <args>``, retrying once without a stale ambient IAM token.

    ``runner`` defaults to :func:`subprocess.run` and is injectable for tests.
    Returns a :class:`KubectlResult`; a returncode of ``127`` means kubectl was
    not found.
    """

    kube_bin = binary or os.environ.get("NPA_KUBECTL_BIN") or shutil.which("kubectl")
    if not kube_bin:
        return KubectlResult(returncode=127, stderr="kubectl not found on PATH")

    cmd = [kube_bin]
    if context:
        cmd += ["--context", context]
    cmd += list(args)

    base_env = dict(os.environ if env is None else env)
    if kubeconfig:
        base_env["KUBECONFIG"] = kubeconfig
    run = runner or subprocess.run

    def _invoke(proc_env: dict[str, str]) -> KubectlResult:
        try:
            proc = run(
                cmd,
                env=proc_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return KubectlResult(returncode=1, stderr=str(exc))
        return KubectlResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    result = _invoke(base_env)
    if result.returncode == 0:
        return result

    combined = f"{result.stderr}\n{result.stdout}"
    if ambient_iam_token_present(base_env) and looks_like_stale_iam_token_error(combined):
        retry = _invoke(strip_iam_token_env(base_env))
        return KubectlResult(
            returncode=retry.returncode,
            stdout=retry.stdout,
            stderr=retry.stderr,
            retried_without_iam_token=True,
        )
    return result


__all__ = [
    "IAM_TOKEN_ENV_KEYS",
    "KubectlResult",
    "ambient_iam_token_present",
    "looks_like_stale_iam_token_error",
    "run_kubectl",
    "strip_iam_token_env",
]
