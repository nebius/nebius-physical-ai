"""Find the durable weight cache in the cluster, so submit does not have to be told.

A claim is storage only the operator can create, so NPA will not invent one. But
once they have applied ``npa/docker/workbench/common/model-weight-cache.yaml``, the
claim is right there with a known name, and making them also export
``NPA_MODEL_CACHE_PVC`` on every shell that submits is a step whose only possible
outcomes are "remembered" and "silently paid for the download again".

So submit looks. If the claim exists, the render behaves exactly as if the variable
had been set; if it does not, nothing changes and the run keeps its ephemeral
defaults. The lookup is read-only, best-effort, and never blocks a submit: a cluster
that cannot answer is the same as a cluster with no claim.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from npa.workbench.model_cache import (
    DEFAULT_MODEL_CACHE_CLAIM,
    MODEL_CACHE_DIR_ENV,
    MODEL_CACHE_HOST_PATH_ENV,
    MODEL_CACHE_PVC_ENV,
    model_cache_disabled,
)

LOOKUP_TIMEOUT_SECONDS = 20


def _already_configured(environ: dict[str, str]) -> bool:
    return any(
        str(environ.get(name, "") or "").strip()
        for name in (MODEL_CACHE_PVC_ENV, MODEL_CACHE_HOST_PATH_ENV, MODEL_CACHE_DIR_ENV)
    )


def find_model_cache_claim(
    *,
    context: str = "",
    kubeconfig: str = "",
    namespace: str = "default",
    claim: str = DEFAULT_MODEL_CACHE_CLAIM,
) -> str:
    """Return ``claim`` when it exists and is Bound in the cluster, else ``""``.

    A Pending claim is deliberately not adopted: it has no volume behind it yet, so
    mounting it would leave every pod stuck in ContainerCreating -- which is exactly
    the failure the manifest's own preflight notes warn about.
    """

    binary = shutil.which("kubectl")
    if not binary:
        return ""
    argv = [binary, "get", "pvc", claim, "-n", namespace, "-o", "json"]
    if context:
        argv[1:1] = ["--context", context]
    env = dict(os.environ)
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=LOOKUP_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return ""
    if str((payload.get("status") or {}).get("phase") or "") != "Bound":
        return ""
    return claim


def adopt_model_cache_claim(
    *,
    context: str = "",
    kubeconfig: str = "",
    namespace: str = "default",
    environ: dict[str, str] | None = None,
) -> str:
    """Export the discovered claim so the renderer picks it up. Returns its name.

    Anything the operator configured explicitly wins untouched, including switching
    the cache off.
    """

    target = os.environ if environ is None else environ
    if model_cache_disabled(target) or _already_configured(target):
        return ""
    found = find_model_cache_claim(
        context=context, kubeconfig=kubeconfig, namespace=namespace
    )
    if found:
        target[MODEL_CACHE_PVC_ENV] = found
    return found


__all__ = ["adopt_model_cache_claim", "find_model_cache_claim"]
