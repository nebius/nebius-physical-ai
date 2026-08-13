"""Bounded argv transport for generated Isaac Job shell programs."""

from __future__ import annotations

import base64
import gzip
import os
import subprocess
from typing import Any

_ARG_CHUNK_CHARS = 60_000
_SCRIPT_PATH = "/tmp/npa-isaac-job-script.sh"
_ISAAC_EULA_VARS = ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA")
_ACCEPTED_EULA_VALUES = frozenset({"1", "TRUE", "Y", "YES"})
_DECODE_STUB = (
    "set -euo pipefail; "
    f"printf '%s' \"$@\" | base64 --decode | gzip --decompress > {_SCRIPT_PATH}; "
    f"chmod 700 {_SCRIPT_PATH}; exec /bin/bash {_SCRIPT_PATH}"
)


def compressed_bash_launch(script: str) -> tuple[list[str], list[str]]:
    """Return a bash command/argv whose individual arguments stay below 128 KiB.

    Linux limits each individual argv/environment string to 128 KiB even when
    the process-wide ``ARG_MAX`` is larger. Generated Isaac programs include
    task source and curated scenarios, so passing the program directly to
    ``bash -lc`` can fail before the container starts. Deterministically gzip
    and base64 the bytes, split them into conservative chunks, and let a small
    login-shell stub reconstruct and execute the exact program in the pod.
    """

    encoded = base64.b64encode(gzip.compress(script.encode("utf-8"), mtime=0)).decode(
        "ascii"
    )
    chunks = [
        encoded[offset : offset + _ARG_CHUNK_CHARS]
        for offset in range(0, len(encoded), _ARG_CHUNK_CHARS)
    ]
    # With ``bash -c COMMAND ARG0 ARG1...``, the label becomes $0 and only the
    # payload chunks become "$@" inside the decoder stub.
    return ["/bin/bash", "-lc"], [_DECODE_STUB, "npa-isaac-payload", *chunks]


def decode_compressed_bash_args(args: list[str]) -> str:
    """Decode :func:`compressed_bash_launch` output for tests/audits."""

    if len(args) < 3 or args[0] != _DECODE_STUB or args[1] != "npa-isaac-payload":
        raise ValueError("not an NPA compressed Isaac bash payload")
    return gzip.decompress(base64.b64decode("".join(args[2:]))).decode("utf-8")


def _require_operator_eula_acceptance(env: dict[str, str]) -> None:
    """Fail before Kit starts unless the operator explicitly accepted both terms."""

    missing = [
        name
        for name in _ISAAC_EULA_VARS
        if str(env.get(name) or "").strip().upper() not in _ACCEPTED_EULA_VALUES
    ]
    if missing:
        raise RuntimeError(
            "inline Isaac execution requires explicit operator EULA acceptance: "
            + " ".join(missing)
        )


def execute_manifest_container_inline(manifest: dict[str, Any]) -> dict[str, Any]:
    """Execute an existing Isaac Job payload in its workflow-owned GPU task.

    The Sim2Real workflow renderer has already selected the immutable image,
    admitted the task through SkyPilot/Kubernetes, mounted credentials, and
    assigned its GPU.  Reusing the exact generated container command here keeps
    the proven Isaac scripts while avoiding a hidden sibling Kubernetes Job.
    """

    containers = (
        manifest.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    if len(containers) != 1:
        raise RuntimeError("inline Isaac execution requires exactly one container")
    container = containers[0]
    expected_image = str(container.get("image") or "").removeprefix("docker:")
    task_image = os.environ.get("NPA_TASK_IMAGE", "").removeprefix("docker:")
    if not expected_image or "@sha256:" not in expected_image:
        raise RuntimeError("inline Isaac execution requires an immutable image digest")
    if not task_image or task_image != expected_image:
        raise RuntimeError(
            "inline Isaac payload image does not match the workflow task image: "
            f"expected={expected_image!r} task={task_image!r}"
        )
    command = [str(item) for item in container.get("command") or []]
    args = [str(item) for item in container.get("args") or []]
    if not command:
        raise RuntimeError("inline Isaac payload has no command")
    env = os.environ.copy()
    for item in container.get("env") or []:
        name = str(item.get("name") or "")
        if name and "value" in item:
            env[name] = str(item["value"])
    _require_operator_eula_acceptance(env)
    subprocess.run([*command, *args], env=env, check=True)
    return {
        "mode": "npa_workflow_skypilot_task",
        "job_name": os.environ.get("SKYPILOT_TASK_ID", "")
        or os.environ.get("SKYPILOT_CLUSTER_NAME", ""),
        "image": expected_image,
        "gpu_product": os.environ.get("NPA_WORKFLOW_GPU_PRODUCT", "")
        or os.environ.get("NPA_SIM2REAL_K8S_GPU_PRODUCT", ""),
        "owner": "standard_npa_workflow_runtime",
    }
