"""Bounded argv transport for generated Isaac Job shell programs."""

from __future__ import annotations

import base64
import gzip
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_ARG_CHUNK_CHARS = 60_000
_SCRIPT_PATH = "/tmp/npa-isaac-job-script.sh"
_ISAAC_EULA_ENV = "ACCEPT_EULA"
_INLINE_SCRIPT_STUB = 'exec /bin/bash "$1"'
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


def embedded_base64_file_block(payload: str, *, destination: str, marker: str) -> str:
    """Materialize embedded text without passing it through a process argv.

    Curated Isaac scenario sets can be several MiB. The generated program is
    already transported as a compressed script, but invoking Python with that
    data in ``--payload`` expands it back into one process argument and exceeds
    Linux ``ARG_MAX``. A quoted here-document keeps the bytes in file transport
    and gives ``base64`` only fixed-size argv entries.
    """

    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", marker):
        raise ValueError("embedded payload marker must be an uppercase shell token")
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return (
        f"base64 --decode > {shlex.quote(destination)} <<'{marker}'\n"
        f"{encoded}\n"
        f"{marker}\n"
    )


def _require_isaac_route_enabled(env: dict[str, str]) -> None:
    """Apply the shared default and fail before Kit starts on explicit opt-out."""

    from npa.serverless_common.env import resolve_isaac_eula_acceptance

    if resolve_isaac_eula_acceptance(env) != "Y":
        raise RuntimeError(
            "inline Isaac execution was explicitly opted out through ACCEPT_EULA"
        )


def _report_cleanup_error(exc: OSError) -> None:
    """Emit a best-effort cleanup diagnostic without masking workload state."""

    message = f"could not remove temporary Isaac script: {exc}\n".encode(
        "utf-8", errors="replace"
    )
    try:
        os.write(2, message)
    except OSError:
        return


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
    _require_isaac_route_enabled(env)
    if command == ["/bin/bash", "-lc"] and args[:2] == [
        _DECODE_STUB,
        "npa-isaac-payload",
    ]:
        # The chunked container argv stays below Linux's per-argument limit, but
        # a large generated program can still exceed the process-wide ARG_MAX
        # when it is re-executed inside an already-running SkyPilot task. Decode
        # the exact payload in-process and pass Bash only a bounded script path.
        script = decode_compressed_bash_args(args)
        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="npa-isaac-job-script-",
                suffix=".sh",
                dir="/tmp",
                delete=False,
            ) as handle:
                script_path = Path(handle.name)
                os.fchmod(handle.fileno(), 0o700)
                handle.write(script)
            subprocess.run(
                [*command, _INLINE_SCRIPT_STUB, "npa-isaac-payload", str(script_path)],
                env=env,
                check=True,
            )
        finally:
            if script_path is not None:
                try:
                    script_path.unlink(missing_ok=True)
                except OSError as exc:
                    _report_cleanup_error(exc)
    else:
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
