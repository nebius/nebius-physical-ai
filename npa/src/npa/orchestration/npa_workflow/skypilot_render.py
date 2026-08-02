"""Render an ``npa.workflow`` execution plan as a SkyPilot multi-doc YAML."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import ExecutionPlan, PlanStep  # noqa: F401
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec

# Map toolRef prefixes / exact names onto CONTAINER_IMAGE_NAMES keys.
# Token Factory is a hosted HTTP API client. Do not pin the heavy cosmos image:
# SkyPilot's k8s apt-ssh runtime setup fails inside npa-cosmos. Use the default
# SkyPilot image and stage npa via NPA_SRC_S3_URI (or an image override).
TOOL_REF_IMAGE_TOOL: dict[str, str] = {
    "workbench.vlm_eval": "cosmos",
    "workbench.cosmos2": "cosmos2-transfer",
    "workbench.cosmos3": "cosmos3-reason",
    "workbench.cosmos_curate": "cosmos-curate",
    "workbench.cosmos_evaluator": "cosmos-evaluator",
    "workbench.lancedb": "lancedb",
    "workbench.detection_training": "detection-training",
    "workbench.fiftyone": "fiftyone",
    "workbench.rl": "isaac-lab",
    "workbench.isaac_lab": "isaac-lab",
    "workbench.lerobot": "lerobot",
    "workbench.sonic": "sonic",
    "workbench.mjlab": "sonic",
    "workbench.retargeting": "retargeting",
    "workbench.sim2real": "lerobot-vlm-rl",
    "workbench.sim2real_envgen": "envgen",
    "workbench.byof": "isaac-lab",
    "workbench.genesis": "genesis",
    "workbench.groot": "groot",
}

SECRET_ENV_HINTS: dict[str, tuple[str, ...]] = {
    "workbench.token_factory": ("NEBIUS_TOKEN_FACTORY_KEY",),
    "workbench.vlm_eval": (),
    # Attribute verification generates and answers its questions on Token Factory.
    "workbench.cosmos_evaluator": ("NEBIUS_TOKEN_FACTORY_KEY",),
    "workbench.cosmos3": ("HF_TOKEN",),
    "workbench.sonic": ("HF_TOKEN", "NGC_API_KEY"),
    "workbench.groot": ("HF_TOKEN", "NGC_API_KEY"),
}


class NpaWorkflowRenderError(NpaWorkflowError):
    """Raised when an npa.workflow plan cannot be rendered to SkyPilot YAML."""


def _default_aws_endpoint_url() -> str:
    """Prefer the operator's configured endpoint over a hard-coded region."""

    import os

    return (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or "https://storage.eu-north1.nebius.cloud"
    )


@dataclass(frozen=True)
class SkypilotRenderOptions:
    """Controls how planned steps become SkyPilot task documents."""

    registry: str = ""
    image_overrides: Mapping[str, str] = field(default_factory=dict)
    default_setup: bool = True
    execution: str = "serial"
    aws_endpoint_url: str = field(default_factory=_default_aws_endpoint_url)
    include_aws_endpoint: bool = True
    gpu_target: str = ""
    image_variant: str = ""
    # When False (``--plan-only``), embed placeholders instead of minting live
    # Nebius registry tokens into rendered YAML that may be printed to stdout.
    materialize_registry_secrets: bool = True


def normalize_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    """Map an npa.workflow resource profile onto a SkyPilot ``resources`` block.

    On Kubernetes, exact ``cpus`` / ``memory`` often fail prechecks when no node
    has that precise free shape. Append ``+`` so SkyPilot can schedule on larger
    nodes (including GPU nodes with spare CPU).
    """

    import os as _os

    # Cluster-specific GPU product override: SkyPilot k8s matches on the node's
    # advertised accelerator name, which varies by cluster (e.g. RTXPRO6000 vs
    # RTXPRO-6000-BLACKWELL-SERVER-EDITION). Let operators override the spec's
    # accelerators at submit time without editing the committed blueprint.
    accel_override = str(_os.environ.get("NPA_WORKFLOW_GPU_ACCELERATOR") or "").strip()

    out: dict[str, Any] = {}
    for key in ("cloud", "accelerators", "cpus", "memory", "use_spot", "region"):
        if key not in resources or resources[key] in (None, ""):
            continue
        value = resources[key]
        if key == "accelerators" and accel_override:
            value = accel_override
        if key == "memory" and isinstance(value, str):
            stripped = value.strip()
            if stripped.lower().endswith("gi"):
                value = stripped[:-2]
            elif stripped.lower().endswith("g"):
                value = stripped[:-1]
        out[key] = value

    cloud = str(out.get("cloud") or "").strip().lower()
    if cloud in {"kubernetes", "k8s"}:
        for key in ("cpus", "memory"):
            if key not in out:
                continue
            raw = str(out[key]).strip()
            if raw and not raw.endswith("+"):
                out[key] = f"{raw}+"
    return out


#: Task-level SkyPilot config fields an npa.workflow resource profile may carry.
#: SkyPilot 0.12 accepts these inside a task's ``config:`` block, and it APPENDS
#: (rather than replaces) lists inside ``kubernetes.pod_config`` -- so a spec can
#: add an imagePullSecret or a volume without discarding the cluster-wide ones.
#: Kept to the fields a workload legitimately needs, so a spec cannot smuggle in
#: arbitrary cluster configuration.
TASK_CONFIG_KUBERNETES_FIELDS = ("pod_config", "provision_timeout")


def normalize_task_config(resources: Mapping[str, Any]) -> dict[str, Any]:
    """Build a task-level SkyPilot ``config:`` block from a resource profile.

    Vendor container images sometimes need pod-level accommodations that
    ``resources:`` cannot express -- e.g. NVIDIA's NRE image ships no ``sudo``
    (which SkyPilot's Kubernetes bootstrap calls unconditionally) and needs a
    ``/dev/shm`` far larger than the 64 MB Kubernetes default. Declaring those on
    the resource profile keeps them versioned with the spec instead of requiring a
    hand-passed global config, which would mean duplicating tenant/project
    identifiers into a committed file.
    """
    kubernetes = resources.get("kubernetes") if isinstance(resources, Mapping) else None
    if not isinstance(kubernetes, Mapping):
        return {}
    selected = {
        key: kubernetes[key]
        for key in TASK_CONFIG_KUBERNETES_FIELDS
        if kubernetes.get(key) not in (None, "", {}, [])
    }
    return {"kubernetes": selected} if selected else {}


def tool_image_key(tool_ref: str) -> str | None:
    """Return the CONTAINER_IMAGE_NAMES key for a toolRef, if known."""

    if not tool_ref:
        return None
    if tool_ref in TOOL_REF_IMAGE_TOOL:
        return TOOL_REF_IMAGE_TOOL[tool_ref]
    # Longest-prefix match.
    best = ""
    for prefix, tool in TOOL_REF_IMAGE_TOOL.items():
        if tool_ref == prefix or tool_ref.startswith(prefix + "."):
            if len(prefix) > len(best):
                best = prefix
    return TOOL_REF_IMAGE_TOOL.get(best)


def resolve_task_image(
    tool_ref: str,
    resources: Mapping[str, Any],
    *,
    options: SkypilotRenderOptions,
) -> str:
    """Resolve a fully-qualified image ref for one planned step."""

    if tool_ref in options.image_overrides:
        return str(options.image_overrides[tool_ref] or "").strip()
    if "*" in options.image_overrides:
        return str(options.image_overrides["*"] or "").strip()

    explicit = str(resources.get("image") or "").strip()
    if explicit:
        return explicit

    tool = tool_image_key(tool_ref)
    if not tool:
        return ""

    from npa.deploy.images import container_image_for_tool

    kwargs: dict[str, Any] = {}
    if options.registry:
        kwargs["registry"] = options.registry
    if tool == "sonic":
        if options.gpu_target:
            kwargs["gpu_target"] = options.gpu_target
        if options.image_variant:
            kwargs["image_variant"] = options.image_variant
    return container_image_for_tool(tool, **kwargs)


def render_task_run_script(command: Sequence[str], *, preamble: str = "") -> str:
    """Turn an argv list into a SkyPilot ``run:`` shell script.

    ``preamble`` is shell inserted just before the command (after the npa
    interpreter shim), e.g. to launch a self-hosted server the command connects
    to.
    """

    if not command:
        raise NpaWorkflowRenderError("cannot render empty command for SkyPilot task")
    quoted = " ".join(shlex.quote(str(part)) for part in command)
    preamble_block = f"{preamble.rstrip(chr(10))}\n" if preamble.strip() else ""
    return (
        "set -euo pipefail\n"
        # Use unbraced $HOME/$PATH so SkyPilot placeholder lint stays clean.
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        # Interpreter-independent import path for npa. `pip install -e` binds npa to
        # whichever python ran pip, and the command below runs through `bash -lc`,
        # whose login profile can resolve a DIFFERENT python3 (observed on SkyPilot's
        # GPU default image: the outer shell imports npa fine, the login shell does
        # not). Prepending the staged source tree unconditionally fixes every shell;
        # it is the same package, so it is a no-op where the install already works.
        # Images activate their toolchain either through docker ENV (inherited) or
        # through profile scripts; source the latter best-effort so dropping the login
        # shell (see scheduler.build_scheduler_task) changes nothing for them.
        "set +u\n"
        "if [ -d /etc/profile.d ]; then\n"
        "  for profile in /etc/profile.d/*.sh; do\n"
        "    [ -r \"$profile\" ] && . \"$profile\" || true\n"
        "  done\n"
        "fi\n"
        # Make `python3` mean "the interpreter npa is installed in" for this stage.
        # setup records its absolute path (sys.executable) in /tmp/npa-python.
        # UNCONDITIONAL: never gate this on whether *this* shell can import npa. The
        # stage command runs in its own `bash -c`, which resolves python3 differently
        # (live: SkyPilot's run shell expanded the Isaac image's `python3` alias and
        # imported npa fine, while the stage's non-login shell got the raw kit python
        # and failed). The shim is a no-op when the recorded interpreter is already
        # what python3 means.
        # A branch overlay has to actually win. Installing it editable only shadows a
        # baked npa if pip's uninstall removes whatever path hook the image left
        # behind, and that is not guaranteed: a workbench image whose own npa was
        # installed by a different pip/backend keeps a .pth pointing at the baked tree,
        # the overlay install reports success, and the stage silently runs the image's
        # older code (live: `No such command 'cosmos2'` from an image built for
        # cosmos2). PYTHONPATH is checked before site-packages, so state the intent
        # instead of relying on the install to displace it.
        # Written without ${...} so the rendered YAML stays free of anything SkyPilot
        # would read as one of its own placeholders.
        "if [ -d /tmp/npa-src-overlay/src ]; then\n"
        "  if [ -n \"$PYTHONPATH\" ]; then\n"
        "    PYTHONPATH=\"/tmp/npa-src-overlay/src:$PYTHONPATH\"\n"
        "  else\n"
        "    PYTHONPATH=/tmp/npa-src-overlay/src\n"
        "  fi\n"
        "  export PYTHONPATH\n"
        "fi\n"
        "npa_python=\"\"\n"
        "if [ -s /tmp/npa-python ]; then\n"
        "  npa_python=\"$(cat /tmp/npa-python)\"\n"
        "  if [ ! -x \"$npa_python\" ]; then\n"
        "    echo \"recorded npa interpreter is not executable: $npa_python\" >&2\n"
        "    npa_python=\"\"\n"
        "  fi\n"
        "fi\n"
        "if [ -n \"$npa_python\" ]; then\n"
        "  mkdir -p /tmp/npa-shim\n"
        "  printf '#!/bin/sh\\nexec \"%s\" \"$@\"\\n' \"$npa_python\" "
        "> /tmp/npa-shim/python3\n"
        "  chmod +x /tmp/npa-shim/python3\n"
        "  export PATH=\"/tmp/npa-shim:$PATH\"\n"
        # Console scripts installed next to that interpreter must be resolvable by
        # name too, which is the same gap the `npa` symlink in setup works around.
        # Live: vLLM's FlashInfer JIT shells out to `ninja`, which ships as a vLLM
        # dependency in the interpreter's bin dir — a directory that is not on the
        # stage shell's PATH, which is the whole reason the shim above exists.
        # Appended, not prepended, so it cannot shadow a system tool.
        "  npa_scripts=\"$(\"$npa_python\" -c 'import sysconfig; "
        "print(sysconfig.get_path(\"scripts\"))' 2>/dev/null || true)\"\n"
        "  if [ -n \"$npa_scripts\" ] && [ -d \"$npa_scripts\" ]; then\n"
        "    export PATH=\"$PATH:$npa_scripts\"\n"
        "  fi\n"
        "  echo \"using npa interpreter $npa_python for this stage\" >&2\n"
        "fi\n"
        "python3 -c 'import npa' >/dev/null 2>&1 || "
        "echo 'warning: python3 in this shell cannot import npa' >&2\n"
        "set -u\n"
        f"{preamble_block}"
        f"{quoted}\n"
    )


#: Readiness budget the eval client waits for a self-hosted server, in seconds.
#: Override per spec with ``config.vlm_ready_timeout_s``.
DEFAULT_VLM_READY_TIMEOUT_S = 1800


def self_hosted_vlm_model(config: Mapping[str, Any]) -> str:
    """Which VLM a self-hosted stage serves. ``config.vlm_model`` wins."""

    try:
        from npa.workbench.vlm_eval import DEFAULT_MODEL as _default_model
    except Exception:  # pragma: no cover - fallback keeps render import-light
        _default_model = "Qwen/Qwen2-VL-7B-Instruct"
    raw = config.get("vlm_model") or config.get("vlm_models") or _default_model
    return str(raw).split(",")[0].strip()


def _vllm_serve_preamble(tool_ref: str, config: Mapping[str, Any]) -> str:
    """Background vLLM server launch for a self-hosted ``vlm_eval`` step.

    The npa.workflow render otherwise only installs vLLM (see
    ``render_setup_for_tool``) but never starts the OpenAI-compatible server the
    eval client connects to, so :8000 is never up. Launch it in the background
    on the eval's default endpoint; the client polls ``/v1/models`` and retries
    connect errors up to ``NPA_VLM_READY_TIMEOUT_S`` (see
    ``npa.workbench.vlm_eval``), so a cold start is a bounded wait.
    """

    backend = str(config.get("vlm_backend") or "").strip().lower()
    if not (tool_ref.startswith("workbench.vlm_eval") and backend in {"self-hosted", "self_hosted"}):
        return ""
    if str(tool_ref).startswith("workbench.vlm_eval.benchmark"):
        # The benchmark twin runs backend=stub / packaged fixtures; no server.
        return ""
    model = self_hosted_vlm_model(config)
    model_q = shlex.quote(model)
    ready_timeout = str(config.get("vlm_ready_timeout_s") or DEFAULT_VLM_READY_TIMEOUT_S).strip()
    return (
        "# Self-hosted vLLM: launch the OpenAI-compatible server in the background\n"
        "# on 127.0.0.1:8000; the eval client waits for readiness. Export the\n"
        "# served model so the client asks for THIS model instead of the library\n"
        "# default (a mismatch is a 404 from the server). Unbraced/plain\n"
        "# assignment keeps SkyPilot placeholder lint clean.\n"
        f"export NPA_VLM_READY_TIMEOUT_S={shlex.quote(ready_timeout)}\n"
        f"export NPA_VLM_SELF_HOSTED_MODEL={model_q}\n"
        # vLLM's FlashInfer sampler JIT-compiles a CUDA kernel on first use, and
        # a task image with a GPU driver does not necessarily ship a CUDA
        # TOOLKIT: live on RTXPRO-6000 the engine died at warmup with
        # "/usr/local/cuda/bin/nvcc: not found". The native torch sampler needs
        # no compiler.
        "export VLLM_USE_FLASHINFER_SAMPLER=0\n"
        "python3 -m vllm.entrypoints.openai.api_server "
        "--host 127.0.0.1 --port 8000 "
        f"--model {model_q} --served-model-name {model_q} --trust-remote-code "
        "> /tmp/vllm-server.log 2>&1 &\n"
        "vllm_pid=$!\n"
        "trap 'kill \"$vllm_pid\" 2>/dev/null || true' EXIT\n"
        # Wait for the server here rather than leaving it to the eval client's
        # retry loop. A server that DIES during startup (live: FlashInfer's JIT
        # sampling kernel needs ninja, which the image lacked) is otherwise
        # indistinguishable from one that is still loading, and the job burns the
        # whole readiness window before reporting a connection error with no
        # server-side detail.
        "vllm_deadline=$(( $(date +%s) + NPA_VLM_READY_TIMEOUT_S ))\n"
        "vllm_ready=0\n"
        "while [ \"$(date +%s)\" -lt \"$vllm_deadline\" ]; do\n"
        "  if ! kill -0 \"$vllm_pid\" 2>/dev/null; then\n"
        "    echo 'vLLM server exited during startup; last 60 log lines:' >&2\n"
        "    tail -n 60 /tmp/vllm-server.log >&2\n"
        "    exit 1\n"
        "  fi\n"
        "  if python3 -c \"import urllib.request; "
        "urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)\" "
        ">/dev/null 2>&1; then\n"
        "    vllm_ready=1\n"
        "    break\n"
        "  fi\n"
        "  sleep 5\n"
        "done\n"
        "if [ \"$vllm_ready\" != 1 ]; then\n"
        "  echo 'vLLM server never became ready; last 60 log lines:' >&2\n"
        "  tail -n 60 /tmp/vllm-server.log >&2\n"
        "  exit 1\n"
        "fi\n"
        "echo \"vLLM server ready on 127.0.0.1:8000\" >&2\n"
    )


def default_npa_setup() -> str:
    """Ensure the ``npa`` CLI is available on the SkyPilot worker.

    Workbench images bake npa at ``/opt/nebius-physical-ai/npa``. When a task
    uses SkyPilot's default image (e.g. Token Factory API twins), setup can:

    1. install from a mounted ``/tmp/npa-src`` (S3 URI via ``NPA_SRC_S3_URI``), or
    2. sync from ``NPA_SRC_S3_URI`` with the AWS CLI / boto3, then install.
    """

    return (
        "set -e\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        # Debian/Ubuntu >= 24.04 mark the system interpreter externally managed
        # (PEP 668), so a plain `pip install` fails with
        # "error: externally-managed-environment". A task container is disposable, so
        # retry with --break-system-packages and then --user before giving up. Live:
        # this is what the Isaac Lab image hit once its system python3 came first on
        # PATH, and any Ubuntu 24.04 based image would hit it too.
        "npa_pip_install() {\n"
        "  target=\"$1\"\n"
        "  shift\n"
        "  python3 -m pip install -q \"$target\" \"$@\" \\\n"
        "    || python3 -m pip install -q \"$target\" \"$@\" --break-system-packages \\\n"
        "    || python3 -m pip install -q \"$target\" \"$@\" --user\n"
        "}\n"
        "if ! command -v npa >/dev/null 2>&1; then\n"
        "  if [ -d /opt/nebius-physical-ai/npa ]; then\n"
        "    npa_pip_install -e /opt/nebius-physical-ai/npa\n"
        "  else\n"
        "    if [ ! -d /tmp/npa-src ] && [ -n \"$NPA_SRC_S3_URI\" ]; then\n"
        "      npa_pip_install boto3\n"
        "      python3 - <<'PY'\n"
        "import os, pathlib\n"
        "from urllib.parse import urlparse\n"
        "import boto3\n"
        "from botocore.client import Config\n"
        "uri = os.environ['NPA_SRC_S3_URI'].rstrip('/')\n"
        "parsed = urlparse(uri if '://' in uri else f's3://{uri}')\n"
        "bucket, prefix = parsed.netloc, parsed.path.lstrip('/')\n"
        "dest = pathlib.Path('/tmp/npa-src')\n"
        "dest.mkdir(parents=True, exist_ok=True)\n"
        "print('syncing', uri, '->', dest, flush=True)\n"
        "kwargs = {'config': Config(signature_version='s3v4')}\n"
        "if os.environ.get('AWS_ENDPOINT_URL'):\n"
        "    kwargs['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']\n"
        "s3 = boto3.client('s3', **kwargs)\n"
        "token = None\n"
        "while True:\n"
        "    kw = {'Bucket': bucket, 'Prefix': prefix}\n"
        "    if token:\n"
        "        kw['ContinuationToken'] = token\n"
        "    resp = s3.list_objects_v2(**kw)\n"
        "    for obj in resp.get('Contents') or ():\n"
        "        key = obj['Key']\n"
        "        rel = key[len(prefix):].lstrip('/') if prefix else key\n"
        "        if not rel or key.endswith('/'):\n"
        "            continue\n"
        "        out = dest / rel\n"
        "        out.parent.mkdir(parents=True, exist_ok=True)\n"
        "        s3.download_file(bucket, key, str(out))\n"
        "    if not resp.get('IsTruncated'):\n"
        "        break\n"
        "    token = resp.get('NextContinuationToken')\n"
        "PY\n"
        "    fi\n"
        "    if [ -d /tmp/npa-src ]; then\n"
        "      npa_pip_install -e /tmp/npa-src\n"
        "    else\n"
        "      echo 'npa CLI not found; set NPA_SRC_S3_URI or use a workbench image' >&2\n"
        "      exit 1\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        # Opt-in branch overlay: reinstall npa from NPA_SRC_S3_URI on TOP of a
        # baked workbench image so branch code (e.g. a new augment prompt path)
        # actually runs on GPU without rebuilding the image. Default off (no-op).
        "if [ \"$NPA_SRC_OVERLAY\" = \"1\" ] && [ -n \"$NPA_SRC_S3_URI\" ]; then\n"
        "  npa_pip_install boto3\n"
        "  python3 - <<'PY'\n"
        "import os, pathlib\n"
        "from urllib.parse import urlparse\n"
        "import boto3\n"
        "from botocore.client import Config\n"
        "uri = os.environ['NPA_SRC_S3_URI'].rstrip('/')\n"
        "parsed = urlparse(uri if '://' in uri else f's3://{uri}')\n"
        "bucket, prefix = parsed.netloc, parsed.path.lstrip('/')\n"
        "dest = pathlib.Path('/tmp/npa-src-overlay')\n"
        "dest.mkdir(parents=True, exist_ok=True)\n"
        "print('overlay syncing', uri, '->', dest, flush=True)\n"
        "kwargs = {'config': Config(signature_version='s3v4')}\n"
        "if os.environ.get('AWS_ENDPOINT_URL'):\n"
        "    kwargs['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']\n"
        "s3 = boto3.client('s3', **kwargs)\n"
        "token = None\n"
        "while True:\n"
        "    kw = {'Bucket': bucket, 'Prefix': prefix}\n"
        "    if token:\n"
        "        kw['ContinuationToken'] = token\n"
        "    resp = s3.list_objects_v2(**kw)\n"
        "    for obj in resp.get('Contents') or ():\n"
        "        key = obj['Key']\n"
        "        rel = key[len(prefix):].lstrip('/') if prefix else key\n"
        "        if not rel or key.endswith('/'):\n"
        "            continue\n"
        "        out = dest / rel\n"
        "        out.parent.mkdir(parents=True, exist_ok=True)\n"
        "        s3.download_file(bucket, key, str(out))\n"
        "    if not resp.get('IsTruncated'):\n"
        "        break\n"
        "    token = resp.get('NextContinuationToken')\n"
        "PY\n"
        "  npa_pip_install -e /tmp/npa-src-overlay --no-deps\n"
        # Same reason as the stage preamble: the install alone is not enough to
        # displace a baked npa, so make the overlay explicit for the rest of setup too
        # (the interpreter recorded below is checked with `import npa`).
        "  if [ -n \"$PYTHONPATH\" ]; then\n"
        "    PYTHONPATH=\"/tmp/npa-src-overlay/src:$PYTHONPATH\"\n"
        "  else\n"
        "    PYTHONPATH=/tmp/npa-src-overlay/src\n"
        "  fi\n"
        "  export PYTHONPATH\n"
        "fi\n"
        # Record the interpreter that can actually import npa, i.e. the one pip just
        # installed into (it has npa AND its dependencies). Stage bodies use it via a
        # PATH shim, because a task image's default `python3` may be a different
        # interpreter entirely: SkyPilot's GPU default image ships /usr/bin/python3
        # with no pip, and the Isaac Lab image's PATH python3 is Isaac's kit python.
        # Record a python COMMAND that can import npa, so stage bodies can be pointed
        # at it. Three candidates are tried in order, because each of them is the right
        # answer on some real image:
        #   1. sys.executable - correct on normal images;
        #   2. the alias target - the Isaac Lab image aliases python3 to
        #      /workspace/isaaclab/_isaac_sim/python.sh, and its embedded kit python
        #      cannot import its own site-packages unless launched through that
        #      wrapper (live run: "could not record a usable npa interpreter");
        #   3. `type -P python3` - the PATH binary, ignoring any alias.
        "python3 -c 'import npa' >/dev/null 2>&1 || "
        "{ echo 'npa is not importable after setup' >&2; exit 1; }\n"
        "npa_python=\"\"\n"
        "alias_target=\"$(alias python3 2>/dev/null | sed -e \"s/^alias python3=//\" "
        "-e \"s/^'//\" -e \"s/'$//\")\"\n"
        "for candidate in \"$(python3 -c 'import sys; print(sys.executable)' "
        "2>/dev/null || true)\" \"$alias_target\" \"$(type -P python3 2>/dev/null "
        "|| true)\"; do\n"
        "  if [ -n \"$candidate\" ] && [ -x \"$candidate\" ] && "
        "\"$candidate\" -c 'import npa' >/dev/null 2>&1; then\n"
        "    npa_python=\"$candidate\"\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "if [ -n \"$npa_python\" ]; then\n"
        "  echo \"$npa_python\" > /tmp/npa-python\n"
        "  echo \"npa interpreter recorded: $npa_python\" >&2\n"
        "else\n"
        "  echo 'warning: no python command outside this shell could import npa' >&2\n"
        "fi\n"
        # toolRef stages invoke the `npa` console script by name; installing into a
        # non-standard interpreter can leave it outside PATH, so link it where every
        # shell will find it.
        "if [ ! -x /usr/local/bin/npa ]; then\n"
        "  scripts_dir=\"$(python3 -c 'import sysconfig; "
        "print(sysconfig.get_path(\"scripts\"))' 2>/dev/null || true)\"\n"
        "  if [ -n \"$scripts_dir\" ] && [ -x \"$scripts_dir/npa\" ]; then\n"
        "    ln -sf \"$scripts_dir/npa\" /usr/local/bin/npa 2>/dev/null || "
        "sudo -n ln -sf \"$scripts_dir/npa\" /usr/local/bin/npa 2>/dev/null || true\n"
        "  fi\n"
        "fi\n"
    )


#: rerun-sdk requirement installed into NuRec stage pods.
#:
#: Must equal npa's own ``viz`` extra in ``npa/pyproject.toml``. There is no import
#: that can enforce that -- pyproject is data, not code, and parsing it at runtime
#: from an installed wheel is unreliable -- so the guarantee is provided by
#: ``test_renderer_nurec_rerun_pin_matches_the_packaged_extra``, which fails if the
#: two ever diverge. (An earlier version imported a ``_rerun_pin`` symbol that does
#: not exist and silently fell back to this literal, so its "cannot drift" promise
#: never actually engaged.)
NUREC_RERUN_PIN = "rerun-sdk==0.31.4"


def _sonic_deps_setup() -> str:
    """Install the torch/ONNX stack a SONIC stage needs, if it is not baked in.

    The npa-sonic image already ships them, and the check below makes this a
    no-op there. On the default SkyPilot image (which is what a workflow run
    with ``--image none`` uses) SONIC train/export/eval would otherwise fail
    with "requires torch" after the cluster is already up.
    """

    return (
        "python3 - <<'PY'\n"
        "import importlib.util\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "REQUIRED = (\n"
        "    ('torch', 'torch>=2.12.1'),\n"
        "    ('onnx', 'onnx>=1.16'),\n"
        "    ('onnxscript', 'onnxscript>=0.5'),\n"
        "    ('onnxruntime', 'onnxruntime>=1.18'),\n"
        ")\n"
        "missing = [spec for module, spec in REQUIRED if importlib.util.find_spec(module) is None]\n"
        "if missing:\n"
        "    base = [sys.executable, '-m', 'pip', 'install', '-q', *missing]\n"
        "    for extra in ([], ['--break-system-packages'], ['--user']):\n"
        "        if subprocess.call(base + extra) == 0:\n"
        "            break\n"
        "    else:\n"
        "        raise SystemExit('failed to install SONIC dependencies: ' + ', '.join(missing))\n"
        "PY\n"
    )


def _vllm_install_setup(model: str) -> str:
    """Install vLLM and pre-fetch the served weights during ``setup``.

    Two things dominate a self-hosted cold start on a fresh node: resolving and
    downloading the vLLM + CUDA wheel set, and pulling the model weights. Use
    ``uv`` for the former (it resolves and downloads in parallel; plain pip took
    long enough that the eval's readiness window expired) and ``hf_transfer``
    for the latter, so the ``run`` phase only has to load already-local weights.
    """

    return (
        f"export NPA_VLM_SETUP_MODEL={shlex.quote(model)}\n"
        "python3 - <<'PY'\n"
        "import importlib.util\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "MODEL = os.environ['NPA_VLM_SETUP_MODEL']\n"
        "\n"
        "def pip_install(*packages):\n"
        "    for prefix in (\n"
        "        [sys.executable, '-m', 'uv', 'pip', 'install', '--python', sys.executable],\n"
        "        [sys.executable, '-m', 'pip', 'install', '-q'],\n"
        "        [sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages'],\n"
        "    ):\n"
        "        if subprocess.call([*prefix, *packages]) == 0:\n"
        "            return True\n"
        "    return False\n"
        "\n"
        "if importlib.util.find_spec('uv') is None:\n"
        "    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])\n"
        "if importlib.util.find_spec('vllm') is None and not pip_install('vllm>=0.8.5'):\n"
        "    raise SystemExit('failed to install vllm for the self-hosted VLM backend')\n"
        "pip_install('hf_transfer')\n"
        "os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')\n"
        "try:\n"
        "    from huggingface_hub import snapshot_download\n"
        "except ImportError:\n"
        "    print('huggingface_hub unavailable; vLLM will fetch weights at startup')\n"
        "else:\n"
        "    print('pre-fetching', MODEL, flush=True)\n"
        "    snapshot_download(MODEL)\n"
        "PY\n"
    )


def render_setup_for_tool(
    tool_ref: str,
    *,
    config: Mapping[str, Any],
    options: SkypilotRenderOptions,
) -> str:
    """Return a SkyPilot ``setup:`` block for a toolRef."""

    if not options.default_setup:
        return ""
    parts = [default_npa_setup()]
    backend = str(config.get("vlm_backend") or "").strip().lower()
    if tool_ref.startswith("workbench.vlm_eval") and backend in {"self-hosted", "self_hosted"}:
        parts.append(_vllm_install_setup(self_hosted_vlm_model(config)))
    if tool_ref.startswith("workbench.sonic"):
        parts.append(_sonic_deps_setup())
    if tool_ref.startswith("workbench.token_factory"):
        # Avoid ${VAR:-} bash forms so SkyPilot placeholder lint stays clean.
        parts.append(
            "if [[ -z \"$NEBIUS_TOKEN_FACTORY_KEY\" ]]; then\n"
            "  echo 'NEBIUS_TOKEN_FACTORY_KEY is required. Pass it with --secret-env "
            "NEBIUS_TOKEN_FACTORY_KEY' >&2\n"
            "  exit 1\n"
            "fi\n"
        )
    if tool_ref.startswith("workbench.nurec"):
        # These stages run inside NVIDIA's NRE container -- a VENDOR image, so it
        # carries none of the tool's runtime dependencies: no Hugging Face CLI
        # (dataset download), no nvidia-ncore (the rig->world pose derivation NRE
        # requires), no rerun-sdk (the run recording; it is only an optional `viz`
        # extra of npa), and no ffmpeg (`nre render --export-video`). The image also
        # ships no `unzip`, which is why the tool extracts with stdlib zipfile.
        # Installing into the interpreter npa was installed into (recorded by
        # default_npa_setup) avoids a second, npa-less python winning on PATH.
        parts.append(
            "set -e\n"
            "if ! command -v ffmpeg >/dev/null 2>&1; then\n"
            "  export DEBIAN_FRONTEND=noninteractive\n"
            "  apt-get update -qq || true\n"
            "  apt-get install -y -qq --no-install-recommends ffmpeg || true\n"
            "fi\n"
            "npa_nurec_py=python3\n"
            "if [ -s /tmp/npa-python ]; then npa_nurec_py=\"$(cat /tmp/npa-python)\"; fi\n"
            # --break-system-packages FIRST: the NRE image is Ubuntu 24.04, whose
            # interpreter is externally managed (PEP 668), so the plain form always
            # fails there and only adds a confusing "error:
            # externally-managed-environment" to the logs before the fallback wins.
            "npa_nurec_pip() {\n"
            "  \"$npa_nurec_py\" -m pip install -q \"$@\" --break-system-packages \\\n"
            "    || \"$npa_nurec_py\" -m pip install -q \"$@\" \\\n"
            "    || \"$npa_nurec_py\" -m pip install -q \"$@\" --user\n"
            "}\n"
            f"npa_nurec_pip 'huggingface_hub>=0.30' 'nvidia-ncore' '{NUREC_RERUN_PIN}' 'pillow>=10.0'\n"
            "\"$npa_nurec_py\" -c 'import ncore, rerun; print(\"nurec runtime deps ready\")'\n"
        )
    return "".join(parts)


def secret_env_hints_for_plan(steps: Sequence[PlanStep]) -> tuple[str, ...]:
    """Collect recommended ``--secret-env`` names for a planned workflow."""

    hints: list[str] = []
    seen: set[str] = set()
    for step in steps:
        tool_ref = step.tool_ref or ""
        for prefix, names in SECRET_ENV_HINTS.items():
            if tool_ref == prefix or tool_ref.startswith(prefix + "."):
                for name in names:
                    if name not in seen:
                        seen.add(name)
                        hints.append(name)
    return tuple(hints)


def build_skypilot_task_doc(
    spec: NpaWorkflowSpec,
    step: PlanStep,
    *,
    run_id: str,
    options: SkypilotRenderOptions,
) -> dict[str, Any]:
    """Build one SkyPilot task document from a planned step."""

    scheduler_task = build_scheduler_task(spec, step, run_id=run_id)
    resources = normalize_resources(scheduler_task.get("resources") or {})
    image = resolve_task_image(
        str(scheduler_task.get("tool_ref") or ""),
        scheduler_task.get("resources") or {},
        options=options,
    )
    if image:
        resources["image_id"] = f"docker:{image}" if not image.startswith("docker:") else image

    command = list(scheduler_task.get("command") or [])
    if not command:
        raise NpaWorkflowRenderError(
            f"planned step {scheduler_task['name']!r} has no command to render"
        )

    envs: dict[str, str] = {
        "NPA_WORKFLOW_NAME": spec.name,
        "NPA_WORKFLOW_RUN_ID": run_id,
        "NPA_WORKFLOW_STATE": str(scheduler_task["name"]),
    }
    if options.include_aws_endpoint and options.aws_endpoint_url:
        envs["AWS_ENDPOINT_URL"] = options.aws_endpoint_url
    if image:
        envs["NPA_TASK_IMAGE"] = image.removeprefix("docker:")
    # Opt-in passthrough: when set at submit, propagate Cosmos input-conditioning
    # knobs to stage pods so the augment conditions on the run's real input clip
    # (edge control) instead of the bundled example. Unset by default → no change.
    import os as _os_cond

    for _cond_var in (
        "NPA_COSMOS_CONDITION_ON_INPUT",
        "NPA_COSMOS_CONTROL",
        "NPA_COSMOS_CONTROL_WEIGHT",
        "NPA_COSMOS_GUIDANCE",
    ):
        _cond_val = str(_os_cond.environ.get(_cond_var) or "").strip()
        if _cond_val:
            envs[_cond_var] = _cond_val

    doc: dict[str, Any] = {
        "name": scheduler_task["name"],
        "resources": resources,
        "envs": envs,
        "run": render_task_run_script(
            command,
            preamble=_vllm_serve_preamble(str(scheduler_task.get("tool_ref") or ""), spec.config),
        ),
    }
    task_config = normalize_task_config(scheduler_task.get("resources") or {})
    if task_config:
        doc["config"] = task_config
    setup = render_setup_for_tool(
        str(scheduler_task.get("tool_ref") or ""),
        config=spec.config,
        options=options,
    )
    if setup.strip():
        doc["setup"] = setup
    # When no workbench image is pinned, point setup at an existing S3 copy of
    # the npa package (SkyPilot local file_mounts create new buckets and fail
    # on Nebius). Operators set NPA_SRC_S3_URI=s3://bucket/prefix/npa.
    import os

    src_uri = (
        os.environ.get("NPA_SRC_S3_URI") or os.environ.get("NPA_E2E_NPA_SRC_S3_URI") or ""
    ).strip()
    if not image:
        if not src_uri:
            raise NpaWorkflowRenderError(
                f"planned step {scheduler_task['name']!r} has no workbench image "
                "and NPA_SRC_S3_URI is unset; set NPA_SRC_S3_URI=s3://bucket/prefix/npa "
                "or pass --image <registry>/npa-<tool>:<tag>"
            )
        envs["NPA_SRC_S3_URI"] = src_uri
        doc["envs"] = envs
    else:
        # A pinned image is EITHER an NPA workbench image with npa baked in, OR a
        # VENDOR image that has never heard of npa (e.g. NVIDIA's NRE container,
        # which is the runtime for the neural-reconstruction workflow). Propagating
        # the source URI serves both: setup's primary install path is guarded by
        # `command -v npa`, so it is a no-op when npa is already present and
        # installs it WITH dependencies when it is not. Without this, a vendor image
        # fails setup with "npa CLI not found; set NPA_SRC_S3_URI or use a workbench
        # image" (observed live on the NRE image).
        if src_uri:
            envs["NPA_SRC_S3_URI"] = src_uri
            doc["envs"] = envs
        # Opt-in overlay: reinstall branch npa ON TOP of a baked image (--no-deps),
        # used to run un-imaged branch code on GPU without rebuilding the image.
        if str(os.environ.get("NPA_SRC_OVERLAY") or "").strip() in {"1", "true", "True"} and src_uri:
            envs["NPA_SRC_OVERLAY"] = "1"
            doc["envs"] = envs
    _inject_nebius_registry_docker_secrets(
        doc,
        materialize=options.materialize_registry_secrets,
    )
    return doc


def _is_nebius_registry_image(image_id: str) -> bool:
    value = image_id.removeprefix("docker:").strip()
    host = value.split("/", 1)[0] if "/" in value else ""
    return host.startswith("cr.") and host.endswith(".nebius.cloud")


def _inject_nebius_registry_docker_secrets(
    doc: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Embed SkyPilot Docker login secrets for private Nebius registry images.

    Matches the burst submit path: ``resources.image_id`` is pulled before YAML
    ``setup`` runs, so registry auth must live in task ``secrets``.

    When ``materialize`` is False (plan-only), embed a placeholder password so
    rendered YAML can be printed without minting or leaking live IAM tokens.
    """

    import os

    resources = doc.get("resources") or {}
    if not isinstance(resources, dict):
        return
    cloud = str(resources.get("cloud") or "").strip().lower()
    image_id = str(resources.get("image_id") or "").strip()
    # Nebius VMs need SKYPILOT_DOCKER_* for private pulls; k8s uses imagePullSecrets
    # but still benefits from secrets when the controller falls back to docker login.
    if cloud not in {"nebius", "kubernetes", "k8s"} or not _is_nebius_registry_image(image_id):
        return

    server = image_id.removeprefix("docker:").split("/", 1)[0]
    # Registry/credentials consistency guard (applies to EVERY stage image, not
    # just Cosmos): SkyPilot logs into the image's registry host using
    # SKYPILOT_DOCKER_PASSWORD. If that password authenticates to a DIFFERENT
    # registry (SKYPILOT_DOCKER_SERVER), the pull is a 403 ErrImagePull that stalls
    # provisioning. Fail fast with an actionable message at submit time. Only
    # enforced when actually materializing creds (real submit), never plan-only.
    if materialize:
        creds_server = str(os.environ.get("SKYPILOT_DOCKER_SERVER") or "").strip()
        if creds_server and creds_server != server:
            raise NpaWorkflowRenderError(
                f"registry mismatch: task image is in {server!r} but the Docker "
                f"credentials (SKYPILOT_DOCKER_SERVER) authenticate to {creds_server!r}. "
                "Pinning images from a registry your credentials cannot pull causes a "
                "403 ErrImagePull for every image. Pass --registry pointing at the "
                f"credentials' registry {creds_server!r} (e.g. the primary workbench "
                f"registry), or set SKYPILOT_DOCKER_* for {server!r}."
            )
    username = (
        os.environ.get("SKYPILOT_DOCKER_USERNAME")
        or os.environ.get("NPA_REGISTRY_USERNAME")
        or "iam"
    )
    if materialize:
        password = (
            os.environ.get("SKYPILOT_DOCKER_PASSWORD")
            or os.environ.get("NPA_REGISTRY_PASSWORD")
            or ""
        )
        if not password:
            try:
                from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token

                password = mint_nebius_registry_token()
            except Exception as exc:  # noqa: BLE001
                raise NpaWorkflowRenderError(
                    "Nebius registry image requires SKYPILOT_DOCKER_PASSWORD "
                    f"(or mintable IAM token); failed to mint: {exc}"
                ) from exc
    else:
        password = "<SKYPILOT_DOCKER_PASSWORD>"

    secrets = doc.setdefault("secrets", {})
    if not isinstance(secrets, dict):
        raise NpaWorkflowRenderError("SkyPilot task secrets must be a mapping")
    secrets.setdefault("SKYPILOT_DOCKER_SERVER", server)
    secrets.setdefault("SKYPILOT_DOCKER_USERNAME", username)
    secrets.setdefault("SKYPILOT_DOCKER_PASSWORD", password)


def render_skypilot_yaml(
    spec: NpaWorkflowSpec,
    plan: ExecutionPlan,
    *,
    run_id: str,
    options: SkypilotRenderOptions | None = None,
) -> str:
    """Return multi-document SkyPilot **pipeline** YAML for a planned npa.workflow.

    This is the default, unchanged path: a flat serial chain. Concurrent fan-out
    is rendered by :func:`render_skypilot_job_group_yaml`, which is a separate
    entry point so that "serial" stays the only mode this function will ever emit.
    """

    opts = options or SkypilotRenderOptions()
    if opts.execution != "serial":
        raise NpaWorkflowRenderError(
            f"npa.workflow/v0.0.1 renderer only supports execution=serial, got {opts.execution!r}"
        )
    if not plan.steps:
        raise NpaWorkflowRenderError(f"workflow {spec.name!r} planned zero steps")
    return _render_docs(spec, plan.steps, run_id=run_id, options=opts, execution="serial")


def render_skypilot_job_group_yaml(
    spec: NpaWorkflowSpec,
    steps: Sequence[PlanStep],
    *,
    run_id: str,
    options: SkypilotRenderOptions | None = None,
    name: str = "",
) -> str:
    """Return a SkyPilot **JobGroup** YAML (``execution: parallel``) for one wave.

    SkyPilot >= 0.12 treats a multi-document YAML whose header sets
    ``execution: parallel`` as a JobGroup: every task shares one managed ``job_id``
    but launches its **own cluster concurrently**. ``primary_tasks`` is intentionally
    omitted, which marks every task primary, so the group only reaches a terminal
    state once all members do — that is the barrier the downstream ``needs:`` state
    waits on.
    """

    opts = options or SkypilotRenderOptions()
    if not steps:
        raise NpaWorkflowRenderError(
            f"workflow {spec.name!r} rendered an empty parallel group"
        )
    if len(steps) < 2:
        raise NpaWorkflowRenderError(
            "a SkyPilot JobGroup needs at least two tasks; render single-task waves "
            "with the serial pipeline renderer"
        )
    return _render_docs(
        spec,
        steps,
        run_id=run_id,
        options=opts,
        execution="parallel",
        name=name or spec.name,
    )


def render_skypilot_steps_yaml(
    spec: NpaWorkflowSpec,
    steps: Sequence[PlanStep],
    *,
    run_id: str,
    options: SkypilotRenderOptions | None = None,
    execution: str = "serial",
    name: str = "",
) -> str:
    """Render one runtime wave: a serial pipeline or a parallel JobGroup."""

    if execution == "parallel" and len(steps) > 1:
        return render_skypilot_job_group_yaml(
            spec, steps, run_id=run_id, options=options, name=name
        )
    opts = options or SkypilotRenderOptions()
    if opts.execution != "serial":
        raise NpaWorkflowRenderError(
            f"npa.workflow/v0.0.1 renderer only supports execution=serial, got {opts.execution!r}"
        )
    if not steps:
        raise NpaWorkflowRenderError(f"workflow {spec.name!r} planned zero steps")
    return _render_docs(
        spec, steps, run_id=run_id, options=opts, execution="serial", name=name or spec.name
    )


def _render_docs(
    spec: NpaWorkflowSpec,
    steps: Sequence[PlanStep],
    *,
    run_id: str,
    options: SkypilotRenderOptions,
    execution: str,
    name: str = "",
) -> str:
    header = {
        "name": name or spec.name,
        "execution": execution,
    }
    docs: list[dict[str, Any]] = [header]
    seen: set[str] = set()
    for step in steps:
        doc = build_skypilot_task_doc(spec, step, run_id=run_id, options=options)
        task_name = str(doc.get("name") or "")
        # Serial pipelines may legitimately repeat a task name (an unrolled loop
        # body re-runs the same state), so only JobGroups — whose tasks run at the
        # same time on distinct clusters — require unique names.
        if execution == "parallel":
            if task_name in seen:
                raise NpaWorkflowRenderError(
                    f"duplicate SkyPilot task name {task_name!r} in parallel group "
                    f"of workflow {spec.name!r}"
                )
            seen.add(task_name)
        docs.append(doc)

    chunks: list[str] = []
    for doc in docs:
        chunks.append(
            yaml.safe_dump(
                doc,
                sort_keys=False,
                default_flow_style=False,
                # Do not fold long lines: a wrapped shell command is unreadable in the
                # rendered YAML and stops `grep`/assertions from finding the command
                # that will actually run.
                width=10_000,
            ).rstrip()
        )
    return "\n---\n".join(chunks) + "\n"


_SKYPILOT_PLACEHOLDER_RE = __import__("re").compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def assert_no_unresolved_placeholders(yaml_text: str) -> None:
    """Fail if rendered YAML still contains SkyPilot-style ``${NAME}`` placeholders.

    Allows bash parameter expansions such as ``$NAME`` (no braces) used in setup
    scripts. Flags only bare ``${NAME}`` forms that SkyPilot would leave literal.
    """

    unresolved = sorted(set(_SKYPILOT_PLACEHOLDER_RE.findall(yaml_text)))
    if unresolved:
        joined = ", ".join(f"${{{name}}}" for name in unresolved)
        raise NpaWorkflowRenderError(
            "rendered SkyPilot YAML still contains unresolved placeholders: "
            f"{joined}; resolve images and config before submit"
        )
