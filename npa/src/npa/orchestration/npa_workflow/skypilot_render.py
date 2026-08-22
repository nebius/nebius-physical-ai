"""Render an ``npa.workflow`` execution plan as a SkyPilot multi-doc YAML."""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import ExecutionPlan, PlanStep  # noqa: F401
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec
from npa.workbench.model_cache import (
    RUNTIME_KUBERNETES,
    RUNTIME_PREMOUNTED,
    model_cache_env,
    model_cache_host_path,
    model_cache_pvc,
    pod_config_with_model_cache,
    render_model_cache_shell,
    resolve_model_cache_root,
)

# Map toolRef prefixes / exact names onto CONTAINER_IMAGE_NAMES keys.
# Token Factory is a hosted HTTP API client. Do not pin the heavy cosmos image:
# SkyPilot's k8s apt-ssh runtime setup fails inside npa-cosmos. Use the default
# SkyPilot image and stage npa via NPA_SRC_S3_URI (or an image override).
TOOL_REF_IMAGE_TOOL: dict[str, str] = {
    # Visualization only needs the prebuilt pinned Rerun runtime, not NuRec.
    "workbench.nurec.visualize": "rerun-viewer",
    "workbench.vlm_eval": "cosmos",
    "workbench.cosmos2": "cosmos2-transfer",
    # Generation runs in the Cosmos 3 framework image; the reason stage runs in the
    # (differently built) Cosmos-Reason VLM image. Exact match wins over the prefix.
    "workbench.cosmos3.generate": "cosmos3",
    "workbench.cosmos3.generate_variants": "cosmos3",
    "workbench.cosmos3.prepare_video_input": "cosmos3",
    "workbench.cosmos3.checkpoint_eval": "cosmos3",
    "workbench.cosmos3": "cosmos3-reason",
    "workbench.cosmos_curate": "cosmos-curate",
    "workbench.cosmos_evaluator": "cosmos-evaluator",
    "workbench.lancedb": "lancedb",
    "workbench.detection_training": "detection-training",
    "workbench.alpamayo2_super": "alpamayo2-super",
    "workbench.fiftyone": "fiftyone",
    "workbench.rl": "isaac-lab",
    "workbench.isaac_lab": "isaac-lab",
    "workbench.lerobot": "lerobot",
    "workbench.sonic": "sonic",
    "workbench.mjlab": "sonic",
    "workbench.retargeting": "retargeting",
    "workbench.sim2real": "lerobot-vlm-rl",
    "workbench.sim2real_envgen": "envgen",
    # BYOF selects its actual workload image from config.base_image inside the
    # BYOF runner.  It is not globally an Isaac workload: public Ubuntu/CUDA
    # profiles such as Wan 2.2 must use the staged NPA runner anonymously and
    # must not inherit Isaac image routing or consent requirements.
    "workbench.genesis": "genesis",
    "workbench.groot": "groot",
}

OPENPI_TERMS_ENV = "NPA_OPENPI_ACCEPT_GEMMA_TERMS"

SECRET_ENV_HINTS: dict[str, tuple[str, ...]] = {
    "workbench.openpi": (OPENPI_TERMS_ENV,),
    "workbench.token_factory": ("NEBIUS_TOKEN_FACTORY_KEY",),
    "workbench.vlm_eval": (),
    # Attribute verification generates and answers its questions on Token Factory.
    "workbench.cosmos_evaluator": ("NEBIUS_TOKEN_FACTORY_KEY",),
    # Material and physics classification call a real hosted vision model. The
    # acquire/validate/package actions do not consume the key, but one prefix
    # hint makes the complete shipped workflow's credential requirement clear.
    "workbench.content_agents": ("NEBIUS_TOKEN_FACTORY_KEY",),
    # This entry explicitly disables the parent Cosmos3 hint: the public Nano
    # checkpoint is downloaded anonymously and this toolRef passes --no-guardrails.
    "workbench.cosmos3.text_to_image": (),
    "workbench.cosmos3": ("HF_TOKEN",),
    # Cosmos-Transfer2.5 downloads its guardrail checkpoints from a gated Hugging Face repo
    # before it will generate anything. Live job 286 got all the way into examples/inference.py
    # and died on `hf download nvidia/Cosmos-Guardrail1` with no token.
    "workbench.cosmos2": ("HF_TOKEN",),
    # Alpamayo2-Super fetches both its OpenMDW checkpoint and the separately
    # gated PhysicalAI-AV sample under the operator's accepted HF identity.
    "workbench.alpamayo2_super": ("HF_TOKEN",),
    # The default GEAR-SONIC and GR00T-N1.7 assets are public. Callers may still
    # pass HF_TOKEN for rate limits or private overrides, but it is not a preflight.
    "workbench.sonic": (),
    "workbench.groot": (),
}

# Optional dependency groups a toolRef's stage needs, declared as npa extras in
# npa/pyproject.toml. A workbench image bakes these already, but a stage running on
# SkyPilot's default image (no `--image`, npa installed from NPA_SRC_S3_URI) gets only
# the base install — and then `npa workbench sonic export` fails with the tool's own
# message: "SONIC ONNX export requires torch ... or use the npa[sonic] extra".
# Installing the extra from the SAME source tree is the existing pattern (the renderer
# already installs vLLM for self-hosted vlm_eval); it is what lets the npa.workflow
# SONIC specs run without a vendor image at all.
TOOL_REF_PIP_EXTRAS: dict[str, str] = {
    "workbench.sonic": "sonic",
    "workflow.groot.emit_learning_rrd": "viz",
    "workflow.groot.publish_learning": "viz",
}

# Declarative metadata, never a package-string passthrough.
DECLARATIVE_PIP_EXTRAS = frozenset({"viz"})

#: toolRef prefix -> third-party pip requirements the tool shells out to, with the executable
#: that proves each is present. `cosmos fetch` runs `huggingface-cli`; the retired
#: cosmos3-ea-fetch.yaml pip-installed `huggingface_hub[cli]` in its setup, and that one line
#: was the only load-bearing part of its ~60-line preamble. Dropping it made the stage fail
#: with "No such file or directory: 'huggingface-cli'" (live job 226).
#: A requirement is (probe, pip requirement). The probe is an executable name checked with
#: ``command -v``, or ``python:<module>`` checked with an import — a library has no binary to
#: look for. `lerobot policy_train` needs the latter: it materialises its dataset with
#: `huggingface_hub`, and the interpreter running npa in a vendor image is not the vendor's own
#: venv, so the library is not necessarily importable there (live job 244).
TOOL_REF_PIP_REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    # The OpenPI BYOF environment intentionally contains only upstream's
    # pinned runtime. Four-mode stages publish/read private object-storage
    # artifacts from that same interpreter, so install the NPA storage client
    # there without resolving the rest of NPA over the vendor JAX closure.
    "workbench.openpi": (("python:boto3", "boto3>=1.34"),),
    # The redistributable image security layer upgrades Transformers with
    # ``--no-deps``. GR00T commit 3df8b382 pins 4.57.3; Transformers 5.3 changes
    # PretrainedConfig dataclass behavior and the pinned GR00T model config then
    # fails during import. Restore the upstream runtime exactly for real model
    # stages. A normal targeted install also restores its Hub/tokenizers closure
    # while leaving torch and the vendor GR00T package untouched.
    "workbench.groot": (
        # The redistributable image's uv-created Python 3.10 environment has no
        # pip. A source overlay can expose the current CLI while its conditional
        # Python <3.11 dependency is still absent; live job 446 then failed on
        # ``import tomli`` before the trainer started. The common installer's uv
        # fallback targets the exact recorded interpreter.
        ("python:tomli", "tomli>=2.0.0"),
        (
            'python:transformers;assert(__import__("importlib.metadata").metadata.version("transformers")=="4.57.3")',
            "transformers==4.57.3",
        ),
    ),
    "workflow.groot.prepare_split": (("python:pyarrow", "pyarrow>=15,<22"),),
    "workflow.groot.compare_learning": (
        ("python:av", "av>=12,<17"),
        ("python:PIL", "Pillow>=10,<12"),
    ),
    "workflow.groot.emit_learning_mcap": (("python:av", "av>=12,<17"),),
    "workflow.groot.emit_learning_rrd": (("python:av", "av>=12,<17"),),
    "workflow.groot.publish_learning": (("python:av", "av>=12,<17"),),
    "workbench.cosmos.fetch": (("huggingface-cli", "huggingface_hub[cli]>=0.23,<1.0"),),
    "workbench.cosmos.check": (("huggingface-cli", "huggingface_hub[cli]>=0.23,<1.0"),),
    "workbench.lerobot.policy_train": (
        ("python:huggingface_hub", "huggingface_hub>=0.23,<1.0"),
    ),
    "workbench.lerobot.policy_rollout": (
        ("python:huggingface_hub", "huggingface_hub>=0.23,<1.0"),
    ),
    # Text-to-image clones the framework and hands its environment to uv, exactly as the
    # retired template did; the checkpoint download still goes through the HF CLI.
    "workbench.cosmos3.text_to_image": (
        ("python:huggingface_hub", "huggingface_hub[cli]>=0.23,<1.0"),
        # `python:` on purpose. Probing for a `uv` EXECUTABLE passes on SkyPilot's default image
        # — it ships one in its runtime directory — and then the stage cannot find it, because
        # that directory is on setup's PATH and not the command's (live job 291:
        # `[Errno 2] No such file or directory: 'uv'`). Probing the MODULE installs uv into the
        # interpreter the stage actually uses, which is the thing that has to be true.
        ("python:uv", "uv>=0.5"),
    ),
}

#: Prefix marking a probe as "is this python module importable?" rather than an executable.
PYTHON_MODULE_PROBE = "python:"

#: toolRef prefix -> the vendor image's own interpreter(s), in preference order.
#:
#: A vendor image keeps its libraries in its own environment (`/opt/lerobot/venv`,
#: `/isaac-sim/python.sh`), while setup installs npa into whatever `python3` resolves to —
#: SkyPilot's miniconda. The stage then runs a tool that imports the vendor library and fails
#: with `No module named 'lerobot'` (live job 245). The retired template avoided this by
#: `source /opt/lerobot/venv/bin/activate` before doing anything.
#:
#: When a candidate exists, setup installs npa INTO it and records it as the stage interpreter,
#: so the tool and the vendor library share one environment.
TOOL_REF_VENDOR_INTERPRETERS: dict[str, tuple[str, ...]] = {
    "workbench.groot.baseline_eval": ("/opt/groot/Isaac-GR00T/.venv/bin/python",),
    "workbench.groot.posttrain_eval": ("/opt/groot/Isaac-GR00T/.venv/bin/python",),
    "workbench.lerobot": ("/opt/lerobot/venv/bin/python",),
    # Isaac Lab's simulator packages live in the Omniverse kit environment, not in the image's
    # system python. Live job 267 installed npa into /usr/bin/python3 and the stage died with
    # `isaaclab is required in the Isaac Lab image: No module named 'isaaclab'`. python.sh is
    # the vendor's own launcher (it sets the kit's library paths); the kit interpreter is the
    # fallback for images that do not ship it.
    "workbench.isaac_lab": (
        "/isaac-sim/python.sh",
        "/isaac-sim/kit/python/bin/python3",
    ),
}


def tool_vendor_interpreters(tool_ref: str) -> tuple[str, ...]:
    """Return the vendor interpreters this toolRef prefers, most specific match first."""

    if tool_ref in TOOL_REF_VENDOR_INTERPRETERS:
        return TOOL_REF_VENDOR_INTERPRETERS[tool_ref]
    best = ""
    for prefix in TOOL_REF_VENDOR_INTERPRETERS:
        if (tool_ref == prefix or tool_ref.startswith(prefix + ".")) and len(
            prefix
        ) > len(best):
            best = prefix
    return TOOL_REF_VENDOR_INTERPRETERS.get(best, ())


def render_vendor_interpreter_setup(candidates: Sequence[str]) -> str:
    """Install npa into the vendor image's interpreter and make it the stage interpreter."""

    if not candidates:
        return ""
    listed = " ".join(candidates)
    # Two attempts per candidate, in this order and for opposite reasons.
    #
    # --no-deps first, because a vendor image ships a PINNED stack and resolving npa's
    # requirements inside it can bump torch, after which the vendor's own compiled extensions
    # stop loading (live job 253: torchcodec's libtorchcodec_core4.so failed with
    # `undefined symbol: _ZN3c1013MessageLogger…`, the classic torch-ABI mismatch). Where the
    # vendor environment already carries npa's dependencies — LeRobot's venv does — this is all
    # that is needed and nothing is perturbed.
    #
    # Then WITH deps, because some vendor environments carry almost none of them. Isaac Lab's
    # Omniverse kit python is one: live job 268 installed npa there with --no-deps, and the
    # probe still failed because typer/boto3/pydantic were absent. A stage that cannot import
    # npa is useless, so a perturbed-but-working environment beats a pristine broken one — and
    # the order means the risky attempt only happens when the safe one was not enough.
    attempts = (
        ("--no-deps ", "without dependencies (protects the vendor's pinned stack)"),
        ("", "with dependencies (the vendor environment lacked them)"),
    )
    install_block = ""
    for flags, why in attempts:
        install_block += (
            f"    if ! \"$npa_vendor_python\" -c 'import npa.workbench' >/dev/null 2>&1; then\n"
            f'      echo "installing npa into $npa_vendor_python {why}" >&2\n'
            f'      "$npa_vendor_python" -m pip install -q {flags}-e "$npa_vendor_src" \\\n'
            f'        || "$npa_vendor_python" -m pip install -q {flags}-e "$npa_vendor_src" '
            "--break-system-packages \\\n"
            f'        || "$npa_vendor_python" -m pip install -q {flags}-e "$npa_vendor_src" '
            "--user || true\n"
            "    fi\n"
        )
    return (
        "# Vendor image: its libraries live in its own environment, so npa has to be installed\n"
        "# there and that interpreter has to be the one the stage runs.\n"
        f"for npa_vendor_python in {listed}; do\n"
        '  [ -x "$npa_vendor_python" ] || continue\n'
        '  npa_vendor_src=""\n'
        "  if [ -s /tmp/npa-src-root ]; then\n"
        '    npa_vendor_src="$(cat /tmp/npa-src-root)"\n'
        "  elif [ -d /opt/nebius-physical-ai/npa ]; then\n"
        "    npa_vendor_src=/opt/nebius-physical-ai/npa\n"
        "  elif [ -d /tmp/npa-src ]; then\n"
        "    npa_vendor_src=/tmp/npa-src\n"
        "  fi\n"
        '  if [ -n "$npa_vendor_src" ]; then\n'
        f"{install_block}"
        "  fi\n"
        # `import npa` is not enough: a vendor image may bake a PARTIAL npa on PYTHONPATH for
        # its own entrypoint, which shadows the real install — `import npa` passes and
        # `import npa.workbench` fails (live job 250). Probe a real subpackage.
        "  if \"$npa_vendor_python\" -c 'import npa.workbench' >/dev/null 2>&1; then\n"
        '    echo "$npa_vendor_python" > /tmp/npa-python\n'
        '    echo "npa interpreter switched to vendor python: $npa_vendor_python" >&2\n'
        "    break\n"
        "  fi\n"
        # Print WHY. A bare warning sent job 268's debugging down the wrong path: the message
        # blamed a shadowing partial npa when the real cause was missing dependencies.
        '  echo "warning: npa.workbench is not importable from $npa_vendor_python:" >&2\n'
        "  \"$npa_vendor_python\" -c 'import npa.workbench' 2>&1 | tail -3 >&2 || true\n"
        "done\n"
    )


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
    # Exact tag/reference -> registry-resolved immutable digest reference. Submit
    # populates this only after pull + bootstrap-contract verification.
    image_digest_pins: Mapping[str, str] = field(default_factory=dict)
    default_setup: bool = True
    execution: str = "serial"
    aws_endpoint_url: str = field(default_factory=_default_aws_endpoint_url)
    include_aws_endpoint: bool = True
    gpu_target: str = ""
    image_variant: str = ""
    # Accelerator specs resolved against the live cluster at submit time, keyed by
    # the spec's own accelerator string. NPA_WORKFLOW_GPU_ACCELERATOR still wins.
    gpu_accelerator_overrides: Mapping[str, str] = field(default_factory=dict)
    # When False (``--plan-only``), embed placeholders instead of explicit live
    # private-registry credentials in YAML that may be printed to stdout.
    materialize_registry_secrets: bool = True
    # Shared scheduler identity for the current runtime wave attempt.  The
    # runtime supplies its durable logical launch id; offline renders derive a
    # deterministic task-local value below.  Cosmos gang workers combine this
    # with SkyPilot's launch incarnation before accepting shard manifests.
    execution_attempt_id: str = ""
    # Durable scheduler-issued ordering fence. Runtime waves monotonically
    # increase sequence; an explicit retry increases attempt within that wave.
    # Workload processes may not manufacture or advance these values.
    execution_fence_sequence: int = 1
    execution_fence_attempt: int = 1
    # Isaac acceptance defaults on so non-interactive workflows do not stop at the
    # vendor prompt. Callers retain an explicit --no-accept-eula opt-out.
    accept_eula: bool = True


def normalize_resources(
    resources: Mapping[str, Any],
    *,
    accelerator_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map an npa.workflow resource profile onto a SkyPilot ``resources`` block.

    On Kubernetes, exact ``cpus`` / ``memory`` often fail prechecks when no node
    has that precise free shape. Append ``+`` so SkyPilot can schedule on larger
    nodes (including GPU nodes with spare CPU).
    """

    import os as _os

    # Cluster-specific GPU product override: SkyPilot k8s matches on the node's
    # advertised accelerator name, which varies by cluster (e.g. RTXPRO6000 vs
    # RTXPRO-6000-BLACKWELL-SERVER-EDITION). A blanket env override still wins so
    # operators can retarget without editing the committed blueprint; otherwise
    # submit-time resolution supplies a per-profile remap.
    accel_override = str(_os.environ.get("NPA_WORKFLOW_GPU_ACCELERATOR") or "").strip()
    gpu_memory_override = str(
        _os.environ.get("NPA_WORKFLOW_GPU_MEMORY") or ""
    ).strip()
    overrides = dict(accelerator_overrides or {})

    out: dict[str, Any] = {}
    # NOTE: `num_nodes` is deliberately absent. SkyPilot puts it at the TASK level, next
    # to `resources`, so the renderer lifts it out of the profile in
    # build_skypilot_task_doc. Adding it here would produce an invalid resources block.
    for key in (
        "cloud",
        "accelerators",
        "cpus",
        "memory",
        "disk_size",
        "use_spot",
        "region",
    ):
        if key not in resources or resources[key] in (None, ""):
            continue
        value = resources[key]
        if key == "accelerators":
            selected_override = accel_override or overrides.get(str(value).strip(), "")
            # A cluster-specific product name should not silently collapse a
            # multi-GPU request. Accept either an exact ``NAME:COUNT`` override
            # or a name-only override that preserves the profile's count.
            if (
                selected_override
                and ":" not in selected_override
                and isinstance(value, str)
                and ":" in value
            ):
                _declared_name, declared_count = value.rsplit(":", 1)
                value = f"{selected_override}:{declared_count}"
            elif selected_override:
                value = selected_override
        if key == "memory":
            if gpu_memory_override and resources.get("accelerators"):
                value = gpu_memory_override
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.lower().endswith("gi"):
                    value = stripped[:-2]
                elif stripped.lower().endswith("g"):
                    value = stripped[:-1]
        out[key] = value

    cloud = str(out.get("cloud") or "").strip().lower()
    if cloud in {"kubernetes", "k8s"}:
        # SkyPilot 0.12.x accepts ``disk_size`` on Kubernetes but explicitly
        # ignores it because pods have no cloud boot disk. Preserve the profile's
        # capacity intent using SkyPilot's supported Kubernetes resource request,
        # which renders as ``ephemeral-storage`` on the pod.
        if "disk_size" in out:
            out["ephemeral_storage"] = out.pop("disk_size")
        for key in ("cpus", "memory"):
            if key not in out:
                continue
            raw = str(out[key]).strip()
            if raw and not raw.endswith("+"):
                out[key] = f"{raw}+"
    else:
        # Preserve the renderer's historical behavior outside Kubernetes. This
        # review deliberately settles only the affected Kubernetes profiles and
        # does not introduce a new VM-cloud boot-disk contract.
        out.pop("disk_size", None)
    return out


def tool_pip_extra(tool_ref: str) -> str:
    """Return the npa extra a toolRef's stage needs, or ``""``.

    Longest-prefix match, mirroring :func:`tool_image_key`.
    """

    if not tool_ref:
        return ""
    if tool_ref in TOOL_REF_PIP_EXTRAS:
        return TOOL_REF_PIP_EXTRAS[tool_ref]
    best = ""
    for prefix in TOOL_REF_PIP_EXTRAS:
        if tool_ref == prefix or tool_ref.startswith(prefix + "."):
            if len(prefix) > len(best):
                best = prefix
    return TOOL_REF_PIP_EXTRAS.get(best, "")


def tool_pip_requirements(tool_ref: str) -> tuple[tuple[str, str], ...]:
    """Return (executable, pip requirement) pairs this toolRef shells out to."""

    if tool_ref in TOOL_REF_PIP_REQUIREMENTS:
        return TOOL_REF_PIP_REQUIREMENTS[tool_ref]
    best = ""
    for prefix in TOOL_REF_PIP_REQUIREMENTS:
        if (tool_ref == prefix or tool_ref.startswith(prefix + ".")) and len(
            prefix
        ) > len(best):
            best = prefix
    return TOOL_REF_PIP_REQUIREMENTS.get(best, ())


def render_pip_requirements_setup(requirements: Sequence[tuple[str, str]]) -> str:
    """Install third-party CLIs a tool shells out to, when they are missing.

    Installed only when the executable is absent, so a purpose-built image that already
    ships it is untouched. Failing here rather than mid-stage is the better signal: the
    alternative is a FileNotFoundError from a subprocess after the stage has started.
    """

    if not requirements:
        return ""
    lines = [
        "# Stage needs third-party packages; install any that are missing INTO the interpreter\n"
        "# the stage will actually run (the vendor one when setup switched to it).\n"
        "npa_req_python=python3\n"
        "if [ -s /tmp/npa-python ]; then\n"
        '  npa_req_python="$(cat /tmp/npa-python)"\n'
        "fi\n"
    ]
    for probe, requirement in requirements:
        if probe.startswith(PYTHON_MODULE_PROBE):
            module = probe[len(PYTHON_MODULE_PROBE) :]
            # A library has no binary to look for, and the interpreter that matters is the one
            # the shim recorded — a vendor image's own venv is a different one.
            condition = f"! \"$npa_req_python\" -c 'import {module}' >/dev/null 2>&1"
            label = module
        else:
            condition = f"! command -v {probe} >/dev/null 2>&1"
            label = probe
        lines.append(
            f"if {condition}; then\n"
            f"  echo 'installing {requirement} for {label}' >&2\n"
            f"  \"$npa_req_python\" -m pip install -q '{requirement}' \\\n"
            f"    || \"$npa_req_python\" -m pip install -q '{requirement}' --break-system-packages \\\n"
            f"    || \"$npa_req_python\" -m pip install -q '{requirement}' --user \\\n"
            f"    || uv pip install -q --python \"$npa_req_python\" '{requirement}'\n"
            "fi\n"
        )
    return "".join(lines)


def render_pip_extra_setup(extra: str) -> str:
    """Install ``npa[<extra>]`` from the tree setup already installed npa from.

    Idempotent and best-effort *only* in the sense that it reports a clear error: if
    the extra cannot be installed the stage would fail anyway with a less obvious
    ImportError, so failing in ``setup`` is the better signal.
    """

    if not extra:
        return ""
    return (
        f"# Stage needs the npa[{extra}] optional dependency group.\n"
        'npa_src_root=""\n'
        "if [ -s /tmp/npa-src-root ]; then\n"
        '  npa_src_root="$(cat /tmp/npa-src-root)"\n'
        "elif [ -d /opt/nebius-physical-ai/npa ]; then\n"
        "  npa_src_root=/opt/nebius-physical-ai/npa\n"
        "elif [ -d /tmp/npa-src ]; then\n"
        "  npa_src_root=/tmp/npa-src\n"
        "fi\n"
        'if [ -n "$npa_src_root" ]; then\n'
        f'  echo "installing npa[{extra}] from $npa_src_root" >&2\n'
        # Compose with printf so the optional-extra suffix is visibly separate from the
        # source path. Braced expansions are valid here because top-level SkyPilot setup
        # and run values are author-controlled shell programs; only declarative and
        # nested fields are subject to the unresolved-placeholder guard.
        f'  npa_extra_target="$(printf \'%s[{extra}]\' "$npa_src_root")"\n'
        '  npa_pip_install -e "$npa_extra_target"\n'
        "else\n"
        f"  echo 'cannot locate the npa source tree to install npa[{extra}]' >&2\n"
        "  exit 1\n"
        "fi\n"
    )


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


def _contains_uid_zero_override(value: object) -> bool:
    """Return whether Kubernetes config explicitly forces a container to UID 0."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "runAsUser" and str(child).strip() == "0":
                return True
            if _contains_uid_zero_override(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_uid_zero_override(child) for child in value)
    return False


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
        resolved = str(options.image_overrides[tool_ref] or "").strip()
    else:
        # A family override such as ``workbench.fiftyone`` applies to its actions.
        # Exact matches above win, and the longest boundary-safe prefix is next.
        best_override = ""
        for prefix in options.image_overrides:
            if prefix == "*":
                continue
            if (tool_ref == prefix or tool_ref.startswith(prefix + ".")) and len(
                prefix
            ) > len(best_override):
                best_override = prefix
        if best_override:
            resolved = str(options.image_overrides[best_override] or "").strip()
        elif "*" in options.image_overrides:
            resolved = str(options.image_overrides["*"] or "").strip()
        else:
            resolved = str(resources.get("image") or "").strip()
            if not resolved:
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
                resolved = container_image_for_tool(tool, **kwargs)
    if resolved.startswith("tool://"):
        image_tool = resolved.removeprefix("tool://").strip()
        if not image_tool:
            raise NpaWorkflowError("tool:// image reference must name a workbench tool")
        from npa.deploy.images import container_image_for_tool

        resolved = container_image_for_tool(
            image_tool,
            registry=options.registry or None,
        )
    return str(options.image_digest_pins.get(resolved, resolved)).strip()


#: How long to wait for a self-hosted model server to answer /health, and how often to ask.
#: The server has to download a multi-GB checkpoint and load it onto the GPU first. The
#: retired ``vlm-eval.yaml`` allowed 120 x 5 s = 600 s; a *cold* HF download of a 7B
#: checkpoint routinely exceeds that, so the default is more generous. Override per spec
#: with ``config.vlm_serve_ready_seconds``.
DEFAULT_VLM_SERVER_READY_SECONDS = 900
VLM_SERVER_POLL_INTERVAL_SECONDS = 5
DEFAULT_VLM_SERVE_PORT = 8000


def render_self_hosted_vlm_preamble(config: Mapping[str, Any]) -> str:
    """Start and health-check a local vLLM server before the stage's command runs.

    ``vlm_backend: self-hosted`` tells the tool to POST to an OpenAI-compatible endpoint
    on localhost, but **nothing in a spec starts that server** — so the stage failed live
    with ``VLM backend request failed: [Errno 111] Connection refused`` (EVIDENCE §5.2b).
    The retired ``vlm-eval.yaml`` template did the serve/wait/teardown in its ``run:``
    block; that is exactly the kind of bash a ``toolRef`` cannot carry, so it moves here.

    The defaults deliberately match the tool's own (``DEFAULT_MODEL`` on port 8000, which
    ``DEFAULT_ENDPOINT_URL`` points at), so a spec needs no extra config to work;
    ``config.vlm_model`` / ``config.vlm_serve_port`` override them.

    The generated program uses ordinary shell expansion. Top-level SkyPilot
    ``setup`` and ``run`` programs are intentionally exempt from
    :func:`assert_no_unresolved_placeholders`; declarative and nested fields are not.
    """

    from npa.workbench.vlm_eval import DEFAULT_MODEL

    model = str(config.get("vlm_model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    port = str(config.get("vlm_serve_port") or DEFAULT_VLM_SERVE_PORT).strip()
    trust_remote_code = str(config.get("vlm_trust_remote_code") or "1").strip() not in {
        "0",
        "false",
        "False",
    }
    trust_flag = " --trust-remote-code" if trust_remote_code else ""
    interval = VLM_SERVER_POLL_INTERVAL_SECONDS
    try:
        ready_seconds = int(
            config.get("vlm_serve_ready_seconds") or DEFAULT_VLM_SERVER_READY_SECONDS
        )
    except (TypeError, ValueError) as exc:
        raise NpaWorkflowRenderError(
            f"config.vlm_serve_ready_seconds must be an integer number of seconds, "
            f"got {config.get('vlm_serve_ready_seconds')!r}"
        ) from exc
    if ready_seconds < interval:
        raise NpaWorkflowRenderError(
            f"config.vlm_serve_ready_seconds must be at least {interval}, got {ready_seconds}"
        )
    attempts = ready_seconds // interval
    flashinfer_sampler = (
        "1"
        if str(config.get("vlm_use_flashinfer_sampler") or "0").strip()
        in {"1", "true", "True"}
        else "0"
    )
    return (
        "# Self-hosted VLM backend: serve the model this stage is about to call.\n"
        # From #236: widen the CLIENT's readiness window too. The preamble health-checks
        # before the command runs, so this is a second net rather than the only one, but a
        # cold 7B load can still be finishing when the first request lands.
        "export NPA_VLM_READY_TIMEOUT_S=1800\n"
        f"npa_vlm_model={shlex.quote(model)}\n"
        # From #238: tell the CLIENT which model this preamble is serving. Without it the eval
        # asks for its own default and a spec that overrode `vlm_model` would score against a
        # model nobody started.
        f"export NPA_VLM_SELF_HOSTED_MODEL={model}\n"
        f"npa_vlm_port={shlex.quote(port)}\n"
        "npa_vlm_log=/tmp/npa-vlm-server.log\n"
        # vLLM's FlashInfer sampler JIT-compiles a CUDA extension and shells out to `ninja`
        # (setup pip-installs it). pip puts console scripts beside the interpreter, which is
        # not necessarily on PATH in this shell, so add it here — in `run:`, since setup runs
        # in a different shell.
        "npa_vlm_scripts=$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"scripts\"))')\n"
        "export PATH=$npa_vlm_scripts:$PATH\n"
        # ... and it needs nvcc, which SkyPilot's default image also lacks: the JIT then
        # failed with "/usr/local/cuda/bin/nvcc: not found" (live job 217). vLLM's own
        # dependencies include the nvidia-cuda-nvcc wheel, so point CUDA_HOME at it when
        # there is no system toolkit.
        "if [ ! -x /usr/local/cuda/bin/nvcc ]; then\n"
        "  npa_cuda_home=$(python3 - <<'PY'\n"
        "import pathlib\n"
        "import sysconfig\n"
        "\n"
        "for root in {sysconfig.get_paths()['purelib'], sysconfig.get_paths()['platlib']}:\n"
        "    candidate = pathlib.Path(root) / 'nvidia' / 'cuda_nvcc'\n"
        "    if (candidate / 'bin' / 'nvcc').is_file():\n"
        "        print(candidate)\n"
        "        break\n"
        "PY\n"
        "  )\n"
        '  if [ -n "$npa_cuda_home" ]; then\n'
        "    export CUDA_HOME=$npa_cuda_home\n"
        "    export PATH=$npa_cuda_home/bin:$PATH\n"
        '    echo "using pip CUDA toolkit at $npa_cuda_home" >&2\n'
        "  fi\n"
        "fi\n"
        # Belt and braces: the sampler that wants the JIT has a pure-PyTorch equivalent, so
        # a task image without a compiler must not be able to break server startup at all.
        # Set config.vlm_use_flashinfer_sampler=1 to opt back in.
        f"export VLLM_USE_FLASHINFER_SAMPLER={flashinfer_sampler}\n"
        'echo "starting vLLM for $npa_vlm_model on port $npa_vlm_port" >&2\n'
        # vLLM 0.26 removed the executable ``__main__`` block from
        # ``vllm.entrypoints.openai.api_server``.  Importing that module with
        # ``python -m`` can therefore leave no OpenAI server behind while an
        # unrelated process on the same port still makes a /health-only probe
        # look ready.  ``vllm serve`` is the supported, versioned CLI entrypoint.
        'vllm serve "$npa_vlm_model" --host 0.0.0.0 '
        '--port "$npa_vlm_port" '
        f'--served-model-name "$npa_vlm_model"{trust_flag} '
        '> "$npa_vlm_log" 2>&1 &\n'
        "npa_vlm_pid=$!\n"
        # Never leave a server (and its GPU memory) behind, on success or failure.
        "trap 'kill \"$npa_vlm_pid\" 2>/dev/null || true' EXIT\n"
        # Health-wait in python3, not curl: curl is not guaranteed in every task image,
        # and python3 is (setup records an interpreter and the shim puts it on PATH).
        'python3 - "$npa_vlm_pid" "$npa_vlm_port" "$npa_vlm_log" '
        "\"$npa_vlm_model\" <<'PY'\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "import urllib.error\n"
        "import urllib.request\n"
        "\n"
        "pid, port, log_path, expected_model = (\n"
        "    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]\n"
        ")\n"
        f"attempts, interval = {attempts}, {interval}\n"
        "\n"
        "def alive() -> bool:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except OSError:\n"
        "        return False\n"
        "    return True\n"
        "\n"
        "def tail() -> str:\n"
        "    try:\n"
        "        with open(log_path, encoding='utf-8', errors='replace') as handle:\n"
        "            return ''.join(handle.readlines()[-200:])\n"
        "    except OSError:\n"
        "        return '(no server log)'\n"
        "\n"
        "for attempt in range(attempts):\n"
        "    try:\n"
        # Probe the OpenAI surface the client will actually use.  A generic
        # /health response is insufficient: live job 364 reached it, then got a
        # 404 from /v1/chat/completions because no OpenAI server was running.
        "        with urllib.request.urlopen(\n"
        "            f'http://127.0.0.1:{port}/v1/models', timeout=5\n"
        "        ) as response:\n"
        "            payload = __import__('json').load(response)\n"
        "            model_ids = [str(item.get('id') or '') for item in payload.get('data', [])]\n"
        "            if not model_ids:\n"
        "                raise RuntimeError('vLLM model list is empty')\n"
        "            if expected_model not in model_ids:\n"
        "                raise RuntimeError(\n"
        "                    f'vLLM serves {model_ids!r}, expected {expected_model!r}'\n"
        "                )\n"
        "            print(f'vLLM server models: {model_ids}', file=sys.stderr)\n"
        "            print(f'vLLM server ready after {attempt * interval}s', file=sys.stderr)\n"
        "            raise SystemExit(0)\n"
        "    except SystemExit:\n"
        "        raise\n"
        "    except Exception:\n"
        "        pass\n"
        "    if not alive():\n"
        "        print('vLLM server exited before becoming ready:', file=sys.stderr)\n"
        "        print(tail(), file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
        "    time.sleep(interval)\n"
        "print(f'vLLM server not ready after {attempts * interval}s:', file=sys.stderr)\n"
        "print(tail(), file=sys.stderr)\n"
        "raise SystemExit(1)\n"
        "PY\n"
    )


def render_run_preamble_for_tool(tool_ref: str, *, config: Mapping[str, Any]) -> str:
    """Return shell that must run *inside the stage* before its command.

    The per-toolRef sibling of :func:`render_setup_for_tool`. A background service has to
    start here rather than in ``setup:``, because SkyPilot runs setup and run as separate
    shells — a server started in setup is gone by the time the command runs.
    """

    content_agents_pythonpath = (
        'if [ -n "$PYTHONPATH" ]; then\n'
        '  export PYTHONPATH="/opt/npa-runtime:/opt/content-agents:'
        '/opt/content-agents/apps:$PYTHONPATH"\n'
        "else\n"
        '  export PYTHONPATH="/opt/npa-runtime:/opt/content-agents:'
        '/opt/content-agents/apps"\n'
        "fi\n"
    )
    if tool_ref in {
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
    }:
        # SkyPilot's Kubernetes bootstrap replaces an image ENTRYPOINT with its
        # own bash command. Bootstrap the immutable operator-owned runtime cache,
        # then restore Xvfb in the shell that actually invokes OVRTX. Fail before
        # the expensive upstream pipeline if the node exposes CUDA devices without
        # the host-mounted graphics userspace OVRTX requires.
        return content_agents_pythonpath + (
            "/opt/venv/bin/python -m npa.workflows.content_agents bootstrap-runtime\n"
            "if ! python3 -c 'import ctypes; "
            'ctypes.CDLL("libGLX_nvidia.so.0")' "' >/dev/null 2>&1; then\n"
            "  echo 'OVRTX requires NVIDIA GPU Operator graphics driver mounts; "
            "libGLX_nvidia.so.0 is unavailable' >&2\n"
            "  exit 1\n"
            "fi\n"
            'npa_ovrtx_display="$(printenv OVRTX_XVFB_DISPLAY 2>/dev/null || true)"\n'
            'if [ -z "$npa_ovrtx_display" ]; then npa_ovrtx_display=99; fi\n'
            'if [ -z "$(printenv DISPLAY 2>/dev/null || true)" ]; then\n'
            "  /usr/local/bin/npa-content-agents-entrypoint /bin/true\n"
            '  export DISPLAY=":$npa_ovrtx_display"\n'
            '  npa_ovrtx_lock="/tmp/.X$npa_ovrtx_display-lock"\n'
            '  npa_ovrtx_xvfb_pid="$(tr -d "[:space:]" < "$npa_ovrtx_lock")"\n'
            '  case "$npa_ovrtx_xvfb_pid" in\n'
            "    ''|*[!0-9]*) echo 'invalid Xvfb pid file' >&2; exit 1 ;;\n"
            "  esac\n"
            "  npa_cleanup_ovrtx_display() {\n"
            '    if [ -r "/proc/$npa_ovrtx_xvfb_pid/comm" ] &&\n'
            '       [ "$(cat "/proc/$npa_ovrtx_xvfb_pid/comm")" = Xvfb ] &&\n'
            '       [ "$(tr -d "[:space:]" < "$npa_ovrtx_lock" 2>/dev/null || true)" = '
            '"$npa_ovrtx_xvfb_pid" ]; then\n'
            '      kill "$npa_ovrtx_xvfb_pid" 2>/dev/null || true\n'
            "    fi\n"
            "  }\n"
            "  trap npa_cleanup_ovrtx_display EXIT\n"
            "fi\n"
        )
    if tool_ref.startswith("workbench.content_agents."):
        # SkyPilot starts setup/run through login shells that may discard Docker
        # ENV values. Keep the narrow baked adapter importable on CPU stages too,
        # without invoking the render-only runtime bootstrap above.
        return content_agents_pythonpath
    if not tool_ref.startswith("workbench.vlm_eval"):
        return ""
    # #236 skipped the benchmark toolRef here, correctly for the twin it had: a `sample`
    # fixture scored with backend=stub needs no server. This branch's benchmark twin scores a
    # real labeled set on the self-hosted backend (EVIDENCE.md §R22), so the decision is made
    # by the backend the spec asks for, below, rather than by the toolRef's name.
    backend = str(config.get("vlm_backend") or "").strip().lower()
    if backend not in {"self-hosted", "self_hosted"}:
        return ""
    return render_self_hosted_vlm_preamble(config)


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
        'export PATH="$HOME/.local/bin:$PATH"\n'
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
        '    [ -r "$profile" ] && . "$profile" || true\n'
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
        '  if [ -n "$PYTHONPATH" ]; then\n'
        '    PYTHONPATH="/tmp/npa-src-overlay/src:$PYTHONPATH"\n'
        "  else\n"
        "    PYTHONPATH=/tmp/npa-src-overlay/src\n"
        "  fi\n"
        "  export PYTHONPATH\n"
        "fi\n"
        'npa_python=""\n'
        "if [ -s /tmp/npa-python ]; then\n"
        '  npa_python="$(cat /tmp/npa-python)"\n'
        '  if [ ! -x "$npa_python" ]; then\n'
        '    echo "recorded npa interpreter is not executable: $npa_python" >&2\n'
        '    npa_python=""\n'
        "  fi\n"
        "fi\n"
        # A vendor image can bake a STALE npa source tree on PYTHONPATH, which shadows every
        # install — editable or not, in any interpreter. Live job 285: the cosmos2-transfer image
        # ships `PYTHONPATH=/opt/npa/src`, whose npa predates the `cosmos2` subcommand, so the
        # stage kept running the old CLI no matter what had just been installed. Third image to
        # do this (lerobot was job 250), so the engine handles it rather than each image.
        # Prepending the recorded source is a no-op wherever the install already wins.
        # (no ${...} expansions here: the renderer's placeholder guard rejects them)
        'if [ -s /tmp/npa-src-root ] && [ -d "$(cat /tmp/npa-src-root)/src" ]; then\n'
        '  npa_src_path="$(cat /tmp/npa-src-root)/src"\n'
        '  if [ -z "$PYTHONPATH" ]; then\n'
        '    export PYTHONPATH="$npa_src_path"\n'
        "  else\n"
        '    case ":$PYTHONPATH:" in\n'
        '      *":$npa_src_path:"*) : ;;\n'
        '      *) export PYTHONPATH="$npa_src_path:$PYTHONPATH" ;;\n'
        "    esac\n"
        "  fi\n"
        '  echo "npa source path: $npa_src_path" >&2\n'
        "fi\n"
        'if [ -n "$npa_python" ]; then\n'
        "  mkdir -p /tmp/npa-shim\n"
        '  printf \'#!/bin/sh\\nexec "%s" "$@"\\n\' "$npa_python" '
        "> /tmp/npa-shim/python3\n"
        "  chmod +x /tmp/npa-shim/python3\n"
        # ... and `npa` must mean the SAME install. A vendor image can bake its own older npa
        # whose console script is first on PATH; setup then skips installing (command -v npa
        # succeeds), the overlay lands in the vendor interpreter, and the stage runs the stale
        # CLI. Live job 284: `No such command 'cosmos2'. Did you mean 'cosmos'?` — from an npa
        # predating the subcommand, while the recorded interpreter had the current one.
        # The distribution intentionally has no top-level ``npa.__main__``.
        # Import and CALL the same lightweight entry function used by the
        # installed console script so the shim stays bound to the recorded
        # interpreter without relying on a scripts directory that may be
        # outside PATH.  Calling it explicitly also supports older baked NPA
        # images whose entry module defines ``main`` but has no ``__main__``
        # guard: ``python -m npa.cli.entry`` silently exited 0 in a live Cosmos
        # Evaluator stage and therefore produced no declared report.
        '  printf \'#!/bin/sh\\nexec "%s" -c "from npa.cli.entry import main; main()" "$@"\\n\' '
        '"$npa_python" > /tmp/npa-shim/npa\n'
        "  chmod +x /tmp/npa-shim/npa\n"
        '  export PATH="/tmp/npa-shim:$PATH"\n'
        # Console scripts installed next to that interpreter must be resolvable by
        # name too, which is the same gap the `npa` symlink in setup works around.
        # Live: vLLM's FlashInfer JIT shells out to `ninja`, which ships as a vLLM
        # dependency in the interpreter's bin dir — a directory that is not on the
        # stage shell's PATH, which is the whole reason the shim above exists.
        # Appended, not prepended, so it cannot shadow a system tool.
        '  npa_scripts="$("$npa_python" -c \'import sysconfig; '
        'print(sysconfig.get_path("scripts"))\' 2>/dev/null || true)"\n'
        '  if [ -n "$npa_scripts" ] && [ -d "$npa_scripts" ]; then\n'
        '    export PATH="$PATH:$npa_scripts"\n'
        "  fi\n"
        '  echo "using npa interpreter $npa_python for this stage" >&2\n'
        "fi\n"
        "python3 -c 'import npa' >/dev/null 2>&1 || "
        "echo 'warning: python3 in this shell cannot import npa' >&2\n"
        "set -u\n"
        # Per-toolRef preamble (e.g. start a self-hosted model server) runs AFTER the
        # interpreter shim, so it uses the same python3 the command will.
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


#: NVIDIA's documented, run-scoped gate on Isaac acquisition/use.
ISAAC_EULA_ENV = "ACCEPT_EULA"
#: Image keys in TOOL_REF_IMAGE_TOOL that resolve to an Isaac-based image.
ISAAC_IMAGE_TOOLS = frozenset({"isaac-lab", "sonic"})


def routes_at_an_isaac_image(
    tool_ref: str,
    resources: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    resolved_image: str = "",
) -> bool:
    """Whether the renderer sends this stage to an Isaac-based runtime.

    ``resolved_image`` adds the actual image selected through a global/tool
    override, registry resolution, or a ``tool://`` reference. The declared
    scheduler resources alone are insufficient for raw-shell states; semantic
    tool routes remain Isaac-backed even when a custom image name is opaque.
    """

    if tool_image_key(tool_ref) in ISAAC_IMAGE_TOOLS:
        return True
    workflow_config = config or {}
    if tool_ref == "workbench.byof.repo":
        base_profile = str(workflow_config.get("base_profile") or "").strip().lower()
        base_image = str(workflow_config.get("base_image") or "").strip().lower()
        if base_profile == "isaac-lab" or "isaac-lab" in base_image:
            return True
    if tool_ref.startswith("workbench.groot"):
        if tool_ref.startswith("workbench.groot.isaac"):
            return True
        for key in ("sim_backend", "simulation_backend", "groot_runtime"):
            if str(workflow_config.get(key) or "").strip().lower() in {
                "isaac",
                "isaac-lab",
                "isaac_sim",
            }:
                return True
        if workflow_config.get("sim") is True:
            return True
    raw = resources or {}
    image = str(resolved_image or raw.get("image") or raw.get("image_id") or "").lower()
    image = image.removeprefix("docker:")
    if "isaac-lab" in image or "npa-sonic" in image:
        return True
    pod = ((raw.get("kubernetes") or {}).get("pod_config") or {}).get("spec") or {}
    for container in pod.get("containers") or []:
        names = {str(item.get("name") or "") for item in container.get("env") or []}
        if "NPA_ISAAC_CACHE_DIR" in names or "NPA_SIM2REAL_ISAAC_CACHE_PVC" in names:
            return True
    return False


def isaac_eula_envs(
    tool_ref: str,
    *,
    resources: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    resolved_image: str = "",
    accepted: bool = True,
) -> dict[str, str]:
    """Declare NVIDIA's acceptance value for an Isaac stage.

    Acceptance defaults on for non-interactive workflows. ``accepted=False`` renders an empty
    value, preserving an explicit opt-out that the runtime bootstrap rejects before download.

    This lives in the renderer rather than in each spec because a spec reaches an Isaac image
    through its ``toolRef``, not by naming an image — so a spec author cannot reasonably be
    expected to know the routing, and a new Isaac toolRef is covered the moment it is added.
    """

    if not routes_at_an_isaac_image(
        tool_ref, resources, config, resolved_image=resolved_image
    ):
        return {}
    return {ISAAC_EULA_ENV: "Y" if accepted else ""}


def default_npa_setup() -> str:
    """Ensure the ``npa`` CLI is available on the SkyPilot worker.

    Workbench images bake npa at ``/opt/nebius-physical-ai/npa``. When a task
    uses SkyPilot's default image (e.g. Token Factory API twins), setup can:

    1. install from a mounted ``/tmp/npa-src`` (S3 URI via ``NPA_SRC_S3_URI``), or
    2. sync from ``NPA_SRC_S3_URI`` with the AWS CLI / boto3, then install.
    """

    return (
        "set -e\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        # Record where npa was installed from so a per-tool extra (see
        # TOOL_REF_PIP_EXTRAS) can be layered on top of the SAME source tree.
        "npa_record_src_root() { printf '%s' \"$1\" > /tmp/npa-src-root; }\n"
        # Thin workbench images keep the installable project at /opt/npa rather than the
        # legacy /opt/nebius-physical-ai/npa path.  Record it even when the baked `npa`
        # launcher is already on PATH: vendor-interpreter setup still needs the source root
        # to install NPA into the runtime-fetched Isaac environment.  Live Isaac job 4
        # otherwise retained NPA_BAKED_PYTHON and failed on `No module named isaaclab`.
        "if [ -f /opt/npa/pyproject.toml ] && [ -d /opt/npa/src/npa ]; then\n"
        "  npa_record_src_root /opt/npa\n"
        "fi\n"
        # Debian/Ubuntu >= 24.04 mark the system interpreter externally managed
        # (PEP 668), so a plain `pip install` fails with
        # "error: externally-managed-environment". A task container is disposable, so
        # retry with --break-system-packages and then --user before giving up. Live:
        # this is what the Isaac Lab image hit once its system python3 came first on
        # PATH, and any Ubuntu 24.04 based image would hit it too.
        "npa_pip_install() {\n"
        '  target="$1"\n'
        "  shift\n"
        # uv-created environments deliberately need not contain pip. GR00T's
        # image is one: `python3 -m pip` exits before the source overlay can be
        # staged even though the image ships uv. Let uv target the exact
        # interpreter that `python3` resolves to; unlike activating another
        # interpreter, this preserves the vendor environment and its pins.
        '  npa_install_python="$(command -v python3)"\n'
        '  if "$npa_install_python" -m pip --version >/dev/null 2>&1; then\n'
        '    "$npa_install_python" -m pip install -q "$target" "$@" \\\n'
        '      || "$npa_install_python" -m pip install -q "$target" "$@" --break-system-packages \\\n'
        '      || "$npa_install_python" -m pip install -q "$target" "$@" --user\n'
        "  elif command -v uv >/dev/null 2>&1; then\n"
        '    uv pip install -q --python "$npa_install_python" "$target" "$@"\n'
        "  else\n"
        '    echo "python3 has no pip and uv is unavailable: $npa_install_python" >&2\n'
        "    return 1\n"
        "  fi\n"
        "}\n"
        "if ! command -v npa >/dev/null 2>&1; then\n"
        "  if [ -d /opt/nebius-physical-ai/npa ]; then\n"
        "    npa_pip_install -e /opt/nebius-physical-ai/npa\n"
        "    npa_record_src_root /opt/nebius-physical-ai/npa\n"
        "  else\n"
        '    if [ ! -d /tmp/npa-src ] && [ -n "$NPA_SRC_S3_URI" ]; then\n'
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
        "      npa_record_src_root /tmp/npa-src\n"
        "    else\n"
        "      echo 'npa CLI not found; set NPA_SRC_S3_URI or use a workbench image' >&2\n"
        "      exit 1\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        # Opt-in branch overlay: reinstall npa from NPA_SRC_S3_URI on TOP of a
        # baked workbench image so branch code (e.g. a new augment prompt path)
        # actually runs on GPU without rebuilding the image. Default off (no-op).
        'if [ "$NPA_SRC_OVERLAY" = "1" ] && [ -n "$NPA_SRC_S3_URI" ]; then\n'
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
        # --no-deps FIRST: the overlay is the same distribution the image already has, so
        # resolving its requirements would only risk moving a pinned vendor stack.
        "  npa_pip_install -e /tmp/npa-src-overlay --no-deps\n"
        # ... and WITH deps if the CLI still will not import. An image that installed npa with
        # its own curated `--no-deps` list leaves the overlay short of whatever that list
        # omitted: live job 309 died on `No module named 'paramiko'` after a clean overlay of a
        # tree that declares paramiko as a dependency. Probe the CLI, not `import npa` — npa
        # imported fine there; it was the command tree that could not load. Same
        # safe-then-sufficient order as the vendor-interpreter install.
        "  if ! python3 -c 'import npa.cli.main' >/dev/null 2>&1; then\n"
        "    echo 'npa CLI is not importable after the overlay; installing its dependencies'"
        " >&2\n"
        "    npa_pip_install -e /tmp/npa-src-overlay\n"
        "  fi\n"
        # The overlay is the freshest tree, so it is the one worth putting on the import path.
        "  npa_record_src_root /tmp/npa-src-overlay\n"
        # Same reason as the stage preamble: the install alone is not enough to
        # displace a baked npa, so make the overlay explicit for the rest of setup too
        # (the interpreter recorded below is checked with `import npa`).
        '  if [ -n "$PYTHONPATH" ]; then\n'
        '    PYTHONPATH="/tmp/npa-src-overlay/src:$PYTHONPATH"\n'
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
        #   1. NPA_BAKED_PYTHON - the image's declared, dependency-complete runtime;
        #   2. sys.executable - correct on normal images;
        #   3. the alias target - the Isaac Lab image aliases python3 to
        #      /workspace/isaaclab/_isaac_sim/python.sh, and its embedded kit python
        #      cannot import its own site-packages unless launched through that
        #      wrapper (live run: "could not record a usable npa interpreter");
        #   4. `type -P python3` - the PATH binary, ignoring any alias.
        "python3 -c 'import npa' >/dev/null 2>&1 || "
        "{ echo 'npa is not importable after setup' >&2; exit 1; }\n"
        'npa_python=""\n'
        'alias_target="$(alias python3 2>/dev/null | sed -e "s/^alias python3=//" '
        '-e "s/^\'//" -e "s/\'$//")"\n'
        'for candidate in "${NPA_BAKED_PYTHON:-}" '
        "\"$(python3 -c 'import sys; print(sys.executable)' "
        '2>/dev/null || true)" "$alias_target" "$(type -P python3 2>/dev/null '
        '|| true)"; do\n'
        '  if [ -n "$candidate" ] && [ -x "$candidate" ] && '
        "\"$candidate\" -c 'import npa' >/dev/null 2>&1; then\n"
        '    npa_python="$candidate"\n'
        "    break\n"
        "  fi\n"
        "done\n"
        'if [ -n "$npa_python" ]; then\n'
        '  echo "$npa_python" > /tmp/npa-python\n'
        '  echo "npa interpreter recorded: $npa_python" >&2\n'
        "else\n"
        "  echo 'warning: no python command outside this shell could import npa' >&2\n"
        "fi\n"
        # toolRef stages invoke the `npa` console script by name; installing into a
        # non-standard interpreter can leave it outside PATH, so link it where every
        # shell will find it.
        "if [ ! -x /usr/local/bin/npa ]; then\n"
        # Look in the USER scheme and $HOME/.local/bin too: npa_pip_install falls back to
        # `--user` under PEP 668, and the console script then lands outside the default
        # scripts dir — the judge stage died with `bash: npa: command not found` on an image
        # where that fallback fired (live job 260).
        "  for scripts_dir in "
        '"$(python3 -c \'import sysconfig; print(sysconfig.get_path("scripts"))\' '
        '2>/dev/null || true)" '
        "\"$(python3 -c 'import sysconfig; "
        'print(sysconfig.get_path("scripts", scheme="posix_user"))\' 2>/dev/null || true)" '
        '"$HOME/.local/bin"; do\n'
        '    if [ -n "$scripts_dir" ] && [ -x "$scripts_dir/npa" ]; then\n'
        '      ln -sf "$scripts_dir/npa" /usr/local/bin/npa 2>/dev/null || '
        'sudo -n ln -sf "$scripts_dir/npa" /usr/local/bin/npa 2>/dev/null || true\n'
        "      break\n"
        "    fi\n"
        "  done\n"
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
        # vLLM's FlashInfer sampler JIT-compiles a CUDA extension on first use and shells out
        # to `ninja`, which SkyPilot's default image does not ship: the server died during
        # engine init with FileNotFoundError: 'ninja' (live job 214). The pip package provides
        # the binary, so no apt or sudo is needed.
        "import shutil\n"
        "if shutil.which('ninja') is None:\n"
        "    pip_install('ninja')\n"
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
    if tool_ref.startswith("workbench.content_agents."):
        # The public Content Agents image deliberately carries only the narrow
        # module adapter used by its five toolRefs. Requiring the full ``npa``
        # console script here would either force unrelated CLI/dependency bytes
        # into that image or make SkyPilot overlay source at runtime. Verify the
        # same zero-OVRTX image boundary used during the build, then record its
        # exact interpreter for the run-shell shim. This inspection never
        # downloads OVRTX; render-stage preambles bootstrap it later.
        return (
            "set -e\n"
            'npa_baked_python="/opt/venv/bin/python"\n'
            'if [ -n "$PYTHONPATH" ]; then\n'
            '  export PYTHONPATH="/opt/npa-runtime:/opt/content-agents:'
            '/opt/content-agents/apps:$PYTHONPATH"\n'
            "else\n"
            '  export PYTHONPATH="/opt/npa-runtime:/opt/content-agents:'
            '/opt/content-agents/apps"\n'
            "fi\n"
            'if [ ! -x "$npa_baked_python" ]; then\n'
            '  echo "Content Agents baked interpreter is unavailable" >&2\n'
            "  exit 69\n"
            "fi\n"
            '"$npa_baked_python" - <<\'PY\'\n'
            "from npa.workflows.content_agents import inspect_image\n"
            "payload = inspect_image()\n"
            "if payload.get('status') != 'image-ready':\n"
            "    raise SystemExit('Content Agents image boundary is not ready')\n"
            "print('Content Agents narrow baked runtime verified')\n"
            "PY\n"
            'printf \'%s\\n\' "$npa_baked_python" > /tmp/npa-python\n'
        )
    require_baked = str(config.get("require_baked_npa") or "").strip().lower()
    if require_baked in {"1", "true", "yes", "on"}:
        return (
            "set -e\n"
            'npa_baked_python="${NPA_BAKED_PYTHON:-}"\n'
            'if [ -z "$npa_baked_python" ]; then\n'
            '  npa_baked_python="$(command -v python3 || true)"\n'
            "fi\n"
            'case "$npa_baked_python" in\n'
            "  /*) ;;\n"
            '  *) echo "baked NPA interpreter must be an absolute path" >&2; exit 68 ;;\n'
            "esac\n"
            'if [ ! -x "$npa_baked_python" ]; then\n'
            '  echo "baked NPA interpreter is not executable: $npa_baked_python" >&2\n'
            "  exit 69\n"
            "fi\n"
            "\"$npa_baked_python\" - <<'PY'\n"
            "import os\n"
            # The stage shim imports npa.cli.main, not merely the intentionally
            # lazy package root.  Probing the same path prevents an immutable
            # image from passing setup and then failing after scheduling because
            # a CLI dependency (as seen live with Typer) was omitted.
            "import npa.cli.main\n"
            "actual = os.environ.get('NPA_IMAGE_SOURCE_SHA', '').strip().lower()\n"
            "expected = os.environ.get('NPA_SIM2REAL_SOURCE_SHA', '').strip().lower()\n"
            "if len(actual) != 40 or actual != expected:\n"
            "    raise SystemExit('baked NPA source attestation does not match workflow source SHA')\n"
            "print('immutable baked NPA runtime verified', actual)\n"
            "PY\n"
            "printf '%s\\n' \"$npa_baked_python\" > /tmp/npa-python\n"
        )
    parts = [default_npa_setup()]
    parts.append(render_vendor_interpreter_setup(tool_vendor_interpreters(tool_ref)))
    extra = tool_pip_extra(tool_ref)
    if extra:
        parts.append(render_pip_extra_setup(extra))
    declared_extra = str(config.get("pip_extra") or "").strip()
    if declared_extra:
        if declared_extra not in DECLARATIVE_PIP_EXTRAS:
            allowed = ", ".join(sorted(DECLARATIVE_PIP_EXTRAS))
            raise NpaWorkflowError(
                f"config.pip_extra {declared_extra!r} is not allowed; choose: {allowed}"
            )
        parts.append(render_pip_extra_setup(declared_extra))
    parts.append(render_pip_requirements_setup(tool_pip_requirements(tool_ref)))
    backend = str(config.get("vlm_backend") or "").strip().lower()
    if tool_ref.startswith("workbench.vlm_eval") and backend in {
        "self-hosted",
        "self_hosted",
    }:
        parts.append(_vllm_install_setup(self_hosted_vlm_model(config)))
    if tool_ref.startswith("workbench.sonic"):
        parts.append(_sonic_deps_setup())
    if tool_ref.startswith("workbench.token_factory"):
        # Avoid ${VAR:-} bash forms so SkyPilot placeholder lint stays clean.
        parts.append(
            'if [[ -z "$NEBIUS_TOKEN_FACTORY_KEY" ]]; then\n'
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
            'if [ -s /tmp/npa-python ]; then npa_nurec_py="$(cat /tmp/npa-python)"; fi\n'
            # --break-system-packages FIRST: the NRE image is Ubuntu 24.04, whose
            # interpreter is externally managed (PEP 668), so the plain form always
            # fails there and only adds a confusing "error:
            # externally-managed-environment" to the logs before the fallback wins.
            "npa_nurec_pip() {\n"
            '  if "$npa_nurec_py" -m pip --version >/dev/null 2>&1; then\n'
            '    "$npa_nurec_py" -m pip install -q "$@" --break-system-packages \\\n'
            '      || "$npa_nurec_py" -m pip install -q "$@" \\\n'
            '      || "$npa_nurec_py" -m pip install -q "$@" --user\n'
            "  elif command -v uv >/dev/null 2>&1; then\n"
            '    uv pip install -q --python "$npa_nurec_py" "$@"\n'
            "  elif command -v uvx >/dev/null 2>&1; then\n"
            '    uvx --from uv uv pip install -q --python "$npa_nurec_py" "$@"\n'
            "  else\n"
            '    echo "NuRec dependencies require pip, uv, or uvx: $npa_nurec_py" >&2\n'
            "    return 1\n"
            "  fi\n"
            "}\n"
            f"npa_nurec_pip 'huggingface_hub>=0.30' 'nvidia-ncore' '{NUREC_RERUN_PIN}' 'pillow>=10.0'\n"
            '"$npa_nurec_py" -c \'import ncore, rerun; print("nurec runtime deps ready")\'\n'
        )
    return "".join(parts)


def secret_env_hints_for_plan(steps: Sequence[PlanStep]) -> tuple[str, ...]:
    """Collect recommended ``--secret-env`` names for a planned workflow."""

    hints: list[str] = []
    seen: set[str] = set()
    for step in steps:
        tool_ref = step.tool_ref or ""
        if tool_ref == "workbench.byof.repo" and any(
            value == "openpi" or "pi05_droid_jointpos_polaris" in value
            for value in step.argv
        ):
            if OPENPI_TERMS_ENV not in seen:
                seen.add(OPENPI_TERMS_ENV)
                hints.append(OPENPI_TERMS_ENV)
        matches = [
            (prefix, names)
            for prefix, names in SECRET_ENV_HINTS.items()
            if tool_ref == prefix or tool_ref.startswith(prefix + ".")
        ]
        names = max(matches, key=lambda item: len(item[0]))[1] if matches else ()
        for name in names:
            if name not in seen:
                seen.add(name)
                hints.append(name)
    return tuple(hints)


def resolve_src_s3_uri() -> str:
    """Return the staged npa source prefix, preferring the environment over config."""

    import os

    value = (
        os.environ.get("NPA_SRC_S3_URI")
        or os.environ.get("NPA_E2E_NPA_SRC_S3_URI")
        or ""
    ).strip()
    if value:
        return value
    try:
        from npa.clients.config import resolve_workflow_src_s3_uri

        return resolve_workflow_src_s3_uri()
    except Exception:  # noqa: BLE001 - a missing/unreadable config is just "unset"
        return ""


def plan_images(
    spec: NpaWorkflowSpec,
    steps: Sequence[PlanStep],
    *,
    run_id: str,
    options: SkypilotRenderOptions,
) -> list[str]:
    """Return the distinct container images a plan's steps will pull, in order."""

    images: list[str] = []
    for step in steps:
        scheduler_task = build_scheduler_task(spec, step, run_id=run_id)
        image = resolve_task_image(
            str(scheduler_task.get("tool_ref") or ""),
            scheduler_task.get("resources") or {},
            options=options,
        )
        image = str(image or "").strip()
        if image and image not in images:
            images.append(image)
    return images


def plan_image_pull_secrets(
    spec: NpaWorkflowSpec,
    steps: Sequence[PlanStep],
    *,
    run_id: str,
    options: SkypilotRenderOptions,
) -> dict[str, tuple[str, ...]]:
    """Return declared Kubernetes pull-secret names for each exact image path.

    If an image is also used by a non-Kubernetes step, its mapping is empty: a
    Kubernetes secret cannot prove that VM execution path can pull the image.
    """

    paths: dict[str, list[tuple[str, ...] | None]] = {}
    for step in steps:
        task = build_scheduler_task(spec, step, run_id=run_id)
        resources = task.get("resources") or {}
        image = str(
            resolve_task_image(
                str(task.get("tool_ref") or ""), resources, options=options
            )
            or ""
        ).strip()
        if not image:
            continue
        cloud = str(resources.get("cloud") or "").strip().casefold()
        if cloud not in {"kubernetes", "k8s"}:
            paths.setdefault(image, []).append(None)
            continue
        kubernetes = resources.get("kubernetes")
        kubernetes = kubernetes if isinstance(kubernetes, dict) else {}
        pod_config = kubernetes.get("pod_config")
        pod_config = pod_config if isinstance(pod_config, dict) else {}
        pod_spec = pod_config.get("spec")
        pod_spec = pod_spec if isinstance(pod_spec, dict) else {}
        raw_names = pod_spec.get("imagePullSecrets")
        raw_names = raw_names if isinstance(raw_names, list) else []
        names = tuple(
            str(item.get("name") or "").strip()
            for item in raw_names
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
        paths.setdefault(image, []).append(names)
    return {
        image: ()
        if any(item is None for item in authorities)
        else tuple(dict.fromkeys(name for item in authorities for name in (item or ())))
        for image, authorities in paths.items()
    }


def build_skypilot_task_doc(
    spec: NpaWorkflowSpec,
    step: PlanStep,
    *,
    run_id: str,
    options: SkypilotRenderOptions,
) -> dict[str, Any]:
    """Build one SkyPilot task document from a planned step."""

    scheduler_task = build_scheduler_task(spec, step, run_id=run_id)
    resources = normalize_resources(
        scheduler_task.get("resources") or {},
        accelerator_overrides=options.gpu_accelerator_overrides,
    )
    image = resolve_task_image(
        str(scheduler_task.get("tool_ref") or ""),
        scheduler_task.get("resources") or {},
        options=options,
    )
    if image:
        resources["image_id"] = (
            f"docker:{image}" if not image.startswith("docker:") else image
        )
    require_baked = str(spec.config.get("require_baked_npa") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    expected_source_sha = str(spec.config.get("source_sha") or "").strip().lower()
    if require_baked:
        from npa.orchestration.skypilot.image_bootstrap_contract import (
            ImageBootstrapContractError,
            parse_oci_reference,
        )

        try:
            parsed_image = parse_oci_reference(image)
        except ImageBootstrapContractError as exc:
            raise NpaWorkflowRenderError(
                f"planned step {scheduler_task['name']!r} requires a "
                "registry-qualified immutable image because "
                "config.require_baked_npa is enabled"
            ) from exc
        if not parsed_image.digest:
            raise NpaWorkflowRenderError(
                f"planned step {scheduler_task['name']!r} requires a "
                "registry-qualified immutable image because "
                "config.require_baked_npa is enabled"
            )
    if require_baked and (
        len(expected_source_sha) != 40
        or any(char not in "0123456789abcdef" for char in expected_source_sha)
    ):
        raise NpaWorkflowRenderError(
            f"planned step {scheduler_task['name']!r} requires an exact source SHA "
            "because config.require_baked_npa is enabled"
        )

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
    attempt_id = str(options.execution_attempt_id or "").strip()
    if not attempt_id:
        material = "\0".join((spec.name, run_id, str(scheduler_task["name"]))).encode(
            "utf-8"
        )
        attempt_id = hashlib.sha256(material).hexdigest()
    envs["NPA_WORKFLOW_ATTEMPT_ID"] = attempt_id
    if options.execution_fence_sequence < 1 or options.execution_fence_attempt < 1:
        raise NpaWorkflowRenderError(
            "workflow execution fence sequence and attempt must be positive"
        )
    envs["NPA_WORKFLOW_FENCE_SEQUENCE"] = str(options.execution_fence_sequence)
    envs["NPA_WORKFLOW_FENCE_ATTEMPT"] = str(options.execution_fence_attempt)
    if options.include_aws_endpoint and options.aws_endpoint_url:
        envs["AWS_ENDPOINT_URL"] = options.aws_endpoint_url
    if image:
        envs["NPA_TASK_IMAGE"] = image.removeprefix("docker:")
    if expected_source_sha:
        if len(expected_source_sha) != 40 or any(
            char not in "0123456789abcdef" for char in expected_source_sha
        ):
            raise NpaWorkflowRenderError(
                "config.source_sha must be an exact 40-character hexadecimal SHA"
            )
        envs["NPA_SIM2REAL_SOURCE_SHA"] = expected_source_sha
    envs.update(
        isaac_eula_envs(
            str(scheduler_task.get("tool_ref") or ""),
            resources=scheduler_task.get("resources") or {},
            config=spec.config,
            resolved_image=image,
            accepted=options.accept_eula,
        )
    )
    # Optional tuning passthrough. Cosmos resolves explicit argv first, these env
    # values second, and a validated run-scoped refinement artifact last. Thus the
    # env values tune direct/first-pass execution but intentionally cannot override
    # a committed retry policy.
    import os as _os_cond

    for _cond_var in (
        "NPA_COSMOS_CONDITION_ON_INPUT",
        "NPA_COSMOS_CONTROL",
        "NPA_COSMOS_CONTROL_WEIGHT",
        "NPA_COSMOS_CONTROL_ASSET",
        "NPA_COSMOS_CONTROL_PROMPT",
        "NPA_COSMOS_MASK_ASSET",
        "NPA_COSMOS_MASK_PROMPT",
        "NPA_COSMOS_GUIDANCE",
        "NPA_COSMOS_VARIANT_PARALLELISM",
        "NPA_COSMOS_IDENTITY_TIMEOUT_S",
        "NPA_COSMOS_SHARD_JOIN_TIMEOUT_S",
        "NPA_COSMOS_VALIDATION_SCOPE",
        "NPA_COSMOS_VALIDATION_DELAY_S",
        "NPA_COSMOS_VALIDATION_DELAY_PHASE",
        "NPA_COSMOS_VALIDATION_DELAY_RANK",
        "NPA_COSMOS_VALIDATION_DELAY_GENERATION",
        "NPA_COSMOS_VALIDATION_FAIL_PHASE",
        "NPA_COSMOS_VALIDATION_FAIL_RANK",
        "NPA_COSMOS_VALIDATION_FAIL_GENERATION",
        "NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS",
    ):
        _cond_val = str(_os_cond.environ.get(_cond_var) or "").strip()
        if _cond_val:
            envs[_cond_var] = _cond_val

    # Weights this stage has to download (the images bake none) belong in the
    # operator's durable cache when one exists, so the next run of the same image
    # is a cache hit instead of another multi-gigabyte pull onto a paid GPU.
    #
    # A claim is only mountable where there is a cluster to mount it in. On any
    # other cloud SkyPilot hands us a fresh VM, so only an explicit
    # NPA_MODEL_CACHE_DIR -- the operator saying the path is already there --
    # can be honored, and the env must not name a path nothing backs.
    cache_on_kubernetes = (
        str(resources.get("cloud") or "").strip().lower() in {"kubernetes", "k8s"}
    )
    cache_root = resolve_model_cache_root(
        runtime=RUNTIME_KUBERNETES if cache_on_kubernetes else RUNTIME_PREMOUNTED
    )
    cache_claim = model_cache_pvc() if cache_on_kubernetes else ""
    cache_host_path = model_cache_host_path() if cache_on_kubernetes else ""
    cache_mounted = bool(cache_root and (cache_claim or cache_host_path))
    envs.update(model_cache_env(cache_root))

    doc: dict[str, Any] = {
        "name": scheduler_task["name"],
        "resources": resources,
        "envs": envs,
        "run": render_task_run_script(
            command,
            preamble=render_model_cache_shell(cache_root, mounted=cache_mounted)
            + render_run_preamble_for_tool(
                str(scheduler_task.get("tool_ref") or ""), config=spec.config
            ),
        ),
    }
    # Multi-node stages: SkyPilot gang-schedules `num_nodes` identical pods for one task
    # and exports SKYPILOT_NODE_RANK / SKYPILOT_NODE_IPS into each. Emitted only when the
    # profile asks for more than one node, so every existing rendered doc is unchanged.
    num_nodes = int(scheduler_task.get("num_nodes") or 1)
    if (
        str(scheduler_task.get("tool_ref") or "")
        == "workbench.cosmos2.transfer_execute"
    ):
        # This renderer/planner value is authoritative.  SkyPilot's runtime
        # variables are independent evidence and the worker cross-checks them.
        envs["NPA_COSMOS_NODE_COUNT"] = str(num_nodes)
    if num_nodes > 1:
        doc["num_nodes"] = num_nodes
    task_config = normalize_task_config(scheduler_task.get("resources") or {})
    if image and task_config:
        from npa.orchestration.skypilot.image_bootstrap_contract import (
            is_trusted_npa_image,
        )

        if is_trusted_npa_image(image) and _contains_uid_zero_override(task_config):
            raise NpaWorkflowRenderError(
                "first-party workflow images must satisfy the SkyPilot bootstrap "
                "contract as their declared image user; runAsUser: 0 overrides are forbidden"
            )
    # A pod is discarded when the stage ends, so on Kubernetes the cache env above
    # only survives the run if it points at a volume that outlives the pod. Mount
    # the operator's claim; a profile that already mounts something at the cache
    # root keeps its own volume (pod_config_with_model_cache leaves it alone).
    if cache_mounted:
        task_config.setdefault("kubernetes", {})["pod_config"] = (
            pod_config_with_model_cache(
                task_config.get("kubernetes", {}).get("pod_config"),
                root=cache_root,
                pvc=cache_claim,
                host_path=cache_host_path,
            )
        )
    if task_config:
        doc["config"] = task_config
    setup = render_setup_for_tool(
        str(scheduler_task.get("tool_ref") or ""),
        config=spec.config,
        options=options,
    )
    if setup.strip():
        # setup is where a stage pre-fetches weights (the self-hosted VLM backend
        # downloads its model here so the eval's readiness window is not spent on
        # it), and SkyPilot runs it in a different shell than run -- so the cache
        # tree has to exist in both.
        doc["setup"] = render_model_cache_shell(cache_root, mounted=cache_mounted) + setup
    # When no workbench image is pinned, point setup at an existing S3 copy of
    # the npa package (SkyPilot local file_mounts create new buckets and fail
    # on Nebius). Operators set NPA_SRC_S3_URI=s3://bucket/prefix/npa, or persist
    # it once with `npa configure --src-s3-uri` so the next shell still finds it.
    import os

    src_uri = resolve_src_s3_uri()
    if require_baked:
        # Exact images must contain the full runtime and pinned dependencies. Never
        # inject a source tree or install packages after a task acquires a GPU.
        pass
    elif not image:
        if not src_uri:
            raise NpaWorkflowRenderError(
                f"planned step {scheduler_task['name']!r} has no workbench image "
                "and NPA_SRC_S3_URI is unset; set NPA_SRC_S3_URI=s3://bucket/prefix/npa, "
                "persist it with `npa configure --src-s3-uri s3://bucket/prefix/npa`, "
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
        if (
            str(os.environ.get("NPA_SRC_OVERLAY") or "").strip()
            in {"1", "true", "True"}
            and src_uri
        ):
            envs["NPA_SRC_OVERLAY"] = "1"
            doc["envs"] = envs
    _inject_operator_registry_docker_secrets(
        doc,
        materialize=options.materialize_registry_secrets,
    )
    return doc


def _inject_operator_registry_docker_secrets(
    doc: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Embed configured exact-host credentials for an operator registry.

    Official public GHCR development and release tags remain anonymous. NPA
    never mints provider IAM tokens or manages Kubernetes pull secrets.
    """

    import os

    resources = doc.get("resources") or {}
    if not isinstance(resources, dict):
        return
    cloud = str(resources.get("cloud") or "").strip().lower()
    image_id = str(resources.get("image_id") or "").strip()
    if cloud not in {"nebius", "kubernetes", "k8s"} or not image_id:
        return

    server = image_id.removeprefix("docker:").split("/", 1)[0]
    creds_server = str(os.environ.get("SKYPILOT_DOCKER_SERVER") or "").strip()
    if not creds_server:
        return
    if creds_server != server:
        from npa.deploy.images import is_public_registry

        image_registry = image_id.removeprefix("docker:").rsplit("/", 1)[0]
        if is_public_registry(image_registry):
            return
        if materialize:
            raise NpaWorkflowRenderError(
                f"registry mismatch: task image is in {server!r} but the Docker "
                "credentials (SKYPILOT_DOCKER_SERVER) authenticate to "
                f"{creds_server!r}. Set SKYPILOT_DOCKER_* for {server!r}, or "
                f"select an image from {creds_server!r}."
            )
        return

    from npa.orchestration.skypilot.registry_preflight import (
        resolve_registry_credentials,
    )

    username, password = resolve_registry_credentials(server)
    if not username:
        return
    if materialize:
        if not password:
            return
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
    return _render_docs(
        spec, plan.steps, run_id=run_id, options=opts, execution="serial"
    )


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
        spec,
        steps,
        run_id=run_id,
        options=opts,
        execution="serial",
        name=name or spec.name,
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


_SKYPILOT_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SKYPILOT_SHELL_FIELDS = frozenset({"run", "setup"})


def _placeholder_names(value: object) -> set[str]:
    """Return bare placeholders from a parsed YAML value."""

    unresolved: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            unresolved.update(_SKYPILOT_PLACEHOLDER_RE.findall(str(raw_key)))
            unresolved.update(_placeholder_names(child))
    elif isinstance(value, list):
        for child in value:
            unresolved.update(_placeholder_names(child))
    elif isinstance(value, str):
        unresolved.update(_SKYPILOT_PLACEHOLDER_RE.findall(value))
    return unresolved


def _document_declarative_placeholder_names(document: object) -> set[str]:
    """Return placeholders outside top-level SkyPilot shell-script fields."""

    if not isinstance(document, Mapping):
        return _placeholder_names(document)
    unresolved: set[str] = set()
    for raw_key, child in document.items():
        key = str(raw_key)
        unresolved.update(_SKYPILOT_PLACEHOLDER_RE.findall(key))
        if key not in _SKYPILOT_SHELL_FIELDS:
            unresolved.update(_placeholder_names(child))
    return unresolved


def assert_no_unresolved_placeholders(yaml_text: str) -> None:
    """Fail on bare ``${NAME}`` placeholders in rendered declarative fields.

    SkyPilot cannot resolve self-references such as ``envs: {PATH: ${PATH}:...}``,
    so declarative fields must be fully materialized before submit. ``setup`` and
    ``run`` are shell programs, however, where both ``${NAME}`` and
    ``${NAME:-default}`` are ordinary author-controlled shell syntax. Parsing the
    rendered YAML also means comments are ignored instead of being mistaken for
    executable placeholders.
    """

    try:
        documents = list(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError as exc:
        raise NpaWorkflowRenderError(
            f"rendered SkyPilot YAML is invalid while checking placeholders: {exc}"
        ) from exc
    unresolved = sorted(
        {
            name
            for document in documents
            for name in _document_declarative_placeholder_names(document)
        }
    )
    if unresolved:
        joined = ", ".join(f"${{{name}}}" for name in unresolved)
        raise NpaWorkflowRenderError(
            "rendered SkyPilot YAML still contains unresolved placeholders: "
            f"{joined}; resolve images and config before submit"
        )
