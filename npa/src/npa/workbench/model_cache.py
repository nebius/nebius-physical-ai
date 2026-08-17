"""One durable cache for the model weights the workbench downloads at run time.

The workbench images deliberately bake no model weights. NVIDIA's Cosmos
checkpoints and guardrails, GR00T, the Qwen VLMs, Wan 2.2, LTX, and the curator
towers are all either license-gated or too large to redistribute, so
``packaging-contract.yaml`` requires them to be fetched with the operator's own
token at run time. That contract is correct, but it only says *when* the bytes
arrive -- not *where they land*. Left unstated, every runtime picked a directory
that happened to be writable inside its image (``/tmp/hf_home``, an ``emptyDir``,
a path with no bind mount at all), so each download was discarded with the
container and the next run paid for it again: tens of gigabytes of egress and
minutes of GPU time per stage, repeated for every run of the same image.

This module is the single place that answers "where do downloaded weights live",
for every runtime NPA drives:

* SkyPilot tasks (:mod:`npa.orchestration.npa_workflow.skypilot_render`)
* Kubernetes sibling Jobs (:mod:`npa.workflows.sim2real.job_scheduling`)
* Workbench Serverless Jobs (:mod:`npa.serverless_common.env`)
* long-lived workbench containers on a VM (:mod:`npa.deploy.configurator`)

The cache root is *opt-in infrastructure*, because durability is a property of
storage that only the operator can supply: a ReadWriteMany PVC on Kubernetes, or
a data disk on a VM. Until one is configured, :func:`resolve_model_cache_root`
returns ``""`` and every caller keeps the ephemeral default it has always used,
so nothing changes implicitly. Once ``NPA_MODEL_CACHE_PVC`` (Kubernetes),
``NPA_MODEL_CACHE_HOST_PATH`` (VM/Docker), or an explicit
``NPA_MODEL_CACHE_DIR`` is set, every runtime redirects *all* of its weight
caches into that one tree and the second run of an image is a cache hit.

Object storage is intentionally not an option here. The Hugging Face hub cache is
a blobs/snapshots tree held together by symlinks, which S3-backed FUSE mounts do
not implement; a bucket mount would corrupt exactly the cache it was meant to
preserve. Durable weight storage has to be a real filesystem.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Explicit in-container cache root. Wins over every other signal.
MODEL_CACHE_DIR_ENV = "NPA_MODEL_CACHE_DIR"
#: ReadWriteMany PVC holding the cache for Kubernetes-scheduled work.
MODEL_CACHE_PVC_ENV = "NPA_MODEL_CACHE_PVC"
#: Host directory holding the cache for Docker-scheduled work on a VM.
MODEL_CACHE_HOST_PATH_ENV = "NPA_MODEL_CACHE_HOST_PATH"

#: Where the cache is mounted inside a container when a PVC or host path is
#: configured without an explicit ``NPA_MODEL_CACHE_DIR``. Deliberately outside
#: ``/tmp`` (which several images size as a small tmpfs) and outside ``$HOME``
#: (read-only in the Isaac and NRE images).
DEFAULT_MODEL_CACHE_MOUNT = "/opt/npa-model-cache"

#: Kubernetes volume name for the cache. SkyPilot appends to the lists inside
#: ``kubernetes.pod_config``, so this must not collide with a cluster-wide name.
MODEL_CACHE_VOLUME_NAME = "npa-model-cache"
#: SkyPilot names the application container in the pods it provisions.
DEFAULT_POD_CONTAINER_NAME = "ray-node"

_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?$")


class ModelCacheError(ValueError):
    """Raised when a configured model cache location is unusable."""


#: Every cache variable a workbench runtime honors, as ``(name, path under root)``.
#:
#: This is one flat family rather than a per-tool selection on purpose. Each name
#: is read by exactly the tool that defined it, so setting all of them everywhere
#: is inert for the tools that ignore it -- whereas selecting a subset per stage
#: means a mis-mapped stage silently sends a multi-gigabyte download back to a
#: container-local directory, which is the failure this module exists to remove.
#: Every entry is a variable something in the tree actually reads:
#:
#: * ``HF_*`` / ``TORCH_HOME``: huggingface_hub, transformers, datasets, torch.
#:   ``HF_HOME`` alone would cover the hub cache, but vendor entrypoints read the
#:   narrower names directly and one unset name is enough to leak a download.
#: * ``COSMOS_HF_CACHE`` / ``NLTK_DATA``: baked by the Cosmos and
#:   Cosmos-Transfer2.5 images (``npa/docker/workbench/cosmos*/Dockerfile``).
#: * ``NPA_COSMOS3_CACHE`` / ``COSMOS_DOWNLOAD_CACHE_DIR``: the Cosmos 3 framework
#:   checkout and checkpoint cache (:mod:`npa.workbench.cosmos.cosmos3`).
#: * ``NPA_COSMOS_REASON*_CACHE``: per-family Reason checkpoints
#:   (:mod:`npa.workbench.cosmos.reason`).
#: * ``NPA_COSMOS_CURATE_WEIGHTS_DIR``: rebinds upstream Cosmos-Curate's
#:   hardcoded ``/config/models`` (:mod:`npa.workbench.cosmos_curate.models`).
#: * ``HF_LEROBOT_HOME`` / ``LEROBOT_HF_HOME``: LeRobot datasets and policies.
#: * ``WAN22_CACHE_DIR`` / ``NPA_LTX_MODEL_CACHE``: the BYOF video models, whose
#:   images ship zero weights.
MODEL_CACHE_LAYOUT: tuple[tuple[str, str], ...] = (
    ("HF_HOME", "huggingface"),
    ("HF_HUB_CACHE", "huggingface/hub"),
    ("HUGGINGFACE_HUB_CACHE", "huggingface/hub"),
    ("TRANSFORMERS_CACHE", "huggingface/hub"),
    ("HF_DATASETS_CACHE", "huggingface/datasets"),
    # Xet is Hugging Face's chunked transfer backend; its chunk cache is as large
    # as the download it accelerates, so re-fetching it defeats the point.
    ("HF_XET_CACHE", "huggingface/xet"),
    ("TORCH_HOME", "torch"),
    ("COSMOS_HF_CACHE", "huggingface"),
    ("NLTK_DATA", "nltk"),
    ("NPA_COSMOS3_CACHE", "cosmos3"),
    ("COSMOS_DOWNLOAD_CACHE_DIR", "cosmos3/downloads"),
    ("NPA_COSMOS_REASON_CACHE", "huggingface/cosmos-reason2"),
    ("NPA_COSMOS_REASON2_CACHE", "huggingface/cosmos-reason2"),
    ("NPA_COSMOS_REASON3_CACHE", "huggingface/cosmos-reason2-2b"),
    ("NPA_COSMOS_CURATE_WEIGHTS_DIR", "cosmos-curate/models"),
    ("HF_LEROBOT_HOME", "lerobot"),
    ("LEROBOT_HF_HOME", "lerobot"),
    ("WAN22_CACHE_DIR", "wan2.2"),
    ("NPA_LTX_MODEL_CACHE", "ltx-2.5"),
)

#: Every variable this module can set. Callers that filter an environment down to
#: an allow-list (the sibling-Job env builder does) need the whole family, or a
#: dropped variable sends one tool's download back to a container-local directory
#: while the rest of the stage uses the durable cache.
MODEL_CACHE_ENV_NAMES: frozenset[str] = frozenset(
    {MODEL_CACHE_DIR_ENV, *(name for name, _ in MODEL_CACHE_LAYOUT)}
)


def _source(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    import os

    return os.environ


def _require_absolute(value: str, *, name: str) -> str:
    path = str(value or "").strip().rstrip("/")
    if not path:
        return ""
    if not path.startswith("/"):
        raise ModelCacheError(
            f"{name} must be an absolute path so every runtime agrees on the "
            f"cache location, got {value!r}"
        )
    return path


def model_cache_pvc(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated ReadWriteMany PVC name holding the cache, or ``""``."""

    pvc = str(_source(environ).get(MODEL_CACHE_PVC_ENV, "") or "").strip()
    if not pvc:
        return ""
    if len(pvc) > 253 or not _DNS_SUBDOMAIN_RE.fullmatch(pvc):
        raise ModelCacheError(
            f"{MODEL_CACHE_PVC_ENV} is not a DNS-safe PVC name: {pvc!r}"
        )
    return pvc


def model_cache_host_path(environ: Mapping[str, str] | None = None) -> str:
    """Return the host directory holding the cache for Docker runs, or ``""``."""

    return _require_absolute(
        str(_source(environ).get(MODEL_CACHE_HOST_PATH_ENV, "") or ""),
        name=MODEL_CACHE_HOST_PATH_ENV,
    )


def resolve_model_cache_root(environ: Mapping[str, str] | None = None) -> str:
    """Return the in-container cache root, or ``""`` when none is configured.

    An empty result is the "no durable storage was supplied" answer, and callers
    must treat it as "keep your existing ephemeral default". It is never a path,
    because guessing one would put multi-gigabyte downloads somewhere the operator
    did not agree to and still lose them at the end of the run.
    """

    source = _source(environ)
    explicit = _require_absolute(
        str(source.get(MODEL_CACHE_DIR_ENV, "") or ""), name=MODEL_CACHE_DIR_ENV
    )
    if explicit:
        return explicit
    if model_cache_pvc(source) or model_cache_host_path(source):
        return DEFAULT_MODEL_CACHE_MOUNT
    return ""


def model_cache_env(root: str) -> dict[str, str]:
    """Return every cache variable a runtime must export to use ``root``.

    Empty when ``root`` is empty, so wiring this into a runtime is a no-op until
    the operator supplies durable storage.
    """

    base = _require_absolute(root, name=MODEL_CACHE_DIR_ENV)
    if not base:
        return {}
    env = {MODEL_CACHE_DIR_ENV: base}
    for name, relative in MODEL_CACHE_LAYOUT:
        env[name] = f"{base}/{relative}"
    return env


def model_cache_dirs(root: str) -> tuple[str, ...]:
    """Return the directories to create before a stage downloads weights.

    ``huggingface_hub`` creates its own tree, but only if the parent is writable;
    on a freshly provisioned volume the mount root is owned by root and the stage
    is not, so the directories are created up front where the failure is a clear
    message instead of a stack trace part-way through a download.
    """

    env = model_cache_env(root)
    return tuple(
        sorted({value for key, value in env.items() if key != MODEL_CACHE_DIR_ENV})
    )


def render_model_cache_shell(root: str) -> str:
    """Return shell that materializes the cache tree and reports its durability.

    The log line matters: a re-download that should have been a cache hit is
    otherwise invisible, because "downloading 40 GB again" looks exactly like
    "downloading 40 GB the first time" in a stage log.
    """

    dirs = model_cache_dirs(root)
    if not dirs:
        return ""
    import shlex

    quoted = " ".join(shlex.quote(path) for path in dirs)
    return (
        f"mkdir -p {quoted}\n"
        f"echo 'npa model cache: {root} (weights persist across runs)' >&2\n"
    )


def pod_config_with_model_cache(
    pod_config: Mapping[str, Any] | None,
    *,
    root: str,
    pvc: str = "",
    host_path: str = "",
    container_names: Sequence[str] = (DEFAULT_POD_CONTAINER_NAME,),
) -> dict[str, Any]:
    """Return ``pod_config`` with the durable cache volume mounted at ``root``.

    Returns the input unchanged when no cache root is given, or when the caller's
    own pod config already mounts something at ``root`` -- a spec that took
    explicit ownership of the path must win over this default.

    The mount is read-write, unlike the Isaac runtime cache: Isaac's closure is
    warmed by one CPU Job and then consumed read-only, while weights arrive
    lazily from whichever stage needs them first and simply accumulate.
    """

    import copy

    base: dict[str, Any] = copy.deepcopy(dict(pod_config or {}))
    mount_root = _require_absolute(root, name=MODEL_CACHE_DIR_ENV)
    if not mount_root:
        return base
    claim = str(pvc or "").strip()
    host = _require_absolute(str(host_path or ""), name=MODEL_CACHE_HOST_PATH_ENV)
    if not claim and not host:
        raise ModelCacheError(
            "a Kubernetes model cache needs either "
            f"{MODEL_CACHE_PVC_ENV} or {MODEL_CACHE_HOST_PATH_ENV}"
        )

    spec = base.setdefault("spec", {})
    if not isinstance(spec, dict):
        raise ModelCacheError("pod_config.spec must be a mapping")
    volumes = spec.setdefault("volumes", [])
    if not isinstance(volumes, list):
        raise ModelCacheError("pod_config.spec.volumes must be a list")
    containers = [
        item for item in (spec.get("containers") or []) if isinstance(item, dict)
    ]
    if any(
        str(mount.get("mountPath") or "").rstrip("/") == mount_root
        for container in containers
        for mount in (container.get("volumeMounts") or [])
        if isinstance(mount, dict)
    ):
        return base
    if any(
        str(item.get("name") or "") == MODEL_CACHE_VOLUME_NAME
        for item in volumes
        if isinstance(item, dict)
    ):
        return base

    volumes.append(
        {"name": MODEL_CACHE_VOLUME_NAME, "persistentVolumeClaim": {"claimName": claim}}
        if claim
        else {
            "name": MODEL_CACHE_VOLUME_NAME,
            "hostPath": {"path": host, "type": "DirectoryOrCreate"},
        }
    )
    by_name = {str(item.get("name") or ""): item for item in containers}
    for raw_name in container_names:
        name = str(raw_name).strip()
        if not name:
            continue
        container = by_name.get(name)
        if container is None:
            container = {"name": name}
            containers.append(container)
            by_name[name] = container
        mounts = container.setdefault("volumeMounts", [])
        if not isinstance(mounts, list):
            raise ModelCacheError(
                f"pod_config container {name!r} volumeMounts must be a list"
            )
        mounts.append({"name": MODEL_CACHE_VOLUME_NAME, "mountPath": mount_root})
    spec["containers"] = containers
    return base


def docker_model_cache_volumes(
    *, root: str = "", host_path: str = "", environ: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Return ``docker run -v`` arguments binding the host cache into a container."""

    source = _source(environ)
    host = _require_absolute(
        str(host_path or "") or model_cache_host_path(source),
        name=MODEL_CACHE_HOST_PATH_ENV,
    )
    mount_root = _require_absolute(
        str(root or "") or resolve_model_cache_root(source), name=MODEL_CACHE_DIR_ENV
    )
    if not host or not mount_root:
        return ()
    return (f"{host}:{mount_root}",)


__all__ = [
    "DEFAULT_MODEL_CACHE_MOUNT",
    "DEFAULT_POD_CONTAINER_NAME",
    "MODEL_CACHE_DIR_ENV",
    "MODEL_CACHE_ENV_NAMES",
    "MODEL_CACHE_HOST_PATH_ENV",
    "MODEL_CACHE_LAYOUT",
    "MODEL_CACHE_PVC_ENV",
    "MODEL_CACHE_VOLUME_NAME",
    "ModelCacheError",
    "docker_model_cache_volumes",
    "model_cache_dirs",
    "model_cache_env",
    "model_cache_host_path",
    "model_cache_pvc",
    "pod_config_with_model_cache",
    "render_model_cache_shell",
    "resolve_model_cache_root",
]
