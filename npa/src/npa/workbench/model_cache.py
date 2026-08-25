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

Caching is on wherever turning it on does not mean inventing storage. A VM deploy
can create a directory on its own disk, so it does
(:data:`DEFAULT_DOCKER_HOST_CACHE`); on Kubernetes, submit adopts a Bound claim
named :data:`DEFAULT_MODEL_CACHE_CLAIM` if the operator applied the shipped
manifest. What stays off is anything that would conjure storage nobody asked for:
this module never creates a claim, chooses a storage class, or bills anyone for a
volume, so Kubernetes is inert until the claim exists and a Serverless Job caches
only once ``NPA_MODEL_CACHE_FILESYSTEM`` names one.
``NPA_MODEL_CACHE_DISABLED=1`` switches all of it off, and an explicit
``NPA_MODEL_CACHE_DIR`` overrides all of it.

Which runtime is asking matters, and callers must say. The variables name storage
in different worlds -- a claim exists only inside a cluster, a host path only on
the machine holding it, a Nebius filesystem only where the platform attaches it --
so a runtime that cannot mount the thing the operator configured must not export
the environment either. Exporting
``HF_HOME=/opt/npa-model-cache/huggingface`` at a runtime with nothing mounted
there does not produce a slow run, it produces a broken one: ``/opt`` is
root-owned in every workbench image and they all run unprivileged, so the first
``mkdir`` fails. That is how one operator exporting ``NPA_MODEL_CACHE_PVC`` for
their Kubernetes workflows would have broken their working Serverless Jobs. Hence
:data:`RUNTIME_KUBERNETES`, :data:`RUNTIME_DOCKER`, :data:`RUNTIME_SERVERLESS`,
:data:`RUNTIME_PREMOUNTED`, and no default for the ``runtime`` argument.

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
#: Nebius filesystem to attach to Serverless Jobs, as ``nebius ai job create
#: --volume`` names it. A Serverless Job has no cluster and no host to borrow
#: storage from, so this is the only thing it can mount.
MODEL_CACHE_FILESYSTEM_ENV = "NPA_MODEL_CACHE_FILESYSTEM"
#: Namespace to look in for the shipped claim, when the operator's SkyPilot pods do
#: not land in `default`.
MODEL_CACHE_NAMESPACE_ENV = "NPA_MODEL_CACHE_NAMESPACE"
#: Turn the cache off everywhere, including the defaults below.
MODEL_CACHE_DISABLED_ENV = "NPA_MODEL_CACHE_DISABLED"

#: The claim the shipped manifest creates. Submit looks for this name so that
#: applying the manifest is the whole opt-in: nothing to remember afterwards.
DEFAULT_MODEL_CACHE_CLAIM = "npa-model-cache"
#: Where a VM deploy keeps the cache when the operator names no other path.
#: Unlike a claim, a host directory needs no provisioning and costs nothing to
#: create, so a Docker deploy can default to caching instead of re-downloading
#: gated weights on every `docker rm -f` cycle. FHS-conventional, and on the same
#: disk the images it pulls already occupy.
DEFAULT_DOCKER_HOST_CACHE = "/var/lib/npa/model-cache"

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

#: A caller that provisions Kubernetes pods: it can mount a claim, and it can mount
#: a node directory (node-local rather than durable, and refused under a
#: ``restricted`` PodSecurity policy -- the operator's call, not ours).
RUNTIME_KUBERNETES = "kubernetes"
#: A caller that runs ``docker`` on a host: it can bind-mount a host directory.
RUNTIME_DOCKER = "docker"
#: A Serverless Job: no cluster, no host, but `nebius ai job create --volume` can
#: attach a Nebius filesystem to it.
RUNTIME_SERVERLESS = "serverless"
#: A caller that mounts nothing itself, and so can only use a cache that already
#: exists at a path someone else mounted -- in-container code reading the
#: environment a renderer set for it.
RUNTIME_PREMOUNTED = "premounted"

#: Which configured signal each runtime is allowed to act on. ``NPA_MODEL_CACHE_DIR``
#: is absent because every runtime honors it: it is the operator asserting the path
#: is already there, which is a claim only they can make.
_RUNTIME_BACKING: dict[str, tuple[str, ...]] = {
    RUNTIME_KUBERNETES: (MODEL_CACHE_PVC_ENV, MODEL_CACHE_HOST_PATH_ENV),
    RUNTIME_DOCKER: (MODEL_CACHE_HOST_PATH_ENV,),
    RUNTIME_SERVERLESS: (MODEL_CACHE_FILESYSTEM_ENV,),
    RUNTIME_PREMOUNTED: (),
}

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
    # OpenPI's policy server keeps its gated checkpoint here; its Deployment used
    # an emptyDir, so a rollout re-downloaded it on an already-running GPU.
    ("OPENPI_DATA_HOME", "openpi"),
    # LeIsaac stages NVIDIA USD scenes and the streaming client into these. Every
    # fetch is hash-verified and skipped when the file is already present, so a warm
    # cache turns the download into a checksum and nothing else changes.
    ("NPA_LEISAAC_CACHE_DIR", "leisaac"),
    ("LEISAAC_ASSETS_ROOT", "leisaac/assets/runtime"),
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


def model_cache_disabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the operator switched the cache off entirely."""

    value = str(_source(environ).get(MODEL_CACHE_DISABLED_ENV, "") or "").strip()
    return value.lower() in {"1", "true", "yes", "on"}


def model_cache_host_path(environ: Mapping[str, str] | None = None) -> str:
    """Return the host directory the operator configured, or ``""``.

    No default here on purpose. Kubernetes accepts a host path too, and turning an
    unset variable into one there would silently mount a node directory into every
    pod -- node-local rather than shared, and refused outright under a
    ``restricted`` PodSecurity policy. The Docker default lives in
    :func:`docker_model_cache_host_path`, where it is the caller's own disk.
    """

    source = _source(environ)
    if model_cache_disabled(source):
        return ""
    return _require_absolute(
        str(source.get(MODEL_CACHE_HOST_PATH_ENV, "") or ""),
        name=MODEL_CACHE_HOST_PATH_ENV,
    )


def docker_model_cache_host_path(environ: Mapping[str, str] | None = None) -> str:
    """Return the host directory a VM deploy should bind, defaulting to a real one.

    A host directory is storage the deploy can always create -- it needs no
    provisioning and costs nothing -- so there is no reason to make an operator ask
    for it by name. The alternative default, discarding every gated download when
    the container is replaced (and this deploy runs ``docker rm -f`` every time),
    is not one anybody would choose deliberately.
    """

    source = _source(environ)
    if model_cache_disabled(source):
        return ""
    return model_cache_host_path(source) or DEFAULT_DOCKER_HOST_CACHE


def resolve_model_cache_root(
    environ: Mapping[str, str] | None = None, *, runtime: str
) -> str:
    """Return the cache root ``runtime`` can use, or ``""`` when it has none.

    An empty result is the "no durable storage this caller can reach" answer, and
    callers must treat it as "keep your existing ephemeral default". It is never a
    path, because guessing one would put multi-gigabyte downloads somewhere the
    operator did not agree to and still lose them at the end of the run.

    ``runtime`` is required, and answers "what can you actually mount". A caller
    that mounts nothing (:data:`RUNTIME_PREMOUNTED`) sees only an explicit
    ``NPA_MODEL_CACHE_DIR``, so a claim configured for Kubernetes cannot send it to
    a path it has no storage for.
    """

    backing = _RUNTIME_BACKING.get(runtime)
    if backing is None:
        raise ModelCacheError(
            f"unknown model cache runtime {runtime!r}; expected one of "
            f"{sorted(_RUNTIME_BACKING)}"
        )
    source = _source(environ)
    if model_cache_disabled(source):
        return ""
    explicit = _require_absolute(
        str(source.get(MODEL_CACHE_DIR_ENV, "") or ""), name=MODEL_CACHE_DIR_ENV
    )
    if explicit:
        return explicit
    if MODEL_CACHE_PVC_ENV in backing and model_cache_pvc(source):
        return DEFAULT_MODEL_CACHE_MOUNT
    if runtime == RUNTIME_DOCKER and docker_model_cache_host_path(source):
        return DEFAULT_MODEL_CACHE_MOUNT
    if MODEL_CACHE_HOST_PATH_ENV in backing and model_cache_host_path(source):
        return DEFAULT_MODEL_CACHE_MOUNT
    if MODEL_CACHE_FILESYSTEM_ENV in backing and model_cache_filesystem(source):
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


def render_model_cache_shell(root: str, *, mounted: bool) -> str:
    """Return shell that materializes the cache tree and reports what it is.

    The log line matters: a re-download that should have been a cache hit is
    otherwise invisible, because "downloading 40 GB again" looks exactly like
    "downloading 40 GB the first time" in a stage log.

    ``mounted`` says whether this caller attached the volume itself. It must not
    claim persistence it cannot vouch for: with an explicit
    ``NPA_MODEL_CACHE_DIR`` the operator asserted the path is already a durable
    mount, and if they were wrong the honest log is the one that says who mounted
    what rather than the one that promises the weights survive.
    """

    dirs = model_cache_dirs(root)
    if not dirs:
        return ""
    import shlex

    quoted = " ".join(shlex.quote(path) for path in dirs)
    note = (
        "mounted here, weights persist across runs"
        if mounted
        else "not mounted by npa; persists only if this path already does"
    )
    return f"mkdir -p {quoted}\necho 'npa model cache: {root} ({note})' >&2\n"


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


def model_cache_filesystem(environ: Mapping[str, str] | None = None) -> str:
    """Return the Nebius filesystem a Serverless Job should mount, or ``""``.

    An ``s3://`` source is refused rather than accepted and quietly broken. The
    Hugging Face hub cache is a blobs/snapshots tree held together by symlinks,
    which a bucket mount does not implement, so it would corrupt exactly the cache
    it was asked to preserve -- and it would do so on the second run, not the first.
    """

    source = _source(environ)
    if model_cache_disabled(source):
        return ""
    value = str(source.get(MODEL_CACHE_FILESYSTEM_ENV, "") or "").strip()
    if not value:
        return ""
    if value.lower().startswith("s3://"):
        raise ModelCacheError(
            f"{MODEL_CACHE_FILESYSTEM_ENV} must name a filesystem, not a bucket: an "
            "S3 mount cannot represent the symlinked blobs/snapshots tree the "
            f"Hugging Face cache is made of, got {value!r}"
        )
    if ":" in value:
        raise ModelCacheError(
            f"{MODEL_CACHE_FILESYSTEM_ENV} is the volume source only; NPA supplies "
            f"the mount path ({DEFAULT_MODEL_CACHE_MOUNT}), got {value!r}"
        )
    return value


def serverless_model_cache_volume(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return ``nebius ai job create --volume`` arguments for the cache."""

    source = _source(environ)
    filesystem = model_cache_filesystem(source)
    root = resolve_model_cache_root(source, runtime=RUNTIME_SERVERLESS)
    if not filesystem or not root:
        return ()
    return (f"{filesystem}:{root}:rw",)


def docker_model_cache_volumes(
    *, root: str = "", host_path: str = "", environ: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Return ``docker run -v`` arguments binding the host cache into a container."""

    source = _source(environ)
    host = _require_absolute(
        str(host_path or "") or docker_model_cache_host_path(source),
        name=MODEL_CACHE_HOST_PATH_ENV,
    )
    mount_root = _require_absolute(
        str(root or "") or resolve_model_cache_root(source, runtime=RUNTIME_DOCKER),
        name=MODEL_CACHE_DIR_ENV,
    )
    if not host or not mount_root:
        return ()
    return (f"{host}:{mount_root}",)


__all__ = [
    "DEFAULT_DOCKER_HOST_CACHE",
    "DEFAULT_MODEL_CACHE_CLAIM",
    "DEFAULT_MODEL_CACHE_MOUNT",
    "DEFAULT_POD_CONTAINER_NAME",
    "MODEL_CACHE_DIR_ENV",
    "MODEL_CACHE_DISABLED_ENV",
    "MODEL_CACHE_ENV_NAMES",
    "MODEL_CACHE_FILESYSTEM_ENV",
    "MODEL_CACHE_HOST_PATH_ENV",
    "MODEL_CACHE_NAMESPACE_ENV",
    "MODEL_CACHE_LAYOUT",
    "MODEL_CACHE_PVC_ENV",
    "MODEL_CACHE_VOLUME_NAME",
    "RUNTIME_DOCKER",
    "RUNTIME_KUBERNETES",
    "RUNTIME_PREMOUNTED",
    "RUNTIME_SERVERLESS",
    "ModelCacheError",
    "docker_model_cache_host_path",
    "docker_model_cache_volumes",
    "model_cache_dirs",
    "model_cache_disabled",
    "model_cache_env",
    "model_cache_filesystem",
    "model_cache_host_path",
    "model_cache_pvc",
    "pod_config_with_model_cache",
    "render_model_cache_shell",
    "resolve_model_cache_root",
    "serverless_model_cache_volume",
]
