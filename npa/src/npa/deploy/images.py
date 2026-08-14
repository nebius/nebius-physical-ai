"""Shared Workbench container image naming."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any

# Primary public Workbench registry (eu-north1). A registry path is a public
# locator, not a credential: pulls are still gated by the registry pull secret /
# IAM token, which are never committed. Operators can override it with NPA_REGISTRY
# or `container_registry` in ~/.npa/config.yaml.
DEFAULT_CONTAINER_REGISTRY_ID = "e00cm0vc6t09m0z5gw"
DEFAULT_CONTAINER_REGISTRY = f"cr.eu-north1.nebius.cloud/{DEFAULT_CONTAINER_REGISTRY_ID}"
# Mirror registry (us-central1) used for region-agnostic failover: every tool
# image is mirrored to both this and the primary (eu-north1) registry, so a pull
# succeeds regardless of the caller's region — e.g. an in-cluster us-central1 pull
# cannot reach the cross-region eu-north1 registry, and vice versa. A registry
# path is a public locator, not a credential. Override with NPA_BACKUP_REGISTRY.
BACKUP_CONTAINER_REGISTRY = "cr.us-central1.nebius.cloud/u00j7q4jjkahvsx0jy"
DEFAULT_VLM_IMAGE_ENV = "NPA_VLM_IMAGE"
DEFAULT_WORKBENCH_IMAGE_ENV = "NPA_WORKBENCH_IMAGE"
SONIC_IMAGE_MANIFEST_RESOURCE = "sonic_image_manifest.json"
WAN_IMAGE_MANIFEST_RESOURCE = "wan2_2_image_manifest.json"

CONTAINER_IMAGE_NAMES = {
    "lerobot": "npa-lerobot",
    "lerobot-policy": "npa-lerobot-policy",
    "genesis": "npa-genesis",
    "isaac-lab": "npa-isaac-lab",
    "cosmos": "npa-cosmos",
    "cosmos2-transfer": "npa-cosmos2-transfer",
    "cosmos3": "npa-cosmos3",
    "cosmos3-reason": "npa-cosmos3-reason",
    "cosmos-curate": "npa-cosmos-curate",
    "cosmos-evaluator": "npa-cosmos-evaluator",
    "groot": "npa-groot",
    "fiftyone": "npa-fiftyone",
    "sonic": "npa-sonic",
    "retargeting": "npa-retargeting",
    "envgen": "npa-envgen",
    "reference-policy": "npa-reference-policy",
    "lerobot-vlm-rl": "npa-lerobot-vlm-rl",
    "loop-eval": "npa-loop-eval",
    "rerun-viewer": "npa-rerun-viewer",
    "foxglove-embed": "npa-foxglove-embed",
    "lichtblick": "npa-lichtblick",
    "lancedb": "npa-lancedb",
    "detection-training": "npa-detection-training",
    "wan2-2": "npa-wan2-2",
}

# Public-image publication must enforce the digest-bound SkyPilot bootstrap
# attestation only for images that declare that build contract in
# docker/workbench/packaging-contract.yaml.  Keep this packaged copy explicit:
# an installed npa wheel does not carry the repository's Docker packaging tree.
# npa/tests/docker/test_packaging_contract.py locks the two inventories together.
SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS: frozenset[str] = frozenset(
    {
        "cosmos2-transfer",
        "cosmos-curate",
        "cosmos-evaluator",
        "fiftyone",
    }
)

# Images for these tool repositories may carry the bootstrap-contract label only
# through a separately checked derived Dockerfile, while the canonical image does
# not satisfy the same contract. A label cannot distinguish those two sources and
# is therefore never sufficient evidence. Submit ignores both the label and any
# cached label-backed result and runs the exact-digest capability probe instead.
# The packaging-contract guard locks this inventory to
# `derived_skypilot_bootstrap_contract.verification: runtime_probe_required`.
SKYPILOT_BOOTSTRAP_RUNTIME_PROBED_TOOLS: frozenset[str] = frozenset({"groot"})


def requires_skypilot_bootstrap_runtime_probe(image: str) -> bool:
    """Whether ``image`` belongs to a repository whose label is only a hint."""

    raw = str(image or "").strip().removeprefix("docker:").partition("@")[0]
    leaf = raw.rsplit("/", 1)[-1].split(":", 1)[0]
    return leaf in {
        CONTAINER_IMAGE_NAMES[tool]
        for tool in SKYPILOT_BOOTSTRAP_RUNTIME_PROBED_TOOLS
    }

# Tools whose built image may NOT be published to a public/anonymous registry,
# because it bakes a runtime we are not licensed to redistribute.
#
# The Isaac-family membership is deliberately empty: those images were
# re-architected to fetch Isaac at runtime. Cosmos3 serving is restricted for a
# separate reason: its pinned vLLM-Omni base embeds the NVIDIA Deep Learning
# Container License and the thin wrapper does not establish the license's
# material-additional-functionality and downstream-terms conditions for an
# anonymous standalone GHCR distribution. Operators may build it into their own
# registry instead.
#
# It used to hold {"isaac-lab", "sonic", "groot"}, because those images baked NVIDIA
# Omniverse Kit (Isaac Sim): the Isaac Sim SOURCE is Apache-2.0, but the shipped
# binary bundles the Kit SDK + NVIDIA assets, and both the isaacsim AND isaaclab
# PyPI packages declare "License: NVIDIA Proprietary Software". Publishing them
# would have made us the third-party redistributor of Omniverse Kit, which needs
# an NVIDIA AI Enterprise license.
#
# They were re-architected to contain no NVIDIA Isaac bytes at all: Isaac Sim and
# Isaac Lab are fetched on first run from pypi.nvidia.com, into a cache volume,
# under the OPERATOR's own EULA acceptance, and the image refuses to start Isaac
# without it (npa/docker/workbench/common/isaac_bootstrap.sh). NVIDIA delivers to
# each operator directly, so we are never the redistributor — the same pattern the
# workbench already uses for gated model weights. Verified mechanically against the
# built images by npa/scripts/scan_image_omniverse_payload.py.
#
# The compatibility name predates this non-Omniverse member. Keep it until a
# deliberate API rename; the behavior is the general restricted-runtime guard.
# Kept in sync with packaging-contract.yaml's `redistribution:` fields by
# npa/tests/deploy/test_public_publish.py.
OMNIVERSE_RESTRICTED_TOOLS: frozenset[str] = frozenset({"cosmos3-serving"})

# Images built FROM a restricted tool image, so they inherit whatever it bakes and
# the same no-public-redistribution rule. They are not separate
# CONTAINER_IMAGE_NAMES entries (they are variants of their parent tool), so they
# never reach publicly_publishable_tools(); they are listed here so operator-facing
# output can name every excluded image without hardcoding it at the call site.
# Empty for the same reason as above: ``sonic-mujoco`` inherits sonic's runtime-fetch
# architecture and adds no Isaac and no Omniverse assets of its own.
OMNIVERSE_RESTRICTED_DERIVED_IMAGES: frozenset[str] = frozenset()

# Public mirror registry for the OSS-redistributable image subset. Nebius CR does
# NOT support anonymous/public pulls and has no cross-tenant / all-authenticated
# grant, so making images pullable by any Nebius tenant (or anyone) means
# mirroring the publicly_publishable_tools() set to a public-capable registry.
# GHCR is the default (public, anonymous pull, native to the GitHub org). A
# registry path is a public locator, not a credential. Override with
# NPA_PUBLIC_REGISTRY; consumers in any tenant pull the OSS images by setting
# NPA_REGISTRY to this value.
PUBLIC_CONTAINER_REGISTRY_ENV = "NPA_PUBLIC_REGISTRY"
DEFAULT_PUBLIC_CONTAINER_REGISTRY = "ghcr.io/nebius/nebius-physical-ai"

# Registry hosts that serve anonymous/public pulls. Resolving a restricted image
# against one of these is always wrong: either it is not there (we never publish
# it) or someone has published a non-redistributable runtime to third parties.
# Private registries are deliberately absent — an operator building the image
# into their OWN registry is the licensed path, whichever registry that is.
PUBLIC_REGISTRY_HOSTS = frozenset(
    {
        "ghcr.io",
        "docker.io",
        "index.docker.io",
        "registry-1.docker.io",
        "quay.io",
        "public.ecr.aws",
    }
)

SUPPORTED_TOOL_VERSIONS = {
    # Default LeRobot image release. Selectable package versions and their
    # image tags live in lerobot_version_manifest.json.
    "lerobot": "cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "lerobot-policy": "0.1.1",
    "genesis": "cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "isaac-lab": "2.3.2.post1",
    "cosmos": "cu128-torch27-sm100-1.0.9-20260803T002017Z",
    "cosmos2-transfer": "2.5.1-skypilot-ready-20260801T053000Z",
    # Additive r2 release of cosmos-framework 1.2.2 (pinned commit 5e67049c) +
    # torch cu130. The immutable 1.2.2-cu130 tag remains rollback provenance.
    # No weights baked; gated Cosmos3 checkpoints download at runtime.
    "cosmos3": "1.2.2-cu130-r2",
    "cosmos3-reason": "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "cosmos-curate": "0.1.2-skypilot-v1-20260813T164700Z",
    "cosmos-evaluator": "0.1.2-skypilot-v1-20260813T164700Z",
    "groot": "0.1.0",
    "fiftyone": "1.15.0.post1",
    "sonic": "cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "retargeting": "0.1.1",
    "envgen": "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "reference-policy": "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "lerobot-vlm-rl": "cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "loop-eval": "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "rerun-viewer": "0.31.4",
    # Tracks the pinned @foxglove/embed SDK release (npa.workbench.foxglove).
    "foxglove-embed": "0.58.0",
    # Lichtblick (MPL-2.0): OSS, Foxglove-compatible static web viewer bundle.
    "lichtblick": "1.26.0",
    "lancedb": "cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z",
    "detection-training": "bdd100k-golden-eval-smoke-20260614T210000Z",
    # Public-eligible Wan source/CPU base; CUDA torch is operator-gated runtime fetch.
    "wan2-2": "2.2-ti2v5b-rtfetch-cu128-20260809T011658Z-r7",
    "nebius-cli": "0.12.254",
    "terraform": "~> 0.5.201",
    "terraform-cli": "1.13.3",
}


@lru_cache(maxsize=1)
def sonic_image_manifest() -> dict[str, Any]:
    """Return the packaged SONIC image compatibility manifest."""

    text = resources.files(__package__).joinpath(SONIC_IMAGE_MANIFEST_RESOURCE).read_text(
        encoding="utf-8"
    )
    payload = json.loads(text)
    if payload.get("format") != "npa_sonic_image_manifest_v1":
        raise RuntimeError("Unsupported SONIC image manifest format")
    return payload


@lru_cache(maxsize=1)
def wan_accepted_image_manifest() -> dict[str, Any]:
    """Return the immutable image/runtime/GPU proof tuple allowed for publication."""

    text = (
        resources.files(__package__)
        .joinpath(WAN_IMAGE_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError("Wan accepted image manifest must be a JSON object")
    if payload.get("format") != "npa_wan_accepted_image_manifest_v1":
        raise RuntimeError("Unsupported Wan accepted image manifest format")
    if payload.get("tag") != SUPPORTED_TOOL_VERSIONS["wan2-2"]:
        raise RuntimeError(
            "Wan accepted image manifest tag drifted from the supported tag"
        )
    return payload


def sonic_image_variants() -> dict[str, dict[str, Any]]:
    """Return SONIC image manifest entries by variant id."""

    variants: dict[str, dict[str, Any]] = {}
    for item in sonic_image_manifest().get("images", []):
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("id", ""))
        if variant_id:
            variants[variant_id] = item
    return variants


def supported_tool_version(tool: str) -> str:
    if tool == "sonic":
        return str(_default_sonic_image()["tag"])

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    for directory in Path(__file__).resolve().parents:
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            return str(data["tool"]["npa"]["supported-tools"][tool])
    try:
        return SUPPORTED_TOOL_VERSIONS[tool]
    except KeyError as exc:
        raise RuntimeError(f"Could not find supported version for tool: {tool}") from exc


def public_mirror_tag_for_tool(tool: str) -> str:
    """Return the exact repository pin that the public mirror must carry.

    SONIC's normal resolver selects a hardware variant and defaults to the L40S
    ``0.1.2`` image. The public inventory contract instead pins the validated
    cross-architecture Kubernetes runtime from ``SUPPORTED_TOOL_VERSIONS``. A
    publisher that called ``supported_tool_version('sonic')`` would silently
    mirror only the default variant and leave the repository pin unavailable.
    """
    if tool == "sonic":
        return SUPPORTED_TOOL_VERSIONS[tool]
    return supported_tool_version(tool)


def supported_lerobot_versions() -> tuple[str, ...]:
    """Return LeRobot versions supported by the workbench (default first)."""

    from npa.workbench.lerobot.version_compat import supported_lerobot_versions as _versions

    return _versions()


def resolve_lerobot_image_tag(version: str | None = None) -> str:
    """Resolve the validated image tag for a supported LeRobot package version."""

    from npa.workbench.lerobot.version_compat import lerobot_version_entry

    entry = lerobot_version_entry(version)
    return str(entry.get("image_tag") or entry["version"])


def sonic_image_variant_for_gpu(gpu_target: str | None = None) -> str:
    """Return the SONIC image variant id for a GPU or provider target."""

    manifest = sonic_image_manifest()
    default = str(manifest.get("default_variant", "sonic-l40s-baked"))
    normalized = _normalize_gpu_target(gpu_target)
    if not normalized:
        return default
    for rule in manifest.get("gpu_selection", []):
        if not isinstance(rule, dict):
            continue
        variant = str(rule.get("variant", ""))
        for match in rule.get("matches", []):
            if str(match).lower() in normalized:
                return variant
    return default


def sonic_image_entry(
    *,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> dict[str, Any]:
    """Return the SONIC manifest entry selected by variant or GPU target."""

    variants = sonic_image_variants()
    if image_variant:
        resolved = _normalize_sonic_variant(image_variant, variants)
    else:
        resolved = sonic_image_variant_for_gpu(gpu_target)
    try:
        return variants[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown SONIC image variant {resolved!r}; choose one of: {choices}") from exc


def container_image_for_tool(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> str:
    """Return the fully qualified image ref for a Workbench tool."""
    if tool == "sonic":
        entry = sonic_image_entry(gpu_target=gpu_target, image_variant=image_variant)
        image_name = str(entry["name"])
        resolved_tag = tag or str(entry["tag"])
    else:
        if image_variant:
            raise ValueError(f"Image variants are only defined for SONIC, got tool={tool!r}")
        image_name = CONTAINER_IMAGE_NAMES[tool]
        resolved_tag = tag or supported_tool_version(tool)
    resolved_registry = registry or _primary_registry()
    if not is_publicly_redistributable(tool) and is_public_registry(resolved_registry):
        raise ValueError(
            f"{tool!r} is not publicly redistributable and is never distributed from a "
            f"public registry, so {resolved_registry!r} cannot serve it. Build it into "
            f"your own registry (npa/docker/workbench/<tool>/build.sh --registry "
            f"<your-registry> --push) and point NPA_REGISTRY at that registry; see "
            f"docs/workbench/container-packaging.md."
        )
    return f"{resolved_registry.rstrip('/')}/{image_name}:{resolved_tag}"


def tool_for_image_name(image_name: str) -> str:
    """Reverse ``CONTAINER_IMAGE_NAMES``: ``npa-cosmos-curate`` -> ``cosmos-curate``."""

    wanted = str(image_name or "").strip()
    if not wanted:
        return ""
    for tool, name in CONTAINER_IMAGE_NAMES.items():
        if name == wanted:
            return tool
    return ""


def build_and_push_command(image: str) -> str:
    """Return the buildx command that produces ``image``, or "" if it is not ours.

    A missing workbench image is the one preflight failure whose fix is entirely
    mechanical, so the remedy carries the command rather than pointing at a guide
    whose tags can drift from these pins.
    """

    ref = str(image or "").removeprefix("docker:").strip()
    if "/" not in ref:
        return ""
    repository = ref.rsplit("/", 1)[-1]
    image_name = repository.rsplit(":", 1)[0] if ":" in repository else repository
    tool = tool_for_image_name(image_name)
    if not tool:
        return ""
    dockerfile = _workbench_dockerfile(tool)
    if not dockerfile:
        # Not every tool builds from npa/docker/workbench/<tool>/Dockerfile
        # (sim2real tools in particular live elsewhere). Printing a command whose
        # -f path does not exist is worse than printing none.
        return ""
    registry = ref.rsplit("/", 1)[0]
    tag = supported_tool_version(tool)
    return (
        f"docker buildx build --push -f {dockerfile} "
        f"-t {registry}/{image_name}:{tag} npa"
    )


def _workbench_dockerfile(tool: str) -> str:
    """Return the repo-relative Dockerfile for ``tool``, or "" if there is none.

    Resolved against the checkout when one is reachable; an installed npa has no
    docker/ tree, and there the conventional path is still the right advice.
    """

    relative = f"npa/docker/workbench/{tool}/Dockerfile"
    package_root = Path(__file__).resolve().parents[2]
    repo_root = package_root.parent.parent
    if not (repo_root / "npa" / "docker").is_dir():
        return relative
    return relative if (repo_root / relative).is_file() else ""


def registry_from_id(registry_id: str) -> str:
    """Build a full registry locator from a bare Nebius registry id.

    A bare id (``NPA_REGISTRY_ID``) is expanded against the primary region so it
    resolves the same way on every registry path (see ``resolve_container_registry``).
    """
    return f"cr.eu-north1.nebius.cloud/{registry_id.strip()}"


def registry_from_env() -> str:
    """Return the registry from NPA_REGISTRY, then NPA_REGISTRY_ID, else ""."""
    explicit = os.environ.get("NPA_REGISTRY", "").strip()
    if explicit:
        return explicit
    registry_id = os.environ.get("NPA_REGISTRY_ID", "").strip()
    if registry_id:
        return registry_from_id(registry_id)
    return ""


def primary_container_registry() -> str:
    """Resolve the primary registry: NPA_REGISTRY, then NPA_REGISTRY_ID, then default."""
    return registry_from_env() or DEFAULT_CONTAINER_REGISTRY


_primary_registry = primary_container_registry


def backup_container_registry() -> str:
    """Resolve the backup registry override, or the committed default."""
    return os.environ.get("NPA_BACKUP_REGISTRY", "").strip() or BACKUP_CONTAINER_REGISTRY


def container_image_candidates(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
    preferred_region: str | None = None,
) -> list[str]:
    """Return image refs to try in order across both mirror registries.

    Callers that support pull failover should iterate these so a pull works
    region-agnostically: every image is mirrored to both registries, and a caller
    that cannot reach one region (cross-region 403, or an identity without read on
    the other project's registry) falls through to the other. ``preferred_region``
    reorders so the caller's local-region registry (``cr.<region>.nebius.cloud``)
    is tried first, avoiding a guaranteed-denied cross-region attempt.
    """
    primary = container_image_for_tool(
        tool, registry=registry, tag=tag, gpu_target=gpu_target, image_variant=image_variant
    )
    candidates = [primary]
    backup_registry = backup_container_registry()
    if backup_registry:
        backup = container_image_for_tool(
            tool, registry=backup_registry, tag=tag, gpu_target=gpu_target, image_variant=image_variant
        )
        if backup != primary:
            candidates.append(backup)
    region = (preferred_region or "").strip().lower()
    if region:
        host_prefix = f"cr.{region}.nebius.cloud/"
        local = [ref for ref in candidates if ref.startswith(host_prefix)]
        other = [ref for ref in candidates if not ref.startswith(host_prefix)]
        candidates = local + other
    return candidates


def public_container_registry() -> str:
    """Return the public mirror registry: ``NPA_PUBLIC_REGISTRY`` or the default."""
    return (
        os.environ.get(PUBLIC_CONTAINER_REGISTRY_ENV, "").strip()
        or DEFAULT_PUBLIC_CONTAINER_REGISTRY
    )


def is_public_registry(registry: str) -> bool:
    """Whether a registry serves anonymous/public pulls.

    True for the well-known public hosts and for whatever registry is configured
    as our public mirror. A Nebius (or other private) registry is not public: an
    operator's own registry is exactly where a restricted image is supposed to
    live.
    """
    candidate = registry.strip().rstrip("/")
    if not candidate:
        return False
    host = candidate.split("/", 1)[0].lower()
    if host in PUBLIC_REGISTRY_HOSTS:
        return True
    mirror = public_container_registry().strip().rstrip("/")
    return bool(mirror) and candidate.lower() == mirror.lower()


def is_publicly_redistributable(tool: str) -> bool:
    """Whether a tool image may be published to a public/anonymous registry.

    ``False`` for any tool in ``OMNIVERSE_RESTRICTED_TOOLS`` — images that bake a
    runtime we may not redistribute, which are licensed for internal-R&D /
    build-your-own use only. See the set's comment for current membership.
    """
    return tool not in OMNIVERSE_RESTRICTED_TOOLS


def omniverse_restricted_image_names() -> list[str]:
    """Return every image name excluded from public registries (tools + variants)."""
    return sorted(OMNIVERSE_RESTRICTED_TOOLS | OMNIVERSE_RESTRICTED_DERIVED_IMAGES)


def publicly_publishable_tools() -> list[str]:
    """Return the workbench tools that are OSS-redistributable to a public registry.

    Excludes anything in ``OMNIVERSE_RESTRICTED_TOOLS``. The Isaac images now
    fetch Isaac Sim / Isaac Lab at run time under the operator's own EULA
    acceptance, so every entry in ``CONTAINER_IMAGE_NAMES`` remains publishable;
    the separately contracted Cosmos3 serving image stays build-your-own.
    """
    return sorted(tool for tool in CONTAINER_IMAGE_NAMES if is_publicly_redistributable(tool))


def default_vlm_image(*, registry: str | None = None) -> str:
    """Return the default self-hosted VLM workflow image, honoring BYO override."""

    override = os.environ.get(DEFAULT_VLM_IMAGE_ENV, "").strip()
    if override:
        return override
    return container_image_for_tool("cosmos", registry=registry)


def default_workbench_image(*, registry: str | None = None) -> str:
    """Return the default generic Workbench workflow image, honoring BYO override."""

    override = os.environ.get(DEFAULT_WORKBENCH_IMAGE_ENV, "").strip()
    if override:
        return override
    return container_image_for_tool("genesis", registry=registry)


def _default_sonic_image() -> dict[str, Any]:
    return sonic_image_entry(image_variant=str(sonic_image_manifest().get("default_variant", "")))


def _normalize_gpu_target(gpu_target: str | None) -> str:
    return (gpu_target or "").strip().lower().replace("_", "-")


def _normalize_sonic_variant(image_variant: str, variants: dict[str, dict[str, Any]]) -> str:
    normalized = image_variant.strip().lower().replace("_", "-")
    aliases = {
        "baked": "sonic-l40s-baked",
        "l40s": "sonic-l40s-baked",
        "l40s-baked": "sonic-l40s-baked",
        "host-mounted": "sonic-k8s-host-mounted",
        "host": "sonic-k8s-host-mounted",
        "k8s": "sonic-k8s-host-mounted",
        "rtx": "sonic-k8s-host-mounted",
        "rtxpro": "sonic-k8s-host-mounted",
        "rtx-pro": "sonic-k8s-host-mounted",
        "rtx6000": "sonic-k8s-host-mounted",
        "rtx-pro-6000": "sonic-k8s-host-mounted",
        "mujoco": "sonic-mujoco-h100-mvp",
        "h100": "sonic-mujoco-h100-mvp",
        "h200": "sonic-mujoco-h100-mvp",
        "sonic-mujoco": "sonic-mujoco-h100-mvp",
        "mvp": "sonic-mujoco-h100-mvp",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown SONIC image variant {image_variant!r}; choose one of: {choices}")
    return resolved
