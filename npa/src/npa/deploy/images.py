"""Shared Workbench container image naming."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
import os
from pathlib import Path
import re
from typing import Any

# Official NPA images use one public GHCR namespace. Immutable
# ``dev-<full-git-sha>`` tags and supported release tags share each image package;
# guarded promotion applies the release tag only to an already validated dev digest.
# ``NPA_REGISTRY`` remains the generic operator execution override. Restricted and
# build-your-own images must use an operator-controlled registry and are refused from
# official GHCR.
PUBLIC_CONTAINER_REGISTRY_ENV = "NPA_PUBLIC_REGISTRY"
DEFAULT_PUBLIC_CONTAINER_REGISTRY = "ghcr.io/nebius/nebius-physical-ai"

# Compatibility name for callers that mean "the normal execution registry". The
# default is the public release channel; it no longer points at Nebius Container
# Registry and carries no registry ID or regional failover behavior.
DEFAULT_CONTAINER_REGISTRY = DEFAULT_PUBLIC_CONTAINER_REGISTRY
DEFAULT_VLM_IMAGE_ENV = "NPA_VLM_IMAGE"
DEFAULT_WORKBENCH_IMAGE_ENV = "NPA_WORKBENCH_IMAGE"
SONIC_IMAGE_MANIFEST_RESOURCE = "sonic_image_manifest.json"
WAN_IMAGE_MANIFEST_RESOURCE = "wan2_2_image_manifest.json"
LTX2_IMAGE_MANIFEST_RESOURCE = "ltx2_image_manifest.json"
CONTENT_AGENTS_IMAGE_MANIFEST_RESOURCE = "content_agents_image_manifest.json"
PUBLIC_RELEASE_MANIFEST_RESOURCE = "public_release_manifest.json"

CONTAINER_IMAGE_NAMES = {
    "lerobot": "npa-lerobot",
    "lerobot-policy": "npa-lerobot-policy",
    "genesis": "npa-genesis",
    "isaac-lab": "npa-isaac-lab",
    "leisaac": "npa-leisaac",
    "cosmos": "npa-cosmos",
    "cosmos2-transfer": "npa-cosmos2-transfer",
    "cosmos3": "npa-cosmos3",
    "cosmos3-ray-serve": "npa-cosmos3-ray-serve",
    "cosmos3-serving": "npa-cosmos3-serving",
    "cosmos3-reason": "npa-cosmos3-reason",
    "cosmos-curate": "npa-cosmos-curate",
    "cosmos-evaluator": "npa-cosmos-evaluator",
    "groot": "npa-groot",
    "fiftyone": "npa-fiftyone",
    "sonic": "npa-sonic",
    "sonic-mujoco": "npa-sonic-mujoco",
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
    "ltx2": "npa-ltx2",
    "alpamayo2-super": "npa-alpamayo2-super",
    "content-agents": "npa-content-agents",
}

# Public-image publication must enforce the digest-bound SkyPilot bootstrap
# attestation only for images that declare that build contract in
# docker/workbench/packaging-contract.yaml.  Keep this packaged copy explicit:
# an installed npa wheel does not carry the repository's Docker packaging tree.
# npa/tests/docker/test_packaging_contract.py locks the two inventories together.
SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS: frozenset[str] = frozenset(
    {
        "cosmos2-transfer",
        "cosmos3",
        "cosmos-curate",
        "cosmos-evaluator",
        "content-agents",
        "fiftyone",
        "rerun-viewer",
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
        CONTAINER_IMAGE_NAMES[tool] for tool in SKYPILOT_BOOTSTRAP_RUNTIME_PROBED_TOOLS
    }


# General public-registry refusal inventories. They intentionally describe the
# redistribution decision, not a particular vendor payload. Both are empty now:
# Cosmos3 serving is a zero-payload runtime bootstrap on a public Python base,
# and sonic-mujoco is rebuilt independently without its quarantined parent.
RESTRICTED_PUBLICATION_TOOLS: frozenset[str] = frozenset()
RESTRICTED_DERIVED_IMAGES: frozenset[str] = frozenset()

# Compatibility exports for installed callers. New code uses the general names.
OMNIVERSE_RESTRICTED_TOOLS = RESTRICTED_PUBLICATION_TOOLS
OMNIVERSE_RESTRICTED_DERIVED_IMAGES = RESTRICTED_DERIVED_IMAGES

# Tools that are licence-eligible for public redistribution but have no accepted
# built/GPU-validated artifact yet.
#
# This is a different question from `RESTRICTED_PUBLICATION_TOOLS`, and conflating
# them would be wrong in both directions: these are not restricted (the licensing
# work is done and the answer was "public"), they are simply unproven. Publishing
# an image whose payload scan and GPU smoke have never run would hand out a claim
# we have not earned, so publish_public refuses them by name rather than relying
# on the push failing because the tag happens not to exist.
#
# Remove a tool from this set in the same change that records its accepted image
# digest and its payload-scan/GPU evidence — not before.
UNVALIDATED_PUBLICATION_TOOLS: frozenset[str] = frozenset()
VALIDATION_CANDIDATE_TOOLS: frozenset[str] = frozenset()
# Compatibility view used by publication callers and public imports. Derive it
# from the two canonical validation-state inventories; never maintain it
# independently.
PUBLICATION_QUARANTINE_TOOLS: frozenset[str] = (
    UNVALIDATED_PUBLICATION_TOOLS | VALIDATION_CANDIDATE_TOOLS
)

# Some newer operator/BYOF pins have not yet been promoted to the supported
# anonymous channel. Public execution stays on the last accepted release while
# an explicit custom registry resolves the newer supported-tool pin.
PUBLIC_RELEASE_TAG_OVERRIDES: dict[str, str] = {
    "cosmos2-transfer": "2.5.1-skypilot-ready-20260801T053000Z",
    "fiftyone": "1.15.0.post1",
    "rerun-viewer": "0.31.4",
}

# Release promotion for the rebuilt surfaces is bound to the exact manifests
# whose filesystem/layers were scanned and whose advertised GPU capability ran.
# A newly built dev tag must earn fresh evidence before this mapping changes.
GPU_ACCEPTED_PUBLIC_IMAGE_SOURCES: dict[str, dict[str, str]] = {
    "cosmos3-ray-serve": {
        "development_sha": "56d8c4f3f05db7aa3b03323441a3e0d7b97ac8da",
        "oci_digest": "sha256:6e42f553a0d14712dc1ed7fa42c72b0f083f4ae3f89b30eaf0e93cfdf64e820d",
    },
    "cosmos3-serving": {
        "development_sha": "d854f6a76cd87ec05ad97ccde6d596f3329efa0e",
        "oci_digest": "sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180",
    },
    "sonic-mujoco": {
        "development_sha": "5b5b5e69e9e686f8d5f305fd735a02f402f6da4b",
        "oci_digest": "sha256:2388d9e97269afaa414966e83a27f676a3f44d4271e9828c57bc13fbdce80f57",
    },
}
GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS: dict[str, str] = {
    tool: source["oci_digest"]
    for tool, source in GPU_ACCEPTED_PUBLIC_IMAGE_SOURCES.items()
}

# Registry hosts that serve anonymous/public pulls. Resolving a restricted image
# against one of these is always wrong: either it is not there (we never publish
# it) or someone has published a non-redistributable runtime to third parties.
# Private registries are deliberately absent — an operator building the image
# into their OWN registry is the licensed path, whichever registry that is.
PUBLIC_REGISTRY_HOSTS = frozenset(
    {
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
    "leisaac": "0.4.0-20260817T231825Z",
    "cosmos": "cu128-torch27-sm100-1.0.9-20260803T002017Z",
    "cosmos2-transfer": "2.5.1-sam2-multigpu-20260817-r2",
    # Additive r2 release of cosmos-framework 1.2.2 (pinned commit 5e67049c) +
    # torch cu130. The immutable predecessor remains rollback provenance.
    # No weights baked; gated Cosmos3 checkpoints download at runtime.
    "cosmos3": "1.2.2-cu130-r6",
    "cosmos3-ray-serve": "ray1-cu130",
    "cosmos3-serving": "0.2.0-oss",
    "cosmos3-reason": "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "cosmos-curate": "0.1.2-skypilot-v1-20260813T164700Z",
    "cosmos-evaluator": "0.1.2-skypilot-v1-20260813T164700Z-r2",
    "groot": "0.1.0",
    "fiftyone": "1.15.0-post1-skypilot-v1-20260815-review5",
    "sonic": "cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "sonic-mujoco": "0.2.0-runtime",
    "retargeting": "0.1.1",
    "envgen": "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "reference-policy": "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "lerobot-vlm-rl": "cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "loop-eval": "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "rerun-viewer": "0.31.4-skypilot-v1-20260815-review5-r2",
    # Tracks the pinned @foxglove/embed SDK release (npa.workbench.foxglove).
    "foxglove-embed": "0.58.0",
    # Lichtblick (MPL-2.0): OSS, Foxglove-compatible static web viewer bundle.
    "lichtblick": "1.26.0",
    "lancedb": "cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z",
    "detection-training": "bdd100k-golden-eval-smoke-20260614T210000Z",
    # Public-eligible Wan source/CPU base; CUDA torch is operator-gated runtime fetch.
    "wan2-2": "2.2-ti2v5b-rtfetch-cu130-20260817",
    # LTX source and weights remain operator-entitled runtime fetches. This tag
    # resolves only to the zero-payload digest recorded in ltx2_image_manifest.json.
    "ltx2": "2.5-rtfetch-20260817",
    "alpamayo2-super": "0.1.0-cu128",
    "content-agents": "0.5.2-npa2",
    "nebius-cli": "0.12.254",
    "terraform": "~> 0.5.201",
    "terraform-cli": "1.13.3",
}


@lru_cache(maxsize=1)
def sonic_image_manifest() -> dict[str, Any]:
    """Return the packaged SONIC image compatibility manifest."""

    text = (
        resources.files(__package__)
        .joinpath(SONIC_IMAGE_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
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


@lru_cache(maxsize=1)
def ltx2_accepted_image_manifest() -> dict[str, Any]:
    """Return the exact zero-payload image and GPU proof allowed for publication."""

    text = (
        resources.files(__package__)
        .joinpath(LTX2_IMAGE_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError("LTX accepted image manifest must be a JSON object")
    if payload.get("format") != "npa_ltx2_accepted_image_manifest_v1":
        raise RuntimeError("Unsupported LTX accepted image manifest format")
    if payload.get("tag") != SUPPORTED_TOOL_VERSIONS["ltx2"]:
        raise RuntimeError(
            "LTX accepted image manifest tag drifted from the supported tag"
        )
    return payload


@lru_cache(maxsize=1)
def content_agents_accepted_image_manifest() -> dict[str, Any]:
    """Return the immutable Content Agents image/runtime/RTX proof tuple."""

    payload = json.loads(
        resources.files(__package__)
        .joinpath(CONTENT_AGENTS_IMAGE_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Content Agents accepted image manifest must be a JSON object")
    if payload.get("format") != "npa_content_agents_accepted_image_manifest_v1":
        raise RuntimeError("Unsupported Content Agents accepted image manifest format")
    if payload.get("tag") != SUPPORTED_TOOL_VERSIONS["content-agents"]:
        raise RuntimeError(
            "Content Agents accepted image manifest tag drifted from the supported tag"
        )
    return payload


@lru_cache(maxsize=1)
def public_release_manifest() -> dict[str, Any]:
    """Load exact anonymously verified release-digest claims."""

    payload = json.loads(
        resources.files("npa.deploy")
        .joinpath(PUBLIC_RELEASE_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Public release manifest must be a JSON object")
    if payload.get("format") != "npa_public_release_manifest_v1":
        raise RuntimeError("Unsupported public release manifest format")
    if payload.get("registry") != DEFAULT_PUBLIC_CONTAINER_REGISTRY:
        raise RuntimeError("Public release manifest registry drifted from official GHCR")
    releases = payload.get("releases")
    pending = payload.get("publication_pending")
    if not isinstance(releases, dict) or not isinstance(pending, dict):
        raise RuntimeError("Public release manifest inventories must be objects")
    if set(releases) | set(pending) != set(publicly_publishable_tools()):
        raise RuntimeError(
            "Public release manifest must partition every publishable tool into "
            "published or publication-pending"
        )
    for tool, entry in releases.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Public release manifest entry {tool!r} must be an object")
        if entry.get("tag") != public_release_tag_for_tool(tool):
            raise RuntimeError(f"Public release tag drifted for {tool!r}")
        if re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(entry.get("published_digest") or "")
        ) is None:
            raise RuntimeError(f"Public release digest is invalid for {tool!r}")
        development_sha = entry.get("development_sha")
        if development_sha is not None:
            development_tag(str(development_sha))
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
        raise RuntimeError(
            f"Could not find supported version for tool: {tool}"
        ) from exc


def public_release_tag_for_tool(tool: str) -> str:
    """Return the exact repository pin that the public release channel must carry.

    SONIC's runtime resolver accepts only the active host-mounted Kubernetes
    variant. The public inventory contract pins that validated cross-architecture
    runtime from ``SUPPORTED_TOOL_VERSIONS`` rather than either quarantined tag.
    """
    if tool == "sonic":
        return SUPPORTED_TOOL_VERSIONS[tool]
    return PUBLIC_RELEASE_TAG_OVERRIDES.get(tool, supported_tool_version(tool))


def supported_lerobot_versions() -> tuple[str, ...]:
    """Return LeRobot versions supported by the workbench (default first)."""

    from npa.workbench.lerobot.version_compat import (
        supported_lerobot_versions as _versions,
    )

    return _versions()


def resolve_lerobot_image_tag(version: str | None = None) -> str:
    """Resolve the validated image tag for a supported LeRobot package version."""

    from npa.workbench.lerobot.version_compat import lerobot_version_entry

    entry = lerobot_version_entry(version)
    return str(entry.get("image_tag") or entry["version"])


def sonic_image_variant_for_gpu(gpu_target: str | None = None) -> str:
    """Return an active SONIC variant or reject unsupported GPU/runtime pairs."""

    manifest = sonic_image_manifest()
    default = str(manifest.get("default_variant", "sonic-k8s-host-mounted"))
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
    raise ValueError(
        f"Unsupported SONIC GPU target {gpu_target!r}. The only published active "
        "variant is sonic-k8s-host-mounted on RTX PRO 6000 Blackwell Kubernetes "
        "nodes with NVIDIA GPU Operator driver mounts. L40S/H100/H200 compute-only "
        "variants are retired and quarantined; supply a separately validated custom "
        "image explicitly or choose gpu-rtx6000 on Kubernetes."
    )


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
        entry = variants[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown SONIC image variant {resolved!r}; choose one of: {choices}"
        ) from exc
    if str(entry.get("status") or "active") != "active":
        status = str(entry.get("status") or "unknown")
        reason = str(entry.get("quarantine_reason") or "image is not accepted")
        raise ValueError(
            f"SONIC image variant {resolved!r} has status {status!r} and cannot be resolved: "
            f"{reason} Use sonic-k8s-host-mounted or build a newly scanned, "
            "license-compatible replacement."
        )
    return entry


def container_image_for_tool(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> str:
    """Return the fully qualified image ref for a Workbench tool."""
    resolved_registry = registry or execution_container_registry()
    if tool == "sonic":
        entry = sonic_image_entry(gpu_target=gpu_target, image_variant=image_variant)
        image_name = str(entry["name"])
        resolved_tag = tag or str(entry["tag"])
    else:
        if image_variant:
            raise ValueError(
                f"Image variants are only defined for SONIC, got tool={tool!r}"
            )
        image_name = CONTAINER_IMAGE_NAMES[tool]
        resolved_tag = tag or (
            public_release_tag_for_tool(tool)
            if is_public_registry(resolved_registry)
            else supported_tool_version(tool)
        )
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


def registry_from_env() -> str:
    """Return the generic operator execution-registry override, if set."""
    return os.environ.get("NPA_REGISTRY", "").strip()


def execution_container_registry() -> str:
    """Resolve an operator override, otherwise the public GHCR release channel."""
    return registry_from_env() or DEFAULT_CONTAINER_REGISTRY


def container_image_candidates(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
    preferred_region: str | None = None,
) -> list[str]:
    """Return the single selected image reference.

    The historical regional mirror/failover behavior was specific to Nebius
    Container Registry and is intentionally gone. ``preferred_region`` remains an
    ignored compatibility argument so older SDK callers do not break.
    """
    del preferred_region
    return [
        container_image_for_tool(
            tool,
            registry=registry,
            tag=tag,
            gpu_target=gpu_target,
            image_variant=image_variant,
        )
    ]


def public_container_registry() -> str:
    """Return the official public GHCR release namespace."""
    value = (
        os.environ.get(PUBLIC_CONTAINER_REGISTRY_ENV, "").strip()
        or DEFAULT_PUBLIC_CONTAINER_REGISTRY
    )
    return _ghcr_namespace(value, channel="public release")


def _ghcr_namespace(value: str, *, channel: str) -> str:
    """Validate an official channel override as ``ghcr.io/<owner>/<namespace>``."""
    normalized = str(value or "").strip().rstrip("/")
    parts = normalized.split("/")
    if len(parts) < 3 or parts[0].lower() != "ghcr.io" or not all(parts[1:]):
        raise ValueError(
            f"{channel} registry must be a GHCR package namespace such as "
            "ghcr.io/<owner>/<namespace>"
        )
    return normalized


def development_tag(git_sha: str) -> str:
    """Return the immutable public development tag for a full Git commit SHA."""
    normalized = str(git_sha or "").strip().lower()
    if len(normalized) != 40 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError("development source SHA must be a full 40-character Git SHA")
    return f"dev-{normalized}"


def development_image_for_tool(
    tool: str,
    *,
    git_sha: str,
    registry: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> str:
    """Return an official public development reference for redistributable bytes."""
    if not is_publicly_redistributable(tool):
        raise ValueError(
            f"{tool!r} is restricted/build-your-own and cannot be pushed to "
            "official GHCR; use an operator-controlled registry"
        )
    resolved_registry = _ghcr_namespace(
        registry or public_container_registry(), channel="public development"
    )
    return container_image_for_tool(
        tool,
        registry=resolved_registry,
        tag=development_tag(git_sha),
        gpu_target=gpu_target,
        image_variant=image_variant,
    )


def is_public_registry(registry: str) -> bool:
    """Whether a registry serves anonymous/public pulls.

    True for conservative public-only hosts, the immutable default official
    namespace, and the configured public release namespace. GHCR is package-
    scoped, so an arbitrary operator GHCR namespace is not assumed public.
    """
    candidate = registry.strip().rstrip("/")
    if not candidate:
        return False
    host = candidate.split("/", 1)[0].lower()
    if host in PUBLIC_REGISTRY_HOSTS:
        return True
    if candidate.lower() == DEFAULT_PUBLIC_CONTAINER_REGISTRY.lower():
        return True
    mirror = public_container_registry().strip().rstrip("/")
    return bool(mirror) and candidate.lower() == mirror.lower()


def is_official_container_registry(registry: str) -> bool:
    """Whether ``registry`` is the official NPA public GHCR namespace."""
    candidate = str(registry or "").strip().rstrip("/").lower()
    return candidate in {
        DEFAULT_PUBLIC_CONTAINER_REGISTRY.lower(),
        public_container_registry().rstrip("/").lower(),
    }


def is_publicly_redistributable(tool: str) -> bool:
    """Whether a tool image may be published to a public/anonymous registry.

    ``False`` for any tool in ``RESTRICTED_PUBLICATION_TOOLS`` — images that bake a
    runtime we may not redistribute, which are licensed for internal-R&D /
    build-your-own use only. See the set's comment for current membership.
    """
    return tool not in RESTRICTED_PUBLICATION_TOOLS


def restricted_image_names() -> list[str]:
    """Return every image name excluded from public registries."""
    return sorted(RESTRICTED_PUBLICATION_TOOLS | RESTRICTED_DERIVED_IMAGES)


def omniverse_restricted_image_names() -> list[str]:
    """Compatibility alias for :func:`restricted_image_names`."""
    return restricted_image_names()


def publicly_publishable_tools() -> list[str]:
    """Return tools accepted for the supported anonymous release inventory.

    Redistribution eligibility is necessary but not sufficient: tools remain out
    while ``PUBLICATION_QUARANTINE_TOOLS`` records that their built-image or GPU
    evidence is incomplete. The trusted build workflow can still create their
    immutable development artifact directly from the public packaging contract.
    """
    return sorted(
        tool
        for tool in CONTAINER_IMAGE_NAMES
        if is_publicly_redistributable(tool)
        and tool not in PUBLICATION_QUARANTINE_TOOLS
    )


def accepted_publication_development_sha(tool: str) -> str | None:
    """Return a tool's exact accepted development SHA when one is recorded."""

    entry = (public_release_manifest().get("releases") or {}).get(tool) or {}
    value = entry.get("development_sha")
    if value is None:
        return None
    normalized = development_tag(str(value)).removeprefix("dev-")
    if tool == "wan2-2" and normalized != wan_accepted_image_manifest().get(
        "development_sha"
    ):
        raise RuntimeError("Wan release and accepted-image development SHAs disagree")
    if tool == "ltx2" and normalized != ltx2_accepted_image_manifest().get(
        "development_sha"
    ):
        raise RuntimeError("LTX release and accepted-image development SHAs disagree")
    gpu_source = GPU_ACCEPTED_PUBLIC_IMAGE_SOURCES.get(tool)
    if gpu_source and normalized != gpu_source.get("development_sha"):
        raise RuntimeError(f"{tool} release and GPU-accepted development SHAs disagree")
    return normalized


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
    return sonic_image_entry(
        image_variant=str(sonic_image_manifest().get("default_variant", ""))
    )


def _normalize_gpu_target(gpu_target: str | None) -> str:
    return (gpu_target or "").strip().lower().replace("_", "-")


def _normalize_sonic_variant(
    image_variant: str, variants: dict[str, dict[str, Any]]
) -> str:
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
        "mujoco": "sonic-mujoco-runtime-fetch",
        "b200": "sonic-mujoco-runtime-fetch",
        "sonic-mujoco": "sonic-mujoco-runtime-fetch",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown SONIC image variant {image_variant!r}; choose one of: {choices}"
        )
    return resolved
