"""Shared Workbench container image naming."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib import resources
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import shlex
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from npa.workbench.gpu_classes import DATACENTER_HEADLESS, classify_gpu_target

# Official NPA images use one public GHCR namespace. Immutable
# ``dev-<full-git-sha>`` tags and supported release tags share each image package;
# guarded promotion applies the release tag only to an already validated dev digest.
# ``NPA_REGISTRY`` remains the generic operator build/BYOF registry. Repository-owned
# runtime defaults never consult it: a stale ambient or saved private registry must not
# redirect supported public releases away from GHCR. Callers that intentionally select
# custom bytes pass ``registry=`` (or a complete image reference) explicitly.
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
NCORE_IMAGE_MANIFEST_RESOURCE = "ncore_image_manifest.json"
LIBERO_IMAGE_MANIFEST_RESOURCE = "libero_image_manifest.json"
PUBLIC_RELEASE_MANIFEST_RESOURCE = "public_release_manifest.json"

LIBERO_CUSTOMER_AUTHORIZATION_SCHEMA = (
    "npa.libero.customer-runtime-authorization.v1"
)
LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_FILE_ENV = (
    "NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_FILE"
)
LIBERO_OFFICIAL_CANDIDATE_PREFIX = (
    "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:"
)
LIBERO_UPSTREAM_SOURCE_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_BUILD_INPUT_PATHS = (
    "npa/docker/workbench/libero/Dockerfile",
    "npa/docker/workbench/libero/REDISTRIBUTION.md",
    "npa/docker/workbench/libero/THIRD_PARTY_NOTICES.md",
    "npa/docker/workbench/libero/debian-packages.lock",
    "npa/docker/workbench/libero/entrypoint.sh",
    "npa/docker/workbench/libero/libero_smoke.py",
    "npa/docker/workbench/libero/runtime-bootstrap.py",
    "npa/docker/workbench/libero/runtime-manifest.json",
    "npa/docker/workbench/libero/runtime-requirements.txt",
    "npa/docker/workbench/libero/skypilot-bootstrap-guard.sh",
    "npa/docker/workbench/libero/smoke.sh",
)
LIBERO_PUBLICATION_ENFORCEMENT_PATHS = (
    ".github/scripts/publish_selected_public_image.py",
    ".github/workflows/publish-public-images.yml",
    "workflows/testing/byof-libero.yaml",
    "workflows/testing/byof-libero.readiness.json",
    *LIBERO_BUILD_INPUT_PATHS,
    "npa/docker/workbench/blackwell-dc-images.json",
    "npa/docker/workbench/libero/build.sh",
    "npa/docker/workbench/packaging-contract.yaml",
    "npa/scripts/run_byof_container_verify.py",
    "npa/scripts/run_byof_repo.py",
    "npa/scripts/scan_image_libero_payload.py",
    "npa/scripts/scan_image_wan_payload.py",
    "npa/src/npa/cli/workbench/byof.py",
    "npa/src/npa/deploy/images.py",
    "npa/src/npa/deploy/libero_image_manifest.json",
    "npa/src/npa/deploy/public_release_manifest.json",
    "npa/src/npa/errors.py",
    "npa/src/npa/execution_preflight.py",
    "npa/src/npa/lifecycle_intent.py",
    "npa/src/npa/smoke/golden_evals.yaml",
    "npa/src/npa/workbench/gpu_classes.py",
    "npa/src/npa/workflows/byof/profiles/byof-solution-smoke-libero-b200-gpu.yaml",
    "npa/tests/deploy/test_public_publish.py",
    "npa/tests/docker/test_image_byte_publish_contract.py",
    "npa/tests/docker/test_libero_image_contract.py",
    "npa/tests/docker/test_libero_image_payload_scan.py",
    "npa/tests/docker/test_libero_runtime_bootstrap.py",
    "npa/tests/e2e/test_byof_onboarding_live_e2e.py",
    "npa/tests/guardrails/test_byof_profiles.py",
    "npa/tests/guardrails/test_e2e_gate_reachability.py",
    "npa/tests/unit/test_execution_preflight.py",
    "npa/tests/workflows/test_byof_container_verify.py",
    "npa/tests/workflows/test_byof_libero.py",
    "npa/tests/workflows/test_byof_repo.py",
    "npa/tests/workflows/test_byof_solution_smokes.py",
)
LIBERO_PUBLICATION_ENFORCEMENT_PYTHON_ROOTS = (
    "npa/scripts",
    "npa/src/npa",
    "npa/tests/e2e",
)
LIBERO_PUBLICATION_ENFORCEMENT_LIBERO_TEST_ROOTS = (
    "npa/tests",
)
LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER = (
    b"# npa: publication-enforcement=libero"
)
LIBERO_PUBLICATION_ENFORCEMENT_CRITICAL_TEST_REFERENCES = (
    b"_cleanup_libero_access_objects",
    b"_libero_external_rbac_inventory_sha256",
    b"libero_publication_enforcement_bundle_sha256",
    b"validate_libero_qualified_image_manifest",
    b"NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
    b"NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
)
LIBERO_REQUIRED_PUBLICATION_REFERRERS = (
    "https://slsa.dev/provenance/v1",
    "https://spdx.dev/Document",
)

CONTAINER_IMAGE_NAMES = {
    "openpi": "npa-openpi",
    "lerobot": "npa-lerobot",
    "sim2real-control": "npa-sim2real-control",
    "lerobot-policy": "npa-lerobot-policy",
    "genesis": "npa-genesis",
    "isaac-lab": "npa-isaac-lab",
    "isaac-arena": "npa-isaac-arena",
    "openarm": "npa-openarm",
    "leisaac": "npa-leisaac",
    "cosmos": "npa-cosmos",
    "cosmos2-transfer": "npa-cosmos2-transfer",
    "cosmos3": "npa-cosmos3",
    "cosmos3-ray-serve": "npa-cosmos3-ray-serve",
    "cosmos3-serving": "npa-cosmos3-serving",
    "cosmos3-super-benchmark": "npa-cosmos3-super-benchmark",
    "cosmos3-nano-video": "npa-cosmos3-nano-video",
    "cosmos3-reason": "npa-cosmos3-reason",
    "cosmos-curate": "npa-cosmos-curate",
    "cosmos-evaluator": "npa-cosmos-evaluator",
    "groot": "npa-groot",
    "fiftyone": "npa-fiftyone",
    "sonic": "npa-sonic",
    "sonic-mujoco": "npa-sonic-mujoco",
    "retargeting": "npa-retargeting",
    "robocasa": "npa-robocasa",
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
    "diffusers": "npa-diffusers",
    "lingbot-world": "npa-lingbot-world",
    "sam2": "npa-sam2",
    "ltx2": "npa-ltx2",
    "alpamayo2-super": "npa-alpamayo2-super",
    "curobo": "npa-curobo",
    "content-agents": "npa-content-agents",
    "ncore": "npa-ncore",
    "libero": "npa-libero",
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
        "cosmos3-reason",
        "cosmos3-super-benchmark",
        "cosmos-curate",
        "cosmos-evaluator",
        "content-agents",
        "ncore",
        "libero",
        "fiftyone",
        "groot",
        "isaac-lab",
        "isaac-arena",
        "openarm",
        "rerun-viewer",
        "sim2real-control",
        "envgen",
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
# redistribution decision, not a particular vendor payload. The Cosmos3-Super
# benchmark wrapper inherits the exact upstream vLLM-Omni runtime and therefore
# remains build-your-own in an operator-controlled registry.
RESTRICTED_PUBLICATION_TOOLS: frozenset[str] = frozenset(
    {"cosmos3-super-benchmark", "cosmos3-nano-video"}
)
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
UNVALIDATED_PUBLICATION_TOOLS: frozenset[str] = frozenset(
    {"openpi", "curobo", "ncore", "libero"}
)
VALIDATION_CANDIDATE_TOOLS: frozenset[str] = frozenset({"robocasa"})
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
    # 0.31.4 (plain) predates the bootstrap contract and cannot host a SkyPilot
    # task: the container exits immediately, the provisioner's exec finds no
    # ray-node container, and the stage retries forever. The 20260903 build is
    # attested (org.nebius.npa.skypilot-bootstrap-contract=skypilot-0.12.2-v1)
    # and anonymously pullable from GHCR.
    "rerun-viewer": "0.31.4-sim2real-coherent-20260904",
}

# Release promotion for the rebuilt surfaces is bound to the exact manifests
# whose filesystem/layers were scanned and whose advertised GPU capability ran.
# A newly built dev tag must earn fresh evidence before this mapping changes.
GPU_ACCEPTED_PUBLIC_IMAGE_SOURCES: dict[str, dict[str, str]] = {
    "isaac-arena": {
        "development_sha": "ae5adea6ab895660996f513f14160c89d06f47e5",
        "oci_digest": "sha256:9c6a417672d6f87499680ba337c90488c2a33d41ac9f7b5452eb5d97d00e097e",
    },
    "alpamayo2-super": {
        "development_sha": "5b693476c113c833e9d9d4f8c7aa492492a27505",
        "oci_digest": "sha256:17a3966a6e743cf34ecaeb2ef684272646c815d07a8a4668ccebf17de6aa0e07",
    },
    "diffusers": {
        "development_sha": "d54eec137d3b2d86ff1acef736e36967b1fad7d3",
        "oci_digest": "sha256:6422a062a00c9816a945623b0c83a78977fa5d5cec1777a0e6d44d1d746fc42e",
    },
    "lingbot-world": {
        "development_sha": "d54eec137d3b2d86ff1acef736e36967b1fad7d3",
        "oci_digest": "sha256:5e2a3998bf7d54987d0916f7c249b4963ef87f489e9e893ec8d94da196286277",
    },
    "sam2": {
        "development_sha": "d54eec137d3b2d86ff1acef736e36967b1fad7d3",
        "oci_digest": "sha256:fbe20454e97452e447e00f79260a267b552deef5538e3bbbc8c3567a3c576f16",
    },
    "cosmos3": {
        "development_sha": "1925834f29983dd9a16659eb3dd350a7f5d13d99",
        "oci_digest": "sha256:d8e1fe370f75e5433455a221b70ae6211c30369255a3bb111d03e5c07240e010",
    },
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
    "detection-training": {
        "development_sha": "408700158b2e9cc9e9f6aad499e9d9c810bebeb1",
        "oci_digest": "sha256:a09126491bd660f314b8f412df7238746dc2b063e5d5b7ca87bba7596dafcb0d",
    },
    "openarm": {
        "development_sha": "01fbf3a554cb7b15066283fd171c5b81f6207eda",
        "oci_digest": "sha256:c30da0d55de0b1b0528b1481a318bf43ad9d95c7128ae44b5d434203e7d1543a",
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
    "openpi": "pi05-full-droid-rlds-cu128-unbuilt",
    # Default LeRobot image release. Selectable package versions and their
    # image tags live in lerobot_version_manifest.json.
    "lerobot": "cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "sim2real-control": "0.1.2-sim2real-coherent-20260904",
    "lerobot-policy": "0.1.1",
    "genesis": "cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "isaac-lab": "3.0.0b2.post1-sim2real-coherent-20260904",
    "isaac-arena": "0.3.0-isaaclab3-20260917-r4",
    "openarm": "2.2.0-isaac0.1.0-rtfetch",
    "leisaac": "0.4.0-20260817T231825Z",
    "cosmos": "cu128-torch27-sm100-1.0.9-20260803T002017Z",
    "cosmos2-transfer": "2.5.1-sim2real-coherent-20260904",
    # Additive r7 release of cosmos-framework 1.2.2 (pinned commit 5e67049c) +
    # torch cu130. The immutable predecessor remains rollback provenance.
    # No weights baked; gated Cosmos3 checkpoints download at runtime.
    "cosmos3": "1.2.2-cu130-r7",
    "cosmos3-ray-serve": "ray1-cu130",
    "cosmos3-serving": "0.2.0-oss",
    "cosmos3-super-benchmark": "0.1.0",
    "cosmos3-nano-video": "0.1.0",
    "cosmos3-reason": "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "cosmos-curate": "0.1.2-skypilot-v1-20260813T164700Z",
    "cosmos-evaluator": "0.1.2-skypilot-v1-20260813T164700Z-r2",
    "groot": "0.1.0",
    "fiftyone": "1.21.0-skypilot-v1-20260915",
    "sonic": "cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "sonic-mujoco": "0.2.0-runtime",
    "retargeting": "0.1.1",
    "envgen": "0.1.2-sim2real-coherent-20260904",
    "robocasa": "0.1.0",
    "reference-policy": "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "lerobot-vlm-rl": "cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "loop-eval": "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
    "rerun-viewer": "0.31.4-sim2real-coherent-20260904",
    # Tracks the pinned @foxglove/embed SDK release (npa.workbench.foxglove).
    "foxglove-embed": "0.58.0",
    # Lichtblick (MPL-2.0): OSS, Foxglove-compatible static web viewer bundle.
    "lichtblick": "1.26.0",
    "lancedb": "cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z",
    "detection-training": "runtime-v1-20260905",
    # Public-eligible Wan source/CPU base; CUDA torch is operator-gated runtime fetch.
    "wan2-2": "2.2-ti2v5b-rtfetch-cu130-20260817",
    "diffusers": "0.38.0-rtfetch-20260916",
    "lingbot-world": "a43bec7-rtfetch-20260916",
    "sam2": "2.1-rtfetch-20260916",
    # LTX source and weights remain operator-entitled runtime fetches. This tag
    # resolves only to the zero-payload digest recorded in ltx2_image_manifest.json.
    "ltx2": "2.5-rtfetch-20260817",
    "alpamayo2-super": "0.1.0-cu128-r3",
    "curobo": "0.8.0-cuda13-b300-unbuilt",
    "content-agents": "0.5.2-npa2",
    # Source packaging inventory only; no accepted public NCore release exists.
    "ncore": "59c698d206da92b406a4f72619fce3b3a2c64bfd-unbuilt",
    "libero": "public-neutral-bootstrap-unbuilt",
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
def libero_image_manifest() -> dict[str, Any]:
    """Return the checked-in LIBERO publication and qualification boundary."""

    try:
        payload = json.loads(
            resources.files(__package__)
            .joinpath(LIBERO_IMAGE_MANIFEST_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("LIBERO image manifest is unavailable or invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "npa.workbench.image-manifest.v1"
        or payload.get("tool") != "libero"
        or payload.get("image_name") != "npa-libero"
    ):
        raise RuntimeError("Unsupported LIBERO image manifest format")
    return payload


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ssh_signature_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def libero_customer_authorization_signature_payload(
    payload: dict[str, Any],
) -> bytes:
    """Return SSHSIG-framed bytes for a customer authorization envelope."""

    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return b"".join(
        (
            b"SSHSIG",
            _ssh_signature_string(b"npa.libero.customer-authorization"),
            _ssh_signature_string(b""),
            _ssh_signature_string(b"sha512"),
            _ssh_signature_string(hashlib.sha512(canonical).digest()),
        )
    )


def _verify_libero_customer_authorization_signature(
    payload: dict[str, Any], *, public_key_file: str = ""
) -> None:
    signature_record = payload.get("signature")
    if not isinstance(signature_record, dict) or set(signature_record) != {
        "algorithm",
        "public_key_sha256",
        "signature_b64",
    }:
        raise RuntimeError(
            "LIBERO customer authorization requires a closed control-plane signature"
        )
    if signature_record.get("algorithm") != "ed25519":
        raise RuntimeError(
            "LIBERO customer authorization requires an Ed25519 signature"
        )
    key_path_value = public_key_file.strip() or os.environ.get(
        LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_FILE_ENV, ""
    ).strip()
    try:
        key_descriptor = os.open(
            key_path_value,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise RuntimeError(
            "LIBERO customer-authorization trust-root file is unavailable"
        ) from exc
    try:
        before = os.fstat(key_descriptor)
        with os.fdopen(key_descriptor, "rb", closefd=False) as key_stream:
            encoded_key = key_stream.read()
        after = os.fstat(key_descriptor)
    finally:
        os.close(key_descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or encoded_key != encoded_key.strip()
    ):
        raise RuntimeError(
            "LIBERO customer-authorization trust-root file is mutable or invalid"
        )
    try:
        public_key = base64.b64decode(encoded_key, validate=True)
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise RuntimeError(
            "LIBERO customer-authorization signature material is invalid"
        ) from exc
    if (
        len(public_key) != 32
        or len(signature) != 64
        or hashlib.sha256(public_key).hexdigest()
        != signature_record.get("public_key_sha256")
    ):
        raise RuntimeError("LIBERO customer-authorization trust root differs")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, libero_customer_authorization_signature_payload(payload)
        )
    except (ValueError, InvalidSignature) as exc:
        raise RuntimeError("LIBERO customer authorization signature is invalid") from exc


def _repository_file_bundle_sha256(
    repository_root: Path,
    *,
    schema: str,
    paths: tuple[str, ...],
    development_sha: str | None = None,
) -> str:
    records = []
    for relative in paths:
        path = repository_root / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"LIBERO build input is unavailable: {relative}") from exc
        records.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    bundle: dict[str, Any] = {"schema": schema, "inputs": records}
    if development_sha is not None:
        bundle["development_sha"] = development_sha
    return _canonical_sha256(bundle)


def libero_build_input_bundle_sha256(
    repository_root: Path, *, development_sha: str
) -> str:
    """Bind every byte copied into the neutral image plus its reproducibility epoch."""

    if re.fullmatch(r"[0-9a-f]{40}", development_sha) is None:
        raise RuntimeError("LIBERO build input bundle requires a full development SHA")
    return _repository_file_bundle_sha256(
        repository_root,
        schema="npa.libero.build-input-bundle.v1",
        paths=LIBERO_BUILD_INPUT_PATHS,
        development_sha=development_sha,
    )


def libero_publication_enforcement_bundle_sha256(repository_root: Path) -> str:
    """Bind the complete trusted publication and live-validation enforcement."""

    return _repository_file_bundle_sha256(
        repository_root,
        schema="npa.libero.publication-enforcement-bundle.v2",
        paths=libero_publication_enforcement_paths(repository_root),
    )


def libero_publication_enforcement_paths(repository_root: Path) -> tuple[str, ...]:
    """Resolve every accepted runner plus its security-critical Python closure."""

    paths = set(LIBERO_PUBLICATION_ENFORCEMENT_PATHS)
    unmarked_critical_tests: list[str] = []
    for relative_root in LIBERO_PUBLICATION_ENFORCEMENT_PYTHON_ROOTS:
        directory = repository_root / relative_root
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(
                f"LIBERO enforcement root is unavailable: {relative_root}"
            )
        discovered = list(directory.rglob("*.py"))
        if not discovered:
            raise RuntimeError(f"LIBERO enforcement root is empty: {relative_root}")
        for candidate in discovered:
            if not candidate.is_file() or candidate.is_symlink():
                raise RuntimeError(
                    "LIBERO enforcement source must be a regular file: "
                    f"{candidate.relative_to(repository_root).as_posix()}"
                )
            paths.add(candidate.relative_to(repository_root).as_posix())
    for relative_root in LIBERO_PUBLICATION_ENFORCEMENT_LIBERO_TEST_ROOTS:
        directory = repository_root / relative_root
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(
                f"LIBERO enforcement test root is unavailable: {relative_root}"
            )
        discovered = list(directory.rglob("*.py"))
        if not discovered:
            raise RuntimeError(
                f"LIBERO enforcement test root is empty: {relative_root}"
            )
        for candidate in discovered:
            if not candidate.is_file() or candidate.is_symlink():
                raise RuntimeError(
                    "LIBERO enforcement test must be a regular file: "
                    f"{candidate.relative_to(repository_root).as_posix()}"
                )
            relative = candidate.relative_to(repository_root).as_posix()
            content = candidate.read_bytes()
            marked = LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER in (
                content.splitlines()
            )
            if any(
                reference in content
                for reference in LIBERO_PUBLICATION_ENFORCEMENT_CRITICAL_TEST_REFERENCES
            ) and not marked:
                unmarked_critical_tests.append(relative)
            if marked:
                paths.add(relative)
    if unmarked_critical_tests:
        raise RuntimeError(
            "LIBERO-critical tests require the publication-enforcement marker: "
            + ", ".join(sorted(unmarked_critical_tests))
        )
    return tuple(sorted(paths))


def libero_publication_lineage_values(
    qualification: dict[str, Any],
    repository_root: Path,
    *,
    development_sha: str,
) -> dict[str, Any]:
    """Resolve only a qualified, byte-identical LIBERO publication source."""

    if qualification.get("development_sha") != development_sha:
        raise RuntimeError("LIBERO qualification development SHA differs")
    observed = libero_build_input_bundle_sha256(
        repository_root, development_sha=development_sha
    )
    if observed != qualification.get("build_input_bundle_sha256"):
        raise RuntimeError("LIBERO neutral build inputs differ from qualification")
    enforcement = libero_publication_enforcement_bundle_sha256(repository_root)
    if enforcement != qualification.get("publication_enforcement_bundle_sha256"):
        raise RuntimeError("LIBERO publication enforcement differs from qualification")
    fields = (
        "candidate_image",
        "oci_digest",
        "platform_manifest_digest",
        "config_digest",
        "canonical_build_metadata_sha256",
        "attestation_manifest_digest",
        "attestation_config_digest",
        "attestation_layers",
        "package_version_digests",
        "publication_enforcement_bundle_sha256",
        "complete_image_inventory_sha256",
        "base_provenance_sha256",
        "publication_bundle_sha256",
        "package_writer_repository",
    )
    return {field: qualification[field] for field in fields}


def _libero_customer_terms(payload: dict[str, Any]) -> list[dict[str, str]]:
    customer_acceptance = payload.get("customer_acceptance")
    if (
        not isinstance(customer_acceptance, dict)
        or set(customer_acceptance)
        != {
            "schema",
            "required",
            "credentials_establish_acceptance",
            "authorization_schema",
            "terms",
        }
        or customer_acceptance.get("schema")
        != "npa.libero.customer-acceptance-requirements.v1"
        or customer_acceptance.get("required") is not True
        or customer_acceptance.get("credentials_establish_acceptance") is not False
        or customer_acceptance.get("authorization_schema")
        != LIBERO_CUSTOMER_AUTHORIZATION_SCHEMA
    ):
        raise RuntimeError("LIBERO customer-acceptance metadata is invalid")
    terms = customer_acceptance.get("terms")
    if not isinstance(terms, list) or len(terms) != 7:
        raise RuntimeError("LIBERO customer terms inventory is incomplete")
    observed_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for term in terms:
        if not isinstance(term, dict) or set(term) != {
            "id",
            "name",
            "official_url",
            "version",
        }:
            raise RuntimeError("LIBERO customer term metadata is not closed")
        term_id = str(term.get("id") or "")
        name = str(term.get("name") or "")
        url = str(term.get("official_url") or "")
        version = str(term.get("version") or "")
        if (
            not term_id
            or term_id in observed_ids
            or not name.strip()
            or not url.startswith("https://")
            or re.fullmatch(r"sha256:[0-9a-f]{64}", version) is None
        ):
            raise RuntimeError("LIBERO customer term identity is invalid")
        observed_ids.add(term_id)
        normalized.append(
            {
                "id": term_id,
                "name": name,
                "official_url": url,
                "version": version,
            }
        )
    return normalized


def libero_customer_acceptance_notification(
    payload: dict[str, Any] | None = None,
    *,
    reason: str = "authorization_missing",
) -> dict[str, Any]:
    """Return the public, side-effect-free customer acknowledgement prompt."""

    manifest = payload or libero_image_manifest()
    terms = _libero_customer_terms(manifest)
    qualification = manifest.get("qualification")
    candidate = (
        str(qualification.get("candidate_image") or "")
        if isinstance(qualification, dict)
        else ""
    )
    return {
        "schema": "npa.libero.customer-acceptance-notification.v1",
        "status": "needs_customer_acceptance",
        "solution": "libero",
        "reason": reason,
        "runtime_manifest_sha256": manifest.get("runtime_manifest_sha256"),
        "candidate_image": candidate or None,
        "terms": terms,
        "acknowledgement": {
            "required": True,
            "instructions": (
                "Review every listed official term in the authenticated NPA "
                "customer/control-plane surface, then obtain a short-lived "
                "authorization for this exact customer and run."
            ),
            "refusal": (
                "Decline or omit authorization to stop before runtime fetch, "
                "installation, cache mutation, or workload creation."
            ),
        },
        "credentials": {
            "purpose": "upstream_access_only",
            "establish_terms_acceptance": False,
        },
    }


def validate_libero_qualified_image_manifest(payload: Any) -> dict[str, Any]:
    """Validate the separately reviewed immutable image qualification."""

    def require(ok: bool, field: str) -> None:
        if not ok:
            raise RuntimeError(f"LIBERO qualification requires valid {field}")

    require(isinstance(payload, dict), "manifest object")
    require(payload.get("schema") == "npa.workbench.image-manifest.v1", "schema")
    require(payload.get("tool") == "libero", "tool")
    require(payload.get("runtime_payloads_baked") is False, "neutral payload")
    require(
        payload.get("customer_runtime_authorization_required") is True,
        "customer authorization boundary",
    )
    _libero_customer_terms(payload)
    qualification = payload.get("qualification")
    require(isinstance(qualification, dict), "qualification record")
    expected_keys = {
        "schema",
        "status",
        "qualification_id",
        "qualified_at",
        "expires_at",
        "candidate_image",
        "oci_digest",
        "platform_manifest_digest",
        "config_digest",
        "canonical_build_metadata_sha256",
        "attestation_manifest_digest",
        "attestation_config_digest",
        "attestation_layers",
        "package_version_digests",
        "complete_image_inventory_sha256",
        "base_provenance_sha256",
        "publication_bundle_sha256",
        "development_sha",
        "upstream_source_revision",
        "build_input_bundle_sha256",
        "publication_enforcement_bundle_sha256",
        "package_writer_repository",
        "runtime_manifest_sha256",
    }
    require(set(qualification) == expected_keys, "closed qualification schema")
    require(
        qualification.get("schema") == "npa.libero.image-qualification.v1",
        "qualification schema",
    )
    require(qualification.get("status") == "qualified", "qualified status")
    require(
        re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(qualification.get("qualification_id") or ""),
        )
        is not None,
        "qualification ID",
    )
    candidate = str(qualification.get("candidate_image") or "")
    require(
        re.fullmatch(
            rf"{re.escape(LIBERO_OFFICIAL_CANDIDATE_PREFIX)}[0-9a-f]{{64}}",
            candidate,
        )
        is not None,
        "official candidate image",
    )
    require(
        candidate.rsplit("@", 1)[1] == qualification.get("oci_digest"),
        "candidate digest",
    )
    for field in (
        "oci_digest",
        "platform_manifest_digest",
        "config_digest",
        "attestation_manifest_digest",
        "attestation_config_digest",
    ):
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(qualification.get(field) or ""))
            is not None,
            field,
        )
    for field in (
        "canonical_build_metadata_sha256",
        "complete_image_inventory_sha256",
        "base_provenance_sha256",
        "publication_bundle_sha256",
        "build_input_bundle_sha256",
        "publication_enforcement_bundle_sha256",
        "runtime_manifest_sha256",
    ):
        require(
            re.fullmatch(r"[0-9a-f]{64}", str(qualification.get(field) or ""))
            is not None,
            field,
        )
    require(
        re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            str(qualification.get("package_writer_repository") or ""),
        )
        is not None,
        "package writer repository",
    )
    attestation_layers = qualification.get("attestation_layers")
    require(
        isinstance(attestation_layers, list)
        and len(attestation_layers) == len(LIBERO_REQUIRED_PUBLICATION_REFERRERS),
        "complete attestation layer set",
    )
    for layer, predicate_type in zip(
        attestation_layers or [], LIBERO_REQUIRED_PUBLICATION_REFERRERS, strict=True
    ):
        require(
            isinstance(layer, dict)
            and set(layer) == {"predicate_type", "digest", "size_bytes"}
            and layer.get("predicate_type") == predicate_type
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(layer.get("digest") or ""))
            is not None
            and isinstance(layer.get("size_bytes"), int)
            and 0 < layer["size_bytes"] <= 64 * 1024 * 1024,
            f"attestation layer {predicate_type}",
        )
    package_version_digests = qualification.get("package_version_digests")
    require(
        isinstance(package_version_digests, list)
        and all(
            isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
            for digest in package_version_digests
        )
        and package_version_digests == sorted(set(package_version_digests))
        and qualification.get("oci_digest") in package_version_digests
        and qualification.get("platform_manifest_digest") in package_version_digests
        and qualification.get("attestation_manifest_digest")
        in package_version_digests,
        "closed package version digest set",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", str(qualification.get("development_sha") or ""))
        is not None,
        "development SHA",
    )
    require(
        qualification.get("upstream_source_revision") == LIBERO_UPSTREAM_SOURCE_REVISION,
        "upstream source revision",
    )
    require(
        qualification.get("runtime_manifest_sha256")
        == payload.get("runtime_manifest_sha256"),
        "runtime manifest binding",
    )
    publication_bundle = {
        "schema": "npa.libero.publication-lineage-bundle.v3",
        "candidate_image": candidate,
        "oci_digest": qualification.get("oci_digest"),
        "platform_manifest_digest": qualification.get("platform_manifest_digest"),
        "config_digest": qualification.get("config_digest"),
        "canonical_build_metadata_sha256": qualification.get(
            "canonical_build_metadata_sha256"
        ),
        "attestation_manifest_digest": qualification.get(
            "attestation_manifest_digest"
        ),
        "attestation_config_digest": qualification.get("attestation_config_digest"),
        "attestation_layers": attestation_layers,
        "package_version_digests": package_version_digests,
        "complete_image_inventory_sha256": qualification.get(
            "complete_image_inventory_sha256"
        ),
        "base_provenance_sha256": qualification.get("base_provenance_sha256"),
        "development_sha": qualification.get("development_sha"),
        "upstream_source_revision": qualification.get("upstream_source_revision"),
        "build_input_bundle_sha256": qualification.get("build_input_bundle_sha256"),
        "publication_enforcement_bundle_sha256": qualification.get(
            "publication_enforcement_bundle_sha256"
        ),
        "package_writer_repository": qualification.get("package_writer_repository"),
    }
    require(
        _canonical_sha256(publication_bundle)
        == qualification.get("publication_bundle_sha256"),
        "canonical publication lineage bundle",
    )
    artifact_review = payload.get("runtime_artifact_review")
    require(
        isinstance(artifact_review, dict)
        and set(artifact_review)
        == {
            "status",
            "pending_size_artifacts",
            "pending_license_artifacts",
            "report_sha256",
        }
        and artifact_review.get("status") == "complete"
        and artifact_review.get("pending_size_artifacts") == 0
        and artifact_review.get("pending_license_artifacts") == 0
        and re.fullmatch(
            r"[0-9a-f]{64}", str(artifact_review.get("report_sha256") or "")
        )
        is not None,
        "complete runtime artifact review",
    )
    try:
        qualified_at = datetime.fromisoformat(
            str(qualification["qualified_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(qualification["expires_at"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LIBERO qualification requires valid UTC timestamps") from exc
    current = datetime.now(timezone.utc)
    require(
        qualified_at.tzinfo is not None
        and expires_at.tzinfo is not None
        and qualified_at <= current + timedelta(minutes=5)
        and qualified_at < expires_at
        and expires_at > current
        and expires_at - qualified_at <= timedelta(days=7),
        "unexpired qualification window",
    )
    return qualification


def libero_qualified_image_manifest() -> dict[str, Any]:
    """Return only a current, complete, checked-in LIBERO qualification."""

    return validate_libero_qualified_image_manifest(libero_image_manifest())


def validate_libero_customer_runtime_authorization(
    authorization_bytes: bytes,
    *,
    image_manifest: dict[str, Any],
    run_id: str,
    customer_identity_sha256: str,
    public_key_file: str = "",
    now: datetime | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate one control-plane-issued customer/run authorization."""

    try:
        authorization = json.loads(authorization_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LIBERO customer authorization is not valid JSON") from exc
    expected_keys = {
        "schema",
        "solution",
        "status",
        "authorization_id",
        "customer_identity_sha256",
        "run_id",
        "candidate_image",
        "runtime_manifest_sha256",
        "terms",
        "issuer",
        "acknowledged_at",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    if not isinstance(authorization, dict) or set(authorization) != expected_keys:
        raise RuntimeError("LIBERO customer authorization schema is not closed")
    qualification = image_manifest.get("qualification")
    if not isinstance(qualification, dict):
        raise RuntimeError("LIBERO image qualification is unavailable")
    expected_terms = [
        {"id": term["id"], "version": term["version"]}
        for term in _libero_customer_terms(image_manifest)
    ]
    if (
        authorization.get("schema") != LIBERO_CUSTOMER_AUTHORIZATION_SCHEMA
        or authorization.get("solution") != "libero"
        or authorization.get("status") != "authorized"
        or re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(authorization.get("authorization_id") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(authorization.get("customer_identity_sha256") or ""),
        )
        is None
        or authorization.get("customer_identity_sha256")
        != customer_identity_sha256
        or authorization.get("run_id") != run_id
        or authorization.get("candidate_image")
        != qualification.get("candidate_image")
        or authorization.get("runtime_manifest_sha256")
        != image_manifest.get("runtime_manifest_sha256")
        or authorization.get("terms") != expected_terms
        or authorization.get("issuer") != "npa-customer-control-plane"
        or re.fullmatch(
            r"[A-Za-z0-9_-]{32,128}", str(authorization.get("nonce") or "")
        )
        is None
    ):
        raise RuntimeError(
            "LIBERO customer authorization does not bind the exact customer/run contract"
        )
    try:
        acknowledged_at = datetime.fromisoformat(
            str(authorization["acknowledged_at"]).replace("Z", "+00:00")
        )
        issued_at = datetime.fromisoformat(
            str(authorization["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(authorization["expires_at"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LIBERO customer authorization timestamps are invalid") from exc
    current = now or datetime.now(timezone.utc)
    if (
        acknowledged_at.tzinfo is None
        or issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or acknowledged_at > issued_at
        or issued_at - acknowledged_at > timedelta(minutes=5)
        or issued_at > current + timedelta(minutes=5)
        or expires_at <= current
        or issued_at >= expires_at
        or expires_at - issued_at > timedelta(hours=24)
    ):
        raise RuntimeError("LIBERO customer authorization is expired or replayable")
    _verify_libero_customer_authorization_signature(
        authorization, public_key_file=public_key_file
    )
    observed_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    return authorization, observed_sha256


def validate_ncore_accepted_image_manifest(payload: Any) -> dict[str, Any]:
    """Validate the reviewed NCore image/full-COLMAP/NRE acceptance tuple offline.

    Hashes identify access-controlled evidence; this does not manufacture or run
    that evidence. ``byte_scan.complete`` means the entire OCI graph, including
    index, attestations, configs, history and every ancestor layer, was covered.
    Publication additionally rechecks the exact registry artifact.
    Keep the template unaccepted and quarantine intact until real results exist.
    """

    def require(ok: bool, field: str) -> None:
        if not ok:
            raise RuntimeError(f"NCore acceptance requires valid {field}")

    def record(parent: dict[str, Any], key: str) -> dict[str, Any]:
        value = parent.get(key)
        require(isinstance(value, dict), key)
        return value

    def match(parent: dict[str, Any], key: str, pattern: str) -> None:
        value = parent.get(key)
        require(
            isinstance(value, str) and re.fullmatch(pattern, value) is not None, key
        )

    def count(parent: dict[str, Any], key: str, minimum: int = 0) -> int:
        value = parent.get(key)
        require(type(value) is int and value >= minimum, key)
        return value

    def equal(parent: dict[str, Any], key: str, expected: Any) -> None:
        value = parent.get(key)
        require(type(value) is type(expected) and value == expected, key)

    require(isinstance(payload, dict), "manifest object")
    equal(payload, "format", "npa_ncore_accepted_image_manifest_v1")
    equal(payload, "status", "accepted")
    equal(payload, "tag", public_release_tag_for_tool("ncore"))
    match(payload, "development_sha", r"[0-9a-f]{40}")
    for field in ("oci_digest", "amd64_manifest", "config_digest"):
        match(payload, field, r"sha256:[0-9a-f]{64}")
    require(
        len({payload[k] for k in ("oci_digest", "amd64_manifest", "config_digest")})
        == 3,
        "distinct index, platform and config digests",
    )
    source = record(payload, "source")
    equal(source, "ncore_revision", "59c698d206da92b406a4f72619fce3b3a2c64bfd")
    for field in ("lock_sha256", "post_patch_inventory_sha256"):
        match(source, field, r"[0-9a-f]{64}")
    for name in ("byte_scan", "payload_scan", "vulnerability_scan", "license_scan"):
        scan = record(payload, name)
        equal(scan, "status", "pass")
        match(scan, "report_sha256", r"[0-9a-f]{64}")
        equal(scan, "image_digest", payload["oci_digest"])
    byte_scan = payload["byte_scan"]
    equal(byte_scan, "complete", True)
    equal(byte_scan, "config_digest", payload["config_digest"])
    equal(byte_scan, "unresolved_findings", 0)
    for field in ("archive_sha256", "policy_sha256"):
        match(byte_scan, field, r"[0-9a-f]{64}")
    for field in ("bytes_scanned", "files_scanned"):
        count(byte_scan, field, 1)
    equal(payload["license_scan"], "unresolved_findings", 0)
    count(payload["payload_scan"], "entries_scanned", 1)
    for field in ("payload_hits", "history_hits"):
        equal(payload["payload_scan"], field, 0)
    # The name scanner also reports harmless Python .pth files. Require reviewed
    # byte evidence and exact count parity, not a filename-based licensing claim.
    count(payload["payload_scan"], "weight_shaped_paths")
    match(payload["payload_scan"], "weight_review_sha256", r"[0-9a-f]{64}")
    vulnerability = payload["vulnerability_scan"]
    for field in ("critical_with_fix", "secrets"):
        equal(vulnerability, field, 0)
    require(
        count(vulnerability, "critical_total")
        == count(vulnerability, "critical_unfixed"),
        "critical vulnerability accounting",
    )

    conversion = record(payload, "conversion")
    equal(conversion, "status", "pass")
    equal(conversion, "exit_code", 0)
    require(
        conversion.get("observed_image_digest")
        in (payload["oci_digest"], payload["amd64_manifest"]),
        "conversion image digest",
    )
    for field in (
        "report_sha256",
        "source_archive_sha256",
        "source_inventory_sha256",
        "converted_inventory_sha256",
    ):
        match(conversion, field, r"[0-9a-f]{64}")
    equal(conversion, "dataset_repository", "nvidia/PhysicalAI-NuRec-PPISP")
    equal(conversion, "dataset_revision", "2521064a3af6ab1c1caa2ba1b01ddde7eecded69")
    equal(conversion, "dataset_root", "struktur28")
    equal(
        conversion,
        "source_archive_sha256",
        "cf7ab7f100da66b2bf05b178ebcfa3a950e1bf2b1d7ff64a6c7a1e1f682afa8d",
    )
    source_counts = record(conversion, "source_counts")
    converted_counts = record(conversion, "converted_counts")
    for field, expected in (("images", 518), ("cameras", 3), ("points", 163453)):
        equal(source_counts, field, expected)
    for field in ("images", "cameras"):
        equal(converted_counts, field, source_counts[field])
    # Upstream removes near-origin SfM points; record that loss rather than
    # requiring a fabricated equality with the unfiltered sparse source count.
    points = count(converted_counts, "points", 1)
    require(
        points + count(conversion, "origin_points_filtered") == source_counts["points"],
        "complete sparse-point accounting",
    )
    for field in (
        "all_members_reopened",
        "member_hashes_verified",
        "calibration_verified",
        "poses_verified",
        "finite_geometry",
    ):
        equal(conversion, field, True)
    equal(conversion, "rig_mode", "derive")
    equal(conversion, "poses_component_group", "npa_rig")

    proof = record(payload, "rtx_proof")
    equal(proof, "status", "pass")
    equal(proof, "conversion_report_sha256", conversion["report_sha256"])
    equal(proof, "converted_inventory_sha256", conversion["converted_inventory_sha256"])
    match(proof, "nre_image", r"nvcr\.io/nvidia/nre/nre-ga@sha256:[0-9a-f]{64}")
    equal(proof, "observed_nre_digest", proof["nre_image"].split("@", 1)[1])
    equal(proof, "gpu_model", "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    count(proof, "gpu_count", 1)
    # Zero means NRE's full native recipe, not a zero-epoch training workload.
    for field in ("max_epochs", "train_exit_code", "render_exit_code"):
        equal(proof, field, 0)
    for field in (
        "training_steps",
        "gaussian_count",
        "usdz_bytes",
        "render_bytes",
        "decoded_frames",
    ):
        count(proof, field, 1)
    for field in ("report_sha256", "usdz_sha256", "render_sha256"):
        match(proof, field, r"[0-9a-f]{64}")
    equal(proof, "rendered_usdz_sha256", proof["usdz_sha256"])
    for field in ("trained_scene_reopened", "finite_pixels", "novel_view"):
        equal(proof, field, True)
    from npa.deploy.ncore_acceptance import (
        validate_full_input_proof,
        validate_selected_base_scan,
    )

    validate_full_input_proof(conversion, proof)
    validate_selected_base_scan(payload)
    return payload


@lru_cache(maxsize=1)
def ncore_accepted_image_manifest() -> dict[str, Any]:
    """Load NCore acceptance only after all required objective evidence exists."""

    try:
        payload = json.loads(
            resources.files(__package__)
            .joinpath(NCORE_IMAGE_MANIFEST_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "NCore accepted image manifest is unavailable or invalid"
        ) from exc
    return validate_ncore_accepted_image_manifest(payload)


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


def sonic_variant_workloads(variant: str) -> tuple[str, ...]:
    """Return the SONIC pipeline stages a variant is published to serve."""

    entry = sonic_image_variants().get(variant, {})
    declared = entry.get("workloads")
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            f"SONIC image variant {variant!r} declares no 'workloads' in "
            "sonic_image_manifest.json. Declare the stages it can serve so GPU "
            "resolution cannot hand a caller a variant with the wrong capability."
        )
    return tuple(str(item) for item in declared)


def sonic_image_variant_for_gpu(
    gpu_target: str | None = None,
    *,
    workload: str | None = None,
) -> str:
    """Return an active SONIC variant or reject unsupported GPU/runtime pairs.

    ``gpu_target`` alone is not enough to pick an image. The variants differ in
    capability, not just in driver provisioning: the only variant that matches a
    datacenter-Blackwell target serves MuJoCo evaluation and cannot fine-tune. So
    when the caller states its ``workload`` (a
    :mod:`npa.workbench.sonic.routing` identifier), the GPU-matched variant must
    also be published for that workload, and a mismatch fails loud instead of
    substituting a different capability.
    """

    manifest = sonic_image_manifest()
    default = str(manifest.get("default_variant", "sonic-k8s-host-mounted"))
    normalized = _normalize_gpu_target(gpu_target)
    requested = (workload or "").strip().lower()
    if not normalized:
        if requested and requested not in sonic_variant_workloads(default):
            raise ValueError(
                f"The default SONIC variant {default!r} does not serve workload "
                f"{workload!r}; it serves "
                f"{', '.join(sonic_variant_workloads(default))}. Select a variant "
                "explicitly with --image-variant or pass a separately validated "
                "image with --image."
            )
        return default
    for rule in manifest.get("gpu_selection", []):
        if not isinstance(rule, dict):
            continue
        variant = str(rule.get("variant", ""))
        for match in rule.get("matches", []):
            token = _normalize_gpu_target(str(match))
            # The family name also occurs in datacenter GPU labels. Those must
            # reach their model-specific rule, never the workstation default.
            if token == "blackwell" and classify_gpu_target(normalized) == DATACENTER_HEADLESS:
                continue
            if token in normalized:
                if not requested:
                    return variant
                served = sonic_variant_workloads(variant)
                if requested in served:
                    return variant
                capable = sorted(
                    other
                    for other, entry in sonic_image_variants().items()
                    if entry.get("status", "active") == "active"
                    and requested in sonic_variant_workloads(other)
                )
                raise ValueError(
                    f"No published SONIC image serves workload {workload!r} on GPU "
                    f"target {gpu_target!r}. That target selects variant "
                    f"{variant!r}, which is published for "
                    f"{', '.join(served)} only. Variants that do serve "
                    f"{workload!r}: {', '.join(capable) or 'none'}. Choose a GPU "
                    "target those variants support, or pass a separately validated "
                    "runtime with --image; npa will not substitute a variant with a "
                    "different capability."
                )
    raise ValueError(
        f"Unsupported SONIC GPU target {gpu_target!r}. Published selection supports "
        "sonic-k8s-host-mounted on RTX PRO 6000 Blackwell Kubernetes nodes with "
        "NVIDIA GPU Operator driver mounts, and sonic-mujoco-runtime-fetch for "
        "B200 MuJoCo evaluation. L40S/H100/H200 compute-only "
        "variants are retired and quarantined; supply a separately validated custom "
        "image explicitly or choose gpu-rtx6000 on Kubernetes."
    )


def sonic_image_entry(
    *,
    gpu_target: str | None = None,
    image_variant: str | None = None,
    workload: str | None = None,
) -> dict[str, Any]:
    """Return the SONIC manifest entry selected by variant or GPU target.

    Pass ``workload`` whenever the caller knows which pipeline stage it is
    resolving an image for, so a GPU target cannot select a variant published for
    a different capability. An explicit ``image_variant`` is still honored, but is
    checked against the workload for the same reason.
    """

    variants = sonic_image_variants()
    if image_variant:
        resolved = _normalize_sonic_variant(image_variant, variants)
        requested = (workload or "").strip().lower()
        if requested and resolved in variants:
            served = sonic_variant_workloads(resolved)
            if requested not in served:
                raise ValueError(
                    f"SONIC image variant {resolved!r} is published for "
                    f"{', '.join(served)} and cannot serve workload {workload!r}. "
                    "Pass a separately validated runtime with --image if that is "
                    "what you intend."
                )
    else:
        resolved = sonic_image_variant_for_gpu(gpu_target, workload=workload)
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
    workload: str | None = None,
) -> str:
    """Return a Workbench image, defaulting repository releases to public GHCR.

    ``registry`` is an explicit custom-image choice. The default deliberately does not
    inherit ``NPA_REGISTRY``: that variable is also used by BYOF/build automation and
    legacy operator configuration, and allowing it to repoint supported runtime images
    made otherwise-public workloads depend on private registry credentials.
    """
    resolved_registry = registry or DEFAULT_CONTAINER_REGISTRY
    if tool == "ncore" and tool in PUBLICATION_QUARANTINE_TOOLS and not tag:
        raise ValueError(
            "NCore has no accepted release image. Supply the validated immutable "
            "image with --image-override workbench.nurec.convert_colmap=IMAGE@sha256:DIGEST "
            "or explicitly select a dev-<full-source-sha> tag for validation."
        )
    if tool == "sonic":
        entry = sonic_image_entry(
            gpu_target=gpu_target,
            image_variant=image_variant,
            workload=workload,
        )
        image_name = str(entry["name"])
        resolved_tag = tag or str(entry["tag"])
    else:
        if image_variant:
            raise ValueError(
                f"Image variants are only defined for SONIC, got tool={tool!r}"
            )
        if workload:
            raise ValueError(
                f"Workload-specific image selection is only defined for SONIC, "
                f"got tool={tool!r}"
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
    if (
        tool == "ncore"
        and is_public_registry(resolved_registry)
        and resolved_tag == public_release_tag_for_tool("ncore")
    ):
        ncore_accepted_image_manifest()
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
    if tool == "ncore":
        # The generic recipe omits the mandatory source revision and would build
        # an unsupported release tag. This helper deliberately never publishes.
        requested_tag = repository.partition(":")[2]
        if not re.fullmatch(r"dev-[0-9a-f]{40}", requested_tag):
            return ""
        return (
            "bash npa/docker/workbench/ncore/build.sh "
            f"--source-sha {requested_tag.removeprefix('dev-')} --image {shlex.quote(ref)}"
        )
    dockerfile = _workbench_dockerfile(tool)
    if not dockerfile:
        # Not every tool builds from npa/docker/workbench/<tool>/Dockerfile
        # (sim2real tools in particular live elsewhere). Printing a command whose
        # -f path does not exist is worse than printing none.
        return ""
    registry = ref.rsplit("/", 1)[0]
    tag = supported_tool_version(tool)
    return (
        "npa/.venv/bin/python npa/src/npa/workflow_build.py "
        "--stage-catalog --package-root npa && "
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
    """Resolve an operator build/BYOF registry, otherwise public GHCR.

    Repository-owned runtime image defaults use :func:`container_image_for_tool`,
    which intentionally does not call this compatibility helper.
    """
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


def is_official_public_image(image: str) -> bool:
    """Whether ``image`` belongs to an official anonymous NPA GHCR namespace."""

    candidate = str(image or "").strip().removeprefix("docker:").lower()
    return any(
        candidate.startswith(f"{registry}/")
        for registry in {
            DEFAULT_PUBLIC_CONTAINER_REGISTRY.lower(),
            public_container_registry().rstrip("/").lower(),
        }
    )


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
    """Return the recorded development SHA, requiring complete NCore acceptance.

    NCore's first promotion requires acceptance before a release record exists.
    Missing or inconsistent evidence therefore raises instead of allowing the
    publisher to fall back to an arbitrary development SHA.

    Args:
        tool: Canonical workbench tool name.
    Returns:
        The accepted source SHA, or None for other tools without a record.
    Raises:
        RuntimeError: Required acceptance is missing or disagrees with release evidence.
        ValueError: A recorded development SHA is malformed.
    """

    entry = (public_release_manifest().get("releases") or {}).get(tool) or {}
    if tool == "ncore":
        accepted = ncore_accepted_image_manifest()
        for release_key, accepted_key in (
            ("development_sha", "development_sha"),
            ("published_digest", "oci_digest"),
        ):
            if entry and entry.get(release_key) != accepted[accepted_key]:
                raise RuntimeError(
                    f"NCore release and accepted-image {release_key} disagree"
                )
        # The first promotion has acceptance evidence before a published record.
        return accepted["development_sha"]
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
