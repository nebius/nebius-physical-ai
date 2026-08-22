from __future__ import annotations

from pathlib import Path
import re

import yaml

from npa.deploy.images import CONTAINER_IMAGE_NAMES, OMNIVERSE_RESTRICTED_TOOLS
from npa.workflows.content_agents import (
    CONTENT_AGENTS_REVISION,
    CONTENT_AGENTS_VERSION,
    OVRTX_VERSION,
)


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa" / "docker" / "workbench" / "content-agents"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
BUILD = IMAGE_DIR / "build.sh"
CONTRACT = ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


def test_source_and_native_renderer_are_immutable() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert f"ARG CONTENT_AGENTS_REVISION={CONTENT_AGENTS_REVISION}" in text
    assert f"ARG CONTENT_AGENTS_VERSION={CONTENT_AGENTS_VERSION}" in text
    assert f'OVRTX_VERSION = "{OVRTX_VERSION}"' in (
        ROOT / "npa" / "src" / "npa" / "workflows" / "content_agents.py"
    ).read_text(encoding="utf-8")
    assert "fetch --depth=1 --filter=blob:none" in text
    assert 'test "$(git rev-parse HEAD)" = "${CONTENT_AGENTS_REVISION}"' in text
    assert "--require-hashes --no-deps --no-config --no-sources" in text
    assert "@sha256:" in text.splitlines()[6]


def test_restricted_image_is_not_a_public_tool() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["content-agents"]
    assert entry["tier"] == "job"
    assert entry["redistribution"] == "restricted"
    assert "content-agents" in OMNIVERSE_RESTRICTED_TOOLS
    assert "content-agents" not in CONTAINER_IMAGE_NAMES
    assert 'npa.redistribution="restricted"' in DOCKERFILE.read_text(encoding="utf-8")


def test_operator_acceptance_is_explicit_and_never_baked() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    gate = "NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS"
    assert gate not in dockerfile
    assert f'if [[ "${{{gate}:-}}" != "YES" ]]' in build
    assert "NVIDIA Software License Agreement" in build
    assert "Product Specific Terms for NVIDIA AI Products" in build
    assert "nvidia-software-license-agreement" in build
    assert "product-specific-terms-for-ai-products" in build
    assert f"env -u {gate}" in build
    for host in ("ghcr.io", "docker.io", "quay.io", "public.ecr.aws"):
        assert host in build


def test_image_excludes_optional_restricted_payloads_and_samples() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "apps/*/tests apps/*/examples docs tests" in dockerfile
    assert "apps/material_agent/data apps/physics_agent/data" in dockerfile
    assert 'find_spec("ovphysx") is None' in dockerfile
    assert "test ! -e .build-resources/scene_optimizer_core" in dockerfile
    assert "COPY .build-resources" not in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "NEBIUS_TOKEN_FACTORY_KEY" not in dockerfile


def test_image_runs_non_root_and_uses_a_forwarding_xvfb_wrapper() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER ubuntu" in text
    assert 'ENTRYPOINT ["/usr/local/bin/npa-content-agents-entrypoint"]' in text
    assert "docker/workbench/content-agents/npa-content-agents-entrypoint" in text
    env_match = re.search(r"\bNVIDIA_DRIVER_CAPABILITIES=([^\s\\]+)", text)
    label_match = re.search(r'npa\.driver_capabilities="([^"]+)"', text)
    assert env_match is not None
    assert label_match is not None
    capabilities = "compute,utility,graphics,display"
    assert env_match.group(1) == capabilities
    assert 'npa.driver_provisioning="gpu-operator-host-mounted"' in text
    assert label_match.group(1) == env_match.group(1)
    assert "EXPOSE" not in text


def test_only_real_upstream_agent_entrypoints_are_installed_and_invoked() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    adapter = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "content_agents.py"
    ).read_text(encoding="utf-8")
    assert "for package in material_agent physics_agent validation_agent" in dockerfile
    for entrypoint in ("material-agent", "physics-agent", "validation-agent"):
        assert entrypoint in adapter
    assert "docker compose" not in dockerfile.lower()
    assert "echo" not in adapter.lower()


def test_image_installs_the_npa_workflow_console_without_dynamic_dependencies() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "uv pip install --python /opt/venv/bin/python /opt/npa" in dockerfile
    assert "--no-deps --no-config --no-sources" in dockerfile
    assert "command -v npa" in dockerfile
    assert "npa --version" in dockerfile


def test_npa_provenance_does_not_invalidate_the_pinned_ovrtx_layer() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    upstream_install = dockerfile.index("pylock.ovrtx-runtime.toml")
    npa_copy = dockerfile.index("COPY --chown=1000:1000 src/npa")
    source_label = dockerfile.index('LABEL npa.source_revision="${NPA_SOURCE_SHA}"')
    assert upstream_install < npa_copy < source_label
