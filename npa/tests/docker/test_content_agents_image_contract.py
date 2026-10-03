from __future__ import annotations

from pathlib import Path
import re

import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    OMNIVERSE_RESTRICTED_TOOLS,
    SUPPORTED_TOOL_VERSIONS,
    PUBLICATION_QUARANTINE_TOOLS,
    content_agents_accepted_image_manifest,
)
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


def test_source_and_runtime_download_description_are_immutable() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert f"ARG CONTENT_AGENTS_REVISION={CONTENT_AGENTS_REVISION}" in text
    assert f"ARG CONTENT_AGENTS_VERSION={CONTENT_AGENTS_VERSION}" in text
    assert f'OVRTX_VERSION = "{OVRTX_VERSION}"' in (
        ROOT / "npa" / "src" / "npa" / "workflows" / "content_agents.py"
    ).read_text(encoding="utf-8")
    assert "fetch --depth=1 --filter=blob:none" in text
    assert 'test "$(git rev-parse HEAD)" = "${CONTENT_AGENTS_REVISION}"' in text
    runtime = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "content_agents_runtime.py"
    ).read_text(encoding="utf-8")
    assert "ed582577175e4a5b32f8b69ef9cdbfc3d7337f3786051d8b076e30a2652f6fa5" in runtime
    assert "a6b2b3c357f6487451c8d71e96cc4f83156c08fd9747d10e1b65f3866bed4b8f" in runtime
    assert "https://pypi.nvidia.com/ovrtx/" in runtime
    assert "@sha256:" in text.splitlines()[6]


def test_public_image_has_immutable_accepted_live_evidence() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["content-agents"]
    assert entry["tier"] == "job"
    assert entry["redistribution"] == "public"
    assert entry["ovrtx_runtime_fetch"] is True
    assert "content-agents" not in OMNIVERSE_RESTRICTED_TOOLS
    assert CONTAINER_IMAGE_NAMES["content-agents"] == "npa-content-agents"
    assert SUPPORTED_TOOL_VERSIONS["content-agents"] == "0.5.2-npa2"
    assert "content-agents" not in PUBLICATION_QUARANTINE_TOOLS
    accepted = content_agents_accepted_image_manifest()
    assert accepted["tag"] == SUPPORTED_TOOL_VERSIONS["content-agents"]
    assert accepted["rtx_proof"]["observed_image_id_digest"] == accepted["oci_digest"]
    assert accepted["payload_scan"]["findings"] == 0
    assert accepted["general_payload_scan"]["payload_hits"] == 0
    assert accepted["general_payload_scan"]["history_hits"] == 0
    publication = accepted["anonymous_publication"]
    assert publication["verified"] is True
    assert publication["oci_digest"] == accepted["oci_digest"]
    assert publication["platform_manifests"] == 1
    assert publication["attestation_manifests"] == 1
    assert publication["redistribution_label"] == "public"
    assert publication["ovrtx_delivery_label"] == "runtime-fetch"
    assert 'npa.redistribution="public"' in DOCKERFILE.read_text(encoding="utf-8")


def test_no_local_acceptance_or_consent_plumbing_exists() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    gate = "NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS"
    assert gate not in dockerfile
    assert gate not in build
    for replacement in ("ACCEPT_EULA", "OMNI_KIT_ACCEPT_EULA"):
        assert replacement not in dockerfile
        assert replacement not in build
    assert "0.5.2-npa2" in build


def test_image_excludes_optional_restricted_payloads_and_samples() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "apps/*/tests apps/*/examples apps/*/data" in dockerfile
    assert "-iname '*.usdz'" in dockerfile
    assert "FROM scratch AS public-image" in dockerfile
    assert "COPY --from=assembled / /" in dockerfile
    assert "rm -rf /usr/local/cuda-*/compat" in dockerfile
    assert "dpkg --purge --force-depends linux-libc-dev" in dockerfile
    assert "-type d ! -name s3 -exec rm -rf" in dockerfile
    assert "COPY --from=uv-bin /uv /usr/local/bin/uv" in dockerfile
    assert "rm -f /usr/local/bin/uv" not in dockerfile
    assert 'find_spec("ovphysx") is None' in dockerfile
    assert 'find_spec("ovrtx") is None' in dockerfile
    assert "test ! -e .build-resources/scene_optimizer_core" in dockerfile
    assert "COPY .build-resources" not in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "NEBIUS_TOKEN_FACTORY_KEY" not in dockerfile
    assert "--provision-only" not in dockerfile
    assert "uv venv" not in dockerfile
    assert "-e '.[telemetry]'" not in dockerfile
    assert "OTEL_SDK_DISABLED=true" in dockerfile


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


def test_image_carries_only_the_npa_modules_reached_by_its_toolrefs() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY src/npa/workflows/content_agents.py" in dockerfile
    assert "COPY src/npa/workflows/content_agents_runtime.py" in dockerfile
    assert "COPY src/npa/clients/storage.py" in dockerfile
    assert "COPY --chown=1000:1000 src/npa" not in dockerfile
    assert "uv pip install --python /opt/venv/bin/python /opt/npa" not in dockerfile
    catalog = (
        ROOT / "npa" / "src" / "npa" / "orchestration" / "npa_workflow" / "catalog.py"
    ).read_text(encoding="utf-8")
    assert (
        '_CONTENT_AGENTS_PIPELINE = [\n    "python3",\n    "-m",\n    "npa.workflows.content_agents",\n]'
        in catalog
    )


def test_build_inspection_proves_ovrtx_is_absent() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "npa.workflows.content_agents inspect-image" in dockerfile
    assert "npa.workflows.content_agents inspect-runtime" not in dockerfile
    assert "test ! -e /opt/content-agents/.ovrtx_venv" in dockerfile
