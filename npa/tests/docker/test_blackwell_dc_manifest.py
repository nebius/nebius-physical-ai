"""Enforce the datacenter Blackwell (B200 sm_100 / B300 sm_103) image manifest.

The manifest is the plan of record for making every workbench image deployable
on datacenter Blackwell. These tests exist to stop the two failure modes that
make such a plan worthless:

* an image quietly falling off the list, so nobody knows whether it was
  considered; and
* a ``blocked`` entry that is really a stub - "blocked" with no upstream reason
  is indistinguishable from "we gave up", and the repo convention is to track
  the vendor, not fake a pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKBENCH_DOCKER = ROOT / "npa" / "docker" / "workbench"
MANIFEST_PATH = WORKBENCH_DOCKER / "blackwell-dc-images.json"
CONTRACT_PATH = WORKBENCH_DOCKER / "packaging-contract.yaml"

EXPECTED_FORMAT = "npa_blackwell_dc_image_manifest_v1"
# A legacy provider-registry location must never be baked into the manifest;
# callers resolve public GHCR or an explicit operator registry through NPA.
REGISTRY_ID_RE = re.compile(r"cr\.[a-z0-9-]+\.nebius\.cloud/[a-z0-9]+")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(manifest: dict) -> list[dict]:
    return manifest["images"]


def test_manifest_format_and_target(manifest: dict) -> None:
    assert manifest["format"] == EXPECTED_FORMAT
    target = manifest["target"]
    gpus = {gpu["name"]: gpu for gpu in target["gpus"]}
    assert set(gpus) == {"B200", "B300 (Blackwell Ultra)"}
    assert gpus["B200"]["sm"] == "sm_100"
    assert gpus["B200"]["compute_capability"] == "10.0"
    assert gpus["B300 (Blackwell Ultra)"]["sm"] == "sm_103"
    assert gpus["B300 (Blackwell Ultra)"]["compute_capability"] == "10.3"
    # Datacenter Blackwell has no RT cores, so rendering must never route here.
    assert target["rt_cores"] is False


def test_manifest_does_not_hardcode_a_live_nebius_identifier(manifest: dict) -> None:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    found = REGISTRY_ID_RE.findall(raw)
    assert not found, f"manifest hardcodes registry ids {found}; use ${{NPA_REGISTRY}}"
    assert "${NPA_REGISTRY}" in manifest["registry_ref"]


def test_every_entry_is_well_formed(manifest: dict, entries: list[dict]) -> None:
    allowed_verdicts = set(manifest["verdicts"])
    allowed_validation = set(manifest["validation_states"])
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names)), "duplicate image names in the manifest"

    for entry in entries:
        name = entry["name"]
        assert name.startswith("npa-"), f"{name} should use its npa-* image name"
        assert entry["verdict"] in allowed_verdicts, (
            f"{name} has verdict {entry['verdict']!r}"
        )
        assert entry["validation"] in allowed_validation, (
            f"{name} has validation state {entry['validation']!r}"
        )
        dockerfile = WORKBENCH_DOCKER / entry["dockerfile"]
        assert dockerfile.is_file(), f"{name} points at missing {entry['dockerfile']}"
        if "alternate_dockerfile" in entry:
            alternate = WORKBENCH_DOCKER / entry["alternate_dockerfile"]
            assert alternate.is_file(), (
                f"{name} points at missing {entry['alternate_dockerfile']}"
            )
        if "build_script" in entry:
            script = WORKBENCH_DOCKER / entry["build_script"]
            assert script.is_file(), f"{name} points at missing {entry['build_script']}"


def test_blocked_entries_track_an_upstream_reason(entries: list[dict]) -> None:
    """A blocked verdict must name the vendor gate, never stand in for a stub."""

    blocked = [entry for entry in entries if entry["verdict"] == "blocked"]
    assert blocked, "the manifest should still record the vendor-paced images"
    for entry in blocked:
        name = entry["name"]
        assert entry.get("blocked_reason", "").strip(), (
            f"{name} is blocked with no reason"
        )
        assert entry.get("upstream_tracking", "").strip(), (
            f"{name} is blocked with nothing to track upstream"
        )


def test_port_entries_name_their_blocker(entries: list[dict]) -> None:
    for entry in [item for item in entries if item["verdict"] == "port"]:
        name = entry["name"]
        assert entry.get("port_blocker", "").strip(), (
            f"{name} needs a port but names no blocker"
        )
        assert entry.get("upstream_tracking", "").strip(), (
            f"{name} needs a port but nothing to track upstream"
        )


def test_gpu_entries_declare_a_real_capability_smoke(entries: list[dict]) -> None:
    """A CUDA probe proves nothing; each GPU image names a functional smoke."""

    for entry in entries:
        if entry["verdict"] == "not-applicable":
            continue
        assert entry.get("smoke", "").strip(), (
            f"{entry['name']} declares no capability smoke"
        )


def test_lerobot_b300_uses_the_supported_python_and_cuda_stack() -> None:
    dockerfile = (WORKBENCH_DOCKER / "lerobot" / "Dockerfile.b300").read_text(
        encoding="utf-8"
    )
    assert "python3.12 -m venv /opt/lerobot/venv" in dockerfile
    assert "torch==2.9.0" in dockerfile
    assert "torchvision==0.24.0" in dockerfile
    assert "torchcodec==0.8.1" in dockerfile
    assert "/opt/npa/venv/bin" in dockerfile, (
        "the inherited base validator venv must remain usable"
    )
    assert dockerfile.index("pip install \\") < dockerfile.index(
        "dpkg --purge --force-depends linux-libc-dev"
    ), "build-time Linux headers must survive until evdev has compiled"


def test_cpu_entries_need_no_arch_validation(entries: list[dict]) -> None:
    for entry in [item for item in entries if item["verdict"] == "not-applicable"]:
        assert entry["validation"] == "not-required", (
            f"{entry['name']} is CPU-only but claims an arch validation state"
        )
        assert "torch_cuda_arch_list" not in entry


def test_manifest_classifies_every_packaged_image() -> None:
    """No workbench image may silently escape a datacenter Blackwell verdict."""

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    classified: set[str] = set()
    for entry in manifest["images"]:
        classified.add(entry["dockerfile"])
        if "alternate_dockerfile" in entry:
            classified.add(entry["alternate_dockerfile"])

    missing = {
        key: spec["dockerfile"]
        for key, spec in contract["images"].items()
        if spec["dockerfile"] not in classified
    }
    assert not missing, (
        "these packaged images have no datacenter Blackwell verdict: "
        f"{sorted(missing)}; add them to blackwell-dc-images.json"
    )


def test_published_tags_are_additive_and_arch_labelled(entries: list[dict]) -> None:
    """A published tag must be collision-proof and say which architectures it carries.

    Overwriting a tag is the failure mode this guards against: the UTC stamp
    makes every build a new tag, and the sm markers keep routing auditable.
    """

    published = [entry for entry in entries if "published_tag" in entry]
    assert published, "at least npa-base should record the tag that was pushed"
    for entry in published:
        tag = entry["published_tag"]
        name = entry["name"]
        if entry.get("publication_model") == "exact-digest-promoted":
            from npa.deploy.images import (
                CONTAINER_IMAGE_NAMES,
                GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS,
                SUPPORTED_TOOL_VERSIONS,
            )

            tool = next(tool for tool, image in CONTAINER_IMAGE_NAMES.items() if image == name)
            assert tag == SUPPORTED_TOOL_VERSIONS[tool]
            assert entry["published_digest"] == GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS[tool]
        else:
            assert tag.startswith(("cuda13-b300-", "cu128-torch27-sm100-")), (
                f"{name} tag {tag!r} is outside the published Blackwell tag families"
            )
            assert re.search(r"-\d{8}T\d{6}Z$", tag), (
                f"{name} tag {tag!r} has no UTC stamp, so a rebuild would overwrite it"
            )
            assert "sm100" in tag, f"{name} tag {tag!r} does not advertise sm_100"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", entry["published_digest"]), (
            f"{name} has a malformed digest"
        )
        expected_registries = {"public-release"}
        if entry.get("publication_model") == "exact-digest-promoted":
            expected_registries.add("public-development")
        assert set(entry["published_registries"]) == expected_registries, (
            f"{name} must record only registries where this exact artifact was published"
        )


def test_redistribution_claims_do_not_drift_from_the_contract(
    entries: list[dict],
) -> None:
    """The manifest must not keep its own stale copy of the redistribution class.

    The Isaac images were re-architected to fetch Isaac Sim at run time and are
    no longer restricted; a hardcoded ``restricted`` here would outlive that.
    """

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    by_dockerfile = {
        spec["dockerfile"]: spec.get("redistribution")
        for spec in contract["images"].values()
    }
    for entry in entries:
        claimed = entry.get("redistribution")
        if claimed is None:
            continue
        actual = by_dockerfile.get(entry["dockerfile"])
        assert claimed == actual, (
            f"{entry['name']} claims redistribution {claimed!r} but the packaging "
            f"contract says {actual!r}"
        )


def test_names_match_the_real_container_image_names(entries: list[dict]) -> None:
    """Manifest rows must use the image names the deploy layer actually resolves.

    A plausible-looking alias (npa-sim2real-envgen for npa-envgen) makes the
    manifest impossible to cross-reference against a registry.
    """

    from npa.deploy.images import CONTAINER_IMAGE_NAMES

    known = set(CONTAINER_IMAGE_NAMES.values())
    # Base and helper images are not deployable tools, so they are not in the map.
    not_deployable_tools = {
        "npa-base",
        "npa-workbench-cuda-base",
        "npa-sonic-mujoco",
        "npa-sonic-export",
        "npa-cosmos3-serving",
        "npa-content-agents",
        "npa-sim2real-control",
    }
    unknown = [
        entry["name"]
        for entry in entries
        if entry["name"] not in known and entry["name"] not in not_deployable_tools
    ]
    assert not unknown, (
        f"{unknown} are not in CONTAINER_IMAGE_NAMES; use the real npa-* image name"
    )


def test_compatibility_matrix_lists_every_image(entries: list[dict]) -> None:
    """The published matrix must not quietly fall behind the manifest."""

    matrix = (ROOT / "docs/workbench/image-gpu-compatibility-matrix.md").read_text(
        encoding="utf-8"
    )
    missing = [entry["name"] for entry in entries if f"`{entry['name']}`" not in matrix]
    assert not missing, (
        f"these images have a verdict but no row in the compatibility matrix: {missing}"
    )


def test_compatibility_matrix_is_linked_from_the_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The property is that a reader can find the matrix from the front page.
    # This also asserted a `## Container registry` heading, which #289 removed
    # while leaving the link in place — an incidental check on the shape of a
    # document rather than on the thing being guarded.
    assert "docs/workbench/image-gpu-compatibility-matrix.md" in readme


def test_measured_arch_lists_back_the_verdicts(entries: list[dict]) -> None:
    """A measured wheel arch set and the image's verdict must agree.

    An image whose wheel carries no sm_100 cannot be 'ready' for datacenter
    Blackwell, and one that does carry it should not be filed as needing a port.
    """

    for entry in entries:
        arch_list = entry.get("measured_arch_list")
        if arch_list is None:
            continue
        covers_sm100 = "sm_100" in arch_list
        verdict = entry["verdict"]
        if verdict == "ready":
            assert covers_sm100, (
                f"{entry['name']} is 'ready' but its measured wheel {arch_list} "
                "has no sm_100"
            )
        if verdict == "port":
            assert not covers_sm100, (
                f"{entry['name']} is filed as 'port' but its measured wheel already "
                f"carries sm_100 ({arch_list}); re-check the verdict"
            )


def test_wan_validation_is_bound_to_an_immutable_accepted_tuple(
    entries: list[dict],
) -> None:
    wan = next(entry for entry in entries if entry["name"] == "npa-wan2-2")
    manifest_path = ROOT / wan["accepted_image_manifest"]
    accepted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert accepted["format"] == "npa_wan_accepted_image_manifest_v1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", accepted["oci_digest"])
    assert re.fullmatch(r"[0-9a-f]{40}", accepted["development_sha"])
    assert re.fullmatch(r"[0-9a-f]{64}", accepted["runtime_requirements_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", accepted["baked_constraints_sha256"])
    for identity_name in ("source", "model", "tokenizer"):
        assert re.fullmatch(r"[0-9a-f]{40}", accepted[identity_name]["revision"])
    proof = accepted["single_gpu_proof"]
    assert proof["gpu_count"] == 1
    assert proof["observed_image_digest"] == accepted["oci_digest"]
    for key in (
        "artifact_sha256",
        "runtime_inventory_sha256",
        "mp4_sha256",
        "rrd_sha256",
        "rrd_manifest_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", proof[key])
    assert proof["record"] == wan["validation_run"]
    assert accepted["payload_scan"]["archives_scanned"] == 21
    assert accepted["payload_scan"]["findings"] == 0
    assert accepted["vulnerability_scan"]["critical_total"] == 27
    assert accepted["vulnerability_scan"]["critical_with_fix"] == 0
    assert accepted["vulnerability_scan"]["critical_unfixed"] == 27
    assert accepted["vulnerability_scan"]["secrets"] == 0


def test_content_agents_validation_is_bound_to_clean_bytes_and_rtx(
    entries: list[dict],
) -> None:
    item = next(entry for entry in entries if entry["name"] == "npa-content-agents")
    accepted = json.loads(
        (ROOT / item["accepted_image_manifest"]).read_text(encoding="utf-8")
    )

    assert item["redistribution"] == "public"
    assert item["verdict"] == "blocked", "B200/B300 are not RT-core renderers"
    assert item["validation"] == "validated"
    assert accepted["format"] == "npa_content_agents_accepted_image_manifest_v1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", accepted["oci_digest"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", accepted["amd64_manifest"])
    assert accepted["runtime"]["version"] == "0.3.0.312915"
    assert accepted["runtime"]["delivery"] == "anonymous-runtime-fetch-from-nvidia"
    assert accepted["runtime"]["ready_markers"] == 1
    assert accepted["runtime"]["reused_render_stages"] == 3
    assert accepted["payload_scan"]["archives_scanned"] == 3
    assert accepted["payload_scan"]["findings"] == 0
    assert accepted["general_payload_scan"]["entries_scanned"] == 28471
    assert accepted["general_payload_scan"]["payload_hits"] == 0
    assert accepted["general_payload_scan"]["history_hits"] == 0
    assert accepted["vulnerability_scan"]["critical_total"] == 0
    assert accepted["vulnerability_scan"]["secrets"] == 0
    publication = accepted["anonymous_publication"]
    assert publication["verified"] is True
    assert publication["oci_digest"] == accepted["oci_digest"]
    assert publication["oci_user"] == "ubuntu"
    proof = accepted["rtx_proof"]
    assert proof["record_id"] == item["validation_run"]
    assert proof["observed_image_id_digest"] == accepted["oci_digest"]
    assert proof["gpu_model"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert proof["gpu_count"] == 1
    assert proof["upstream_validation"] == "pass"
    assert [
        proof["material_render_count"],
        proof["physics_render_count"],
        proof["validation_render_count"],
    ] == [6, 6, 1]
    assert proof["rigid_physics"]["fixed"] is False
    assert proof["rigid_physics"]["mass_or_density"] > 0
    assert 0.1 <= proof["rigid_physics"]["friction"] <= 2.0


def test_ltx_validation_is_bound_to_zero_payload_bytes_and_gpu_evidence(
    entries: list[dict],
) -> None:
    ltx = next(entry for entry in entries if entry["name"] == "npa-ltx2")
    accepted = json.loads(
        (ROOT / ltx["accepted_image_manifest"]).read_text(encoding="utf-8")
    )

    assert accepted["format"] == "npa_ltx2_accepted_image_manifest_v1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", accepted["oci_digest"])
    assert re.fullmatch(r"[0-9a-f]{40}", accepted["development_sha"])
    assert accepted["payload_scan"] == {
        "scanner": "npa/scripts/scan_image_ltx_payload.py",
        "findings": 0,
        "source_baked": False,
        "weights_baked": False,
        "credentials_baked": False,
    }
    assert re.fullmatch(r"[0-9a-f]{40}", accepted["source"]["revision"])
    assert re.fullmatch(r"[0-9a-f]{40}", accepted["weights"]["resolved_revision"])
    proof = accepted["gpu_proof"]
    assert proof["observed_image_digest"] == accepted["oci_digest"]
    assert proof["gpu_count"] == 1
    assert proof["deferred"] == []
    assert proof["source_baked"] is False
    assert proof["weights_baked"] is False
    for key in ("artifact_sha256", "refusal_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", proof[key])
    assert re.fullmatch(r"[0-9a-f]{64}", proof["video"]["sha256"])
    assert proof["video"]["frame_count"] >= 24


def test_base_image_covers_both_blackwell_majors(entries: list[dict]) -> None:
    """npa-base gates the tree, so its arch list must span sm_100 and sm_120."""

    base = next(entry for entry in entries if entry["name"] == "npa-base")
    arch_list = base["torch_cuda_arch_list"].split()
    assert {"8.0", "9.0", "10.0", "10.3", "12.0"} <= set(arch_list), arch_list

    dockerfile = (WORKBENCH_DOCKER / base["dockerfile"]).read_text(encoding="utf-8")
    declared = re.search(r'ARG TORCH_CUDA_ARCH_LIST="([^"]+)"', dockerfile)
    assert declared, "npa-base must declare TORCH_CUDA_ARCH_LIST as a build arg"
    assert declared.group(1).split() == arch_list, (
        "the manifest arch list drifted from the Dockerfile default"
    )
