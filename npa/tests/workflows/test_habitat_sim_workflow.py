"""Validate the dedicated Habitat-Sim workflow and runtime refusal paths."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import zlib

import pytest
import yaml

from npa.deploy.images import container_image_for_tool
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows import habitat_sim_smoke as H


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/habitat-sim-smoke.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
LIVE_SPEC = importlib.util.spec_from_file_location(
    "habitat_live_selector", ROOT / "npa/tests/e2e/test_habitat_sim_image_live_e2e.py"
)
assert LIVE_SPEC is not None and LIVE_SPEC.loader is not None
LIVE = importlib.util.module_from_spec(LIVE_SPEC)
LIVE_SPEC.loader.exec_module(LIVE)


class Response(io.BytesIO):
    def geturl(self) -> str:
        return H.ARCHIVE_URL

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _archive(scene: bytes = b"scene", navmesh: bytes = b"navmesh") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(H.MEMBER_SPECS[H.SCENE_NAME]["archive_member"], scene)
        bundle.writestr(H.MEMBER_SPECS[H.NAVMESH_NAME]["archive_member"], navmesh)
        bundle.writestr("unrelated/other.glb", b"never extracted")
    return stream.getvalue()


def _configure_archive(monkeypatch, payload: bytes) -> None:
    scene, navmesh = b"scene", b"navmesh"
    monkeypatch.setattr(H, "ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(H, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        H,
        "MEMBER_SPECS",
        {
            H.SCENE_NAME: {
                "archive_member": H.MEMBER_SPECS[H.SCENE_NAME]["archive_member"],
                "bytes": len(scene),
                "crc32": f"{zlib.crc32(scene):08x}",
                "sha256": hashlib.sha256(scene).hexdigest(),
            },
            H.NAVMESH_NAME: {
                "archive_member": H.MEMBER_SPECS[H.NAVMESH_NAME]["archive_member"],
                "bytes": len(navmesh),
                "crc32": f"{zlib.crc32(navmesh):08x}",
                "sha256": hashlib.sha256(navmesh).hexdigest(),
            },
        },
    )


def test_one_state_workflow_validates_plans_and_never_selects_b200() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="habitat-contract")
    payload = yaml.safe_load(WORKFLOW.read_text())
    assert spec.metadata["name"] == "habitat-sim-smoke"
    assert set(spec.states) == {"render-traversal"}
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_ref == "workflow.habitat_sim.smoke"
    resource = payload["resources"]["rtx-renderer"]
    assert resource["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert resource["image"] == "tool://habitat-sim"
    assert "B200" not in json.dumps(payload)
    assert "habitat-sim-smoke.json" in json.dumps(payload["states"])


def test_readiness_binds_exact_workflow_and_records_unbuilt_blocker() -> None:
    payload = json.loads(READINESS.read_text())
    assert payload["schema_version"] == "workflow-readiness/v1"
    assert (
        payload["workflow_sha256"] == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert payload["planning"]["validation"]["status"] == "verified"
    assert payload["planning"]["task_fidelity"]["status"] == "verified"
    assert payload["prerequisites"]["target_runtime"]["status"] == "blocked"


def test_unbuilt_default_and_explicit_private_image_references_are_resolvable() -> None:
    assert container_image_for_tool("habitat-sim").endswith(
        "/npa-habitat-sim:0.3.3-public-unbuilt"
    )
    assert (
        container_image_for_tool(
            "habitat-sim",
            registry="registry.invalid/run-owned",
            tag="reviewed-private-stage",
        )
        == "registry.invalid/run-owned/npa-habitat-sim:reviewed-private-stage"
    )


def test_official_archive_extracts_only_exact_members_and_removes_bundle(
    tmp_path, monkeypatch
) -> None:
    payload = _archive()
    _configure_archive(monkeypatch, payload)
    root = tmp_path / "run-owned-cache"
    scene, navmesh, archive, members = H.fetch_scene_assets(
        root, opener=lambda *_args, **_kwargs: Response(payload)
    )
    assert scene.read_bytes() == b"scene" and navmesh.read_bytes() == b"navmesh"
    assert scene.stat().st_mode & 0o777 == 0o600
    assert archive["ephemeral_copy_removed"] is True
    assert archive["unrelated_members_extracted"] is False
    assert members[H.SCENE_NAME]["sha256"] == hashlib.sha256(b"scene").hexdigest()
    assert {path.name for path in root.iterdir()} == {H.NAVMESH_NAME, H.SCENE_NAME}


@pytest.mark.parametrize("payload", [b"unavailable", b"not-a-zip"])
def test_missing_or_corrupt_archive_never_leaves_partial_payload(
    tmp_path, monkeypatch, payload
) -> None:
    root = tmp_path / "run-owned-cache"
    if payload == b"unavailable":

        def opener(*_args, **_kwargs):
            raise OSError("official source unavailable")

        with pytest.raises(OSError, match="official source unavailable"):
            H.fetch_scene_assets(root, opener=opener)
    else:
        monkeypatch.setattr(H, "ARCHIVE_BYTES", len(payload))
        monkeypatch.setattr(H, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
        with pytest.raises(H.SmokeFailure, match="not a valid ZIP"):
            H.fetch_scene_assets(root, opener=lambda *_a, **_k: Response(payload))
    assert not list(root.iterdir())


def test_archive_download_stops_before_exceeding_the_pinned_size(
    tmp_path, monkeypatch
) -> None:
    payload = b"one byte too many"
    monkeypatch.setattr(H, "ARCHIVE_BYTES", len(payload) - 1)
    destination = tmp_path / "oversized.zip.part"

    with pytest.raises(H.SmokeFailure, match="exceeded its pinned byte boundary"):
        H._download(destination, opener=lambda *_a, **_k: Response(payload))

    assert not destination.exists()


def test_runtime_cache_cleanup_is_verified(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "run-owned-cache"
    cache.mkdir()
    (cache / "scene").write_bytes(b"exact")
    monkeypatch.setattr(H.shutil, "rmtree", lambda _path: None)

    with pytest.raises(H.SmokeFailure, match="cleanup did not complete"):
        H._remove_runtime_cache(cache)

    assert cache.exists()


def _live_receipt(tmp_path: Path) -> dict[str, object]:
    owner = tmp_path / "owner-only"
    owner.mkdir(mode=0o700)
    readback = {
        key: marker * 64
        for key, marker in zip(
            ("capacity", "cluster", "node-group"), ("a", "b", "c"), strict=True
        )
    }
    provider = {
        "schema_version": "npa.nebius.strict-capacity-binding.v1",
        "accelerator": "RTX PRO 6000 Blackwell",
        "capacity_block_group_id": "reservation-fixture",
        "gpu_count": 1,
        "kubernetes_context": "context-fixture",
        "node_group_reservation_policy": {
            "policy": "STRICT",
            "reservation_ids": ["reservation-fixture"],
        },
        "project": "project-fixture",
        "project_id": "project-id-fixture",
        "provider_readback_sha256": readback,
        "state": "READY",
        "verified_at": "2026-09-11T00:00:00Z",
    }
    provider_path = owner / "strict-provider.json"
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    provider_path.chmod(0o600)
    return {
        "registry": "private.invalid/task-owned",
        "image": "private.invalid/task-owned/npa-habitat-sim@sha256:" + "a" * 64,
        "kubernetes_context": provider["kubernetes_context"],
        "project": provider["project"],
        "project_id": provider["project_id"],
        "target": {
            "policy": "STRICT",
            "accelerator": provider["accelerator"],
            "gpu_count": 1,
        },
        "reservation": {
            "accelerator": provider["accelerator"],
            "capacity_block_group_id": provider["capacity_block_group_id"],
            "gpu_count": 1,
            "provider_readback_sha256": readback,
            "provider_receipt_path": str(provider_path),
            "provider_receipt_sha256": hashlib.sha256(
                provider_path.read_bytes()
            ).hexdigest(),
            "state": provider["state"],
            "verified_at": provider["verified_at"],
        },
    }


def test_live_receipt_binds_private_image_and_strict_provider_readback(
    tmp_path,
) -> None:
    receipt = _live_receipt(tmp_path)
    LIVE._assert_private_image(receipt)
    LIVE._assert_provider_binding(receipt)

    provider_path = Path(receipt["reservation"]["provider_receipt_path"])
    provider_bytes = provider_path.read_bytes()
    provider_path.write_bytes(provider_bytes + b" ")
    with pytest.raises(AssertionError):
        LIVE._assert_provider_binding(receipt)
    provider_path.write_bytes(provider_bytes)

    receipt["image"] = (
        "ghcr.io/nebius/nebius-physical-ai/npa-habitat-sim@sha256:" + "a" * 64
    )
    receipt["registry"] = "ghcr.io/nebius/nebius-physical-ai"
    with pytest.raises(AssertionError):
        LIVE._assert_private_image(receipt)


def test_live_selector_requires_exact_digest_and_terminated_zero_exit() -> None:
    image = "private.invalid/task/npa-habitat-sim@sha256:" + "a" * 64
    pod = {
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "imageID": "docker-pullable://" + image,
                    "state": {"terminated": {"exitCode": 0}},
                }
            ],
        }
    }
    assert LIVE._assert_pod_completion(pod, image).endswith("@sha256:" + "a" * 64)
    pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, image)
    pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 0
    pod["status"]["containerStatuses"][0]["imageID"] = (
        "docker-pullable://private.invalid/task/npa-habitat-sim@sha256:" + "b" * 64
    )
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, image)


def test_member_hash_duplicate_encryption_and_link_refuse(
    tmp_path, monkeypatch
) -> None:
    payload = _archive()
    _configure_archive(monkeypatch, payload)
    specs = H.MEMBER_SPECS
    specs[H.SCENE_NAME]["sha256"] = "0" * 64
    with pytest.raises(H.SmokeFailure, match="member hash mismatch"):
        H.fetch_scene_assets(
            tmp_path / "hash-cache", opener=lambda *_a, **_k: Response(payload)
        )

    member = zipfile.ZipInfo("fixture")
    member.flag_bits |= 0x1
    with pytest.raises(H.SmokeFailure, match="encrypted"):
        H._validate_zip_member(member, {"bytes": 0, "crc32": "00000000"})
    member.flag_bits = 0
    member.external_attr = 0o120777 << 16
    with pytest.raises(H.SmokeFailure, match="symbolic link"):
        H._validate_zip_member(member, {"bytes": 0, "crc32": "00000000"})

    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as bundle:
        member_name = H.MEMBER_SPECS[H.SCENE_NAME]["archive_member"]
        bundle.writestr(member_name, b"scene")
        with pytest.warns(UserWarning, match="Duplicate name"):
            bundle.writestr(member_name, b"scene")
        bundle.writestr(H.MEMBER_SPECS[H.NAVMESH_NAME]["archive_member"], b"navmesh")
    payload = duplicate.getvalue()
    _configure_archive(monkeypatch, payload)
    with pytest.raises(H.SmokeFailure, match="member count"):
        H.fetch_scene_assets(
            tmp_path / "duplicate-cache", opener=lambda *_a, **_k: Response(payload)
        )


def test_later_member_failure_removes_an_already_verified_member(
    tmp_path, monkeypatch
) -> None:
    payload = _archive()
    _configure_archive(monkeypatch, payload)
    H.MEMBER_SPECS[H.NAVMESH_NAME]["sha256"] = "0" * 64
    root = tmp_path / "partial-cache"

    with pytest.raises(H.SmokeFailure, match="member hash mismatch"):
        H.fetch_scene_assets(root, opener=lambda *_a, **_k: Response(payload))

    assert not list(root.iterdir())


def test_runtime_contract_requires_real_gpu_egl_bullet_navigation_and_attribution() -> (
    None
):
    source = (ROOT / "npa/src/npa/workflows/habitat_sim_smoke.py").read_text()
    required = (
        "RTXPRO6000BLACKWELL",
        'compute != "12.0"',
        "built_with_bullet",
        "make_greedy_follower",
        "find_path",
        "simulator.step(action",
        "libEGL_nvidia.so",
        "rendered_rgb_frame_count",
        "rendered_depth_frame_count",
        "finite_depth_statistics",
        "agent_displacement",
        "bullet_step_count",
        "pod_observed_image_digest",
        "The King's Hall",
        "CC BY 4.0",
        "modification_notice",
        '"exit_status": 0',
        '"schema_version": "npa.habitat-sim.smoke.v1"',
    )
    for token in required:
        assert token in source
    assert "huggingface.co" not in source.lower()
    assert H.ARCHIVE_URL.startswith("https://dl.fbaipublicfiles.com/")


def test_storage_upload_reads_back_hashes_and_records_media_types(
    tmp_path, monkeypatch
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "proof.json").write_bytes(b"{}\n")
    (outputs / "rgb.png").write_bytes(b"png")
    (outputs / "depth.npy").write_bytes(b"npy")

    class Storage:
        def __init__(self):
            self.objects = {}

        def upload_file(self, path, bucket, key):
            self.objects[(bucket, key)] = Path(path).read_bytes()

        def head_object(self, Bucket, Key):
            return {"ContentLength": len(self.objects[(Bucket, Key)])}

        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    storage = Storage()
    monkeypatch.setattr("boto3.client", lambda *_a, **_k: storage)
    receipt = H._upload_directory(outputs, "s3://fixture-bucket/run-owned/")
    assert receipt["object_count"] == 3
    assert {row["media_type"] for row in receipt["objects"]} == {
        "application/json",
        "application/x-npy",
        "image/png",
    }
