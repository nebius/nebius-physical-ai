"""Validate the dedicated Habitat-Sim workflow and runtime refusal paths."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import zlib
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from npa.deploy.images import container_image_for_tool
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.workflows import habitat_sim_smoke as H


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/habitat-sim-smoke.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
FIXTURE_IMAGE = "registry.invalid/run-owned/habitat-sim@sha256:" + "a" * 64
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


class Storage:
    def __init__(self, *, fail_after: int | None = None):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.fail_after = fail_after
        self.uploads = 0
        self.deleted: list[tuple[str, str]] = []

    def _store(self, bucket: str, key: str, payload: bytes) -> None:
        self.uploads += 1
        if self.fail_after is not None and self.uploads > self.fail_after:
            raise OSError("fixture interrupted upload")
        self.objects[(bucket, key)] = payload

    def upload_file(self, path, bucket, key):
        self._store(bucket, key, Path(path).read_bytes())

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        assert ContentType in {
            "application/json",
            "application/x-npy",
            "image/png",
        }
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            raise OSError("fixture immutable object already exists")
        self._store(Bucket, Key, bytes(Body))

    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        assert ContinuationToken is None
        return {
            "IsTruncated": False,
            "Contents": [
                {"Key": key}
                for bucket, key in sorted(self.objects)
                if bucket == Bucket and key.startswith(Prefix)
            ],
        }

    def delete_object(self, *, Bucket, Key):
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)


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
    assert "habitat-sim-publication-ready.json" in json.dumps(payload["states"])
    assert payload["config"]["plan_sha256"] == "0" * 64


def test_renderer_preserves_the_exact_one_rtx_habitat_placement() -> None:
    spec = load_spec(WORKFLOW)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="habitat-placement"),
        run_id="habitat-placement",
        options=SkypilotRenderOptions(
            image_overrides={"workflow.habitat_sim.smoke": FIXTURE_IMAGE},
            materialize_registry_secrets=False,
        ),
    )
    assert "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in rendered


def test_renderer_uses_only_the_baked_habitat_runtime(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://fixture/source-overlay")
    monkeypatch.setenv("NPA_SRC_OVERLAY", "1")
    spec = load_spec(WORKFLOW)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="habitat-baked-runtime"),
        run_id="habitat-baked-runtime",
        options=SkypilotRenderOptions(
            image_overrides={"workflow.habitat_sim.smoke": FIXTURE_IMAGE},
            materialize_registry_secrets=False,
        ),
    )
    task = list(yaml.safe_load_all(rendered))[1]
    assert task["envs"].get("NPA_SRC_S3_URI") is None
    assert task["envs"].get("NPA_SRC_OVERLAY") is None
    assert "/opt/venv/bin/python" in task["setup"]
    assert "sha256sum -c npa-source-manifest.sha256" in task["setup"]
    assert "pip install" not in task["setup"]
    assert "/" + "tmp/npa-src-overlay" in task["setup"]


@pytest.mark.parametrize("field", ["pip_extra", "source_overlay"])
def test_renderer_refuses_habitat_dependency_or_source_overlays(field) -> None:
    spec = load_spec(WORKFLOW)
    spec.config[field] = "hostile"
    with pytest.raises(NpaWorkflowRenderError, match="forbids dependency and source"):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="habitat-overlay-refusal"),
            run_id="habitat-overlay-refusal",
            options=SkypilotRenderOptions(
                image_overrides={"workflow.habitat_sim.smoke": FIXTURE_IMAGE},
                materialize_registry_secrets=False,
            ),
        )


@pytest.mark.parametrize(
    "override",
    [
        "B200:1",
        "H100:1",
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:2",
        "RTXPRO6000:1",
    ],
)
def test_renderer_refuses_non_exact_habitat_accelerator_overrides(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("NPA_WORKFLOW_GPU_ACCELERATOR", override)
    spec = load_spec(WORKFLOW)

    with pytest.raises(
        NpaWorkflowRenderError,
        match="Habitat-Sim rendering requires exactly",
    ):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="habitat-hostile-placement"),
            run_id="habitat-hostile-placement",
            options=SkypilotRenderOptions(materialize_registry_secrets=False),
        )


@pytest.mark.parametrize("remap", ["B200:1", "H100:1", "RTXPRO6000:1"])
def test_renderer_refuses_submit_time_habitat_accelerator_remap(remap) -> None:
    spec = load_spec(WORKFLOW)
    declared = "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"

    with pytest.raises(
        NpaWorkflowRenderError,
        match="Habitat-Sim rendering requires exactly",
    ):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="habitat-hostile-remap"),
            run_id="habitat-hostile-remap",
            options=SkypilotRenderOptions(
                gpu_accelerator_overrides={declared: remap},
                materialize_registry_secrets=False,
            ),
        )


@pytest.mark.parametrize("require_baked", [None, False, "false", True])
@pytest.mark.parametrize(
    "image",
    [
        "",
        "registry.invalid/run-owned/habitat-sim:latest",
        "habitat-sim@sha256:" + "a" * 64,
        "namespace/habitat-sim@sha256:" + "a" * 64,
        "registry.invalid/habitat-sim@sha256:incomplete",
    ],
)
def test_habitat_image_requires_registry_digest_independently_of_baked_flag(
    require_baked, image
) -> None:
    spec = load_spec(WORKFLOW)
    if require_baked is not None:
        spec.config["require_baked_npa"] = require_baked
    with pytest.raises(
        NpaWorkflowRenderError, match="registry-qualified immutable image"
    ):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="habitat-image-refusal"),
            run_id="habitat-image-refusal",
            options=SkypilotRenderOptions(
                image_overrides={"workflow.habitat_sim.smoke": image},
                materialize_registry_secrets=False,
            ),
        )


@pytest.mark.parametrize("require_baked", [None, False, True])
def test_habitat_accepts_inert_digest_and_identity_only_accelerator_map(
    require_baked,
) -> None:
    spec = load_spec(WORKFLOW)
    if require_baked is not None:
        spec.config["require_baked_npa"] = require_baked
    spec.config["source_sha"] = "a" * 40
    declared = "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="habitat-image-identity"),
        run_id="habitat-image-identity",
        options=SkypilotRenderOptions(
            image_overrides={"workflow.habitat_sim.smoke": FIXTURE_IMAGE},
            gpu_accelerator_overrides={declared: declared},
            materialize_registry_secrets=False,
        ),
    )
    task = list(yaml.safe_load_all(rendered))[1]
    assert task["resources"]["image_id"] == "docker:" + FIXTURE_IMAGE
    assert task["resources"]["accelerators"] == declared


def test_readiness_binds_exact_workflow_and_records_unbuilt_blocker() -> None:
    payload = json.loads(READINESS.read_text())
    assert payload["schema_version"] == "workflow-readiness/v1"
    assert (
        payload["workflow_sha256"] == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert payload["planning"]["validation"]["status"] == "verified"
    assert payload["planning"]["task_fidelity"]["status"] == "verified"
    assert payload["prerequisites"]["target_runtime"]["status"] == "blocked"


def test_pending_public_image_is_refused_and_private_reference_is_resolvable() -> None:
    with pytest.raises(
        ValueError, match="pending corresponding-source closure"
    ) as error:
        container_image_for_tool("habitat-sim")
    assert "--push" not in str(error.value)
    assert "never distributed" not in str(error.value)
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


def test_archive_redirect_is_refused_before_target_contact(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    class Opener:
        def __init__(self, handler):
            self.handler = handler

        def open(self, request, timeout):
            assert timeout == 120 and request.full_url == H.ARCHIVE_URL
            events.append("official-origin")
            with pytest.raises(H.SmokeFailure, match="redirects are not permitted"):
                self.handler.redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {},
                    "https://redirect-target.invalid/archive.zip",
                )
            events.append("refused-before-target")
            raise H.SmokeFailure("redirects are not permitted")

    def build_opener(handler):
        assert isinstance(handler, H._RefuseRedirects)
        return Opener(handler)

    monkeypatch.setattr(H.urllib.request, "build_opener", build_opener)
    with pytest.raises(H.SmokeFailure, match="redirects are not permitted"):
        H._download(tmp_path / "archive.part")
    assert events == ["official-origin", "refused-before-target"]
    assert not (tmp_path / "archive.part").exists()


def test_archive_origin_is_validated_before_open(tmp_path, monkeypatch) -> None:
    contacted = False

    def opener(*_args, **_kwargs):
        nonlocal contacted
        contacted = True
        raise AssertionError("opener must not be called")

    monkeypatch.setattr(
        H,
        "ARCHIVE_URL",
        "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",
    )
    with pytest.raises(H.SmokeFailure, match="pinned HTTPS origin"):
        H._download(tmp_path / "archive.part", opener=opener)
    assert contacted is False


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


def _cache_stat(inode, *, directory=False, **changes):
    """Construct inert metadata, never a filesystem entry."""
    values = dict(
        st_dev=1,
        st_ino=inode,
        st_uid=1000,
        st_mode=0o40700 if directory else 0o100600,
        st_nlink=2 if directory else 1,
        st_size=64,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )
    return SimpleNamespace(**(values | changes))


class _CacheObservations:
    """Model only named observations; every filesystem operation is a mock."""

    def __init__(self, monkeypatch):
        self.owner = H._RuntimeCacheOwnership(
            Path("/fixture/output"),
            ".cache",
            descriptors={"parent": 101, "cache": 202, "marker": 303},
            identities={"parent": (1, 10), "cache": (1, 20)},
            marker_sha256=hashlib.sha256(b"a" * 64).hexdigest(),
            created=True,
        )
        self.parent = _cache_stat(10, directory=True)
        self.cache = _cache_stat(20, directory=True)
        self.entries = {
            ".npa-run-owner": _cache_stat(30),
            H.SCENE_NAME: _cache_stat(40),
        }
        self.parent_names = [self.owner.name]
        self.stats = {
            101: self.parent,
            202: self.cache,
            303: self.entries[".npa-run-owner"],
        }
        self.os = self._mock_cache_os()
        monkeypatch.setattr(H, "os", self.os)

    def _mock_cache_os(self):
        """Keep mock system calls isolated from pytest's own filesystem access."""
        constants = {
            k: getattr(H.os, k)
            for k in (
                "O_RDONLY",
                "O_RDWR",
                "O_DIRECTORY",
                "O_CLOEXEC",
                "O_NOFOLLOW",
                "O_CREAT",
                "O_EXCL",
            )
        }
        return SimpleNamespace(
            **constants,
            environ={},
            geteuid=Mock(return_value=1000),
            fstat=Mock(side_effect=self.stats.__getitem__),
            stat=Mock(side_effect=self._stat),
            pread=Mock(return_value=b"a" * 64),
            listdir=Mock(side_effect=self._listdir),
            unlink=Mock(side_effect=self._unlink),
            rmdir=Mock(side_effect=self._rmdir),
            close=Mock(),
            open=Mock(side_effect=[101, 202, 303]),
            mkdir=Mock(),
            write=Mock(return_value=64),
            fsync=Mock(),
        )

    def _stat(self, name, *, dir_fd=None, follow_symlinks):
        assert follow_symlinks is False
        if name == self.owner.output_dir and dir_fd is None:
            return self.parent
        if name == self.owner.name and dir_fd == 101:
            return self.cache
        assert dir_fd == 202
        return self.entries[name]

    def _listdir(self, descriptor):
        return list(self.parent_names if descriptor == 101 else self.entries)

    def _unlink(self, name, *, dir_fd):
        assert dir_fd == 202
        del self.entries[name]

    def _rmdir(self, name, *, dir_fd):
        assert dir_fd == 101 and not self.entries
        self.parent_names.remove(name)


def test_cache_cleanup_uses_retained_descriptors_only(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    H._finish_runtime_cache(fixture.owner)
    assert fixture.owner.removed and fixture.owner.descriptors == {}
    fixture.os.open.assert_not_called()
    assert fixture.os.unlink.call_args_list == [
        ((H.SCENE_NAME,), {"dir_fd": 202}),
        ((".npa-run-owner",), {"dir_fd": 202}),
    ]
    fixture.os.rmdir.assert_called_once_with(".cache", dir_fd=101)
    assert [call.args[0] for call in fixture.os.close.call_args_list] == [303, 202, 101]


@pytest.mark.parametrize(
    "boundary", ["parent_fd", "parent_name", "cache_fd", "cache_name", "marker"]
)
def test_cache_ownership_mismatch_refuses_before_removal(monkeypatch, boundary):
    fixture = _CacheObservations(monkeypatch)
    if boundary == "parent_fd":
        fixture.stats[101] = _cache_stat(99, directory=True)
    elif boundary == "parent_name":
        fixture.parent = _cache_stat(99, directory=True)
    elif boundary == "cache_fd":
        fixture.stats[202] = _cache_stat(99, directory=True)
    elif boundary == "cache_name":
        fixture.cache = _cache_stat(99, directory=True)
    else:
        fixture.os.pread.return_value = b"b" * 64
    with pytest.raises(H.SmokeFailure) as raised:
        H._finish_runtime_cache(fixture.owner)
    assert "unresolved" in " ".join(raised.value.__notes__)
    fixture.os.unlink.assert_not_called()
    fixture.os.rmdir.assert_not_called()
    assert fixture.os.close.call_count == 3 and not fixture.owner.removed


@pytest.mark.parametrize(
    "changes", [{"st_uid": 2000}, {"st_mode": 0o40755}, {"st_nlink": 0}]
)
def test_cache_private_parent_loss_refuses_effects(monkeypatch, changes):
    fixture = _CacheObservations(monkeypatch)
    fixture.stats[101] = _cache_stat(10, directory=True, **changes)
    with pytest.raises(H.SmokeFailure, match="ownership"):
        H._finish_runtime_cache(fixture.owner)
    fixture.os.unlink.assert_not_called()
    fixture.os.rmdir.assert_not_called()


@pytest.mark.parametrize("entry", ["unknown", "directory", "nonprivate", "hardlink"])
def test_cache_unresolved_members_are_never_adopted(monkeypatch, entry):
    fixture = _CacheObservations(monkeypatch)
    if entry == "unknown":
        fixture.entries["unclaimed"] = _cache_stat(50)
    else:
        changes = {
            "directory": {"st_mode": 0o40700},
            "nonprivate": {"st_mode": 0o100644},
            "hardlink": {"st_nlink": 2},
        }[entry]
        fixture.entries[H.SCENE_NAME] = _cache_stat(40, **changes)
    with pytest.raises(H.SmokeFailure):
        H._finish_runtime_cache(fixture.owner)
    fixture.os.unlink.assert_not_called()
    fixture.os.rmdir.assert_not_called()


def test_cache_cleanup_retains_all_primary_and_release_diagnostics(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    fixture.entries[H.NAVMESH_NAME] = _cache_stat(50)
    fixture.os.unlink.side_effect = OSError("fixture unlink refused")
    fixture.os.close.side_effect = OSError("fixture close refused")
    primary = H.SmokeFailure("fixture fetch failed")
    H._finish_runtime_cache(fixture.owner, primary)
    notes = "\n".join(primary.__notes__)
    assert H.SCENE_NAME in notes and H.NAVMESH_NAME in notes
    assert "unresolved" in notes and str(primary) == "fixture fetch failed"
    assert all(
        role + " descriptor release" in notes for role in ("marker", "cache", "parent")
    )
    assert fixture.os.close.call_count == 3 and fixture.owner.descriptors == {}
    fixture.os.rmdir.assert_not_called()
    assert fixture.os.unlink.call_count == 2


def test_cache_primary_survives_unexpected_cleanup_exception(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    monkeypatch.setattr(
        H, "_remove_runtime_cache", Mock(side_effect=RuntimeError("fixture bug"))
    )
    primary = ValueError("initial failure")
    H._finish_runtime_cache(fixture.owner, primary)
    assert "RuntimeError: fixture bug" in " ".join(primary.__notes__)
    assert fixture.os.close.call_count == 3
    assert str(primary) == "initial failure" and not fixture.owner.removed


def test_cache_release_failure_blocks_success_without_cleanup_retry(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    fixture.os.close.side_effect = [OSError("fixture marker close"), None, None]
    with pytest.raises(H.SmokeFailure, match="release incomplete"):
        H._finish_runtime_cache(fixture.owner)
    assert fixture.owner.removed and fixture.os.close.call_count == 3
    assert fixture.os.rmdir.call_count == 1


def test_cache_cleanup_postcondition_is_verified_inertly(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    fixture.os.rmdir.side_effect = None
    with pytest.raises(H.SmokeFailure, match="cleanup did not complete"):
        H._finish_runtime_cache(fixture.owner)
    assert not fixture.owner.removed and fixture.os.close.call_count == 3


def _inert_main(monkeypatch):
    """Bypass all runtime effects and inject a descriptor-only ownership model."""
    fixture = _CacheObservations(monkeypatch)
    owner = fixture.owner
    monkeypatch.setattr(
        H, "_execution_identity", lambda *_args: ("fixture-run", "a" * 64)
    )
    monkeypatch.setattr(
        H,
        "_prepare_output",
        lambda _args: (owner.output_dir, owner.output_dir / owner.name, owner),
    )
    proof = Mock(return_value={"schema_version": "npa.habitat-sim.smoke.v1"})
    upload = Mock()
    monkeypatch.setattr(H, "_produce_proof", proof)
    monkeypatch.setattr(H, "_upload_directory", upload)
    argv = [
        "--output-dir",
        str(owner.output_dir),
        "--output-uri",
        "s3://fixture/output",
        "--run-id",
        "fixture-run",
        "--plan-sha256",
        "a" * 64,
    ]
    return fixture, proof, upload, argv


def test_main_preserves_primary_after_inert_cache_cleanup_failure(monkeypatch):
    fixture, proof, upload, argv = _inert_main(monkeypatch)
    primary = H.SmokeFailure("fixture fetch failed")
    proof.side_effect = primary
    fixture.os.unlink.side_effect = OSError("fixture cleanup failed")
    with pytest.raises(H.SmokeFailure) as raised:
        H.main(argv)
    assert raised.value is primary and "unresolved" in " ".join(primary.__notes__)
    upload.assert_not_called()
    assert fixture.os.close.call_count == 3


def _inert_cache_claim(monkeypatch):
    """Observe the proposed creation protocol without creating any entry."""
    fixture = _CacheObservations(monkeypatch)
    fixture.owner.name = ".habitat-scene-cache-" + "a" * 32
    fixture.parent_names.clear()
    monkeypatch.setattr(H.secrets, "token_hex", lambda size: "a" * (size * 2))
    return fixture


def test_cache_claim_is_private_child_with_held_no_follow_descriptors(monkeypatch):
    fixture = _inert_cache_claim(monkeypatch)
    owner = H._claim_runtime_cache(fixture.owner.output_dir, (1, 10))
    assert owner.output_dir / owner.name == Path("/fixture/output") / fixture.owner.name
    assert owner.descriptors == {"parent": 101, "cache": 202, "marker": 303}
    fixture.os.mkdir.assert_called_once_with(owner.name, mode=0o700, dir_fd=101)
    assert all(
        call.args[1] & fixture.os.O_NOFOLLOW for call in fixture.os.open.call_args_list
    )
    assert fixture.os.open.call_args_list[-1].args[1] & fixture.os.O_EXCL
    assert fixture.os.open.call_args_list[-1].args[2] == 0o600
    fixture.os.close.assert_not_called()
    fixture.os.unlink.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["nonempty", "parent_identity", "preexisting", "short_marker"]
)
def test_cache_partial_claim_is_preserved_without_adoption(monkeypatch, failure):
    fixture = _inert_cache_claim(monkeypatch)
    if failure == "nonempty":
        fixture.parent_names.append("not-owned")
    elif failure == "parent_identity":
        fixture.parent = _cache_stat(99, directory=True)
    elif failure == "preexisting":
        fixture.os.mkdir.side_effect = FileExistsError("fixture entry exists")
    else:
        fixture.os.write.return_value = 63
    with pytest.raises((H.SmokeFailure, FileExistsError)) as raised:
        H._claim_runtime_cache(fixture.owner.output_dir, (1, 10))
    assert "preserved, not adopted" in " ".join(raised.value.__notes__)
    fixture.os.unlink.assert_not_called()
    fixture.os.rmdir.assert_not_called()
    assert fixture.os.close.call_count == (3 if failure == "short_marker" else 1)
    if failure in {"nonempty", "parent_identity"}:
        fixture.os.mkdir.assert_not_called()


def test_prepare_output_places_cache_beneath_created_private_output(monkeypatch):
    fixture = _CacheObservations(monkeypatch)
    fixture.os.chmod = Mock()
    output = MagicMock(spec=Path)
    output.resolve.return_value = output
    output.stat.return_value = _cache_stat(10, directory=True)
    claim = Mock(return_value=fixture.owner)
    monkeypatch.setattr(H, "_claim_runtime_cache", claim)
    actual_output, cache, owner = H._prepare_output(SimpleNamespace(output_dir=output))
    output.mkdir.assert_called_once_with(parents=True, exist_ok=False, mode=0o700)
    claim.assert_called_once_with(output, (1, 10))
    assert actual_output is output and owner is fixture.owner
    assert cache is output.__truediv__.return_value
    output.__truediv__.assert_called_once_with(owner.name)


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
        "node_group_id": "node-group-fixture",
        "kubernetes_node": {
            "name": "worker-fixture",
            "provider_id": "nebius://instance-fixture",
        },
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
    image = "private.invalid/task-owned/npa-habitat-sim@sha256:" + "a" * 64
    registry_evidence = {
        "schema_version": "npa.registry.private-pull-refusal.v1",
        "registry": "private.invalid/task-owned",
        "image": image,
        "resolved_digest": "sha256:" + "a" * 64,
        "anonymous_pull_denied": True,
        "anonymous_status": 401,
        "authenticated_pull_succeeded": True,
    }
    registry_path = owner / "private-registry.json"
    registry_path.write_text(json.dumps(registry_evidence), encoding="utf-8")
    registry_path.chmod(0o600)
    return {
        "schema_version": "npa.habitat-sim.image-live.v1",
        "head": "fixture-head",
        "workflow_sha256": hashlib.sha256(WORKFLOW.read_bytes()).hexdigest(),
        "run_id": "habitat-fixture",
        "transaction_started_at": "2026-09-10T23:59:59Z",
        "registry": "private.invalid/task-owned",
        "image": image,
        "registry_evidence": {
            "path": str(registry_path),
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
        "kubernetes_context": provider["kubernetes_context"],
        "project": provider["project"],
        "project_id": provider["project_id"],
        "namespace": "habitat-fixture",
        "pod_name": "habitat-fixture-pod",
        "pod_uid": "habitat-fixture-pod-uid",
        "kubeconfig": str(owner / "kubeconfig"),
        "storage": {
            "endpoint": "https://storage.invalid",
            "bucket": "fixture-bucket",
            "prefix": "run-owned/habitat-fixture",
        },
        "target": {
            "policy": "STRICT",
            "accelerator": provider["accelerator"],
            "gpu_count": 1,
        },
        "reservation": {
            "accelerator": provider["accelerator"],
            "capacity_block_group_id": provider["capacity_block_group_id"],
            "gpu_count": 1,
            "node_group_id": provider["node_group_id"],
            "kubernetes_node": provider["kubernetes_node"],
            "provider_readback_sha256": readback,
            "provider_receipt_path": str(provider_path),
            "provider_receipt_sha256": hashlib.sha256(
                provider_path.read_bytes()
            ).hexdigest(),
            "state": provider["state"],
            "verified_at": provider["verified_at"],
        },
    }


def _bound_live_fixture(tmp_path: Path):
    import numpy as np

    receipt = _live_receipt(tmp_path)
    owner = Path(receipt["reservation"]["provider_receipt_path"]).parent
    image = receipt["image"]
    runtime_output = str(tmp_path / "runtime-output")
    plan = {
        "schema_version": "npa.habitat-sim.rendered-plan.v1",
        "workflow_name": "habitat-sim-smoke",
        "workflow_sha256": receipt["workflow_sha256"],
        "run_id": receipt["run_id"],
        "image": image,
        "namespace": receipt["namespace"],
        "pod_name": receipt["pod_name"],
        "node_name": "worker-fixture",
        "output_dir": runtime_output,
        "output_uri": "s3://fixture-bucket/run-owned/habitat-fixture/",
        "pod_command": ["python3", "-m", "npa.workflows.habitat_sim_smoke"],
        "pod_args": [
            "--output-dir",
            runtime_output,
            "--output-uri",
            "s3://fixture-bucket/run-owned/habitat-fixture/",
            "--run-id",
            "habitat-fixture",
            "--plan-sha256",
            LIVE.PLAN_PLACEHOLDER,
        ],
        "environment": {
            "NPA_TASK_IMAGE": image,
            "NPA_WORKFLOW_NAME": "habitat-sim-smoke",
            "NPA_WORKFLOW_RUN_ID": receipt["run_id"],
            "NPA_WORKFLOW_STATE": "render-traversal",
        },
        "runtime_uid": 1000,
        "gpu": {
            "accelerator": "RTX PRO 6000 Blackwell",
            "count": 1,
            "compute_capability": "12.0",
        },
        "submission_id": "skypilot-task-fixture",
    }
    setup = (
        "set -euo pipefail\n"
        "echo 'Habitat-Sim refuses a Python source overlay' >/dev/null\n"
        "test -x /opt/venv/bin/python\n"
    )
    run = (
        "set -euo pipefail\n"
        + " ".join(
            str(item)
            for item in LIVE._expected_plan_argv(plan, receipt, plan["output_uri"])
        )
        + "\n"
    )
    task = {
        "name": "render-traversal",
        "resources": {"image_id": "docker:" + image},
        "envs": plan["environment"],
        "setup": setup,
        "run": run,
    }
    skypilot_payload = yaml.safe_dump_all(
        [{"name": "habitat-sim-smoke", "execution": "serial"}, task],
        explicit_start=False,
        sort_keys=False,
    ).encode()
    skypilot_path = owner / "rendered-skypilot.yaml"
    skypilot_path.write_bytes(skypilot_payload)
    skypilot_path.chmod(0o600)
    plan.update(
        skypilot_sha256=hashlib.sha256(skypilot_payload).hexdigest(),
        setup_sha256=hashlib.sha256(setup.encode()).hexdigest(),
        run_sha256=hashlib.sha256(run.encode()).hexdigest(),
    )
    receipt["rendered_skypilot"] = {
        "path": str(skypilot_path),
        "sha256": plan["skypilot_sha256"],
    }
    plan_path = owner / "rendered-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    plan_path.chmod(0o600)
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    receipt["rendered_plan"] = {"path": str(plan_path), "sha256": plan_sha256}
    submission = {
        "schema_version": "npa.habitat-sim.submitted-task.v1",
        "task_id": plan["submission_id"],
        "task_name": task["name"],
        "skypilot_sha256": plan["skypilot_sha256"],
        "setup_sha256": plan["setup_sha256"],
        "run_sha256": plan["run_sha256"],
        "image": image,
        "pod_command": plan["pod_command"],
        "pod_args": [
            plan_sha256 if item == LIVE.PLAN_PLACEHOLDER else item
            for item in plan["pod_args"]
        ],
        "platform_environment": [],
    }
    submission_path = owner / "submitted-task.json"
    submission_path.write_text(json.dumps(submission, sort_keys=True), encoding="utf-8")
    submission_path.chmod(0o600)
    receipt["submitted_task"] = {
        "path": str(submission_path),
        "sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
    }

    outputs = tmp_path / "outputs"
    observations = outputs / "habitat-sim-observations"
    observations.mkdir(parents=True)
    frame_records = []
    for index in range(2):
        rgb = f"rgb-{index}".encode()
        depth = f"depth-{index}".encode()
        preview = f"preview-{index}".encode()
        rgb_path = observations / f"rgb-{index:04d}.png"
        depth_path = observations / f"depth-{index:04d}.npy"
        preview_path = observations / f"depth-{index:04d}.png"
        rgb_path.write_bytes(rgb)
        depth_path.write_bytes(depth)
        preview_path.write_bytes(preview)
        frame_records.append(
            {
                "index": index,
                "action": "move_forward",
                "rgb_shape": [240, 320, 4],
                "rgb_raw_sha256": hashlib.sha256(rgb + b"-raw").hexdigest(),
                "rgb_png_path": str(rgb_path.relative_to(outputs)),
                "rgb_png_media_type": "image/png",
                "rgb_png_bytes": len(rgb),
                "rgb_png_sha256": hashlib.sha256(rgb).hexdigest(),
                "depth_shape": [240, 320],
                "depth_raw_sha256": hashlib.sha256(depth + b"-raw").hexdigest(),
                "depth_npy_path": str(depth_path.relative_to(outputs)),
                "depth_npy_media_type": "application/x-npy",
                "depth_npy_bytes": len(depth),
                "depth_npy_sha256": hashlib.sha256(depth).hexdigest(),
                "depth_preview_png_path": str(preview_path.relative_to(outputs)),
                "depth_preview_png_media_type": "image/png",
                "depth_preview_png_bytes": len(preview),
                "depth_preview_png_sha256": hashlib.sha256(preview).hexdigest(),
            }
        )
    member_records = {
        H.SCENE_NAME: {
            "name": H.MEMBER_SPECS[H.SCENE_NAME]["archive_member"],
            "bytes": H.MEMBER_SPECS[H.SCENE_NAME]["bytes"],
            "compressed_bytes": 1,
            "crc32": H.MEMBER_SPECS[H.SCENE_NAME]["crc32"],
            "sha256": H.SCENE_SHA256,
        },
        H.NAVMESH_NAME: {
            "name": H.MEMBER_SPECS[H.NAVMESH_NAME]["archive_member"],
            "bytes": H.MEMBER_SPECS[H.NAVMESH_NAME]["bytes"],
            "compressed_bytes": 1,
            "crc32": H.MEMBER_SPECS[H.NAVMESH_NAME]["crc32"],
            "sha256": H.NAVMESH_SHA256,
        },
    }
    archive = {
        "url": H.ARCHIVE_URL,
        "url_role": "mutable official locator only",
        "checked_at_utc": "2026-09-11T00:00:00+00:00",
        "response_metadata": {
            "status": 200,
            "content_length": str(H.ARCHIVE_BYTES),
            "content_type": "application/zip",
            "etag": None,
            "last_modified": None,
        },
        "bytes": H.ARCHIVE_BYTES,
        "sha256": H.ARCHIVE_SHA256,
        "url_is_mutable": True,
        "zip_integrity": "pass",
        "ephemeral_copy_removed": True,
        "unrelated_members_extracted": False,
    }
    traversal = {
        "count": 2,
        "finite_depth": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([1.0, 0.0, 0.0]),
        "goal": np.array([2.0, 0.0, 0.0]),
        "displacement": 1.0,
        "geodesic": 2.0,
        "actions": ["move_forward", "move_forward"],
        "collisions": 0,
        "physics_start": 0.0,
        "physics_end": 2.0 / 60.0,
        "fps": 30.0,
        "egl": {
            "backend": "EGL",
            "display_unset": True,
            "gl": {
                "vendor": "NVIDIA Corporation",
                "renderer": "NVIDIA RTX PRO 6000 Blackwell",
                "version": "fixture",
            },
            "libraries": ["/usr/lib/libEGL.so.1", "/usr/lib/libEGL_nvidia.so.0"],
        },
        "rgb_hash": "a" * 64,
        "depth_hash": "b" * 64,
        "records": frame_records,
    }
    source_manifest = ROOT / "npa/docker/workbench/habitat-sim/source-manifest.json"
    source = {
        "repository": "https://github.com/facebookresearch/habitat-sim",
        "requested_revision": H.SOURCE_REVISION,
        "observed_revision": H.SOURCE_REVISION,
        "license": "MIT",
        "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    }
    gpu = {
        "count": 1,
        "model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "architecture": "Blackwell",
        "compute_capability": "12.0",
        "observation_source": "nvidia-smi inside the workload pod",
    }
    proof = H._proof(
        source,
        image,
        image.rsplit("@", 1)[1],
        gpu,
        archive,
        member_records,
        traversal,
        receipt["run_id"],
        plan_sha256,
    )
    proof_path = outputs / "habitat-sim-smoke.json"
    proof_path.write_text(
        json.dumps(proof, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    storage = Storage()
    publication = H._upload_directory(
        outputs,
        "s3://fixture-bucket/run-owned/habitat-fixture/",
        client=storage,
        stage_token="1" * 32,
    )
    termination_path = tmp_path / "termination.json"
    H._write_termination_receipt(publication, proof, termination_path)
    termination = json.loads(termination_path.read_text(encoding="utf-8"))
    pod = {
        "metadata": {
            "uid": receipt["pod_uid"],
            "labels": {"npa.nebius.com/task-name": submission["task_name"]},
            "annotations": {"npa.nebius.com/task-id": submission["task_id"]},
        },
        "spec": {
            "nodeName": "worker-fixture",
            "containers": [
                {
                    "name": "smoke",
                    "image": image,
                    "command": submission["pod_command"],
                    "args": submission["pod_args"],
                    "env": [
                        {"name": name, "value": value}
                        for name, value in plan["environment"].items()
                    ],
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 1000},
                }
            ],
        },
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "smoke",
                    "imageID": "docker-pullable://" + image,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "message": json.dumps(termination),
                        }
                    },
                }
            ],
        },
    }
    node = {
        "metadata": {"name": "worker-fixture"},
        "spec": {"providerID": "nebius://instance-fixture"},
    }
    return receipt, plan, plan_sha256, pod, node, storage, publication


def test_live_receipt_binds_private_image_and_strict_provider_readback(
    tmp_path,
) -> None:
    receipt = _live_receipt(tmp_path)
    LIVE._assert_private_image(receipt)
    LIVE._assert_provider_binding(receipt)

    registry_path = Path(receipt["registry_evidence"]["path"])
    registry_bytes = registry_path.read_bytes()
    registry_path.write_bytes(registry_bytes + b" ")
    with pytest.raises(AssertionError):
        LIVE._assert_private_image(receipt)
    registry_path.write_bytes(registry_bytes)

    provider_path = Path(receipt["reservation"]["provider_receipt_path"])
    provider_bytes = provider_path.read_bytes()
    provider_path.write_bytes(provider_bytes + b" ")
    with pytest.raises(AssertionError):
        LIVE._assert_provider_binding(receipt)
    provider_path.write_bytes(provider_bytes)

    private_receipt = receipt.copy()
    receipt["image"] = (
        "ghcr.io/nebius/nebius-physical-ai/npa-habitat-sim@sha256:" + "a" * 64
    )
    receipt["registry"] = "ghcr.io/nebius/nebius-physical-ai"
    with pytest.raises(AssertionError):
        LIVE._assert_private_image(receipt)

    for registry in (
        "GHCR.IO:443/nebius/nebius-physical-ai",
        "GHCR.IO.:443/nebius/nebius-physical-ai",
        "docker.io/task-owned",
        "quay.io/task-owned",
        "public.ecr.aws/task-owned",
    ):
        receipt = {
            **private_receipt,
            "registry": registry,
            "image": registry + "/npa-habitat-sim@sha256:" + "a" * 64,
        }
        with pytest.raises(AssertionError):
            LIVE._assert_private_image(receipt)


def test_provider_binding_requires_ready_transaction_fresh_readback(tmp_path) -> None:
    receipt = _live_receipt(tmp_path)
    provider_path = Path(receipt["reservation"]["provider_receipt_path"])
    provider = json.loads(provider_path.read_text(encoding="utf-8"))

    provider["state"] = receipt["reservation"]["state"] = "FAILED"
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    receipt["reservation"]["provider_receipt_sha256"] = hashlib.sha256(
        provider_path.read_bytes()
    ).hexdigest()
    with pytest.raises(AssertionError):
        LIVE._assert_provider_binding(receipt)

    freshness = tmp_path / "freshness"
    freshness.mkdir()
    receipt = _live_receipt(freshness)
    receipt["transaction_started_at"] = "2026-09-11T00:00:01Z"
    with pytest.raises(AssertionError):
        LIVE._assert_provider_binding(receipt)


@pytest.mark.parametrize(
    "invalid_timestamp",
    (
        "2026-09-11 00:00:00Z",
        "2026-09-11T00:00:00+00:00",
        "2026-9-11T00:00:00Z",
    ),
)
def test_provider_binding_rejects_non_rfc3339_utc_timestamps(
    tmp_path, invalid_timestamp
) -> None:
    receipt = _live_receipt(tmp_path)
    provider_path = Path(receipt["reservation"]["provider_receipt_path"])
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    provider["verified_at"] = invalid_timestamp
    receipt["reservation"]["verified_at"] = invalid_timestamp
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    receipt["reservation"]["provider_receipt_sha256"] = hashlib.sha256(
        provider_path.read_bytes()
    ).hexdigest()

    with pytest.raises(AssertionError):
        LIVE._assert_provider_binding(receipt)


def test_named_smoke_proof_declares_schema_at_artifact_root() -> None:
    import numpy as np

    records = {
        H.SCENE_NAME: {"bytes": 1},
        H.NAVMESH_NAME: {"bytes": 2},
    }
    traversal = {
        "count": 2,
        "finite_depth": np.array([1.0, 2.0], dtype=np.float32),
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([1.0, 0.0, 0.0]),
        "goal": np.array([2.0, 0.0, 0.0]),
        "displacement": 1.0,
        "geodesic": 2.0,
        "actions": ["move_forward", "move_forward"],
        "collisions": 0,
        "physics_start": 0.0,
        "physics_end": 0.2,
        "fps": 10.0,
        "egl": {"backend": "EGL"},
        "rgb_hash": "a" * 64,
        "depth_hash": "b" * 64,
        "records": [],
    }
    proof = H._proof(
        {"revision": H.SOURCE_REVISION},
        "private.invalid/task/image@sha256:" + "c" * 64,
        "sha256:" + "c" * 64,
        {"model": "fixture", "architecture": "Blackwell", "count": 1},
        {"bytes": 3, "sha256": "d" * 64},
        records,
        traversal,
        "fixture-run",
        "d" * 64,
    )

    assert proof["schema_version"] == "npa.habitat-sim.smoke.v1"
    assert "schema_version" not in proof["scene"]


def test_live_selector_binds_pod_node_to_provider_receipt(tmp_path) -> None:
    receipt = _live_receipt(tmp_path)
    provider = LIVE._assert_provider_binding(receipt)
    pod = {"spec": {"nodeName": "worker-fixture"}}
    node = {
        "metadata": {"name": "worker-fixture"},
        "spec": {"providerID": "nebius://instance-fixture"},
    }
    LIVE._assert_pod_provider_node(pod, node, provider)

    pod["spec"]["nodeName"] = "other-worker"
    with pytest.raises(AssertionError):
        LIVE._assert_pod_provider_node(pod, node, provider)
    pod["spec"]["nodeName"] = "worker-fixture"
    node["spec"]["providerID"] = "nebius://other-instance"
    with pytest.raises(AssertionError):
        LIVE._assert_pod_provider_node(pod, node, provider)


def _validate_bound_fixture(tmp_path: Path, monkeypatch):
    receipt, _plan, _plan_hash, pod, node, storage, _publication = _bound_live_fixture(
        tmp_path
    )
    monkeypatch.setattr(LIVE, "_storage_client", lambda _receipt: storage)
    provider = LIVE._assert_provider_binding(receipt)
    plan, plan_hash, submission = LIVE._rendered_plan(receipt, provider)
    LIVE._assert_pod_provider_node(pod, node, provider)
    image_id, termination = LIVE._assert_pod_completion(
        pod, receipt, plan, plan_hash, submission
    )
    ready, manifest, client = LIVE._publication(receipt, termination)
    proof, payload = LIVE._proof(receipt, ready, client)
    LIVE._assert_proof(proof, receipt, provider, plan_hash, manifest)
    LIVE._assert_observation_readback(proof, receipt, manifest, client)
    return {
        "receipt": receipt,
        "provider": provider,
        "plan": plan,
        "submission": submission,
        "plan_hash": plan_hash,
        "pod": pod,
        "node": node,
        "storage": storage,
        "ready": ready,
        "manifest": manifest,
        "proof": proof,
        "payload": payload,
        "image_id": image_id,
        "termination": termination,
    }


def test_live_selector_accepts_only_fully_bound_mock_evidence(
    tmp_path, monkeypatch
) -> None:
    validated = _validate_bound_fixture(tmp_path, monkeypatch)
    assert validated["image_id"].endswith("@sha256:" + "a" * 64)
    assert validated["proof"]["exit_status"] == 0


@pytest.mark.parametrize("field", ["command", "args", "image", "uid", "plan"])
def test_live_selector_rejects_unbound_pod_execution(tmp_path, field) -> None:
    receipt, plan, plan_hash, pod, _node, _storage, _publication = _bound_live_fixture(
        tmp_path
    )
    _bound_plan, _bound_hash, submission = LIVE._rendered_plan(
        receipt, LIVE._assert_provider_binding(receipt)
    )
    if field == "command":
        pod["spec"]["containers"][0]["command"] = ["python3", "--help"]
    elif field == "args":
        pod["spec"]["containers"][0]["args"] = ["--help"]
    elif field == "image":
        pod["spec"]["containers"][0]["image"] = (
            "private.invalid/task/npa-habitat-sim@sha256:" + "b" * 64
        )
    elif field == "uid":
        pod["spec"]["containers"][0]["securityContext"]["runAsUser"] = 0
    else:
        terminated = pod["status"]["containerStatuses"][0]["state"]["terminated"]
        message = json.loads(terminated["message"])
        message["rendered_plan_sha256"] = "b" * 64
        terminated["message"] = json.dumps(message)
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, receipt, plan, plan_hash, submission)


def test_live_selector_rejects_unbound_render_submit_and_environment(tmp_path) -> None:
    receipt, plan, plan_hash, pod, _node, _storage, _publication = _bound_live_fixture(
        tmp_path
    )
    provider = LIVE._assert_provider_binding(receipt)
    skypilot_path = Path(receipt["rendered_skypilot"]["path"])
    original = skypilot_path.read_bytes()
    skypilot_path.write_bytes(original + b"# hostile\n")
    with pytest.raises(AssertionError):
        LIVE._rendered_plan(receipt, provider)
    skypilot_path.write_bytes(original)

    bound_plan, bound_hash, submission = LIVE._rendered_plan(receipt, provider)
    pod["metadata"]["annotations"]["npa.nebius.com/task-id"] = "other-task"
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, receipt, bound_plan, bound_hash, submission)
    pod["metadata"]["annotations"]["npa.nebius.com/task-id"] = submission["task_id"]

    pod["spec"]["containers"][0]["env"].append(
        {"name": "PYTHONPATH", "value": "/" + "tmp/overlay"}
    )
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, receipt, bound_plan, bound_hash, submission)
    pod["spec"]["containers"][0]["env"].pop()
    pod["spec"]["containers"][0]["envFrom"] = [{"secretRef": {"name": "undeclared"}}]
    with pytest.raises(AssertionError):
        LIVE._assert_pod_completion(pod, receipt, bound_plan, bound_hash, submission)


def test_live_selector_allows_only_declared_secret_reference_without_value(
    tmp_path,
) -> None:
    receipt, _plan, _plan_hash, pod, _node, _storage, _publication = (
        _bound_live_fixture(tmp_path)
    )
    submission_path = Path(receipt["submitted_task"]["path"])
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    secret = {
        "name": "AWS_ACCESS_KEY_ID",
        "valueFrom": {
            "secretKeyRef": {"name": "run-owned-storage", "key": "access-key"}
        },
    }
    submission["platform_environment"] = [secret]
    submission_path.write_text(json.dumps(submission, sort_keys=True), encoding="utf-8")
    receipt["submitted_task"]["sha256"] = hashlib.sha256(
        submission_path.read_bytes()
    ).hexdigest()
    pod["spec"]["containers"][0]["env"].append(secret)
    plan, plan_hash, bound_submission = LIVE._rendered_plan(
        receipt, LIVE._assert_provider_binding(receipt)
    )
    LIVE._assert_pod_completion(pod, receipt, plan, plan_hash, bound_submission)


def test_live_selector_rejects_wrong_node_gpu_workflow_and_extra_proof(
    tmp_path, monkeypatch
) -> None:
    receipt, _plan, _plan_hash, pod, node, storage, _publication = _bound_live_fixture(
        tmp_path
    )
    monkeypatch.setattr(LIVE, "_storage_client", lambda _receipt: storage)
    provider = LIVE._assert_provider_binding(receipt)
    plan, plan_hash, submission = LIVE._rendered_plan(receipt, provider)
    _image_id, termination = LIVE._assert_pod_completion(
        pod, receipt, plan, plan_hash, submission
    )
    ready, manifest, client = LIVE._publication(receipt, termination)
    proof, _payload = LIVE._proof(receipt, ready, client)

    node["metadata"]["name"] = "wrong-worker"
    with pytest.raises(AssertionError):
        LIVE._assert_pod_provider_node(pod, node, provider)
    node["metadata"]["name"] = "worker-fixture"

    proof["observed_gpu"]["model"] = "NVIDIA B200"
    with pytest.raises(AssertionError):
        LIVE._assert_proof(proof, receipt, provider, plan_hash, manifest)
    proof["observed_gpu"]["model"] = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

    proof["execution_binding"]["workflow_name"] = "fabricated-workflow"
    with pytest.raises(AssertionError):
        LIVE._assert_proof(proof, receipt, provider, plan_hash, manifest)
    proof["execution_binding"]["workflow_name"] = "habitat-sim-smoke"
    proof["caller_selected_extra"] = True
    with pytest.raises(AssertionError):
        LIVE._assert_proof(proof, receipt, provider, plan_hash, manifest)


def test_live_selector_rejects_fabricated_proof_and_incomplete_inventory(
    tmp_path, monkeypatch
) -> None:
    receipt, plan, plan_hash, pod, _node, storage, publication = _bound_live_fixture(
        tmp_path
    )
    monkeypatch.setattr(LIVE, "_storage_client", lambda _receipt: storage)
    _bound_plan, _bound_hash, submission = LIVE._rendered_plan(
        receipt, LIVE._assert_provider_binding(receipt)
    )
    _image_id, termination = LIVE._assert_pod_completion(
        pod, receipt, plan, plan_hash, submission
    )
    ready, _manifest, client = LIVE._publication(receipt, termination)
    proof_key = (receipt["storage"]["bucket"], ready["proof_key"])
    storage.objects[proof_key] += b" "
    with pytest.raises(AssertionError):
        LIVE._proof(receipt, ready, client)

    storage.objects[proof_key] = storage.objects[proof_key][:-1]
    observation_key = next(
        row["key"] for row in publication["objects"] if row["path"].endswith(".npy")
    )
    storage.objects.pop((receipt["storage"]["bucket"], observation_key))
    with pytest.raises(AssertionError):
        LIVE._publication(receipt, termination)


def test_live_selector_ignores_stage_without_ready_marker(
    tmp_path, monkeypatch
) -> None:
    receipt, plan, plan_hash, pod, _node, storage, publication = _bound_live_fixture(
        tmp_path
    )
    monkeypatch.setattr(LIVE, "_storage_client", lambda _receipt: storage)
    _bound_plan, _bound_hash, submission = LIVE._rendered_plan(
        receipt, LIVE._assert_provider_binding(receipt)
    )
    _image_id, termination = LIVE._assert_pod_completion(
        pod, receipt, plan, plan_hash, submission
    )
    storage.objects.pop((receipt["storage"]["bucket"], publication["ready_key"]))
    with pytest.raises(KeyError):
        LIVE._publication(receipt, termination)


def test_live_receipt_reader_rejects_symlink_and_foreign_owner(
    tmp_path, monkeypatch
) -> None:
    owner = tmp_path / "owner-only"
    owner.mkdir(mode=0o700)
    receipt = owner / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    receipt.chmod(0o600)
    linked = owner / "linked.json"
    linked.symlink_to(receipt)

    with pytest.raises(OSError):
        LIVE._private_json(linked)

    current_uid = LIVE.os.geteuid()
    monkeypatch.setattr(LIVE.os, "geteuid", lambda: current_uid + 1)
    with pytest.raises(AssertionError):
        LIVE._private_json(receipt)


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


def test_interrupted_storage_publication_deletes_only_exact_staged_keys(
    tmp_path,
) -> None:
    _bound_live_fixture(tmp_path)
    storage = Storage(fail_after=8)
    ready_key = "run-owned/interrupted/habitat-sim-publication-ready.json"
    with pytest.raises(
        H.SmokeFailure,
        match=f"unresolved object ownership after failed write: {ready_key}",
    ) as raised:
        H._upload_directory(
            tmp_path / "outputs",
            "s3://fixture-bucket/run-owned/interrupted/",
            client=storage,
            stage_token="2" * 32,
        )
    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "fixture interrupted upload"
    assert storage.objects == {}
    assert len(storage.deleted) == 8
    assert all(
        key.startswith("run-owned/interrupted/.staging/" + "2" * 32 + "/")
        for _bucket, key in storage.deleted
    )
    assert ("fixture-bucket", ready_key) not in storage.deleted


def test_transactional_publication_records_verified_media_and_commits_last(
    tmp_path,
) -> None:
    *_prefix, storage, publication = _bound_live_fixture(tmp_path)
    assert publication["object_count"] == 7
    assert {row["media_type"] for row in publication["objects"]} == {
        "application/json",
        "application/x-npy",
        "image/png",
    }
    ready = storage.objects[("fixture-bucket", publication["ready_key"])]
    assert hashlib.sha256(ready).hexdigest() == publication["ready_sha256"]
    assert storage.uploads == publication["object_count"] + 2


def test_ready_marker_race_preserves_the_unowned_object(tmp_path) -> None:
    _bound_live_fixture(tmp_path)

    class ReadyMarkerRace(Storage):
        def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
            if Key.endswith("/" + H.READY_MARKER_NAME):
                self.objects[(Bucket, Key)] = b"foreign-ready-marker\n"
                raise OSError("fixture immutable object already exists")
            return super().put_object(
                Bucket=Bucket,
                Key=Key,
                Body=Body,
                ContentType=ContentType,
                IfNoneMatch=IfNoneMatch,
            )

    storage = ReadyMarkerRace()
    ready_key = "run-owned/race/habitat-sim-publication-ready.json"
    with pytest.raises(OSError, match="immutable object already exists"):
        H._upload_directory(
            tmp_path / "outputs",
            "s3://fixture-bucket/run-owned/race/",
            client=storage,
            stage_token="3" * 32,
        )
    assert storage.objects == {("fixture-bucket", ready_key): b"foreign-ready-marker\n"}
    assert ("fixture-bucket", ready_key) not in storage.deleted


def test_ambiguous_ready_write_is_reported_without_deleting_shared_marker(
    tmp_path,
) -> None:
    _bound_live_fixture(tmp_path)

    class AmbiguousReady(Storage):
        def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
            if Key.endswith("/" + H.READY_MARKER_NAME):
                self.objects[(Bucket, Key)] = bytes(Body)
                raise OSError("fixture lost acknowledgement")
            return super().put_object(
                Bucket=Bucket,
                Key=Key,
                Body=Body,
                ContentType=ContentType,
                IfNoneMatch=IfNoneMatch,
            )

        def get_object(self, *, Bucket, Key):
            if Key.endswith("/" + H.READY_MARKER_NAME):
                raise OSError("fixture readback unavailable")
            return super().get_object(Bucket=Bucket, Key=Key)

    storage = AmbiguousReady()
    ready_key = "run-owned/ambiguous/habitat-sim-publication-ready.json"
    with pytest.raises(H.SmokeFailure, match="unresolved object ownership"):
        H._upload_directory(
            tmp_path / "outputs",
            "s3://fixture-bucket/run-owned/ambiguous/",
            client=storage,
            stage_token="4" * 32,
        )
    assert ("fixture-bucket", ready_key) in storage.objects
    assert ("fixture-bucket", ready_key) not in storage.deleted


def test_main_removes_run_cache_before_publication_commit(monkeypatch) -> None:
    fixture, _proof, upload, argv = _inert_main(monkeypatch)

    def publication(*_args, **_kwargs):
        assert fixture.owner.removed and fixture.owner.descriptors == {}
        assert not fixture.entries and not fixture.parent_names
        return {}

    upload.side_effect = publication
    assert H.main(argv) == 0
    upload.assert_called_once()


def test_main_refuses_publication_after_inert_cleanup_failure(monkeypatch):
    fixture, _proof, upload, argv = _inert_main(monkeypatch)
    fixture.os.pread.return_value = b"b" * 64
    with pytest.raises(H.SmokeFailure, match="marker identity"):
        H.main(argv)
    upload.assert_not_called()
    fixture.os.unlink.assert_not_called()
    assert fixture.os.close.call_count == 3


def _mock_publication_stage(monkeypatch):
    """Stage inert bytes through a memory-only client; never contact storage."""

    def stage(client, bucket, prefix, manifest_key, output_dir, ledger):
        del output_dir
        proof_key = prefix + "/habitat-sim-smoke.json"
        payload = b"{}\n"
        for key in (proof_key, manifest_key):
            H._put_owned_object(
                client,
                bucket,
                key,
                payload,
                "application/json",
                key,
                ledger,
                exclusive_key=True,
            )
        return [
            H._object_record(
                "habitat-sim-smoke.json", proof_key, payload, "application/json"
            )
        ], payload

    monkeypatch.setattr(H, "_stage_publication", stage)


def test_success_callback_follows_exact_ready_marker_readback(tmp_path, monkeypatch):
    _mock_publication_stage(monkeypatch)
    storage = Storage()
    verified = []
    original = H._verify_object

    def verify(client, bucket, key, payload, label):
        original(client, bucket, key, payload, label)
        verified.append(key)

    monkeypatch.setattr(H, "_verify_object", verify)
    receipts = []

    def acknowledge(publication):
        assert verified[-1] == publication["ready_key"]
        actual = storage.objects[("fixture-bucket", publication["ready_key"])]
        assert hashlib.sha256(actual).hexdigest() == publication["ready_sha256"]
        receipts.append(publication)

    result = H._upload_directory(
        tmp_path,
        "s3://fixture-bucket/run-owned/",
        client=storage,
        stage_token="5" * 32,
        after_commit=acknowledge,
    )
    assert receipts == [result]
    assert storage.deleted == []


@pytest.mark.parametrize("boundary", ["put", "head", "readback"])
def test_failed_ready_commit_never_emits_success_receipt(
    tmp_path, monkeypatch, boundary
):
    _mock_publication_stage(monkeypatch)
    storage = Storage()
    method = {"put": "put_object", "head": "head_object", "readback": "get_object"}[
        boundary
    ]
    original = getattr(storage, method)
    failure = OSError("fixture ready " + boundary + " unavailable")

    def refuse(**kwargs):
        if kwargs["Key"].endswith("/" + H.READY_MARKER_NAME):
            raise failure
        return original(**kwargs)

    monkeypatch.setattr(storage, method, refuse)
    acknowledge = Mock()
    with pytest.raises((H.SmokeFailure, OSError)) as raised:
        H._upload_directory(
            tmp_path,
            "s3://fixture-bucket/run-owned/",
            client=storage,
            stage_token="5" * 32,
            after_commit=acknowledge,
        )
    assert raised.value is failure or raised.value.__cause__ is failure
    acknowledge.assert_not_called()
    assert storage.objects == {}


def test_cleanup_aggregates_all_diagnostics_and_unresolved_ownership(monkeypatch):
    client = Mock()
    client.delete_object.side_effect = [OSError("second delete failed"), None]
    client.list_objects_v2.side_effect = [
        OSError("first verify failed"),
        {
            "Contents": [{"Key": "owned/second"}],
            "IsTruncated": False,
        },
    ]
    ledger = H._ObjectWriteLedger(
        owned=["owned/first", "owned/second"], unresolved=["unresolved/ready"]
    )
    initiating = RuntimeError("fixture publication interrupted")
    with pytest.raises(H.SmokeFailure) as raised:
        H._publication_failure(client, "fixture-bucket", ledger, initiating)
    assert raised.value.__cause__ is initiating
    for diagnostic in (
        "RuntimeError: fixture publication interrupted",
        "delete owned/second: OSError: second delete failed",
        "verify owned/first: OSError: first verify failed",
        "verify owned/second: object remains",
        "unresolved object ownership after failed write: unresolved/ready",
    ):
        assert diagnostic in str(raised.value)
    assert [call.kwargs["Key"] for call in client.delete_object.call_args_list] == [
        "owned/second",
        "owned/first",
    ]
    assert [
        call.kwargs["Prefix"] for call in client.list_objects_v2.call_args_list
    ] == [
        "owned/first",
        "owned/second",
    ]


def test_successful_cleanup_preserves_the_original_publication_exception(monkeypatch):
    client = Mock()
    client.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    initiating = ValueError("fixture publication failure")
    ledger = H._ObjectWriteLedger(owned=["owned/only"])
    # No secondary failure is manufactured: the caller re-raises its original error.
    assert H._publication_failure(client, "fixture-bucket", ledger, initiating) is None
    client.delete_object.assert_called_once_with(
        Bucket="fixture-bucket", Key="owned/only"
    )
