"""Integrity and genuine native-output contracts for customer runtime delivery."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from npa.orchestration.npa_workflow import robotwin_customer as customer
from npa.orchestration.npa_workflow import robotwin_preflight as preflight

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "npa/docker/workbench/robotwin"
LOCK = DIRECTORY / "runtime-lock.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch = load("robotwin_fetch")
collector = load("robotwin_collect")


def test_complete_runtime_lock_binds_provider_artifacts_and_correct_cudnn():
    lock = json.loads(LOCK.read_bytes())
    artifacts = fetch.validate_artifacts(lock)
    assert len(artifacts) == 474
    assert sum(item["size_bytes"] for item in artifacts) > 5_000_000_000
    assert (
        next(item for item in artifacts if item["name"] == "nvidia-cudnn-cu12")[
            "version"
        ]
        == "9.7.1.26"
    )
    assert lock["access"]["terms_without_vendor_gate"] == list(preflight.CUSTOMER_TERMS)
    assert lock["runtime_delivery"]["live_capability_status"] == "not-yet-executed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "../escape"),
        ("url", "http://files.pythonhosted.org/a"),
        ("url", "https://untrusted.example/a"),
        ("sha256", "a"),
        ("size_bytes", -1),
        ("kind", "shell"),
    ],
)
def test_invalid_artifact_lock_refuses_without_files(tmp_path, field, value):
    lock = json.loads(LOCK.read_bytes())
    lock["runtime_artifacts"][0][field] = value
    with pytest.raises(fetch.RuntimeFailure):
        fetch.validate_artifacts(lock)
    assert list(tmp_path.iterdir()) == []


def test_download_hash_mismatch_never_exposes_an_installer_input(tmp_path, monkeypatch):
    response = io.BytesIO(b"corrupt")
    response.geturl = lambda: "https://files.pythonhosted.org/a"
    monkeypatch.setattr(
        fetch, "build_opener", lambda *_: SimpleNamespace(open=lambda _: response)
    )
    path = tmp_path / "a.whl"
    with pytest.raises(fetch.RuntimeFailure, match="integrity-mismatch"):
        fetch.fetch(
            {
                "url": "https://files.pythonhosted.org/a",
                "size_bytes": 7,
                "sha256": "a" * 64,
            },
            path,
        )
    assert list(tmp_path.iterdir()) == []


def test_download_exact_bytes_are_published_after_verification(tmp_path, monkeypatch):
    response = io.BytesIO(b"verified")
    response.geturl = lambda: "https://files.pythonhosted.org/a"
    monkeypatch.setattr(
        fetch, "build_opener", lambda *_: SimpleNamespace(open=lambda _: response)
    )
    path = tmp_path / "a.whl"
    fetch.fetch(
        {
            "url": "https://files.pythonhosted.org/a",
            "size_bytes": 8,
            "sha256": hashlib.sha256(b"verified").hexdigest(),
        },
        path,
    )
    assert path.read_bytes() == b"verified"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("version", ["1.2.3-4", "7:1.2.3-4"])
def test_offline_apt_cache_preserves_verified_bytes_and_encodes_epoch(
    tmp_path, version
):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    source = downloads / "example_1.2.3-4_amd64.deb"
    source.write_bytes(b"verified-deb-bytes")
    cache = tmp_path / "apt-cache"
    debs = fetch.stage_apt_archives(
        [
            {
                "kind": "deb",
                "name": "example",
                "version": version,
                "filename": source.name,
            }
        ],
        downloads,
        cache,
    )
    assert debs == [str(source)]
    assert cache.stat().st_mode & 0o777 == 0o700
    expected = cache / f"example_{version.replace(':', '%3a')}_amd64.deb"
    assert list(cache.iterdir()) == [expected]
    assert expected.read_bytes() == source.read_bytes() == b"verified-deb-bytes"


@pytest.mark.parametrize(
    ("field", "value"), [("name", "../escape"), ("version", "../escape")]
)
def test_offline_apt_cache_refuses_escaping_metadata(tmp_path, field, value):
    item = {
        "kind": "deb",
        "name": "example",
        "version": "1",
        "filename": "example_1_all.deb",
    }
    item[field] = value
    cache = tmp_path / "apt-cache"
    with pytest.raises(fetch.RuntimeFailure, match="apt-cache-filename-invalid"):
        fetch.stage_apt_archives([item], tmp_path, cache)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("member", ["/escape", "root/../../escape", "root/dir\\escape"])
def test_cuda_archive_refuses_escaping_member(tmp_path, member):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as writer:
        entry = tarfile.TarInfo(member)
        entry.size = 1
        writer.addfile(entry, io.BytesIO(b"a"))
    with pytest.raises(fetch.RuntimeFailure, match="archive-member-invalid"):
        fetch.extract_tar(archive, tmp_path / "cuda")
    assert not (tmp_path / "escape").exists()


def test_cuda_archive_rejects_escaping_link(tmp_path):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as writer:
        entry = tarfile.TarInfo("root/link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../escape"
        writer.addfile(entry)
    with pytest.raises(fetch.RuntimeFailure):
        fetch.extract_tar(archive, tmp_path / "cuda")


def test_asset_archive_rejects_link(tmp_path):
    archive = tmp_path / "embodiments.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        info = zipfile.ZipInfo("embodiments/link")
        info.external_attr = 0o120777 << 16
        writer.writestr(info, "../../escape")
    with pytest.raises(fetch.RuntimeFailure, match="member-type-invalid"):
        fetch.extract_assets(archive, tmp_path / "assets", "embodiments")


def write_asset_member(writer, name, content=b"asset", mode=0o100644):
    info = zipfile.ZipInfo(name)
    info.external_attr = mode << 16
    writer.writestr(info, content)


def test_asset_archive_ignores_only_paired_appledouble_metadata(tmp_path):
    archive = tmp_path / "embodiments.zip"
    header = b"\x00\x05\x16\x07\x00\x02\x00\x00"
    with zipfile.ZipFile(archive, "w") as writer:
        write_asset_member(writer, "embodiments/", b"", 0o40755)
        write_asset_member(writer, "embodiments/robot.urdf")
        write_asset_member(writer, "__MACOSX/._embodiments", header)
        write_asset_member(writer, "__MACOSX/embodiments/._robot.urdf", header)
    destination = tmp_path / "assets"
    fetch.extract_assets(archive, destination, "embodiments")
    assert (destination / "embodiments/robot.urdf").read_bytes() == b"asset"
    assert sorted(
        str(path.relative_to(destination)) for path in destination.rglob("*")
    ) == ["embodiments", "embodiments/robot.urdf"]


@pytest.mark.parametrize(
    ("name", "content", "mode"),
    [
        ("__MACOSX/embodiments/._robot.urdf", b"not-appledouble", 0o100644),
        (
            "__MACOSX/embodiments/._robot.urdf",
            b"\x00\x05\x16\x07\x00\x01\x00\x00",
            0o100644,
        ),
        ("__MACOSX/embodiments/._robot.urdf", None, 0o120777),
        ("__MACOSX/embodiments/._robot.urdf", None, 0o40755),
        ("__MACOSX/embodiments/robot.urdf", None, 0o100644),
        ("__MACOSX/embodiments/._missing.urdf", None, 0o100644),
        ("__MACOSX/other/._robot.urdf", None, 0o100644),
        ("__MACOSX/embodiments/../._robot.urdf", None, 0o100644),
        ("other/robot.urdf", None, 0o100644),
    ],
)
def test_asset_metadata_never_weakens_archive_boundaries(tmp_path, name, content, mode):
    archive = tmp_path / "embodiments.zip"
    if content is None:
        content = b"\x00\x05\x16\x07\x00\x02\x00\x00"
    with zipfile.ZipFile(archive, "w") as writer:
        write_asset_member(writer, "embodiments/robot.urdf")
        write_asset_member(writer, name, content, mode)
    with pytest.raises(fetch.RuntimeFailure):
        fetch.extract_assets(archive, tmp_path / "assets", "embodiments")


@pytest.mark.parametrize("metadata", [False, True])
def test_asset_duplicate_file_still_refuses(tmp_path, metadata):
    archive = tmp_path / "embodiments.zip"
    name = "__MACOSX/embodiments/._robot.urdf" if metadata else "embodiments/robot.urdf"
    content = b"\x00\x05\x16\x07\x00\x02\x00\x00" if metadata else b"asset"
    with zipfile.ZipFile(archive, "w") as writer:
        if metadata:
            write_asset_member(writer, "embodiments/robot.urdf")
        write_asset_member(writer, name, content)
        with pytest.warns(UserWarning, match="Duplicate name"):
            write_asset_member(writer, name, content)
    with pytest.raises((fetch.RuntimeFailure, FileExistsError)):
        fetch.extract_assets(archive, tmp_path / "assets", "embodiments")


@pytest.mark.parametrize("directory_first", [False, True])
def test_asset_file_directory_collision_still_refuses(tmp_path, directory_first):
    archive = tmp_path / "embodiments.zip"
    members = [("embodiments/robot", 0o100644), ("embodiments/robot/", 0o40755)]
    if directory_first:
        members.reverse()
    with zipfile.ZipFile(archive, "w") as writer:
        for name, mode in members:
            write_asset_member(writer, name, mode=mode)
    with pytest.raises(FileExistsError):
        fetch.extract_assets(archive, tmp_path / "assets", "embodiments")


def test_runtime_child_environment_excludes_authorization_and_credentials(tmp_path):
    env = fetch.runtime_environment(
        tmp_path,
        {
            "AWS_SECRET_ACCESS_KEY": "secret-sentinel",
            "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1": "authorization-sentinel",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "PYTHONPATH": "/injection",
        },
    )
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1" not in env
    assert "PYTHONPATH" not in env
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert env["TORCH_CUDA_ARCH_LIST"] == "12.0"


def request():
    return preflight.CustomerAuthorizationRequest(
        customer.ISSUER,
        "synthetic-customer",
        "synthetic-run",
        preflight.RUNTIME_LOCK_SHA256,
        customer.ACTIVITY,
        tuple((term["id"], term["url"]) for term in preflight.CUSTOMER_TERMS),
    )


def decision():
    expected = request()
    return {
        "schema_version": customer.SCHEMA,
        "issuer": customer.ISSUER,
        "customer_scope_id": expected.customer_scope_id,
        "run_id": expected.run_id,
        "runtime_manifest_sha256": expected.runtime_manifest_sha256,
        "issued_at": "2000-01-01T00:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
        "decision": "accepted",
        "intended_activity": customer.ACTIVITY,
        "terms": list(preflight.CUSTOMER_TERMS),
        "assertion_id": "customer-terminal-synthetic-test",
        "nonce": "0" * 64,
    }


def write_decision(tmp_path, updates=None):
    tmp_path.chmod(0o700)
    value = {**decision(), **(updates or {})}
    path = tmp_path / "synthetic-decision.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def test_customer_decision_consumes_once_and_matches_existing_boundary(tmp_path):
    path = write_decision(tmp_path)
    boundary = customer.CustomerTerminalBoundary(path)
    assertion = boundary.consume_once(request())
    verified = preflight._canonical_customer_authorization(
        assertion,
        customer_scope_id=request().customer_scope_id,
        run_id=request().run_id,
        expected_issuer=customer.ISSUER,
        context=b"synthetic",
    )
    assert verified.sha256
    with pytest.raises(customer.CustomerDecisionError, match="replayed"):
        boundary.consume_once(request())


@pytest.mark.parametrize(
    "updates",
    [
        {"decision": "declined"},
        {"expires_at": "2001-01-01T00:00:00Z"},
        {"run_id": "other"},
        {"customer_scope_id": "other"},
        {"runtime_manifest_sha256": "0" * 64},
        {"terms": []},
        {"intended_activity": "commercial-service"},
    ],
)
def test_invalid_customer_decision_refuses_before_consumption(tmp_path, updates):
    path = write_decision(tmp_path, updates)
    with pytest.raises(customer.CustomerDecisionError):
        customer.CustomerTerminalBoundary(path).consume_once(request())
    assert list(tmp_path.iterdir()) == [path]


def test_customer_decision_missing_refuses_without_writes(tmp_path):
    tmp_path.chmod(0o700)
    with pytest.raises(customer.CustomerDecisionError, match="receipt-unreadable"):
        customer.CustomerTerminalBoundary(tmp_path / "absent").consume_once(request())
    assert list(tmp_path.iterdir()) == []


def test_customer_decline_command_has_no_receipt_side_effect(tmp_path, capsys):
    path = tmp_path / "declined.json"
    result = customer.record_decision(
        lock_path=LOCK,
        run_id="synthetic-run",
        customer_scope_id="synthetic-customer",
        expires_at="2099-01-01T00:00:00Z",
        decision="decline",
        receipt=path,
    )
    assert result == 78
    assert not path.exists()
    assert "declined" in capsys.readouterr().out


def test_customer_accept_requires_actual_terminal_before_receipt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(customer.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(
        customer.CustomerDecisionError, match="customer-terminal-required"
    ):
        customer.record_decision(
            lock_path=LOCK,
            run_id="synthetic-run",
            customer_scope_id="synthetic-customer",
            expires_at="2099-01-01T00:00:00Z",
            decision="accept",
            receipt=tmp_path / "never.json",
        )
    assert list(tmp_path.iterdir()) == []


def native_fixture(output, *, corrupt_action=False, static=False, video_frames=4):
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    base = output / "native/demo_clean/beat_block_hammer/aloha_agilex"
    (base / "data").mkdir(parents=True)
    (base / "video").mkdir()
    frames = []
    for index in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :32] = (40 + (0 if static else index * 40), 100, 180)
        frame[:, 32:] = (180, 40, 100)
        frames.append(frame)
    with h5py.File(base / "data/episode_0000000.hdf5", "w") as file:
        file.attrs["source_format"] = "RoboTwin"
        for name in (
            "left_arm_joint_states",
            "right_arm_joint_states",
            "left_ee_joint_states",
            "right_ee_joint_states",
        ):
            states = np.arange(4, dtype=np.float32)[:, None]
            file.create_dataset("state/" + name, data=states[:-1])
            file.create_dataset(
                "action/" + name, data=states[1:] + (2 if corrupt_action else 0)
            )
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            encoded = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in frames[:3]]
            file.create_dataset(
                "vision/" + camera + "/colors", data=encoded, dtype="S4096"
            )
    writer = cv2.VideoWriter(
        str(base / "video/episode_0000000.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (64, 64),
    )
    assert writer.isOpened()
    for frame in frames[:video_frames]:
        writer.write(frame)
    writer.release()


def test_native_validation_decodes_video_and_aligns_state_action(tmp_path):
    native_fixture(tmp_path)
    result = collector.validate_native(tmp_path)
    assert result["state_action_pairs"] == 3
    assert result["decoded_video_frames"] == 4
    assert len(list((tmp_path / "frames").glob("*.png"))) == 3


@pytest.mark.parametrize(
    "options", [{"corrupt_action": True}, {"static": True}, {"video_frames": 2}]
)
def test_native_validation_rejects_broken_or_static_evidence(tmp_path, options):
    native_fixture(tmp_path, **options)
    with pytest.raises(ValueError):
        collector.validate_native(tmp_path)


def test_runtime_failure_retains_owner_only_process_log(tmp_path, monkeypatch):
    env = fetch.runtime_environment(tmp_path, {})

    def process(argv, *, stdout, **kwargs):
        stdout.write(b"synthetic compiler diagnosis\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(fetch.subprocess, "run", process)
    with pytest.raises(fetch.RuntimeFailure, match="private-log-retained"):
        fetch.command(["compiler", "source"], env=env)
    log = tmp_path / "logs/command-0001.log"
    assert log.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "logs").stat().st_mode & 0o777 == 0o700
    assert b"synthetic compiler diagnosis" in log.read_bytes()


def test_preprovision_probe_checks_current_manifest_before_network(monkeypatch):
    monkeypatch.setattr(
        customer,
        "build_opener",
        lambda *_: pytest.fail("network before manifest validation"),
    )
    with pytest.raises(customer.CustomerDecisionError, match="manifest-mismatch"):
        customer.probe_runtime_inputs(LOCK, expected_sha256="0" * 64)


def test_preprovision_probe_reads_one_byte_and_persists_nothing(tmp_path, monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size):
            assert size == 1
            return b"v"

    def opener(*_):
        def open_request(request):
            assert request.headers["Range"] == "bytes=0-0"
            calls.append(request.full_url)
            return Response()

        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(customer, "build_opener", opener)
    result = customer.probe_runtime_inputs(
        LOCK, expected_sha256=preflight.RUNTIME_LOCK_SHA256
    )
    assert result == {"payloads_probed": 478}
    assert len(calls) == len(set(calls)) == 478
    assert list(tmp_path.iterdir()) == []
