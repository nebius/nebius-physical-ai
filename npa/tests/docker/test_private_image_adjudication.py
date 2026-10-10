"""Exercise the hosted review transport using real local children and private files."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

import test_private_image_qualification as T

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "npa/scripts"))
import private_image_adjudication as H  # noqa: E402

export, private_root = T.export, T.private_root

Q = H.Q


def request_body(selector, scan, review):
    return {
        "schema_version": "npa.private-image-adjudication.v1",
        "image_manifest_sha256": selector,
        "scan_run_id": "123-1",
        "scan_receipt_sha256": Q._sha(scan),
        "scan_receipt_bytes": len(scan),
        "retention_sha256": "a" * 64,
        "review_tar_sha256": Q._sha(review),
        "review_tar_bytes": len(review),
        "disposition_manifest_sha256": "b" * 64,
        "independent_review_sha256": "c" * 64,
        "workspace_bytes": 8192,
    }


@pytest.fixture
def transport_case(export, private_root, monkeypatch):
    scan, review = b"synthetic retained scan", b"synthetic independent review"
    body = request_body(export[1], scan, review)
    payload = Q._json_bytes(body)
    selector = Q._sha(payload)
    request_dir = private_root / Q.ADJUDICATION_ROOT / selector
    request_dir.mkdir(parents=True, mode=0o700)
    Q._write(request_dir / "request.json", payload)
    Q._write(request_dir / "review.tar", review)
    original = private_root / Q.RECEIPT_ROOT / export[1] / body["scan_run_id"]
    descriptor = Q._directory(original, create=True)
    os.close(descriptor)
    Q._write(original / "result.tar", scan)
    code = Path(Q.__file__).read_text()
    monkeypatch.setattr(
        Q,
        "_remote_command",
        lambda _ssh, *args: [sys.executable, "-I", "-c", code, "remote", *args],
    )
    monkeypatch.setattr(
        Q,
        "_clean_environment",
        lambda **_kwargs: {"HOME": str(private_root), "PATH": os.environ["PATH"]},
    )
    return selector, body, scan, review


def test_actual_remote_protocol_retrieves_only_exact_owned_request_and_receipt(
    transport_case, private_root
):
    selector, body, scan, review = transport_case
    H._fetch([], selector, "request", private_root / "received-request.json")
    assert json.loads((private_root / "received-request.json").read_bytes()) == body
    for role, data in (("receipt", scan), ("review", review)):
        target = private_root / ("received-" + role)
        H._fetch([], selector, role, target, size=len(data), digest=Q._sha(data))
        assert target.read_bytes() == data


@pytest.mark.parametrize(
    "change",
    ["run_path", "wrong_receipt", "negative_bytes", "bool_bytes", "extra_policy"],
)
def test_request_refuses_scope_escape_and_changed_contract(transport_case, change):
    _, request, *_ = transport_case
    if change == "run_path":
        request["scan_run_id"] = "../../other"
    elif change == "wrong_receipt":
        request["scan_receipt_sha256"] = "not-a-hash"
    elif change == "negative_bytes":
        request["scan_receipt_bytes"] = -1
    elif change == "bool_bytes":
        request["workspace_bytes"] = True
    else:
        request["policy"] = "synthetic forbidden input"
    data = Q._json_bytes(request)
    with pytest.raises(Q._QualificationError):
        Q._adjudication_request(data, Q._sha(data))


def tar_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, kind, body in members:
            entry = tarfile.TarInfo(name)
            entry.type, entry.size = kind, len(body)
            archive.addfile(entry, io.BytesIO(body))
    return output.getvalue()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/escape", tarfile.REGTYPE),
        ("manifest.json", tarfile.SYMTYPE),
        ("manifest.json", tarfile.LNKTYPE),
        ("manifest.json", tarfile.FIFOTYPE),
        ("policy.json", tarfile.REGTYPE),
        ("nested/manifest.json", tarfile.REGTYPE),
    ],
)
def test_review_unpack_refuses_paths_and_nonregular_members(private_root, name, kind):
    path = private_root / "review.tar"
    Q._write(path, tar_bytes([(name, kind, b"")]))
    with pytest.raises(Q._QualificationError):
        H._unpack(path, private_root / "review", review=True)
    assert not (private_root / "escape").exists()


def test_review_unpack_preserves_exact_bodies_and_refuses_duplicate(private_root):
    path = private_root / "review.tar"
    Q._write(
        path,
        tar_bytes(
            [
                ("manifest.json", tarfile.REGTYPE, b"one"),
                ("review.json", tarfile.REGTYPE, b"two"),
            ]
        ),
    )
    names = H._unpack(path, private_root / "review", review=True)
    assert names == {"manifest.json", "review.json"}
    assert (private_root / "review/manifest.json").read_bytes() == b"one"
    duplicate = private_root / "duplicate.tar"
    Q._write(
        duplicate,
        tar_bytes(
            [
                ("manifest.json", tarfile.REGTYPE, b"one"),
                ("manifest.json", tarfile.REGTYPE, b"replacement"),
            ]
        ),
    )
    with pytest.raises(Q._QualificationError):
        H._unpack(duplicate, private_root / "duplicate", review=True)
    assert (private_root / "duplicate/manifest.json").read_bytes() == b"one"


def test_hosted_adjudication_is_default_branch_only_and_never_uploads_private_artifacts():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/private-image-adjudication.yml").read_bytes()
    )
    job = workflow["jobs"]["adjudicate"]
    assert "github.event.repository.default_branch" in job["if"]
    assert workflow["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "ubuntu-latest"
    steps = job["steps"]
    assert not any("upload-artifact" in step.get("uses", "") for step in steps)
    source = next(
        step for step in steps if step.get("with", {}).get("path") == "scanner"
    )
    assert source["with"]["ref"] == Q.SCANNER_REVISION
    caller = steps[-1]
    assert caller["env"]["REQUEST_SHA256"] == "$" + "{{ inputs.request_sha256 }}"
    assert "--request-sha256" in caller["run"]
    assert "private_image_adjudication.py" in caller["run"]


@pytest.mark.parametrize("signals", [0, 1, 2])
def test_actual_receipt_child_and_late_signals_preserve_failed_acceptance(
    transport_case, private_root, monkeypatch, capsys, signals
):
    selector, request, *_ = transport_case
    root = private_root / "hosted"
    root.mkdir(mode=0o700)
    args = SimpleNamespace(scanner_root=private_root, request_sha256=selector, run_id="124-1")

    def complete(*_args):
        Q._write(root / "request.json", Q._json_bytes(request))
        return {"status": "passed", "accepted": True}

    original = H._retain

    def retain(*arguments):
        for _ in range(signals):
            os.kill(os.getpid(), signal.SIGTERM)
        return original(*arguments)

    monkeypatch.setattr(H, "_steps", complete)
    monkeypatch.setattr(H, "_retain", retain)
    assert H._execute_adjudication(args, root, []) == int(bool(signals))
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["accepted"] is (not signals)
    receipt = private_root / Q.RECEIPT_ROOT / request["image_manifest_sha256"] / "124-1/result.tar"
    assert Q._sha(receipt.read_bytes()) == summary["receipt_sha256"]
    if signals:
        assert summary["failure_code"] == "qualification_cancelled"


def test_real_upload_child_failure_cannot_report_acceptance(
    transport_case, private_root, monkeypatch, capsys
):
    selector, request, *_ = transport_case
    root = private_root / "hosted"
    root.mkdir(mode=0o700)
    args = SimpleNamespace(scanner_root=private_root, request_sha256=selector, run_id="125-1")

    def complete(*_args):
        Q._write(root / "request.json", Q._json_bytes(request))
        return {"status": "passed", "accepted": True}

    monkeypatch.setattr(H, "_steps", complete)
    monkeypatch.setattr(Q, "_remote_command", lambda *_: [sys.executable, "-c", "raise SystemExit(7)"])
    assert H._execute_adjudication(args, root, []) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["accepted"] is False
    assert summary["failure_code"] == "private_receipt_not_retained"
    assert "receipt_sha256" not in summary


def test_malformed_tar_is_sanitized_and_original_failure_retained(
    transport_case, private_root, monkeypatch, capsys
):
    selector, request, *_ = transport_case
    root = private_root / "hosted"
    root.mkdir(mode=0o700)
    args = SimpleNamespace(scanner_root=private_root, request_sha256=selector, run_id="126-1")

    def malformed(*_args):
        Q._write(root / "request.json", Q._json_bytes(request))
        Q._write(root / "bad.tar", b"synthetic-private-text")
        H._unpack(root / "bad.tar", root / "review", review=True)

    monkeypatch.setattr(H, "_steps", malformed)
    assert H._execute_adjudication(args, root, []) == 1
    output = capsys.readouterr().out
    summary = json.loads(output.splitlines()[-1])
    assert summary["accepted"] is False
    assert "synthetic-private-text" not in output
    receipt = private_root / Q.RECEIPT_ROOT / request["image_manifest_sha256"] / "126-1/result.tar"
    with tarfile.open(receipt) as archive:
        assert json.load(archive.extractfile("summary.json"))["accepted"] is False


def test_review_unpack_refuses_extended_metadata(private_root):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        entry = tarfile.TarInfo("manifest.json")
        entry.pax_headers = {"comment": "synthetic unreviewed metadata"}
        entry.size = 2
        archive.addfile(entry, io.BytesIO(b"{}"))
    path = private_root / "extended.tar"
    Q._write(path, output.getvalue())
    with pytest.raises(Q._QualificationError):
        H._unpack(path, private_root / "review", review=True)
