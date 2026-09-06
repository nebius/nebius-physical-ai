"""Synthetic independently interposed output races; no actual image is read."""

import json
import os
from pathlib import Path

import pytest

import test_image_byte_public_entrypoint as ENTRY

prepared = ENTRY.prepared


def test_policy_refuses_same_inode_receipt_payload_changed_before_success(
    prepared, monkeypatch
):
    W, _P, argv, _out, *_ = prepared
    original = W.write_private_json

    def altered(directory, name, value):
        result = original(directory, name, value)
        if name == "public-policy-acceptance.json":
            path = directory / name
            before = path.stat().st_ino
            changed = json.loads(path.read_bytes())
            changed["accepted"] = False
            path.write_text(json.dumps(changed))
            assert path.stat().st_ino == before
        return result

    monkeypatch.setattr(W, "write_private_json", altered)
    assert W.main(argv) == 1


def test_policy_refuses_output_directory_replaced_after_report_check(
    prepared, monkeypatch
):
    W, P, argv, _out, *_ = prepared
    original = P.FreshPolicyReview.accept_fresh_scan

    def replaced(self, report, directory):
        saved = directory.with_name(directory.name + "-original")
        directory.rename(saved)
        directory.mkdir(mode=0o700)
        # The attacker copies only exact original inputs; all byte hashes remain
        # correct while the held root output directory identity is different.
        for file in saved.iterdir():
            if file.is_file():
                (directory / file.name).write_bytes(file.read_bytes())
                (directory / file.name).chmod(0o600)
        return original(self, report, directory)

    monkeypatch.setattr(P.FreshPolicyReview, "accept_fresh_scan", replaced)
    assert W.main(argv) == 1


@pytest.mark.parametrize("error_kind", ["ordinary", "oserror", "cancellation"])
def test_exception_after_acceptance_cannot_return_success(
    prepared, monkeypatch, capsys, error_kind
):
    core, policy, argv, _out, *_ = prepared
    original = policy.FreshPolicyReview.accept_fresh_scan
    reached = []

    def fail_after_acceptance(self, report, directory):
        original(self, report, directory)
        assert self.accepted is True
        reached.append(True)
        if error_kind == "cancellation":
            raise KeyboardInterrupt
        if error_kind == "oserror":
            raise OSError("synthetic-output-failure")
        raise core.ScanError("synthetic-output-failure")

    monkeypatch.setattr(
        policy.FreshPolicyReview, "accept_fresh_scan", fail_after_acceptance
    )
    assert core.main(argv) == 1
    assert reached == [True]
    assert "gate passed" not in capsys.readouterr().out


@pytest.mark.parametrize("mutation", ["different-inode", "permissions", "hardlink"])
def test_receipt_metadata_mutation_cannot_return_success(
    prepared, monkeypatch, mutation
):
    core, _policy, argv, _out, *_ = prepared
    original = core.write_private_json
    reached = []

    def mutate(directory, name, result):
        original_fingerprint = original(directory, name, result)
        if name == "public-policy-acceptance.json":
            path = directory / name
            if mutation == "different-inode":
                old = path.with_name("retained-original-receipt")
                path.rename(old)
                path.write_bytes(old.read_bytes())
                path.chmod(0o600)
                assert path.stat().st_ino != old.stat().st_ino
            elif mutation == "permissions":
                path.chmod(0o640)
            else:
                os.link(path, path.with_name("synthetic-extra-link"))
                assert path.stat().st_nlink == 2
            reached.append(True)
        return original_fingerprint

    monkeypatch.setattr(core, "write_private_json", mutate)
    assert core.main(argv) == 1
    assert reached == [True]


def test_raw_report_mutation_is_still_fatal(prepared, monkeypatch):
    core, _policy, argv, _out, *_ = prepared
    original = core.write_private_json

    def mutate(directory, name, result):
        fingerprint = original(directory, name, result)
        if name == "report.json":
            path = directory / name
            raw = json.loads(path.read_bytes())
            raw["complete"] = False
            path.write_text(json.dumps(raw))
        return fingerprint

    monkeypatch.setattr(core, "write_private_json", mutate)
    assert core.main(argv) == 1


def test_unchanged_policy_success_preserves_failed_raw_verdict(prepared):
    core, _policy, argv, output, *_ = prepared
    assert core.main(argv) == 0
    raw = json.loads((output / "report.json").read_bytes())
    receipt = json.loads((output / "public-policy-acceptance.json").read_bytes())
    assert raw["complete"] is True and raw["valid"] is False
    assert receipt["accepted"] is True
    assert receipt["raw_scan_valid"] is False


def test_writer_closes_directory_descriptor_when_pending_unlink_fails(
    prepared, monkeypatch
):
    core, _policy, argv, output, *_ = prepared
    opened = []
    real_directory_fd = core.directory_fd
    real_close = core.os.close
    real_unlink = core.os.unlink

    def tracked(path):
        descriptor = real_directory_fd(path)
        opened.append(descriptor)
        return descriptor

    def refuse_pending(name, *args, **kwargs):
        if name == "synthetic-receipt.json.pending":
            raise OSError("synthetic-unlink-failure")
        return real_unlink(name, *args, **kwargs)

    with core.authorized_roots(Path(argv[1]), Path(argv[3])):
        output.mkdir(mode=0o700)
        monkeypatch.setattr(core, "directory_fd", tracked)
        monkeypatch.setattr(core.os, "unlink", refuse_pending)
        try:
            with pytest.raises(OSError):
                core.write_private_json(
                    output, "synthetic-receipt.json", {"synthetic": True}
                )
            assert opened
            for descriptor in opened:
                with pytest.raises(OSError):
                    core.os.fstat(descriptor)
        finally:
            # Close only descriptors opened by this synthetic writer invocation.
            for descriptor in opened:
                try:
                    core.os.fstat(descriptor)
                except OSError:
                    continue
                real_close(descriptor)
