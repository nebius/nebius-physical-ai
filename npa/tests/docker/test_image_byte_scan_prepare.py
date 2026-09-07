"""Hermetic preparation checks; these never install Go or native dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / "npa/scripts"))
from image_byte_scan import core as W, prepare as P  # noqa: E402


def test_policy_preflight_needs_no_roots_or_tools_and_preserves_no_secret(monkeypatch, capsys):
    marker = "synthetic-private-policy-value"
    monkeypatch.setenv("CUSTOMER_DENYLIST", marker)
    monkeypatch.delenv("INFRA_DENYLIST", raising=False)
    monkeypatch.setattr(P, "dependencies", lambda *_: pytest.fail("unexpected dependency work"))
    assert P.main(["check-policy", "--policy-mode", "ci-regex"]) == 0
    output = capsys.readouterr()
    assert marker not in output.out and output.err == ""
    receipt = json.loads(output.out)
    assert receipt["customer"] == "configured" and receipt["infra"] == "not_configured"


@pytest.mark.parametrize("pattern", [None, "", "  ", "["])
def test_policy_preflight_refuses_missing_invalid_config_without_diagnostics(monkeypatch, capsys, pattern):
    monkeypatch.delenv("CUSTOMER_DENYLIST", raising=False)
    if pattern is not None:
        monkeypatch.setenv("CUSTOMER_DENYLIST", pattern)
    assert P.main(["check-policy"]) == 1
    output = capsys.readouterr()
    assert output.out == "image byte preparation failed\n" and output.err == ""


def test_authorize_rejects_missing_policy_before_output_or_dependency_access(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CUSTOMER_DENYLIST", raising=False)
    monkeypatch.setattr(P, "tools_bindings", lambda *_: pytest.fail("unexpected tool access"))
    output = tmp_path / "output"
    args = ["authorize", "--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT), "--output-dir", str(output),
            "--tools-receipt", "unread", "--native-receipt", "unread", "--archive", "unread", "--verification-report", "unread", "--expected-image-id", "unread"]
    assert P.main(args) == 1 and not output.exists()
    assert capsys.readouterr().out == "image byte preparation failed\n"


def test_invalid_cli_never_echoes_value(capsys):
    assert P.main(["check-policy", "--unrecognized", "synthetic-private-argument"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "image byte preparation failed\n" and captured.err == ""


def test_analysis_output_cannot_enter_trusted_checkout(tmp_path):
    with pytest.raises(W.ScanError, match="analysis_root_inside_build_context"):
        with W.authorized_roots(tmp_path, tmp_path):
            pytest.fail("invalid roots accepted")


def test_save_private_policy_has_exact_hash_and_no_overwrite(tmp_path):
    with W.authorized_roots(tmp_path, CHECKOUT):
        result = P.save_bytes(tmp_path, "policy.json", b'{"synthetic":true}')
        assert W.bound_json(result) == {"synthetic": True}
        assert (tmp_path / "policy.json").stat().st_mode & 0o077 == 0
        with pytest.raises(FileExistsError):
            P.save_bytes(tmp_path, "policy.json", b"changed")


def test_scanner_cli_invalid_input_retains_private_incomplete_report(tmp_path, capsys):
    auth = tmp_path / "authorization.json"
    auth.write_text("{}")
    auth.chmod(0o600)
    output = tmp_path / "scan"
    result = W.main(["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT), "--authorization", str(auth), "--output-dir", str(output)])
    assert result == 1
    captured = capsys.readouterr()
    assert captured.out == "complete image byte scan failed\n" and captured.err == ""
    report = json.loads((output / "report.json").read_bytes())
    assert not report["complete"] and not report["valid"]
    assert (output / "report.json").stat().st_mode & 0o077 == 0
    assert not list(output.glob("*.pending"))


def test_prepare_exact_image_and_policy_inputs_are_bound(tmp_path, monkeypatch):
    from test_image_byte_scan import FakeDetector, fixture
    with W.authorized_roots(tmp_path, CHECKOUT):
        authorization = fixture(tmp_path)
        monkeypatch.setattr(P, "tools_bindings", lambda _: (authorization["helper"], authorization["config"]))
        monkeypatch.setattr(P, "native_engine", lambda _: {"kind": "synthetic-unexecuted-binding"})
        monkeypatch.setattr(W, "Detector", FakeDetector)
        monkeypatch.setenv("CUSTOMER_DENYLIST", "synthetic-required-policy")
        monkeypatch.delenv("INFRA_DENYLIST", raising=False)
        monkeypatch.setattr(W, "input_snapshots", lambda _: [])  # Native import has a distinct mandatory real check.
        directory = tmp_path / "prepared"
        directory.mkdir(mode=0o700)
        args = SimpleNamespace(tools_receipt=authorization["tools_receipt"]["path"], native_receipt=None, archive=authorization["archive"]["path"],
                               verification_report=authorization["verification_report"]["path"], expected_image_id=authorization["expected_image_id"],
                               policy_mode="ci-regex", literal_inventory=None)
        result = P.authorize(args, directory)
        assert result["archive"] == authorization["archive"]
        assert W.bound_json(result["confidentiality"])["customer_pattern"] == "synthetic-required-policy"
        assert result["sources"] == W.source_bindings()
        assert json.loads((directory / "authorization.json").read_bytes()) == result
        assert all(path.stat().st_mode & 0o077 == 0 for path in directory.iterdir())


def test_no_secret_environment_is_inherited_by_detector_fixture(tmp_path, monkeypatch):
    from test_image_byte_scan import fixture
    with W.authorized_roots(tmp_path, CHECKOUT):
        authorization = fixture(tmp_path)
        monkeypatch.setenv("CUSTOMER_DENYLIST", "synthetic-secret")
        def inspect(*args, **kwargs):
            assert kwargs["env"] == {"PATH": os.defpath}
            assert kwargs["start_new_session"] is True and len(kwargs["pass_fds"]) == 2
            raise OSError("synthetic spawn failure")
        monkeypatch.setattr(W.subprocess, "Popen", inspect)
        with pytest.raises(OSError):
            W.Detector(authorization, tmp_path / "helper-stderr.jsonl")


def test_explicit_native_command_fails_when_tools_are_missing(tmp_path, capsys):
    from image_byte_scan import real_helper_checks as N
    output = tmp_path / "native-checks"
    assert N.main(["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT),
                   "--tools-receipt", str(tmp_path / "absent-tools.json"), "--native-receipt", str(tmp_path / "absent-native.json"),
                   "--output-dir", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "native image byte checks failed\n" and captured.err == ""
    result = json.loads((output / "native-checks-failure.json").read_bytes())
    assert result["passed"] is False and result["synthetic_only"] is True


def test_cancellation_scope_restores_handler_and_marks_controlled_failure():
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(W.ScanError, match="scan_cancelled"), W.cancellation_scope():
        os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_analysis_ancestor_cannot_create_outputs_inside_trusted_tree(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir(mode=0o700)
    with W.authorized_roots(tmp_path, trusted):
        with pytest.raises(W.ScanError, match="output_directory_scope"):
            W.create_output(trusted / "forbidden-output")
        assert not (trusted / "forbidden-output").exists()
        private_file = trusted / "private.json"
        private_file.write_bytes(b"{}")
        private_file.chmod(0o600)
        with pytest.raises(W.ScanError, match="input_outside_authorized_roots"):
            W.open_private_fd(private_file)
