"""Exercise signed-review authority and phase boundaries with hermetic inputs."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / "npa/scripts"))
from image_byte_scan import private_review_gate as G  # noqa: E402

SOURCE = "a" * 40
KEY = bytes(range(32))
KEY_PIN = G.W.sha(KEY)
VERIFIER = "b" * 64


def environment():
    return {"GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        "GITHUB_WORKFLOW_REF": "nebius/nebius-physical-ai/.github/workflows/publish-public-images.yml@refs/heads/synthetic-review",
        "GITHUB_WORKFLOW_SHA": SOURCE, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": "build-development"}


def write(path, value):
    path.write_bytes(G.W.canonical(value))
    path.chmod(0o600)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    phase = tmp_path / "pre"
    phase.mkdir(mode=0o700)
    inbox = phase / "review-inbox"
    inbox.mkdir(mode=0o700)
    identity = G._identity(SOURCE, "pre", environment())
    envelope = {"schema_version": G.SCHEMA, "algorithm": "Ed25519", "identity": identity,
        "context": {"synthetic_private_context": "never exported"},
        "public_key_sha256": KEY_PIN, "manifest_sha256": "c" * 64,
        "review_sha256": "d" * 64, "verifier_sha256": VERIFIER, "scan_command_sha256": "f" * 64}
    signed = {"envelope": envelope, "public_key_hex": KEY.hex(), "signature_hex": "e" * 128}
    path = inbox / "signed-envelope.json"
    write(path, signed)
    packets = []
    monkeypatch.setattr(G.S, "verified_binary", lambda _: {"path": "synthetic", "sha256": VERIFIER})
    monkeypatch.setattr(G, "_run_verifier", lambda binary, packet: packets.append(packet))
    with G.W.authorized_roots(tmp_path, CHECKOUT):
        yield SimpleNamespace(phase=phase, path=path, signed=signed, identity=identity,
                              packets=packets, root=tmp_path)


def verify(bundle, *, identity=None, pin=KEY_PIN):
    return G._signed_bundle(bundle.phase, pin, identity or bundle.identity, bundle.root / "tools.json")


def test_exact_envelope_passes_domain_and_full_canonical_bytes_to_crypto(bundle):
    envelope, binding = verify(bundle)
    assert envelope == bundle.signed["envelope"]
    assert binding["sha256"] == G.W.sha(bundle.path.read_bytes())
    assert bundle.packets == [KEY + bytes.fromhex("e" * 128) + G.DOMAIN + G.W.canonical(envelope)]


@pytest.mark.parametrize("pin", ["", "1" * 64, "a" * 63, "A" * 64, None, True])
def test_dispatch_key_pin_cannot_be_inferred_from_bundle(bundle, pin):
    with pytest.raises(G.W.ScanError):
        verify(bundle, pin=pin)
    assert not bundle.packets


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other"), ("algorithm", "none"), ("public_key_sha256", "0" * 64),
    ("manifest_sha256", ""), ("review_sha256", None), ("verifier_sha256", "f" * 64),
])
def test_unsigned_or_unbound_envelope_fields_refuse(bundle, field, value):
    bundle.signed["envelope"][field] = value
    write(bundle.path, bundle.signed)
    with pytest.raises(G.W.ScanError):
        verify(bundle)
    assert not bundle.packets


@pytest.mark.parametrize("field,value", [
    ("phase", "post"), ("source_sha", "0" * 40), ("run_id", "456"),
    ("run_attempt", "2"), ("repository", "synthetic/untrusted"), ("tool", "openpi"),
    ("job", "resolve"), ("workflow_sha", "f" * 40), ("workflow_ref", "other"),
])
def test_signature_cannot_replay_across_execution_identity(bundle, field, value):
    bundle.signed["envelope"]["identity"][field] = value
    expected = G._identity(SOURCE, "pre", environment())
    write(bundle.path, bundle.signed)
    with pytest.raises(G.W.ScanError):
        verify(bundle, identity=expected)


@pytest.mark.parametrize("field,value", [
    ("public_key_hex", "00" * 31), ("public_key_hex", "gg" * 32),
    ("signature_hex", "00" * 63), ("signature_hex", "00" * 65),
    ("signature_hex", "AA" * 64), ("signature_hex", None),
])
def test_crypto_inputs_require_exact_lengths_and_encoding(bundle, field, value):
    bundle.signed[field] = value
    write(bundle.path, bundle.signed)
    with pytest.raises(G.W.ScanError):
        verify(bundle)
    assert not bundle.packets


def test_duplicate_json_key_and_extra_envelope_field_refuse(bundle):
    bundle.path.write_bytes(b'{"envelope":{},"envelope":{}}')
    with pytest.raises(G.W.ScanError):
        verify(bundle)
    bundle.signed["envelope"]["automatic_approval"] = True
    write(bundle.path, bundle.signed)
    with pytest.raises(G.W.ScanError):
        verify(bundle)


def test_bad_crypto_verdict_cannot_fall_back_to_native_acceptance(bundle, monkeypatch):
    def refused(*_):
        raise G.W.ScanError("private_review_signature")
    monkeypatch.setattr(G, "_run_verifier", refused)
    with pytest.raises(G.W.ScanError, match="private_review_signature"):
        verify(bundle)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public_mode"])
def test_private_bundle_transport_rejects_aliases_and_open_permissions(bundle, kind):
    if kind == "public_mode":
        bundle.path.chmod(0o644)
    else:
        original = bundle.path.with_name("original.json")
        bundle.path.rename(original)
        if kind == "symlink":
            bundle.path.symlink_to(original)
        else:
            os.link(original, bundle.path)
    with pytest.raises(G.W.ScanError):
        verify(bundle)


def test_bundle_replacement_during_signature_check_refuses(bundle, monkeypatch):
    def replace(*_):
        changed = copy.deepcopy(bundle.signed)
        changed["envelope"]["review_sha256"] = "f" * 64
        write(bundle.path, changed)
    monkeypatch.setattr(G, "_run_verifier", replace)
    with pytest.raises(G.W.ScanError):
        verify(bundle)


@pytest.mark.parametrize("raw_exit", [0, 1])
def test_no_private_pin_preserves_exact_native_command_and_exit(tmp_path, monkeypatch, raw_exit):
    args = G._arguments(["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT),
        "--authorization", str(tmp_path / "authorization.json"), "--output-dir", str(tmp_path / "scan"),
        "--public-native-policy", str(tmp_path / "catalog.json"), "--public-native-policy-sha256", "b" * 64,
        "--source-sha", SOURCE, "--phase", "pre"])
    calls = []
    monkeypatch.setattr(G.W, "main", lambda argv: calls.append(argv) or raw_exit)
    monkeypatch.setattr(G, "_accept", lambda *args: pytest.fail("private review must not run"))
    assert G._run(args) == raw_exit
    assert calls == [["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT),
        "--authorization", str(tmp_path / "authorization.json"), "--output-dir", str(tmp_path / "scan"),
        "--public-native-policy", str(tmp_path / "catalog.json"), "--public-native-policy-sha256", "b" * 64]]


@pytest.mark.parametrize("key,value", [
    ("GITHUB_WORKFLOW_SHA", "b" * 40), ("GITHUB_REPOSITORY", "synthetic/other"),
    ("GITHUB_RUN_ID", "0"), ("GITHUB_RUN_ATTEMPT", "-1"), ("GITHUB_JOB", "resolve"),
    ("GITHUB_WORKFLOW_REF", "nebius/nebius-physical-ai/.github/workflows/other.yml@refs/heads/main"),
])
def test_current_identity_requires_trusted_workflow_fields(key, value):
    selected = environment()
    selected[key] = value
    with pytest.raises(G.W.ScanError):
        G._identity(SOURCE, "pre", selected)


def test_missing_actual_scan_never_waits_for_or_issues_approval(bundle, monkeypatch):
    monkeypatch.setattr(G, "_request", lambda *args: pytest.fail("no request before complete scan"))
    with pytest.raises(OSError):
        G._accept(bundle.phase, KEY_PIN, bundle.identity, bundle.root / "tools.json", 1)


def test_private_gate_cli_error_does_not_print_sensitive_arguments(capsys):
    assert G.main(["--synthetic-private-argument"]) == 1
    assert capsys.readouterr().out == "image byte publication gate failed\n"


def test_cancellation_after_child_result_joins_separate_native_session(tmp_path):
    trusted = tmp_path / "trusted"
    script = trusted / "npa/scripts/scan_image_bytes.py"
    script.parent.mkdir(parents=True)
    phase = tmp_path / "analysis/pre"
    phase.mkdir(parents=True)
    marker = tmp_path / "owned-process.json"
    # The synthetic scanner publishes its child identity before a late cancel.
    script.write_text(
        "import json,os,signal,subprocess,sys\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import signal;signal.pause()'],start_new_session=True)\n"
        "def stop(signum,frame):\n"
        " child.terminate();child.wait();raise SystemExit(1)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'child':child.pid}}))\n"
        "os.kill(os.getppid(),signal.SIGTERM)\n"
        "signal.pause()\n"
    )
    with pytest.raises(G.W.ScanError), G.W.cancellation_scope():
        G._scan_command(trusted, phase, [])
    child = json.loads(marker.read_text())["child"]
    assert not Path(f"/proc/{child}").exists()
    assert not (phase / "raw-scan-exit.json").exists()
    assert not (phase / "private-review-request.json").exists()


@pytest.mark.parametrize("exit_code,complete,interrupted", [
    (True, True, False), (2, True, False), (-15, True, False),
    (1, False, False), (1, True, True), (1, 1, False), (1, True, 0),
])
def test_command_receipt_cannot_claim_interruption_as_completed(bundle, exit_code, complete, interrupted):
    write(bundle.phase / "raw-scan-exit.json",
          {"exit_code": exit_code, "completed": complete, "interrupted": interrupted})
    with pytest.raises(G.W.ScanError):
        G._command_exit(bundle.phase)


def test_review_gate_normalizes_real_verifier_receipt_alias_error(tmp_path, monkeypatch, capsys):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(receipt)
    monkeypatch.setattr(G, "_run", lambda _: G.S.verified_binary(alias))
    monkeypatch.setattr(G, "_arguments", lambda _: None)
    assert G.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == "image byte publication gate failed\n"
    assert captured.err == "" and str(alias) not in captured.out


@pytest.mark.parametrize("tracking_id", [None, "", "synthetic-runner-process-scope"])
def test_actual_scan_child_preserves_only_runner_tracking(tmp_path, monkeypatch, tracking_id):
    monkeypatch.delenv("RUNNER_TRACKING_ID", raising=False)
    if tracking_id is not None:
        monkeypatch.setenv("RUNNER_TRACKING_ID", tracking_id)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-not-for-child")
    monkeypatch.setenv("CUSTOMER_DENYLIST", "synthetic-not-for-child")
    monkeypatch.setenv("UNRELATED_PARENT_VARIABLE", "synthetic-not-for-child")
    trusted = tmp_path / "trusted"
    script = trusted / "npa/scripts/scan_image_bytes.py"
    script.parent.mkdir(parents=True)
    script.write_text("import json,os\nprint(json.dumps(dict(os.environ)))\n")
    phase = tmp_path / "analysis/pre"
    phase.mkdir(parents=True)
    assert G._scan_command(trusted, phase, []) == 0
    child_environment = json.loads((phase / "raw-scan.stdout.log").read_text())
    assert child_environment.get("RUNNER_TRACKING_ID") == (tracking_id or None)
    assert child_environment["PATH"] == os.defpath
    # CPython may add its locale-coercion setting after exec.
    assert set(child_environment) <= {"PATH", "LC_CTYPE", "RUNNER_TRACKING_ID"}
