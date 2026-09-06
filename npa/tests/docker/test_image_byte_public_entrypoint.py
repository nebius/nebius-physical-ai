"""Exercise the real CLI with synthetic trusted source and a framing oracle.

The separate pinned native gate provides native image-security validation.
"""

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/tests/docker"))
sys.path.insert(0, str(ROOT / "npa/scripts"))
import image_byte_scan  # noqa: E402
import test_image_byte_adjudication as CASE  # noqa: E402
import test_image_byte_scan as F  # noqa: E402


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(image_byte_scan, name.rsplit(".", 1)[1], module, raising=False)
    return module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    trusted = tmp_path / "synthetic-trusted-source"
    trusted.mkdir()
    analysis = tmp_path / "private-analysis"
    analysis.mkdir(mode=0o700)
    original = ROOT / "npa/scripts/image_byte_scan"
    destination = trusted / "npa/scripts/image_byte_scan"
    destination.mkdir(parents=True)
    for source in original.rglob("*"):
        if (
            source.is_file()
            and "__pycache__" not in source.parts
            and source.suffix in {".py", ".go", ".mod", ".sum", ".json", ".md"}
            or source.is_file()
            and source.name.startswith("LICENSE")
        ):
            target = destination / source.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name in [
        ".gitleaks.toml",
        "npa/scripts/scan_image_bytes.py",
        "npa/tests/docker/test_image_byte_go_build.py",
    ]:
        target = trusted / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    W = load(monkeypatch, "image_byte_scan.core", destination / "core.py")
    A = load(monkeypatch, "image_byte_scan.adjudicate", destination / "adjudicate.py")
    P = load(
        monkeypatch,
        "image_byte_scan.public_native_policy",
        destination / "public_native_policy.py",
    )
    monkeypatch.setattr(F, "W", W)
    monkeypatch.setattr(CASE, "W", W)
    monkeypatch.setattr(CASE, "A", A)
    monkeypatch.setattr(CASE, "CHECKOUT", trusted)
    with W.authorized_roots(analysis, trusted):
        args, auth, _ = CASE.complete_case(analysis)
        body = b"neutral private-operator-marker body"
        native = {"rule_id": "generic-api-key", "start_line": 0, "end_line": 0}

        class RecordingDetector(F.FakeDetector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.findings = 0

            def end(self, length, value):
                super().end(length, value)
                result = (
                    [copy.deepcopy(native), copy.deepcopy(native)]
                    if self.current == body
                    else []
                )
                self.findings += len(result)
                return result

            def finish(self):
                result = super().finish()
                result["findings"] = self.findings
                return result

        monkeypatch.setattr(W._scan, "__defaults__", (RecordingDetector,))
        proof = destination / "synthetic-public-proof.md"
        proof.write_text(
            "Synthetic independently approved noncredential fixture evidence.\n"
        )
        proof_binding = {
            "path": str(proof.relative_to(trusted)),
            "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        }
        catalog = {
            "schema_version": P.SCHEMA,
            "detector_identity": {
                "helper_sha256": auth["helper"]["sha256"],
                "config_sha256": auth["config"]["sha256"],
            },
            "entries": [
                {
                    "record_kind": "layer_regular_content",
                    "record_sha256": hashlib.sha256(body).hexdigest(),
                    "record_bytes": len(body),
                    "native_findings": [native, copy.deepcopy(native)],
                    "semantic_role": "cryptographic-self-test",
                    "operational_credential": False,
                    "public_provenance": proof_binding,
                    "semantic_proof": copy.deepcopy(proof_binding),
                }
            ],
        }
        policy = destination / "synthetic-policy.json"
        policy.write_text(json.dumps(catalog))
        inventory = Path(auth["literal_inventory"]["path"])
        inventory.write_text(json.dumps({"literals": ["absent confidential fixture"]}))
        auth["literal_inventory"]["sha256"] = hashlib.sha256(
            inventory.read_bytes()
        ).hexdigest()
        auth["sources"] = W.source_bindings()
        F.write(args.authorization, F.js(auth))
        committed = {
            name: Path(binding["path"]).read_bytes()
            for name, binding in auth["sources"].items()
        }
    original_git = A.subprocess.check_output
    head = original_git(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    def git_objects(argv, **kwargs):
        if argv[:3] == ["git", "-C", str(trusted)]:
            if argv[3:] == ["rev-parse", "HEAD"]:
                return head + "\n"
            if argv[3:5] == ["cat-file", "blob"]:
                commit, name = argv[5].split(":", 1)
                if commit != head or name not in committed:
                    raise subprocess.CalledProcessError(128, argv)
                return committed[name]
        return original_git(argv, **kwargs)

    monkeypatch.setattr(A.subprocess, "check_output", git_objects)
    output = analysis / "fresh-scan"
    argv = [
        "--analysis-root",
        str(analysis),
        "--trusted-root",
        str(trusted),
        "--authorization",
        str(args.authorization),
        "--output-dir",
        str(output),
        "--public-native-policy",
        str(policy),
        "--public-native-policy-sha256",
        hashlib.sha256(policy.read_bytes()).hexdigest(),
    ]
    return W, P, argv, output, auth, policy, proof


def test_fresh_cli_keeps_raw_failure_and_counts_every_native_occurrence(
    prepared, capsys
):
    W, _P, argv, out, *_ = prepared
    assert W.main(argv) == 0
    assert capsys.readouterr().out == "image byte public-policy gate passed\n"
    raw = json.loads((out / "report.json").read_text())
    receipt = json.loads((out / "public-policy-acceptance.json").read_text())
    assert raw["complete"] is True and raw["valid"] is False and raw["findings"] == 4
    assert receipt["accepted"] is True and receipt["accepted_native_occurrences"] == 4
    assert (out / "public-policy-acceptance.json").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("target", ["policy", "proof", "authorization", "archive"])
def test_changed_inputs_cannot_emit_accepted_receipt(prepared, target):
    W, _P, argv, out, auth, policy, proof = prepared
    path = {
        "policy": policy,
        "proof": proof,
        "authorization": Path(argv[5]),
        "archive": Path(auth["archive"]["path"]),
    }[target]
    path.write_bytes(path.read_bytes() + b"changed")
    assert W.main(argv) == 1
    assert not (out / "public-policy-acceptance.json").exists()


@pytest.mark.parametrize("target", ["ledger", "raw_report", "authorization", "proof"])
def test_post_scan_mutations_are_rechecked_before_receipt(
    prepared, monkeypatch, target
):
    W, P, argv, out, _auth, _policy, proof = prepared
    original = P.FreshPolicyReview.accept_fresh_scan

    def racing(self, report, directory):
        path = {
            "ledger": directory / "records.jsonl",
            "raw_report": directory / "report.json",
            "authorization": Path(argv[5]),
            "proof": proof,
        }[target]
        path.write_bytes(path.read_bytes() + b"changed after scan")
        return original(self, report, directory)

    monkeypatch.setattr(P.FreshPolicyReview, "accept_fresh_scan", racing)
    assert W.main(argv) == 1
    assert not (out / "public-policy-acceptance.json").exists()


@pytest.mark.parametrize("family", ["literal", "customer_regex", "infra_regex"])
def test_every_confidentiality_family_stays_fatal_after_reviewed_native_matches(
    prepared, family
):
    W, _P, argv, out, auth, _policy, _proof = prepared
    authpath = Path(argv[5])
    if family == "literal":
        path = Path(auth["literal_inventory"]["path"])
        path.write_text(json.dumps({"literals": ["private-operator-marker"]}))
        auth["literal_inventory"]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    else:
        path = authpath.parent / "confidentiality.json"
        config = {
            "customer_pattern": "private-operator-marker"
            if family == "customer_regex"
            else "absent-customer",
            "infra_pattern": "private-operator-marker"
            if family == "infra_regex"
            else None,
        }
        auth["confidentiality"] = F.write(path, F.js(config))
    F.write(authpath, F.js(auth))
    assert W.main(argv) == 1
    raw = json.loads((out / "report.json").read_text())
    assert raw["complete"] is True and raw["valid"] is False and raw["findings"] > 4
    assert not (out / "public-policy-acceptance.json").exists()


@pytest.mark.parametrize("failure", ["omitted_row", "callback_error"])
def test_observer_failure_never_becomes_accepted_scan(prepared, monkeypatch, failure):
    W, P, argv, out, *_ = prepared
    original = P.FreshPolicyReview.observe
    invoked = False

    def failing(self, emitted):
        nonlocal invoked
        if not invoked:
            invoked = True
            if failure == "callback_error":
                raise W.ScanError("synthetic_observer_failure")
            return
        original(self, emitted)

    monkeypatch.setattr(P.FreshPolicyReview, "observe", failing)
    assert W.main(argv) == 1
    assert invoked and not (out / "public-policy-acceptance.json").exists()
    raw = json.loads((out / "report.json").read_text())
    if failure == "callback_error":
        assert raw["complete"] is False and raw["helper_joined"] is True


def test_cli_cannot_accept_a_caller_supplied_forged_report(prepared, capsys):
    W, _P, argv, out, *_ = prepared
    assert W.main(argv + ["--report", "synthetic-forged-report.json"]) == 1
    captured = capsys.readouterr()
    assert "synthetic-forged-report" not in captured.out + captured.err
    assert not (out / "public-policy-acceptance.json").exists()


def test_catalog_and_substantive_proof_are_in_committed_source_closure(prepared):
    W, _P, argv, _out, auth, policy, proof = prepared
    root = Path(argv[3])
    for path in [policy, proof]:
        binding = auth["sources"][str(path.relative_to(root))]
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert W.main(argv) == 0
