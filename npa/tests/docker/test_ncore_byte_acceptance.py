"""Test clean/adjudicated control flow and fail-closed byte evidence bindings."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import attribution, byte_acceptance as gate  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(W.canonical(value))
    path.chmod(0o600)
    return W.sha(path.read_bytes())


def _report(clean=True):
    return {
        "valid": clean,
        "complete": True,
        "helper_joined": True,
        "findings": 0 if clean else 2,
        "scanned_bytes": 1024,
        "regular_files": 1,
        "expected_image_id": "sha256:" + "a" * 64,
        "archive_sha256": "b" * 64,
        "image_config_digest": "sha256:" + "c" * 64,
        "confidentiality_policy": W.C.compile_policy(
            "synthetic-customer", None
        ).receipt(),
    }


def _fixture(tmp_path, monkeypatch, clean=True):
    report = _report(clean)
    directory = tmp_path / "gate"
    bindings = {
        "report_sha256": _write(directory / "bytes/report.json", report),
        "records_sha256": _write(
            directory / "bytes/records.jsonl", {"record": "synthetic"}
        ),
    }
    prepublication = {}
    if not clean:
        for field, relative in gate.ATTRIBUTION_FILES.items():
            prepublication[field] = _write(directory / relative, {"synthetic": field})
    resolution = "raw-clean" if clean else "public-attribution"
    manifest = {
        "oci_digest": report["expected_image_id"],
        "development_sha": "d" * 40,
        "prepublication": prepublication,
        "byte_scan": gate._result(
            report, bindings, resolution, gate._policy_identity(report)
        ),
    }
    monkeypatch.setattr(
        attribution, "_scan_inputs", lambda _: ({}, report, [], bindings)
    )
    monkeypatch.setattr(gate.artifact, "inspect", lambda *_: ({}, {}))
    monkeypatch.setattr(attribution, "_authorization_binding", lambda *_: [])
    monkeypatch.setattr(attribution, "_recheck", lambda *_: None)
    return manifest, report, directory, bindings


@pytest.mark.parametrize("clean", [True, False])
def test_both_branches_require_full_replay_and_preserve_raw_verdict(
    tmp_path, monkeypatch, clean
):
    manifest, report, directory, _ = _fixture(tmp_path, monkeypatch, clean)
    calls = []
    monkeypatch.setattr(
        gate, "_replay", lambda *args: calls.append(("replay", args[-1]))
    )
    monkeypatch.setattr(
        attribution,
        "verify",
        lambda *args, **kwargs: calls.append(("attribution", kwargs)),
    )
    before = json.dumps(report, sort_keys=True)
    with W.authorized_roots(tmp_path, ROOT):
        assert gate.verify(manifest, tmp_path, directory) == manifest["byte_scan"]
    assert calls == (
        [("replay", 0)]
        if clean
        else [("replay", 1), ("attribution", {"retained": True})]
    )
    assert json.dumps(report, sort_keys=True) == before


@pytest.mark.parametrize(
    "damage",
    ["findings", "complete", "helper", "failure", "valid", "bool-count", "attribution"],
)
def test_clean_failures_stop_before_replay(tmp_path, monkeypatch, damage):
    manifest, report, directory, _ = _fixture(tmp_path, monkeypatch)
    if damage == "attribution":
        manifest["prepublication"]["attribution_receipt_sha256"] = "e" * 64
    else:
        key, value = {
            "findings": ("findings", 2),
            "complete": ("complete", False),
            "helper": ("helper_joined", False),
            "failure": ("failure_code", "incomplete"),
            "valid": ("valid", 1),
            "bool-count": ("findings", False),
        }[damage]
        report[key] = value
    monkeypatch.setattr(
        gate, "_replay", lambda *_: pytest.fail("invalid raw result replayed")
    )
    with W.authorized_roots(tmp_path, ROOT), pytest.raises(ValueError):
        gate.verify(manifest, tmp_path, directory)


@pytest.mark.parametrize("field", list(gate.ATTRIBUTION_FILES))
def test_adjudication_requires_all_bound_receipts(tmp_path, monkeypatch, field):
    manifest, _report_value, directory, _ = _fixture(tmp_path, monkeypatch, False)
    manifest["prepublication"].pop(field)
    monkeypatch.setattr(
        gate, "_replay", lambda *_: pytest.fail("unbound disposition replayed")
    )
    with (
        W.authorized_roots(tmp_path, ROOT),
        pytest.raises(ValueError, match="attribution_evidence_binding"),
    ):
        gate.verify(manifest, tmp_path, directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_valid", 1),
        ("raw_findings", False),
        ("policy_sha256", "0" * 64),
        ("policy_kind", "exact-literals-v1"),
        ("archive_sha256", "0" * 64),
        ("bytes_scanned", 1025),
    ],
)
def test_manifest_must_equal_rederived_result_without_bool_coercion(
    tmp_path, monkeypatch, field, value
):
    manifest, _report_value, directory, _ = _fixture(tmp_path, monkeypatch)
    manifest["byte_scan"][field] = value
    monkeypatch.setattr(gate, "_replay", lambda *_: None)
    with (
        W.authorized_roots(tmp_path, ROOT),
        pytest.raises(ValueError, match="byte_scan_results"),
    ):
        gate.verify(manifest, tmp_path, directory)


@pytest.mark.parametrize("clean", [True, False])
@pytest.mark.parametrize("damage", [None, "exit", "report", "ledger"])
def test_real_replay_driver_requires_status_and_byte_identical_population(
    tmp_path, monkeypatch, clean, damage
):
    directory = tmp_path / "gate"
    report, ledger = W.canonical(_report(clean)), b'{"synthetic":"ledger"}\n'
    bindings = {"report_sha256": W.sha(report), "records_sha256": W.sha(ledger)}

    def scanner(argv, _output):
        assert argv[argv.index("--authorization") + 1] == str(
            directory / "authorization/authorization.json"
        )
        target = Path(argv[argv.index("--output-dir") + 1])
        target.mkdir(mode=0o700)
        for name, raw, damaged in (
            ("report.json", report, "report"),
            ("records.jsonl", ledger, "ledger"),
        ):
            path = target / name
            path.write_bytes(raw + (b"tampered" if damage == damaged else b""))
            path.chmod(0o600)
        return int(clean) if damage == "exit" else int(not clean)

    monkeypatch.setattr(gate, "run_byte_scanner", scanner)
    with W.authorized_roots(tmp_path, ROOT):
        if damage is None:
            gate._replay(tmp_path, directory, bindings, int(not clean))
        else:
            with pytest.raises(ValueError, match="byte_replay"):
                gate._replay(tmp_path, directory, bindings, int(not clean))


def test_exact_literal_policy_identity_never_claims_regex_equivalence():
    binding = {
        "kind": "exact-substring-v1",
        "inventory_sha256": "a" * 64,
        "matcher_sha256": "b" * 64,
        "pattern_count": 129,
    }
    assert gate._policy_identity(
        {"confidentiality_policy": {"mode": "exact-literals-v1", "binding": binding}}
    ) == ("exact-literals-v1", W.sha(W.canonical(binding)))


def test_retained_provenance_refuses_missing_archive_without_downloading(tmp_path):
    api = SimpleNamespace(
        _ARCHIVES=[SimpleNamespace(cache_name="pinned.tar")],
        verify_public_notice=lambda *_: pytest.fail("missing provenance fetched"),
    )
    with pytest.raises(ValueError, match="upstream_missing"):
        attribution._retained_upstream(api, b"notice", tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "policy_sha256",
        "upstream_proof_sha256",
        "source_sha",
        "report_sha256",
        "occurrences",
        "raw_valid",
    ],
)
def test_retained_attribution_must_equal_freshly_recomputed_receipt(tmp_path, field):
    expected = {
        "policy_sha256": "a" * 64,
        "upstream_proof_sha256": "b" * 64,
        "source_sha": "c" * 40,
        "report_sha256": "d" * 64,
        "occurrences": ["e" * 64, "f" * 64],
        "raw_valid": False,
    }
    _write(tmp_path / "attribution.json", expected)
    with W.authorized_roots(tmp_path, ROOT):
        attribution._retain_receipt(tmp_path, expected, True)
        damaged = dict(expected)
        damaged[field] = "forged"
        _write(tmp_path / "attribution.json", damaged)
        with pytest.raises(ValueError, match="receipt_changed"):
            attribution._retain_receipt(tmp_path, expected, True)
