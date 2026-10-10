"""Exercise fresh-gate binding without authorizing real image or registry writes."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W, prepare as P  # noqa: E402
from ncore_publication import acceptance, attribution, cli, gates, publication_binding  # noqa: E402
from ncore_publication.process import file_sha  # noqa: E402


def _write(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(W.canonical(value))
    path.chmod(0o600)
    return file_sha(path)


def _authorization(directory):
    policy_path = directory / "policy.json"
    policy = {"customer_pattern": "synthetic-customer", "infra_pattern": None}
    _write(policy_path, policy)
    engine = {"kind": "aho-corasick-v1"}
    engine.update(
        {
            key: {"path": "/synthetic/" + key, "sha256": value}
            for key, value in W.AHO_PINS.items()
        }
    )
    authorization = {
        "confidentiality": P.binding(policy_path),
        "literal_engine": engine,
        "sources": {"npa/scripts/image_byte_scan/aho_matcher.py": engine["source"]},
    }
    return _write(directory / "authorization/authorization.json", authorization)


def _report(directory, build, graph, adjudicated):
    auth_sha = _authorization(directory)
    return {
        "valid": not adjudicated,
        "complete": True,
        "helper_joined": True,
        "findings": 2 if adjudicated else 0,
        "scanned_bytes": 1024,
        "regular_files": 1,
        "expected_image_id": build["image_digest"],
        "archive_sha256": build["archive_sha256"],
        "image_config_digest": graph["image_config_digest"],
        "authorization_sha256": auth_sha,
        "confidentiality_policy": W.C.compile_policy(
            **dict(customer_pattern="synthetic-customer", infra_pattern=None)
        ).receipt(),
        "literal_engine": attribution._expected_native_engine_receipt(),
        "input_snapshot_receipts": [
            {
                "role": "policy",
                "sha256": file_sha(directory / "policy.json"),
                "stat": [str(directory)],
            }
        ],
    }


def _raw_gate(root, directory, build, graph, *, adjudicated=False):
    report = _report(directory, build, graph, adjudicated)
    report_sha = _write(directory / "bytes/report.json", report)
    ledger_sha = _write(
        directory / "bytes/records.jsonl",
        {"synthetic": "full finding bytes", "findings": report["findings"]},
    )
    if adjudicated:
        _write(
            directory / "attribution.json",
            {
                "authorization_sha256": report["authorization_sha256"],
                "report_sha256": report_sha,
                "records_sha256": ledger_sha,
                "occurrences": ["exact synthetic occurrence"],
                "policy_sha256": report["confidentiality_policy"]["policy_sha256"],
                "suppressed": False,
            },
        )
        _write(directory / "attribution-replay/report.json", report)
        _write(
            directory / "attribution-replay/records.jsonl",
            {"synthetic": "full finding bytes", "findings": report["findings"]},
        )
    _write(directory / "source-guards.log", {"successful_run_path": str(directory)})
    _security_fixtures(directory)
    return report, ledger_sha


def _selected_fixture(directory, trivy):
    sbom = {
        "documentNamespace": str(directory),
        "creationInfo": {"created": str(directory)},
        "annotations": [
            {"annotationDate": str(directory), "comment": "same reviewed scope"}
        ],
        "packages": [{"name": "synthetic-component", "versionInfo": "1"}],
    }
    _write(
        directory / "selected-base/selected.receipt.json",
        {
            "sbom_sha256": _write(directory / "selected-base/selected.spdx.json", sbom),
            "report_sha256": _write(
                directory / "selected-base/selected.trivy.json", trivy
            ),
            "findings": [{"id": "synthetic-advisory", "blocking": False}],
        },
    )


def _component_fixture(directory, trivy):
    license_report = dict(trivy, ArtifactName=str(directory / "components/licenses"))
    license_hash = _write(
        directory / "components/licenses.trivy.stdout", license_report
    )
    grype_hash = _write(
        directory / "components/synthetic.grype.stdout",
        {"log_path": str(directory), "matches": []},
    )
    _write(
        directory / "components/receipt.json",
        {
            "licenses": {
                "report_sha256": license_hash,
                "observed": ["synthetic-license"],
            },
            "evaluations": [
                {
                    "component": "synthetic",
                    "method": "grype-cpe",
                    "report_sha256": grype_hash,
                    "database_status": {
                        "path": str(directory / "components/db/6/vulnerability.db"),
                        "built": "same-build",
                    },
                    "database_sha256": "a" * 64,
                    "findings": [{"id": "synthetic-advisory", "blocking": False}],
                }
            ],
        },
    )


def _security_fixtures(directory):
    trivy = {
        "CreatedAt": str(directory),
        "ReportID": str(directory),
        "Results": [
            {
                "Vulnerabilities": [
                    {"VulnerabilityID": "synthetic-advisory", "Severity": "LOW"}
                ]
            }
        ],
    }
    for name in ("trivy-all.json", "trivy-policy.json"):
        _write(directory / name, trivy)
    _write(
        directory / "payload.json",
        {"image": str(directory / "inspection.tar"), "payload_hits": []},
    )
    _write(directory / "payload-history.json", {"history_hits": []})
    _selected_fixture(directory, trivy)
    _component_fixture(directory, trivy)


def _proposed_manifest(root, build, graph, adjudicated):
    directory = root / "check"
    directory.mkdir(mode=0o700)
    report, ledger_sha = _raw_gate(
        root, directory, build, graph, adjudicated=adjudicated
    )
    manifest_path = cli._gate_evidence_manifest(directory, "a" * 40, build, graph)
    _write(directory / "prepublication.json", {"status": "pass"})
    return {
        "development_sha": "a" * 40,
        "oci_digest": build["image_digest"],
        "amd64_manifest": graph["image_manifest_digest"],
        "config_digest": graph["image_config_digest"],
        "prepublication": {
            "archive_sha256": build["archive_sha256"],
            "evidence_manifest_sha256": file_sha(manifest_path),
            "raw_byte_report_sha256": file_sha(directory / "bytes/report.json"),
            "raw_byte_ledger_sha256": ledger_sha,
        },
        "byte_scan": {
            "resolution": "public-attribution" if adjudicated else "raw-clean",
            "policy_sha256": report["confidentiality_policy"]["policy_sha256"],
        },
    }


def _statement(root, directory, manifest, monkeypatch):
    evidence = root / "workload"
    _write(evidence / "qualification-audit.json", {"synthetic": True})
    proposed = root / "proposed.json"
    _write(proposed, manifest)
    # These tests isolate publication/statement binding, not GPU, image scanner,
    # hosted VLM or manifest-schema acceptance; those boundaries have own controls.
    monkeypatch.setattr(
        acceptance.images, "validate_ncore_accepted_image_manifest", lambda value: value
    )
    for name in ("_prepublication", "_qualification", "_visual", "_retained_inventory"):
        monkeypatch.setattr(acceptance, name, lambda *_: {})
    statement_path = root / acceptance.STATEMENT_PATH
    return acceptance.build_statement(
        analysis_root=root,
        gate_dir=directory,
        evidence_root=evidence,
        proposed_manifest_path=proposed,
        output_path=statement_path,
    )


def _accepted(root, build, graph, monkeypatch, *, adjudicated=False):
    manifest = _proposed_manifest(root, build, graph, adjudicated)
    statement = _statement(root, root / "check", manifest, monkeypatch)
    statement_path = root / acceptance.STATEMENT_PATH
    review = {
        "format": acceptance.REVIEW_FORMAT,
        "verdict": "ACCEPTED",
        "candidate_commit": "a" * 40,
        "statement_sha256": file_sha(statement_path),
        "evidence_inventory_sha256": statement["evidence_inventory_sha256"],
        "manifest_sha256": statement["manifest_sha256"],
        "objective_evidence_reviewed": True,
        "visual_evidence_reviewed": True,
        "cleanup_reviewed": True,
        "reviewer_id": "synthetic-independent-lane",
    }
    review_path = root / acceptance.REVIEW_PATH
    _write(review_path, review)
    return acceptance.finalize_acceptance(
        analysis_root=root,
        statement_path=statement_path,
        review_path=review_path,
        output_path=root / acceptance.FINAL_PATH,
    )


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    build = {
        "source_sha": "a" * 40,
        "context_sha256": "f" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "archive_sha256": "c" * 64,
    }
    graph = {
        "image_manifest_digest": "sha256:" + "d" * 64,
        "image_config_digest": "sha256:" + "e" * 64,
    }
    _write(tmp_path / "build/build.json", build)
    # Real scanning belongs to the unchanged gates/replay; record the boundary so
    # a positive binding cannot silently skip the mandatory fresh replay.
    replays = []
    monkeypatch.setattr(
        publication_binding.byte_acceptance,
        "verify",
        lambda *args: replays.append(args),
    )
    with W.authorized_roots(tmp_path, ROOT):
        yield tmp_path, build, graph, replays


def _publish(root, build, graph, monkeypatch, *, adjudicated=False, damage=None):
    args = SimpleNamespace(
        action="publish",
        source_sha="a" * 40,
        analysis_root=root,
        output_dir=root / "publish",
        acceptance=root / acceptance.FINAL_PATH,
    )

    def fresh(_args, directory, _build):
        _raw_gate(root, directory, build, graph, adjudicated=adjudicated)
        if damage:
            damage(directory)
        _args.source_sha = _build["source_sha"]
        return graph, {}

    monkeypatch.setattr(cli, "_build_receipt", lambda *_: build)
    monkeypatch.setattr(gates, "verify", fresh)
    transfers = []
    monkeypatch.setattr(
        cli.registry, "transfer", lambda *_: transfers.append("synthetic transfer")
    )
    cli._check_or_publish(args)
    return transfers


@pytest.mark.parametrize("adjudicated", [False, True])
def test_fresh_logs_and_snapshot_stats_preserve_exact_review(
    candidate, monkeypatch, adjudicated
):
    root, build, graph, replays = candidate
    accepted = _accepted(root, build, graph, monkeypatch, adjudicated=adjudicated)
    before = {
        path: path.read_bytes()
        for path in (root / "check").rglob("*")
        if path.is_file()
    }
    assert _publish(root, build, graph, monkeypatch, adjudicated=adjudicated) == [
        "synthetic transfer"
    ]
    receipt = acceptance._json(root / "publish/accepted-publication-binding.json")
    assert (
        receipt["reviewed_evidence_manifest_sha256"]
        != receipt["fresh_evidence_manifest_sha256"]
    )
    assert receipt["reviewed_raw_report_sha256"] != receipt["fresh_raw_report_sha256"]
    assert (
        receipt["identical_raw_ledger_sha256"]
        == accepted["prepublication"]["raw_byte_ledger_sha256"]
    )
    assert receipt["review"] == accepted["acceptance_verification"]
    assert len(replays) == 1
    assert all(path.read_bytes() == content for path, content in before.items())


def _damage(root, build, graph, damage):
    def change(directory):
        if damage == "ledger":
            _write(
                directory / "bytes/records.jsonl",
                {"same_totals": 2, "different_finding_bytes": True},
            )
        elif damage == "suppression":
            path = directory / "attribution.json"
            receipt = acceptance._json(path)
            receipt["suppressed"] = True
            path.unlink()
            _write(path, receipt)
        elif damage in {"source", "image", "config"}:
            if damage == "source":
                build["source_sha"] = "0" * 40
            elif damage == "image":
                build["archive_sha256"] = "0" * 64
            else:
                graph["image_config_digest"] = "sha256:" + "0" * 64
        elif damage == "reviewer":
            path = root / acceptance.REVIEW_PATH
            review = acceptance._json(path)
            review["reviewer_id"] = "substituted-lane"
            path.unlink()
            _write(path, review)
        elif damage == "missing-original":
            (root / "check/source-guards.log").unlink()
        else:
            _damage_report(directory, damage)

    return change


def _damage_report(directory, damage):
    path = directory / "bytes/report.json"
    report = acceptance._json(path)
    if damage == "policy":
        report["confidentiality_policy"]["policy_sha256"] = "0" * 64
    elif damage == "snapshot-hash":
        report["input_snapshot_receipts"][0]["sha256"] = "0" * 64
    else:
        report["layers"] = [{"different_bytes": True}]
    path.unlink()
    _write(path, report)


@pytest.mark.parametrize(
    "damage,expected",
    [
        ("ledger", "publication_reviewed_byte_population_changed"),
        ("policy", "publication_reviewed_byte_population_changed"),
        ("suppression", "publication_reviewed_disposition_changed"),
        ("source", "accepted_ncore_evidence_does_not_match_candidate"),
        ("image", "accepted_ncore_evidence_does_not_match_candidate"),
        ("config", "accepted_ncore_evidence_does_not_match_candidate"),
        ("layer", "publication_reviewed_byte_population_changed"),
        ("snapshot-hash", "publication_reviewed_byte_population_changed"),
        ("reviewer", "accepted_manifest_review_binding"),
        ("missing-original", "accepted evidence is outside"),
    ],
)
def test_changed_population_or_review_never_reaches_transfer(
    candidate, monkeypatch, damage, expected
):
    root, build, graph, _replays = candidate
    _accepted(root, build, graph, monkeypatch, adjudicated=True)
    change = _damage(root, build, graph, damage)

    with pytest.raises(ValueError, match=expected):
        _publish(root, build, graph, monkeypatch, adjudicated=True, damage=change)
    assert not (root / "publish/transfer").exists()


def test_fresh_native_replay_failure_cannot_reach_transfer(candidate, monkeypatch):
    root, build, graph, _replays = candidate
    _accepted(root, build, graph, monkeypatch)

    def failed_replay(*_):
        raise ValueError("synthetic fresh replay failure")

    monkeypatch.setattr(publication_binding.byte_acceptance, "verify", failed_replay)
    with pytest.raises(ValueError, match="synthetic fresh replay failure"):
        _publish(root, build, graph, monkeypatch)
    assert not (root / "publish/transfer").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "new",
        "removed",
        "substituted",
        "component-disposition",
        "database",
        "payload",
        "unknown-metadata",
    ],
)
def test_material_security_changes_require_new_review(candidate, monkeypatch, damage):
    root, build, graph, _replays = candidate
    _accepted(root, build, graph, monkeypatch)

    def change(directory):
        relative = (
            "components/receipt.json"
            if damage in {"component-disposition", "database"}
            else "payload.json"
            if damage == "payload"
            else "trivy-all.json"
        )
        path = directory / relative
        report = acceptance._json(path)
        if damage in {"new", "removed", "substituted"}:
            findings = report["Results"][0]["Vulnerabilities"]
            if damage == "new":
                findings.append(
                    {"VulnerabilityID": "new-nonblocking", "Severity": "LOW"}
                )
            elif damage == "removed":
                findings.clear()
            else:
                findings[0]["VulnerabilityID"] = "same-count-different-id"
        elif damage == "component-disposition":
            report["evaluations"][0]["findings"][0]["blocking"] = True
        elif damage == "database":
            report["evaluations"][0]["database_sha256"] = "b" * 64
        elif damage == "payload":
            report["payload_hits"] = ["changed population"]
        else:
            report["unknown_metadata"] = "never normalize unknown fields"
        _write(path, report)

    with pytest.raises(
        ValueError, match="publication_reviewed_security_population_changed"
    ):
        _publish(root, build, graph, monkeypatch, damage=change)
    assert not (root / "publish/transfer").exists()


def test_exact_literal_publication_rejected_before_inventory_access():
    args = SimpleNamespace(
        action="publish",
        policy_mode="exact-literals",
        literal_inventory=Path("/not/read"),
    )
    with pytest.raises(ValueError, match="publication_requires_ci_regex_policy"):
        cli._policy_input(args)


def test_reviewer_hash_cannot_be_substituted_in_final_manifest(candidate, monkeypatch):
    root, build, graph, _replays = candidate
    manifest = _accepted(root, build, graph, monkeypatch)
    manifest["acceptance_verification"]["reviewer_id_sha256"] = hashlib.sha256(
        b"different-reviewer"
    ).hexdigest()
    path = root / acceptance.FINAL_PATH
    path.unlink()
    _write(path, manifest)
    with pytest.raises(ValueError, match="accepted_manifest_review_binding"):
        acceptance.verify_final_acceptance(root, path)
