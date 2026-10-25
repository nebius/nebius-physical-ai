"""Hostile offline controls for genuine-shaped retained receipt contracts."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import core as W  # noqa: E402
from ncore_publication import retained_receipts as R, retained_security as S  # noqa: E402
from ncore_acceptance_fixture import _s3_probe  # noqa: E402


@pytest.fixture
def probe(tmp_path):
    tmp_path.chmod(0o700)
    manifest = {"qualification_controls": {}}
    _s3_probe(tmp_path, manifest)
    raw = tmp_path / "s3-handoff-probe.json"
    manifest["qualification_controls"]["s3_probe_receipt_sha256"] = hashlib.sha256(
        raw.read_bytes()
    ).hexdigest()
    return (
        manifest,
        json.loads(raw.read_bytes()),
        json.loads((tmp_path / "s3-probe-provenance.json").read_bytes()),
    )


def test_original_v1_probe_controls_and_scope_are_accepted(probe, tmp_path):
    manifest, payload, _ = probe
    with W.authorized_roots(tmp_path, ROOT):
        R.s3_probe(manifest, tmp_path, payload)


@pytest.mark.parametrize("field", R.S3_CONTROLS)
@pytest.mark.parametrize("value", [None, False, 1, "true", []])
def test_each_literal_probe_control_is_required(probe, field, value):
    manifest, payload, provenance = probe
    payload[field] = value
    with pytest.raises(ValueError, match="s3_controls"):
        R.validate_s3_probe(
            payload,
            provenance,
            receipt_sha256=manifest["qualification_controls"][
                "s3_probe_receipt_sha256"
            ],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pass"),
        ("status", "OK"),
        ("status", True),
        ("status", "failed"),
        ("format", "unknown"),
        ("delete_status", "requested"),
        ("payload_bytes", 0),
        ("payload_bytes", True),
        ("payload_bytes", "257"),
        ("payload_bytes", -1),
        ("scope_sha256", "b" * 64),
        ("payload_sha256", ""),
        ("etag_sha256", "abc"),
        ("before_inventory_sha256", None),
        ("during_inventory_sha256", "A" * 64),
        ("after_inventory_sha256", "abc"),
    ],
)
def test_probe_does_not_coerce_invalid_literals(probe, field, value):
    manifest, payload, provenance = probe
    payload[field] = value
    with pytest.raises(ValueError, match="retained_s3"):
        R.validate_s3_probe(
            payload,
            provenance,
            receipt_sha256=manifest["qualification_controls"][
                "s3_probe_receipt_sha256"
            ],
        )


@pytest.mark.parametrize(
    "changed", ["scope", "receipt", "producer", "execution", "selection"]
)
def test_probe_rejects_substituted_originals(probe, tmp_path, changed):
    manifest, payload, provenance = probe
    if changed == "scope":
        provenance["scope_sha256"] = "b" * 64
    elif changed == "receipt":
        provenance["receipt_sha256"] = "b" * 64
    elif changed == "producer":
        provenance["producer"]["sha256"] = "b" * 64
    else:
        key = "execution" if changed == "execution" else "scope_selection"
        provenance[key]["sha256"] = "b" * 64
    path = tmp_path / "s3-probe-provenance.json"
    path.write_text(json.dumps(provenance))
    manifest["qualification_controls"]["s3_probe_provenance_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with (
        W.authorized_roots(tmp_path, ROOT),
        pytest.raises(ValueError, match="retained_"),
    ):
        R.s3_probe(manifest, tmp_path, payload)


@pytest.mark.parametrize(
    "results",
    [
        None,
        "",
        {},
        [None],
        ["finding"],
        [{"Vulnerabilities": None}],
        [{"Secrets": {}}],
        [{"Vulnerabilities": [None]}],
        [{"Vulnerabilities": [{}]}],
        [{"Vulnerabilities": [{"Severity": "CRITICAL", "FixedVersion": False}]}],
    ],
)
def test_trivy_never_discards_malformed_rows(results):
    with pytest.raises(ValueError, match="acceptance_trivy"):
        S.counts({"Results": results}, omission_authenticated=True)


def test_only_authenticated_omission_is_empty():
    with pytest.raises(ValueError, match="results"):
        S.counts({})
    assert S.counts({}, omission_authenticated=True) == dict(
        critical_total=0, critical_with_fix=0, critical_unfixed=0, secrets=0
    )


def test_present_findings_preserve_fixed_unfixed_and_secret_accounting():
    report = {
        "Results": [
            {
                "Vulnerabilities": [
                    {"Severity": "CRITICAL"},
                    {"Severity": "CRITICAL", "FixedVersion": "1.2"},
                    {"Severity": "HIGH"},
                ],
                "Secrets": [{"RuleID": "synthetic-rule", "Severity": "LOW"}],
            }
        ]
    }
    original = copy.deepcopy(report)
    assert S.counts(report) == dict(
        critical_total=2, critical_with_fix=1, critical_unfixed=1, secrets=1
    )
    assert report == original


@pytest.fixture
def trivy_identity():
    digest = "sha256:" + "a" * 64
    layers = ["sha256:" + "b" * 64, "sha256:" + "c" * 64]
    graph = {"image_config_digest": digest, "verified_layer_diff_ids": layers}
    report = {
        "SchemaVersion": 2,
        "ArtifactType": "container_image",
        "ArtifactID": digest,
        "Trivy": {"Version": "0.72.0"},
        "Metadata": {"ImageID": digest, "DiffIDs": layers},
    }
    return report, graph


@pytest.mark.parametrize(
    "change",
    [None, "schema", "type", "tool", "artifact", "metadata", "config", "layers"],
)
def test_omitted_report_requires_external_config_and_ordered_layers(
    trivy_identity, change
):
    report, graph = copy.deepcopy(trivy_identity)
    if change in {"schema", "type", "tool", "artifact", "metadata"}:
        field = {
            "schema": "SchemaVersion",
            "type": "ArtifactType",
            "tool": "Trivy",
            "artifact": "ArtifactID",
            "metadata": "Metadata",
        }[change]
        report[field] = None
    elif change == "config":
        report["Metadata"]["ImageID"] = "sha256:" + "d" * 64
    elif change == "layers":
        report["Metadata"]["DiffIDs"] = list(reversed(report["Metadata"]["DiffIDs"]))
    if change:
        with pytest.raises(ValueError, match="external_image_identity"):
            S._identity(report, graph)
    else:
        S._identity(report, graph)


def _driver_records():
    manifest = {
        "development_sha": "a" * 40,
        "oci_digest": "sha256:" + "b" * 64,
        "prepublication": {"archive_sha256": "c" * 64},
    }
    argv = [
        "/synthetic/npa/.venv/bin/python",
        "npa/scripts/publish_ncore_oci.py",
        "check",
    ]
    for key, value in {
        "--source-sha": manifest["development_sha"],
        "--analysis-root": "/evidence",
        "--output-dir": "/evidence/check",
        "--annex": "/evidence/source",
        "--native-source": "/evidence/native",
        "--metadata": "/evidence/meta",
        "--bootstrap-source": "/evidence/bootstrap",
    }.items():
        argv.extend([key, value])
    driver = {
        "returncode": 0,
        "source_sha": manifest["development_sha"],
        "image_digest": manifest["oci_digest"],
        "archive_sha256": "c" * 64,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:01:00+00:00",
        "argv": argv,
    }
    provenance = {
        "driver": {},
        "driver_argv": copy.deepcopy(argv),
        "original_checkout": "/synthetic",
    }
    return manifest, driver, provenance


@pytest.mark.parametrize(
    "change",
    [
        None,
        "exit",
        "bool_exit",
        "incomplete",
        "time",
        "source",
        "image",
        "archive",
        "argument",
        "checkout",
    ],
)
def test_original_driver_requires_completed_exact_process(
    monkeypatch, tmp_path, change
):
    manifest, driver, provenance = _driver_records()
    if change == "exit":
        driver["returncode"] = 1
    elif change == "bool_exit":
        driver["returncode"] = False
    elif change == "incomplete":
        driver.pop("completed_at")
    elif change == "time":
        driver["completed_at"] = driver["started_at"]
    elif change in {"source", "image", "archive"}:
        driver[
            {
                "source": "source_sha",
                "image": "image_digest",
                "archive": "archive_sha256",
            }[change]
        ] = "wrong"
    elif change == "argument":
        driver["argv"].extend(["--authfile", "/other"])
    elif change == "checkout":
        provenance["original_checkout"] = "/other"
    monkeypatch.setattr(R, "bound_json", lambda *_: driver)
    if change:
        with pytest.raises((ValueError, KeyError), match="retained_trivy|completed_at"):
            S._driver(provenance, tmp_path, manifest)
    else:
        assert S._driver(provenance, tmp_path, manifest) == "/evidence/check"


@pytest.mark.parametrize(
    "missing",
    [
        "selected",
        "selected_sbom",
        "selected_report",
        "components",
        "component_inventory",
        "licenses",
        "source_delivery",
        "source_inventory",
        "sbom",
        "base_lock",
        "source_lock",
    ],
)
def test_absent_supplement_cannot_qualify_omission(tmp_path, missing):
    names = {
        "selected",
        "selected_sbom",
        "selected_report",
        "components",
        "component_inventory",
        "licenses",
        "source_delivery",
        "source_inventory",
        "sbom",
        "base_lock",
        "source_lock",
    }
    with pytest.raises(ValueError, match="supplement_population"):
        S._supplements(
            tmp_path, {"supplements": dict.fromkeys(names - {missing}, {})}, {}, {}
        )
