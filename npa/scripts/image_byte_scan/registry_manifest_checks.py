"""Exercise original-manifest byte coverage with actual native scanner dependencies."""

from __future__ import annotations

import gzip
import json

from . import (
    core as W,
    prepare as P,
    registry_manifest_verification as R,
    synthetic as F,
)

CANARY = "synthetic-registry-private-value"


def _blob(files, data, media):
    digest = "sha256:" + W.sha(data)
    files["blobs/sha256/" + digest[7:]] = data
    return {"mediaType": media, "digest": digest, "size": len(data)}


def _configuration(raw, body):
    return F.js(
        {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {
                "type": "layers",
                "diff_ids": ["sha256:" + W.sha(b) for b in raw],
            },
            "history": [{"created_by": body}, {"created_by": "remove ancestor"}],
        }
    )


def _fixture(docker, finding):
    files = {"oci-layout": F.js({"imageLayoutVersion": "1.0.0"})}
    body = CANARY if finding else "public-control"
    raw = [
        F.tar_data([F.file("opt/removed", body.encode())]),
        F.tar_data([F.file("opt/.wh.removed"), F.file("opt/current", b"control")]),
    ]
    media = (
        (R.DOCKER + "distribution.manifest.v2+json")
        if docker
        else R.OCI + "manifest.v1+json"
    )
    config_media, layer_media = R.FORMATS[media]
    codec = next(m for m in layer_media if m.endswith(("+gzip", ".gzip")))
    config = _blob(files, _configuration(raw, body), config_media)
    layers = [_blob(files, gzip.compress(b, mtime=0), codec) for b in raw]
    manifest = _blob(
        files,
        F.js(
            {"schemaVersion": 2, "mediaType": media, "config": config, "layers": layers}
        ),
        media,
    )
    files["index.json"] = F.js(
        {
            "schemaVersion": 2,
            "mediaType": R.OCI + "index.v1+json",
            "manifests": [manifest],
        }
    )
    return files, manifest["digest"]


def _authorization(case, files, expected, tools_receipt, engine):
    archive = F.write(
        case / "image.tar", F.tar_data([F.file(n, b) for n, b in files.items()])
    )
    verification = F.write(
        case / "verification.json", F.js(R.verify(archive, expected))
    )
    helper, config = P.tools_bindings(tools_receipt)
    return {
        "schema_version": "npa.image-byte-scan-authorization.v1",
        "accepted_verification": True,
        "archive": archive,
        "verification_report": verification,
        "expected_image_id": expected,
        "helper": helper,
        "config": config,
        "tools_receipt": P.binding(tools_receipt),
        "sources": W.source_bindings(),
        "literal_engine": engine,
        "confidentiality": F.write(
            case / "policy.json", F.js({"customer_pattern": CANARY})
        ),
    }


def _check_report(report, output, finding):
    W.require(
        report["complete"] and report["helper_joined"], "registry_native_complete"
    )
    W.require(report["valid"] is (not finding), "registry_native_finding_boundary")
    W.require(
        report["regular_files"] == 3 and len(report["layers"]) == 2,
        "registry_native_ancestors",
    )
    W.require(
        report["oci_graph"]["attestations"] == R.ATTESTATIONS,
        "registry_native_attestations",
    )
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()
    ]
    W.require(
        CANARY not in json.dumps(report) + json.dumps(records),
        "registry_native_disclosure",
    )


def _check_case(directory, docker, finding, tools_receipt, engine):
    name = ("docker" if docker else "oci") + ("-finding" if finding else "-control")
    case = directory / name
    case.mkdir(mode=0o700)
    files, expected = _fixture(docker, finding)
    authorization = _authorization(case, files, expected, tools_receipt, engine)
    output = case / "scan"
    output.mkdir(mode=0o700)
    report = W._scan(authorization, output)
    W.write_private_json(output, "report.json", report)
    _check_report(report, output, finding)
    return {
        "case": name,
        "passed": True,
        "expected_image_id": expected,
        "complete": True,
        "helper_joined": True,
        "findings": report["findings"],
        "regular_files": report["regular_files"],
        "scanned_bytes": report["scanned_bytes"],
        "archive_sha256": authorization["archive"]["sha256"],
    }


def checks(directory, tools_receipt, engine):
    """Prove native control and finding behavior for both original manifest types.

    Args:
        directory: Private output directory for the synthetic checks.
        tools_receipt: Verified native detector receipt.
        engine: Verified native literal-matcher binding.
    Returns:
        Four actual native scan receipts, never qualification of a real image.
    Raises:
        W.ScanError: Native identity, accounting, detection or child join fails.
    """
    directory.mkdir(mode=0o700)
    return [
        _check_case(directory, docker, finding, tools_receipt, engine)
        for docker in (False, True)
        for finding in (False, True)
    ]
