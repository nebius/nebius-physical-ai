"""Recompute raw-clean or separately adjudicated NCore byte-scan acceptance."""

from pathlib import Path
import tempfile
from types import SimpleNamespace

from image_byte_scan import core as W

from . import artifact, attribution
from .process import PYTHON, ROOT, file_sha, run_byte_scanner


ATTRIBUTION_FILES = {
    "attribution_receipt_sha256": "attribution.json",
    "attribution_replay_report_sha256": "attribution-replay/report.json",
    "attribution_replay_ledger_sha256": "attribution-replay/records.jsonl",
}


def _replay(analysis_root, directory, bindings, expected_exit):
    retained = Path(
        tempfile.mkdtemp(prefix="acceptance-byte-replay-", dir=analysis_root)
    )
    argv = [
        str(PYTHON),
        "npa/scripts/scan_image_bytes.py",
        "--analysis-root",
        str(analysis_root),
        "--trusted-root",
        str(ROOT),
        "--authorization",
        str(directory / "authorization/authorization.json"),
        "--output-dir",
        str(retained / "bytes"),
    ]
    status = run_byte_scanner(argv, retained / "scanner.log")
    W.require(status == expected_exit, "acceptance_byte_replay_exit")
    for name, field in (
        ("report.json", "report_sha256"),
        ("records.jsonl", "records_sha256"),
    ):
        _raw, digest, _stat = attribution._read_private(retained / "bytes" / name)
        W.require(digest == bindings[field], "acceptance_byte_replay_population")


def _policy_identity(report):
    policy = report.get("confidentiality_policy")
    W.require(isinstance(policy, dict), "acceptance_byte_policy_missing")
    if policy.get("mode") == "exact-literals-v1":
        binding = policy.get("binding")
        W.require(
            isinstance(binding, dict) and binding.get("kind") == "exact-substring-v1",
            "acceptance_byte_literal_policy",
        )
        return "exact-literals-v1", W.sha(W.canonical(binding))
    W.require(
        policy.get("schema") == "npa.image-byte-confidentiality-policy.v1"
        and isinstance(policy.get("policy_sha256"), str),
        "acceptance_byte_regex_policy",
    )
    return "regex-v1", policy["policy_sha256"]


def _resolution(report, directory, prepublication):
    W.require(
        report.get("complete") is True
        and report.get("helper_joined") is True
        and "failure_code" not in report
        and type(report.get("findings")) is int,
        "acceptance_byte_scan_incomplete",
    )
    if report.get("valid") is True:
        W.require(report["findings"] == 0, "acceptance_clean_byte_findings")
        for field, relative in ATTRIBUTION_FILES.items():
            W.require(
                prepublication.get(field) is None
                and not (directory / relative).exists(),
                "acceptance_clean_has_attribution",
            )
        return "raw-clean", 0
    W.require(
        report.get("valid") is False and report["findings"] == 2,
        "acceptance_attribution_raw_findings",
    )
    for field, relative in ATTRIBUTION_FILES.items():
        W.require(
            prepublication.get(field) == file_sha(directory / relative),
            "acceptance_attribution_evidence_binding",
        )
    return "public-attribution", 1


def _result(report, bindings, resolution, policy):
    return {
        "status": "pass",
        "resolution": resolution,
        "raw_valid": report["valid"],
        "raw_findings": report["findings"],
        "dispositioned_findings": 0 if resolution == "raw-clean" else 2,
        "unresolved_findings": 0,
        "complete": True,
        "report_sha256": bindings["report_sha256"],
        "image_digest": report["expected_image_id"],
        "archive_sha256": report["archive_sha256"],
        "config_digest": report["image_config_digest"],
        "bytes_scanned": report["scanned_bytes"],
        "files_scanned": report["regular_files"],
        "policy_kind": policy[0],
        "policy_sha256": policy[1],
    }


def _adjudicate(manifest, analysis_root, directory, verification, resolution):
    if resolution != "public-attribution":
        return
    args = SimpleNamespace(
        source_sha=manifest["development_sha"], analysis_root=analysis_root
    )
    attribution.verify(
        args,
        directory,
        analysis_root / "build/image.oci.tar",
        manifest["oci_digest"],
        verification,
        1,
        retained=True,
    )


def verify(manifest, analysis_root, directory):
    """Recompute byte acceptance without rewriting any original scan evidence.

    Args:
        manifest: Candidate acceptance manifest, including explicit raw disposition.
        analysis_root: Private root holding the original image and scanner inputs.
        directory: Original prepublication gate directory.
    Returns:
        Receipt fields derived from authenticated, fully replayed byte evidence.
    Raises:
        ValueError, OSError: Any image, policy, provenance or scan binding fails.
    """
    archive = analysis_root / "build/image.oci.tar"
    digest = manifest["oci_digest"]
    authorization, report, _rows, bindings = attribution._scan_inputs(directory)
    _graph, verification = artifact.inspect(archive, digest)
    snapshots = attribution._authorization_binding(
        authorization, report, archive, digest, verification
    )
    resolution, expected_exit = _resolution(
        report, directory, manifest["prepublication"]
    )
    _replay(analysis_root, directory, bindings, expected_exit)
    policy = _policy_identity(report)
    _adjudicate(manifest, analysis_root, directory, verification, resolution)
    result = _result(report, bindings, resolution, policy)
    W.require(
        W.canonical(manifest["byte_scan"]) == W.canonical(result),
        "acceptance_byte_scan_results",
    )
    attribution._recheck(directory, bindings, snapshots, archive, verification)
    return result
