"""Require NCore component evaluation and delivered-notice coverage before publication."""

from __future__ import annotations

import io
import json
from pathlib import Path
import re
import tarfile

from npa.deploy import ncore_component_advisories as advisories
from npa.deploy import ncore_component_inventory as inventory
from npa.deploy import ncore_component_sources as sources

_TRIVY_VERSION = "0.72.0"
_TRIVY_ARCHIVE_SHA256 = "bbb64b9695866ce4a7a8f5c9592002c5961cab378577fa3f8a040df362b9b2ea"
_TRIVY_BINARY_SHA256 = "0e69edd134a3c338baa1a6806920773615d682b18cbc6a0cba2a3b658ef9b63e"
_TRIVY_URL = "https://github.com/aquasecurity/trivy/releases/download/v0.72.0/trivy_0.72.0_Linux-64bit.tar.gz"
_LICENSE_IDS = {
    "cpython": {"Python-2.0", "BeOpen", "CNRI-Python-GPL-Compatible", "BSD-0-Clause"},
    "cpython-incorporated": {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Python-2.0"},
    "expat": {"MIT"}, "mpdecimal": {"BSD-2-Clause"}, "hacl": {"MIT"},
    "hacl-krml": {"Apache-2.0"}, "hacl-fstar": {"Apache-2.0"},
    "blake2-python": {"CC0-1.0"}, "blake2": {"CC0-1.0"}, "ncore": {"Apache-2.0"}, "pycolmap": {"MIT"},
    "npa": {"Apache-2.0"},
}

# Short upstream headers invoke these complete terms. Both inputs must be
# delivered and authenticated before their concatenation is scanned.
_LICENSE_TERMS = {
    "Apache-2.0": {"path": "usr/share/common-licenses/Apache-2.0",
                   "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"},
    "CC0-1.0": {"path": "usr/share/common-licenses/CC0-1.0",
                "sha256": "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"},
}
_NOTICE_TERMS = {"hacl-krml": "Apache-2.0", "hacl-fstar": "Apache-2.0",
                 "blake2": "CC0-1.0", "blake2-python": "CC0-1.0"}


def _trivy_binary(directory, executable):
    if executable is not None:
        raw = Path(executable).read_bytes()
    else:
        archive = advisories._download_scanner(_TRIVY_URL)
        inventory._require(inventory._sha(archive) == _TRIVY_ARCHIVE_SHA256, "Trivy download hash differs")
        (directory / "trivy.tar.gz").write_bytes(archive)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            members = [member for member in source if member.name == "trivy"]
            inventory._require(len(members) == 1 and members[0].isfile(), "Trivy archive member differs")
            raw = source.extractfile(members[0]).read()
    inventory._require(inventory._sha(raw) == _TRIVY_BINARY_SHA256, "Trivy executable hash differs")
    binary = directory / "trivy"
    binary.write_bytes(raw)
    binary.chmod(0o700)
    return binary


def _notice_population(population):
    notices = dict(population["required_notices"])
    for component in population["components"]:
        if component["name"] not in {"ncore", "pycolmap", "npa"}:
            continue
        notices[component["name"]] = {"path": component["license_path"],
                                     "sha256": component["license_sha256"]}
    return notices


def _stage_notices(path, population, directory):
    target = directory / "licenses"
    target.mkdir(mode=0o700)
    staged, failures = {}, []
    with tarfile.open(path) as archive:
        members = inventory.selected._archive_members(archive)
        for name, notice in _notice_population(population).items():
            origins = [notice]
            if name in _NOTICE_TERMS:
                origins.append(_LICENSE_TERMS[_NOTICE_TERMS[name]])
            raw = _delivered_notices(archive, members, population, origins)
            if raw is None:
                failures.append("missing or changed delivered notice: " + name)
                continue
            staged[name] = _stage_notice(target, name, raw, origins)
    return target, staged, failures


def _delivered_notices(archive, members, population, origins):
    contents = []
    for notice in origins:
        record = population["files"].get(notice["path"], {})
        member = members.get(notice["path"])
        if (not notice["sha256"] or record.get("sha256") != notice["sha256"]
                or member is None or not member.isfile()):
            return None
        raw = inventory._read(archive, members, notice["path"])
        inventory._require(inventory._sha(raw) == notice["sha256"], "notice changed while staging")
        contents.append(raw)
    return b"\n".join(contents)


def _stage_notice(target, name, raw, origins):
    # Trivy's loose-license analyzer recognizes LICENSE, not NPA-LICENSE. The
    # scanner-only input retains every original byte, joining multiple delivered
    # inputs with one LF; origins and the resulting hash bind that transformation.
    path = target / name / "LICENSE"
    path.parent.mkdir(mode=0o700)
    path.write_bytes(raw)
    return {"scan_path": name + "/LICENSE", "sha256": inventory._sha(raw), "origins": origins}


def license_findings(payload: dict, staged: dict, target: Path) -> dict:
    """Require recognized loose-file license results for each exact staged notice.

    Args:
        payload: Pinned Trivy full-license filesystem result.
        staged: Notice-to-source/scan-file hash mapping produced from the image.
        target: Exact private directory supplied to Trivy.
    Returns:
        Scanner identifiers and missing scope; no redistribution approval is inferred.
    Raises:
        ValueError: Wrong scanner, scan target, schema or unexpected notice population.
        KeyError, TypeError: Malformed report fields.
    """
    inventory._require(payload.get("SchemaVersion") == 2
                       and payload.get("Trivy", {}).get("Version") == _TRIVY_VERSION
                       and payload.get("ArtifactType") == "filesystem"
                       and payload.get("ArtifactName") == str(target), "Trivy license identity differs")
    inventory._require(isinstance(payload.get("Results"), list), "Trivy license results missing")
    observed = {name: set() for name in staged}
    by_path = {row["scan_path"]: name for name, row in staged.items()}
    for row in payload["Results"]:
        inventory._require(row.get("Class") == "license-file", "unexpected license analyzer scope")
        inventory._require(isinstance(row.get("Licenses"), list), "license findings missing")
        for finding in row["Licenses"]:
            name = by_path.get(finding["FilePath"])
            inventory._require(name is not None, "license result has an unexpected file")
            observed[name].add(finding["Name"])
    missing = [name for name, expected in _LICENSE_IDS.items()
               if not expected <= observed.get(name, set())]
    return {"observed": {name: sorted(values) for name, values in observed.items()},
            "missing_license_scope": sorted(missing)}


def _license_evaluation(path, population, directory, executable):
    target, staged, failures = _stage_notices(path, population, directory)
    advisories._write(directory / "license-inputs.json", staged)
    binary = _trivy_binary(directory, executable)
    (directory / "trivy.json").write_text("{}\n")
    (directory / "ignore").write_text("")
    command = [str(binary), "fs", "--config", str(directory / "trivy.json"),
               "--ignorefile", str(directory / "ignore"), "--scanners", "license",
               "--license-full", "--format", "json", str(target)]
    raw = advisories._run(command, directory, "licenses.trivy")
    result = license_findings(json.loads(raw), staged, target)
    for row in staged.values():
        inventory._require(advisories._file_hash(target / row["scan_path"]) == row["sha256"],
                           "staged license changed during scan")
    inventory._require(advisories._file_hash(binary) == _TRIVY_BINARY_SHA256,
                       "Trivy changed during evaluation")
    return {**result, "delivery_failures": failures, "inputs": staged,
            "report_sha256": inventory._sha(raw), "scanner_sha256": _TRIVY_BINARY_SHA256}


def _bundled_coverage(component, row, expected):
    name = component["name"]
    profile = inventory._BUNDLED_SOURCE_PROFILES.get(name)
    if profile is None:
        return
    inventory._require(component.get("source_mapping") == profile
                       and row.get("source_mapping") == profile
                       and component["query"] == profile["query"]
                       and component["version"] == ("2.5.1" if name == "mpdecimal"
                                                     else inventory._CPYTHON_COMMIT)
                       and re.fullmatch(r"[0-9a-f]{64}", row.get("source_proof_sha256", "")),
                       "bundled source proof missing or substituted")
    parent = expected["cpython"]
    inventory._require(parent["version"] == inventory._CPYTHON_VERSION
                       and parent["query"] == {"cpe": "cpe:2.3:a:python:python:"
                                               + inventory._CPYTHON_VERSION + ":*:*:*:*:*:*:*"}
                       and row.get("parent_advisory_scope") == {
                           "component": "cpython", "version": parent["version"], "query": parent["query"]},
                       "CPython patch advisory scope differs")
    if name == "mpdecimal":
        inventory._require(row.get("upstream_review") == sources._MPDECIMAL_REVIEW,
                           "mpdecimal upstream release review missing or substituted")


def coverage_failures(population: dict, evaluations: list[dict], licenses: dict) -> list[str]:
    """Require the exact evaluated component population and all delivered licenses.

    Args:
        population: Inventory computed from authenticated rootfs/lock/source inputs.
        evaluations: Fresh evaluation results produced by this module's scanner calls.
        licenses: Fresh delivered-notice verification and full Trivy license result.
    Returns:
        Publication refusal reasons; empty only when every required scope is covered.
    Raises:
        ValueError: A component identity is duplicated, substituted or omitted.
        KeyError, TypeError: Malformed required evidence.
    """
    expected = {row["name"]: row for row in population["components"]}
    actual = {row["component"]: row for row in evaluations}
    inventory._require(set(expected) == inventory._COMPONENT_NAMES
                       and len(expected) == len(population["components"])
                       and len(actual) == len(evaluations) and actual.keys() == expected.keys(),
                       "evaluation component population differs")
    failures = list(licenses["delivery_failures"])
    failures += ["license scanner did not cover: " + name for name in licenses["missing_license_scope"]]
    for name, row in actual.items():
        component = expected[name]
        inventory._require(all(row[field] == component[field]
                               for field in ("version", "files_sha256", "query")),
                           "evaluation component binding differs")
        query = component["query"]
        method = "unmapped" if query is None else "grype-cpe" if "cpe" in query else "osv-commit"
        inventory._require(row["method"] == method, "evaluation method differs")
        if row["method"] == "unmapped":
            failures.append("unmapped vulnerability evaluation: " + name)
        else:
            _bundled_coverage(component, row, expected)
        for finding in row["findings"]:
            if finding["blocking"]:
                failures.append("blocking advisory: " + name + ":" + finding["id"])
    return failures


def scan_archive(path: Path, directory: Path, *, base_lock: bytes, source_lock: bytes,
                 source_sha: str, committed_files: dict[str, str], image_digest: str,
                 platform_digest: str, config_digest: str,
                 grype_executable: Path | None = None, trivy_executable: Path | None = None) -> dict:
    """Run required supplemental NCore scans and retain a receipt even on policy refusal.

    Args:
        path: Read-only merged tar whose derivation the publication OCI gate proves.
        directory: New private evidence directory; existing paths are refused.
        base_lock, source_lock: Reviewed committed lock bytes, never image-only claims.
        source_sha, committed_files: Authenticated source-closure inputs to inventory_archive.
        image_digest, platform_digest, config_digest: Exact verified OCI graph identities.
        grype_executable, trivy_executable: Optional exact-hash host scanner binaries.
    Returns:
        Byte/population-bound receipt only after all coverage and policy checks pass.
    Raises:
        ValueError: Any missing coverage, unsupported component, finding, or identity failure.
        OSError, KeyError, TypeError: Scanner/network/input failure; publication must stop.
    """
    identity = inventory.selected._scan_identity(image_digest, platform_digest, config_digest)
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    before = advisories._file_hash(path)
    population = inventory.inventory_archive(path, base_lock=base_lock, source_lock=source_lock,
                                             source_sha=source_sha, committed_files=committed_files)
    inventory_hash = advisories._write(directory / "inventory.json", population)
    evaluations = advisories.evaluate_components(population["components"], directory,
                                                 grype_executable=grype_executable)
    licenses = _license_evaluation(path, population, directory, trivy_executable)
    inventory._require(advisories._file_hash(path) == before, "rootfs changed during evaluation")
    failures = coverage_failures(population, evaluations, licenses)
    receipt = {"format": "npa_ncore_component_scan_v1", **identity, "rootfs_sha256": before,
               "inventory_sha256": inventory_hash, "population_sha256": population["population_sha256"],
               "status": "refused" if failures else "pass", "failures": failures,
               "evaluations": evaluations, "licenses": licenses,
               "scope": "Merged NCore files only; selected Debian and ordinary image/byte scans remain required"}
    advisories._write(directory / "receipt.json", receipt)
    inventory._require(not failures, "required coverage/policy refused; see private component receipt")
    return receipt
