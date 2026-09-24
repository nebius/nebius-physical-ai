"""NCore's selected Debian files as SPDX 2.3; never an installed dpkg database.

The lock-only mode is a source inventory/probe, not image verification. Publication
uses an exact-digest crane export and checks the shipped lock against actual files.
Package-level CVEs are conservative for partial packages; absence of an affected
file is not a vulnerability exception. CPython, runtime-fetched dependencies and
ancestor layers remain the responsibility of the existing image/byte scans.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from urllib.parse import quote
from uuid import uuid4

LOCK_PATH = "opt/ncore/base-sources/recipes/base-source-lock.json"
ANNEX = "opt/ncore/base-sources/"
SCOPE = "Selected Debian files only: partial packages, not installed dpkg packages."
SCAN_FORMAT = "npa_ncore_selected_base_scan_v1"


def _require(ok, message):
    if not ok:
        raise ValueError("NCore selected-base " + message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(name):
    _require(isinstance(name, str) and "\x00" not in name, "unsafe path")
    path = PurePosixPath(name)
    _require(
        not path.is_absolute() and ".." not in path.parts and bool(path.parts),
        "unsafe path",
    )
    return str(path)


def _id(kind, value):
    return "SPDXRef-" + kind + "-" + _sha(value.encode())


def _inventory(raw):
    lock = json.loads(raw)
    _require(
        lock["schema"] == 2 and "-slim-bookworm@sha256:" in lock["base_image"],
        "unsupported lock/distro",
    )
    packages, files = {}, {}

    for package in lock["debian_binaries"]:
        _add_inventory_package(packages, files, package)
    _require(bool(packages), "empty package inventory")
    for item in lock["notices"]:
        _add_inventory_file(files, item)
    _add_inventory_artifacts(lock, files)
    return lock, packages, files


def _add_inventory_file(files, item):
    name = _path(item["path"])
    identity = {key: item[key] for key in ("sha256", "link") if key in item}
    _require(len(identity) == 1, "missing or ambiguous file identity")
    if "sha256" in identity:
        _require(re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "invalid file hash")
    _require(name not in files or files[name] == identity, "conflicting file identity")
    files[name] = identity


def _add_inventory_package(packages, files, package):
    key = tuple(package[key] for key in ("name", "version", "architecture"))
    _require(
        all(isinstance(value, str) and value for value in key) and package["files"],
        "empty package identity/scope",
    )
    identity = {key: package[key] for key in ("source", "url", "sha256")}
    _require(re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "invalid deb hash")
    if key in packages:
        _require(
            all(packages[key][field] == value for field, value in identity.items()),
            "conflicting package identity",
        )
        packages[key]["files"] += package["files"]
    else:
        packages[key] = {**package, "files": list(package["files"])}
    for item in package["files"]:
        _add_inventory_file(files, item)


def _add_inventory_artifacts(lock, files):
    metadata_hashes = {
        digest
        for repository in lock.get("debian_repositories", [])
        for digest in [
            repository["inrelease"],
            *(index["artifact"] for index in repository["indexes"].values()),
        ]
    }
    required_source_hashes = {
        digest
        for component in lock["components"]
        for digest in component.get("artifacts", [])
    }
    for item in lock["artifacts"]:
        if item.get("delivery") == "build-only":
            _require(
                item["sha256"] in metadata_hashes
                and item["sha256"] not in required_source_hashes
                and _path(item["path"]).startswith("metadata/")
                and "transformation" not in item,
                "invalid build-only repository metadata",
            )
            continue
        _add_inventory_file(
            files, {"path": ANNEX + _path(item["path"]), "sha256": item["sha256"]}
        )


def _source_components(lock):
    sources = {}
    for component in lock["components"]:
        _require(component["id"] not in sources, "duplicate source component")
        if component["kind"] == "debian-source":
            _require(
                component["id"] == f"debian:{component['name']}@{component['version']}",
                "source component identity differs",
            )
        sources[component["id"]] = component
    return sources


def build_spdx(raw: bytes, *, sha1s: dict | None = None) -> dict:
    """Translate only lock-declared identities; hash-lock claims are not byte proof.

    Args:
        raw: Selected-base lock JSON bytes.
        sha1s: Verified file SHA1 checksums, or None for a lock-only inventory.
    Returns:
        An SPDX 2.3 document for the selected Debian files and sources.
    Raises:
        ValueError: Lock JSON or a required inventory identity is invalid.
        KeyError: A required lock field or verified checksum is missing.
        TypeError: A lock field has an incompatible type.
    """
    lock, packages, files = _inventory(raw)
    lock_hash = _sha(raw)
    files[LOCK_PATH] = {"sha256": lock_hash}
    operating_system_id = "SPDXRef-OperatingSystem-debian-12"
    document = _spdx_document(lock_hash, operating_system_id)
    _add_spdx_files(document, files, sha1s)
    _annotate_spdx_files(document, files, sha1s)
    sources = _source_components(lock)
    artifacts = {
        artifact.get("transformation", {}).get(
            "input_sha256", artifact["sha256"]
        ): artifact
        for artifact in lock["artifacts"]
    }
    for key, package in sorted(packages.items()):
        _add_spdx_package(
            document,
            sha1s,
            key,
            package,
            sources,
            artifacts,
            files,
            operating_system_id,
        )
    return document


def _spdx_document(lock_hash, operating_system_id):
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "ncore-selected-debian-files",
        "documentNamespace": "https://nebius.com/spdx/ncore-selected-base/"
        + str(uuid4()),
        "creationInfo": {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: npa-ncore-selected-sbom"],
        },
        "comment": SCOPE
        + " Hashes and provenance from shipped "
        + LOCK_PATH
        + "; sha256="
        + lock_hash,
        "packages": [_operating_system_package(operating_system_id)],
        "files": [],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": operating_system_id,
            }
        ],
    }


def _operating_system_package(operating_system_id):
    return {
        "SPDXID": operating_system_id,
        "name": "debian",
        "versionInfo": "12",
        "primaryPackagePurpose": "OPERATING_SYSTEM",
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "comment": "Distribution context (bookworm), not a complete Debian installation.",
    }


def _add_spdx_files(document, files, sha1s):
    for name, identity in sorted(files.items()):
        if "link" in identity or sha1s is None:
            continue  # No invented checksum of symlink content; targets recorded below.
        document["files"].append(
            {
                "SPDXID": _id("File", name),
                "fileName": "./" + name,
                "checksums": [
                    {"algorithm": "SHA256", "checksumValue": identity["sha256"]},
                    {"algorithm": "SHA1", "checksumValue": sha1s[name]},
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )


def _annotate_spdx_files(document, files, sha1s):
    # SPDX 2.3 requires SHA1 for File elements. A lock-only vulnerability probe
    # has only SHA256, so carry factual file evidence in a standard annotation;
    # the verified-export path adds proper File elements with computed SHA1 too.
    document["annotations"] = [
        {
            "annotationType": "OTHER",
            "annotator": "Tool: npa-ncore-selected-sbom",
            "annotationDate": document["creationInfo"]["created"],
            "comment": json.dumps(
                {
                    "source": "verified export"
                    if sha1s is not None
                    else "lock-only; shipped bytes not verified",
                    "files": files,
                },
                sort_keys=True,
            ),
        }
    ]


def _add_spdx_package(
    document, sha1s, key, package, sources, artifacts, files, operating_system_id
):
    name, version, architecture = key
    source = sources[package["source"]]
    _require(source["kind"] == "debian-source", "invalid source component")
    purl = (
        f"pkg:deb/debian/{quote(name, safe='')}@{quote(version, safe='')}"
        f"?arch={quote(architecture, safe='')}&distro=debian-12"
    )
    package_id = _id("Package", purl)
    links = {item["path"]: item["link"] for item in package["files"] if "link" in item}
    _append_spdx_package(
        document, sha1s, package, source, package_id, purl, links, name, version
    )
    _add_package_verification(document, sha1s, package)
    _relate(document, sha1s, operating_system_id, "CONTAINS", package_id)
    _relate(
        document,
        sha1s,
        package_id,
        "OTHER",
        _id("File", LOCK_PATH),
        "Exact selected-file/source correspondence lock",
    )
    for item in package["files"]:
        if "sha256" in item:
            _relate(
                document,
                sha1s,
                package_id,
                "CONTAINS",
                _id("File", _path(item["path"])),
            )
    _relate_package_sources(document, sha1s, package_id, name, files, source, artifacts)


def _add_package_verification(document, sha1s, package):
    if sha1s is not None:
        selected_hashes = sorted(
            sha1s[path]
            for path in {item["path"] for item in package["files"] if "sha256" in item}
        )
        document["packages"][-1]["packageVerificationCode"] = {
            "packageVerificationCodeValue": hashlib.sha1(
                "".join(selected_hashes).encode(), usedforsecurity=False
            ).hexdigest(),
        }
        document["packages"][-1]["licenseInfoFromFiles"] = ["NOASSERTION"]


def _append_spdx_package(
    document, sha1s, package, source, package_id, purl, links, name, version
):
    document["packages"].append(
        {
            "SPDXID": package_id,
            "name": name,
            "versionInfo": version,
            "filesAnalyzed": sha1s is not None,
            "downloadLocation": package["url"],
            "licenseDeclared": "NOASSERTION",
            "licenseConcluded": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            # Trivy's SPDX decoder reads Debian source identity from this field.
            "sourceInfo": f"built package from: {source['name']} {source['version']}",
            "comment": SCOPE
            + " File analysis/verification covers selected regular files only."
            + " Original .deb SHA256: "
            + package["sha256"]
            + ". Selected symlinks (path: target): "
            + json.dumps(links, sort_keys=True),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": purl,
                }
            ],
        }
    )


def _relate_package_sources(
    document, sha1s, package_id, name, files, source, artifacts
):
    notice = f"usr/share/doc/{name}/copyright"
    _require(notice in files and "sha256" in files[notice], "missing package notice")
    _require(source["delivery"] in ("source", "notice"), "invalid source delivery")
    _require(
        bool(source["artifacts"]) == (source["delivery"] == "source"),
        "source delivery requires corresponding artifacts",
    )
    _relate(
        document,
        sha1s,
        package_id,
        "OTHER",
        _id("File", notice),
        "Delivered copyright/license notice; no license inference",
    )
    for digest in source["artifacts"]:
        artifact = artifacts[digest]
        _require(
            artifact.get("delivery") != "build-only", "source omitted as build-only"
        )
        _relate(
            document,
            sha1s,
            package_id,
            "OTHER" if artifact.get("transformation") else "GENERATED_FROM",
            _id("File", ANNEX + _path(artifact["path"])),
            "Corresponding source delivery; upstream URL: "
            + artifact["url"]
            + "; any source transformation is recorded in the linked lock",
        )


def _relate(document, sha1s, parent, kind, child, comment=None):
    if child.startswith("SPDXRef-File-") and sha1s is None:
        return
    row = {
        "spdxElementId": parent,
        "relationshipType": kind,
        "relatedSpdxElement": child,
    }
    if comment:
        row["comment"] = comment
    if row not in document["relationships"]:
        document["relationships"].append(row)


def from_archive(path: Path) -> dict:
    """Verify a merged crane export without extracting or executing image content.

    Args:
        path: Merged exact-digest crane export archive.
    Returns:
        SPDX inventory with checksums computed from verified archive bytes.
    Raises:
        ValueError: Export paths, identities, distro or archive format are invalid.
        OSError: The archive cannot be read.
        TypeError: A shipped lock field has an incompatible type.
    """
    try:
        return _from_archive(path)
    except (KeyError, tarfile.TarError) as exc:
        raise ValueError(
            "NCore selected-base missing file or malformed export"
        ) from exc


def _from_archive(path: Path) -> dict:
    with tarfile.open(path) as archive:
        members = _archive_members(archive)
        with _read_archive_file(archive, members, LOCK_PATH) as stream:
            raw = stream.read()
        sha1s = {LOCK_PATH: hashlib.sha1(raw, usedforsecurity=False).hexdigest()}
        _, _, files = _inventory(raw)
        _verify_archive_distro(archive, members)
        for name, identity in files.items():
            sha1 = _verify_archive_file(archive, members, name, identity)
            if sha1 is not None:
                sha1s[name] = sha1
    return build_spdx(raw, sha1s=sha1s)


def _archive_members(archive):
    members = {}
    for member in archive:
        if member.name in (".", "./"):
            _require(member.isdir(), "invalid archive root")
            continue
        name = _path(member.name)
        _require(name not in members, "duplicate archive path")
        members[name] = member
    for name in members:
        for parent in PurePosixPath(name).parents:
            if str(parent) in members:
                _require(members[str(parent)].isdir(), "non-directory archive ancestor")
    return members


def _read_archive_file(archive, members, name):
    member = members[name]
    _require(member.isfile(), "expected regular file")
    return archive.extractfile(member)


def _verify_archive_distro(archive, members):
    with _read_archive_file(archive, members, "etc/os-release") as stream:
        distro = dict(
            line.split("=", 1)
            for line in stream.read().decode().splitlines()
            if "=" in line and not line.startswith("#")
        )
    _require(
        distro.get("ID", "").strip('"') == "debian"
        and distro.get("VERSION_ID", "").strip('"') == "12",
        "wrong shipped distro",
    )


def _verify_archive_file(archive, members, name, identity):
    member = members[name]
    if "link" in identity:
        _require(
            member.issym() and member.linkname == identity["link"],
            "symlink changed",
        )
        return
    digest = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    with _read_archive_file(archive, members, name) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            sha1.update(block)
    _require(
        digest.hexdigest() == identity["sha256"],
        "shipped file hash differs from lock",
    )
    return sha1.hexdigest()


def scan_counts(payload: dict, sbom: dict) -> dict:
    """Require every exact binary/source version in Trivy --list-all-pkgs output.

    Args:
        payload: Trivy JSON report with all evaluated packages.
        sbom: Selected-base SPDX document used by that scan.
    Returns:
        Critical vulnerability, secret and evaluated-package counts.
    Raises:
        ValueError: Coverage differs, secrets exist or a CRITICAL has a fix.
        KeyError: Required SPDX or finding fields are missing.
        TypeError: A report field has an incompatible type.
    """
    expected = _expected_trivy_packages(sbom)
    os_info = payload.get("Metadata", {}).get("OS", {})
    _require(
        os_info.get("Family") == "debian" and os_info.get("Name") == "12",
        "Trivy omitted Debian 12",
    )
    actual, critical = set(), {}
    _require(isinstance(payload.get("Results"), list), "invalid Trivy report")
    for result in payload["Results"]:
        _collect_trivy_result(result, actual, critical)
    _require(
        expected and actual == expected,
        "Trivy package coverage differs from selected inventory",
    )
    return {
        "critical_total": len(critical),
        "critical_with_fix": 0,
        "critical_unfixed": len(critical),
        "secrets": 0,
        "packages_evaluated": len(actual),
    }


def _expected_trivy_packages(sbom):
    from urllib.parse import parse_qs, urlsplit

    expected = set()
    for package in sbom["packages"]:
        if not package.get("externalRefs"):
            continue
        purl = package["externalRefs"][0]["referenceLocator"]
        architecture = parse_qs(urlsplit(purl).query)["arch"][0]
        source, version = (
            package["sourceInfo"].removeprefix("built package from: ").split(" ")
        )
        expected.add(
            (package["name"], package["versionInfo"], source, version, architecture)
        )
    return expected


def _collect_trivy_result(result, actual, critical):
    _require(isinstance(result, dict), "invalid Trivy result")
    _require(not result.get("Secrets"), "secret findings")
    _collect_trivy_criticals(result, critical)
    if result.get("Class") != "os-pkgs" or result.get("Type") != "debian":
        _require(not result.get("Packages"), "unexpected Trivy package population")
        return
    packages = result.get("Packages")
    _require(isinstance(packages, list), "Trivy omitted --list-all-pkgs")
    for package in packages:
        _require(isinstance(package, dict), "invalid Trivy package")
        actual.add(_trivy_package_identity(package))


def _trivy_package_identity(package):
    source_version = package.get("SrcVersion", "")
    if package.get("SrcRelease"):
        source_version += "-" + package["SrcRelease"]
    if package.get("SrcEpoch"):
        source_version = str(package["SrcEpoch"]) + ":" + source_version
    return (
        package.get("Name"),
        package.get("Version"),
        package.get("SrcName"),
        source_version,
        package.get("Arch"),
    )


def _collect_trivy_criticals(result, critical):
    findings = result.get("Vulnerabilities") or []
    _require(isinstance(findings, list), "invalid Trivy vulnerabilities")
    for finding in findings:
        _require(isinstance(finding, dict), "invalid Trivy finding")
        if str(finding["Severity"]).upper() != "CRITICAL":
            continue
        _require(not finding.get("FixedVersion"), "fixed CRITICAL vulnerabilities")
        identity = tuple(
            finding[key] for key in ("VulnerabilityID", "PkgName", "InstalledVersion")
        )
        critical[identity] = finding


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return _file_sha256(path)


def _verified_counts(sbom):
    evidence = json.loads(sbom["annotations"][0]["comment"])
    _require(evidence["source"] == "verified export", "lock-only scan is diagnostic")
    lock = next(item for item in sbom["files"] if item["fileName"] == "./" + LOCK_PATH)
    packages = sorted(_expected_trivy_packages(sbom))
    return {
        "lock_sha256": lock["checksums"][0]["checksumValue"],
        "files_verified": len(sbom["files"]) - 1,
        "symlinks_verified": sum("link" in item for item in evidence["files"].values()),
        "packages_sha256": _sha(json.dumps(packages, separators=(",", ":")).encode()),
    }


def _trivy_arguments(command, directory):
    return [
        *command,
        "sbom",
        "--config",
        str(directory / "trivy.yaml"),
        "--ignorefile",
        str(directory / "ignore"),
        "--ignore-unfixed=false",
        "--scanners",
        "vuln",
        "--pkg-types",
        "os",
        "--list-all-pkgs",
        "--severity",
        "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL",
        "--format",
        "json",
        "--quiet",
        "--exit-code",
        "0",
        str(directory / "selected.spdx.json"),
    ]


def _run_trivy(directory, command):
    # Ignore ambient filters and working-directory files; the ordinary image
    # vuln/secret scan remains mandatory and is not replaced by this SBOM scan.
    (directory / "trivy.yaml").write_text("{}\n", encoding="utf-8")
    (directory / "ignore").write_text("", encoding="utf-8")
    scan_directory = directory
    command = list(command)
    if len(command) != 1:  # The existing exact-digest Docker fallback.
        command[-1:-1] = ["--volume", f"{directory}:/npa-sbom:ro"]
        scan_directory = Path("/npa-sbom")
    completed = subprocess.run(
        _trivy_arguments(command, scan_directory),
        capture_output=True,
        text=True,
        check=False,
        cwd=directory,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("TRIVY_")
        },
    )
    (directory / "selected.trivy.json").write_text(completed.stdout, encoding="utf-8")
    _require(completed.returncode == 0, "Trivy SBOM scan failed")
    return json.loads(completed.stdout)


def scan_archive(
    path: Path,
    directory: Path,
    *,
    image_digest: str,
    platform_digest: str,
    config_digest: str,
    trivy_command: list[str] | None = None,
) -> dict:
    """Retain supplemental SPDX, Trivy output and a byte-bound scan receipt.

    Args:
        path: Final merged filesystem tar derived from the exact OCI artifact.
        directory: New evidence directory; existing directories are refused.
        image_digest: OCI index digest whose export the caller has verified.
        platform_digest: Its sole linux/amd64 manifest digest.
        config_digest: That platform's config digest.
        trivy_command: Supported host Trivy or publication Docker fallback argv.
    Returns:
        A receipt binding supplied OCI identities to observed bytes and counts.
        The caller must prove export derivation from that OCI graph separately.
    Raises:
        ValueError: Invalid identity, bytes, coverage, secret or fixed CRITICAL.
        OSError: Inputs, evidence outputs or the scanner cannot be accessed.
        KeyError, TypeError: A required lock/report identity is malformed.
    """
    identity = _scan_identity(image_digest, platform_digest, config_digest)
    if trivy_command is None:
        from npa.deploy.publish_public import _trivy_command

        trivy_command = _trivy_command()
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    return _scan_archive(path, directory, identity, trivy_command)


def _scan_identity(image_digest, platform_digest, config_digest):
    identity = dict(
        image_digest=image_digest,
        platform_digest=platform_digest,
        config_digest=config_digest,
    )
    for digest in identity.values():
        _require(
            isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest),
            "scan requires exact OCI digests",
        )
    _require(len(set(identity.values())) == 3, "scan requires distinct OCI identities")
    return identity


def _scan_archive(path, directory, identity, command):
    archive_hash = _file_sha256(path)
    sbom = from_archive(path)
    _require(_file_sha256(path) == archive_hash, "export changed during verification")
    sbom_hash = _write_json(directory / "selected.spdx.json", sbom)
    payload = _run_trivy(directory, command)
    receipt = {
        "format": SCAN_FORMAT,
        "status": "pass",
        **identity,
        "rootfs_sha256": archive_hash,
        "sbom_sha256": sbom_hash,
        "report_sha256": _file_sha256(directory / "selected.trivy.json"),
        **_verified_counts(sbom),
        **scan_counts(payload, sbom),
    }
    _write_json(directory / "selected.receipt.json", receipt)
    return receipt


def _cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--lock", type=Path, help="Diagnostic inventory; cannot pass a scan"
    )
    inputs.add_argument(
        "--rootfs-tar", type=Path, help="Final merged filesystem export"
    )
    parser.add_argument("--output", type=Path, help="SPDX output for inventory mode")
    parser.add_argument(
        "--scan-output", type=Path, help="New directory for scan evidence"
    )
    for name in ("image-digest", "platform-digest", "config-digest"):
        parser.add_argument(
            "--" + name, help="Exact OCI identity; required with --scan-output"
        )
    return parser


def main():
    """Write a diagnostic inventory or scan verified final filesystem bytes.

    Args:
        None. Inputs are parsed from command-line arguments.
    Returns:
        None.
    Raises:
        SystemExit: Command-line arguments are invalid or help was requested.
        OSError: Input or output files cannot be accessed.
        ValueError: Lock or export validation fails.
        KeyError: A required lock field is absent.
        TypeError: An input field has an incompatible type.
    """
    parser = _cli_parser()
    args = parser.parse_args()
    identities = (args.image_digest, args.platform_digest, args.config_digest)
    if args.scan_output:
        if not args.rootfs_tar or args.output or not all(identities):
            parser.error(
                "scan requires --rootfs-tar and all OCI digests, without --output"
            )
        scan_archive(
            args.rootfs_tar,
            args.scan_output,
            image_digest=args.image_digest,
            platform_digest=args.platform_digest,
            config_digest=args.config_digest,
        )
        return
    if not args.output or any(identities):
        parser.error("inventory requires --output; OCI digests require --scan-output")
    sbom = (
        from_archive(args.rootfs_tar)
        if args.rootfs_tar
        else build_spdx(args.lock.read_bytes())
    )
    _write_json(args.output, sbom)


if __name__ == "__main__":
    main()
