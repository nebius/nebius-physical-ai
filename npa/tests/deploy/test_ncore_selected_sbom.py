"""Selected loose-file provenance, scanner coverage and fail-closed acceptance."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from npa.deploy import ncore_selected_sbom as selected


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def lock():
    return {
        "schema": 2,
        "base_image": "python:3.12.12-slim-bookworm@sha256:" + "a" * 64,
        "debian_binaries": [
            {
                "name": "libc6",
                "version": "2.36-9+deb12u14",
                "architecture": "amd64",
                "source": "debian:glibc@2.36-9+deb12u14",
                "url": "https://example.com/libc6.deb",
                "sha256": "b" * 64,
                "files": [{"path": "usr/lib/libc.so.6", "sha256": sha(b"libc")}],
            }
        ],
        "components": [
            {
                "id": "debian:glibc@2.36-9+deb12u14",
                "kind": "debian-source",
                "name": "glibc",
                "version": "2.36-9+deb12u14",
                "delivery": "source",
                "artifacts": [sha(b"source")],
            }
        ],
        "artifacts": [
            {
                "path": "sources/glibc.tar.xz",
                "sha256": sha(b"source"),
                "url": "https://example.com/glibc.tar.xz",
            }
        ],
        "notices": [
            {"path": "usr/share/doc/libc6/copyright", "sha256": sha(b"notice")}
        ],
    }


def test_build_only_signed_indexes_are_not_claimed_as_image_files(lock):
    metadata_hash = sha(b"signed metadata")
    lock["artifacts"].append(
        {
            "path": "metadata/debian/InRelease",
            "sha256": metadata_hash,
            "delivery": "build-only",
            "url": "https://example.com/InRelease",
        }
    )
    lock["debian_repositories"] = [{"inrelease": metadata_hash, "indexes": {}}]
    _, _, files = selected._inventory(json.dumps(lock).encode())
    assert selected.ANNEX + "metadata/debian/InRelease" not in files
    assert selected.ANNEX + "sources/glibc.tar.xz" in files


def test_required_source_cannot_be_omitted_as_build_only(lock):
    lock["artifacts"][0]["delivery"] = "build-only"
    with pytest.raises(ValueError, match="build-only"):
        selected._inventory(json.dumps(lock).encode())


def _fixture_sha1s(raw):
    return {
        path: hashlib.sha1(body, usedforsecurity=False).hexdigest()
        for path, body in {
            "usr/lib/libc.so.6": b"libc",
            "usr/share/doc/libc6/copyright": b"notice",
            "opt/ncore/base-sources/sources/glibc.tar.xz": b"source",
            selected.LOCK_PATH: raw,
        }.items()
    }


def test_partial_inventory_has_real_versions_hashes_and_source_links(lock):
    lock["debian_binaries"].append(copy.deepcopy(lock["debian_binaries"][0]))
    raw = json.dumps(lock).encode()
    sha1s = _fixture_sha1s(raw)
    sbom = selected.build_spdx(raw, sha1s=sha1s)
    assert sbom["packages"][0]["primaryPackagePurpose"] == "OPERATING_SYSTEM"
    packages = [p for p in sbom["packages"] if p["name"] == "libc6"]
    assert len(packages) == 1
    pkg = packages[0]
    assert pkg["versionInfo"] == "2.36-9+deb12u14"
    assert pkg["sourceInfo"] == "built package from: glibc 2.36-9+deb12u14"
    assert pkg["externalRefs"][0]["referenceLocator"] == (
        "pkg:deb/debian/libc6@2.36-9%2Bdeb12u14?arch=amd64&distro=debian-12"
    )
    assert pkg["filesAnalyzed"] is True
    assert (
        pkg["packageVerificationCode"]["packageVerificationCodeValue"]
        == hashlib.sha1(
            sha1s["usr/lib/libc.so.6"].encode(), usedforsecurity=False
        ).hexdigest()
    )
    assert "partial" in pkg["comment"] and "not installed" in pkg["comment"]
    assert pkg["licenseDeclared"] == pkg["licenseConcluded"] == "NOASSERTION"
    files = {f["fileName"]: f for f in sbom["files"]}
    assert files["./usr/lib/libc.so.6"]["checksums"][0]["checksumValue"] == sha(b"libc")
    for path, relation in [
        ("./usr/lib/libc.so.6", "CONTAINS"),
        ("./usr/share/doc/libc6/copyright", "OTHER"),
        ("./opt/ncore/base-sources/sources/glibc.tar.xz", "GENERATED_FROM"),
    ]:
        assert any(
            r["spdxElementId"] == pkg["SPDXID"]
            and r["relationshipType"] == relation
            and r["relatedSpdxElement"] == files[path]["SPDXID"]
            for r in sbom["relationships"]
        )


def test_conflicting_duplicate_package_is_rejected(lock):
    duplicate = copy.deepcopy(lock["debian_binaries"][0])
    duplicate["source"] = "debian:wrong@1"
    lock["debian_binaries"].append(duplicate)
    with pytest.raises(ValueError):
        selected.build_spdx(json.dumps(lock).encode())


def test_checked_in_lock_has_no_invented_source_packages():
    raw = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/ncore/base-source-lock.json"
    ).read_bytes()
    lock = json.loads(raw)
    sbom = selected.build_spdx(raw)
    actual = {
        (p["name"], p["versionInfo"]) for p in sbom["packages"] if p.get("externalRefs")
    }
    assert actual == {(p["name"], p["version"]) for p in lock["debian_binaries"]}


def archive(tmp_path, lock, damage=None):
    contents = {
        selected.LOCK_PATH: json.dumps(lock).encode(),
        "etc/os-release": b'ID=debian\nVERSION_ID="12"\nVERSION_CODENAME=bookworm\n',
        "usr/lib/libc.so.6": b"libc",
        "usr/share/doc/libc6/copyright": b"notice",
        "opt/ncore/base-sources/sources/glibc.tar.xz": b"source",
    }
    if damage:
        contents[damage] = b"changed"
    path = tmp_path / "rootfs.tar"
    with tarfile.open(path, "w") as tar:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return path


def test_export_checks_actual_files_and_delivered_source(tmp_path, lock):
    sbom = selected.from_archive(archive(tmp_path, lock))
    assert any(p["name"] == "libc6" for p in sbom["packages"])
    libc = next(f for f in sbom["files"] if f["fileName"] == "./usr/lib/libc.so.6")
    assert {
        "algorithm": "SHA1",
        "checksumValue": hashlib.sha1(b"libc", usedforsecurity=False).hexdigest(),
    } in libc["checksums"]


def test_lock_only_inventory_does_not_invent_required_spdx_sha1(lock):
    sbom = selected.build_spdx(json.dumps(lock).encode())
    assert not sbom["files"]
    evidence = json.loads(sbom["annotations"][0]["comment"])
    assert evidence["files"]["usr/lib/libc.so.6"] == {"sha256": sha(b"libc")}
    assert evidence["source"] == "lock-only; shipped bytes not verified"


def test_transformed_source_links_actual_delivered_hash(tmp_path, lock):
    original = sha(b"original source")
    lock["components"][0]["artifacts"] = [original]
    lock["artifacts"][0]["transformation"] = {"input_sha256": original}
    sbom = selected.from_archive(archive(tmp_path, lock))
    source = next(f for f in sbom["files"] if f["fileName"].endswith("/glibc.tar.xz"))
    assert source["checksums"][0]["checksumValue"] == sha(b"source")
    assert any(
        r["relatedSpdxElement"] == source["SPDXID"] and r["relationshipType"] == "OTHER"
        for r in sbom["relationships"]
    )


@pytest.mark.parametrize("changed", [False, True])
def test_selected_symlinks_are_verified_without_fake_hashes(tmp_path, lock, changed):
    lock["debian_binaries"][0]["files"].append(
        {"path": "usr/lib/libc.so", "link": "libc.so.6"}
    )
    path = archive(tmp_path, lock)
    with tarfile.open(path, "a") as tar:
        link = tarfile.TarInfo("usr/lib/libc.so")
        link.type = tarfile.SYMTYPE
        link.linkname = "wrong" if changed else "libc.so.6"
        tar.addfile(link)
    if changed:
        with pytest.raises(ValueError, match="symlink changed"):
            selected.from_archive(path)
    else:
        sbom = selected.from_archive(path)
        assert not any(f["fileName"] == "./usr/lib/libc.so" for f in sbom["files"])
        evidence = json.loads(sbom["annotations"][0]["comment"])
        assert evidence["files"]["usr/lib/libc.so"] == {"link": "libc.so.6"}


@pytest.mark.parametrize(
    "damage",
    [
        "usr/lib/libc.so.6",
        "usr/share/doc/libc6/copyright",
        "opt/ncore/base-sources/sources/glibc.tar.xz",
        "etc/os-release",
    ],
)
def test_export_refuses_lock_byte_or_distro_drift(tmp_path, lock, damage):
    with pytest.raises(ValueError):
        selected.from_archive(archive(tmp_path, lock, damage))


def report():
    return {
        "Metadata": {"OS": {"Family": "debian", "Name": "12"}},
        "Results": [
            {
                "Class": "os-pkgs",
                "Type": "debian",
                "Packages": [
                    {
                        "Name": "libc6",
                        "Version": "2.36-9+deb12u14",
                        "SrcName": "glibc",
                        "SrcVersion": "2.36",
                        "SrcRelease": "9+deb12u14",
                        "Arch": "amd64",
                    }
                ],
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-synthetic",
                        "PkgName": "libc6",
                        "InstalledVersion": "2.36-9+deb12u14",
                        "Severity": "CRITICAL",
                    }
                ],
            }
        ],
    }


def test_scan_counts_unfixed_critical_without_inventing_zero(lock):
    counts = selected.scan_counts(
        report(), selected.build_spdx(json.dumps(lock).encode())
    )
    assert counts == {
        "critical_total": 1,
        "critical_with_fix": 0,
        "critical_unfixed": 1,
        "secrets": 0,
        "packages_evaluated": 1,
    }


def test_scan_handles_epochs_and_deduplicates_packages_and_findings(lock):
    lock["debian_binaries"][0]["version"] = "1:2.36-9+deb12u14"
    lock["components"][0]["version"] = "1:2.36-9+deb12u14"
    lock["components"][0]["id"] = "debian:glibc@1:2.36-9+deb12u14"
    lock["debian_binaries"][0]["source"] = lock["components"][0]["id"]
    payload = report()
    result = payload["Results"][0]
    result["Packages"][0]["Version"] = "1:2.36-9+deb12u14"
    result["Packages"][0]["SrcEpoch"] = 1
    result["Packages"] *= 2
    result["Vulnerabilities"][0]["InstalledVersion"] = "1:2.36-9+deb12u14"
    result["Vulnerabilities"] *= 2
    counts = selected.scan_counts(
        payload, selected.build_spdx(json.dumps(lock).encode())
    )
    assert counts["critical_total"] == counts["packages_evaluated"] == 1


@pytest.mark.parametrize(
    "damage", ["empty", "distro", "missing", "version", "source", "fixed", "secret"]
)
def test_scan_refuses_incomplete_or_unsafe_report(lock, damage):
    payload = report()
    result = payload["Results"][0]
    if damage == "empty":
        payload["Results"] = []
    elif damage == "distro":
        payload["Metadata"]["OS"]["Name"] = "13"
    elif damage == "missing":
        result["Packages"] = []
    elif damage == "version":
        result["Packages"][0]["Version"] = "0"
    elif damage == "source":
        result["Packages"][0]["SrcName"] = "libc6"
    elif damage == "fixed":
        result["Vulnerabilities"][0]["FixedVersion"] = "next"
    else:
        result["Secrets"] = [{"RuleID": "synthetic"}]
    with pytest.raises(ValueError):
        selected.scan_counts(payload, selected.build_spdx(json.dumps(lock).encode()))


@pytest.mark.parametrize(
    "damage",
    ["missing-lock", "missing-file", "duplicate", "symlink", "traversal", "malformed"],
)
def test_archive_rejects_missing_or_ambiguous_bytes(tmp_path, lock, damage):
    path = archive(tmp_path, lock)
    if damage == "malformed":
        path.write_bytes(b"not a tar archive")
    else:
        with tarfile.open(path) as tar:
            contents = [(m, tar.extractfile(m).read()) for m in tar.getmembers()]
        if damage.startswith("missing"):
            missing = (
                selected.LOCK_PATH if damage == "missing-lock" else "usr/lib/libc.so.6"
            )
            contents = [(m, raw) for m, raw in contents if m.name != missing]
        if damage == "duplicate":
            contents.append(contents[-1])
        if damage == "symlink":
            contents[2][0].type = tarfile.SYMTYPE
            contents[2][0].linkname = "/outside"
            contents[2][0].size = 0
        if damage == "traversal":
            contents[2][0].name = "../outside"
        with tarfile.open(path, "w") as tar:
            for member, raw in contents:
                tar.addfile(member, io.BytesIO(raw))
    with pytest.raises(ValueError):
        selected.from_archive(path)


@pytest.mark.parametrize(
    "entry_type", [tarfile.LNKTYPE, tarfile.SYMTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE]
)
@pytest.mark.parametrize(
    "name",
    [selected.LOCK_PATH, "usr/lib/libc.so.6", selected.ANNEX + "sources/glibc.tar.xz"],
)
def test_regular_identities_refuse_link_and_special_entries(
    tmp_path, lock, name, entry_type
):
    path = archive(tmp_path, lock)
    with tarfile.open(path) as source:
        entries = [(member, source.extractfile(member).read()) for member in source]
    with tarfile.open(path, "w") as destination:
        for member, body in entries:
            if member.name == name:
                member.type, member.size = entry_type, 0
                member.linkname = "usr/share/doc/libc6/copyright"
                body = b""
            destination.addfile(member, io.BytesIO(body))
    with pytest.raises(ValueError, match="expected regular file"):
        selected.from_archive(path)


def test_archive_cannot_hide_selected_bytes_beneath_symlink(tmp_path, lock):
    path = archive(tmp_path, lock)
    with tarfile.open(path, "a") as output:
        member = tarfile.TarInfo("usr/lib")
        member.type, member.linkname = tarfile.SYMTYPE, "/elsewhere"
        output.addfile(member)
    with pytest.raises(ValueError, match="non-directory archive ancestor"):
        selected.from_archive(path)


@pytest.mark.parametrize(
    "field,value", [("Arch", "arm64"), ("SrcRelease", "other"), ("SrcEpoch", 2)]
)
def test_exact_package_coverage_includes_architecture_and_source_version(
    lock, field, value
):
    payload = report()
    payload["Results"][0]["Packages"][0][field] = value
    with pytest.raises(ValueError, match="coverage differs"):
        selected.scan_counts(payload, selected.build_spdx(json.dumps(lock).encode()))


@pytest.mark.parametrize("finding", ["Secrets", "Vulnerabilities", "Packages"])
def test_non_debian_result_cannot_hide_findings_or_extra_packages(lock, finding):
    payload = report()
    row = {"Class": "lang-pkgs", "Type": "python-pkg"}
    row[finding] = [{"Severity": "CRITICAL", "FixedVersion": "next"}]
    payload["Results"].append(row)
    with pytest.raises(ValueError):
        selected.scan_counts(payload, selected.build_spdx(json.dumps(lock).encode()))


IDENTITY = {
    "image_digest": "sha256:" + "a" * 64,
    "platform_digest": "sha256:" + "b" * 64,
    "config_digest": "sha256:" + "c" * 64,
}


def _scanner(monkeypatch, calls, *, docker=False, failure=None):
    from types import SimpleNamespace

    def run(command, **options):
        calls.append(command)
        directory = Path(options["cwd"])
        assert (directory / "trivy.yaml").read_text() == "{}\n"
        assert (directory / "ignore").read_text() == ""
        assert not any(key.startswith("TRIVY_") for key in options["env"])
        assert "--list-all-pkgs" in command and "--ignore-unfixed=false" in command
        assert (
            command[command.index("--severity") + 1]
            == "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL"
        )
        if docker:
            assert command[command.index("--volume") + 1] == f"{directory}:/npa-sbom:ro"
            assert command[-1] == "/npa-sbom/selected.spdx.json"
        else:
            assert command[-1] == str(directory / "selected.spdx.json")
        payload = report()
        if failure == "coverage":
            payload["Results"][0]["Packages"] = []
        return SimpleNamespace(
            returncode=int(failure == "trivy"), stdout=json.dumps(payload)
        )

    monkeypatch.setenv("TRIVY_SKIP_FILES", "everything")
    monkeypatch.setattr(selected.subprocess, "run", run)
    return ["docker", "run", "--rm", "trivy-image"] if docker else ["trivy"]


@pytest.mark.parametrize("docker", [False, True])
def test_archive_scan_retains_actual_spdx_report_and_receipt(
    tmp_path, lock, monkeypatch, docker
):
    calls = []
    command = _scanner(monkeypatch, calls, docker=docker)
    path = archive(tmp_path, lock)
    output = tmp_path / "evidence"
    receipt = selected.scan_archive(path, output, **IDENTITY, trivy_command=command)
    assert len(calls) == 1
    assert receipt["rootfs_sha256"] == sha(path.read_bytes())
    assert receipt["lock_sha256"] == sha(json.dumps(lock).encode())
    assert receipt["sbom_sha256"] == sha((output / "selected.spdx.json").read_bytes())
    assert receipt["report_sha256"] == sha(
        (output / "selected.trivy.json").read_bytes()
    )
    assert receipt["files_verified"] == 3 and receipt["symlinks_verified"] == 0
    assert receipt["packages_evaluated"] == receipt["critical_unfixed"] == 1
    assert receipt == json.loads((output / "selected.receipt.json").read_bytes())
    assert {key: receipt[key] for key in IDENTITY} == IDENTITY


@pytest.mark.parametrize("failure", ["trivy", "coverage"])
def test_failed_archive_scan_keeps_report_without_passing_receipt(
    tmp_path, lock, monkeypatch, failure
):
    command = _scanner(monkeypatch, [], failure=failure)
    output = tmp_path / "evidence"
    with pytest.raises(ValueError):
        selected.scan_archive(
            archive(tmp_path, lock), output, **IDENTITY, trivy_command=command
        )
    assert (output / "selected.trivy.json").is_file()
    assert not (output / "selected.receipt.json").exists()


def test_existing_evidence_directory_is_never_reused(tmp_path, lock):
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "selected.receipt.json").write_text("old receipt")
    with pytest.raises(FileExistsError):
        selected.scan_archive(
            archive(tmp_path, lock), output, **IDENTITY, trivy_command=["trivy"]
        )
    assert (output / "selected.receipt.json").read_text() == "old receipt"


@pytest.mark.parametrize("input_mode", ["--lock", "--rootfs-tar"])
def test_inventory_cli_cannot_produce_scan_receipt(
    tmp_path, lock, monkeypatch, input_mode
):
    import sys

    path = archive(tmp_path, lock)
    if input_mode == "--lock":
        path = tmp_path / "lock.json"
        path.write_text(json.dumps(lock))
    output = tmp_path / "selected.spdx.json"
    monkeypatch.setattr(
        sys, "argv", ["sbom", input_mode, str(path), "--output", str(output)]
    )
    selected.main()
    sbom = json.loads(output.read_text())
    if input_mode == "--lock":
        with pytest.raises(ValueError, match="lock-only"):
            selected._verified_counts(sbom)
    else:
        assert selected._verified_counts(sbom)["files_verified"] == 3
    assert not (tmp_path / "selected.receipt.json").exists()


def test_cli_refuses_lock_only_scan_before_scanner(tmp_path, lock, monkeypatch):
    import sys

    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(
        sys,
        "argv",
        ["sbom", "--lock", str(path), "--scan-output", str(tmp_path / "scan")],
    )
    with pytest.raises(SystemExit):
        selected.main()
    assert not (tmp_path / "scan").exists()


def test_cli_scan_uses_same_retained_evidence_api(tmp_path, lock, monkeypatch):
    import sys

    from npa.deploy import publish_public as publish

    command = _scanner(monkeypatch, [])
    monkeypatch.setattr(publish, "_trivy_command", lambda: command)
    path = archive(tmp_path, lock)
    output = tmp_path / "scan"
    args = ["sbom", "--rootfs-tar", str(path), "--scan-output", str(output)]
    for name, value in IDENTITY.items():
        args.extend(["--" + name.replace("_", "-"), value])
    monkeypatch.setattr(sys, "argv", args)
    selected.main()
    assert (
        json.loads((output / "selected.receipt.json").read_bytes())["status"] == "pass"
    )


def _publication_exporter(path, calls, scanner, failure):
    from types import SimpleNamespace

    def run(args, **options):
        if args[0] != "crane":
            return scanner(args, **options)
        calls.append(args)
        assert args[:-1] == [
            "crane",
            "export",
            "--platform",
            "linux/amd64",
            "example.invalid/ncore@" + IDENTITY["platform_digest"],
        ]
        Path(args[-1]).write_bytes(path.read_bytes())
        return SimpleNamespace(returncode=int(failure == "export"))

    return run


@pytest.mark.parametrize("failure", [None, "export", "coverage"])
def test_publisher_exports_exact_platform_and_scans_verified_bytes(
    tmp_path, lock, monkeypatch, failure
):
    from npa.deploy import publish_public as publish

    path = archive(tmp_path, lock)
    calls = []
    command = _scanner(monkeypatch, calls, failure=failure)
    run = _publication_exporter(path, calls, selected.subprocess.run, failure)

    monkeypatch.setattr(publish.subprocess, "run", run)
    monkeypatch.setattr(publish, "_trivy_command", lambda: command)
    args = {key: value for key, value in IDENTITY.items() if key != "image_digest"}
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            publish._scan_ncore_selected_base_exact_digest(
                "example.invalid/ncore@" + IDENTITY["image_digest"], **args
            )
        assert len(calls) == (1 if failure == "export" else 2)
        return
    receipt = publish._scan_ncore_selected_base_exact_digest(
        "example.invalid/ncore@" + IDENTITY["image_digest"], **args
    )
    assert len(calls) == 2 and receipt["lock_sha256"] == sha(json.dumps(lock).encode())


@pytest.mark.parametrize(
    "image", ["example.invalid/ncore:latest", "example.invalid/ncore@sha256:123"]
)
def test_publisher_refuses_mutable_or_invalid_digest_before_export(image, monkeypatch):
    from npa.deploy import publish_public as publish

    def run(*args, **kwargs):
        pytest.fail("must not export a mutable image")

    monkeypatch.setattr(publish.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exact digest"):
        publish._scan_ncore_selected_base_exact_digest(
            image,
            platform_digest=IDENTITY["platform_digest"],
            config_digest=IDENTITY["config_digest"],
        )
