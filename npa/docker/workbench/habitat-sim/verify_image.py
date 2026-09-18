#!/usr/bin/env python3
"""Verify a Habitat-Sim OCI archive without starting the image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import uuid
from contextlib import contextmanager
from pathlib import Path

NPA_ROOT = Path(__file__).resolve().parents[3]
CHECKOUT_ROOT = NPA_ROOT.parent
sys.path.insert(0, str(NPA_ROOT / "scripts"))

from image_byte_scan import core as W  # noqa: E402
from image_byte_scan import habitat_sim_verification as H  # noqa: E402


NPA_SOURCE_PATHS = (
    "docker/workbench/habitat-sim/Dockerfile",
    "docker/workbench/habitat-sim/REDISTRIBUTION.md",
    "docker/workbench/habitat-sim/THIRD_PARTY_NOTICES.md",
    "docker/workbench/habitat-sim/apt-build.lock",
    "docker/workbench/habitat-sim/apt-runtime.lock",
    "docker/workbench/habitat-sim/build.sh",
    "docker/workbench/habitat-sim/entrypoint.sh",
    "docker/workbench/habitat-sim/licenses.json",
    "docker/workbench/habitat-sim/prepare_source.py",
    "docker/workbench/habitat-sim/requirements-build.lock",
    "docker/workbench/habitat-sim/requirements-runtime.lock",
    "docker/workbench/habitat-sim/runtime-payload.json",
    "docker/workbench/habitat-sim/source-manifest.json",
    "docker/workbench/habitat-sim/verify_apt_artifacts.sh",
    "docker/workbench/habitat-sim/verify_image.py",
    "docker/workbench/packaging-contract.yaml",
    "src/npa/__init__.py",
    "src/npa/workflows/__init__.py",
    "src/npa/workflows/habitat_sim_smoke.py",
)
PROVENANCE_ROOT = "usr/share/doc/npa-habitat-sim/npa-source-provenance"
PROVENANCE_SCHEMA = "npa.source-provenance.v1"
EXECUTABLE_SOURCE_DESTINATIONS = {
    "inputs/src/npa/__init__.py": ("/opt/npa-runtime/npa/__init__.py", 0o644),
    "inputs/src/npa/workflows/__init__.py": (
        "/opt/npa-runtime/npa/workflows/__init__.py",
        0o644,
    ),
    "inputs/src/npa/workflows/habitat_sim_smoke.py": (
        "/opt/npa-runtime/npa/workflows/habitat_sim_smoke.py",
        0o644,
    ),
    "inputs/docker/workbench/habitat-sim/entrypoint.sh": (
        "/usr/local/bin/npa-habitat-entrypoint",
        0o755,
    ),
}
SYSTEM_FILE_BYTES = {
    "/etc/ssh/sshd_config.d/99-npa-worker.conf": (
        b"PasswordAuthentication no\nPermitRootLogin no\n"
    ),
    "/etc/sudoers.d/90-npa-skypilot": b"ubuntu ALL=(ALL) NOPASSWD:ALL\n",
}


def _validate_source_contract(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    rows = manifest["source"]["required_projection_files"]
    expected = {f"/usr/src/habitat-sim/{row['path']}": row["sha256"] for row in rows}
    prefix = "/usr/src/habitat-sim/data/pbr/"
    observed = {
        path: digest
        for path, digest in contract["required_final_file_sha256"].items()
        if path.startswith(prefix)
    }
    if observed != expected:
        raise ValueError("runtime PBR byte contract differs from source manifest")
    required_paths = set(contract["required_final_paths"])
    if not expected.keys() <= required_paths:
        raise ValueError("runtime PBR path contract is incomplete")
    notice = "/usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md"
    if (
        notice not in required_paths
        or notice not in contract["required_final_file_sha256"]
    ):
        raise ValueError("runtime PBR notice contract is incomplete")


def _load_contract() -> dict[str, object]:
    package = Path(__file__).resolve().parent
    contract = json.loads((package / "runtime-payload.json").read_text())
    manifest = json.loads((package / "source-manifest.json").read_text())
    _validate_source_contract(contract, manifest)
    return contract


def _source_manifest_from_git(revision: str) -> bytes:
    """Recreate the build manifest from the exact committed Git objects."""

    head = subprocess.run(
        ["git", "-C", str(CHECKOUT_ROOT), "rev-parse", "--verify", "HEAD^{commit}"],
        check=False,
        capture_output=True,
        text=True,
    )
    W.require(head.returncode == 0, "trusted_source_revision_unreadable")
    W.require(head.stdout.strip() == revision, "trusted_source_revision_mismatch")
    lines: list[bytes] = []
    for path in NPA_SOURCE_PATHS:
        repository_path = f"npa/{path}"
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(CHECKOUT_ROOT),
                "cat-file",
                "blob",
                f"{revision}:{repository_path}",
            ],
            check=False,
            capture_output=True,
        )
        W.require(blob.returncode == 0, "trusted_source_object_missing")
        try:
            working = (NPA_ROOT / path).read_bytes()
        except OSError:
            raise W.ScanError("trusted_source_worktree_unreadable") from None
        W.require(working == blob.stdout, "trusted_source_worktree_mismatch")
        digest = hashlib.sha256(blob.stdout).hexdigest()
        lines.append(f"{digest}  inputs/{path}\n".encode())
    return b"".join(sorted(lines, key=lambda row: row.split(b"  ", 1)[1]))


def _provenance_bytes(revision: str, manifest_sha256: str) -> bytes:
    return (
        '{"manifest_sha256":"%s","schema_version":"%s",'
        '"source_revision":"%s"}\n' % (manifest_sha256, PROVENANCE_SCHEMA, revision)
    ).encode()


def _parse_source_manifest(manifest: bytes) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in manifest.splitlines():
        match = re.fullmatch(rb"([0-9a-f]{64})  (inputs/[A-Za-z0-9_./-]+)", raw)
        W.require(match is not None, "source_manifest_row_invalid")
        digest = match.group(1).decode()
        path = match.group(2).decode()
        W.require(path not in expected, "source_manifest_duplicate_path")
        expected[path] = digest
    W.require(
        tuple(path.removeprefix("inputs/") for path in expected) == NPA_SOURCE_PATHS,
        "source_manifest_path_set_mismatch",
    )
    return expected


def _bind_provenance_files(
    contract: dict, revision: str, expected: dict[str, str], manifest_sha256: str
) -> None:
    """Bind the attested manifest and provenance population to image paths."""
    provenance = _provenance_bytes(revision, manifest_sha256)
    labels = contract["required_labels"]
    labels["org.nebius.npa.source-manifest-sha256"] = manifest_sha256
    labels["org.nebius.npa.source-provenance-schema"] = PROVENANCE_SCHEMA
    labels["org.nebius.npa.sudo-bootstrap-contract"] = "habitat-sim-skypilot-0.12.2-v1"
    files = contract["required_final_file_sha256"]
    paths = contract["required_final_paths"]
    manifest_path = f"/{PROVENANCE_ROOT}/npa-source-manifest.sha256"
    provenance_path = f"/{PROVENANCE_ROOT}/npa-source-provenance.json"
    for path in (f"/{PROVENANCE_ROOT}", manifest_path, provenance_path):
        if path not in paths:
            paths.append(path)
    files[manifest_path] = manifest_sha256
    files[provenance_path] = hashlib.sha256(provenance).hexdigest()
    for path, digest in expected.items():
        image_path = f"/{PROVENANCE_ROOT}/{path}"
        paths.append(image_path)
        files[image_path] = digest


def _bind_executable_sources(contract: dict, expected: dict[str, str]) -> None:
    """Bind installed entrypoints to the same verified source input bytes."""
    paths = contract["required_final_paths"]
    files = contract["required_final_file_sha256"]
    metadata = contract.setdefault("required_final_metadata", {})
    bindings = contract.setdefault("executable_source_bindings", {})
    for source, (destination, mode) in EXECUTABLE_SOURCE_DESTINATIONS.items():
        W.require(source in expected, "executable_source_input_missing")
        digest = expected[source]
        if destination not in paths:
            paths.append(destination)
        files[destination] = digest
        metadata[destination] = {
            "kind": "file",
            "uid": 0,
            "gid": 0,
            "mode": mode,
        }
        bindings[destination] = {**metadata[destination], "sha256": digest}


def _bind_bootstrap_files(contract: dict) -> None:
    """Preserve the exact account, SSH and sudo bootstrap byte/mode contract."""
    paths = contract["required_final_paths"]
    files = contract["required_final_file_sha256"]
    metadata = contract.setdefault("required_final_metadata", {})
    for destination, payload in SYSTEM_FILE_BYTES.items():
        paths.append(destination)
        files[destination] = hashlib.sha256(payload).hexdigest()
    for path, kind, owner, mode in (
        ("/etc/group", "file", 0, 0o644),
        ("/etc/passwd", "file", 0, 0o644),
        ("/etc/ssh/sshd_config.d/99-npa-worker.conf", "file", 0, 0o644),
        ("/etc/sudoers.d/90-npa-skypilot", "file", 0, 0o440),
        ("/home/ubuntu/.ssh", "directory", 1000, 0o700),
    ):
        metadata[path] = {"kind": kind, "uid": owner, "gid": owner, "mode": mode}
    for path in metadata:
        if path not in paths:
            paths.append(path)


def _bind_source_contract(
    contract: dict[str, object], revision: str, manifest: bytes, manifest_sha256: str
) -> tuple[dict[str, object], dict[str, str]]:
    """Add exact revision, manifest, provenance, and input bytes to the scan."""
    W.require(
        hashlib.sha256(manifest).hexdigest() == manifest_sha256,
        "source_manifest_digest_mismatch",
    )
    expected = _parse_source_manifest(manifest)
    _bind_provenance_files(contract, revision, expected, manifest_sha256)
    _bind_executable_sources(contract, expected)
    _bind_bootstrap_files(contract)
    return contract, expected


def _record_provenance_member(archive, member, relative, expected, observed, findings):
    """Check one regular provenance member without extracting it to disk."""
    if not member.isfile():
        findings.append({"code": "source_provenance_nonregular_path", "path": relative})
        return
    body = archive.extractfile(member)
    W.require(body is not None, "source_provenance_file_read")
    digest = hashlib.sha256(body.read()).hexdigest()
    if relative not in expected:
        findings.append({"code": "source_provenance_unexpected_path", "path": relative})
        return
    if digest != expected[relative]:
        findings.append({"code": "source_provenance_file_mismatch", "path": relative})
    observed[relative] = digest


def _inspect_provenance_layer(fd, layer, layer_index, expected, observed, findings):
    """Account for each provenance-layer member, including removals and types."""
    prefix = PROVENANCE_ROOT + "/"
    with tarfile.open(fileobj=H._decoded(fd, layer), mode="r|") as archive:
        for member in archive:
            path = W.safe_name(member.name)
            if not path.startswith(prefix):
                continue
            relative = path.removeprefix(prefix)
            if not relative or member.isdir():
                continue
            if "/.wh." in f"/{relative}" or relative.startswith(".wh."):
                findings.append(
                    {"code": "source_provenance_whiteout", "layer": layer_index}
                )
                continue
            _record_provenance_member(
                archive, member, relative, expected, observed, findings
            )


def _source_provenance_findings(
    fd: int,
    length: int,
    expected_image_id: str,
    expected_inputs: dict[str, str],
    manifest: bytes,
    revision: str,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    """Require the exact source-provenance population in every image layer."""

    expected = {
        "npa-source-manifest.sha256": manifest_sha256,
        "npa-source-provenance.json": hashlib.sha256(
            _provenance_bytes(revision, manifest_sha256)
        ).hexdigest(),
        **expected_inputs,
    }
    observed: dict[str, str] = {}
    findings: list[dict[str, object]] = []
    graph = H.inspect(fd, length, expected_image_id)
    for layer_index, layer in enumerate(graph["layers"]):
        _inspect_provenance_layer(fd, layer, layer_index, expected, observed, findings)
    for path in sorted(set(expected) - set(observed)):
        findings.append({"code": "source_provenance_path_missing", "path": path})
    if observed.get("npa-source-manifest.sha256") == manifest_sha256:
        # Binding the bytes here is independent of trusting the OCI label.
        W.require(
            hashlib.sha256(manifest).hexdigest() == manifest_sha256,
            "source_manifest_digest_mismatch",
        )
    return findings


def _require_root(path: Path, missing_code: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        raise W.ScanError(missing_code) from None


class _ArchiveIdentityError(W.ScanError):
    """Refuse report publication when the held archive identity is unstable."""


def _confirm_archive_descriptor(fd: int, initial: os.stat_result) -> None:
    """Compare the held object's device, inode, bytes, times and ownership."""
    if W.stat_fingerprint(os.fstat(fd)) != W.stat_fingerprint(initial):
        raise _ArchiveIdentityError("archive_changed_during_verification")


@contextmanager
def _bound_archive_descriptor(path: Path):
    """Hash and verify one private descriptor without reopening its pathname."""
    _, fd, initial = W.open_private_fd(path)
    try:
        _confirm_archive_descriptor(fd, initial)
        digest = W.descriptor_digest(fd)
        _confirm_archive_descriptor(fd, initial)
        yield fd, initial.st_size, digest
        _confirm_archive_descriptor(fd, initial)
    finally:
        os.close(fd)


def _archive_report(args, fd, length, digest, contract, expected_inputs, manifest):
    """Combine complete-byte and per-layer source verification on the same fd."""
    report = H.verify(
        fd,
        length,
        args.expected_image_id,
        contract,
        digest,
        args.expected_source_revision,
        args.expected_dpkg_inventory_sha256,
        args.expected_python_venv_inventory_sha256,
        args.expected_native_closure_sha256,
    )
    report["findings"].extend(
        _source_provenance_findings(
            fd,
            length,
            args.expected_image_id,
            expected_inputs,
            manifest,
            args.expected_source_revision,
            args.expected_npa_source_manifest_sha256,
        )
    )
    report["valid"] = not report["findings"]
    report["npa_source_manifest_sha256"] = args.expected_npa_source_manifest_sha256
    report["npa_source_file_count"] = len(expected_inputs)
    return report


def _verify_archive(args: argparse.Namespace) -> dict[str, object]:
    """Verify trusted source and complete archive evidence under private roots."""
    _require_root(args.analysis_root, "analysis_root_missing")
    _require_root(args.trusted_root, "trusted_root_missing")
    with W.authorized_roots(args.analysis_root, args.trusted_root) as roots:
        W.require(roots[1] == CHECKOUT_ROOT, "trusted_source_root_mismatch")
        manifest = _source_manifest_from_git(args.expected_source_revision)
        W.require(
            re.fullmatch(r"[0-9a-f]{64}", args.expected_npa_source_manifest_sha256)
            is not None,
            "source_manifest_expected_digest_invalid",
        )
        contract, expected_inputs = _bind_source_contract(
            _load_contract(),
            args.expected_source_revision,
            manifest,
            args.expected_npa_source_manifest_sha256,
        )
        with _bound_archive_descriptor(args.oci_archive) as (fd, length, digest):
            return _archive_report(
                args, fd, length, digest, contract, expected_inputs, manifest
            )


def _failure_report(code: str) -> dict[str, object]:
    return {
        "schema_version": H.SCHEMA,
        "valid": False,
        "findings": [{"code": code}],
    }


def _require_private_directory(fd: int) -> None:
    info = os.fstat(fd)
    W.require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.geteuid()
        and not info.st_mode & 0o077,
        "report_directory_permissions",
    )


def _open_report_parent(analysis_root: Path, output: Path) -> int:
    root, output = analysis_root.absolute(), output.absolute()
    W.require(".." not in root.parts + output.parts, "report_parent_component")
    W.require(
        output.parent.is_relative_to(root) and not root.is_relative_to(CHECKOUT_ROOT),
        "report_output_scope",
    )
    W.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output.name), "report_name")
    _require_root(root, "analysis_root_missing")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        _require_private_directory(fd)
        for component in output.parent.relative_to(root).parts:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            _require_private_directory(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _confirm_report_parent(args: argparse.Namespace, held: int) -> None:
    current = _open_report_parent(args.analysis_root, args.json)
    try:
        expected, observed = os.fstat(held), os.fstat(current)
        W.require(
            (expected.st_dev, expected.st_ino) == (observed.st_dev, observed.st_ino),
            "report_directory_changed",
        )
    finally:
        os.close(current)


def _require_report_absent(parent: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise W.ScanError("report_output_exists")


def _close_report_descriptor(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        print("Habitat-Sim report descriptor cleanup failed", file=sys.stderr)


@contextmanager
def _report_destination(args: argparse.Namespace):
    parent = _open_report_parent(args.analysis_root, args.json)
    try:
        _require_report_absent(parent, args.json.name)
        yield parent
    finally:
        _close_report_descriptor(parent)


def _cleanup_owned_report(parent: int, name: str, held: int) -> None:
    try:
        expected = os.fstat(held)
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
            print("Habitat-Sim report cleanup identity changed", file=sys.stderr)
            return
        os.unlink(name, dir_fd=parent)
    except FileNotFoundError:
        return
    except OSError:
        print("Habitat-Sim owned report cleanup failed", file=sys.stderr)


@contextmanager
def _temporary_report(parent: int):
    name = f".habitat-report-{uuid.uuid4().hex}.pending"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, 0o600, dir_fd=parent)
    try:
        yield name, fd
    finally:
        _cleanup_owned_report(parent, name, fd)
        _close_report_descriptor(fd)


def _write_report_bytes(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        W.require(written > 0, "report_short_write")
        remaining = remaining[written:]
    os.fsync(fd)


def _confirm_report(parent: int, name: str, held: int, payload: bytes) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    fd = os.open(name, flags, dir_fd=parent)
    try:
        expected, before = os.fstat(held), os.fstat(fd)
        W.require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_nlink == 2
            and W.stat_fingerprint(expected) == W.stat_fingerprint(before),
            "report_identity_changed",
        )
        W.require(W.descriptor_bytes(fd) == payload, "report_bytes_changed")
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        W.require(
            W.stat_fingerprint(before)
            == W.stat_fingerprint(os.fstat(fd))
            == W.stat_fingerprint(after),
            "report_identity_changed",
        )
    finally:
        _close_report_descriptor(fd)


def _publish_report(args: argparse.Namespace, parent: int, report: dict) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    _confirm_report_parent(args, parent)
    with _temporary_report(parent) as (temporary, fd):
        linked = verified = False
        try:
            _write_report_bytes(fd, payload)
            _confirm_report_parent(args, parent)
            temporary_info = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
            W.require(
                W.stat_fingerprint(temporary_info) == W.stat_fingerprint(os.fstat(fd)),
                "report_temporary_changed",
            )
            os.link(
                temporary,
                args.json.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            linked = True
            os.fsync(parent)
            _confirm_report(parent, args.json.name, fd, payload)
            _confirm_report_parent(args, parent)
            verified = True
        finally:
            if linked and not verified:
                _cleanup_owned_report(parent, args.json.name, fd)


def _verification_report(args: argparse.Namespace) -> dict[str, object]:
    try:
        return _verify_archive(args)
    except _ArchiveIdentityError:
        raise
    except W.ScanError as error:
        return _failure_report(str(error))
    except W.INPUT_ERRORS as error:
        report = _failure_report("unreadable_or_incomplete_image_evidence")
        report["scanner_error_type"] = type(error).__name__
        print(f"Habitat-Sim scanner failure: {type(error).__name__}", file=sys.stderr)
        return report


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--oci-archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-npa-source-manifest-sha256", required=True)
    parser.add_argument("--expected-dpkg-inventory-sha256", required=True)
    parser.add_argument("--expected-python-venv-inventory-sha256", required=True)
    parser.add_argument("--expected-native-closure-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify an OCI archive and publish one owner-scoped, non-overwriting report.

    Args:
        argv: Command arguments; defaults to the process arguments.
    Returns:
        Zero only after valid evidence and verified report publication, else one.
    Raises:
        SystemExit: Argument parsing rejects an invalid command line.
    """
    os.umask(0o077)
    args = _arguments(argv)
    try:
        with _report_destination(args) as parent:
            report = _verification_report(args)
            _publish_report(args, parent, report)
    except W.ScanError as error:
        print(f"Habitat-Sim report refused: {error}", file=sys.stderr)
        return 1
    except W.INPUT_ERRORS as error:
        print(f"Habitat-Sim report failure: {type(error).__name__}", file=sys.stderr)
        return 1
    print(
        "Habitat-Sim complete image verification "
        + ("passed" if report["valid"] else "failed")
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
