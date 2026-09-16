#!/usr/bin/env python3
"""Verify a Habitat-Sim OCI archive without starting the image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

NPA_ROOT = Path(__file__).resolve().parents[3]
CHECKOUT_ROOT = NPA_ROOT.parent
sys.path.insert(0, str(NPA_ROOT / "scripts"))

from image_byte_scan import core as W  # noqa: E402
from image_byte_scan import habitat_sim_verification as H  # noqa: E402
from image_byte_scan import prepare as P  # noqa: E402


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


def _bind_source_contract(
    contract: dict[str, object], revision: str, manifest: bytes, manifest_sha256: str
) -> tuple[dict[str, object], dict[str, str]]:
    """Add exact revision, manifest, provenance, and input bytes to the scan."""

    W.require(
        hashlib.sha256(manifest).hexdigest() == manifest_sha256,
        "source_manifest_digest_mismatch",
    )
    expected = _parse_source_manifest(manifest)
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
    for destination, payload in SYSTEM_FILE_BYTES.items():
        paths.append(destination)
        files[destination] = hashlib.sha256(payload).hexdigest()
    metadata.update(
        {
            "/etc/group": {"kind": "file", "uid": 0, "gid": 0, "mode": 0o644},
            "/etc/passwd": {"kind": "file", "uid": 0, "gid": 0, "mode": 0o644},
            "/etc/ssh/sshd_config.d/99-npa-worker.conf": {
                "kind": "file",
                "uid": 0,
                "gid": 0,
                "mode": 0o644,
            },
            "/etc/sudoers.d/90-npa-skypilot": {
                "kind": "file",
                "uid": 0,
                "gid": 0,
                "mode": 0o440,
            },
            "/home/ubuntu/.ssh": {
                "kind": "directory",
                "uid": 1000,
                "gid": 1000,
                "mode": 0o700,
            },
        }
    )
    for path in metadata:
        if path not in paths:
            paths.append(path)
    return contract, expected


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
    prefix = PROVENANCE_ROOT + "/"
    graph = H.inspect(fd, length, expected_image_id)
    for layer_index, layer in enumerate(graph["layers"]):
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
                if not member.isfile():
                    findings.append(
                        {
                            "code": "source_provenance_nonregular_path",
                            "path": relative,
                        }
                    )
                    continue
                body = archive.extractfile(member)
                W.require(body is not None, "source_provenance_file_read")
                digest = hashlib.sha256(body.read()).hexdigest()
                if relative not in expected:
                    findings.append(
                        {"code": "source_provenance_unexpected_path", "path": relative}
                    )
                    continue
                if digest != expected[relative]:
                    findings.append(
                        {"code": "source_provenance_file_mismatch", "path": relative}
                    )
                observed[relative] = digest
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


def _verify_archive(args: argparse.Namespace) -> dict[str, object]:
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
        archive, fd, info = W.open_private_fd(args.oci_archive)
        try:
            archive_hash = P.binding(archive)["sha256"]
            report = H.verify(
                fd,
                info.st_size,
                args.expected_image_id,
                contract,
                archive_hash,
                args.expected_source_revision,
                args.expected_dpkg_inventory_sha256,
                args.expected_python_venv_inventory_sha256,
                args.expected_native_closure_sha256,
            )
            provenance_findings = _source_provenance_findings(
                fd,
                info.st_size,
                args.expected_image_id,
                expected_inputs,
                manifest,
                args.expected_source_revision,
                args.expected_npa_source_manifest_sha256,
            )
            report["findings"].extend(provenance_findings)
            report["valid"] = not report["findings"]
            report["npa_source_manifest_sha256"] = (
                args.expected_npa_source_manifest_sha256
            )
            report["npa_source_file_count"] = len(expected_inputs)
            return report
        finally:
            os.close(fd)


def _failure_report(code: str) -> dict[str, object]:
    return {
        "schema_version": H.SCHEMA,
        "valid": False,
        "findings": [{"code": code}],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the complete Habitat OCI verification command."""

    os.umask(0o077)
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
    args = parser.parse_args(argv)
    report: dict[str, object]
    try:
        report = _verify_archive(args)
    except W.ScanError as error:
        report = _failure_report(str(error))
    except W.INPUT_ERRORS:
        report = _failure_report("unreadable_or_incomplete_image_evidence")
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "Habitat-Sim complete image verification "
        + ("passed" if report["valid"] else "failed")
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
