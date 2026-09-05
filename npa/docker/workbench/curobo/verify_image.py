"""Verify every saved image layer against independently reviewed runtime bytes.

No container is started and no archive member is extracted to the filesystem.
Diagnostics use entry ordinals, never untrusted image paths or file contents.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


CONTRACT = Path(__file__).with_name("runtime-payload.json")
_SDK_NAME = re.compile(r"(?i)(?:cudnn\w*\.(?:h|hpp|hxx|cuh)|libcudnn\w*\.(?:a|lib))$")
_RUNTIME_NAME = re.compile(r"(?i)libcudnn\w*\.so(?:\.\d+)*$")


class ImageVerificationError(ValueError):
    """Saved image evidence cannot be completely interpreted."""


def _path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ImageVerificationError("unsafe archive path")
    return str(path)


def _hash_stream(stream):
    digest = hashlib.sha256()
    count = 0
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        count += len(chunk)
    return digest.hexdigest(), count


def _layer_diff_id(stream):
    signature = stream.read(2)
    stream.seek(0)
    decoded = gzip.GzipFile(fileobj=stream) if signature == b"\x1f\x8b" else stream
    return "sha256:" + _hash_stream(decoded)[0]


def verify_image(tarball: Path, *, expected_image_id: str, contract: dict | None = None) -> dict:
    """Read all regular bytes, preserving layer history and final whiteout state."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id):
        raise ImageVerificationError("require exact independently inspected image ID")
    contract = contract if contract is not None else json.loads(CONTRACT.read_text())
    if contract.get("schema_version") != "npa.curobo.runtime-payload.v1":
        raise ImageVerificationError("unsupported runtime payload contract")
    cudnn = contract["cudnn"]
    root = _path(cudnn["install_root"])
    expected = {f"{root}/{row['path']}": row for row in cudnn["retained"]}
    runtimes = [row for row in cudnn["retained"] if row["kind"] == "runtime"]
    if len(runtimes) != 8 or len(expected) != 9:
        raise ImageVerificationError("expected exactly eight runtimes and their license")
    notice = contract["nvshmem_notice"]
    expected[_path(notice["path"])] = notice
    excluded_hashes = {row["sha256"] for row in cudnn["excluded_sdk"]}
    runtime_hashes = {row["sha256"] for row in runtimes}
    if len(cudnn["excluded_sdk"]) != 14 or len({row["path"] for row in cudnn["excluded_sdk"]}) != 14:
        raise ImageVerificationError("expected complete reviewed SDK inventory")
    metadata_root = f"{root}/nvidia_cudnn_cu13-{cudnn['version']}.dist-info"
    metadata = {f"{metadata_root}/{name}" for name in (
        "METADATA", "WHEEL", "RECORD", "top_level.txt", "INSTALLER", "REQUESTED",
        "licenses/License.txt",
    )}
    ancestors = {str(parent) for path in expected for parent in PurePosixPath(path).parents}
    observed = {}
    findings = []
    entries_read = regular_files_read = content_bytes_read = 0

    def issue(code, layer, entry):
        findings.append({"code": code, "layer_index": layer, "entry_index": entry})

    with tarball.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with tarfile.open(tarball, "r:*") as archive:
        outer = archive.getmembers()
        members = {_path(member.name): member for member in outer}
        names = [_path(member.name) for member in outer]
        if len(set(names)) != len(names):
            raise ImageVerificationError("duplicate saved-image archive entry")
        manifest_member = members["manifest.json"]
        if not manifest_member.isfile():
            raise ImageVerificationError("saved-image manifest must be regular")
        manifest = json.load(archive.extractfile(manifest_member))
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ImageVerificationError("save exactly one image for verification")
        if not isinstance(manifest[0], dict):
            raise ImageVerificationError("saved-image manifest entry must be an object")
        layers = manifest[0].get("Layers")
        if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers):
            raise ImageVerificationError("saved-image layers must be complete and unique")
        config_member = members[_path(manifest[0]["Config"])]
        if not config_member.isfile():
            raise ImageVerificationError("saved-image config must be regular")
        config_bytes = archive.extractfile(config_member).read()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        if PurePosixPath(config_member.name).stem != config_hash or expected_image_id != "sha256:" + config_hash:
            raise ImageVerificationError("saved-image config digest mismatch")
        config = json.loads(config_bytes)
        if not isinstance(config, dict) or not isinstance(config.get("rootfs"), dict):
            raise ImageVerificationError("image config requires root filesystem identity")
        if config["rootfs"].get("type") != "layers":
            raise ImageVerificationError("unsupported image root filesystem")
        diff_ids = config.get("rootfs", {}).get("diff_ids")
        if not isinstance(diff_ids, list) or len(diff_ids) != len(layers):
            raise ImageVerificationError("config and complete layer population disagree")
        for layer_index, layer_name in enumerate(layers):
            member = members[_path(layer_name)]
            if not member.isfile():
                raise ImageVerificationError("saved-image layer must be regular")
            if _layer_diff_id(archive.extractfile(member)) != diff_ids[layer_index]:
                raise ImageVerificationError("saved-image layer diff ID mismatch")
            # Whiteouts affect lower layers, including when recorded after new files.
            current = {}
            seen_paths = set()
            with tarfile.open(fileobj=archive.extractfile(member), mode="r|*") as layer:
                for entry_index, entry in enumerate(layer):
                    entries_read += 1
                    path = _path(entry.name)
                    if path in seen_paths:
                        issue("duplicate_layer_path", layer_index, entry_index)
                    seen_paths.add(path)
                    basename = PurePosixPath(path).name
                    directory = str(PurePosixPath(path).parent)
                    whiteout = basename.startswith(".wh.")
                    malformed_whiteout = whiteout and (not entry.isfile() or entry.size != 0 or basename[4:] in {"", ".", ".."})
                    if malformed_whiteout:
                        issue("malformed_whiteout", layer_index, entry_index)
                    elif basename == ".wh..wh..opq":
                        observed = {p: v for p, v in observed.items() if directory != "." and not p.startswith(directory + "/")}
                    elif whiteout:
                        target = str(PurePosixPath(directory) / basename[4:])
                        observed = {p: v for p, v in observed.items() if p != target and not p.startswith(target + "/")}
                    else:
                        # Every type replacement invalidates proof of that exact file.
                        observed.pop(path, None)
                        current.pop(path, None)
                        if not entry.isdir():
                            observed = {p: v for p, v in observed.items() if not p.startswith(path + "/")}
                            current = {p: v for p, v in current.items() if not p.startswith(path + "/")}
                        # Required files may not be reached through links or a regular
                        # ancestor. Other system links are never followed or extracted.
                        if path in ancestors and not entry.isdir():
                            issue("required_payload_ancestor_not_directory", layer_index, entry_index)
                        if path in expected and not entry.isfile():
                            issue("retained_payload_not_regular", layer_index, entry_index)
                        # Ancestor bytes remain distributed even if later hidden.
                        if not entry.isdir():
                            if _SDK_NAME.fullmatch(basename):
                                issue("excluded_cudnn_sdk_path", layer_index, entry_index)
                            if "/nvidia/cudnn/" in "/" + path and path not in expected:
                                issue("unreviewed_cudnn_namespace_payload", layer_index, entry_index)
                            if _RUNTIME_NAME.fullmatch(basename) and path not in expected:
                                issue("unexpected_cudnn_runtime_location", layer_index, entry_index)
                            if "nvidia_cudnn_" in path and ".dist-info/" in path and path not in metadata:
                                issue("unreviewed_cudnn_distribution_payload", layer_index, entry_index)
                            if basename.lower().startswith("nvidia_cudnn") and basename.lower().endswith(".whl"):
                                issue("cached_cudnn_wheel", layer_index, entry_index)
                    if entry.isfile():
                        digest, size = _hash_stream(layer.extractfile(entry))
                        if size != entry.size:
                            raise ImageVerificationError("truncated layer member")
                        regular_files_read += 1
                        content_bytes_read += size
                        if digest in excluded_hashes:
                            issue("excluded_cudnn_sdk_bytes", layer_index, entry_index)
                        if digest in runtime_hashes and path not in expected:
                            issue("unexpected_cudnn_runtime_copy", layer_index, entry_index)
                        if path in expected and not whiteout:
                            matches = digest == expected[path]["sha256"] and size == expected[path]["size"]
                            current[path] = {"sha256": digest, "size": size, "matches": matches}
                            if not matches:
                                issue("retained_payload_hash_mismatch", layer_index, entry_index)
            observed.update(current)
        for path in expected:
            if path not in observed:
                # Expected names originate in reviewed code, never the image.
                findings.append({"code": "required_payload_missing", "expected_path": path})
    return {
        "schema_version": "npa.curobo.image-verification.v1",
        "valid": not findings,
        "docker_save_sha256": archive_hash,
        "image_config_digest": "sha256:" + config_hash,
        "verified_layer_diff_ids": diff_ids,
        "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "cudnn_source_wheel_sha256": cudnn["wheel_sha256"],
        "layer_count": len(layers),
        "entries_read": entries_read,
        "regular_files_read": regular_files_read,
        "content_bytes_read": content_bytes_read,
        "retained_runtime_count": sum(path in observed and observed[path]["matches"] for path in expected if expected[path].get("kind") == "runtime"),
        "required_payload_count": len(expected),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True, help="Exact docker image inspect .Id for the scanned image")
    args = parser.parse_args()
    try:
        report = verify_image(args.docker_save, expected_image_id=args.expected_image_id)
    except (OSError, EOFError, ValueError, KeyError, TypeError, tarfile.TarError):
        # Do not echo a malformed archive member or exception with private paths.
        report = {"schema_version": "npa.curobo.image-verification.v1", "valid": False,
                  "findings": [{"code": "unreadable_or_incomplete_image_evidence"}]}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("cuRobo complete-layer verification " + ("passed" if report["valid"] else "failed"))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
