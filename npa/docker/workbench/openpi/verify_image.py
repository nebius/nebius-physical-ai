"""Verify OpenPI's reviewed vendor boundary across every saved image layer.

No image executes, no archive member is extracted, and all regular-file bytes
are hashed. The general publication security scan remains independently required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

# Reuse only the existing format/identity decoder. OpenPI's policy and inventory
# remain separate; no cuRobo globals or classification behavior are changed.
_SPEC = importlib.util.spec_from_file_location(
    "npa_openpi_saved_image_format", Path(__file__).parents[1] / "curobo/verify_image.py"
)
_FORMAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORMAT)
CONTRACT = Path(__file__).with_name("runtime-payload.json")
_SDK = re.compile(r"(?i)(?:cudnn\w*\.(?:h|hpp|hxx|cuh)|libcudnn\w*\.(?:a|lib))$")
_RUNTIME = re.compile(r"(?i)libcudnn\w*\.so(?:\.\d+)*$")


def _identity(archive, members, manifest, expected_id):
    config_member = members[_FORMAT._path(manifest["Config"])]
    config_bytes = archive.extractfile(config_member).read() if config_member.isfile() else b""
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if PurePosixPath(config_member.name).stem != config_hash:
        raise ValueError("saved image config digest mismatch")
    config = json.loads(config_bytes)
    if not isinstance(config, dict) or not isinstance(config.get("rootfs"), dict):
        raise ValueError("image config requires a root filesystem identity")
    manifest_id, descriptors = _FORMAT._oci_graph(
        archive, members, config_member, config_hash, manifest["Layers"]
    )
    if expected_id not in {"sha256:" + config_hash, manifest_id}:
        raise ValueError("saved image identity differs from independent inspect")
    rootfs = config.get("rootfs", {})
    diff_ids = rootfs.get("diff_ids")
    if rootfs.get("type") != "layers" or not isinstance(diff_ids, list) or len(diff_ids) != len(manifest["Layers"]):
        raise ValueError("config and saved layer population differ")
    return "sha256:" + config_hash, manifest_id, descriptors, diff_ids


def _saved_graph(archive, expected_id):
    outer = archive.getmembers()
    members = {_FORMAT._path(member.name): member for member in outer}
    if len(members) != len(outer):
        raise ValueError("duplicate saved image archive member")
    manifest = _FORMAT._json_member(archive, members["manifest.json"])
    if not isinstance(manifest, list) or len(manifest) != 1 or not isinstance(manifest[0], dict):
        raise ValueError("save exactly one image")
    manifest = manifest[0]
    layers = manifest.get("Layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(name, str) for name in layers):
        raise ValueError("saved image layer population is incomplete")
    return members, layers, _identity(archive, members, manifest, expected_id)


def _verify_layer_identity(archive, member, layer_name, descriptor, diff_id):
    if not member.isfile():
        raise ValueError("saved image layer must be regular")
    digest, size = _FORMAT._hash_stream(archive.extractfile(member))
    if layer_name.startswith("blobs/") and layer_name != "blobs/sha256/" + digest:
        raise ValueError("saved image layer blob digest mismatch")
    if descriptor is not None:
        if descriptor["digest"] != "sha256:" + digest or descriptor["size"] != size:
            raise ValueError("OCI layer blob identity mismatch")
        gzip = archive.extractfile(member).read(2) == b"\x1f\x8b"
        if gzip != _FORMAT._LAYER_TYPES[descriptor["mediaType"]]:
            raise ValueError("OCI layer compression differs from its media type")
    if _FORMAT._layer_diff_id(archive.extractfile(member)) != diff_id:
        raise ValueError("saved image layer diff ID mismatch")


def _inventory(contract):
    if contract.get("schema_version") != "npa.openpi.runtime-payload.v1":
        raise ValueError("unsupported OpenPI inventory")
    expected = {_FORMAT._path(row["path"]): row for row in contract["required"]}
    if len(expected) != len(contract["required"]):
        raise ValueError("duplicate required OpenPI payload")
    kinds = [row["kind"] for row in expected.values()]
    if kinds.count("cudnn_runtime") != 8 or kinds.count("cudnn_license") != 1:
        raise ValueError("require exactly eight cuDNN runtimes and their notice")
    if len(contract["excluded_sdk"]) != 14 or len({row["path"] for row in contract["excluded_sdk"]}) != 14:
        raise ValueError("require all fourteen reviewed cuDNN SDK headers")
    if kinds.count("nvshmem_notice") != 1 or kinds.count("nccl_license") != 1 or kinds.count("torch_license") != 1:
        raise ValueError("required vendor and adapter notices are incomplete")
    if kinds.count("torch_adapter") != 53 or kinds.count("nvshmem_payload") != 56 or kinds.count("nccl_payload") != 2:
        raise ValueError("reviewed application dependency inventory is incomplete")
    for row in [*expected.values(), *contract["excluded_sdk"]]:
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("invalid exact-byte inventory")
    return expected


class _LayerAudit:
    def __init__(self, contract):
        self.expected = _inventory(contract)
        self.excluded = {row["sha256"] for row in contract["excluded_sdk"]}
        self.cudnn_wheel = contract["source_artifacts"]["cudnn"]["sha256"]
        self.runtimes = {row["sha256"] for row in self.expected.values() if row["kind"] == "cudnn_runtime"}
        self.ancestors = {str(parent) for path in self.expected for parent in PurePosixPath(path).parents}
        self.metadata = set(contract["cudnn_metadata_paths"])
        self.observed, self.current = {}, {}
        self.findings = []
        self.entries = self.files = self.bytes = 0

    def issue(self, code, layer, entry):
        self.findings.append({"code": code, "layer_index": layer, "entry_index": entry})

    def _whiteout(self, entry, path, layer_index, entry_index):
        # OCI whiteouts affect parent layers only. Same-layer additions remain
        # in current regardless of whether a marker precedes or follows them.
        basename = PurePosixPath(path).name
        if not basename.startswith(".wh."):
            return False
        directory = str(PurePosixPath(path).parent)
        if not entry.isfile() or entry.size != 0 or basename[4:] in {"", ".", ".."}:
            self.issue("malformed_whiteout", layer_index, entry_index)
        elif basename == ".wh..wh..opq":
            self.observed = {p: v for p, v in self.observed.items() if directory != "." and not p.startswith(directory + "/")}
        else:
            target = str(PurePosixPath(directory) / basename[4:])
            self.observed = {p: v for p, v in self.observed.items() if p != target and not p.startswith(target + "/")}
        return True

    def _entry_policy(self, entry, path, layer_index, entry_index):
        self.observed.pop(path, None)
        self.current.pop(path, None)
        if not entry.isdir():
            self.observed = {p: v for p, v in self.observed.items() if not p.startswith(path + "/")}
            self.current = {p: v for p, v in self.current.items() if not p.startswith(path + "/")}
        if path in self.ancestors and not entry.isdir():
            self.issue("required_payload_ancestor_not_directory", layer_index, entry_index)
        if path in self.expected and not entry.isfile():
            self.issue("retained_payload_not_regular", layer_index, entry_index)
        if entry.isdir():
            return
        basename = PurePosixPath(path).name
        adapter = self.expected.get(path, {}).get("kind") == "torch_adapter"
        if _SDK.fullmatch(basename) and not adapter:
            self.issue("excluded_cudnn_sdk_path", layer_index, entry_index)
        if "/nvidia/cudnn/" in "/" + path and path not in self.expected:
            self.issue("unreviewed_cudnn_namespace_payload", layer_index, entry_index)
        if _RUNTIME.fullmatch(basename) and path not in self.expected:
            self.issue("unexpected_cudnn_runtime_location", layer_index, entry_index)
        if "nvidia_cudnn_" in path and ".dist-info/" in path and path not in self.metadata:
            self.issue("unreviewed_cudnn_distribution_payload", layer_index, entry_index)
        if basename.lower().startswith("nvidia_cudnn") and basename.lower().endswith(".whl"):
            self.issue("cached_cudnn_wheel", layer_index, entry_index)

    def _regular_bytes(self, layer, entry, path, layer_index, entry_index, whiteout):
        digest, size = _FORMAT._hash_stream(layer.extractfile(entry))
        if size != entry.size:
            raise ValueError("truncated layer member")
        self.files += 1
        self.bytes += size
        if digest in self.excluded:
            self.issue("excluded_cudnn_sdk_bytes", layer_index, entry_index)
        if digest == self.cudnn_wheel:
            self.issue("cached_cudnn_wheel_bytes", layer_index, entry_index)
        if digest in self.runtimes and path not in self.expected:
            self.issue("unexpected_cudnn_runtime_copy", layer_index, entry_index)
        if path in self.expected and not whiteout:
            matches = digest == self.expected[path]["sha256"] and size == self.expected[path]["size"]
            self.current[path] = matches
            if not matches:
                self.issue("retained_payload_hash_mismatch", layer_index, entry_index)

    def scan(self, stream, layer_index):
        self.current = {}
        seen = set()
        with _FORMAT._layer_stream(stream) as decoded, tarfile.open(fileobj=decoded, mode="r|") as layer:
            for entry_index, entry in enumerate(layer):
                self.entries += 1
                path = _FORMAT._path(entry.name)
                if path in seen:
                    self.issue("duplicate_layer_path", layer_index, entry_index)
                seen.add(path)
                whiteout = self._whiteout(entry, path, layer_index, entry_index)
                if not whiteout:
                    self._entry_policy(entry, path, layer_index, entry_index)
                if entry.isfile():
                    self._regular_bytes(layer, entry, path, layer_index, entry_index, whiteout)
        self.observed.update(self.current)

    def report(self):
        for path in self.expected:
            if path not in self.observed:
                self.findings.append({"code": "required_payload_missing", "expected_path": path})
        return {
            "valid": not self.findings, "findings": self.findings,
            "entries_read": self.entries, "regular_files_read": self.files,
            "content_bytes_read": self.bytes, "required_payload_count": len(self.expected),
            "retained_runtime_count": sum(self.observed.get(path, False) for path, row in self.expected.items() if row["kind"] == "cudnn_runtime"),
            "verified_torch_adapter_count": sum(self.observed.get(path, False) for path, row in self.expected.items() if row["kind"] == "torch_adapter"),
        }


def verify_image(tarball: Path, *, expected_image_id: str, contract: dict | None = None) -> dict:
    """Bind the image identity and inspect its complete ordered layer population.

    Args:
        tarball: Docker-save archive of exactly one independently inspected image.
        expected_image_id: Exact manifest or config digest from that inspection.
        contract: Reviewed byte inventory; the committed OpenPI inventory is default.

    Returns:
        Aggregate byte coverage, identity and findings without untrusted payload text.

    Raises:
        ValueError: The identity, archive or reviewed inventory is invalid.
        OSError: The saved archive cannot be read.
    """
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id):
        raise ValueError("require exact independently inspected image ID")
    contract = contract if contract is not None else json.loads(CONTRACT.read_text())
    audit = _LayerAudit(contract)
    with tarball.open("rb") as stream:
        archive_hash, _ = _FORMAT._hash_stream(stream)
    with tarfile.open(tarball, "r:*") as archive:
        members, layers, identity = _saved_graph(archive, expected_image_id)
        config_id, manifest_id, descriptors, diff_ids = identity
        for index, name in enumerate(layers):
            member = members[_FORMAT._path(name)]
            _verify_layer_identity(archive, member, name, descriptors[index] if descriptors else None, diff_ids[index])
            audit.scan(archive.extractfile(member), index)
    return {
        "schema_version": "npa.openpi.image-verification.v1", **audit.report(),
        "docker_save_sha256": archive_hash, "expected_image_id": expected_image_id,
        "image_config_digest": config_id, "image_manifest_digest": manifest_id,
        "verified_layer_diff_ids": diff_ids, "layer_count": len(layers),
        "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "cudnn_source_wheel_sha256": contract["source_artifacts"]["cudnn"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    args = parser.parse_args()
    try:
        report = verify_image(args.docker_save, expected_image_id=args.expected_image_id)
    except (OSError, EOFError, ValueError, KeyError, TypeError, tarfile.TarError):
        report = {"schema_version": "npa.openpi.image-verification.v1", "valid": False,
                  "findings": [{"code": "unreadable_or_incomplete_image_evidence"}]}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("OpenPI complete-layer verification " + ("passed" if report["valid"] else "failed"))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
