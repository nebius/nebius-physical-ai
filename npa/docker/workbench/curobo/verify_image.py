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
CLEAN_ROOT_CONFIG = Path(__file__).with_name("clean-root-config.json")
_SDK_NAME = re.compile(r"(?i)(?:cudnn\w*\.(?:h|hpp|hxx|cuh)|libcudnn\w*\.(?:a|lib))$")
_RUNTIME_NAME = re.compile(r"(?i)libcudnn\w*\.so(?:\.\d+)*$")
_MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
_CONFIG_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
_LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar": False,
    "application/vnd.oci.image.layer.v1.tar+gzip": True,
    "application/vnd.docker.image.rootfs.diff.tar": False,
    "application/vnd.docker.image.rootfs.diff.tar.gzip": True,
}


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


def _layer_stream(stream):
    signature = stream.read(2)
    stream.seek(0)
    return gzip.GzipFile(fileobj=stream) if signature == b"\x1f\x8b" else stream


def _layer_diff_id(stream):
    with _layer_stream(stream) as decoded:
        return "sha256:" + _hash_stream(decoded)[0]


def _descriptor_member(descriptor, members, media_types):
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("mediaType") not in media_types
    ):
        raise ImageVerificationError("unsupported image descriptor media type")
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ImageVerificationError("invalid image descriptor digest")
    if type(size) is not int or size < 0:
        raise ImageVerificationError("invalid image descriptor size")
    member = members["blobs/sha256/" + digest.removeprefix("sha256:")]
    if not member.isfile() or member.size != size:
        raise ImageVerificationError("image descriptor blob size or type mismatch")
    return member


def _json_member(archive, member):
    if not member.isfile():
        raise ImageVerificationError("image metadata must be regular")
    return json.load(archive.extractfile(member))


def _oci_graph(archive, members, config_member, config_hash, layers):
    """Bind Docker's compatibility manifest to the single OCI descriptor graph.

    Docker's containerd store reports the manifest digest as inspect .Id. Classic
    stores report the config digest. Both identities must bind the same bytes;
    annotations alone are never evidence. Repeated layer *references* are valid.
    """
    if "index.json" not in members and "oci-layout" not in members:
        return None, None
    layout = _json_member(archive, members["oci-layout"])
    index = _json_member(archive, members["index.json"])
    if not isinstance(layout, dict) or layout.get("imageLayoutVersion") != "1.0.0":
        raise ImageVerificationError("unsupported OCI image layout")
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 2
        or index.get("mediaType", "application/vnd.oci.image.index.v1+json")
        != "application/vnd.oci.image.index.v1+json"
        or not isinstance(index.get("manifests"), list)
        or len(index["manifests"]) != 1
    ):
        raise ImageVerificationError("save exactly one OCI image manifest")
    descriptor = index["manifests"][0]
    member = _descriptor_member(descriptor, members, _MANIFEST_TYPES)
    data = archive.extractfile(member).read()
    if "sha256:" + hashlib.sha256(data).hexdigest() != descriptor["digest"]:
        raise ImageVerificationError("OCI manifest digest mismatch")
    manifest = json.loads(data)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 2
        or manifest.get("mediaType") != descriptor["mediaType"]
    ):
        raise ImageVerificationError("OCI manifest schema or media type mismatch")
    described_config = _descriptor_member(
        manifest.get("config"), members, _CONFIG_TYPES
    )
    if (
        described_config is not config_member
        or manifest["config"]["digest"] != "sha256:" + config_hash
    ):
        raise ImageVerificationError("OCI and saved-image config digest mismatch")
    annotations = descriptor.get("annotations", {})
    if (
        not isinstance(annotations, dict)
        or annotations.get("config.digest", "sha256:" + config_hash)
        != "sha256:" + config_hash
    ):
        raise ImageVerificationError("OCI config annotation mismatch")
    described_layers = manifest.get("layers")
    if not isinstance(described_layers, list) or len(described_layers) != len(layers):
        raise ImageVerificationError("OCI and saved-image layer population disagree")
    for layer_name, layer_descriptor in zip(layers, described_layers, strict=True):
        if (
            _descriptor_member(layer_descriptor, members, _LAYER_TYPES)
            is not members[_path(layer_name)]
        ):
            raise ImageVerificationError("OCI and saved-image layer order disagree")
    return descriptor["digest"], described_layers


def _clean_root_runtime_matches(config, expected):
    actual = config.get("config")
    if not isinstance(actual, dict):
        return False
    labels = actual.get("Labels", {})
    if not isinstance(labels, dict):
        return False
    revision = labels.get("org.opencontainers.image.revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False
    environment = actual.get("Env")
    if not isinstance(environment, list) or not all(
        isinstance(value, str) for value in environment
    ):
        return False
    pinned_environment = [
        value.replace("${NPA_SOURCE_SHA}", revision) for value in expected["Env"]
    ]
    if sorted(environment) != sorted(pinned_environment):
        return False
    return all(
        actual.get(key) == value for key, value in expected.items() if key != "Env"
    )


def _clean_root_findings(config, layers, contract, layer_identities):
    pinned_hash = contract.get("clean_root_config_sha256")
    if pinned_hash is None:
        return []
    raw = CLEAN_ROOT_CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned_hash:
        raise ImageVerificationError("clean-root runtime contract digest mismatch")
    expected = json.loads(raw)
    if expected["schema_version"] != "npa.curobo.clean-root-config.v1":
        raise ImageVerificationError("unsupported clean-root runtime contract")
    findings = []
    # BuildKit may encode WORKDIR after COPY as this exact empty gzip/tar.
    # Both compressed bytes and the verified uncompressed diff ID are pinned:
    # an arbitrary empty tar, hidden gzip metadata, or a whiteout is not allowed.
    canonical_empty_tail = (
        len(layers) == expected["layer_count"] + 1
        and layer_identities[-1] == expected["optional_empty_metadata_layer"]
    )
    if len(layers) != expected["layer_count"] and not canonical_empty_tail:
        findings.append({"code": "clean_root_layer_population_mismatch"})
    if any(config.get(key) != value for key, value in expected["platform"].items()):
        findings.append({"code": "clean_root_platform_mismatch"})
    if not _clean_root_runtime_matches(config, expected["config"]):
        findings.append({"code": "clean_root_runtime_config_mismatch"})
    return findings


def verify_image(
    tarball: Path, *, expected_image_id: str, contract: dict | None = None
) -> dict:
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
        raise ImageVerificationError(
            "expected exactly eight runtimes and their license"
        )
    notice = contract["nvshmem_notice"]
    expected[_path(notice["path"])] = notice
    libgomp = contract["libgomp"]
    libgomp_payload = [libgomp["runtime"], *libgomp["license_files"]]
    if len(libgomp_payload) != 3:
        raise ImageVerificationError(
            "expected libgomp runtime, package copyright and GPL text"
        )
    for row in libgomp_payload:
        expected[_path(row["path"])] = row
    libgomp_soname = _path(libgomp["runtime"]["soname_path"])
    expected_links = {
        libgomp_soname: libgomp["runtime"]["soname_target"],
    }
    runtime_import_receipt = contract["runtime_import_receipt"]
    expected[_path(runtime_import_receipt["path"])] = runtime_import_receipt
    seeds = contract["bootstrap_seeds"]
    seed_payload = {_path(row["path"]): row for row in seeds["retained"]}
    if len(seeds["retained"]) != 3 or set(seed_payload) != {
        "usr/share/python-wheels/pip-26.2.1+npa.1-py3-none-any.whl",
        "usr/share/python-wheels/setuptools-84.0.0-py3-none-any.whl",
        "usr/share/doc/npa-curobo/secure-pip-build.json",
    }:
        raise ImageVerificationError(
            "expected both replacement seeds and build receipt"
        )
    forbidden_seeds = {_path(path) for path in seeds["forbidden_paths"]}
    if forbidden_seeds != {
        "usr/share/python-wheels/pip-24.0-py3-none-any.whl",
        "usr/share/python-wheels/setuptools-68.1.2-py3-none-any.whl",
    }:
        raise ImageVerificationError("expected complete superseded seed inventory")
    expected.update(seed_payload)
    distro = contract["distro_source_closure"]
    distro_manifest = distro["manifest"]
    expected[_path(distro_manifest["path"])] = distro_manifest
    versioned = {}
    for row in distro["copyright_files"]:
        path = _path(row["path"])
        if path in expected and any(
            row[key] != expected[path][key] for key in ("sha256", "size")
        ):
            raise ImageVerificationError("conflicting distro notice identities")
        expected[path] = row
        versioned[path] = {
            (entry["sha256"], entry["size"]) for entry in row["ancestor_versions"]
        }
        if (row["sha256"], row["size"]) not in versioned[path]:
            raise ImageVerificationError("final notice is absent from its history")
    databases = {row["sha256"]: row for row in distro["package_databases"]}
    if not databases or len(databases) != len(distro["package_databases"]):
        raise ImageVerificationError("missing or repeated distro package databases")
    final_database = databases[distro["final_dpkg_database_sha256"]]
    database_path = "var/lib/dpkg/status"
    expected[database_path] = final_database
    versioned[database_path] = {
        (row["sha256"], row["size"]) for row in databases.values()
    }
    observed_databases = set()
    # PyTorch's generated cuDNN operator declarations are BSD-licensed adapters,
    # not NVIDIA SDK headers. Only exact independently verified wheel bytes at
    # these exact paths qualify, together with the wheel's complete license.
    adapters = {}
    torch_contract = contract.get("torch_cudnn_adapters")
    if torch_contract is not None:
        adapters = {
            f"{root}/{_path(row['path'])}": row for row in torch_contract["headers"]
        }
        if len(adapters) != 52 or len(torch_contract["headers"]) != 52:
            raise ImageVerificationError(
                "expected complete reviewed PyTorch adapter inventory"
            )
        expected.update(adapters)
        torch_license = torch_contract["license"]
        expected[f"{root}/{_path(torch_license['path'])}"] = torch_license
        # The patched wheel carries third-party notices as separate files;
        # its short project LICENSE no longer contains the complete notices.
        notices = torch_contract["third_party_notices"]
        notice_paths = {row["path"] for row in notices}
        if len(notices) != 97 or len(notice_paths) != 97:
            raise ImageVerificationError(
                "expected complete reviewed PyTorch notice inventory"
            )
        for row in notices:
            expected[f"{root}/{_path(row['path'])}"] = row
    excluded_hashes = {row["sha256"] for row in cudnn["excluded_sdk"]}
    runtime_hashes = {row["sha256"] for row in runtimes}
    if (
        len(cudnn["excluded_sdk"]) != 14
        or len({row["path"] for row in cudnn["excluded_sdk"]}) != 14
    ):
        raise ImageVerificationError("expected complete reviewed SDK inventory")
    metadata_root = f"{root}/nvidia_cudnn_cu13-{cudnn['version']}.dist-info"
    metadata = {
        f"{metadata_root}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "RECORD",
            "top_level.txt",
            "INSTALLER",
            "REQUESTED",
            "licenses/License.txt",
        )
    }
    ancestors = {
        str(parent)
        for path in (*expected, *expected_links)
        for parent in PurePosixPath(path).parents
    }
    observed = {}
    observed_links = {}
    findings = []
    entries_read = regular_files_read = content_bytes_read = 0

    def issue(code, layer, entry):
        findings.append({"code": code, "layer_index": layer, "entry_index": entry})

    with tarball.open("rb") as stream:
        archive_hash, _ = _hash_stream(stream)
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
        if (
            not isinstance(layers, list)
            or not layers
            or not all(isinstance(name, str) for name in layers)
        ):
            raise ImageVerificationError("saved-image layers must be complete")
        config_member = members[_path(manifest[0]["Config"])]
        if not config_member.isfile():
            raise ImageVerificationError("saved-image config must be regular")
        config_bytes = archive.extractfile(config_member).read()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        if PurePosixPath(config_member.name).stem != config_hash:
            raise ImageVerificationError("saved-image config digest mismatch")
        manifest_digest, layer_descriptors = _oci_graph(
            archive, members, config_member, config_hash, layers
        )
        if expected_image_id not in {"sha256:" + config_hash, manifest_digest}:
            raise ImageVerificationError(
                "saved-image manifest or config digest mismatch"
            )
        config = json.loads(config_bytes)
        if not isinstance(config, dict) or not isinstance(config.get("rootfs"), dict):
            raise ImageVerificationError(
                "image config requires root filesystem identity"
            )
        if config["rootfs"].get("type") != "layers":
            raise ImageVerificationError("unsupported image root filesystem")
        diff_ids = config.get("rootfs", {}).get("diff_ids")
        if not isinstance(diff_ids, list) or len(diff_ids) != len(layers):
            raise ImageVerificationError(
                "config and complete layer population disagree"
            )
        layer_identities = []
        for layer_index, layer_name in enumerate(layers):
            member = members[_path(layer_name)]
            if not member.isfile():
                raise ImageVerificationError("saved-image layer must be regular")
            # A blob may occur more than once in the layer stack. Verify and scan
            # each occurrence in order, while rejecting duplicate outer entries.
            blob_hash, blob_size = _hash_stream(archive.extractfile(member))
            if (
                _path(layer_name).startswith("blobs/")
                and _path(layer_name) != "blobs/sha256/" + blob_hash
            ):
                raise ImageVerificationError("saved-image layer blob digest mismatch")
            if layer_descriptors is not None:
                descriptor = layer_descriptors[layer_index]
                if (
                    descriptor["digest"] != "sha256:" + blob_hash
                    or descriptor["size"] != blob_size
                ):
                    raise ImageVerificationError(
                        "OCI layer blob digest or size mismatch"
                    )
                signature = archive.extractfile(member).read(2)
                if (signature == b"\x1f\x8b") != _LAYER_TYPES[descriptor["mediaType"]]:
                    raise ImageVerificationError(
                        "OCI layer compression media type mismatch"
                    )
            if _layer_diff_id(archive.extractfile(member)) != diff_ids[layer_index]:
                raise ImageVerificationError("saved-image layer diff ID mismatch")
            layer_identities.append(
                {
                    "blob_sha256": blob_hash,
                    "blob_bytes": blob_size,
                    "diff_id": diff_ids[layer_index],
                }
            )
            # Whiteouts affect lower layers, including when recorded after new files.
            current = {}
            current_links = {}
            seen_paths = set()
            # Use exactly the same decoding for the diff ID and tar contents.
            # Auto-detection here would accept an undeclared bz2/xz codec while
            # hashing its compressed bytes as a false uncompressed diff ID.
            with (
                _layer_stream(archive.extractfile(member)) as decoded,
                tarfile.open(fileobj=decoded, mode="r|") as layer,
            ):
                for entry_index, entry in enumerate(layer):
                    entries_read += 1
                    path = _path(entry.name)
                    # A later whiteout cannot remove bytes from an exported ancestor.
                    # Reject every entry type at superseded seed paths in every layer.
                    if path in forbidden_seeds:
                        issue("superseded_bootstrap_seed", layer_index, entry_index)
                    if path in seen_paths:
                        issue("duplicate_layer_path", layer_index, entry_index)
                    seen_paths.add(path)
                    basename = PurePosixPath(path).name
                    directory = str(PurePosixPath(path).parent)
                    whiteout = basename.startswith(".wh.")
                    malformed_whiteout = whiteout and (
                        not entry.isfile()
                        or entry.size != 0
                        or basename[4:] in {"", ".", ".."}
                    )
                    if malformed_whiteout:
                        issue("malformed_whiteout", layer_index, entry_index)
                    elif basename == ".wh..wh..opq":
                        observed = {
                            p: v
                            for p, v in observed.items()
                            if directory != "." and not p.startswith(directory + "/")
                        }
                        observed_links = {
                            p: v
                            for p, v in observed_links.items()
                            if directory != "." and not p.startswith(directory + "/")
                        }
                    elif whiteout:
                        target = str(PurePosixPath(directory) / basename[4:])
                        observed = {
                            p: v
                            for p, v in observed.items()
                            if p != target and not p.startswith(target + "/")
                        }
                        observed_links = {
                            p: v
                            for p, v in observed_links.items()
                            if p != target and not p.startswith(target + "/")
                        }
                    else:
                        # Every type replacement invalidates proof of that exact file.
                        observed.pop(path, None)
                        observed_links.pop(path, None)
                        current.pop(path, None)
                        current_links.pop(path, None)
                        if not entry.isdir():
                            observed = {
                                p: v
                                for p, v in observed.items()
                                if not p.startswith(path + "/")
                            }
                            current = {
                                p: v
                                for p, v in current.items()
                                if not p.startswith(path + "/")
                            }
                            observed_links = {
                                p: v
                                for p, v in observed_links.items()
                                if not p.startswith(path + "/")
                            }
                            current_links = {
                                p: v
                                for p, v in current_links.items()
                                if not p.startswith(path + "/")
                            }
                        # Required files may not be reached through links or a regular
                        # ancestor. Other system links are never followed or extracted.
                        if path in ancestors and not entry.isdir():
                            issue(
                                "required_payload_ancestor_not_directory",
                                layer_index,
                                entry_index,
                            )
                        if path in expected and not entry.isfile():
                            issue(
                                "retained_payload_not_regular", layer_index, entry_index
                            )
                        if path in expected_links and not entry.issym():
                            issue("retained_link_not_symlink", layer_index, entry_index)
                        # Ancestor bytes remain distributed even if later hidden.
                        if not entry.isdir():
                            if _SDK_NAME.fullmatch(basename) and path not in adapters:
                                issue(
                                    "excluded_cudnn_sdk_path", layer_index, entry_index
                                )
                            if "/nvidia/cudnn/" in "/" + path and path not in expected:
                                issue(
                                    "unreviewed_cudnn_namespace_payload",
                                    layer_index,
                                    entry_index,
                                )
                            if (
                                _RUNTIME_NAME.fullmatch(basename)
                                and path not in expected
                            ):
                                issue(
                                    "unexpected_cudnn_runtime_location",
                                    layer_index,
                                    entry_index,
                                )
                            if (
                                "nvidia_cudnn_" in path
                                and ".dist-info/" in path
                                and path not in metadata
                            ):
                                issue(
                                    "unreviewed_cudnn_distribution_payload",
                                    layer_index,
                                    entry_index,
                                )
                            if basename.lower().startswith(
                                "nvidia_cudnn"
                            ) and basename.lower().endswith(".whl"):
                                issue("cached_cudnn_wheel", layer_index, entry_index)
                        if path in expected_links and entry.issym():
                            matches = entry.linkname == expected_links[path]
                            current_links[path] = {
                                "target": entry.linkname,
                                "matches": matches,
                            }
                            if not matches:
                                issue(
                                    "retained_link_target_mismatch",
                                    layer_index,
                                    entry_index,
                                )
                    if entry.isfile():
                        digest, size = _hash_stream(layer.extractfile(entry))
                        if size != entry.size:
                            raise ImageVerificationError("truncated layer member")
                        regular_files_read += 1
                        content_bytes_read += size
                        if digest in excluded_hashes:
                            issue("excluded_cudnn_sdk_bytes", layer_index, entry_index)
                        if digest in runtime_hashes and path not in expected:
                            issue(
                                "unexpected_cudnn_runtime_copy",
                                layer_index,
                                entry_index,
                            )
                        if path in expected and not whiteout:
                            matches = (
                                digest == expected[path]["sha256"]
                                and size == expected[path]["size"]
                            )
                            current[path] = {
                                "sha256": digest,
                                "size": size,
                                "matches": matches,
                            }
                            if path == database_path:
                                observed_databases.add(digest)
                            allowed_version = (digest, size) in versioned.get(
                                path, set()
                            )
                            if not matches and not allowed_version:
                                issue(
                                    "retained_payload_hash_mismatch",
                                    layer_index,
                                    entry_index,
                                )
            observed.update(current)
            observed_links.update(current_links)
        findings.extend(
            _clean_root_findings(config, layers, contract, layer_identities)
        )
        for path in expected:
            if path not in observed:
                # Expected names originate in reviewed code, never the image.
                findings.append(
                    {"code": "required_payload_missing", "expected_path": path}
                )
            elif not observed[path]["matches"]:
                findings.append(
                    {"code": "final_payload_hash_mismatch", "expected_path": path}
                )
        if observed_databases != set(databases):
            findings.append({"code": "distro_package_history_mismatch"})
        for path in expected_links:
            if path not in observed_links:
                findings.append(
                    {"code": "required_link_missing", "expected_path": path}
                )
    return {
        "schema_version": "npa.curobo.image-verification.v1",
        "valid": not findings,
        "docker_save_sha256": archive_hash,
        "image_config_digest": "sha256:" + config_hash,
        "expected_image_id": expected_image_id,
        "image_manifest_digest": manifest_digest,
        "verified_layer_diff_ids": diff_ids,
        "verified_layer_identities": layer_identities,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "clean_root_config_sha256": contract.get("clean_root_config_sha256"),
        "cudnn_source_wheel_sha256": cudnn["wheel_sha256"],
        "torch_source_wheel_sha256": torch_contract["wheel_sha256"]
        if torch_contract is not None
        else None,
        "verified_torch_adapter_count": sum(
            path in observed and observed[path]["matches"] for path in adapters
        ),
        "layer_count": len(layers),
        "entries_read": entries_read,
        "regular_files_read": regular_files_read,
        "content_bytes_read": content_bytes_read,
        "retained_runtime_count": sum(
            path in observed and observed[path]["matches"]
            for path in expected
            if expected[path].get("kind") == "runtime"
        ),
        "verified_libgomp_payload_count": sum(
            row["path"] in observed and observed[row["path"]]["matches"]
            for row in libgomp_payload
        ),
        "verified_bootstrap_seed_payload_count": sum(
            path in observed and observed[path]["matches"] for path in seed_payload
        ),
        "verified_libgomp_soname_link": (
            libgomp_soname in observed_links
            and observed_links[libgomp_soname]["matches"]
        ),
        "runtime_import_receipt_verified": (
            runtime_import_receipt["path"] in observed
            and observed[runtime_import_receipt["path"]]["matches"]
        ),
        "verified_distro_database_versions": len(observed_databases & set(databases)),
        "verified_distro_copyright_files": sum(
            row["path"] in observed and observed[row["path"]]["matches"]
            for row in distro["copyright_files"]
        ),
        "distro_source_manifest_verified": (
            distro_manifest["path"] in observed
            and observed[distro_manifest["path"]]["matches"]
        ),
        "required_payload_count": len(expected),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--expected-image-id",
        required=True,
        help="Exact docker image inspect .Id for the scanned image",
    )
    args = parser.parse_args()
    try:
        report = verify_image(
            args.docker_save, expected_image_id=args.expected_image_id
        )
    except (OSError, EOFError, ValueError, KeyError, TypeError, tarfile.TarError):
        # Do not echo a malformed archive member or exception with private paths.
        report = {
            "schema_version": "npa.curobo.image-verification.v1",
            "valid": False,
            "findings": [{"code": "unreadable_or_incomplete_image_evidence"}],
        }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "cuRobo complete-layer verification "
        + ("passed" if report["valid"] else "failed")
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
