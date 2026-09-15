"""Bind a closed OCI index graph without extracting files or trusting filenames.

Supports one runtime platform and BuildKit in-toto attestation manifests (legacy
and OCI artifact forms). This is structural byte coverage, not attestation trust,
license approval, or publication authorization. Unknown graph shapes fail closed.
"""
from __future__ import annotations

import base64
import hashlib
import os
import tarfile

from . import core as W

INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
EMPTY = "application/vnd.oci.empty.v1+json"
INTOTO = "application/vnd.in-toto+json"
ATTESTATION = "application/vnd.docker.attestation.manifest.v1+json"
LAYERS = {"application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip"}


def inspect(fd, length, expected_id, *, platform):
    """Return exact layer ranges and a hash-only receipt for every graph blob."""
    W.require(isinstance(expected_id, str) and W.DIGEST.fullmatch(expected_id), "oci_expected_digest")
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "rb") as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
        members = {}
        for member in archive.getmembers():
            name = W.safe_name(member.name)
            W.require(name not in members, "oci_duplicate_path")
            W.require(member.isfile() or member.isdir(), "oci_outer_entry_type")
            W.require(member.offset_data + member.size <= length, "oci_member_range")
            members[name] = member

        def payload(name):
            W.require(name in members, "oci_missing_blob")
            member = members[name]
            W.require(member.isfile(), "oci_blob_not_regular")
            return archive.extractfile(member).read()

        def obj(data):
            value = W.json_object(data)
            W.require(isinstance(value, dict), "oci_json_object")
            return value

        W.require(obj(payload("oci-layout")) == {"imageLayoutVersion": "1.0.0"}, "oci_layout_version")
        root_bytes = payload("index.json")
        root_digest = "sha256:" + W.sha(root_bytes)
        root = obj(root_bytes)
        blobs, stored, visited = {}, set(), set()
        runtime, attestations = [], []

        def bound(desc, allowed):
            W.require(isinstance(desc, dict) and {"mediaType", "digest", "size"} <= set(desc)
                      <= {"mediaType", "digest", "size", "annotations", "platform", "artifactType", "data"}, "oci_descriptor_schema")
            digest, media, size = desc["digest"], desc["mediaType"], desc["size"]
            W.require(isinstance(digest, str) and W.DIGEST.fullmatch(digest), "oci_descriptor_digest")
            W.require(isinstance(media, str) and media in allowed, "oci_descriptor_media_type")
            W.require(type(size) is int and size >= 0, "oci_descriptor_size")
            if "annotations" in desc:
                W.require(isinstance(desc["annotations"], dict) and all(isinstance(k, str) and isinstance(v, str)
                          for k, v in desc["annotations"].items()), "oci_descriptor_annotations")
            inline = None
            if "data" in desc:
                # BuildKit's empty OCI config is the only supported inline blob.
                # Its decoded bytes are fixed, so no opaque base64 payload escapes scanning.
                W.require(media == EMPTY and desc["data"] == "e30=", "oci_inline_descriptor")
                inline = base64.b64decode(desc["data"], validate=True)
                W.require(size == len(inline) and digest == "sha256:" + W.sha(inline), "oci_inline_digest")
            name = "blobs/sha256/" + digest[7:]
            identity = {"digest": digest, "mediaType": media, "size": size}
            W.require(digest not in blobs or blobs[digest] == identity, "oci_conflicting_descriptor")
            if name in members:
                member = members[name]
                W.require(member.isfile() and member.size == size, "oci_blob_size")
                if digest not in blobs:
                    reader = W.Slice(fd, member.offset_data, size)
                    value = hashlib.sha256()
                    while chunk := reader.read(W.CHUNK):
                        value.update(chunk)
                    W.require("sha256:" + value.hexdigest() == digest, "oci_blob_digest")
                stored.add(name)
            else:
                W.require(inline is not None, "oci_missing_blob")
            blobs[digest] = identity
            return name, inline

        def document(desc, allowed):
            name, inline = bound(desc, allowed)
            return obj(inline if inline is not None else payload(name))

        def index_schema(value):
            W.require(type(value.get("schemaVersion")) is int and value["schemaVersion"] == 2
                      and value.get("mediaType", INDEX) == INDEX and isinstance(value.get("manifests"), list)
                      and value["manifests"] and "subject" not in value and "artifactType" not in value, "oci_index_schema")

        # A local OCI export may wrap the publication index in one or more
        # single-index descriptors. Never select one image out of a wider graph.
        selected, selected_digest = root, root_digest
        while selected_digest != expected_id:
            index_schema(selected)
            W.require(len(selected["manifests"]) == 1, "oci_publication_root_binding")
            desc = selected["manifests"][0]
            selected = document(desc, {INDEX})
            selected_digest = desc["digest"]
            W.require(selected_digest not in visited, "oci_graph_cycle")
            visited.add(selected_digest)
        visited.clear()

        def visit_index(value):
            index_schema(value)
            for desc in value["manifests"]:
                manifest = document(desc, {INDEX, MANIFEST})
                W.require(desc["digest"] not in visited, "oci_duplicate_graph_node")
                visited.add(desc["digest"])
                if desc["mediaType"] == INDEX:
                    visit_index(manifest)
                    continue
                W.require(type(manifest.get("schemaVersion")) is int and manifest["schemaVersion"] == 2
                          and manifest.get("mediaType") == MANIFEST and isinstance(manifest.get("layers"), list), "oci_manifest_schema")
                annotations = desc.get("annotations", {})
                artifact = manifest.get("artifactType") == ATTESTATION
                legacy = annotations.get("vnd.docker.reference.type") == "attestation-manifest"
                if artifact or legacy:
                    W.require(desc.get("platform") == {"os": "unknown", "architecture": "unknown"}, "oci_attestation_platform")
                    W.require(manifest["layers"], "oci_empty_attestation")
                    config = document(manifest["config"], {EMPTY} if artifact else {CONFIG})
                    if artifact:
                        W.require(config == {}, "oci_attestation_empty_config")
                        bound(manifest["subject"], {MANIFEST})
                        target = manifest["subject"]["digest"]
                        W.require(not legacy or annotations.get("vnd.docker.reference.digest") == target, "oci_attestation_subject_disagreement")
                    else:
                        W.require("subject" not in manifest and "artifactType" not in manifest, "oci_attestation_schema")
                        target = annotations.get("vnd.docker.reference.digest")
                    for layer in manifest["layers"]:
                        document(layer, {INTOTO})
                    if not artifact:
                        W.require(config.get("architecture") == "unknown" and config.get("os") == "unknown"
                                  and config.get("rootfs") == {"type": "layers", "diff_ids": [layer["digest"] for layer in manifest["layers"]]}, "oci_attestation_config_layers")
                    attestations.append(target)
                else:
                    W.require("artifactType" not in manifest and "subject" not in manifest
                              and "vnd.docker.reference.type" not in annotations, "oci_runtime_manifest_schema")
                    W.require(desc.get("platform") == platform, "oci_runtime_platform")
                    config = document(manifest["config"], {CONFIG})
                    W.require(all(config.get(key) == value for key, value in platform.items()), "oci_config_platform")
                    rootfs = config.get("rootfs", {})
                    diff_ids = rootfs.get("diff_ids")
                    W.require(rootfs.get("type") == "layers" and isinstance(diff_ids, list)
                              and len(diff_ids) == len(manifest["layers"])
                              and all(isinstance(d, str) and W.DIGEST.fullmatch(d) for d in diff_ids), "oci_runtime_diff_ids")
                    layers = []
                    for ordinal, (layer, diff_id) in enumerate(zip(manifest["layers"], diff_ids, strict=True)):
                        name, _ = bound(layer, LAYERS)
                        member = members[name]
                        layers.append({"ordinal": ordinal, "name": name, "offset": member.offset_data,
                                       "size": member.size, "diff_id": diff_id, "descriptor": layer})
                    runtime.append((desc["digest"], manifest["config"]["digest"], layers))

        visit_index(root)
        W.require(len(runtime) == 1, "oci_runtime_population")
        manifest_digest, config_digest, layers = runtime[0]
        W.require(attestations and all(target == manifest_digest for target in attestations), "oci_attestation_target")
        regular = {name for name, member in members.items() if member.isfile()}
        W.require(regular == stored | {"index.json", "oci-layout"}, "oci_unreferenced_blob_or_file")
        W.require(all(name in {".", "blobs", "blobs/sha256"} for name, member in members.items() if member.isdir()), "oci_unexpected_directory")
        return {"layers": layers, "image_manifest_digest": manifest_digest, "image_config_digest": config_digest,
                "receipt": {"archive_index_digest": root_digest, "image_index_digest": expected_id,
                            "runtime_platform": platform, "blob_count": len(stored),
                            "blob_bytes": sum(members[name].size for name in stored),
                            "blobs": sorted(blobs.values(), key=lambda row: row["digest"]),
                            "attestation_manifest_count": len(attestations)}}
