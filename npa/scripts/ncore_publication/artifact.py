"""Derive inspection archives from the verified, unchanged NCore OCI graph."""

import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile

from image_byte_scan import core as W, ncore_verification as N, prepare as P
from .process import file_sha, stream_sha, write_json

_EMPTY_DIFF_ID = "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
_EMPTY_BLOB = "sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1"
_DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
_DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
_DOCKER_LAYERS = {
    "application/vnd.oci.image.layer.v1.tar+gzip": "application/vnd.docker.image.rootfs.diff.tar.gzip",
    "application/vnd.oci.image.layer.v1.tar": "application/vnd.docker.image.rootfs.diff.tar",
}


def inspect(archive, digest):
    """Validate the existing strict OCI graph and every decoded layer digest.

    Args:
        archive: Private original OCI archive.
        digest: Expected publication index digest from buildx metadata.
    Returns:
        Graph offsets and native verification receipt.
    Raises:
        ValueError, OSError: The graph, archive or layer binding is invalid.
    """
    binding = P.binding(archive)
    verification = N.verify(binding, digest)
    with W.bound_open(binding) as (_, fd, info):
        graph = N.inspect(fd, info.st_size, digest)
    return graph, verification


def _blob(archive, digest):
    name = "blobs/sha256/" + digest.removeprefix("sha256:")
    if name not in archive.getnames():
        raw = archive.extractfile("index.json").read()
        W.require("sha256:" + W.sha(raw) == digest, "missing_publication_index")
        return raw
    return archive.extractfile(name).read()


def documents(path, digest, graph):
    """Read digest-checked index/config/attestation documents for local gates.

    Args:
        path: OCI archive already checked by inspect.
        digest: Publication index identity.
        graph: Validated graph.
    Returns:
        Index, config and manifest/blob JSON map.
    Raises:
        ValueError, OSError: Documents cannot be read.
    """
    with tarfile.open(path) as archive:
        blobs = {row["digest"]: json.loads(_blob(archive, row["digest"]))
                 for row in graph["receipt"]["blobs"]
                 if row["mediaType"].endswith("+json") and row["size"] > 2}
        index = json.loads(_blob(archive, digest))
    return index, blobs[graph["image_config_digest"]], blobs


def _add_bytes(archive, name, raw):
    member = tarfile.TarInfo(name)
    member.size = len(raw)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(raw))


def inspection_archives(path, digest, graph, directory):
    """Preserve physical layers while deriving the assembled scratch rootfs.

    Args:
        path: Verified original OCI archive, retained for scanning/publication.
        digest: Original publication index digest.
        graph: Validated graph with layer ranges and diff IDs.
        directory: Private output directory.
    Returns:
        Derivation receipt binding original, config and inspection bytes.
    Raises:
        ValueError: Layers are not an assembled root plus an optional exact no-op.
        OSError, tarfile.TarError: Archive I/O failed.
    """
    original = file_sha(path)
    _scratch_layers(path, graph["layers"])
    layer = graph["layers"][0]
    rootfs = directory / "rootfs.tar"
    with tarfile.open(path) as archive, archive.extractfile(layer["name"]) as stream:
        decoded = gzip.GzipFile(fileobj=stream) if layer["descriptor"]["mediaType"].endswith("+gzip") else stream
        with rootfs.open("xb") as output:
            shutil.copyfileobj(decoded, output)
        config = _blob(archive, graph["image_config_digest"])
    W.require("sha256:" + file_sha(rootfs) == layer["diff_id"], "inspection_diff_id")
    _validate_rootfs(rootfs)
    W.require("sha256:" + W.sha(config) == graph["image_config_digest"], "inspection_config_digest")
    _docker_archive(directory / "inspection.tar", path, graph["layers"], config)
    W.require(file_sha(path) == original, "original_archive_changed")
    receipt = dict(image_digest=digest, archive_sha256=original,
                   platform_digest=graph["image_manifest_digest"],
                   config_digest=graph["image_config_digest"],
                   layers=[dict(row["descriptor"], diff_id=row["diff_id"]) for row in graph["layers"]],
                   rootfs_sha256=file_sha(rootfs),
                   inspection_sha256=file_sha(directory / "inspection.tar"))
    write_json(directory / "inspection.json", receipt)
    return receipt


def _scratch_layers(path, layers):
    W.require(len(layers) in {1, 2}, "ncore_requires_one_scratch_layer")
    if len(layers) == 1:
        return
    tail = layers[1]
    descriptor = tail["descriptor"]
    W.require(descriptor["mediaType"] in _DOCKER_LAYERS, "inspection_empty_codec")
    compressed = descriptor["mediaType"] == "application/vnd.oci.image.layer.v1.tar+gzip"
    W.require(tail["diff_id"] == _EMPTY_DIFF_ID
              and descriptor["digest"] == (_EMPTY_BLOB if compressed else _EMPTY_DIFF_ID)
              and descriptor["size"] == (32 if compressed else 1024), "ncore_requires_one_scratch_layer")
    with tarfile.open(path) as archive:
        raw = archive.extractfile(tail["name"]).read()
    W.require(len(raw) == descriptor["size"] and "sha256:" + W.sha(raw) == descriptor["digest"],
              "inspection_empty_blob")
    decoded = gzip.decompress(raw) if compressed else raw
    W.require(decoded == bytes(1024), "inspection_empty_layer")


def _validate_rootfs(path):
    seen = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = W.safe_name(member.name)
            W.require(name not in seen, "duplicate_rootfs_path")
            W.require(not Path(name).name.startswith(".wh."), "scratch_layer_whiteout")
            W.require(member.isfile() or member.isdir() or member.issym(), "rootfs_special_or_hardlink")
            seen[name] = member
    for name in seen:
        for parent in Path(name).parents:
            W.require(str(parent) not in seen or seen[str(parent)].isdir(), "rootfs_link_ancestor")


def _docker_archive(output, path, layers, config):
    config_name = W.sha(config) + ".json"
    manifest = [{"Config": config_name, "RepoTags": [], "Layers": [row["name"] for row in layers]}]
    with tarfile.open(path) as source, tarfile.open(output, "x") as archive:
        _add_bytes(archive, config_name, config)
        _add_bytes(archive, "manifest.json", json.dumps(manifest).encode())
        for row in layers:
            member = tarfile.TarInfo(row["name"])
            member.size = row["descriptor"]["size"]
            member.mode = 0o644
            with source.extractfile(row["name"]) as stream:
                archive.addfile(member, stream)


def verify_local_export(path, inspected, expected):
    """Bind a daemon's runnable identity to the original config and every layer.

    Args:
        path: Private Docker re-export selected by the captured immutable ID.
        inspected: Single Docker image-inspect object for that same ID.
        expected: Fresh inspection derivation receipt from the verified OCI.
    Returns:
        Local representation relationship; never a publication graph.
    Raises:
        ValueError, OSError, KeyError: Identity, bytes or graph coverage differs.
    """
    original = file_sha(path)
    local_id = inspected["Id"]
    W.require(isinstance(local_id, str) and W.DIGEST.fullmatch(local_id), "loaded_image_id")
    classic = local_id == expected["config_digest"]
    with tarfile.open(path) as archive:
        members = _saved_members(archive)
        config, names, layers = _saved_payload(archive, members, expected, classic)
        _inspected_config(inspected, config)
        if classic:
            W.require(not inspected.get("Descriptor"), "unexpected_classic_descriptor")
        else:
            names |= _saved_manifest(archive, members, inspected, expected, len(config))
        regular = {name for name, member in members.items() if member.isfile()}
        W.require(regular == names, "loaded_export_unreferenced_bytes")
    W.require(file_sha(path) == original, "loaded_export_changed")
    return dict(local_image_id=local_id, identity_kind="config" if classic else "docker-v2-manifest",
                image_digest=expected["image_digest"], platform_digest=expected["platform_digest"],
                config_digest=expected["config_digest"], layers=layers,
                archive_sha256=expected["archive_sha256"],
                inspection_sha256=expected["inspection_sha256"], export_sha256=original)


def _saved_members(archive):
    members = {}
    for member in archive:
        name = W.safe_name(member.name)
        W.require(name not in members, "loaded_export_duplicate_path")
        W.require(member.isfile() or member.isdir(), "loaded_export_entry_type")
        members[name] = member
    return members


def _saved_bytes(archive, members, name):
    W.require(name in members and members[name].isfile(), "loaded_export_missing_file")
    return archive.extractfile(members[name]).read()


def _saved_payload(archive, members, expected, classic):
    manifest = W.json_object(_saved_bytes(archive, members, "manifest.json"))
    W.require(isinstance(manifest, list) and len(manifest) == 1, "loaded_image_population")
    row = manifest[0]
    config = _saved_bytes(archive, members, row["Config"])
    W.require("sha256:" + W.sha(config) == expected["config_digest"], "loaded_config_bytes")
    names = row["Layers"]
    W.require(isinstance(names, list) and len(names) == len(expected["layers"]), "loaded_layer_population")
    W.require(W.json_object(config)["rootfs"] == {
        "type": "layers", "diff_ids": [layer["diff_id"] for layer in expected["layers"]]}, "loaded_config_layers")
    layers = [_saved_layer(archive, members, name, layer, classic)
              for name, layer in zip(names, expected["layers"], strict=True)]
    return config, {"manifest.json", row["Config"], *names}, layers


def _saved_layer(archive, members, name, expected, classic):
    W.require(name in members and members[name].isfile(), "loaded_export_missing_layer")
    member = members[name]
    with archive.extractfile(member) as stream:
        stored = "sha256:" + stream_sha(stream)
    exact = stored == expected["digest"] and member.size == expected["size"]
    # Classic Docker may re-export the exact decoded tar. Bind all its bytes to
    # the original diff ID; recompression or merely equal extracted files fails.
    W.require(exact or (classic and stored == expected["diff_id"]), "loaded_layer_bytes")
    with archive.extractfile(member) as stream:
        compressed = exact and expected["mediaType"].endswith("+gzip")
        decoded = gzip.GzipFile(fileobj=stream) if compressed else stream
        diff_id = "sha256:" + stream_sha(decoded)
    W.require(diff_id == expected["diff_id"], "loaded_layer_diff_id")
    return dict(original_digest=expected["digest"], exported_digest=stored, diff_id=diff_id,
                byte_relation="stored" if exact else "decoded")


def _inspected_config(inspected, raw):
    config = W.json_object(raw)
    W.require(inspected.get("Config") == config["config"], "loaded_inspect_config")
    W.require(inspected.get("RootFS") == {"Type": "layers", "Layers": config["rootfs"]["diff_ids"]},
              "loaded_inspect_layers")
    W.require(inspected.get("Os") == config["os"] and inspected.get("Architecture") == config["architecture"],
              "loaded_inspect_platform")


def _saved_manifest(archive, members, inspected, expected, config_size):
    name = "blobs/sha256/" + inspected["Id"][7:]
    raw = _saved_bytes(archive, members, name)
    W.require("sha256:" + W.sha(raw) == inspected["Id"], "loaded_manifest_digest")
    descriptor = dict(mediaType=_DOCKER_MANIFEST, digest=inspected["Id"], size=len(raw))
    _local_descriptor(inspected.get("Descriptor"), descriptor)
    manifest = dict(schemaVersion=2, mediaType=_DOCKER_MANIFEST,
                    config=dict(mediaType=_DOCKER_CONFIG, digest=expected["config_digest"], size=config_size),
                    layers=[dict(mediaType=_DOCKER_LAYERS[layer["mediaType"]], digest=layer["digest"], size=layer["size"])
                            for layer in expected["layers"]])
    W.require(W.json_object(raw) == manifest, "loaded_manifest_relationship")
    layout = W.json_object(_saved_bytes(archive, members, "oci-layout"))
    W.require(layout == {"imageLayoutVersion": "1.0.0"}, "loaded_layout_version")
    index = W.json_object(_saved_bytes(archive, members, "index.json"))
    W.require(index.get("schemaVersion") == 2 and index.get("mediaType") == "application/vnd.oci.image.index.v1+json"
              and isinstance(index.get("manifests"), list) and len(index["manifests"]) == 1, "loaded_index_population")
    _local_descriptor(index["manifests"][0], descriptor)
    names = {name, "index.json", "oci-layout"}
    for entry in [manifest["config"], *manifest["layers"]]:
        blob = "blobs/sha256/" + entry["digest"][7:]
        W.require(blob in members and members[blob].isfile() and members[blob].size == entry["size"],
                  "loaded_manifest_missing_blob")
        with archive.extractfile(members[blob]) as stream:
            W.require("sha256:" + stream_sha(stream) == entry["digest"],
                      "loaded_manifest_blob_digest")
        names.add(blob)
    return names


def _local_descriptor(value, expected):
    W.require(isinstance(value, dict) and set(expected) <= set(value) <= {*expected, "annotations"}
              and all(value[key] == field for key, field in expected.items()), "loaded_manifest_descriptor")


def assert_unchanged(path, verification):
    """Recheck the original archive hash immediately before a transfer.

    Args:
        path: Original OCI archive.
        verification: Native graph receipt from the completed gates.
    Returns:
        None.
    Raises:
        ValueError, OSError: Archive bytes changed or cannot be read.
    """
    W.require(file_sha(path) == verification["archive_sha256"], "gated_archive_changed")
