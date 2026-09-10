"""Reject altered inspection/runtime representations before executing an image."""

import copy
import gzip
import io
import json
from pathlib import Path
import tarfile

import pytest

from test_ncore_oci_publication import W, _archive, _tar, artifact, process
from test_ncore_oci_publication import private as private
from ncore_publication import bootstrap


EMPTY = bytes.fromhex("1f8b08000000000000ff621805a360148c5800080000ffff2eafb5ef00040000")
MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
CONFIG = "application/vnd.docker.container.image.v1+json"


def _files(path):
    with tarfile.open(path) as archive:
        return {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}


@pytest.mark.parametrize("tail", [EMPTY, bytes(1024)])
def test_canonical_tail_preserves_config_physical_layers_and_rootfs(private, tail):
    path, digest, rootfs = _archive(private, tails=(tail,))
    graph, verification = artifact.inspect(path, digest)
    receipt = artifact.inspection_archives(path, digest, graph, private)
    files = _files(private / "inspection.tar")
    original = _files(path)
    manifest = json.loads(files["manifest.json"])[0]
    assert manifest["RepoTags"] == []
    assert files[manifest["Config"]] == original["blobs/sha256/" + graph["image_config_digest"][7:]]
    assert len(manifest["Layers"]) == verification["layer_count"] == len(receipt["layers"]) == 2
    for name, row in zip(manifest["Layers"], graph["layers"], strict=True):
        assert files[name] == original[row["name"]]
        assert "sha256:" + W.sha(files[name]) == row["descriptor"]["digest"]
    assert (private / "rootfs.tar").read_bytes() == rootfs
    assert receipt["rootfs_sha256"] == verification["verified_layer_diff_ids"][0][7:]
    assert receipt["archive_sha256"] == process.file_sha(path)
    assert verification["regular_files_read"] == 2


def _named_empty():
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, filename="unexpected-metadata", mode="wb", mtime=0) as stream:
        stream.write(bytes(1024))
    return output.getvalue()


@pytest.mark.parametrize("tails", [
    (EMPTY[:4] + b"\x01" + EMPTY[5:],), (_named_empty(),), (EMPTY + bytes(512),),
    (bytes(1536),), (gzip.compress(bytes(1536), mtime=0),),
    (_tar({"unexpected": b""}),), (EMPTY, EMPTY),
])
def test_noncanonical_headers_padding_entries_and_extra_layers_fail(private, tails):
    path, digest, _ = _archive(private, tails=tails)
    with pytest.raises(ValueError):
        graph, _ = artifact.inspect(path, digest)
        artifact.inspection_archives(path, digest, graph, private)
    assert not (private / "inspection.json").exists()


def _prepared(private, tails=(EMPTY,)):
    path, digest, _ = _archive(private, tails=tails)
    graph, _ = artifact.inspect(path, digest)
    return artifact.inspection_archives(path, digest, graph, private)


def _daemon_files(private, expected, mode):
    source = _files(private / "inspection.tar")
    row = json.loads(source["manifest.json"])[0]
    config = source[row["Config"]]
    config_name = "blobs/sha256/" + W.sha(config)
    files = {config_name: config}
    names = []
    for ordinal, name in enumerate(row["Layers"]):
        raw = source[name]
        if mode == "classic-decoded":
            raw = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
        saved_name = f"layer-{ordinal}/layer.tar" if mode.startswith("classic") else name
        files[saved_name] = raw
        names.append(saved_name)
    files["manifest.json"] = json.dumps([dict(Config=config_name, Layers=names, RepoTags=[])]).encode()
    config_object = json.loads(config)
    inspected = dict(Id=expected["config_digest"], Config=config_object["config"], Os="linux", Architecture="amd64",
                     RootFS=dict(Type="layers", Layers=config_object["rootfs"]["diff_ids"]))
    if mode == "containerd":
        _containerd_manifest(files, inspected, expected, len(config))
    return files, inspected


def _containerd_manifest(files, inspected, expected, config_size):
    manifest = dict(schemaVersion=2, mediaType=MANIFEST,
                    config=dict(mediaType=CONFIG, digest=expected["config_digest"], size=config_size),
                    layers=[dict(mediaType="application/vnd.docker.image.rootfs.diff.tar.gzip",
                                 digest=layer["digest"], size=layer["size"]) for layer in expected["layers"]])
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    inspected["Id"] = "sha256:" + W.sha(raw)
    descriptor = dict(mediaType=MANIFEST, digest=inspected["Id"], size=len(raw))
    inspected["Descriptor"] = descriptor
    files["blobs/sha256/" + inspected["Id"][7:]] = raw
    files["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    files["index.json"] = json.dumps(dict(schemaVersion=2, mediaType="application/vnd.oci.image.index.v1+json",
                                         manifests=[descriptor])).encode()


@pytest.mark.parametrize("mode", ["containerd", "classic-stored", "classic-decoded"])
@pytest.mark.parametrize("tails", [(), (EMPTY,)])
def test_daemon_identity_requires_exact_config_and_every_layer(private, mode, tails):
    expected = _prepared(private, tails)
    files, inspected = _daemon_files(private, expected, mode)
    exported = private / "daemon.tar"
    exported.write_bytes(_tar(files))
    receipt = artifact.verify_local_export(exported, inspected, expected)
    assert receipt["local_image_id"] == inspected["Id"]
    assert receipt["config_digest"] == expected["config_digest"]
    assert receipt["image_digest"] == expected["image_digest"]
    assert receipt["platform_digest"] == expected["platform_digest"]
    assert len(receipt["layers"]) == len(expected["layers"])
    assert receipt["identity_kind"] == ("docker-v2-manifest" if mode == "containerd" else "config")
    assert {row["byte_relation"] for row in receipt["layers"]} == ({"decoded"} if mode == "classic-decoded" else {"stored"})


def _alter_export(files, inspected, change):
    row = json.loads(files["manifest.json"])[0]
    if change == "config":
        files[row["Config"]] += b" "
    elif change in {"first-layer", "tail-layer", "recompressed"}:
        name = row["Layers"][change == "tail-layer"]
        files[name] = (gzip.compress(gzip.decompress(files[name]), mtime=1)
                       if change == "recompressed" else files[name] + b"changed")
    elif change == "missing-layer":
        del files[row["Layers"][-1]]
    elif change == "extra-blob":
        files["blobs/sha256/" + "f" * 64] = b"unreferenced graph"
    elif change == "inspect-config":
        inspected["Config"]["User"] = "root"
    elif change == "inspect-rootfs":
        inspected["RootFS"]["Layers"].pop()
    elif change in {"partial", "reordered", "extra-layer", "multiple-images"}:
        if change == "partial":
            row["Layers"].pop()
        elif change == "reordered":
            row["Layers"].reverse()
        elif change == "extra-layer":
            row["Layers"].append(row["Layers"][-1])
        files["manifest.json"] = json.dumps([row] * (2 if change == "multiple-images" else 1)).encode()
    else:
        raise AssertionError(change)


@pytest.mark.parametrize("mode", ["containerd", "classic-stored", "classic-decoded"])
@pytest.mark.parametrize("change", ["config", "first-layer", "tail-layer", "missing-layer", "extra-blob",
                                   "inspect-config", "inspect-rootfs", "partial", "reordered", "extra-layer", "multiple-images"])
def test_different_or_partial_loaded_bytes_are_rejected(private, mode, change):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, mode)
    _alter_export(files, inspected, change)
    exported = private / "daemon.tar"
    exported.write_bytes(_tar(files))
    with pytest.raises(ValueError):
        artifact.verify_local_export(exported, inspected, expected)


@pytest.mark.parametrize("mode", ["containerd", "classic-stored"])
def test_semantically_equal_recompressed_layers_are_rejected(private, mode):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, mode)
    _alter_export(files, inspected, "recompressed")
    exported = private / "daemon.tar"
    exported.write_bytes(_tar(files))
    with pytest.raises(ValueError, match="loaded_layer_bytes"):
        artifact.verify_local_export(exported, inspected, expected)


def _mock_docker(monkeypatch, private, files, inspected, load=None):
    commands = []

    def run(argv, output, **kwargs):
        commands.append((argv, kwargs))
        if argv[1] == "load":
            output.write_text(load if load is not None else "Loaded image ID: " + inspected["Id"] + "\n")
        elif argv[1:3] == ["image", "inspect"]:
            output.write_text(json.dumps([inspected]))
        elif argv[1:3] == ["image", "save"]:
            Path(argv[argv.index("--output") + 1]).write_bytes(_tar(files))
            output.write_text("")
        elif argv[1] == "run":
            assert (private / "local-image-binding.json").exists()
            assert inspected["Id"] in argv and "--pull=never" in argv
        else:
            pytest.fail("unexpected Docker operation")

    monkeypatch.setattr(bootstrap, "run", run)
    monkeypatch.setattr(bootstrap, "_apt_script", lambda _: (
        "# synthetic pinned APT script\n"
        "function synthetic_apt() {\n"
        f"  local log={private}/apt.log\n"
        "}\n"
    ))
    return commands


@pytest.mark.parametrize("mode", ["containerd", "classic-decoded"])
def test_binding_precedes_all_offline_cold_warm_and_bash_probes(private, monkeypatch, mode):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, mode)
    commands = _mock_docker(monkeypatch, private, files, inspected)
    bootstrap.verify(private, expected["config_digest"], private)
    assert [argv[1] for argv, _ in commands] == ["load", "image", "image", "run", "run", "run"]
    assert commands[1][0][-1] == commands[2][0][-1] == inspected["Id"]
    assert "--network=none" in commands[3][0] and "--image-only" in commands[3][0]
    assert "--entrypoint" not in commands[4][0]
    assert commands[5][0][commands[5][0].index("--entrypoint") + 1] == "/bin/bash"
    for _, kwargs in commands[4:]:
        script = kwargs["input_bytes"].decode()
        assert script.count("/opt/venv/bin/python /opt/ncore/bin/verify-packaging.py") == 2
        assert "ssh-keygen -A" in script and "rsync --version" in script
        assert "synthetic pinned APT script" in script


@pytest.mark.parametrize("change", ["config", "first-layer", "tail-layer", "missing-layer", "partial",
                                   "multiple-images", "unknown-id", "changed-inspection", "ambiguous-load", "tag-load"])
def test_failed_binding_never_reaches_container_execution(private, monkeypatch, change):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, "containerd")
    load = None
    if change == "unknown-id":
        inspected["Id"] = "sha256:" + "e" * 64
    elif change == "changed-inspection":
        with (private / "inspection.tar").open("ab") as stream:
            stream.write(b"changed")
    elif change == "ambiguous-load":
        load = ("Loaded image ID: " + inspected["Id"] + "\n") * 2
    elif change == "tag-load":
        load = "Loaded image: unexpected:tag\n"
    else:
        _alter_export(files, inspected, change)
    commands = _mock_docker(monkeypatch, private, files, inspected, load)
    with pytest.raises(ValueError):
        bootstrap.verify(private, expected["config_digest"], private)
    assert all(argv[1] != "run" for argv, _ in commands)
    assert not (private / "local-image-binding.json").exists()


@pytest.mark.parametrize("change", ["manifest", "descriptor", "index", "index-population", "missing-manifest", "duplicate-path"])
def test_containerd_requires_the_exact_local_manifest_graph(private, change):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, "containerd")
    manifest_name = "blobs/sha256/" + inspected["Id"][7:]
    if change == "manifest":
        files[manifest_name] += b" "
    elif change == "descriptor":
        inspected["Descriptor"]["digest"] = expected["platform_digest"]
    elif change == "missing-manifest":
        del files[manifest_name]
    elif change in {"index", "index-population"}:
        index = json.loads(files["index.json"])
        if change == "index":
            index["manifests"][0]["digest"] = expected["image_digest"]
        else:
            index["manifests"].append(copy.deepcopy(index["manifests"][0]))
        files["index.json"] = json.dumps(index).encode()
    exported = private / "daemon.tar"
    exported.write_bytes(_tar(files))
    if change == "duplicate-path":
        with tarfile.open(exported, "a") as archive:
            member = tarfile.TarInfo("manifest.json")
            member.size = len(files["manifest.json"])
            archive.addfile(member, io.BytesIO(files["manifest.json"]))
    with pytest.raises(ValueError):
        artifact.verify_local_export(exported, inspected, expected)


@pytest.mark.parametrize("change", ["config", "layer", "codec", "subject"])
def test_hash_valid_local_manifest_must_have_the_original_relationship(private, change):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, "containerd")
    name = "blobs/sha256/" + inspected["Id"][7:]
    manifest = json.loads(files.pop(name))
    if change == "config":
        manifest["config"]["digest"] = "sha256:" + "f" * 64
    elif change == "layer":
        manifest["layers"].pop()
    elif change == "codec":
        manifest["layers"][0]["mediaType"] = "application/vnd.docker.image.rootfs.diff.tar"
    else:
        manifest["subject"] = manifest["config"]
    raw = json.dumps(manifest).encode()
    inspected["Id"] = "sha256:" + W.sha(raw)
    files["blobs/sha256/" + inspected["Id"][7:]] = raw
    inspected["Descriptor"].update(digest=inspected["Id"], size=len(raw))
    index = json.loads(files["index.json"])
    index["manifests"] = [inspected["Descriptor"]]
    files["index.json"] = json.dumps(index).encode()
    exported = private / "daemon.tar"
    exported.write_bytes(_tar(files))
    with pytest.raises(ValueError, match="loaded_manifest_relationship"):
        artifact.verify_local_export(exported, inspected, expected)


@pytest.mark.parametrize("observed", [[], [{}, {}], [None], [{"Id": "sha256:" + "f" * 64}]])
def test_inspect_must_return_the_captured_single_identity(private, monkeypatch, observed):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, "containerd")
    commands = _mock_docker(monkeypatch, private, files, inspected)
    original_run = bootstrap.run

    def run(argv, output, **kwargs):
        if argv[1:3] == ["image", "inspect"]:
            output.write_text(json.dumps(observed))
            return
        original_run(argv, output, **kwargs)

    monkeypatch.setattr(bootstrap, "run", run)
    with pytest.raises(ValueError, match="loaded_inspect"):
        bootstrap.verify(private, expected["config_digest"], private)
    assert all(argv[1] != "run" for argv, _ in commands)
    assert not (private / "loaded-image.tar").exists()


def test_daemon_export_cannot_replace_existing_evidence(private, monkeypatch):
    expected = _prepared(private)
    files, inspected = _daemon_files(private, expected, "containerd")
    commands = _mock_docker(monkeypatch, private, files, inspected)
    exported = private / "loaded-image.tar"
    exported.write_bytes(b"prior evidence")
    with pytest.raises(ValueError, match="loaded_export_must_be_new"):
        bootstrap.verify(private, expected["config_digest"], private)
    assert all(argv[1] != "run" for argv, _ in commands)
    assert exported.read_bytes() == b"prior evidence"
