"""Exercise saved image evidence with synthetic bytes, without Docker or vendors."""

from __future__ import annotations

import copy
import bz2
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/curobo"
SPEC = importlib.util.spec_from_file_location("curobo_image_verifier", IMAGE / "verify_image.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entry(name, data=b"", kind=tarfile.REGTYPE, link=""):
    return name, data, kind, link


def tar_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data, kind, link in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = link
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return stream.getvalue()


@pytest.fixture
def payload():
    """Tiny independent contract; no proprietary payload or fabricated GPU output."""
    contract = copy.deepcopy(json.loads((IMAGE / "runtime-payload.json").read_text()))
    contract.pop("torch_cudnn_adapters")  # Adapter qualification has separate exact-byte fixtures.
    entries = []
    cudnn = contract["cudnn"]
    for index, row in enumerate(cudnn["retained"]):
        data = b"\x7fELF synthetic runtime " + bytes([index]) if row["kind"] == "runtime" else b"Synthetic full cuDNN notice."
        row.update(sha256=digest(data), size=len(data))
        entries.append(entry(f"{cudnn['install_root']}/{row['path']}", data))
    # The real fourteen headers contain seven pairs of identical files.
    excluded = []
    for index, row in enumerate(cudnn["excluded_sdk"]):
        data = f"Synthetic excluded SDK pair {index // 2}".encode()
        row.update(sha256=digest(data), size=len(data))
        excluded.append(data)
    notice = b"Synthetic complete NVSHMEM notice including third-party licenses."
    contract["nvshmem_notice"].update(sha256=digest(notice), size=len(notice))
    entries.append(entry(contract["nvshmem_notice"]["path"], notice))
    return contract, entries, excluded


def save_image(tmp_path, layers, *, compressed=False, oci_paths=False, config_update=None,
               manifest_update=None, config_path=None, layer_replacement=None, outer_entries=()):
    raw_layers = [tar_bytes(layer) for layer in layers]
    config = {"rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(layer) for layer in raw_layers]}}
    if config_update:
        config_update(config)
    config_data = json.dumps(config).encode()
    image_id = "sha256:" + digest(config_data)
    config_name = config_path or ("blobs/sha256/" + digest(config_data) if oci_paths else digest(config_data) + ".json")
    layer_data = [gzip.compress(layer, mtime=0) if compressed else layer for layer in raw_layers]
    layer_names = [("blobs/sha256/" + digest(data) if oci_paths else f"layer-{index}/layer.tar") for index, data in enumerate(layer_data)]
    manifest = [{"Config": config_name, "Layers": list(layer_names), "RepoTags": ["synthetic:test"]}]
    if manifest_update:
        manifest_update(manifest)
    if layer_replacement:
        layer_data[0] = layer_replacement
    members = [entry("manifest.json", json.dumps(manifest).encode()), entry(config_name, config_data)]
    members.extend(entry(name, data) for name, data in zip(layer_names, layer_data, strict=True))
    members.extend(outer_entries)
    archive = tmp_path / "image.tar"
    archive.write_bytes(tar_bytes(members))
    return archive, image_id


def verify(tmp_path, payload, layers=None, **kwargs):
    contract, entries, _ = payload
    archive, image_id = save_image(tmp_path, layers or [entries], **kwargs)
    return VERIFIER.verify_image(archive, expected_image_id=image_id, contract=contract)


def codes(report):
    return {finding["code"] for finding in report["findings"]}


@pytest.mark.parametrize("compressed,oci_paths", [(False, False), (False, True), (True, True)])
def test_complete_image_binds_config_all_diff_ids_and_independent_bytes(tmp_path, payload, compressed, oci_paths):
    report = verify(tmp_path, payload, compressed=compressed, oci_paths=oci_paths)
    assert report["valid"] is True
    assert report["retained_runtime_count"] == 8
    assert report["required_payload_count"] == 10
    assert report["layer_count"] == 1
    assert len(report["verified_layer_diff_ids"]) == 1
    assert report["regular_files_read"] == 10
    assert report["content_bytes_read"] == sum(len(row[1]) for row in payload[1])
    assert report["docker_save_sha256"] == digest((tmp_path / "image.tar").read_bytes())


def test_normal_root_and_system_links_are_never_extracted_or_followed(tmp_path, payload):
    layers = [[entry(".", kind=tarfile.DIRTYPE), entry("bin", kind=tarfile.SYMTYPE, link="usr/bin"), *payload[1]]]
    assert verify(tmp_path, payload, layers)["valid"] is True
    assert not (tmp_path / "bin").exists()


@pytest.mark.parametrize("replacement", [b"changed runtime", b""])
def test_rejects_changed_or_empty_required_runtime(tmp_path, payload, replacement):
    entries = list(payload[1])
    entries[0] = entry(entries[0][0], replacement)
    report = verify(tmp_path, payload, [entries])
    assert not report["valid"]
    assert "retained_payload_hash_mismatch" in codes(report)


def test_full_notice_is_verified_independently_of_image_authored_manifest(tmp_path, payload):
    entries = list(payload[1])
    entries[-1] = entry(entries[-1][0], b"license truncated")
    entries.append(entry("usr/share/doc/npa-curobo/cudnn-runtime.json", b'{"valid":true}'))
    assert "retained_payload_hash_mismatch" in codes(verify(tmp_path, payload, [entries]))


@pytest.mark.parametrize("name", ["opt/elsewhere/cudnn.h", "usr/lib/libcudnn_static.a", "opt/renamed.dat"])
def test_rejects_relocated_sdk_even_when_deleted_later(tmp_path, payload, name):
    before = [entry(name, payload[2][0])]
    whiteout = str(Path(name).parent / (".wh." + Path(name).name))
    report = verify(tmp_path, payload, [before, [entry(whiteout), *payload[1]]])
    assert not report["valid"]
    assert "excluded_cudnn_sdk_bytes" in codes(report)
    assert any(finding.get("layer_index") == 0 for finding in report["findings"])


@pytest.mark.parametrize("name", ["opt/relocated/libcudnn_extra.so.9", "opt/renamed.bin"])
def test_rejects_runtime_copies_outside_reviewed_locations(tmp_path, payload, name):
    report = verify(tmp_path, payload, [[*payload[1], entry(name, payload[1][0][1])]])
    assert "unexpected_cudnn_runtime_copy" in codes(report)


@pytest.mark.parametrize("suffix", ["nvidia/cudnn/unrecorded.txt", "nvidia_cudnn_cu13-9.13.0.50.dist-info/secret", "nvidia_cudnn_cu13-9.12.dist-info/METADATA"])
def test_rejects_unreviewed_wheel_payload_and_versions(tmp_path, payload, suffix):
    name = payload[0]["cudnn"]["install_root"] + "/" + suffix
    report = verify(tmp_path, payload, [[*payload[1], entry(name, b"unreviewed")]])
    assert not report["valid"]


def test_rejects_cached_unfiltered_cudnn_wheel(tmp_path, payload):
    report = verify(tmp_path, payload, [[*payload[1], entry("tmp/nvidia_cudnn_cu13-cache.whl", b"cache")]])
    assert "cached_cudnn_wheel" in codes(report)


@pytest.mark.parametrize("kind", [tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_later_type_replacement_cannot_keep_old_regular_proof(tmp_path, payload, kind):
    replacement = entry(payload[1][0][0], kind=kind, link=payload[1][1][0])
    report = verify(tmp_path, payload, [payload[1], [replacement]])
    assert "required_payload_missing" in codes(report)
    assert "retained_payload_not_regular" in codes(report)
    assert report["retained_runtime_count"] == 7


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.REGTYPE])
@pytest.mark.parametrize("before", [True, False])
def test_required_parent_links_and_files_fail_in_either_layer_order(tmp_path, payload, kind, before):
    parent = entry("opt/npa-venv", kind=kind, link="../../escape")
    layers = [[parent], payload[1]] if before else [payload[1], [parent]]
    report = verify(tmp_path, payload, layers)
    assert "required_payload_ancestor_not_directory" in codes(report)
    assert not report["valid"]


@pytest.mark.parametrize("whiteout", [".wh..wh..opq", "opt/.wh..wh..opq", "opt/.wh.npa-venv"])
def test_whiteouts_remove_lower_proof_including_root_opaque(tmp_path, payload, whiteout):
    report = verify(tmp_path, payload, [payload[1], [entry(whiteout)]])
    assert "required_payload_missing" in codes(report)
    assert report["retained_runtime_count"] == 0


def test_direct_whiteout_removes_one_runtime(tmp_path, payload):
    path = Path(payload[1][0][0])
    whiteout = str(path.parent / (".wh." + path.name))
    report = verify(tmp_path, payload, [payload[1], [entry(whiteout)]])
    assert report["retained_runtime_count"] == 7
    assert "required_payload_missing" in codes(report)


@pytest.mark.parametrize("whiteout_first", [True, False])
def test_opaque_whiteout_does_not_erase_new_same_layer_files(tmp_path, payload, whiteout_first):
    current = [entry(".wh..wh..opq"), *payload[1]] if whiteout_first else [*payload[1], entry(".wh..wh..opq")]
    assert verify(tmp_path, payload, [[entry("opt/old", b"old")], current])["valid"] is True


def test_later_directory_metadata_merges_existing_children(tmp_path, payload):
    assert verify(tmp_path, payload, [payload[1], [entry("opt/npa-venv", kind=tarfile.DIRTYPE)]])["valid"] is True


@pytest.mark.parametrize("path", ["../escape", "/absolute", "opt/../../escape"])
def test_layer_traversal_is_rejected_without_extraction(tmp_path, payload, path):
    with pytest.raises(VERIFIER.ImageVerificationError, match="unsafe archive path"):
        verify(tmp_path, payload, [[*payload[1], entry(path, b"untrusted")]])


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0]["Layers"].append(rows[0]["Layers"][0]),
    lambda rows: rows[0]["Layers"].clear(),
])
def test_rejects_ambiguous_or_incomplete_saved_images(tmp_path, payload, mutation):
    with pytest.raises(VERIFIER.ImageVerificationError):
        verify(tmp_path, payload, manifest_update=mutation)


def test_duplicate_normalized_outer_paths_are_rejected(tmp_path, payload):
    with pytest.raises(VERIFIER.ImageVerificationError, match="duplicate"):
        verify(tmp_path, payload, outer_entries=[entry("./manifest.json", b"[]")])


def test_config_name_and_expected_image_id_bind_actual_image(tmp_path, payload):
    with pytest.raises(VERIFIER.ImageVerificationError, match="config digest"):
        verify(tmp_path, payload, config_path="0" * 64 + ".json")
    archive, _ = save_image(tmp_path, [payload[1]])
    with pytest.raises(VERIFIER.ImageVerificationError, match="config digest"):
        VERIFIER.verify_image(archive, expected_image_id="sha256:" + "0" * 64, contract=payload[0])


@pytest.mark.parametrize("expected", ["", "synthetic:test", "sha256:short"])
def test_mutable_or_incomplete_expected_identity_rejected(tmp_path, payload, expected):
    archive, _ = save_image(tmp_path, [payload[1]])
    with pytest.raises(VERIFIER.ImageVerificationError, match="exact independently inspected"):
        VERIFIER.verify_image(archive, expected_image_id=expected, contract=payload[0])


def test_layer_tampering_rejected_even_when_tar_still_parses(tmp_path, payload):
    altered = tar_bytes([entry("opt/changed", b"tampered"), *payload[1]])
    with pytest.raises(VERIFIER.ImageVerificationError, match="layer diff ID"):
        verify(tmp_path, payload, layer_replacement=altered)


@pytest.mark.parametrize("update", [
    lambda config: config["rootfs"]["diff_ids"].clear(),
    lambda config: config["rootfs"].update(type="unknown"),
])
def test_invalid_rootfs_population_is_not_accepted(tmp_path, payload, update):
    with pytest.raises(VERIFIER.ImageVerificationError):
        verify(tmp_path, payload, config_update=update)


def test_cli_failure_report_never_echoes_archive_private_paths(tmp_path):
    archive = tmp_path / "private-input.tar"
    archive.write_bytes(tar_bytes([entry("../private-sensitive-path", b"private-sensitive-content")]))
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(IMAGE / "verify_image.py"), "--docker-save", str(archive),
                             "--expected-image-id", "sha256:" + "0" * 64, "--json", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert json.loads(output.read_text())["findings"] == [{"code": "unreadable_or_incomplete_image_evidence"}]
    for text in (output.read_text(), result.stdout, result.stderr):
        assert "private-" not in text
        assert str(tmp_path) not in text


def test_production_contract_matches_locked_artifacts_and_notice():
    contract = json.loads((IMAGE / "runtime-payload.json").read_text())
    cudnn = contract["cudnn"]
    lock = (IMAGE / "requirements.lock").read_text()
    assert f"{cudnn['name']}=={cudnn['version']}" in lock
    assert f"--hash=sha256:{cudnn['wheel_sha256']}" in lock
    assert cudnn["wheel_sha256"] == "2150b4850725d30653ec3e365f0732e3e2e3eb8633cf3bd2d3117628dea8b4f9"
    assert len(cudnn["retained"]) == 9
    assert sum(row["kind"] == "runtime" for row in cudnn["retained"]) == 8
    assert len(cudnn["excluded_sdk"]) == 14
    assert len({row["sha256"] for row in cudnn["excluded_sdk"]}) == 7
    notice = contract["nvshmem_notice"]
    assert notice["sha256"] == "43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f"
    assert notice["size"] == 26566
    docker = (IMAGE / "Dockerfile").read_text()
    assert notice["sha256"] in docker and notice["url"] in docker and "/" + notice["path"] in docker


def test_trusted_workflow_checks_local_bytes_before_push_and_exact_pushed_bytes():
    text = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    workflow = yaml.safe_load(text)
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["build-development"]["steps"])
    assert commands.count("npa/docker/workbench/curobo/verify_image.py") == 2
    push = commands.index('docker push "$IMAGE"')
    first = commands.index('npa/docker/workbench/curobo/verify_image.py')
    second = commands.rindex('npa/docker/workbench/curobo/verify_image.py')
    assert first < push < second
    assert '--expected-image-id "$(docker image inspect --format \'{{.Id}}\' "$IMAGE")"' in commands[:push]
    assert '--expected-image-id "$(docker image inspect --format \'{{.Id}}\' "$exact")"' in commands[push:]
    assert '--docker-save "$RUNNER_TEMP/${TOOL}-pushed.tar"' in commands[second:]
    assert commands.count('npa/scripts/scan_image_omniverse_payload.py') == 2


def test_no_member_size_cap_and_no_required_file_sample(tmp_path, payload):
    data = b"complete unreviewed neutral file\n" * 350_000
    report = verify(tmp_path, payload, [[entry("opt/neutral-large-file", data), *payload[1]]])
    assert report["valid"] is True
    assert report["content_bytes_read"] == len(data) + sum(len(row[1]) for row in payload[1])
    assert report["regular_files_read"] == 11


def test_same_layer_directory_replacement_invalidates_regular_proof(tmp_path, payload):
    report = verify(tmp_path, payload, [[*payload[1], entry(payload[1][0][0], kind=tarfile.DIRTYPE)]])
    assert "retained_payload_not_regular" in codes(report)
    assert "required_payload_missing" in codes(report)


def test_bad_ancestor_runtime_stays_rejected_after_correct_replacement(tmp_path, payload):
    report = verify(tmp_path, payload, [[entry(payload[1][0][0], b"unreviewed old bytes")], payload[1]])
    assert "retained_payload_hash_mismatch" in codes(report)
    assert report["retained_runtime_count"] == 8
    assert not report["valid"]


@pytest.mark.parametrize("whiteout", [entry(".wh..wh..opq", b"not empty"), entry(".wh..wh..opq", kind=tarfile.SYMTYPE, link="escape"), entry("opt/.wh.")])
def test_malformed_whiteout_fails_closed(tmp_path, payload, whiteout):
    assert "malformed_whiteout" in codes(verify(tmp_path, payload, [[whiteout, *payload[1]]]))


@pytest.mark.parametrize("update", [lambda rows: rows.__setitem__(0, "invalid"), lambda rows: rows[0].update(Layers=None)])
def test_malformed_manifest_fails_closed(tmp_path, payload, update):
    with pytest.raises(VERIFIER.ImageVerificationError):
        verify(tmp_path, payload, manifest_update=update)


def test_nonobject_rootfs_fails_closed(tmp_path, payload):
    with pytest.raises(VERIFIER.ImageVerificationError):
        verify(tmp_path, payload, config_update=lambda config: config.update(rootfs=[]))


@pytest.mark.parametrize("alias", [False, True])
def test_duplicate_canonical_inner_paths_are_rejected(tmp_path, payload, alias):
    original = payload[1][0]
    duplicate = entry(("./" if alias else "") + original[0], original[1])
    report = verify(tmp_path, payload, [[*payload[1], duplicate]])
    assert "duplicate_layer_path" in codes(report)
    assert report["regular_files_read"] == 11  # Duplicate bytes are still scanned.
    assert report["retained_runtime_count"] == 8
    assert not report["valid"]


@pytest.mark.parametrize("whiteout", ["opt/.wh..", "opt/.wh...", ".wh..", ".wh..."])
def test_dot_and_parent_whiteout_targets_are_malformed(tmp_path, payload, whiteout):
    report = verify(tmp_path, payload, [[*payload[1], entry(whiteout)]])
    assert "malformed_whiteout" in codes(report)
    assert not report["valid"]


def test_same_file_in_distinct_layers_is_valid_replacement(tmp_path, payload):
    assert verify(tmp_path, payload, [payload[1], [payload[1][0]]])["valid"] is True


def save_oci_image(tmp_path, layers, *, docker_media=False, compressed=True,
                   update_manifest=None, update_index=None, update_saved=None,
                   replace_blob=None, duplicate_blob=False):
    """Independent complete descriptor graph, including deduplicated blob storage."""
    raw_layers = [tar_bytes(rows) for rows in layers]
    config = json.dumps({"rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(data) for data in raw_layers]}}).encode()
    blobs = {digest(config): config}

    def descriptor(data, media):
        blobs[digest(data)] = data
        return {"mediaType": media, "digest": "sha256:" + digest(data), "size": len(data)}

    layer_media = ("application/vnd.docker.image.rootfs.diff.tar" if docker_media else "application/vnd.oci.image.layer.v1.tar")
    if compressed:
        layer_media += ".gzip" if docker_media else "+gzip"
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json" if docker_media else "application/vnd.oci.image.manifest.v1+json",
        "config": descriptor(config, "application/vnd.docker.container.image.v1+json" if docker_media else "application/vnd.oci.image.config.v1+json"),
        "layers": [descriptor(gzip.compress(data, mtime=0) if compressed else data, layer_media) for data in raw_layers],
    }
    saved = [{"Config": "blobs/sha256/" + digest(config), "Layers": ["blobs/sha256/" + row["digest"][7:] for row in manifest["layers"]]}]
    if update_manifest:
        update_manifest(manifest)
    manifest_data = json.dumps(manifest).encode()
    manifest_descriptor = descriptor(manifest_data, manifest["mediaType"])
    manifest_descriptor["annotations"] = {"config.digest": "sha256:" + digest(config)}
    index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [manifest_descriptor]}
    if update_index:
        update_index(index)
    if update_saved:
        update_saved(saved)
    if replace_blob:
        replace_blob(blobs, manifest, manifest_descriptor)
    members = [entry("manifest.json", json.dumps(saved).encode()), entry("index.json", json.dumps(index).encode()),
               entry("oci-layout", b'{"imageLayoutVersion":"1.0.0"}')]
    members.extend(entry("blobs/sha256/" + sha, data) for sha, data in blobs.items())
    if duplicate_blob:
        members.append(members[-1])
    archive = tmp_path / "image.tar"
    archive.write_bytes(tar_bytes(members))
    return archive, "sha256:" + digest(manifest_data), "sha256:" + digest(config)


@pytest.mark.parametrize("docker_media,compressed", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("config_id", [False, True])
def test_oci_graph_binds_manifest_and_classic_ids_with_repeated_ordered_blobs(tmp_path, payload, docker_media, compressed, config_id):
    layers = [[], payload[1], []]
    archive, manifest_id, classic_id = save_oci_image(tmp_path, layers, docker_media=docker_media, compressed=compressed)
    report = VERIFIER.verify_image(archive, expected_image_id=classic_id if config_id else manifest_id, contract=payload[0])
    assert report["valid"]
    assert report["layer_count"] == 3
    assert report["verified_layer_diff_ids"][0] == report["verified_layer_diff_ids"][2]
    assert report["image_config_digest"] == classic_id
    assert report["image_manifest_digest"] == manifest_id
    assert report["regular_files_read"] == 10


def test_repeated_nonempty_blob_is_scanned_for_every_occurrence(tmp_path, payload):
    archive, image_id, _ = save_oci_image(tmp_path, [payload[1], payload[1]])
    report = VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])
    assert report["valid"]
    assert report["regular_files_read"] == 20
    assert report["content_bytes_read"] == 2 * sum(len(row[1]) for row in payload[1])


@pytest.mark.parametrize("update", [
    lambda m: m["config"].update(size=m["config"]["size"] + 1),
    lambda m: m["config"].update(digest="sha256:" + "0" * 64),
    lambda m: m["config"].update(mediaType="unsupported"),
    lambda m: m["layers"][0].update(size=True),
    lambda m: m["layers"][0].update(size=-1),
    lambda m: m["layers"][0].update(size=m["layers"][0]["size"] + 1),
    lambda m: m["layers"][0].update(digest="sha256:" + "0" * 64),
    lambda m: m["layers"][0].update(mediaType="application/vnd.oci.image.layer.v1.tar+zstd"),
    lambda m: m["layers"][0].update(mediaType="application/vnd.oci.image.layer.v1.tar"),
    lambda m: m["layers"].reverse(),
    lambda m: m["layers"].pop(),
    lambda m: m.update(schemaVersion=1),
])
def test_oci_graph_rejects_descriptor_tampering(tmp_path, payload, update):
    archive, image_id, _ = save_oci_image(tmp_path, [[], payload[1]], update_manifest=update)
    with pytest.raises((VERIFIER.ImageVerificationError, KeyError)):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


@pytest.mark.parametrize("update", [
    lambda i: i["manifests"].append(copy.deepcopy(i["manifests"][0])),
    lambda i: i["manifests"][0].update(digest="sha256:" + "0" * 64),
    lambda i: i["manifests"][0].update(size=i["manifests"][0]["size"] + 1),
    lambda i: i["manifests"][0]["annotations"].update({"config.digest": "sha256:" + "0" * 64}),
    lambda i: i.update(schemaVersion=1),
])
def test_oci_index_cannot_redirect_or_ambiguously_select_image(tmp_path, payload, update):
    archive, image_id, _ = save_oci_image(tmp_path, [payload[1]], update_index=update)
    with pytest.raises((VERIFIER.ImageVerificationError, KeyError)):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def test_oci_manifest_and_blob_bytes_are_hashed_not_trusted_from_names(tmp_path, payload):
    def replace(blobs, manifest, descriptor):
        sha = descriptor["digest"][7:]
        blobs[sha] = blobs[sha].replace(b'"schemaVersion": 2', b'"schemaVersion": 1')

    archive, image_id, _ = save_oci_image(tmp_path, [payload[1]], replace_blob=replace)
    with pytest.raises(VERIFIER.ImageVerificationError, match="manifest digest"):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def test_oci_compressed_blob_identity_detects_same_diff_id_with_different_compression(tmp_path, payload):
    def replace(blobs, manifest, descriptor):
        sha = manifest["layers"][0]["digest"][7:]
        blobs[sha] = gzip.compress(gzip.decompress(blobs[sha]), mtime=1)

    archive, image_id, _ = save_oci_image(tmp_path, [payload[1]], replace_blob=replace)
    with pytest.raises(VERIFIER.ImageVerificationError, match="blob digest"):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def test_repeated_references_do_not_allow_duplicate_outer_entries(tmp_path, payload):
    archive, image_id, _ = save_oci_image(tmp_path, [[], payload[1], []], duplicate_blob=True)
    with pytest.raises(VERIFIER.ImageVerificationError, match="duplicate saved-image"):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def test_oci_compatibility_manifest_cannot_reorder_the_described_layers(tmp_path, payload):
    archive, image_id, _ = save_oci_image(tmp_path, [[], payload[1]], update_saved=lambda rows: rows[0]["Layers"].reverse())
    with pytest.raises(VERIFIER.ImageVerificationError, match="layer order"):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


@pytest.fixture
def adapter_payload(payload):
    contract, entries, excluded = copy.deepcopy(payload)
    adapters = copy.deepcopy(json.loads((IMAGE / "runtime-payload.json").read_text())["torch_cudnn_adapters"])
    root = contract["cudnn"]["install_root"]
    for index, row in enumerate([*adapters["headers"], adapters["license"]]):
        data = f"Synthetic independently reviewed PyTorch BSD adapter or full notice {index}.".encode()
        row.update(sha256=digest(data), size=len(data))
        entries.append(entry(f"{root}/{row['path']}", data))
    contract["torch_cudnn_adapters"] = adapters
    return contract, entries, excluded


def test_reviewed_torch_adapter_bytes_and_complete_license_pass(tmp_path, adapter_payload):
    report = verify(tmp_path, adapter_payload)
    assert report["valid"]
    assert report["verified_torch_adapter_count"] == 52
    assert report["required_payload_count"] == 63
    assert report["regular_files_read"] == 63


@pytest.mark.parametrize("change", ["modified", "missing", "relocated", "sdk_bytes", "symlink", "missing_license", "modified_license"])
def test_adapter_exception_never_trusts_path_alone(tmp_path, adapter_payload, change):
    contract, entries, excluded = adapter_payload
    first_adapter = 10
    if change == "modified":
        entries[first_adapter] = entry(entries[first_adapter][0], b"unreviewed SDK header")
    elif change == "missing":
        del entries[first_adapter]
    elif change == "relocated":
        entries[first_adapter] = entry("opt/elsewhere/cudnn_affine_grid_generator.h", entries[first_adapter][1])
    elif change == "sdk_bytes":
        entries[first_adapter] = entry(entries[first_adapter][0], excluded[0])
    elif change == "symlink":
        entries[first_adapter] = entry(entries[first_adapter][0], kind=tarfile.SYMTYPE, link="untrusted")
    elif change == "missing_license":
        entries.pop()
    elif change == "modified_license":
        entries[-1] = entry(entries[-1][0], b"truncated notice")
    report = verify(tmp_path, (contract, entries, excluded))
    assert not report["valid"]
    if change == "sdk_bytes":
        assert "excluded_cudnn_sdk_bytes" in codes(report)


def test_unknown_header_under_torch_namespace_is_still_rejected(tmp_path, adapter_payload):
    contract, entries, _ = adapter_payload
    root = contract["cudnn"]["install_root"]
    entries.append(entry(f"{root}/torch/include/ATen/ops/cudnn_unreviewed.h", b"unreviewed"))
    assert "excluded_cudnn_sdk_path" in codes(verify(tmp_path, adapter_payload))


def test_ancestor_adapter_tampering_remains_rejected_after_valid_replacement(tmp_path, adapter_payload):
    path = adapter_payload[1][10][0]
    report = verify(tmp_path, adapter_payload, [[entry(path, b"old forbidden header")], adapter_payload[1]])
    assert not report["valid"]
    assert report["verified_torch_adapter_count"] == 52
    assert "retained_payload_hash_mismatch" in codes(report)


def test_adapter_whiteout_cannot_keep_prior_proof(tmp_path, adapter_payload):
    path = Path(adapter_payload[1][10][0])
    report = verify(tmp_path, adapter_payload, [adapter_payload[1], [entry(str(path.parent / (".wh." + path.name)))]])
    assert not report["valid"]
    assert report["verified_torch_adapter_count"] == 51


def test_adapter_inventory_is_bound_to_locked_official_wheel_and_full_notice():
    adapters = json.loads((IMAGE / "runtime-payload.json").read_text())["torch_cudnn_adapters"]
    assert adapters["name"] == "torch" and adapters["version"] == "2.9.1+cu130"
    assert adapters["wheel_sha256"] == "e70e1b18881e6b3c1ce402d0a989da39f956a3a057526e03c354df23d704ce9b"
    assert "--hash=sha256:" + adapters["wheel_sha256"] in (IMAGE / "requirements.lock").read_text()
    assert adapters["wheel_url"] == "https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert len(adapters["headers"]) == 52
    assert len({row["path"] for row in adapters["headers"]}) == 52
    assert all(row["path"].startswith("torch/include/ATen/ops/cudnn_") for row in adapters["headers"])
    assert adapters["license"]["sha256"] == "776e43288ab54330aa1d942737754b0edef0e7a07b62b736b595136ca5ccae88"
    assert adapters["license"]["size"] == 499846
    assert adapters["source_license_sha256"] == "a8a2c6b67cf8a597ffaa5c82380a6bd1b82629c0417e00673d8013d662eb500a"


@pytest.mark.parametrize("compress", [bz2.compress, lzma.compress])
def test_declared_raw_oci_layer_cannot_auto_decode_an_unsupported_codec(tmp_path, payload, compress):
    compressed = compress(tar_bytes(payload[1]))
    config = json.dumps({"rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(compressed)]}}).encode()
    manifest = {
        "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": "sha256:" + digest(config), "size": len(config)},
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": "sha256:" + digest(compressed), "size": len(compressed)}],
    }
    manifest_bytes = json.dumps(manifest).encode()
    index = {"schemaVersion": 2, "manifests": [{"mediaType": manifest["mediaType"], "digest": "sha256:" + digest(manifest_bytes), "size": len(manifest_bytes)}]}
    saved = [{"Config": "blobs/sha256/" + digest(config), "Layers": ["blobs/sha256/" + digest(compressed)]}]
    archive = tmp_path / "unsupported-codec.tar"
    archive.write_bytes(tar_bytes([
        entry("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'), entry("index.json", json.dumps(index).encode()),
        entry("manifest.json", json.dumps(saved).encode()), entry("blobs/sha256/" + digest(config), config),
        entry("blobs/sha256/" + digest(manifest_bytes), manifest_bytes), entry("blobs/sha256/" + digest(compressed), compressed),
    ]))
    with pytest.raises(tarfile.ReadError):
        VERIFIER.verify_image(archive, expected_image_id="sha256:" + digest(manifest_bytes), contract=payload[0])
