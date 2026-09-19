from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCANNER = ROOT / "npa/scripts/scan_image_gymnasium_robotics_payload.py"
WORKFLOW = ROOT / ".github/workflows/publish-public-images.yml"


def test_scanner_covers_config_all_layers_whiteouts_and_rootfs_entries() -> None:
    text = SCANNER.read_text(encoding="utf-8")
    whitespace_independent_text = "".join(text.split())
    assert "_zip_central_filename_bytes(path,info)" in whitespace_independent_text
    assert 'layer_names=entry.get("Layers")' in whitespace_independent_text
    assert (
        "ifnotisinstance(layer_names,list)ornotlayer_names:"
        in whitespace_independent_text
    )
    assert "iflen(layer_names)>MAX_ORDERED_LAYERS:" in whitespace_independent_text
    assert (
        "_validated_tar_members(path,content,max_members="
        "MAX_NESTED_ARCHIVE_MEMBERS)"
        in whitespace_independent_text
    )
    for token in (
        '"manifest.json"',
        '_scan_decoded_member_bytes("exact image config", config_raw)',
        "_docker_save_config_digest(config_name) != config_digest",
        're.fullmatch(r"([0-9a-f]{64})\\.json", config_name)',
        're.fullmatch(r"blobs/sha256/([0-9a-f]{64})", config_name)',
        'config_rootfs.get("diff_ids") != layer_diff_ids',
        '_scan_raw_blob_bytes(f"raw layer bytes: {layer_name}", raw)',
        '_scan_raw_blob_bytes("complete Docker-save archive", archive_bytes)',
        '_scan_archive_representation_bytes(',
        '_validate_zip_compressed_stream(path, info, content[data_start:data_end])',
        'stream.unused_data',
        'f"decoded member: {path}"',
        'f"raw archive member: {path}"',
        "_whiteout_metadata(layer, item, path, layer_name)",
        'leaf == ".wh..wh..opq"',
        'leaf.startswith(".wh.")',
        "rootfs[path] = content",
        'kind = "symlink" if item.issym() else "hardlink"',
        "zipfile.is_zipfile(io.BytesIO(content))",
        "compression_kind = next(",
        "is_tar = _looks_like_tar(content)",
        "MAX_NESTED_ARCHIVE_MEMBERS = 10_000",
        "MAX_OCI_DESCRIPTOR_GRAPH_VISITS = MAX_DOCKER_SAVE_OUTER_MEMBERS",
        "graph_budget.reserve(label)",
        "graph_budget.validated_blobs.get(digest)",
        "nonzero tar link body",
        "max(disk_entries, total_entries) > MAX_NESTED_ARCHIVE_MEMBERS",
        "_validated_zip_infos(path, content, budget=budget)",
        "_validate_zip_data_descriptor(path, descriptor, info)",
        "zip local filename does not match central directory",
        "unsupported zip extra field",
        "zip local descriptor metadata is not zero",
        'struct.unpack("<3L", fields)',
        "_nested_archive_members(",
        "nested_budget=nested_budget",
        "allowed_system_wheel_path=",
        '"requirements.lock": "30d48e4b2bfcf0c590b47ed569393104dd759476d720a608aa9f441cd9976e4a"',
        "KNOWN_FORBIDDEN_CONTENT_SHA256",
        "forbidden upstream/runtime byte",
        "six-boundary runtime delivery classification changed",
        "runtime artifact closure is incomplete",
        "neutral image corresponding-source closure is incomplete",
        "final image must declare the non-root ubuntu user",
        '"unresolved_findings": 0',
        '"upstream_runtime_payload_count": 0',
        '"shadow_asset_count": 0',
        '"runtime_cache_entry_count": 0',
        '"whiteout_metadata_sha256"',
        '"release_authorized": False',
    ):
        assert token in text


def test_product_scan_is_staged_before_push_and_after_exact_pull() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("scan_image_gymnasium_robotics_payload.py") == 2
    before_push = text.index("scan_image_gymnasium_robotics_payload.py")
    push = text.index('docker push "$IMAGE"')
    after_pull = text.rindex("scan_image_gymnasium_robotics_payload.py")
    assert before_push < push < after_pull
    for gate in (
        "Generate pre-publication SBOM",
        "Attest exact pushed digest provenance",
        "Attest exact pushed digest SBOM",
        "--scanners vuln,secret,license",
        'anonymous_config="$(mktemp -d)"',
    ):
        assert gate in text


def test_neutral_payload_scan_is_verified_before_development_image_push() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    source_gate = text.index("Require Gymnasium neutral payload scan before development push")
    push = text.index('docker push "$IMAGE"')
    assert source_gate < push
    for token in (
        '--metadata-file "$RUNNER_TEMP/${TOOL}-build-metadata.json"',
        '.["containerimage.config.digest"]',
        'payload="$RUNNER_TEMP/${TOOL}-gymnasium-payload.json"',
        '.status == "passed"',
        '.distributed_blob_scan_complete == true',
        '.accepted_manifest_present == false',
        '.release_authorized == false',
    ):
        assert token in text


def test_trusted_workflow_allows_neutral_development_selection_before_build() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    build = text.index("docker buildx build")
    assert "pre-registration candidate has no build authority" not in text
    assert "Require Gymnasium neutral payload scan before development push" in text
    assert build > text.index("Resolve immutable public development plan")


def test_public_workflow_uses_hash_bound_graph_not_candidate_digests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    scanner_text = SCANNER.read_text(encoding="utf-8")
    assert (
        "ef9104df9ec9c2f85a26a2ea38db3b1c27565b3e68eb84cfff508d6969ae5ba5"
        in scanner_text
    )
    assert '"schema": "npa.gymnasium-robotics.oci-graph.v1"' in scanner_text
    assert (
        "500cf71fa1d9e0a75d4964145ad5cabeb8f4dab213daa47e3eb4062daa26d8ee"
        in scanner_text
    )
    assert "--reviewed-graph-record" in text
    assert text.count('gymnasium_scan_args=(--oci-layout "$archive")') == 2
    assert text.count('gymnasium_scan_args=(--docker-save "$archive")') == 2
    assert text.count('ambiguous Gymnasium archive format') == 1
    assert text.count('ambiguous pushed Gymnasium archive format') == 1
    assert text.count('.archive_format == "oci-layout"') == 2
    assert text.count('.archive_format == "docker-save"') == 2
    assert text.count('(.ordered_layer_descriptors | length) == .layer_count') == 2
    assert "if args.reviewed_graph_record and (" in (
        SCANNER.read_text(encoding="utf-8")
    )
    assert "--expected-config-sha256" not in text
    assert "--expected-layer-diff-ids-json" not in text
    assert "docker image inspect --format '{{json .RootFS.Layers}}'" not in text


def test_future_runtime_stage_proves_the_non_root_user_before_switching() -> None:
    dockerfile = (
        ROOT / "npa/docker/workbench/gymnasium-robotics/Dockerfile"
    ).read_text(encoding="utf-8")
    proof = dockerfile.index('RUN test "$(id -u ubuntu)" = 1000')
    user = dockerfile.index("USER ubuntu")
    assert proof < user


def test_dockerfile_never_copies_runtime_or_upstream_payload() -> None:
    dockerfile = (
        ROOT / "npa/docker/workbench/gymnasium-robotics/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "COPY --from=" not in dockerfile
    assert "/opt/venv" not in dockerfile
    assert ".whl" not in dockerfile
    assert "runtime-bootstrap.py" in dockerfile
