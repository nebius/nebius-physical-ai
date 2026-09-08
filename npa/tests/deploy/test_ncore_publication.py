"""Synthetic acceptance evidence only; no fixture establishes a public release."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from npa.deploy import images, publish_public as publish


SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
PLATFORM = "sha256:" + "c" * 64
CONFIG = "sha256:" + "d" * 64
HASH = "e" * 64
REPOSITORY = images.DEFAULT_PUBLIC_CONTAINER_REGISTRY + "/npa-ncore"


@pytest.fixture
def accepted():
    scan = {"status": "pass", "report_sha256": HASH, "image_digest": DIGEST}
    return {
        "format": "npa_ncore_accepted_image_manifest_v1",
        "status": "accepted",
        "tag": images.public_release_tag_for_tool("ncore"),
        "development_sha": SHA,
        "oci_digest": DIGEST,
        "amd64_manifest": PLATFORM,
        "config_digest": CONFIG,
        "source": {
            "ncore_revision": "59c698d206da92b406a4f72619fce3b3a2c64bfd",
            "lock_sha256": HASH,
            "post_patch_inventory_sha256": HASH,
        },
        "byte_scan": {
            **scan,
            "archive_sha256": HASH,
            "policy_sha256": HASH,
            "config_digest": CONFIG,
            "complete": True,
            "bytes_scanned": 1024,
            "files_scanned": 10,
            "unresolved_findings": 0,
        },
        "payload_scan": {
            **scan,
            "entries_scanned": 10,
            "payload_hits": 0,
            "history_hits": 0,
            "weight_shaped_paths": 0,
            "weight_review_sha256": HASH,
        },
        "vulnerability_scan": {
            **scan,
            "critical_total": 0,
            "critical_with_fix": 0,
            "critical_unfixed": 0,
            "secrets": 0,
        },
        "license_scan": {**scan, "unresolved_findings": 0},
        "selected_base_scan": {
            **scan,
            "format": "npa_ncore_selected_base_scan_v1",
            "platform_digest": PLATFORM,
            "config_digest": CONFIG,
            "rootfs_sha256": HASH,
            "lock_sha256": HASH,
            "sbom_sha256": HASH,
            "packages_sha256": HASH,
            "files_verified": 3,
            "symlinks_verified": 1,
            "packages_evaluated": 1,
            "critical_total": 0,
            "critical_with_fix": 0,
            "critical_unfixed": 0,
            "secrets": 0,
        },
        "conversion": {
            "status": "pass",
            "exit_code": 0,
            "observed_image_digest": PLATFORM,
            "report_sha256": HASH,
            "source_archive_sha256": "cf7ab7f100da66b2bf05b178ebcfa3a950e1bf2b1d7ff64a6c7a1e1f682afa8d",
            "source_inventory_sha256": HASH,
            "converted_inventory_sha256": HASH,
            "dataset_repository": "nvidia/PhysicalAI-NuRec-PPISP",
            "dataset_revision": "2521064a3af6ab1c1caa2ba1b01ddde7eecded69",
            "dataset_root": "struktur28",
            "source_counts": {"images": 518, "cameras": 3, "points": 163453},
            "converted_counts": {"images": 518, "cameras": 3, "points": 163450},
            "origin_points_filtered": 3,
            "camera_frame_counts": {"camera1": 200, "camera2": 200, "camera3": 118},
            "camera_frame_inventory_sha256": HASH,
            "all_members_reopened": True,
            "member_hashes_verified": True,
            "calibration_verified": True,
            "poses_verified": True,
            "finite_geometry": True,
            "rig_mode": "derive",
            "poses_component_group": "npa_rig",
        },
        "rtx_proof": {
            "status": "pass",
            "report_sha256": HASH,
            "conversion_report_sha256": HASH,
            "converted_inventory_sha256": HASH,
            "nre_image": "nvcr.io/nvidia/nre/nre-ga@sha256:" + "f" * 64,
            "observed_nre_digest": "sha256:" + "f" * 64,
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_count": 1,
            "max_epochs": 0,
            "train_exit_code": 0,
            "training_steps": 100,
            "gaussian_count": 1000,
            "usdz_sha256": HASH,
            "usdz_bytes": 1024,
            "trained_scene_reopened": True,
            "render_exit_code": 0,
            "rendered_usdz_sha256": HASH,
            "render_sha256": HASH,
            "render_bytes": 1024,
            "decoded_frames": 10,
            "finite_pixels": True,
            "novel_view": True,
            "training_input": {
                "inventory_report_sha256": HASH,
                "parsed_config_sha256": HASH,
                "native_recipe_sha256": HASH,
                "datasource_summary_sha256": HASH,
                "split_report_sha256": HASH,
                "native_recipe": "configs/experimental/3dgut/3dgut_colmap.yaml",
                "native_epochs": 1,
                "resolved_epochs": 1,
                "native_samples_per_epoch": 30000,
                "resolved_samples_per_epoch": 30000,
                "native_recipe_completed": True,
                "loaded_camera_frame_counts": {
                    "camera1": 200,
                    "camera2": 200,
                    "camera3": 118,
                },
                "eligible_training_camera_frame_counts": {
                    "camera1": 190,
                    "camera2": 190,
                    "camera3": 108,
                },
                "validation_camera_frame_counts": {
                    "camera1": 10,
                    "camera2": 10,
                    "camera3": 10,
                },
                "covered_camera_frame_counts": {
                    "camera1": 200,
                    "camera2": 200,
                    "camera3": 118,
                },
                "source_frame_inventory_sha256": HASH,
                "covered_frame_inventory_sha256": HASH,
                "independent_frame_readback": True,
            },
            "rrd": {
                "report_sha256": HASH,
                "sha256": HASH,
                "bytes": 1024,
                "decoded_rows": 20,
                "decoded_frames": 10,
                "conversion_report_sha256": HASH,
                "source_inventory_sha256": HASH,
                "lineage_verified": True,
                "frame_artifacts_verified": True,
            },
        },
    }


def test_checked_in_ncore_remains_unaccepted():
    assert images.is_publicly_redistributable("ncore")
    assert "ncore" in images.PUBLICATION_QUARANTINE_TOOLS
    assert "ncore" not in images.publicly_publishable_tools()
    assert images.development_image_for_tool("ncore", git_sha=SHA).endswith(
        "dev-" + SHA
    )
    with pytest.raises(ValueError, match="no accepted release"):
        images.container_image_for_tool("ncore")
    with pytest.raises(RuntimeError, match="NCore"):
        images.ncore_accepted_image_manifest()


def test_acceptance_validates_complete_evidence(accepted):
    assert images.validate_ncore_accepted_image_manifest(accepted) == accepted


@pytest.mark.parametrize(
    "path,value",
    [
        ("status", "pending"),
        ("development_sha", "main"),
        ("oci_digest", "sha256:abc"),
        ("amd64_manifest", None),
        ("tag", "wrong"),
        ("source.ncore_revision", "a" * 40),
        ("source.post_patch_inventory_sha256", ""),
        ("byte_scan.complete", False),
        ("byte_scan.bytes_scanned", True),
        ("byte_scan.files_scanned", 0),
        ("byte_scan.unresolved_findings", 1),
        ("byte_scan.config_digest", DIGEST),
        ("byte_scan.image_digest", PLATFORM),
        ("license_scan.unresolved_findings", 1),
        ("payload_scan.payload_hits", False),
        ("payload_scan.history_hits", 1),
        ("payload_scan.weight_shaped_paths", -1),
        ("payload_scan.weight_review_sha256", ""),
        ("vulnerability_scan.critical_with_fix", 1),
        ("vulnerability_scan.secrets", 1),
        ("vulnerability_scan.critical_total", 1),
        ("selected_base_scan.format", "lock-only"),
        ("selected_base_scan.status", "unverified"),
        ("selected_base_scan.image_digest", PLATFORM),
        ("selected_base_scan.platform_digest", DIGEST),
        ("selected_base_scan.config_digest", DIGEST),
        ("selected_base_scan.rootfs_sha256", ""),
        ("selected_base_scan.sbom_sha256", ""),
        ("selected_base_scan.report_sha256", ""),
        ("selected_base_scan.lock_sha256", ""),
        ("selected_base_scan.packages_sha256", ""),
        ("selected_base_scan.files_verified", True),
        ("selected_base_scan.files_verified", 0),
        ("selected_base_scan.symlinks_verified", -1),
        ("selected_base_scan.packages_evaluated", 0),
        ("selected_base_scan.critical_total", 1),
        ("selected_base_scan.critical_with_fix", 1),
        ("selected_base_scan.secrets", 1),
        ("conversion.observed_image_digest", CONFIG),
        ("conversion.dataset_root", "struktur28_auto"),
        ("conversion.source_counts.images", 59),
        ("conversion.converted_counts.cameras", 2),
        ("conversion.converted_counts.points", 163449),
        ("conversion.calibration_verified", False),
        ("conversion.member_hashes_verified", "true"),
        ("conversion.finite_geometry", False),
        ("conversion.rig_mode", "preserve"),
        ("rtx_proof.conversion_report_sha256", "1" * 64),
        ("rtx_proof.converted_inventory_sha256", "2" * 64),
        ("rtx_proof.gpu_model", "NVIDIA H200"),
        ("rtx_proof.nre_image", "nvcr.io/nvidia/nre/nre-ga:latest"),
        ("rtx_proof.observed_nre_digest", DIGEST),
        ("rtx_proof.max_epochs", 1),
        ("rtx_proof.train_exit_code", 1),
        ("rtx_proof.training_steps", 0),
        ("rtx_proof.gaussian_count", 0),
        ("rtx_proof.usdz_bytes", 0),
        ("rtx_proof.trained_scene_reopened", False),
        ("rtx_proof.rendered_usdz_sha256", "3" * 64),
        ("rtx_proof.decoded_frames", 0),
        ("rtx_proof.render_exit_code", 1),
        ("rtx_proof.finite_pixels", False),
        ("rtx_proof.novel_view", False),
        ("conversion.camera_frame_counts.camera3", 117),
        ("rtx_proof.training_input.loaded_camera_frame_counts.camera1", 199),
        (
            "rtx_proof.training_input.eligible_training_camera_frame_counts",
            {"camera1": 200},
        ),
        ("rtx_proof.training_input.eligible_training_camera_frame_counts.camera3", 0),
        ("rtx_proof.training_input.eligible_training_camera_frame_counts.camera3", 119),
        ("rtx_proof.training_input.validation_camera_frame_counts.camera1", 0),
        ("rtx_proof.training_input.covered_camera_frame_counts.camera1", 199),
        ("rtx_proof.training_input.covered_frame_inventory_sha256", "f" * 64),
        ("rtx_proof.training_input.source_frame_inventory_sha256", "f" * 64),
        ("rtx_proof.training_input.independent_frame_readback", False),
        ("rtx_proof.training_input.native_recipe", "unverified.yaml"),
        ("rtx_proof.training_input.native_epochs", 2),
        ("rtx_proof.training_input.resolved_epochs", 2),
        ("rtx_proof.training_input.resolved_samples_per_epoch", 15000),
        ("rtx_proof.training_input.native_recipe_completed", False),
        ("rtx_proof.rrd.conversion_report_sha256", "f" * 64),
        ("rtx_proof.rrd.source_inventory_sha256", "f" * 64),
        ("rtx_proof.rrd.decoded_frames", 0),
        ("rtx_proof.rrd.decoded_rows", True),
        ("rtx_proof.rrd.lineage_verified", False),
        ("rtx_proof.rrd.frame_artifacts_verified", False),
    ],
)
def test_acceptance_rejects_unproven_or_mismatched_evidence(accepted, path, value):
    target = accepted
    parts = path.split(".")
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    with pytest.raises(RuntimeError, match="NCore"):
        images.validate_ncore_accepted_image_manifest(accepted)


def test_every_evidence_field_is_required(accepted):
    def leaves(record, prefix=()):
        for key, value in record.items():
            yield prefix + (key,)
            if isinstance(value, dict):
                yield from leaves(value, prefix + (key,))

    for path in leaves(accepted):
        candidate = copy.deepcopy(accepted)
        target = candidate
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
        with pytest.raises(RuntimeError, match="NCore"):
            images.validate_ncore_accepted_image_manifest(candidate)


def test_source_archive_must_be_the_pinned_public_capture(accepted):
    accepted["conversion"]["source_archive_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="source_archive_sha256"):
        images.validate_ncore_accepted_image_manifest(accepted)


def test_first_release_plan_uses_accepted_source(accepted, monkeypatch):
    monkeypatch.setattr(images, "ncore_accepted_image_manifest", lambda: accepted)
    monkeypatch.setattr(images, "public_release_manifest", lambda: {"releases": {}})
    assert images.accepted_publication_development_sha("ncore") == SHA


@pytest.mark.parametrize(
    "field,value", [("development_sha", "f" * 40), ("published_digest", CONFIG)]
)
def test_release_record_cannot_disagree(accepted, monkeypatch, field, value):
    entry = {"development_sha": SHA, "published_digest": DIGEST}
    entry[field] = value
    monkeypatch.setattr(images, "ncore_accepted_image_manifest", lambda: accepted)
    monkeypatch.setattr(
        images, "public_release_manifest", lambda: {"releases": {"ncore": entry}}
    )
    with pytest.raises(RuntimeError, match="NCore"):
        images.accepted_publication_development_sha("ncore")


def test_removing_quarantine_alone_does_not_enable_default_resolution(monkeypatch):
    monkeypatch.setattr(images, "PUBLICATION_QUARANTINE_TOOLS", frozenset())
    with pytest.raises(RuntimeError, match="NCore"):
        images.container_image_for_tool("ncore")


@pytest.fixture
def registry(accepted, monkeypatch):
    """Offline buildx-shaped SPDX and SLSA with actual statement blob hashes."""
    statements = [
        {
            "predicateType": "https://spdx.dev/Document",
            "predicate": {"packages": [{"name": "synthetic"}]},
        },
        {
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "predicate": {
                "buildType": "https://mobyproject.org/buildkit@v1",
                "materials": [
                    {"uri": "pkg:docker/python", "digest": {"sha256": "a" * 64}}
                ],
                "invocation": {"parameters": {"args": {"build-arg:SOURCE_SHA": SHA}}},
            },
        },
    ]
    for statement in statements:
        statement["_type"] = "https://in-toto.io/Statement/v0.1"
        statement["subject"] = [
            {"name": "synthetic", "digest": {"sha256": PLATFORM[7:]}}
        ]
    layers = [
        {
            "digest": "sha256:" + hashlib.sha256(json.dumps(s).encode()).hexdigest(),
            "annotations": {"in-toto.io/predicate-type": s["predicateType"]},
        }
        for s in statements
    ]
    attestation_digest = "sha256:" + "4" * 64
    index = {
        "manifests": [
            {"digest": PLATFORM, "platform": {"architecture": "amd64", "os": "linux"}},
            {
                "digest": attestation_digest,
                "platform": {"architecture": "unknown", "os": "unknown"},
                "annotations": {
                    "vnd.docker.reference.type": "attestation-manifest",
                    "vnd.docker.reference.digest": PLATFORM,
                },
            },
        ]
    }
    platform = {
        "config": {"digest": CONFIG},
        "layers": [{"digest": "sha256:" + "5" * 64}],
    }
    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "User": "ubuntu",
            "Labels": {
                "org.opencontainers.image.revision": SHA,
                "npa.ncore.revision": accepted["source"]["ncore_revision"],
            },
        },
    }

    def crane(args):
        if args[0] == "config":
            assert args[1] == REPOSITORY + "@" + PLATFORM
            return config
        return {
            DIGEST: index,
            PLATFORM: platform,
            attestation_digest: {"layers": layers},
        }[args[1].split("@")[1]]

    calls = []
    monkeypatch.setattr(images, "ncore_accepted_image_manifest", lambda: accepted)
    monkeypatch.setattr(publish, "_crane_digest", lambda ref: (True, DIGEST))
    monkeypatch.setattr(publish, "_crane_json", crane)
    monkeypatch.setattr(
        publish,
        "_crane_blob_json",
        lambda repo, digest: statements[
            next(i for i, layer in enumerate(layers) if layer["digest"] == digest)
        ],
    )
    counts = {
        k: accepted["payload_scan"][k]
        for k in (
            "entries_scanned",
            "payload_hits",
            "history_hits",
            "weight_shaped_paths",
        )
    }
    monkeypatch.setattr(
        publish,
        "_scan_ncore_payload_exact_digest",
        lambda ref: calls.append(ref) or counts,
    )
    vulnerability = {
        k: accepted["vulnerability_scan"][k]
        for k in ("critical_total", "critical_with_fix", "critical_unfixed", "secrets")
    }
    monkeypatch.setattr(
        publish,
        "_scan_trivy_exact_digest",
        lambda ref, **kwargs: calls.append(ref) or vulnerability,
    )
    def selected_scan(ref, *, platform_digest, config_digest):
        assert platform_digest == PLATFORM and config_digest == CONFIG
        calls.append(ref)
        return copy.deepcopy(accepted["selected_base_scan"])

    monkeypatch.setattr(publish, "_scan_ncore_selected_base_exact_digest", selected_scan)
    return index, statements, layers, config, calls


def item():
    return publish.PublishItem(
        "ncore",
        REPOSITORY + "@" + DIGEST,
        REPOSITORY + ":" + images.public_release_tag_for_tool("ncore"),
    )


def test_exact_attested_index_passes(accepted, registry):
    assert publish.verify_ncore_publication_source(item())[0]
    assert registry[-1] == [REPOSITORY + "@" + DIGEST] * 3


def test_buildx_v1_provenance_and_oci_artifact_subject_pass(registry, monkeypatch):
    _, statements, layers, _, _ = registry
    provenance = statements[1]
    predicate = provenance["predicate"]
    provenance["_type"] = "https://in-toto.io/Statement/v1"
    provenance["predicateType"] = "https://slsa.dev/provenance/v1"
    provenance["predicate"] = {
        "buildDefinition": {
            "buildType": "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md",
            "externalParameters": {"request": predicate["invocation"]["parameters"]},
            "resolvedDependencies": predicate["materials"],
        }
    }
    layers[1]["annotations"]["in-toto.io/predicate-type"] = provenance["predicateType"]
    original = publish._crane_json

    def crane(args):
        result = original(args)
        if result.get("layers") is layers:
            result["artifactType"] = (
                "application/vnd.docker.attestation.manifest.v1+json"
            )
            result["subject"] = {"digest": PLATFORM}
        return result

    monkeypatch.setattr(publish, "_crane_json", crane)
    assert publish.verify_ncore_publication_source(item())[0]


def test_trusted_build_npa_source_arg_passes(registry):
    args = registry[1][1]["predicate"]["invocation"]["parameters"]["args"]
    args["build-arg:NPA_SOURCE_SHA"] = args.pop("build-arg:SOURCE_SHA")
    assert publish.verify_ncore_publication_source(item())[0]


@pytest.mark.parametrize(
    "damage",
    [
        "conflicting-source",
        "missing-material-digest",
        "index-drift",
        "config-drift",
        "root-zero-padded",
        "artifact-subject",
    ],
)
def test_additional_live_bindings_are_fail_closed(registry, monkeypatch, damage):
    _, statements, _, config, _ = registry
    if damage == "conflicting-source":
        statements[1]["predicate"]["invocation"]["parameters"]["args"][
            "build-arg:NPA_SOURCE_SHA"
        ] = "f" * 40
    elif damage == "missing-material-digest":
        statements[1]["predicate"]["materials"][0]["digest"] = {}
    elif damage == "index-drift":
        monkeypatch.setattr(publish, "_crane_digest", lambda ref: (True, PLATFORM))
    elif damage == "root-zero-padded":
        config["config"]["User"] = "00:1000"
    else:
        original = publish._crane_json

        def crane(args):
            result = original(args)
            if damage == "config-drift" and args == [
                "manifest",
                REPOSITORY + "@" + PLATFORM,
            ]:
                result["config"]["digest"] = DIGEST
            elif damage == "artifact-subject" and result.get("layers") is registry[2]:
                result["subject"] = {"digest": DIGEST}
            return result

        monkeypatch.setattr(publish, "_crane_json", crane)
    assert not publish.verify_ncore_publication_source(item())[0]


@pytest.mark.parametrize("field", ["source_ref", "target_ref"])
def test_publication_cannot_select_other_source_or_target(registry, field):
    from dataclasses import replace

    changed = replace(item(), **{field: REPOSITORY + ":unaccepted"})
    assert not publish.verify_ncore_publication_source(changed)[0]
    assert not registry[-1]


@pytest.mark.parametrize("gate", ["payload", "vulnerability"])
def test_live_scans_must_reproduce_recorded_counts(registry, monkeypatch, gate):
    if gate == "payload":
        monkeypatch.setattr(publish, "_scan_ncore_payload_exact_digest", lambda ref: {})
    else:
        monkeypatch.setattr(
            publish,
            "_scan_trivy_exact_digest",
            lambda ref, **kw: {
                "critical_total": 1,
                "critical_with_fix": 0,
                "critical_unfixed": 1,
                "secrets": 0,
            },
        )
    ok, detail = publish.verify_ncore_publication_source(item())
    assert not ok
    assert "counts differ" in detail


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "lock-only"),
        ("image_digest", PLATFORM),
        ("platform_digest", DIGEST),
        ("config_digest", DIGEST),
        ("lock_sha256", "f" * 64),
        ("packages_sha256", "f" * 64),
        ("files_verified", 4),
        ("symlinks_verified", 2),
        ("packages_evaluated", 2),
        ("critical_total", 1),
        ("critical_unfixed", 1),
        ("critical_with_fix", 1),
        ("secrets", False),
    ],
)
def test_live_selected_scan_refuses_mismatched_evidence(
    accepted, registry, monkeypatch, field, value
):
    scan = {**accepted["selected_base_scan"], field: value}
    monkeypatch.setattr(
        publish, "_scan_ncore_selected_base_exact_digest", lambda *args, **kwargs: scan
    )
    ok, detail = publish.verify_ncore_publication_source(item())
    assert not ok and "live selected-base" in detail


def test_supplemental_spdx_cannot_replace_or_duplicate_buildx_predicate(registry):
    registry[2].append(copy.deepcopy(registry[2][0]))
    ok, detail = publish.verify_ncore_publication_source(item())
    assert not ok and "duplicate attestation" in detail


@pytest.mark.parametrize(
    "field,value",
    [
        ("scan_complete", False),
        ("history_only", True),
        ("digest", CONFIG),
        ("entries_scanned", 0),
        ("entries_scanned", True),
        ("verdict", "failed"),
        ("payload_hits", ["synthetic"]),
        ("history_hits", None),
        ("weight_shaped_paths", None),
    ],
)
def test_payload_scan_rejects_incomplete_or_unsafe_reports(
    monkeypatch, tmp_path, field, value
):
    from types import SimpleNamespace

    result = {
        "format": "npa_restricted_payload_scan_v2",
        "scan_complete": True,
        "history_only": False,
        "digest": DIGEST,
        "entries_scanned": 10,
        "verdict": "clean",
        "payload_hits": [],
        "history_hits": [],
        "weight_shaped_paths": [],
    }
    result[field] = value

    def run(args, **kwargs):
        from pathlib import Path

        assert args[-3] == REPOSITORY + "@" + DIGEST
        Path(args[-1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(publish.subprocess, "run", run)
    monkeypatch.setattr(publish.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="NCore"):
        publish._scan_ncore_payload_exact_digest(REPOSITORY + "@" + DIGEST)


@pytest.mark.parametrize(
    "damage",
    [
        "extra-platform",
        "unbound",
        "subject",
        "empty-sbom",
        "wrong-source",
        "root",
        "revision",
        "duplicate-predicate",
        "missing-provenance",
    ],
)
def test_live_artifact_gate_refuses_drift(registry, damage):
    index, statements, layers, config, calls = registry
    if damage == "extra-platform":
        index["manifests"].append(
            {"digest": CONFIG, "platform": {"architecture": "arm64", "os": "linux"}}
        )
    elif damage == "unbound":
        index["manifests"][1]["annotations"]["vnd.docker.reference.digest"] = CONFIG
    elif damage == "subject":
        statements[0]["subject"][0]["digest"]["sha256"] = CONFIG[7:]
    elif damage == "empty-sbom":
        statements[0]["predicate"]["packages"] = []
    elif damage == "wrong-source":
        statements[1]["predicate"]["invocation"]["parameters"]["args"][
            "build-arg:SOURCE_SHA"
        ] = "f" * 40
    elif damage == "root":
        config["config"]["User"] = "0:1000"
    elif damage == "revision":
        config["config"]["Labels"]["org.opencontainers.image.revision"] = "f" * 40
    elif damage == "duplicate-predicate":
        layers.append(copy.deepcopy(layers[0]))
    else:
        layers.pop()
    assert not publish.verify_ncore_publication_source(item())[0]
    assert not calls


def test_preflight_calls_ncore_gate_after_quarantine_is_lifted(monkeypatch):
    monkeypatch.setattr(images, "PUBLICATION_QUARANTINE_TOOLS", frozenset())
    monkeypatch.setattr(publish, "_crane_manifest_readable", lambda ref: (True, "ok"))
    monkeypatch.setattr(
        publish, "verify_bootstrap_publication_source", lambda item: (True, "ok")
    )
    monkeypatch.setattr(
        publish,
        "verify_ncore_publication_source",
        lambda item: (False, "missing full capture"),
    )
    assert publish.preflight_sources([item()]) == [
        (item(), "NCORE GATE — missing full capture")
    ]
