"""Static LIBERO neutral-bootstrap and hard-gate contract tests."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / "workflows" / "testing" / "byof-libero.yaml"
READINESS_PATH = ROOT / "workflows" / "testing" / "byof-libero.readiness.json"
PROFILE_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-libero-b200-gpu.yaml"
)
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "libero"
RUNTIME_MANIFEST_PATH = IMAGE_ROOT / "runtime-manifest.json"
SMOKE_PATH = IMAGE_ROOT / "libero_smoke.py"
IMAGE_MANIFEST_PATH = (
    ROOT / "npa" / "src" / "npa" / "deploy" / "libero_image_manifest.json"
)

SOURCE_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DATASET_REF = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DATASET_SHA256 = "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"
LANGUAGE_MODEL_REF = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
LANGUAGE_MODEL_MODEL_SHA256 = (
    "d6992b8cd27d7a132eafce6a8210272329a371b1c762d453588795dd3835593e"
)
LANGUAGE_MODEL_SOURCE_SHA256 = (
    "d1df48c6984a2938d60eebf70ba1c61cd2ea512e859fa0ed11abfc550beee3f1"
)
CAPABILITY = "libero_spatial_bc_rnn_train_reload_heldout"
PAYLOAD_SERVICE_ACCOUNT = "npa-byof-libero-payload"


def _workflow() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _config() -> dict[str, object]:
    config = _workflow()["config"]
    assert isinstance(config, dict)
    return config


def _manifest() -> dict[str, object]:
    payload = json.loads(RUNTIME_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _smoke_source() -> str:
    source = SMOKE_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    return source


def test_libero_workflow_uses_only_quarantined_prebuilt_managed_path() -> None:
    workflow = _workflow()
    config = _config()

    assert config["repo_url"] == "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    assert config["repo_ref"] == SOURCE_REF
    assert config["base_profile"] == "prebuilt"
    assert config["base_image"] == "tool://libero"
    assert config["build_command"] == ""
    assert config["source_prune_path"] == ""
    assert config["resource_profile_yaml"] == "byof-solution-smoke-libero-b200-gpu"
    assert config["capability_name"] == CAPABILITY
    assert config["smoke_artifact_name"] == "libero-smoke.json"
    assert config["wait_timeout"] == -1
    assert config["libero_acceptance_candidate_image"] == ""
    assert config["libero_runtime_use_decision_file"] == ""
    assert config["libero_runtime_use_decision_sha256"] == ""
    assert config["libero_build_metadata_sha256"] == ""
    resources = workflow["resources"]
    assert isinstance(resources, dict)
    assert resources["gpu"]["image"] == "{{config.base_image}}"
    assert resources["gpu"]["accelerators"] == "B200:1"
    state = workflow["states"]["byof-run"]
    assert state["toolRef"] == "workbench.byof.repo"
    assert state["terminal"] is True


def test_runtime_manifest_closes_source_data_task_model_and_runtime_identity() -> None:
    manifest = _manifest()
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    task = manifest["task"]
    model = manifest["language_model"]
    artifacts = manifest["runtime_artifacts"]

    assert manifest["schema"] == "npa.libero.runtime-manifest.v1"
    assert source["revision"] == SOURCE_REF
    assert source["license"] == "MIT"
    assert source["forbidden_paths"] == ["libero/libero/assets", ".git"]
    assert "libero/libero/assets" not in source["sparse_paths"]
    assert demonstration["revision"] == DATASET_REF
    assert demonstration["license"] == "CC-BY-4.0"
    assert demonstration["sha256"] == DATASET_SHA256
    assert demonstration["size_bytes"] == 508_779_600
    assert task["suite"] == "libero_spatial"
    assert task["bddl"]["sha256"] == (
        "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b"
    )
    assert task["initial_states"]["sha256"] == (
        "c3a6a01fdc53ae1914fe24c8935088d723baee8f6ee3cd5f8d68e86aea3e2f1c"
    )
    assert task["embedding_source"]["sha256"] == LANGUAGE_MODEL_SOURCE_SHA256
    assert model["revision"] == LANGUAGE_MODEL_REF
    assert model["license"] == "Apache-2.0"
    assert {item["filename"]: item["sha256"] for item in model["files"]}[
        "pytorch_model.bin"
    ] == LANGUAGE_MODEL_MODEL_SHA256
    assert manifest["runtime_artifact_count"] == 135
    assert len(artifacts) == 135
    assert len({item["filename"] for item in artifacts}) == 135
    assert all(len(item["sha256"]) == 64 for item in artifacts)
    assert all(item["url"].startswith("https://") for item in artifacts)
    assert all("license_expression" in item for item in artifacts)


def test_libero_smoke_uses_real_upstream_conditioned_training_and_heldout() -> None:
    smoke = _smoke_source()

    for contract in (
        "from libero.lifelong.algos.base import Sequential",
        "from libero.lifelong.datasets import SequenceVLDataset, get_dataset",
        "AutoTokenizer.from_pretrained",
        "AutoModel.from_pretrained",
        "pooler_output",
        "algorithm.observe(batch)",
        "TRAIN_STEPS = 8",
        "torch_save_model(algorithm.policy",
        "torch_load_model(",
        "load_state_dict(state_dict, strict=True)",
        "reloaded.policy.compute_loss(batch)",
        "reloaded.policy.forward(batch)",
        "torch.isfinite(actions).all()",
        "set(train_demo_ids) & set(heldout_demo_ids)",
        '"strategy": "deterministic_trajectory_disjoint_sha256_rank"',
        "libero_upstream_bert_task_conditioning",
        "libero_trajectory_disjoint_heldout_split",
    ):
        assert contract in smoke
    assert smoke.count("revision=LANGUAGE_MODEL_REVISION") == 2
    assert "urllib.request" not in smoke
    assert "embedding_bytes" not in smoke
    assert "np.resize" not in smoke
    assert "deterministic_single_task_768d_sha256" not in smoke
    assert '"rendering_invoked": False' in smoke
    assert ".render(" not in smoke
    assert "offscreen" not in smoke.lower()


def test_libero_profile_binds_payload_identity_runtime_decision_and_headless_gpu() -> None:
    documents = list(yaml.safe_load_all(PROFILE_PATH.read_text(encoding="utf-8")))
    assert len(documents) == 2
    task = documents[1]
    resources = task["resources"]
    assert resources["accelerators"] == "B200:1"
    pod_spec = resources["kubernetes"]["pod_config"]["spec"]
    assert pod_spec["serviceAccountName"] == PAYLOAD_SERVICE_ACCOUNT
    assert pod_spec["serviceAccountName"] not in {"default", "skypilot-service-account"}
    assert task["envs"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert "NVIDIA_VISIBLE_DEVICES" not in task["envs"]
    profile = PROFILE_PATH.read_text(encoding="utf-8")
    for contract in (
        "NPA_LIBERO_RUNTIME_USE_DECISION_B64",
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256",
        "NPA_LIBERO_EXPECTED_BUILD_METADATA_SHA256",
        "/opt/npa/libero/runtime-bootstrap.py ensure",
        "/opt/npa/libero/smoke.sh",
        "npa-byof-libero-payload",
    ):
        assert contract in profile
    assert '"rendering_invoked": False' in profile
    assert ".render(" not in profile


def test_libero_workload_requires_pod_identity_and_independent_build_lineage() -> None:
    smoke = _smoke_source()

    for contract in (
        "NPA_LIBERO_EXPECTED_BUILD_METADATA_SHA256",
        "PUBLIC_BASE_IMAGE_DIGEST",
        "PUBLIC_BASE_ROOTFS_MATERIAL_DIGEST",
        "observe_own_pod_image",
        "kubernetes_status_containerStatuses_imageID",
    ):
        assert contract in smoke
    assert 'service_account_name != "npa-byof-libero-payload"' in smoke
    assert "torch.cuda.device_count() != 1" in smoke
    assert '"B200" not in gpu_name.upper()' in smoke
    assert "compute_capability != (10, 0)" in smoke


def test_libero_image_manifest_remains_quarantined_and_unpublished() -> None:
    manifest = json.loads(IMAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["tool"] == "libero"
    assert manifest["version"] == "public-neutral-bootstrap-unbuilt"
    assert manifest["status"] == "quarantined_unbuilt_unvalidated"
    assert manifest["payload"] == "neutral_bootstrap_only"
    assert manifest["source_revision"] == ""
    assert manifest["oci_digest"] == ""
    assert manifest["gpu_validation"] is None
    assert manifest["anonymous_pull_verified"] is False
    assert manifest["runtime_payloads_baked"] is False
    assert manifest["catalog_release"] is False
    assert manifest["base_provenance"] == {
        "repository": "index.docker.io/library/python",
        "blob_sha256": (
            "0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b"
        ),
        "predicate_type": "https://slsa.dev/provenance/v0.2",
        "builder_id": "https://github.com/docker-library",
        "source_revision": "688a0b86bb44289df16a363e9f41d90514c1a5f9",
    }
    assert manifest["base_sbom"] == {
        "format": "https://spdx.dev/Document",
        "blob_sha256": (
            "b290dbd3087fc5d2cf4af106f1d253c417a26080b70bdcf440ff96314a12c2bb"
        ),
    }


def test_libero_readiness_hashes_bind_every_execution_input() -> None:
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    expected_paths = {
        "workflow": WORKFLOW_PATH,
        "resource_profile": PROFILE_PATH,
        "runtime_manifest": RUNTIME_MANIFEST_PATH,
        "runtime_bootstrap": IMAGE_ROOT / "runtime-bootstrap.py",
        "smoke": SMOKE_PATH,
        "image_manifest": IMAGE_MANIFEST_PATH,
    }

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["solution"] == "libero"
    assert readiness["status"] == "quarantined_unbuilt_unvalidated"
    hashes = readiness["input_sha256"]
    assert set(hashes) == set(expected_paths)
    for name, path in expected_paths.items():
        assert hashes[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert readiness["public_image_digest"] == ""
    assert readiness["anonymous_pull_verified"] is False
    assert readiness["live_validated"] is False
