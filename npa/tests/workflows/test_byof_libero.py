"""Static LIBERO neutral-bootstrap and hard-gate contract tests."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from npa.deploy.images import (
    LIBERO_PUBLICATION_ENFORCEMENT_PYTHON_ROOTS,
    LIBERO_PUBLICATION_ENFORCEMENT_LIBERO_TEST_ROOTS,
    LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER,
    LIBERO_REQUIRED_PUBLICATION_REFERRERS,
    libero_acceptance_signature_payload,
    libero_accepted_image_manifest,
    libero_build_input_bundle_sha256,
    libero_publication_enforcement_bundle_sha256,
    libero_publication_enforcement_paths,
    libero_publication_lineage_values,
    validate_libero_accepted_image_manifest,
)


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
PUBLICATION_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "publish-public-images.yml"

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
    assert "libero_runtime_use_decision_sha256" not in config
    assert "libero_build_metadata_sha256" not in config
    resources = workflow["resources"]
    assert isinstance(resources, dict)
    assert resources["gpu"]["image"] == "{{config.base_image}}"
    assert resources["gpu"]["accelerators"] == "B200:1"
    state = workflow["states"]["byof-run"]
    assert state["toolRef"] == "workbench.byof.repo"
    assert state["terminal"] is True


def test_publication_workflow_binds_anonymous_tag_to_pushed_digest() -> None:
    workflow = PUBLICATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    anonymous = workflow[workflow.index('anonymous_config="$(mktemp -d)"') :]

    resolve = anonymous.index(
        'anonymous_digest="$(DOCKER_CONFIG="$anonymous_config" crane digest "$anonymous_reference"'
    )
    readable = anonymous.index(
        'DOCKER_CONFIG="$anonymous_config" crane manifest "$anonymous_reference"'
    )
    compare = anonymous.index('test "$anonymous_digest" = "$DIGEST"')
    assert resolve < readable < compare


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
    assert "runtime_materialized_this_run" in smoke
    assert 'receipt.get("warm_reuse") is not False' in smoke
    assert 'receipt.get("manifest_sha256") != RUNTIME_MANIFEST_SHA256' in smoke
    assert 'receipt.get("decision_sha256")' in smoke
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
    assert "NPA_LIBERO_RUNTIME_USE_DECISION_B64" not in task["envs"]
    assert "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256" not in task["envs"]
    profile = PROFILE_PATH.read_text(encoding="utf-8")
    for contract in (
        "NPA_LIBERO_RUNTIME_USE_DECISION_B64",
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256",
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
        "/opt/npa/libero/runtime-bootstrap.py ensure",
        "/opt/npa/libero/runtime-bootstrap.py execute",
        "/opt/npa/libero/runtime-bootstrap.py execute-and-upload",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        "npa_runtime_bootstrap.json",
        "/opt/npa/libero/smoke.sh",
        "npa-byof-libero-payload",
    ):
        assert contract in profile
    bootstrap = (IMAGE_ROOT / "runtime-bootstrap.py").read_text(encoding="utf-8")
    assert '"rendering_invoked": False' in bootstrap
    assert "root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)" in bootstrap
    assert "dir_fd=root_fd" in bootstrap
    assert "stat.S_ISREG(info.st_mode)" in bootstrap
    assert "info.st_nlink != 1" in bootstrap
    assert "os.O_NOFOLLOW" in bootstrap
    assert "os.fstat(stream.fileno())" in bootstrap
    assert '"if-none-match": "*"' in bootstrap
    assert '"x-amz-checksum-mode": "ENABLED"' in bootstrap
    assert "headers.get(\"x-amz-checksum-sha256\") != checksum" in bootstrap
    assert '"schema": "npa.libero.s3-upload-readback.v1"' in bootstrap
    assert "set(os.listdir(root_fd)) != set(OUTPUT_SIZE_LIMITS)" in bootstrap
    assert "MAX_OUTPUT_BYTES" in bootstrap
    assert "STORAGE_SECRET_ENV_NAMES" in bootstrap
    assert profile.count("unset NPA_LIBERO_RUNTIME_USE_DECISION_B64") == 2
    assert (
        "/usr/local/bin/python /opt/npa/libero/runtime-bootstrap.py "
        "execute-and-upload"
    ) in profile
    assert "runtime-bootstrap.py upload" not in profile
    assert "execute-python" not in profile
    assert ".render(" not in profile


def test_libero_workload_requires_pod_identity_and_independent_build_lineage() -> None:
    smoke = _smoke_source()

    for contract in (
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
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
    assert manifest["runtime_artifact_review"] == {
        "status": "incomplete",
        "pending_size_artifacts": 135,
        "pending_license_artifacts": 65,
        "report_sha256": "",
    }
    assert manifest["acceptance"]["status"] == "not_accepted"
    assert manifest["acceptance"]["candidate_image"] == ""
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

    with pytest.raises(RuntimeError, match="accepted status"):
        libero_accepted_image_manifest()


def test_libero_acceptance_closes_candidate_publication_and_infrastructure(
    monkeypatch,
) -> None:
    manifest = json.loads(IMAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    digest = "sha256:" + "1" * 64
    manifest["runtime_artifact_review"] = {
        "status": "complete",
        "pending_size_artifacts": 0,
        "pending_license_artifacts": 0,
        "report_sha256": "2" * 64,
    }
    acceptance = manifest["acceptance"]
    acceptance.update(
        {
            "status": "accepted",
            "acceptance_id": "libero-qualification-acceptance-0001",
            "accepted_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "candidate_image": (
                "ghcr.io/nebius/nebius-physical-ai/npa-libero@" + digest
            ),
            "oci_digest": digest,
            "platform_manifest_digest": "sha256:" + "3" * 64,
            "config_digest": "sha256:" + "4" * 64,
            "canonical_build_metadata_sha256": "5" * 64,
            "attestation_manifest_digest": "sha256:" + "e" * 64,
            "attestation_config_digest": "sha256:" + "f" * 64,
            "attestation_layers": [
                {
                    "predicate_type": predicate_type,
                    "digest": "sha256:" + digit * 64,
                    "size_bytes": 1024,
                }
                for predicate_type, digit in zip(
                    LIBERO_REQUIRED_PUBLICATION_REFERRERS, ("1", "2"), strict=True
                )
            ],
            "package_version_digests": sorted(
                [digest, "sha256:" + "3" * 64, "sha256:" + "e" * 64]
            ),
            "complete_image_inventory_sha256": "6" * 64,
            "base_provenance_sha256": "7" * 64,
            "publication_bundle_sha256": "8" * 64,
            "development_sha": "d" * 40,
            "upstream_source_revision": SOURCE_REF,
            "build_input_bundle_sha256": libero_build_input_bundle_sha256(
                ROOT, development_sha="d" * 40
            ),
            "publication_enforcement_bundle_sha256": (
                libero_publication_enforcement_bundle_sha256(ROOT)
            ),
            "package_writer_repository": "nebius/nebius-physical-ai",
            "runtime_use_decision_sha256": "9" * 64,
            "infrastructure_bundle_sha256": "a" * 64,
        }
    )
    acceptance["infrastructure"] = {
        "run_id": "libero-qualification-run-0001",
        **{
            key: "b" * 64
            for key in acceptance["infrastructure"]
            if key != "run_id"
        },
    }
    acceptance["infrastructure_bundle_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": "npa.libero.infrastructure-bundle.v1",
                **acceptance["infrastructure"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    acceptance["publication_bundle_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": "npa.libero.publication-lineage-bundle.v2",
                "candidate_image": acceptance["candidate_image"],
                "oci_digest": acceptance["oci_digest"],
                "platform_manifest_digest": acceptance["platform_manifest_digest"],
                "config_digest": acceptance["config_digest"],
                "canonical_build_metadata_sha256": acceptance[
                    "canonical_build_metadata_sha256"
                ],
                "attestation_manifest_digest": acceptance[
                    "attestation_manifest_digest"
                ],
                "attestation_config_digest": acceptance[
                    "attestation_config_digest"
                ],
                "attestation_layers": acceptance["attestation_layers"],
                "package_version_digests": acceptance[
                    "package_version_digests"
                ],
                "complete_image_inventory_sha256": acceptance[
                    "complete_image_inventory_sha256"
                ],
                "base_provenance_sha256": acceptance["base_provenance_sha256"],
                "development_sha": acceptance["development_sha"],
                "upstream_source_revision": acceptance[
                    "upstream_source_revision"
                ],
                "build_input_bundle_sha256": acceptance[
                    "build_input_bundle_sha256"
                ],
                "publication_enforcement_bundle_sha256": acceptance[
                    "publication_enforcement_bundle_sha256"
                ],
                "package_writer_repository": acceptance[
                    "package_writer_repository"
                ],
                "infrastructure_bundle_sha256": acceptance[
                    "infrastructure_bundle_sha256"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        "NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64",
        base64.b64encode(public_key).decode("ascii"),
    )
    acceptance["manager_signature"] = {
        "algorithm": "ed25519",
        "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
        "signature_b64": base64.b64encode(
            private_key.sign(libero_acceptance_signature_payload(manifest))
        ).decode("ascii"),
    }

    assert validate_libero_accepted_image_manifest(manifest) == acceptance
    lineage = libero_publication_lineage_values(
        acceptance, ROOT, development_sha="d" * 40
    )
    assert lineage["candidate_image"] == acceptance["candidate_image"]
    assert lineage["platform_manifest_digest"] == acceptance[
        "platform_manifest_digest"
    ]

    signature = acceptance["manager_signature"]["signature_b64"]
    acceptance["manager_signature"]["signature_b64"] = base64.b64encode(
        b"\0" * 64
    ).decode("ascii")
    with pytest.raises(RuntimeError, match="manager signature is invalid"):
        validate_libero_accepted_image_manifest(manifest)
    acceptance["manager_signature"]["signature_b64"] = signature

    untrusted_key = Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        "NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64",
        base64.b64encode(untrusted_key).decode("ascii"),
    )
    with pytest.raises(RuntimeError, match="manager trust root differs"):
        validate_libero_accepted_image_manifest(manifest)
    monkeypatch.setenv(
        "NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64",
        base64.b64encode(public_key).decode("ascii"),
    )

    acceptance["candidate_image"] = (
        "ghcr.io/attacker/example/npa-libero@" + digest
    )
    with pytest.raises(RuntimeError, match="official candidate image"):
        validate_libero_accepted_image_manifest(manifest)


def test_publication_enforcement_bundle_detects_descendant_policy_drift(
    tmp_path,
) -> None:
    enforcement_paths = libero_publication_enforcement_paths(ROOT)
    for relative in enforcement_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    accepted = libero_publication_enforcement_bundle_sha256(tmp_path)
    assert set(LIBERO_PUBLICATION_ENFORCEMENT_PYTHON_ROOTS) == {
        "npa/scripts",
        "npa/src/npa",
        "npa/tests/e2e",
    }
    assert set(LIBERO_PUBLICATION_ENFORCEMENT_LIBERO_TEST_ROOTS) == {
        "npa/tests",
    }
    assert {
        "npa/src/npa/execution_preflight.py",
        "npa/src/npa/orchestration/skypilot/cleanup.py",
        "npa/src/npa/orchestration/skypilot/launch_transaction.py",
        "npa/src/npa/orchestration/skypilot/signal_teardown.py",
        "npa/src/npa/orchestration/skypilot/workflow.py",
        "npa/src/npa/orchestration/skypilot/workflow_state.py",
        "npa/src/npa/teardown_receipts.py",
        "npa/src/npa/cleanup_identity.py",
        "npa/src/npa/config_schema.py",
        "npa/src/npa/progress.py",
        "npa/src/npa/verification.py",
        "npa/src/npa/workflows/byof/live.py",
        "npa/tests/e2e/agent_live_helpers.py",
        "npa/tests/e2e/conftest.py",
        "npa/tests/e2e/npa_workflow_live_helpers.py",
        "npa/tests/e2e/test_byof_onboarding_live_e2e.py",
        "npa/tests/guardrails/test_byof_profiles.py",
        "npa/tests/guardrails/test_e2e_gate_reachability.py",
    } <= set(enforcement_paths)
    representatives = {
        ".github/workflows/publish-public-images.yml",
        "npa/scripts/run_byof_container_verify.py",
        "npa/src/npa/cleanup_identity.py",
        "npa/tests/e2e/conftest.py",
        "npa/tests/guardrails/test_byof_profiles.py",
        "npa/tests/guardrails/test_e2e_gate_reachability.py",
    }
    assert representatives <= set(enforcement_paths)
    for relative in representatives:
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n# policy drift\n")
        assert libero_publication_enforcement_bundle_sha256(tmp_path) != accepted
        path.write_bytes(original)

    future = tmp_path / "npa/tests/guardrails/test_future_policy.py"
    future.write_bytes(
        LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER
        + b"\ndef test_future_policy():\n    pass\n"
    )
    assert future.relative_to(tmp_path).as_posix() in (
        libero_publication_enforcement_paths(tmp_path)
    )
    assert libero_publication_enforcement_bundle_sha256(tmp_path) != accepted
    future.unlink()

    incidental = tmp_path / "npa/tests/guardrails/test_unmarked_policy.py"
    incidental.write_text(
        "def test_unmarked_policy():\n    assert 'libero'\n", encoding="utf-8"
    )
    assert incidental.relative_to(tmp_path).as_posix() not in (
        libero_publication_enforcement_paths(tmp_path)
    )
    incidental.unlink()

    assert libero_publication_enforcement_bundle_sha256(tmp_path) == accepted


def test_libero_readiness_hashes_bind_every_execution_input() -> None:
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    expected_paths = {
        "workflow": WORKFLOW_PATH,
        "resource_profile": PROFILE_PATH,
        "runtime_manifest": RUNTIME_MANIFEST_PATH,
        "runtime_requirements": IMAGE_ROOT / "runtime-requirements.txt",
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
