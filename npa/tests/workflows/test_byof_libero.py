# npa: publication-enforcement=libero
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
    LiberoCustomerAuthorizationDenied,
    LIBERO_PUBLICATION_ENFORCEMENT_PYTHON_ROOTS,
    LIBERO_PUBLICATION_ENFORCEMENT_LIBERO_TEST_ROOTS,
    LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER,
    LIBERO_REQUIRED_PUBLICATION_REFERRERS,
    libero_customer_acceptance_notification,
    libero_customer_authorization_signature_payload,
    libero_authenticated_caller_signature_payload,
    libero_output_storage_authorization_signature_payload,
    libero_qualified_image_manifest,
    libero_build_input_bundle_sha256,
    libero_publication_enforcement_bundle_sha256,
    libero_publication_enforcement_paths,
    libero_publication_lineage_values,
    validate_libero_customer_runtime_authorization,
    validate_libero_authenticated_caller_assertion,
    validate_libero_output_storage_authorization,
    validate_libero_qualified_image_manifest,
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
    assert config["libero_qualified_candidate_image"] == ""
    assert config["libero_customer_runtime_authorization_file"] == ""
    assert config["libero_authenticated_caller_identity_file"] == ""
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
    assert all(item["license_expression"].strip() for item in artifacts)
    assert all(item["size_bytes"] > 0 for item in artifacts)
    assert sum(item["size_bytes"] for item in artifacts) == 3_277_640_175


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
    assert 'receipt.get("governing_terms_fetched_this_invocation") is not True' in smoke
    assert 'receipt.get("manifest_sha256") != RUNTIME_MANIFEST_SHA256' in smoke
    assert 'receipt.get("customer_authorization_sha256")' in smoke
    assert 'receipt.get("customer_identity_sha256")' in smoke
    assert ".render(" not in smoke
    assert "offscreen" not in smoke.lower()


def test_libero_profile_binds_payload_identity_customer_authorization_and_headless_gpu() -> (
    None
):
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
    assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64" not in task["envs"]
    assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256" not in task["envs"]
    profile = PROFILE_PATH.read_text(encoding="utf-8")
    for contract in (
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256",
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
        "/opt/npa/libero/runtime-bootstrap.py ensure",
        "/opt/npa/libero/runtime-bootstrap.py execute",
        "/opt/npa/libero/runtime-bootstrap.py execute-and-upload",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        "/workspace/.cache/npa/libero/run-receipts",
        "/opt/npa/libero/smoke.sh",
        "npa-byof-libero-payload",
    ):
        assert contract in profile
    bootstrap = (IMAGE_ROOT / "runtime-bootstrap.py").read_text(encoding="utf-8")
    assert '"rendering_invoked": False' in bootstrap
    assert (
        "output_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)"
        in bootstrap
    )
    assert "upload_outputs(smoke_exit_code, root_fd=output_fd)" in bootstrap
    assert "output staging directory changed during execution" in bootstrap
    assert "dir_fd=root_fd" in bootstrap
    assert "stat.S_ISREG(info.st_mode)" in bootstrap
    assert "info.st_nlink != 1" in bootstrap
    assert "os.O_NOFOLLOW" in bootstrap
    assert "os.fstat(stream.fileno())" in bootstrap
    assert '"if-none-match": "*"' in bootstrap
    assert '"x-amz-checksum-mode": "ENABLED"' in bootstrap
    assert 'headers.get("x-amz-checksum-sha256") != checksum' in bootstrap
    assert 'OUTPUT_RECEIPT_SCHEMA = "npa.libero.s3-upload-readback.v2"' in bootstrap
    assert 'OUTPUT_RECEIPT_NAME = "npa_upload_receipt.json"' in bootstrap
    assert "current_authorization = _storage_authorization" in bootstrap
    assert "set(os.listdir(root_fd)) != set(OUTPUT_SIZE_LIMITS)" in bootstrap
    assert "MAX_OUTPUT_BYTES" in bootstrap
    assert "STORAGE_SECRET_ENV_NAMES" in bootstrap
    assert "_execution_uid_processes()" in bootstrap
    assert "os.fchmod(output_fd, 0o700)" in bootstrap
    assert "snapshots.append((name, payload, digest))" in bootstrap
    assert '"PATH": "/usr/bin:/bin"' in bootstrap
    assert '["git"' not in bootstrap
    assert '"/usr/bin/git"' in bootstrap
    assert "_discard_new_cache_entry" in bootstrap
    workflow_source = (
        ROOT / "npa/src/npa/orchestration/skypilot/workflow.py"
    ).read_text(encoding="utf-8")
    assert "if libero_submission:" in workflow_source
    assert "generated_config_path.chmod(0o400)" in workflow_source
    assert "prepared_yaml.chmod(0o400)" in workflow_source
    assert profile.count("unset NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64") == 2
    assert (
        "/usr/local/bin/python /opt/npa/libero/runtime-bootstrap.py execute-and-upload"
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
        "status": "complete",
        "pending_size_artifacts": 0,
        "pending_license_artifacts": 0,
        "report_sha256": (
            "8caa49fc844d8a78ac44ee632ea2cbb393f009af424daa0392d13d7763abc9a8"
        ),
    }
    assert manifest["qualification"]["status"] == "not_qualified"
    assert manifest["qualification"]["candidate_image"] == ""
    assert manifest["qualification"]["customer_authorization_public_key_sha256"] == ""
    assert (
        manifest["qualification"]["output_storage_authorization_public_key_sha256"]
        == ""
    )
    assert manifest["customer_runtime_authorization_required"] is True
    notification = libero_customer_acceptance_notification(manifest)
    assert notification["status"] == "needs_customer_acceptance"
    assert notification["credentials"]["establish_terms_acceptance"] is False
    assert len(notification["terms"]) == 7
    assert all(term["name"] and term["official_url"] for term in notification["terms"])
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

    with pytest.raises(RuntimeError, match="qualified status"):
        libero_qualified_image_manifest()


def test_libero_qualification_and_customer_authorization_are_separate(
    monkeypatch, tmp_path
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
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_file = tmp_path / "customer-authorization-public-key.b64"
    key_file.write_bytes(base64.b64encode(public_key))
    key_file.chmod(0o600)
    customer_private_key = Ed25519PrivateKey.generate()
    customer_public_key = customer_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    customer_key_file = tmp_path / "direct-customer-public-key.b64"
    customer_key_file.write_bytes(base64.b64encode(customer_public_key))
    customer_key_file.chmod(0o600)
    storage_private_key = Ed25519PrivateKey.generate()
    storage_public_key = storage_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    storage_key_file = tmp_path / "output-storage-authorization-public-key.b64"
    storage_key_file.write_bytes(base64.b64encode(storage_public_key))
    storage_key_file.chmod(0o600)
    qualification = manifest["qualification"]
    qualification.update(
        {
            "status": "qualified",
            "qualification_id": "libero-image-qualification-0001",
            "qualified_at": now.isoformat(),
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
            "customer_authorization_public_key_sha256": hashlib.sha256(
                public_key
            ).hexdigest(),
            "output_storage_authorization_public_key_sha256": hashlib.sha256(
                storage_public_key
            ).hexdigest(),
        }
    )
    qualification["publication_bundle_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": "npa.libero.publication-lineage-bundle.v3",
                "candidate_image": qualification["candidate_image"],
                "oci_digest": qualification["oci_digest"],
                "platform_manifest_digest": qualification["platform_manifest_digest"],
                "config_digest": qualification["config_digest"],
                "canonical_build_metadata_sha256": qualification[
                    "canonical_build_metadata_sha256"
                ],
                "attestation_manifest_digest": qualification[
                    "attestation_manifest_digest"
                ],
                "attestation_config_digest": qualification["attestation_config_digest"],
                "attestation_layers": qualification["attestation_layers"],
                "package_version_digests": qualification["package_version_digests"],
                "complete_image_inventory_sha256": qualification[
                    "complete_image_inventory_sha256"
                ],
                "base_provenance_sha256": qualification["base_provenance_sha256"],
                "development_sha": qualification["development_sha"],
                "upstream_source_revision": qualification["upstream_source_revision"],
                "build_input_bundle_sha256": qualification["build_input_bundle_sha256"],
                "publication_enforcement_bundle_sha256": qualification[
                    "publication_enforcement_bundle_sha256"
                ],
                "package_writer_repository": qualification["package_writer_repository"],
                "customer_authorization_public_key_sha256": qualification[
                    "customer_authorization_public_key_sha256"
                ],
                "output_storage_authorization_public_key_sha256": qualification[
                    "output_storage_authorization_public_key_sha256"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert validate_libero_qualified_image_manifest(manifest) == qualification
    lineage = libero_publication_lineage_values(
        qualification, ROOT, development_sha="d" * 40
    )
    assert lineage["candidate_image"] == qualification["candidate_image"]
    assert (
        lineage["platform_manifest_digest"] == qualification["platform_manifest_digest"]
    )

    run_id = "libero-customer-run-0001"
    authorization = {
        "schema": "npa.libero.customer-runtime-authorization.v2",
        "solution": "libero",
        "status": "authorized",
        "authorization_id": "libero-customer-authorization-0001",
        "customer_identity_sha256": "9" * 64,
        "run_id": run_id,
        "candidate_image": qualification["candidate_image"],
        "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
        "workflow_profile_sha256": "a" * 64,
        "upstream_source_revision": qualification["upstream_source_revision"],
        "terms": [
            {"id": term["id"], "version": term["version"]}
            for term in manifest["customer_acceptance"]["terms"]
        ],
        "issuer": "customer",
        "evidence_type": "customer-controlled-signature",
        "customer_signer_public_key_b64": base64.b64encode(customer_public_key).decode(
            "ascii"
        ),
        "acknowledged_at": now.isoformat(),
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "nonce": "customer-authorization-nonce-000001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": hashlib.sha256(customer_public_key).hexdigest(),
            "signature_b64": "",
        },
    }
    authorization["signature"]["signature_b64"] = base64.b64encode(
        customer_private_key.sign(
            libero_customer_authorization_signature_payload(authorization)
        )
    ).decode("ascii")
    authorization_bytes = json.dumps(authorization, sort_keys=True).encode()
    validated, observed_sha256 = validate_libero_customer_runtime_authorization(
        authorization_bytes,
        image_manifest=manifest,
        run_id=run_id,
        customer_identity_sha256="9" * 64,
        customer_signer_public_key_sha256=hashlib.sha256(
            customer_public_key
        ).hexdigest(),
        executable_profile_sha256="a" * 64,
        public_key_file=str(customer_key_file),
        now=now,
    )
    assert validated == authorization
    assert observed_sha256 == hashlib.sha256(authorization_bytes).hexdigest()

    caller_assertion = {
        "schema": "npa.libero.authenticated-caller.v1",
        "issuer": "npa-authenticated-caller-control-plane",
        "session_id": "libero-caller-session-0001",
        "customer_identity_sha256": authorization["customer_identity_sha256"],
        "customer_signer_public_key_sha256": hashlib.sha256(
            customer_public_key
        ).hexdigest(),
        "run_id": run_id,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "nonce": "authenticated-caller-nonce-00000001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
            "signature_b64": "",
        },
    }
    caller_assertion["signature"]["signature_b64"] = base64.b64encode(
        private_key.sign(
            libero_authenticated_caller_signature_payload(caller_assertion)
        )
    ).decode("ascii")
    caller_bytes = json.dumps(caller_assertion, sort_keys=True).encode()
    validated_caller, caller_sha256 = validate_libero_authenticated_caller_assertion(
        caller_bytes,
        run_id=run_id,
        public_key_file=str(key_file),
        now=now,
    )
    assert (
        validated_caller["customer_identity_sha256"]
        == authorization["customer_identity_sha256"]
    )
    assert caller_sha256 == hashlib.sha256(caller_bytes).hexdigest()

    output_prefix = f"s3://customer-output/libero/{run_id}/"
    endpoint = "https://storage.example.invalid"
    access_key = "temporary-access-key"
    secret_key = "temporary-secret-key"
    session_token = "temporary-session-token"
    policy_sha256 = "a" * 64
    storage_authorization = {
        "schema": "npa.libero.output-storage-authorization.v3",
        "issuer": "npa-output-storage-control-plane",
        "capability_id": "libero-output-capability-0001",
        "customer_identity_sha256": authorization["customer_identity_sha256"],
        "run_id": run_id,
        "candidate_image": authorization["candidate_image"],
        "runtime_manifest_sha256": authorization["runtime_manifest_sha256"],
        "output_prefix": output_prefix,
        "endpoint_url": endpoint,
        "access_key_id_sha256": hashlib.sha256(access_key.encode()).hexdigest(),
        "secret_access_key_sha256": hashlib.sha256(secret_key.encode()).hexdigest(),
        "session_token_sha256": hashlib.sha256(session_token.encode()).hexdigest(),
        "policy_sha256": policy_sha256,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=30)).isoformat(),
        "nonce": "output-storage-capability-nonce-0001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": hashlib.sha256(storage_public_key).hexdigest(),
            "signature_b64": "",
        },
    }
    storage_authorization["signature"]["signature_b64"] = base64.b64encode(
        storage_private_key.sign(
            libero_output_storage_authorization_signature_payload(storage_authorization)
        )
    ).decode("ascii")
    storage_bytes = json.dumps(storage_authorization, sort_keys=True).encode()
    validated_storage, storage_sha256 = validate_libero_output_storage_authorization(
        storage_bytes,
        image_manifest=manifest,
        customer_authorization=authorization,
        run_id=run_id,
        output_prefix=output_prefix,
        endpoint_url=endpoint,
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token,
        expected_policy_sha256=policy_sha256,
        customer_public_key_file=str(key_file),
        storage_public_key_file=str(storage_key_file),
        now=now,
    )
    assert validated_storage == storage_authorization
    assert storage_sha256 == hashlib.sha256(storage_bytes).hexdigest()

    forged_storage = json.loads(storage_bytes)
    forged_storage["signature"]["signature_b64"] = base64.b64encode(b"\0" * 64).decode(
        "ascii"
    )
    with pytest.raises(RuntimeError, match="signature is invalid"):
        validate_libero_output_storage_authorization(
            json.dumps(forged_storage, sort_keys=True).encode(),
            image_manifest=manifest,
            customer_authorization=authorization,
            run_id=run_id,
            output_prefix=output_prefix,
            endpoint_url=endpoint,
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=session_token,
            expected_policy_sha256=policy_sha256,
            customer_public_key_file=str(key_file),
            storage_public_key_file=str(storage_key_file),
            now=now,
        )

    denied = json.loads(authorization_bytes)
    denied["status"] = "denied"
    denied["signature"]["signature_b64"] = base64.b64encode(
        customer_private_key.sign(
            libero_customer_authorization_signature_payload(denied)
        )
    ).decode("ascii")
    with pytest.raises(
        LiberoCustomerAuthorizationDenied, match="declined the required runtime terms"
    ):
        validate_libero_customer_runtime_authorization(
            json.dumps(denied, sort_keys=True).encode(),
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            public_key_file=str(customer_key_file),
            now=now,
        )

    forged = json.loads(authorization_bytes)
    forged["signature"]["signature_b64"] = base64.b64encode(b"\0" * 64).decode()
    with pytest.raises(RuntimeError, match="signature is invalid"):
        validate_libero_customer_runtime_authorization(
            json.dumps(forged, sort_keys=True).encode(),
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            public_key_file=str(customer_key_file),
            now=now,
        )

    other_public_key = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    wrong_key_file = tmp_path / "wrong-customer-authorization-public-key.b64"
    wrong_key_file.write_bytes(base64.b64encode(other_public_key))
    wrong_key_file.chmod(0o600)
    with pytest.raises(RuntimeError, match="transported customer signer differs"):
        validate_libero_customer_runtime_authorization(
            authorization_bytes,
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            public_key_file=str(wrong_key_file),
            now=now,
        )

    attacker_private_key = Ed25519PrivateKey.generate()
    attacker_public_key = attacker_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    attacker_key_file = tmp_path / "attacker-customer-authorization-public-key.b64"
    attacker_key_file.write_bytes(base64.b64encode(attacker_public_key))
    attacker_key_file.chmod(0o600)
    attacker_authorization = json.loads(authorization_bytes)
    attacker_authorization["customer_signer_public_key_b64"] = base64.b64encode(
        attacker_public_key
    ).decode("ascii")
    attacker_authorization["signature"]["public_key_sha256"] = hashlib.sha256(
        attacker_public_key
    ).hexdigest()
    attacker_authorization["signature"]["signature_b64"] = base64.b64encode(
        attacker_private_key.sign(
            libero_customer_authorization_signature_payload(attacker_authorization)
        )
    ).decode("ascii")
    with pytest.raises(RuntimeError, match="customer signer identity differs"):
        validate_libero_customer_runtime_authorization(
            json.dumps(attacker_authorization, sort_keys=True).encode(),
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            now=now,
        )

    customer_key_file.chmod(0o640)
    with pytest.raises(RuntimeError, match="trust-root file is mutable"):
        validate_libero_customer_runtime_authorization(
            authorization_bytes,
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            public_key_file=str(customer_key_file),
            now=now,
        )
    customer_key_file.chmod(0o600)

    linked_key_file = tmp_path / "linked-customer-authorization-public-key.b64"
    linked_key_file.symlink_to(customer_key_file)
    with pytest.raises(RuntimeError, match="trust-root file is unavailable"):
        validate_libero_customer_runtime_authorization(
            authorization_bytes,
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            public_key_file=str(linked_key_file),
            now=now,
        )

    monkeypatch.delenv(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_FILE", raising=False
    )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_B64",
        base64.b64encode(public_key).decode(),
    )
    validated_without_transport, _ = validate_libero_customer_runtime_authorization(
        authorization_bytes,
        image_manifest=manifest,
        run_id=run_id,
        customer_identity_sha256="9" * 64,
        customer_signer_public_key_sha256=hashlib.sha256(
            customer_public_key
        ).hexdigest(),
        executable_profile_sha256="a" * 64,
        now=now,
    )
    assert validated_without_transport == authorization

    with pytest.raises(RuntimeError, match="exact customer/run contract"):
        validate_libero_customer_runtime_authorization(
            json.dumps({**authorization, "workflow_profile_sha256": "0" * 64}).encode(),
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            now=now,
        )

    authorization["run_id"] = "libero-other-customer-run"
    with pytest.raises(RuntimeError, match="exact customer/run contract"):
        validate_libero_customer_runtime_authorization(
            json.dumps(authorization).encode(),
            image_manifest=manifest,
            run_id=run_id,
            customer_identity_sha256="9" * 64,
            customer_signer_public_key_sha256=hashlib.sha256(
                customer_public_key
            ).hexdigest(),
            executable_profile_sha256="a" * 64,
            now=now,
        )

    qualification["candidate_image"] = "ghcr.io/attacker/example/npa-libero@" + digest
    with pytest.raises(RuntimeError, match="official candidate image"):
        validate_libero_qualified_image_manifest(manifest)


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
        ".github/scripts/publish_selected_public_image.py",
        ".github/workflows/publish-public-images.yml",
        "workflows/testing/byof-libero.yaml",
        "workflows/testing/byof-libero.readiness.json",
        "npa/docker/workbench/libero/Dockerfile",
        "npa/docker/workbench/libero/build.sh",
        "npa/docker/workbench/libero/runtime-bootstrap.py",
        "npa/docker/workbench/libero/runtime-manifest.json",
        "npa/docker/workbench/libero/smoke.sh",
        "npa/docker/workbench/packaging-contract.yaml",
        "npa/scripts/run_byof_container_verify.py",
        "npa/src/npa/cleanup_identity.py",
        "npa/src/npa/deploy/libero_image_manifest.json",
        "npa/src/npa/deploy/public_release_manifest.json",
        "npa/src/npa/smoke/golden_evals.yaml",
        "npa/src/npa/workflows/byof/profiles/byof-solution-smoke-libero-b200-gpu.yaml",
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

    critical = tmp_path / "npa/tests/guardrails/test_unmarked_libero_critical.py"
    critical.write_text(
        "def test_unmarked_libero_critical():\n"
        "    assert NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="publication-enforcement marker"):
        libero_publication_enforcement_paths(tmp_path)
    critical.write_bytes(
        LIBERO_PUBLICATION_ENFORCEMENT_TEST_MARKER + b"\n" + critical.read_bytes()
    )
    assert critical.relative_to(tmp_path).as_posix() in (
        libero_publication_enforcement_paths(tmp_path)
    )
    critical.unlink()

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
