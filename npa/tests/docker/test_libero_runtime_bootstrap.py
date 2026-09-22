# npa: publication-enforcement=libero
"""Fail-closed and atomic tests for the LIBERO runtime materializer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa" / "docker" / "workbench" / "libero" / "runtime-bootstrap.py"


@pytest.fixture(autouse=True)
def _restore_process_environment() -> object:
    """Keep every runtime-bootstrap test's process environment isolated."""

    original = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "libero_runtime_bootstrap_test", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object, *, mode: int = 0o600) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[object, argparse.Namespace, dict[str, object]]:
    module = _load_module()
    module.RUNTIME_EXECUTION_GROUP = module.grp.getgrgid(os.getgid()).gr_name
    module.RUNTIME_SUPERVISOR_USER = module.pwd.getpwuid(os.getuid()).pw_name
    license_bytes = b"fixture MIT license\n"
    bddl_bytes = b"fixture bddl\n"
    initial_bytes = b"fixture initial states\n"
    embedding_bytes = b"fixture upstream embedding implementation\n"
    demonstration_bytes = b"fixture demonstration\n"
    model_bytes = b"fixture BERT config\n"
    artifacts = [
        {
            "name": f"package-{index}",
            "version": "1.0",
            "filename": f"package_{index}-1.0-py3-none-any.whl",
            "url": (
                "https://files.pythonhosted.org/packages/00/00/"
                f"package_{index}-1.0-py3-none-any.whl"
            ),
            "size_bytes": len(f"artifact-{index}".encode()),
            "sha256": _sha(f"artifact-{index}".encode()),
            "license_expression": "MIT",
            "metadata_source": f"https://pypi.org/pypi/package-{index}/1.0/json",
        }
        for index in range(135)
    ]
    manifest: dict[str, object] = {
        "schema": module.SCHEMA,
        "solution": "libero",
        "governing_terms": [
            {
                "id": term_id,
                "name": f"Fixture term {index}",
                "boundary": boundary,
                "version": "sha256:" + _sha(f"term-{index}".encode()),
                "url": f"https://www.apache.org/licenses/{index}.txt",
                "size_bytes": len(f"term-{index}"),
                "sha256": _sha(f"term-{index}".encode()),
            }
            for index, (term_id, boundary) in enumerate(
                [
                    ("libero-mit", "source"),
                    ("dataset-cc-by-4.0", "demonstration"),
                    ("bert-apache-2.0", "language_model"),
                    ("pytorch-bsd", "runtime_packages"),
                    ("cuda-eula", "runtime_packages"),
                    ("nvidia-software-license", "runtime_packages"),
                    ("cudnn-eula", "runtime_packages"),
                ]
            )
        ],
        "source": {
            "repository": "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "license": "MIT",
            "license_file": "LICENSE",
            "license_sha256": _sha(license_bytes),
            "sparse_paths": [
                "LICENSE",
                "libero/lifelong/utils.py",
                "libero/libero/bddl_files/task.bddl",
                "libero/libero/init_files/task.init",
            ],
            "forbidden_paths": ["libero/libero/assets", ".git"],
        },
        "demonstration": {
            "repository": "yifengzhu-hf/LIBERO-datasets",
            "revision": "3" * 40,
            "license": "CC-BY-4.0",
            "attribution": "LIBERO, Lifelong Robot Learning",
            "filename": "task_demo.hdf5",
            "url": "https://huggingface.co/datasets/example/resolve/"
            + "3" * 40
            + "/task_demo.hdf5",
            "size_bytes": len(demonstration_bytes),
            "sha256": _sha(demonstration_bytes),
        },
        "task": {
            "suite": "libero_spatial",
            "name": "task",
            "description": "fixture task description",
            "bddl": {
                "path": "libero/libero/bddl_files/task.bddl",
                "sha256": _sha(bddl_bytes),
            },
            "initial_states": {
                "path": "libero/libero/init_files/task.init",
                "sha256": _sha(initial_bytes),
            },
            "embedding_source": {
                "path": "libero/lifelong/utils.py",
                "sha256": _sha(embedding_bytes),
            },
        },
        "language_model": {
            "repository": "google-bert/bert-base-cased",
            "revision": "4" * 40,
            "license": "Apache-2.0",
            "files": [
                {
                    "filename": "config.json",
                    "size_bytes": len(model_bytes),
                    "sha256": _sha(model_bytes),
                }
            ],
        },
        "runtime_artifact_count": len(artifacts),
        "runtime_artifacts": artifacts,
        "runtime_python": dict(module.EXPECTED_RUNTIME_PYTHON),
        "customer_runtime_authorization": dict(
            module.EXPECTED_CUSTOMER_AUTHORIZATION_METADATA
        ),
        "boundaries": dict(module.EXPECTED_BOUNDARIES),
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = manifest_sha
    module.EXECUTABLE_PROFILE_ROOT = tmp_path / "byof-runs"
    requirements_path = tmp_path / "runtime-requirements.txt"
    requirements_path.write_text(
        "\n".join(
            f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']} # {item['url']}"
            for item in artifacts
        )
        + "\n",
        encoding="utf-8",
    )
    module.EXPECTED_RUNTIME_REQUIREMENTS_SHA256 = _sha(requirements_path.read_bytes())
    authorization = {
        "schema": module.CUSTOMER_AUTHORIZATION_SCHEMA,
        "solution": "libero",
        "status": "authorized",
        "authorization_id": "libero-customer-authorization-0001",
        "customer_identity_sha256": "8" * 64,
        "run_id": "libero-runtime-bootstrap-fixture",
        "candidate_image": "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:"
        + "9" * 64,
        "runtime_manifest_sha256": manifest_sha,
        "workflow_profile_sha256": "",
        "upstream_source_revision": manifest["source"]["revision"],
        "terms": [
            {"id": term["id"], "version": term["version"]}
            for term in manifest["governing_terms"]
        ],
        "issuer": "customer",
        "evidence_type": "customer-controlled-signature",
        "customer_signer_public_key_b64": "",
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "nonce": "libero-customer-authorization-nonce-0001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": "",
            "signature_b64": "",
        },
    }
    customer_key = Ed25519PrivateKey.generate()
    customer_public_key = customer_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    caller_key = Ed25519PrivateKey.generate()
    caller_public_key = caller_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    authorization["customer_signer_public_key_b64"] = module.base64.b64encode(
        customer_public_key
    ).decode("ascii")
    customer_signer_registry = tmp_path / "customer-signer-roots"
    customer_signer_registry.mkdir(mode=0o755)
    customer_signer_root = customer_signer_registry / (
        authorization["customer_identity_sha256"] + ".b64"
    )
    customer_signer_root.write_bytes(module.base64.b64encode(customer_public_key))
    customer_signer_root.chmod(0o444)
    module.CUSTOMER_SIGNER_REGISTRY_ROOT = customer_signer_registry
    module.CUSTOMER_SIGNER_REGISTRY_OWNER_UID = os.getuid()
    storage_signer = Ed25519PrivateKey.generate()
    storage_public_key = storage_signer.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    storage_trust_root = tmp_path / "output-storage-authorization-public-key.b64"
    storage_trust_root.write_bytes(module.base64.b64encode(storage_public_key))
    storage_trust_root.chmod(0o444)
    module.OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY = storage_trust_root
    module.OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID = os.getuid()
    authorization["signature"]["public_key_sha256"] = _sha(customer_public_key)
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        customer_key.sign(
            module._customer_authorization_signature_payload(authorization)
        )
    ).decode("ascii")
    caller_assertion = {
        "schema": module.AUTHENTICATED_CALLER_SCHEMA,
        "issuer": "npa-authenticated-caller-control-plane",
        "session_id": "libero-fixture-session-0001",
        "customer_identity_sha256": authorization["customer_identity_sha256"],
        "customer_signer_public_key_sha256": _sha(customer_public_key),
        "run_id": authorization["run_id"],
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        "nonce": "libero-authenticated-caller-nonce-0001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": _sha(caller_public_key),
            "signature_b64": "",
        },
    }
    caller_assertion["signature"]["signature_b64"] = module.base64.b64encode(
        caller_key.sign(
            module._sshsig_signature_payload(
                module.AUTHENTICATED_CALLER_NAMESPACE,
                module._canonical_unsigned_authenticated_caller(caller_assertion),
            )
        )
    ).decode("ascii")
    caller_bytes = json.dumps(caller_assertion, sort_keys=True).encode() + b"\n"
    caller_trust_root = tmp_path / "authenticated-caller-public-key.b64"
    caller_trust_root.write_bytes(module.base64.b64encode(caller_public_key))
    caller_trust_root.chmod(0o444)
    module.AUTHENTICATED_CALLER_TRUST_ROOT = caller_trust_root
    module.AUTHENTICATED_CALLER_TRUST_ROOT_OWNER_UID = os.getuid()
    caller_trust_raw = tmp_path / "authenticated-caller-public-key.raw"
    caller_trust_raw.write_bytes(caller_public_key)
    caller_trust_raw.chmod(0o600)
    caller_assertion_path = tmp_path / "authenticated-caller.json"
    caller_assertion_path.write_bytes(caller_bytes)
    caller_assertion_path.chmod(0o600)
    authorization_path = tmp_path / "customer-authorization.json"
    profile_bytes = b'{"fixture":"libero-executable-profile"}'
    profile_path = module.EXECUTABLE_PROFILE_ROOT / authorization["run_id"] / module.EXECUTABLE_PROFILE_NAME
    profile_path.parent.mkdir(mode=0o770, parents=True)
    profile_path.write_bytes(profile_bytes)
    profile_path.chmod(0o440)
    authorization["workflow_profile_sha256"] = _sha(profile_bytes)
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        customer_key.sign(module._customer_authorization_signature_payload(authorization))
    ).decode("ascii")
    authorization_sha256 = _write_json(authorization_path, authorization)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        requirements=str(requirements_path),
        cache_root=str(tmp_path / "cache"),
        authorization=str(authorization_path),
        authorization_sha256=authorization_sha256,
        output_dir=str(tmp_path / "output"),
    )
    os.environ.update(
        {
            "BYOF_IMAGE": authorization["candidate_image"],
            "NPA_BYOF_RUN_ID": authorization["run_id"],
            "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256": authorization[
                "customer_identity_sha256"
            ],
            "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256": _sha(
                customer_public_key
            ),
            "NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256": authorization[
                "workflow_profile_sha256"
            ],
            "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT": authorization[
                "expires_at"
            ],
            "NPA_LIBERO_AUTHENTICATED_CALLER_B64": module.base64.b64encode(
                caller_bytes
            ).decode("ascii"),
            "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256": _sha(caller_bytes),
            "NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_FILE": str(
                caller_trust_root
            ),
            "NPA_LIBERO_AUTHENTICATED_CALLER_FILE": str(caller_assertion_path),
        }
    )
    fixture = {
        "license": license_bytes,
        "bddl": bddl_bytes,
        "initial": initial_bytes,
        "embedding": embedding_bytes,
        "demonstration": demonstration_bytes,
        "model": model_bytes,
        "manifest_sha": manifest_sha,
        "profile_sha256": _sha(profile_bytes),
        "scope_sha": module._runtime_cache_scope_sha256(
            manifest_sha,
            authorization["customer_identity_sha256"],
            authorization["run_id"],
            authorization_sha256,
        ),
        "customer_private_key": customer_key,
        "customer_public_key": customer_public_key,
        "storage_private_key": storage_signer,
        "storage_public_key": storage_public_key,
        "customer_signer_sha256": _sha(customer_public_key),
        "customer_signer_registry": customer_signer_root,
        "caller_bytes": caller_bytes,
        "caller_assertion_path": caller_assertion_path,
        "caller_trust_root": caller_trust_root,
        "caller_trust_raw": caller_trust_raw,
        "authorization_path": authorization_path,
        "customer_identity_sha256": authorization["customer_identity_sha256"],
        "run_id": authorization["run_id"],
    }
    return module, args, fixture


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("unknown_top_level", "top-level schema"),
        ("stale_authorization_schema", "runtime-authorization metadata"),
        ("runtime_python_drift", "Python identity"),
        ("output_boundary_drift", "boundary metadata"),
    ],
)
def test_runtime_manifest_rejects_unknown_or_stale_contract_metadata(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    module, args, fixture = _fixture(tmp_path)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unknown_top_level":
        manifest["unreviewed"] = True
    elif mutation == "stale_authorization_schema":
        manifest["customer_runtime_authorization"]["schema"] = (
            "npa.libero.customer-runtime-authorization.v0"
        )
    elif mutation == "runtime_python_drift":
        manifest["runtime_python"]["abi"] = "cp311"
    else:
        manifest["boundaries"]["outputs"] = "anywhere"
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(manifest_path, manifest)

    with pytest.raises(module.BootstrapRefusal, match=expected):
        module._validate_manifest(manifest_path)


def test_fetched_execution_environment_excludes_every_storage_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    for name in module.STORAGE_SECRET_ENV_NAMES:
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64", "authorization-secret")
    monkeypatch.setenv("HF_TOKEN", "provider-secret")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/attacker-controlled-libraries")
    monkeypatch.setenv("NPA_BYOF_RUN_ID", "libero-runtime-environment")
    monkeypatch.setenv("PATH", "/attacker-controlled-bin")

    environment = module._runtime_execution_environment(Path("/proc/self/fd/7"))

    assert module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
    assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64" not in environment
    assert "HF_TOKEN" not in environment
    assert "LD_LIBRARY_PATH" not in environment
    assert environment["NPA_BYOF_RUN_ID"] == "libero-runtime-environment"
    assert environment["LIBERO_RUNTIME_ROOT"] == "/proc/self/fd/7"
    assert environment["PATH"] == "/usr/bin:/bin"


def test_output_storage_authorization_is_hash_bound_scoped_and_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, args, fixture = _fixture(tmp_path)
    module.DEFAULT_MANIFEST = Path(args.manifest)
    run_id = fixture["run_id"]
    prefix = f"s3://fixture/byof/{run_id}/"
    endpoint = "https://storage.fixture.invalid"
    credentials = {
        "AWS_ACCESS_KEY_ID": "temporary-access",
        "AWS_SECRET_ACCESS_KEY": "temporary-secret",
        "AWS_SESSION_TOKEN": "temporary-session",
    }
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    policy_sha256 = "a" * 64
    lease_key = f"byof/{run_id}/.npa-output-lease"
    lease_etag = '"lease-etag"'
    lease_version = "lease-version"
    lease_nonce = "lease-nonce-000000000000000000000000000000"
    authorization = {
        "schema": module.OUTPUT_STORAGE_AUTHORIZATION_SCHEMA,
        "issuer": "npa-output-storage-control-plane",
        "capability_id": "libero-output-capability-fixture-0001",
        "customer_identity_sha256": fixture["customer_identity_sha256"],
        "run_id": run_id,
        "candidate_image": os.environ["BYOF_IMAGE"],
        "runtime_manifest_sha256": fixture["manifest_sha"],
        "output_prefix": prefix,
        "endpoint_url": endpoint,
        "access_key_id_sha256": _sha(credentials["AWS_ACCESS_KEY_ID"].encode()),
        "secret_access_key_sha256": _sha(credentials["AWS_SECRET_ACCESS_KEY"].encode()),
        "session_token_sha256": _sha(credentials["AWS_SESSION_TOKEN"].encode()),
        "policy_sha256": policy_sha256,
        "lease_key": lease_key,
        "lease_etag": lease_etag,
        "lease_version_id": lease_version,
        "lease_nonce": lease_nonce,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "nonce": "libero-output-authorization-nonce-0001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": _sha(fixture["storage_public_key"]),
            "signature_b64": "",
        },
    }
    unsigned = json.loads(json.dumps(authorization))
    unsigned.pop("signature")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        fixture["storage_private_key"].sign(
            module._sshsig_signature_payload(
                module.OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE, canonical
            )
        )
    ).decode("ascii")
    payload = (json.dumps(authorization, sort_keys=True) + "\n").encode()
    monkeypatch.setenv(
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        module.base64.b64encode(payload).decode(),
    )
    monkeypatch.setenv(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256", _sha(payload)
    )
    monkeypatch.setenv(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256", _sha(prefix.encode())
    )
    monkeypatch.setenv(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256", policy_sha256
    )
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY", lease_key)
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG", lease_etag)
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID", lease_version)
    monkeypatch.setenv(
        "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
    )

    assert module._storage_authorization(prefix, run_id, endpoint) == authorization

    with pytest.raises(module.BootstrapRefusal, match="invalid or expired"):
        module._storage_authorization(prefix, run_id, "https://other.invalid")

    monkeypatch.delenv("AWS_SESSION_TOKEN")
    with pytest.raises(module.BootstrapRefusal, match="invalid or expired"):
        module._storage_authorization(prefix, run_id, endpoint)


def test_sigv4_storage_request_uses_a_direct_tls_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    for name, value in {
        "AWS_ACCESS_KEY_ID": "temporary-access",
        "AWS_SECRET_ACCESS_KEY": "temporary-secret",
        "AWS_SESSION_TOKEN": "temporary-session",
        "AWS_DEFAULT_REGION": "fixture-region",
    }.items():
        monkeypatch.setenv(name, value)

    observed: dict[str, object] = {}

    class Response:
        status = 200

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return b"read-back"

        def getheaders(self) -> list[tuple[str, str]]:
            return [("X-Amz-Checksum-Sha256", "fixture-checksum")]

        def close(self) -> None:
            observed["response_closed"] = True

    class Connection:
        def __init__(self, hostname: str, port: int, timeout: int) -> None:
            observed.update(hostname=hostname, port=port, timeout=timeout)

        def request(
            self,
            method: str,
            target: str,
            *,
            body: bytes | None,
            headers: dict[str, str],
        ) -> None:
            observed.update(method=method, target=target, body=body, headers=headers)

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            observed["connection_closed"] = True

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)

    object_url = module._s3_object_url(
        "https://storage.example:8443",
        "fixture-bucket",
        "byof/libero-owned-run-0001/libero smoke.json",
    )
    assert object_url.endswith(
        "/fixture-bucket/byof/libero-owned-run-0001/libero%20smoke.json"
    )
    status_code, headers, body = module._sigv4_request(
        "GET",
        object_url,
        extra_headers={"x-amz-checksum-mode": "ENABLED"},
    )

    assert status_code == 200
    assert body == b"read-back"
    assert headers == {"x-amz-checksum-sha256": "fixture-checksum"}
    assert observed["hostname"] == "storage.example"
    assert observed["port"] == 8443
    assert observed["target"] == (
        "/fixture-bucket/byof/libero-owned-run-0001/libero%20smoke.json"
    )
    assert observed["body"] is None
    assert "authorization" in observed["headers"]
    assert observed["response_closed"] is True
    assert observed["connection_closed"] is True
    with pytest.raises(module.BootstrapRefusal, match="credential-free HTTPS"):
        module._sigv4_request("GET", "https://user@storage.example/bucket/object")
    with pytest.raises(module.BootstrapRefusal, match="not canonical"):
        module._sigv4_request(
            "GET", "https://storage.example/fixture-bucket/byof%2Frun/artifact"
        )


def _assert_storage_signature_matches_botocore(method, target, payload, headers):
    """Check the real stdlib signer against the installed AWS SDK without I/O."""
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    request = AWSRequest(
        method=method,
        url="https://storage.example" + target,
        data=payload or b"",
        headers={
            name: value for name, value in headers.items() if name != "authorization"
        },
    )
    request.context["timestamp"] = headers["x-amz-date"]
    signer = S3SigV4Auth(
        Credentials("temporary-access", "temporary-secret", "temporary-session"),
        "s3",
        "fixture-region",
    )
    canonical = signer.canonical_request(request)
    expected = signer.signature(signer.string_to_sign(request, canonical), request)
    assert headers["authorization"].endswith("Signature=" + expected)
    assert canonical.splitlines()[2] == target.partition("?")[2]


def test_version_bound_put_readback_and_cleanup_use_the_real_signer(monkeypatch):
    module = _load_module()
    for name, value in {
        "AWS_ACCESS_KEY_ID": "temporary-access",
        "AWS_SECRET_ACCESS_KEY": "temporary-secret",
        "AWS_SESSION_TOKEN": "temporary-session",
        "AWS_DEFAULT_REGION": "fixture-region",
    }.items():
        monkeypatch.setenv(name, value)
    payload = b"immutable fixture snapshot"
    version = "opaque+/=%2F"
    digest = _sha(payload)
    response_headers = {
        "x-amz-version-id": version,
        "etag": '"fixture-etag"',
        "x-amz-checksum-sha256": module.base64.b64encode(
            bytes.fromhex(digest)
        ).decode(),
        "content-length": str(len(payload)),
    }
    responses = iter([(200, b""), (200, payload), (200, b""), (204, b""), (404, b"")])
    calls = []

    class Response:
        def __init__(self):
            self.status, self.body = next(responses)

        def read(self, _limit):
            return self.body

        def getheaders(self):
            return list(response_headers.items())

        def close(self):
            pass

    class Connection:
        def __init__(self, hostname, port, timeout):
            assert (hostname, port, timeout) == ("storage.example", 443, 120)

        def request(self, method, target, *, body, headers):
            _assert_storage_signature_matches_botocore(method, target, body, headers)
            calls.append((method, target, body, headers))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    attempted = {}
    record = module._verified_output_put(
        endpoint="https://storage.example",
        bucket="fixture-bucket",
        object_key="owned-run/artifact.json",
        name="artifact.json",
        payload=payload,
        digest=digest,
        transaction_token="a" * 32,
        attempted=attempted,
    )
    assert record["version_id"] == version and record["sha256"] == digest
    module._cleanup_output_attempts(
        endpoint="https://storage.example", bucket="fixture-bucket", attempted=attempted
    )
    assert [call[0] for call in calls] == ["PUT", "GET", "HEAD", "DELETE", "HEAD"]
    assert calls[0][1] == "/fixture-bucket/owned-run/artifact.json"
    assert calls[0][2] == payload and calls[0][3]["if-none-match"] == "*"
    assert calls[0][3]["x-amz-meta-npa-transaction-token"] == "a" * 32
    assert calls[1][3]["x-amz-checksum-mode"] == "ENABLED"
    assert calls[3][3]["if-match"] == '"fixture-etag"'
    for _, target, body, _ in calls[1:]:
        assert target == calls[0][1] + "?versionId=opaque%2B%2F%3D%252F"
        assert body is None


@pytest.mark.parametrize(
    "method,query",
    [
        ("PUT", "versionId=version"),
        ("GET", "other=version"),
        ("GET", "versionId=one&versionId=two"),
        ("GET", "versionId=one&other=two"),
        ("GET", "versionId"),
        ("GET", "versionId="),
        ("GET", "versionId=null"),
        ("HEAD", "versionId=a+b"),
        ("DELETE", "versionId=a%2fb"),
        ("GET", "versionId=%FF"),
        ("GET", "versionId=%00"),
        ("GET", "versionId=%ZZ"),
        ("GET", "versionId=" + "a" * 1025),
    ],
)
def test_storage_version_query_refuses_ambiguity_before_transport(
    monkeypatch, method, query
):
    module = _load_module()
    for name in module.STORAGE_SECRET_ENV_NAMES:
        monkeypatch.setenv(name, "fixture-only")
    monkeypatch.setattr(
        module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid version query reached transport"
        ),
    )
    with pytest.raises(module.BootstrapRefusal, match="version"):
        module._sigv4_request(method, "https://storage.example/bucket/object?" + query)


@pytest.mark.parametrize("profile_state", ["bound", "missing", "wrong"])
def test_sudo_allowlist_preserves_real_profile_bound_signature_validation(
    monkeypatch, tmp_path, profile_state
):
    module, args, fixture = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    authorization_bytes = Path(args.authorization).read_bytes()
    dockerfile = (SCRIPT.parent / "Dockerfile").read_text(encoding="utf-8")
    line = next(
        line
        for line in dockerfile.splitlines()
        if "Defaults!NPA_LIBERO_EXEC env_keep" in line
    )
    keep = set(line.split('env_keep += "', 1)[1].split('"', 1)[0].split())
    environment = module._runtime_execution_environment(Path(args.cache_root))
    filtered = {name: value for name, value in environment.items() if name in keep}
    assert filtered["NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256"] == fixture["profile_sha256"]
    assert module.STORAGE_SECRET_ENV_NAMES.isdisjoint(filtered)
    for name in tuple(os.environ):
        monkeypatch.delenv(name)
    for name, value in filtered.items():
        monkeypatch.setenv(name, value)
    if profile_state == "missing":
        monkeypatch.delenv("NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256")
    elif profile_state == "wrong":
        monkeypatch.setenv("NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256", "b" * 64)
    if profile_state != "bound":
        with pytest.raises(
            module.CustomerAcceptanceRequired, match="authorization wrong scope"
        ):
            module._validate_customer_authorization_bytes(
                authorization_bytes,
                args.authorization_sha256,
                manifest,
                module.EXPECTED_RUNTIME_MANIFEST_SHA256,
                authenticated_signer_sha256=fixture["customer_signer_sha256"],
            )
        return
    observed, observed_sha = module._validate_customer_authorization_bytes(
        authorization_bytes,
        args.authorization_sha256,
        manifest,
        module.EXPECTED_RUNTIME_MANIFEST_SHA256,
        authenticated_signer_sha256=fixture["customer_signer_sha256"],
    )
    assert observed["workflow_profile_sha256"] == fixture["profile_sha256"]
    assert observed_sha == args.authorization_sha256


@pytest.mark.parametrize(
    "object_key",
    (
        "byof//artifact.json",
        "byof/./artifact.json",
        "byof/../artifact.json",
    ),
)
def test_s3_object_url_refuses_noncanonical_segments(object_key: str) -> None:
    module = _load_module()

    with pytest.raises(module.BootstrapRefusal, match="object identity"):
        module._s3_object_url("https://storage.example", "bucket", object_key)


def _install_fake_materializers(
    monkeypatch, module, fixture: dict[str, object], *, python_exit: int = 0
) -> None:
    def fetch_source(
        root: Path, source: dict[str, object], *, deadline=None
    ) -> None:
        del deadline
        source_root = root / "source"
        files = {
            str(source["license_file"]): fixture["license"],
            "libero/libero/bddl_files/task.bddl": fixture["bddl"],
            "libero/libero/init_files/task.init": fixture["initial"],
            "libero/lifelong/utils.py": fixture["embedding"],
        }
        for relative, content in files.items():
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def install_runtime(
        root: Path,
        _artifacts: list[dict[str, object]],
        _requirements: list[str],
        *,
        published_root: Path,
        deadline=None,
    ) -> None:
        del deadline
        python = root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            '#!/bin/sh\nset -eu\ntest -f "$LIBERO_CONFIG_PATH/config.yaml"\n'
            f"exit {python_exit}\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        site_packages = root / "venv" / "lib" / "site-packages"
        site_packages.mkdir(parents=True)
        (site_packages / "npa-libero-source.pth").write_text(
            str(published_root / "source") + "\n", encoding="utf-8"
        )

    def fetch_inputs(root: Path, manifest: dict[str, object], *, deadline=None) -> None:
        del deadline
        demo = manifest["demonstration"]
        data = root / "data" / demo["filename"]
        data.parent.mkdir(parents=True)
        data.write_bytes(fixture["demonstration"])
        model = manifest["language_model"]
        model_root = root / "models" / f"bert-base-cased-{model['revision']}"
        model_root.mkdir(parents=True)
        for item in model["files"]:
            (model_root / item["filename"]).write_bytes(fixture["model"])

    monkeypatch.setattr(module, "_fetch_source", fetch_source)
    monkeypatch.setattr(module, "_install_runtime", install_runtime)
    monkeypatch.setattr(module, "_fetch_inputs", fetch_inputs)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda manifest, **_kwargs: module._governing_terms_identity(manifest),
    )


def test_missing_customer_authorization_refuses_before_network_or_cache_mutation(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    Path(args.authorization).unlink()
    monkeypatch.setattr(
        module,
        "_fetch_source",
        lambda *_args: pytest.fail("source fetch started before authorization"),
    )

    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization missing"
    ):
        module.ensure(args)
    assert not Path(args.cache_root).exists()


def test_ensure_revalidates_authorization_before_terms_or_cache_creation(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    original_validate = module._validate_customer_authorization_bytes
    validations = 0

    def validate(*call_args, **call_kwargs):
        nonlocal validations
        validations += 1
        result = original_validate(*call_args, **call_kwargs)
        if validations == 2:
            raise module.CustomerAcceptanceRequired(
                "authorization expired or replayable"
            )
        return result

    monkeypatch.setattr(module, "_validate_customer_authorization_bytes", validate)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda *_args: pytest.fail("terms resolution began after authorization drift"),
    )

    with pytest.raises(
        module.CustomerAcceptanceRequired,
        match="authorization expired or replayable",
    ):
        module.ensure(args)

    assert validations == 2
    assert not Path(args.cache_root).exists()


def test_missing_authorization_notification_names_exact_terms_and_refusal(
    tmp_path,
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest, manifest_sha256 = module._validate_manifest(Path(args.manifest))

    notification = module._customer_acceptance_notification(
        manifest,
        manifest_sha256=manifest_sha256,
        reason="authorization_missing",
    )

    assert notification["status"] == "needs_customer_acceptance"
    assert notification["runtime_manifest_sha256"] == manifest_sha256
    assert notification["reason"] == "authorization_missing"
    assert notification["terms"] == [
        {
            "id": term["id"],
            "name": term["name"],
            "official_url": term["url"],
            "version": term["version"],
        }
        for term in manifest["governing_terms"]
    ]
    assert "Decline" in notification["acknowledgement"]["refusal"]
    assert notification["credentials"] == {
        "purpose": "upstream_access_only",
        "establish_terms_acceptance": False,
    }


def test_invalid_authorization_emits_structured_acceptance_state(
    capsys, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)

    status = module.main(
        [
            "ensure",
            "--manifest",
            args.manifest,
            "--requirements",
            args.requirements,
            "--cache-root",
            args.cache_root,
            "--authorization",
            args.authorization,
            "--authorization-sha256",
            "0" * 64,
            "--output-dir",
            args.output_dir,
        ]
    )

    notification = json.loads(capsys.readouterr().err)
    fixture_manifest_sha = module.EXPECTED_RUNTIME_MANIFEST_SHA256
    assert status == 3
    assert notification["status"] == "needs_customer_acceptance"
    assert notification["reason"] == "authorization_hash_mismatch"
    assert notification["runtime_manifest_sha256"] == fixture_manifest_sha
    assert len(fixture_manifest_sha) == 64


def test_locally_invented_customer_authorization_refuses_before_network_or_cache(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    payload = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    attacker = Ed25519PrivateKey.generate()
    attacker_public = attacker.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    signature = payload["signature"]
    signature["public_key_sha256"] = _sha(attacker_public)
    signature["signature_b64"] = module.base64.b64encode(
        attacker.sign(module._customer_authorization_signature_payload(payload))
    ).decode("ascii")
    args.authorization_sha256 = _write_json(Path(args.authorization), payload)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda *_args: pytest.fail("network authorization began for a local signer"),
    )

    with pytest.raises(
        module.BootstrapRefusal, match="authorization signature invalid"
    ):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


def test_customer_authorization_payload_cannot_substitute_authenticated_signer(
    tmp_path,
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    payload = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    attacker = Ed25519PrivateKey.generate()
    attacker_public = attacker.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload["customer_signer_public_key_b64"] = module.base64.b64encode(
        attacker_public
    ).decode("ascii")
    payload["signature"]["public_key_sha256"] = _sha(attacker_public)
    payload["signature"]["signature_b64"] = module.base64.b64encode(
        attacker.sign(module._customer_authorization_signature_payload(payload))
    ).decode("ascii")
    args.authorization_sha256 = _write_json(Path(args.authorization), payload)

    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization signature invalid"
    ):
        module._validate_customer_authorization(
            Path(args.authorization),
            args.authorization_sha256,
            json.loads(Path(args.manifest).read_text(encoding="utf-8")),
        module.EXPECTED_RUNTIME_MANIFEST_SHA256,
    )


def test_customer_authorization_requires_external_signer_registration(tmp_path) -> None:
    module, args, fixture = _fixture(tmp_path)
    fixture["customer_signer_registry"].unlink()
    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization signature invalid"
    ):
        module._validate_customer_authorization_bytes(
            Path(args.authorization).read_bytes(),
            args.authorization_sha256,
            json.loads(Path(args.manifest).read_text(encoding="utf-8")),
            module.EXPECTED_RUNTIME_MANIFEST_SHA256,
            authenticated_signer_sha256=fixture["customer_signer_sha256"],
        )


def test_customer_authorization_rejects_descriptor_metadata_race(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    authorization_path = Path(args.authorization)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    original_fstat = module.os.fstat
    first_fstat = True

    def racing_fstat(descriptor):
        nonlocal first_fstat
        metadata = original_fstat(descriptor)
        if first_fstat:
            first_fstat = False
            os.utime(
                authorization_path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
            )
        return metadata

    monkeypatch.setattr(module.os, "fstat", racing_fstat)

    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization file invalid"
    ):
        module._validate_customer_authorization(
            authorization_path,
            args.authorization_sha256,
            manifest,
            module.EXPECTED_RUNTIME_MANIFEST_SHA256,
        )


def test_private_input_reader_names_the_supplied_input(tmp_path: Path) -> None:
    module = _load_module()
    candidate = tmp_path / "payload-kubeconfig"
    candidate.write_bytes(b"private input\n")
    candidate.chmod(0o644)
    descriptor = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with pytest.raises(
            module.BootstrapRefusal,
            match="payload kubeconfig file is not stable owner-private",
        ):
            module._read_private_regular_descriptor(
                descriptor,
                limit=1024,
                owner_uid=os.getuid(),
                input_name="payload kubeconfig file",
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("caller_fingerprint", [None, "0" * 64])
def test_absent_or_stale_authenticated_signer_refuses_before_network_or_cache(
    monkeypatch, tmp_path, caller_fingerprint
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    if caller_fingerprint is None:
        monkeypatch.delenv(
            "NPA_LIBERO_AUTHENTICATED_CALLER_B64", raising=False
        )
    else:
        monkeypatch.setenv(
            "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256",
            caller_fingerprint,
        )
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda *_args: pytest.fail("network began without a current caller signer"),
    )

    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization signature invalid"
    ):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


@pytest.mark.parametrize("field", ["size_bytes", "license_expression"])
def test_incomplete_runtime_artifact_review_refuses_before_cache_mutation(
    tmp_path, field
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["runtime_artifacts"][0].pop(field)
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(Path(args.manifest), manifest)

    with pytest.raises(module.BootstrapRefusal, match="size/license review"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


@pytest.mark.parametrize("version", [None, "", "1.0 # unbound", {"value": "1.0"}])
def test_invalid_runtime_artifact_version_refuses_before_requirements_dereference(
    tmp_path: Path, version: object
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["runtime_artifacts"][0]["version"] = version
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(Path(args.manifest), manifest)

    with pytest.raises(module.BootstrapRefusal, match="runtime artifact version"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


@pytest.mark.parametrize(
    "filename",
    ["../escape.whl", "/absolute.whl", "nested/package.whl", ".", "..", "bad name.whl"],
)
def test_runtime_artifact_filename_must_be_a_canonical_leaf(
    tmp_path: Path, filename: str
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["runtime_artifacts"][0]["filename"] = filename
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(Path(args.manifest), manifest)

    with pytest.raises(module.BootstrapRefusal, match="filename"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


def test_runtime_artifact_total_size_budget_refuses_before_cache_mutation(
    tmp_path,
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["runtime_artifacts"][0]["size_bytes"] = (
        module.MAX_RUNTIME_CACHE_DOWNLOAD_BYTES + 1
    )
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(Path(args.manifest), manifest)

    with pytest.raises(module.BootstrapRefusal, match="cache budget"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "run",
        "customer",
        "manifest",
        "workflow-profile",
        "source-revision",
        "customer-signer",
        "expired",
        "denied",
        "acceptance-proxy",
    ],
)
def test_mismatched_customer_authorization_refuses_before_cache_mutation(
    tmp_path, mutation
) -> None:
    module, args, fixture = _fixture(tmp_path)
    if mutation == "hash":
        args.authorization_sha256 = "0" * 64
    else:
        authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
        if mutation == "run":
            authorization["run_id"] = "libero-wrong-customer-run"
        elif mutation == "customer":
            authorization["customer_identity_sha256"] = "7" * 64
        elif mutation == "manifest":
            authorization["runtime_manifest_sha256"] = "0" * 64
        elif mutation == "workflow-profile":
            authorization["workflow_profile_sha256"] = "0" * 64
        elif mutation == "source-revision":
            authorization["upstream_source_revision"] = "0" * 40
        elif mutation == "customer-signer":
            authorization["customer_signer_public_key_b64"] = module.base64.b64encode(
                b"x" * 32
            ).decode("ascii")
            authorization["signature"]["public_key_sha256"] = hashlib.sha256(
                b"x" * 32
            ).hexdigest()
        elif mutation == "expired":
            authorization["expires_at"] = "2000-01-01T00:00:00+00:00"
        elif mutation == "denied":
            authorization["status"] = "denied"
        else:
            authorization["ACCEPT_LIBERO_TERMS"] = "YES"
        authorization["signature"]["signature_b64"] = module.base64.b64encode(
            fixture["customer_private_key"].sign(
                module._customer_authorization_signature_payload(authorization)
            )
        ).decode("ascii")
        args.authorization_sha256 = _write_json(Path(args.authorization), authorization)

    with pytest.raises(module.CustomerAcceptanceRequired):
        module.ensure(args)
    assert not Path(args.cache_root).exists()


def test_customer_authorization_rejects_empty_profile_digest(
    tmp_path: Path,
) -> None:
    module, args, fixture = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    authorization["workflow_profile_sha256"] = ""
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        fixture["customer_private_key"].sign(
            module._customer_authorization_signature_payload(authorization)
        )
    ).decode("ascii")
    authorization_bytes = json.dumps(authorization, sort_keys=True).encode() + b"\n"

    with pytest.raises(module.CustomerAcceptanceRequired, match="authorization wrong scope"):
        module._validate_customer_authorization_bytes(
            authorization_bytes,
            hashlib.sha256(authorization_bytes).hexdigest(),
            manifest,
            module.EXPECTED_RUNTIME_MANIFEST_SHA256,
            authenticated_signer_sha256=fixture["customer_signer_sha256"],
        )


def test_executable_profile_bytes_are_verified_before_cache_effect(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    profile = module._executable_profile_path()
    profile.chmod(0o600)
    profile.write_bytes(b'{"tampered":true}')
    profile.chmod(0o440)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda *_args, **_kwargs: pytest.fail("cache effect began after profile drift"),
    )
    with pytest.raises(module.BootstrapRefusal, match="profile bytes differ"):
        module.ensure(args)
    assert not Path(args.cache_root).exists()


def test_customer_run_reuses_signature_without_caller_or_storage_services(monkeypatch, tmp_path):
    module, args, fixture = _fixture(tmp_path)
    monkeypatch.setenv("NPA_LIBERO_RUNTIME_DELIVERY", module.CUSTOMER_RUN_MODE)
    monkeypatch.setattr(module, "_customer_run_key", lambda: fixture["customer_public_key"])
    monkeypatch.delenv("NPA_LIBERO_AUTHENTICATED_CALLER_B64")
    monkeypatch.delenv("NPA_LIBERO_AUTHENTICATED_CALLER_SHA256")
    caller, key, signer = module._authenticated_caller_binding_from_environment()
    assert caller == b"" and key == fixture["customer_public_key"]
    manifest = json.loads(Path(args.manifest).read_bytes())
    payload = Path(args.authorization).read_bytes()
    accepted, _ = module._validate_customer_authorization_bytes(
        payload, args.authorization_sha256, manifest, fixture["manifest_sha"],
        authenticated_signer_sha256=signer,
    )
    assert accepted["run_id"] == fixture["run_id"]
    tampered = json.loads(payload)
    tampered["customer_identity_sha256"] = "f" * 64
    changed = json.dumps(tampered).encode()
    with pytest.raises(module.CustomerAcceptanceRequired):
        module._validate_customer_authorization_bytes(
            changed, _sha(changed), manifest, fixture["manifest_sha"],
            authenticated_signer_sha256=signer,
        )
    assert not Path(args.cache_root).exists()


def test_customer_mode_must_be_present_in_customer_signed_profile(monkeypatch, tmp_path):
    module, _args, fixture = _fixture(tmp_path)
    monkeypatch.setenv("NPA_LIBERO_RUNTIME_DELIVERY", module.CUSTOMER_RUN_MODE)
    with pytest.raises(module.BootstrapRefusal, match="mode is absent"):
        module._validate_executable_profile_digest(fixture["profile_sha256"])
    profile = module._executable_profile_path()
    profile.chmod(0o600)
    payload = json.dumps([{"resources": {"kubernetes": {"pod_config": {"spec": {"automountServiceAccountToken": False}}}}, "envs": {"NPA_LIBERO_RUNTIME_DELIVERY": module.CUSTOMER_RUN_MODE}}]).encode()
    profile.write_bytes(payload)
    profile.chmod(0o440)
    module._validate_executable_profile_digest(_sha(payload))


def test_unnamed_supervisor_handoff_remains_distinct_from_authorization_file():
    module = _load_module()
    with tempfile.TemporaryFile() as handoff:
        handoff.write(b"synthetic-public-key")
        handoff.flush()
        assert os.fstat(handoff.fileno()).st_nlink == 0
        assert module._read_private_regular_descriptor(
            handoff.fileno(), limit=1024, owner_uid=os.getuid(),
            input_name="supervisor handoff", allow_unlinked=True,
        ) == b"synthetic-public-key"
        with pytest.raises(module.BootstrapRefusal, match="owner-private"):
            module._read_private_regular_descriptor(
                handoff.fileno(), limit=1024, owner_uid=os.getuid(),
                input_name="customer authorization file",
            )


def test_customer_signer_environment_binding_is_required(monkeypatch, tmp_path) -> None:
    module, args, fixture = _fixture(tmp_path)
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256", "0" * 64)
    with pytest.raises(module.CustomerAcceptanceRequired, match="signer binding"):
        module._validate_customer_authorization(
            Path(args.authorization),
            args.authorization_sha256,
            module._validate_manifest(Path(args.manifest))[0],
            module.EXPECTED_RUNTIME_MANIFEST_SHA256,
        )


def test_download_deadline_refuses_before_transport(monkeypatch, tmp_path) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_open_https_download",
        lambda *_args, **_kwargs: pytest.fail("expired authorization reached transport"),
    )
    with pytest.raises(module.CustomerAcceptanceRequired, match="expired or replayable"):
        module._download_verified(
            tmp_path / "artifact",
            url="https://files.pythonhosted.org/artifact.whl",
            sha256="0" * 64,
            size=1,
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )


def test_signed_customer_denial_notifies_before_network_or_cache_mutation(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    authorization["status"] = "denied"
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        fixture["customer_private_key"].sign(
            module._customer_authorization_signature_payload(authorization)
        )
    ).decode("ascii")
    args.authorization_sha256 = _write_json(Path(args.authorization), authorization)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda *_args: pytest.fail("network began after a signed customer denial"),
    )

    with pytest.raises(module.CustomerAcceptanceRequired, match="authorization denied"):
        module.ensure(args)
    assert not Path(args.cache_root).exists()


def test_terms_resolution_refuses_before_cache_mutation(monkeypatch, tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest, **_kwargs: (_ for _ in ()).throw(
            module.BootstrapRefusal("governing terms source no longer resolves")
        ),
    )

    with pytest.raises(module.BootstrapRefusal, match="terms source"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


def test_governing_terms_use_the_separate_terms_allowlist_and_exact_hashes(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest, _ = module._validate_manifest(Path(args.manifest))
    calls: list[dict[str, object]] = []

    def fake_download(_destination: Path, **kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(module, "_download_verified", fake_download)

    digest = module._verify_governing_terms(manifest)

    assert len(calls) == 7
    assert all(call["terms"] is True for call in calls)
    assert all(call["sha256"] for call in calls)
    assert all(call["size"] for call in calls)
    assert len(digest) == 64


def test_runtime_install_uses_only_hash_locked_no_dependency_commands(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    for name in module.STORAGE_SECRET_ENV_NAMES:
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("HF_TOKEN", "unrelated-provider-secret")
    manifest, _ = module._validate_manifest(Path(args.manifest))
    lines, _ = module._validate_requirements(Path(args.requirements), manifest)
    commands: list[tuple[list[str], dict[str, str]]] = []

    def fake_download(destination: Path, **_kwargs) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")

    def fake_run(command: list[str], **kwargs) -> None:
        commands.append((command, kwargs["environment"]))
        if command[1:4] == ["-m", "venv", "--copies"]:
            assert command[4] == str(tmp_path / "runtime" / "venv")
            python = tmp_path / "runtime" / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")

    site_packages = tmp_path / "runtime" / "venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(module, "_download_verified", fake_download)
    monkeypatch.setattr(module, "_run", fake_run)
    check_output_environments: list[dict[str, str]] = []

    def fake_check_output(*_args, **kwargs) -> str:
        check_output_environments.append(kwargs["env"])
        return str(site_packages)

    monkeypatch.setattr(module.subprocess, "check_output", fake_check_output)

    published_root = tmp_path / "cache" / "published-runtime"
    module._install_runtime(
        tmp_path / "runtime",
        manifest["runtime_artifacts"],
        lines,
        published_root=published_root,
    )

    installs = [command for command, _environment in commands if "install" in command]
    assert len(installs) == 2
    assert any("--copies" in command for command, _environment in commands)
    assert all("--require-hashes" in command for command in installs)
    assert all("--no-deps" in command for command in installs)
    assert all("--no-index" in command for command in installs)
    assert all("--find-links" in command for command in installs)
    assert all("--no-cache-dir" in command for command in installs)
    assert all("--only-binary=:all:" in command for command in installs)
    subprocess_environments = [environment for _command, environment in commands]
    subprocess_environments.extend(check_output_environments)
    assert len(subprocess_environments) == 4
    assert all(
        module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
        for environment in subprocess_environments
    )
    assert all("HF_TOKEN" not in environment for environment in subprocess_environments)
    assert all(
        environment["HOME"] == str(tmp_path / "runtime")
        for environment in subprocess_environments
    )
    source_path = (
        tmp_path
        / "runtime"
        / "venv"
        / "lib"
        / "site-packages"
        / "npa-libero-source.pth"
    )
    assert (
        source_path.read_text(encoding="utf-8") == str(published_root / "source") + "\n"
    )


def test_source_archives_become_identity_checked_hash_locked_wheels(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    wheelhouse = tmp_path / "downloads"
    wheelhouse.mkdir()
    build_wheelhouse = tmp_path / ".built-wheelhouse"
    artifact = {
        "name": "fixture-source",
        "version": "1.0",
        "filename": "fixture-source-1.0.tar.gz",
        "sha256": "a" * 64,
    }
    source_line = (
        "fixture-source==1.0 --hash=sha256:"
        + "a" * 64
        + " # https://files.pythonhosted.org/fixture-source-1.0.tar.gz"
    )

    def fake_run(command, **_kwargs):
        assert command[:7] == [
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--pid",
            "--fork",
        ]
        assert command[7:10] == ["--", "/bin/sh", "-ceu"]
        script = command[10]
        assert "mount --make-rprivate /" in script
        assert "/usr/sbin/chroot" in script
        assert "mount --bind" in script and "mount -o remount,ro,bind" in script
        assert "--no-index" in script and "--find-links /npa-build/input" in script
        assert "--wheel-dir /npa-build/output" in script
        assert "/npa-build/source-requirements.txt" in script
        build_wheelhouse.mkdir(mode=0o700, exist_ok=True)
        wheel = build_wheelhouse / "fixture_source-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "fixture_source-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: fixture-source\nVersion: 1.0\n",
            )
        wheel.chmod(0o400)

    monkeypatch.setattr(module, "_run", fake_run)
    module.REVIEWED_SOURCE_BUILD_NAMES = frozenset({"fixture-source"})

    generated = module._build_source_wheels(
        "/venv/bin/python",
        [artifact],
        [source_line],
        wheelhouse,
        build_wheelhouse,
        {"HOME": str(tmp_path)},
    )

    assert generated == [
        "fixture-source==1.0 --hash=sha256:"
        + hashlib.sha256(
            (build_wheelhouse / "fixture_source-1.0-py3-none-any.whl").read_bytes()
        ).hexdigest()
    ]


def test_source_fetch_uses_only_the_credential_free_materialization_environment(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    for name in module.STORAGE_SECRET_ENV_NAMES:
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("HF_TOKEN", "unrelated-provider-secret")
    license_bytes = b"MIT fixture\n"
    source = {
        "repository": "https://example.invalid/LIBERO.git",
        "revision": "a" * 40,
        "tree": "b" * 40,
        "sparse_paths": ["LICENSE"],
        "license_file": "LICENSE",
        "license_sha256": _sha(license_bytes),
    }
    subprocess_environments: list[dict[str, str]] = []
    (tmp_path / "runtime").mkdir()

    def fake_run(command: list[str], **kwargs) -> None:
        subprocess_environments.append(kwargs["environment"])
        if command[1:3] == ["init", "--quiet"]:
            destination = kwargs["cwd"]
            (destination / ".git").mkdir()
            (destination / "LICENSE").write_bytes(license_bytes)

    def fake_check_output(command: list[str], **kwargs) -> str:
        subprocess_environments.append(kwargs["env"])
        return source["revision"] if "^{commit}" in command[-1] else source["tree"]

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module.subprocess, "check_output", fake_check_output)

    module._fetch_source(tmp_path / "runtime", source)

    assert len(subprocess_environments) == 10
    assert all(
        module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
        for environment in subprocess_environments
    )
    assert all("HF_TOKEN" not in environment for environment in subprocess_environments)
    assert all(
        environment["HOME"] == str(tmp_path / "runtime")
        for environment in subprocess_environments
    )


def test_source_fetch_propagates_authorization_deadline_to_every_subprocess(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    license_bytes = b"MIT fixture\n"
    source = {
        "repository": "https://example.invalid/LIBERO.git",
        "revision": "a" * 40,
        "tree": "b" * 40,
        "sparse_paths": ["LICENSE"],
        "license_file": "LICENSE",
        "license_sha256": _sha(license_bytes),
    }
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    runs: list[datetime | None] = []
    captures: list[datetime | None] = []
    (tmp_path / "runtime").mkdir()

    def fake_run(command: list[str], **kwargs) -> None:
        runs.append(kwargs["deadline"])
        if command[1:3] == ["init", "--quiet"]:
            destination = kwargs["cwd"]
            (destination / ".git").mkdir()
            (destination / "LICENSE").write_bytes(license_bytes)

    def fake_capture(command: list[str], **kwargs) -> str:
        captures.append(kwargs["deadline"])
        return source["revision"] if "^{commit}" in command[-1] else source["tree"]

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_run_capture", fake_capture)

    module._fetch_source(tmp_path / "runtime", source, deadline=deadline)

    assert len(runs) == 8
    assert len(captures) == 2
    assert runs == [deadline] * 8
    assert captures == [deadline] * 2


def test_source_fetch_capture_terminates_the_process_group_at_expiry() -> None:
    module = _load_module()
    started = datetime.now(timezone.utc)
    with pytest.raises(module.CustomerAcceptanceRequired, match="expired or replayable"):
        module._run_capture(
            ["/bin/sh", "-ceu", "sleep 10"],
            deadline=started + timedelta(milliseconds=50),
        )
    assert (datetime.now(timezone.utc) - started).total_seconds() < 2


def test_authorized_materialization_is_atomic_and_warm_reusable(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)

    cold = module.ensure(args)
    warm = module.ensure(args)

    final = Path(args.cache_root) / fixture["scope_sha"]
    assert cold["warm_reuse"] is False
    assert warm["warm_reuse"] is True
    assert cold["governing_terms_fetched_this_invocation"] is True
    assert warm["governing_terms_fetched_this_invocation"] is False
    assert final.is_dir()
    assert (final / ".complete.json").is_file()
    assert (final / ".content-inventory.json").is_file()
    assert cold["content_inventory_entry_count"] > 0
    assert cold["content_inventory_sha256"] == warm["content_inventory_sha256"]
    assert (Path(args.cache_root) / "current").resolve() == final.resolve()
    assert not list(Path(args.cache_root).glob(".*.partial-*"))
    assert not (final / "source" / ".git").exists()
    assert not (final / "source" / "libero" / "libero" / "assets").exists()
    assert all(
        path.is_symlink() or path.stat().st_mode & 0o222 == 0
        for path in (final, *final.rglob("*"))
    )
    assert all(
        path.is_symlink() or path.stat().st_gid == os.getgid()
        for path in (final, *final.rglob("*"))
    )
    assert all(
        path.is_symlink()
        or (path.is_dir() and path.stat().st_mode & 0o050 == 0o050)
        or (path.is_file() and path.stat().st_mode & 0o040 == 0o040)
        for path in (final, *final.rglob("*"))
    )


def test_materialized_source_path_targets_published_cache_after_rename(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)

    module.ensure(args)

    final = Path(args.cache_root) / fixture["scope_sha"]
    source_record = final / "venv" / "lib" / "site-packages" / "npa-libero-source.pth"
    recorded_path = source_record.read_text(encoding="utf-8").strip()
    assert recorded_path == str(final / "source")
    assert Path(recorded_path).is_dir()
    assert ".partial-" not in recorded_path
    assert not list(Path(args.cache_root).glob(".*.partial-*"))


def _prepared_execute(monkeypatch, tmp_path):
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    monkeypatch.setattr(module, "DEFAULT_MANIFEST", Path(args.manifest))
    monkeypatch.setattr(module, "DEFAULT_REQUIREMENTS", Path(args.requirements))
    monkeypatch.setattr(module, "DEFAULT_CACHE", Path(args.cache_root))
    monkeypatch.setattr(
        module,
        "RUNTIME_SUPERVISOR_USER",
        module.pwd.getpwuid(os.getuid()).pw_name,
    )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", args.authorization_sha256
    )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        fixture["customer_identity_sha256"],
    )
    monkeypatch.setenv("NPA_BYOF_RUN_ID", fixture["run_id"])
    final = Path(args.cache_root) / fixture["scope_sha"]
    return module, args, final, Path(args.cache_root) / "current"


@contextmanager
def _execution_descriptors(module, args, final):
    root_fd = os.open(args.cache_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    cache_fd = os.open(final, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    authorization_fd = os.open(args.authorization, os.O_RDONLY | os.O_NOFOLLOW)
    caller_fd = os.open(
        os.environ["NPA_LIBERO_AUTHENTICATED_CALLER_FILE"],
        os.O_RDONLY | os.O_NOFOLLOW,
    )
    trust_fd = os.open(
        module.AUTHENTICATED_CALLER_TRUST_ROOT.with_suffix(".raw"),
        os.O_RDONLY | os.O_NOFOLLOW,
    )
    try:
        with ExitStack() as stack:
            execution_lock = stack.enter_context(
                module._cache_lock(
                    root_fd, ".execution.lock", exclusive=True, create=True
                )
            )
            bootstrap_lock = stack.enter_context(
                module._cache_lock(
                    root_fd, ".bootstrap.lock", exclusive=False, create=False
                )
            )
            yield (
                cache_fd,
                authorization_fd,
                execution_lock,
                bootstrap_lock,
                caller_fd,
                trust_fd,
            )
    finally:
        for descriptor in (trust_fd, caller_fd, authorization_fd, cache_fd, root_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_execute_returns_the_exact_smoke_exit_code(monkeypatch, tmp_path) -> None:
    module, args, final, _current = _prepared_execute(monkeypatch, tmp_path)
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    expected_signature_input = module._canonical_unsigned_customer_authorization(
        authorization
    )
    original_run = module.subprocess.run

    def smoke(command, **kwargs):
        if command[:3] == ["/usr/bin/ssh-keygen", "-Y", "verify"]:
            if command[6] != "customer":
                return original_run(command, **kwargs)
            assert len(command) == 11
            allowed_signers = Path(command[4])
            signature = Path(command[10])
            assert command[3:] == [
                "-f",
                str(allowed_signers),
                "-I",
                "customer",
                "-n",
                module.CUSTOMER_AUTHORIZATION_NAMESPACE.decode("ascii"),
                "-s",
                str(signature),
            ]
            assert allowed_signers.name == "allowed-signers"
            assert signature == allowed_signers.with_name("authorization.sig")
            assert kwargs["input"] == expected_signature_input
            assert kwargs["stdout"] == module.subprocess.DEVNULL
            assert kwargs["stderr"] == module.subprocess.DEVNULL
            assert kwargs["env"] == {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"}
            assert kwargs["check"] is False
            assert "capture_output" not in kwargs
            assert "text" not in kwargs
            return original_run(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", smoke)

    class Completed:
        pid = 12345
        returncode = 23

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    def smoke_process(command, **kwargs):
        assert command == ["/opt/npa/libero/smoke.sh"]
        assert kwargs["env"]["LIBERO_RUNTIME_ROOT"].startswith("/proc/self/fd/")
        assert len(kwargs["pass_fds"]) == 1
        for descriptor in (descriptors[1], descriptors[4], descriptors[5]):
            with pytest.raises(OSError):
                os.fstat(descriptor)
        return Completed()

    monkeypatch.setattr(module, "_Popen", smoke_process)

    with _execution_descriptors(module, args, final) as descriptors:
        assert module.execute(*descriptors) == 23


def test_execute_revalidates_expiry_immediately_before_smoke(
    monkeypatch, tmp_path
) -> None:
    module, args, final, _current = _prepared_execute(monkeypatch, tmp_path)
    actual_datetime = module.datetime

    class FutureDateTime(actual_datetime):
        @classmethod
        def now(cls, tz=None):
            return actual_datetime.now(tz) + timedelta(hours=2)

    monkeypatch.setattr(module, "datetime", FutureDateTime)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("smoke began after authorization expiry"),
    )

    with _execution_descriptors(module, args, final) as descriptors:
        with pytest.raises(
            module.CustomerAcceptanceRequired,
            match="authorization expired or replayable",
        ):
            module.execute(*descriptors)


def test_execute_terminates_smoke_when_authorization_expires_during_run(
    monkeypatch, tmp_path
) -> None:
    module, args, final, _current = _prepared_execute(monkeypatch, tmp_path)
    actual_datetime = module.datetime
    now_calls = 0

    class ExpiringDateTime(actual_datetime):
        @classmethod
        def now(cls, tz=None):
            nonlocal now_calls
            now_calls += 1
            current = actual_datetime.now(tz)
            return current if now_calls <= 2 else current + timedelta(hours=2)

    class RunningChild:
        pid = 12347
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 143
            return self.returncode

    child = RunningChild()
    monkeypatch.setattr(module, "datetime", ExpiringDateTime)
    monkeypatch.setattr(module, "_Popen", lambda *_a, **_k: child)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signal_number: killed.append((pid, signal_number)),
    )

    with _execution_descriptors(module, args, final) as descriptors:
        with pytest.raises(
            module.CustomerAcceptanceRequired,
            match="authorization expired or replayable",
        ):
            module.execute(*descriptors)
    assert killed == [(child.pid, module.signal.SIGTERM)]


@pytest.mark.parametrize("drift", ["current-removed", "cache-replaced"])
def test_execute_rejects_post_smoke_cache_identity_drift(
    monkeypatch, tmp_path, drift
) -> None:
    module, args, final, current = _prepared_execute(monkeypatch, tmp_path)
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    expected_signature_input = module._canonical_unsigned_customer_authorization(
        authorization
    )
    original_run = module.subprocess.run

    def smoke(command, **kwargs):
        if command[:3] == ["/usr/bin/ssh-keygen", "-Y", "verify"]:
            if command[6] != "customer":
                return original_run(command, **kwargs)
            assert len(command) == 11
            allowed_signers = Path(command[4])
            signature = Path(command[10])
            assert command[3:] == [
                "-f",
                str(allowed_signers),
                "-I",
                "customer",
                "-n",
                module.CUSTOMER_AUTHORIZATION_NAMESPACE.decode("ascii"),
                "-s",
                str(signature),
            ]
            assert allowed_signers.name == "allowed-signers"
            assert signature == allowed_signers.with_name("authorization.sig")
            assert kwargs["input"] == expected_signature_input
            assert kwargs["stdout"] == module.subprocess.DEVNULL
            assert kwargs["stderr"] == module.subprocess.DEVNULL
            assert kwargs["env"] == {
                "HOME": "/nonexistent",
                "PATH": "/usr/bin:/bin",
            }
            assert kwargs["check"] is False
            assert "capture_output" not in kwargs
            assert "text" not in kwargs
            return original_run(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", smoke)

    class Completed:
        pid = 12346
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    def smoke_process(command, **kwargs):
        assert command == ["/opt/npa/libero/smoke.sh"]
        if drift == "current-removed":
            current.unlink()
        else:
            stale = final.with_name(".stale-execution-cache")
            final.rename(stale)
            final.mkdir(mode=0o550)
        return Completed()

    monkeypatch.setattr(module, "_Popen", smoke_process)

    with _execution_descriptors(module, args, final) as descriptors:
        with pytest.raises(module.BootstrapRefusal, match="changed|current link"):
            module.execute(*descriptors)


def test_execution_process_inventory_requires_runtime_account(
    monkeypatch, tmp_path
) -> None:
    module, _args, _fixture_values = _fixture(tmp_path)

    def missing(_name):
        raise KeyError("absent")

    monkeypatch.setattr(module.pwd, "getpwnam", missing)

    with pytest.raises(module.BootstrapRefusal, match="execution account"):
        module._execution_uid_processes()


def test_execution_process_cleanup_terminates_owned_group_and_process(
    monkeypatch,
):
    module = _load_module()
    signals: list[tuple[str, int, int]] = []
    remaining = iter([[4321], []])

    monkeypatch.setattr(
        module, "_execution_process_group_ids", lambda processes: [5432]
    )
    monkeypatch.setattr(
        module,
        "_execution_uid_processes",
        lambda: next(remaining),
    )
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda group, signal_number: signals.append(("group", group, signal_number)),
    )
    monkeypatch.setattr(
        module.os,
        "kill",
        lambda pid, signal_number: signals.append(("pid", pid, signal_number)),
    )

    module._terminate_execution_processes([4321])

    assert signals == [
        ("group", 5432, module.signal.SIGTERM),
        ("pid", 4321, module.signal.SIGTERM),
    ]


@pytest.mark.parametrize("python_exit", [0, 23])
def test_smoke_propagates_status_and_never_writes_into_sealed_cache(
    monkeypatch, tmp_path, python_exit
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture, python_exit=python_exit)
    module.ensure(args)
    final = Path(args.cache_root) / fixture["scope_sha"]
    output = Path(args.output_dir)
    output.mkdir()
    before = module._inventory_entries(final)
    environment = {
        "LIBERO_RUNTIME_ROOT": str(final),
        "NPA_SMOKE_OUTPUT_DIR": str(output),
    }

    completed = subprocess.run(
        [ROOT / "npa/docker/workbench/libero/smoke.sh"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == python_exit, completed.stderr
    assert module._inventory_entries(final) == before
    assert not (final / "libero-config").exists()
    assert not (output / ".libero-config").exists()


def test_verified_warm_cache_reuse_performs_no_network_fetch(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)

    def reject_network(_manifest) -> str:
        raise AssertionError("warm cache reuse must not resolve the network")

    monkeypatch.setattr(module, "_verify_governing_terms", reject_network)

    warm = module.ensure(args)

    assert warm["warm_reuse"] is True


def test_renewed_authorization_selects_a_distinct_bound_cache(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    first = module.ensure(args)
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    authorization["authorization_id"] = "libero-customer-authorization-0002"
    authorization["nonce"] = "libero-customer-authorization-nonce-0002"
    authorization["signature"]["signature_b64"] = module.base64.b64encode(
        fixture["customer_private_key"].sign(
            module._customer_authorization_signature_payload(authorization)
        )
    ).decode("ascii")
    args.authorization_sha256 = _write_json(Path(args.authorization), authorization)

    renewed = module.ensure(args)

    assert first["warm_reuse"] is False
    assert renewed["warm_reuse"] is False
    assert first["cache_path"] != renewed["cache_path"]
    assert Path(first["cache_path"]).is_dir()
    assert Path(renewed["cache_path"]).is_dir()


def test_manifest_cache_entry_symlink_to_output_refuses_without_network(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    cache_root = Path(args.cache_root)
    cache_root.mkdir()
    output = Path(args.output_dir)
    output.mkdir()
    (cache_root / fixture["scope_sha"]).symlink_to(output, target_is_directory=True)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest: pytest.fail("symlink refusal must precede network"),
    )

    with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
        module.ensure(args)
    assert module.status(args)["materialized"] is False

    assert not (cache_root / "current").exists()


def test_manifest_cache_entry_swap_is_refused_before_descriptor_validation(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    cache_root = Path(args.cache_root)
    final = cache_root / fixture["scope_sha"]
    preserved = cache_root / ".preserved-final"
    output = Path(args.output_dir)
    output.mkdir()
    original_identity = module._cache_entry_identity_at
    first = True
    swapped_identity = None

    def swap_after_initial_identity(parent_descriptor: int, name: str):
        nonlocal first, swapped_identity
        identity = original_identity(parent_descriptor, name)
        if first and name == fixture["scope_sha"]:
            first = False
            assert identity is not None
            swapped_identity = identity
            final.rename(preserved)
            final.symlink_to(output, target_is_directory=True)
        return identity

    monkeypatch.setattr(module, "_cache_entry_identity_at", swap_after_initial_identity)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest: pytest.fail("warm race refusal must remain offline"),
    )
    try:
        with pytest.raises(
            module.BootstrapRefusal,
            match="must be a real directory|rollback was incomplete",
        ):
            module.ensure(args)
    finally:
        try:
            preserved_info = os.lstat(preserved)
        except FileNotFoundError:
            preserved_info = None
        if preserved_info is not None:
            final_info = os.lstat(final)
            assert swapped_identity is not None
            assert stat.S_ISDIR(preserved_info.st_mode)
            assert (preserved_info.st_dev, preserved_info.st_ino) == swapped_identity
            assert stat.S_ISLNK(final_info.st_mode)
            assert os.readlink(final) == str(output)
            final.unlink()
            preserved.rename(final)

    assert (cache_root / "current").resolve() == final.resolve()


def test_manifest_cache_entry_swap_during_validation_removes_current_link(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    cache_root = Path(args.cache_root)
    final = cache_root / fixture["scope_sha"]
    preserved = cache_root / ".preserved-final"
    output = Path(args.output_dir)
    output.mkdir()
    original_validate = module._validate_complete

    def swap_after_descriptor_validation(*validate_args, **validate_kwargs):
        record = original_validate(*validate_args, **validate_kwargs)
        final.rename(preserved)
        final.symlink_to(output, target_is_directory=True)
        return record

    monkeypatch.setattr(module, "_validate_complete", swap_after_descriptor_validation)
    try:
        with pytest.raises(
            module.BootstrapRefusal,
            match="must be a real directory|rollback was incomplete",
        ):
            module.ensure(args)
        assert not (cache_root / "current").exists()
    finally:
        final.unlink(missing_ok=True)
        preserved.rename(final)


def test_cold_cache_entry_swap_after_publication_removes_current_link(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    cache_root = Path(args.cache_root)
    final = cache_root / fixture["scope_sha"]
    preserved = cache_root / ".preserved-final"
    output = Path(args.output_dir)
    output.mkdir()
    original_publish = module._publish_current_cache_link

    def swap_after_publication(*publish_args, **publish_kwargs):
        original_publish(*publish_args, **publish_kwargs)
        final.rename(preserved)
        final.symlink_to(output, target_is_directory=True)

    monkeypatch.setattr(module, "_publish_current_cache_link", swap_after_publication)
    try:
        with pytest.raises(
            module.BootstrapRefusal,
            match="must be a real directory|rollback was incomplete",
        ):
            module.ensure(args)
        assert not (cache_root / "current").exists()
    finally:
        final.unlink(missing_ok=True)
        preserved.rename(final)


@pytest.mark.parametrize(
    "relative",
    ["source/libero/lifelong/utils.py", "venv/bin/python"],
)
def test_warm_cache_refuses_tampered_source_or_runtime(
    monkeypatch, tmp_path, relative
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    target = Path(args.cache_root) / fixture["scope_sha"] / relative
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b"tampered")
    target.chmod(0o400)

    with pytest.raises(module.BootstrapRefusal, match="differs from inventory"):
        module.ensure(args)


def test_warm_cache_refuses_writable_tree(monkeypatch, tmp_path) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    target = (
        Path(args.cache_root)
        / fixture["scope_sha"]
        / "source"
        / "libero"
        / "lifelong"
        / "utils.py"
    )
    target.chmod(0o600)

    with pytest.raises(module.BootstrapRefusal, match="cache is writable"):
        module.ensure(args)


@pytest.mark.parametrize(
    "target", ["inside", "../outside", "/untrusted-external-cache-target"]
)
def test_runtime_cache_refuses_every_symlink(tmp_path, target) -> None:
    module = _load_module()
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "inside").write_text("fixture", encoding="utf-8")
    (root / "link").symlink_to(target)

    with pytest.raises(module.BootstrapRefusal, match="may not contain symlinks"):
        module._inventory_entries(root)


def test_status_validates_inventory_before_reporting_materialized(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    target = (
        Path(args.cache_root)
        / fixture["scope_sha"]
        / "source"
        / "libero"
        / "lifelong"
        / "utils.py"
    )
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b"tampered")
    target.chmod(0o400)

    with pytest.raises(module.BootstrapRefusal, match="differs from inventory"):
        module.status(args)


def test_failed_materialization_removes_partial_cache(monkeypatch, tmp_path) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    monkeypatch.setattr(
        module,
        "_fetch_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    with pytest.raises(RuntimeError, match="fixture failure"):
        module.ensure(args)
    cache = Path(args.cache_root)
    assert not (cache / fixture["scope_sha"]).exists()
    assert not list(cache.glob(".*.partial-*"))
    assert not (cache / "current").exists()


@pytest.mark.parametrize("failure", ["final-validation", "link-publication"])
def test_failed_post_rename_publication_discards_only_new_cache(
    monkeypatch, tmp_path, failure
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)

    if failure == "final-validation":
        monkeypatch.setattr(
            module,
            "_validate_complete",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                module.BootstrapRefusal("post-rename validation failure")
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "_publish_current_cache_link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("post-rename link failure")
            ),
        )

    with pytest.raises((module.BootstrapRefusal, OSError)):
        module.ensure(args)

    cache = Path(args.cache_root)
    assert not (cache / fixture["scope_sha"]).exists()
    assert not (cache / "current").exists()
    assert not list(cache.glob(".*.partial-*"))
    assert not list(cache.glob(".*.failed-*"))


@pytest.mark.parametrize(
    "fault", ["identity", "current-unlink", "quarantine", "descendant-cleanup"]
)
def test_post_rename_rollback_attempts_independent_cleanup_after_fault(
    monkeypatch, tmp_path, fault
) -> None:
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    final = cache / ("a" * 64)
    (final / "nested").mkdir(parents=True)
    (final / "nested" / "payload").write_bytes(b"accepted-cache-bytes")
    (final / "nested" / "payload").chmod(0o400)
    (final / "nested").chmod(0o500)
    final.chmod(0o500)
    expected = module._cache_entry_identity(final)
    assert expected is not None
    (cache / "current").symlink_to(final.name)
    parent_descriptor = os.open(cache, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    calls = 0

    if fault == "identity":
        original = module._cache_entry_identity_at

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise module.BootstrapRefusal("injected identity failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_cache_entry_identity_at", fail_once)
    elif fault == "current-unlink":
        original = module._remove_current_cache_link

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected unlink failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_remove_current_cache_link", fail_once)
    elif fault == "quarantine":
        original = module.tempfile.mkdtemp

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected quarantine failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(module.tempfile, "mkdtemp", fail_once)
    else:
        original = module._remove_cache_tree

        def fail_once(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ["injected descendant cleanup failure"]
            return original(path)

        monkeypatch.setattr(module, "_remove_cache_tree", fail_once)

    try:
        errors = module._rollback_new_cache_entry(
            cache,
            final,
            expected,
            parent_descriptor=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)

    assert errors
    assert not final.exists()
    assert not (cache / "current").exists()
    assert not list(cache.glob(".*.failed-*"))


def test_inner_execution_rejects_path_substitution_after_outer_descriptor_open(
    monkeypatch, tmp_path
) -> None:
    module, args, final, _current = _prepared_execute(monkeypatch, tmp_path)
    preserved = final.with_name(".outer-validated-cache")
    try:
        with _execution_descriptors(module, args, final) as descriptors:
            final.rename(preserved)
            final.mkdir(mode=0o550)
            with pytest.raises(
                module.BootstrapRefusal, match="descriptor identity changed"
            ):
                module.execute(*descriptors)
    finally:
        if final.exists():
            final.rmdir()
        if preserved.exists():
            preserved.rename(final)


def test_cache_output_overlap_and_cache_symlink_refuse(tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    args.output_dir = str(Path(args.cache_root) / "outputs")
    with pytest.raises(module.BootstrapRefusal, match="boundaries overlap"):
        module.ensure(args)

    args.output_dir = str(tmp_path / "output")
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    Path(args.cache_root).symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(
        module.BootstrapRefusal,
        match="runtime cache root contains a link or invalid component",
    ):
        module.ensure(args)


def test_cache_root_creation_callback_refuses_concurrent_appearance(tmp_path) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"

    def create_replacement(parent_descriptor: int, name: str) -> None:
        parent = os.fstat(parent_descriptor)
        expected_parent = os.stat(tmp_path, follow_symlinks=False)
        assert (parent.st_dev, parent.st_ino) == (
            expected_parent.st_dev,
            expected_parent.st_ino,
        )
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        replacement = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            marker = os.open(
                "racer-owned",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=replacement,
            )
            os.write(marker, b"preserve")
            os.close(marker)
        finally:
            os.close(replacement)

    with pytest.raises(module.BootstrapRefusal, match="appeared during creation"):
        with module._open_cache_root_descriptor(
            cache_root,
            create=True,
            before_create=create_replacement,
        ):
            pytest.fail("concurrent cache root was trusted")

    assert (cache_root / "racer-owned").read_bytes() == b"preserve"


def test_cache_lock_refuses_a_symlink_through_the_retained_root(tmp_path) -> None:
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"")
    (cache / ".bootstrap.lock").symlink_to(outside)
    descriptor = os.open(cache, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(module.BootstrapRefusal, match="lock is unavailable"):
            with module._cache_lock(
                descriptor, ".bootstrap.lock", exclusive=True, create=False
            ):
                pytest.fail("symlinked lock was acquired")
    finally:
        os.close(descriptor)


def test_customer_authorization_must_be_owner_private_regular_file(tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    os.chmod(args.authorization, 0o644)
    with pytest.raises(
        module.CustomerAcceptanceRequired, match="authorization file invalid"
    ):
        module.ensure(args)


def test_verified_download_follows_only_an_allowlisted_https_redirect(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    content = b"immutable runtime artifact"

    class Response:
        def __init__(self, status: int, payload: bytes = b"", location: str = ""):
            self.status = status
            self.payload = payload
            self.location = location
            self.closed = False

        def getheader(self, name: str) -> str | None:
            return self.location if name == "Location" else None

        def read(self, _size: int = -1) -> bytes:
            payload, self.payload = self.payload, b""
            return payload

        def close(self) -> None:
            self.closed = True

    responses = [
        Response(302, location="https://cdn-lfs.huggingface.co/object"),
        Response(200, payload=content),
    ]
    connections = []

    class Connection:
        def __init__(self, host: str, port: int, timeout: int):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.closed = False
            connections.append(self)

        def request(self, method: str, target: str, headers: dict[str, str]) -> None:
            assert method == "GET"
            assert target in {"/object", "/package.whl"}
            assert headers == {"User-Agent": "npa-libero-runtime/1"}

        def getresponse(self) -> Response:
            return responses.pop(0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    destination = tmp_path / "package.whl"
    module._download_verified(
        destination,
        url="https://huggingface.co/package.whl",
        sha256=_sha(content),
        size=len(content),
    )

    assert destination.read_bytes() == content
    assert [connection.host for connection in connections] == [
        "huggingface.co",
        "cdn-lfs.huggingface.co",
    ]
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize(
    ("content_length", "payload", "message"),
    [
        ("99", b"short", "Content-Length differs"),
        (None, b"one-byte-too-many", "exceeded expected size"),
    ],
)
def test_verified_download_enforces_size_before_publication(
    monkeypatch, tmp_path, content_length, payload, message
) -> None:
    module = _load_module()

    class Response:
        def __init__(self) -> None:
            self.remaining = payload

        def getheader(self, name: str) -> str | None:
            return content_length if name == "Content-Length" else None

        def read(self, _size: int = -1) -> bytes:
            value, self.remaining = self.remaining, b""
            return value

        def close(self) -> None:
            pass

    class Connection:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        module,
        "_open_https_download",
        lambda *_a, **_k: (Connection(), Response()),
    )
    destination = tmp_path / "package.whl"

    with pytest.raises(module.BootstrapRefusal, match=message):
        module._download_verified(
            destination,
            url="https://files.pythonhosted.org/package.whl",
            sha256=_sha(payload),
            size=len(payload) - 1,
        )

    assert not destination.exists()
    assert not destination.with_name(".package.whl.partial").exists()


def test_verified_download_creates_private_partial_atomically(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    content = b"immutable-runtime-byte"
    observed_modes: list[int] = []

    class Response:
        def __init__(self) -> None:
            self.remaining = content

        def getheader(self, name: str) -> str | None:
            return str(len(content)) if name == "Content-Length" else None

        def read(self, _size: int = -1) -> bytes:
            partial = tmp_path / ".package.whl.partial"
            observed_modes.append(stat.S_IMODE(partial.stat().st_mode))
            value, self.remaining = self.remaining, b""
            return value

        def close(self) -> None:
            pass

    class Connection:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        module,
        "_open_https_download",
        lambda *_a, **_k: (Connection(), Response()),
    )
    destination = tmp_path / "package.whl"

    module._download_verified(
        destination,
        url="https://files.pythonhosted.org/package.whl",
        sha256=_sha(content),
        size=len(content),
    )

    assert observed_modes == [0o600, 0o600]


def test_verified_download_refuses_redirect_outside_allowlist(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()

    class Response:
        status = 302

        @staticmethod
        def getheader(name: str) -> str | None:
            return "https://example.invalid/artifact" if name == "Location" else None

        @staticmethod
        def close() -> None:
            return None

    class Connection:
        def __init__(self, _host: str, _port: int, timeout: int):
            assert timeout == 60

        @staticmethod
        def request(_method: str, _target: str, headers: dict[str, str]) -> None:
            assert headers == {"User-Agent": "npa-libero-runtime/1"}

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    destination = tmp_path / "artifact"

    with pytest.raises(module.BootstrapRefusal, match="redirect left"):
        module._download_verified(
            destination,
            url="https://huggingface.co/artifact",
            sha256="0" * 64,
            size=1,
        )
    assert not destination.exists()


@pytest.mark.parametrize("replace_output_root", [False, True])
def test_execute_and_upload_holds_descriptor_and_cache_lock_through_readback(
    monkeypatch, tmp_path, replace_output_root
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    output = tmp_path / "run-output"
    output.mkdir()
    output.chmod(0o1770)
    protected_authorization = output / module.PROTECTED_AUTHORIZATION_NAME
    protected_authorization.write_bytes(Path(args.authorization).read_bytes())
    protected_authorization.chmod(0o600)
    output_identity = (output.stat().st_dev, output.stat().st_ino)
    replaced_output = tmp_path / "replaced-run-output"
    (output / "libero-smoke.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "DEFAULT_MANIFEST", Path(args.manifest))
    monkeypatch.setattr(module, "DEFAULT_REQUIREMENTS", Path(args.requirements))
    monkeypatch.setattr(module, "DEFAULT_CACHE", Path(args.cache_root))
    monkeypatch.setattr(module, "_run_output_root", lambda _run_id: output)
    receipt = Path(args.cache_root) / "run-receipts" / f"{fixture['run_id']}.json"
    receipt.parent.mkdir(mode=0o750)
    receipt.write_text(
        json.dumps(
            {
                "schema": "npa.libero.runtime-cache.v1",
                "solution": "libero",
                "status": "ready",
                "run_id": fixture["run_id"],
                "manifest_sha256": module.EXPECTED_RUNTIME_MANIFEST_SHA256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt.chmod(0o640)
    monkeypatch.setenv("NPA_BYOF_RUN_ID", fixture["run_id"])
    monkeypatch.setenv("BYOF_SMOKE_ARTIFACT_NAME", "libero-smoke.json")
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", args.authorization_sha256
    )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        fixture["customer_identity_sha256"],
    )

    class Completed:
        returncode = 0

    original_run = module.subprocess.run

    def fake_run(command, **kwargs):
        if command[:3] == ["/usr/bin/ssh-keygen", "-Y", "verify"]:
            if command[6] != "customer":
                return original_run(command, **kwargs)
            assert len(command) == 11
            allowed_signers = Path(command[4])
            signature = Path(command[10])
            assert command[3:] == [
                "-f",
                str(allowed_signers),
                "-I",
                "customer",
                "-n",
                module.CUSTOMER_AUTHORIZATION_NAMESPACE.decode("ascii"),
                "-s",
                str(signature),
            ]
            assert allowed_signers.name == "allowed-signers"
            assert signature == allowed_signers.with_name("authorization.sig")
            return original_run(command, **kwargs)
        assert command == [
            "/usr/bin/sudo",
            "--close-from",
            str(module.INHERITED_CUSTOMER_TRUST_DESCRIPTOR + 1),
            "--user=npa-libero-exec",
            "/opt/npa/libero/runtime-bootstrap.py",
            "execute",
        ]
        assert kwargs["pass_fds"] == (
            module.INHERITED_CACHE_DESCRIPTOR,
            module.INHERITED_AUTHORIZATION_DESCRIPTOR,
            module.INHERITED_EXECUTION_LOCK_DESCRIPTOR,
            module.INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR,
            module.INHERITED_CALLER_DESCRIPTOR,
            module.INHERITED_CUSTOMER_TRUST_DESCRIPTOR,
        )
        opened = os.fstat(module.INHERITED_CACHE_DESCRIPTOR)
        assert stat.S_ISDIR(opened.st_mode)
        assert stat.S_ISREG(os.fstat(module.INHERITED_AUTHORIZATION_DESCRIPTOR).st_mode)
        environment = kwargs["env"]
        assert module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
        assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64" not in environment
        assert "HF_TOKEN" not in environment
        assert environment["NPA_BYOF_RUN_ID"] == fixture["run_id"]
        assert environment["NPA_LIBERO_AUTHENTICATED_CALLER_SHA256"] == _sha(
            fixture["caller_bytes"]
        )
        assert environment["NPA_LIBERO_BOOTSTRAP_RECEIPT"] == str(receipt)
        assert environment["HOME"] == "/nonexistent"
        assert environment["PATH"] == "/usr/bin:/bin"
        with (Path(args.cache_root) / ".execution.lock").open("rb") as competing:
            with pytest.raises(BlockingIOError):
                module.fcntl.flock(
                    competing,
                    module.fcntl.LOCK_EX | module.fcntl.LOCK_NB,
                )
        if replace_output_root:
            output.rename(replaced_output)
            output.mkdir()
            output.chmod(0o1770)
        return Completed()

    lock_path = Path(args.cache_root) / ".bootstrap.lock"

    def fake_upload(smoke_exit_code, *, root_fd):
        assert smoke_exit_code == 0
        opened = os.fstat(root_fd)
        assert (opened.st_dev, opened.st_ino) == output_identity
        with lock_path.open("rb") as competing:
            with pytest.raises(BlockingIOError):
                module.fcntl.flock(
                    competing,
                    module.fcntl.LOCK_EX | module.fcntl.LOCK_NB,
                )
        with (Path(args.cache_root) / ".execution.lock").open("rb") as competing:
            with pytest.raises(BlockingIOError):
                module.fcntl.flock(
                    competing,
                    module.fcntl.LOCK_EX | module.fcntl.LOCK_NB,
                )
        return {"status": "verified"}

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_execution_uid_processes", lambda: [])
    monkeypatch.setattr(module, "upload_outputs", fake_upload)

    if replace_output_root:
        with pytest.raises(
            module.BootstrapRefusal,
            match="output staging directory changed during execution",
        ):
            module.execute_and_upload()
        assert not (output / "npa_runtime_bootstrap.json").exists()
        return

    assert module.execute_and_upload() == 0
    assert (output / "solution_smoke_stdout.log").is_file()
    assert (output / "solution_smoke_stderr.log").is_file()
    assert (output / "npa_runtime_bootstrap.json").read_bytes() == receipt.read_bytes()
    assert (output / "npa_runtime_metadata.json").is_file()
    assert output.stat().st_mode & 0o7777 == 0o700
    assert not protected_authorization.exists()


def _valid_upload_payloads(module, run_id: str) -> dict[str, bytes]:
    digest = "a" * 64
    metadata = {
        "schema": "npa.libero.runtime-cache.v1",
        "solution": "libero",
        "manifest_sha256": digest,
        "customer_authorization_sha256": digest,
        "customer_identity_sha256": digest,
        "run_id": run_id,
        "runtime_requirements_sha256": digest,
        "governing_terms_sha256": digest,
        "governing_terms_count": 1,
        "source_revision": "source-revision",
        "source_tree": digest,
        "source_license_sha256": digest,
        "runtime_artifact_count": 135,
        "demonstration_sha256": digest,
        "demonstration_size_bytes": 10,
        "task_bddl_sha256": digest,
        "task_initial_states_sha256": digest,
        "language_model_revision": "model-revision",
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
        "content_inventory_sha256": digest,
        "content_inventory_entry_count": 3,
    }
    smoke = {
        "schema": "npa.workbench.libero.bc-smoke.v1",
        "status": "passed",
        "exit_status": 0,
        "solution": "libero",
        "capability": "libero_spatial_bc_rnn_train_reload_heldout",
        "capabilities_exercised": ["libero_spatial_bc_rnn_train_reload_heldout"],
        "source": {"repository": "https://example.invalid/libero", "revision": "r", "license": "MIT"},
        "dataset": {
            "repository": "dataset",
            "revision": "r",
            "suite": "libero_spatial",
            "task": "task",
            "task_description": "description",
            "url": "https://example.invalid/data",
            "expected_sha256": digest,
            "expected_size_bytes": 10,
            "license": "CC-BY-4.0",
            "attribution": "LIBERO",
        },
        "task_language_model": {"repository": "model", "revision": "r", "license": "Apache-2.0", "delivery": "runtime_fetch"},
    }
    summary = {
        "status": "success",
        "tool": "byof",
        "workload": "solution-smoke-libero-b200",
        "run_id": run_id,
        "image": "registry.invalid/libero@sha256:" + digest,
        "solution_name": "libero",
        "capability_name": "libero_spatial_bc_rnn_train_reload_heldout",
        "smoke_artifact_name": "libero-smoke.json",
        "smoke_exit_code": 0,
        "runtime_cache_uploaded": False,
        "rendering_invoked": False,
        "created_unix": 1.0,
    }
    return {
        "libero-smoke.json": json.dumps(smoke).encode(),
        "npa_byof_summary.json": json.dumps(summary).encode(),
        "npa_runtime_bootstrap.json": json.dumps({"schema": metadata["schema"], "solution": metadata["solution"], "status": "ready", "run_id": run_id, "manifest_sha256": digest}).encode(),
        "npa_runtime_metadata.json": json.dumps(metadata).encode(),
    }


@pytest.mark.parametrize(
    "failure",
    [None, "artifact-put", "receipt-put", "delete", "absence"],
)
def test_output_upload_is_commit_last_and_cleans_exact_failed_transaction(
    monkeypatch, tmp_path, failure
) -> None:
    module, _args, fixture = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    run_id = fixture["run_id"]
    payloads = _valid_upload_payloads(module, run_id)
    for name in module.OUTPUT_SIZE_LIMITS:
        if name == "npa_byof_summary.json":
            continue
        if name in payloads:
            (output / name).write_bytes(payloads[name])
        else:
            (output / name).write_bytes(f"fixture:{name}\n".encode())
    monkeypatch.setenv("NPA_BYOF_RUN_ID", run_id)
    monkeypatch.setenv("S3_OUTPUT_PREFIX", f"s3://fixture-bucket/byof/{run_id}/")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.fixture.invalid")
    lease_key = f"byof/{run_id}/.npa-output-lease"
    lease_etag = '"lease-etag"'
    lease_version = "lease-version"
    lease_nonce = "lease-nonce-000000000000000000000000000000"
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY", lease_key)
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG", lease_etag)
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID", lease_version)
    capability = {
        "capability_id": "libero-output-capability-fixture-0001",
        "lease_key": lease_key,
        "lease_etag": lease_etag,
        "lease_version_id": lease_version,
        "lease_nonce": lease_nonce,
    }
    validation_calls = 0

    def validate_storage(*_args, **_kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return capability

    objects: dict[str, tuple[bytes, str, str, str]] = {}
    puts: list[str] = []
    deletes: list[str] = []
    injected = False
    lease_present = True

    def request(
        method,
        url,
        *,
        payload=b"",
        extra_headers=None,
        expected_statuses=frozenset({200}),
    ):
        nonlocal injected, lease_present
        base_url, _, query = url.partition("?versionId=")
        if lease_key in base_url:
            assert query == lease_version
            if method == "HEAD":
                if not lease_present:
                    return 404, {}, b""
                return (
                    200,
                    {
                        "etag": lease_etag,
                        "x-amz-version-id": lease_version,
                        "x-amz-meta-npa-lease-nonce": lease_nonce,
                    },
                    b"",
                )
            assert method == "DELETE"
            assert lease_present
            assert extra_headers == {"if-match": lease_etag}
            deletes.append(url)
            lease_present = False
            return 204, {}, b""
        if method == "PUT":
            puts.append(url)
            if failure in {"artifact-put", "delete", "absence"} and len(puts) == 2:
                raise module.BootstrapRefusal("injected staging upload failure")
            if failure == "receipt-put" and url.endswith("/npa_upload_receipt.json"):
                raise module.BootstrapRefusal("injected receipt upload failure")
            checksum = extra_headers["x-amz-checksum-sha256"]
            etag = f'"fixture-{len(puts)}"'
            version = f"version-{len(puts)}"
            objects[url] = (payload, checksum, etag, version)
            return 200, {"etag": etag, "x-amz-version-id": version}, b""
        if method == "GET":
            observed, checksum, etag, version = objects[base_url]
            assert query == version
            return (
                200,
                {
                    "x-amz-checksum-sha256": checksum,
                    "x-amz-version-id": version,
                    "etag": etag,
                },
                observed,
            )
        if method == "HEAD":
            if base_url not in objects:
                return 404, {}, b""
            observed, checksum, etag, version = objects[base_url]
            assert query == version
            return (
                200,
                {
                    "content-length": str(len(observed)),
                    "etag": etag,
                    "x-amz-checksum-sha256": checksum,
                    "x-amz-version-id": version,
                },
                b"",
            )
        assert method == "DELETE"
        deletes.append(url)
        assert extra_headers == {"if-match": objects[base_url][2]}
        if failure == "delete" and not injected:
            injected = True
            raise module.BootstrapRefusal("injected delete failure")
        if failure != "absence" or injected:
            objects.pop(base_url, None)
        else:
            injected = True
        return 204, {}, b""

    monkeypatch.setattr(module, "_storage_authorization", validate_storage)
    monkeypatch.setattr(module, "_sigv4_request", request)
    root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if failure is not None:
            expected = (
                "cleanup is incomplete"
                if failure in {"delete", "absence"}
                else "injected"
            )
            with pytest.raises(module.BootstrapRefusal, match=expected):
                module.upload_outputs(0, root_fd=root_fd)
            assert not any(url.endswith("/npa_upload_receipt.json") for url in objects)
            if failure in {"artifact-put", "receipt-put"}:
                assert objects == {}
            else:
                assert objects
            return
        result = module.upload_outputs(0, root_fd=root_fd)
    finally:
        os.close(root_fd)

    assert result["status"] == "verified"
    assert puts[-1].endswith("/npa_upload_receipt.json")
    assert all("/.npa-transactions/" in url for url in puts[:-1])
    assert deletes == [
        "https://storage.fixture.invalid/fixture-bucket/"
        f"{lease_key}?versionId={lease_version}"
    ]
    receipt = json.loads(objects[puts[-1]][0])
    assert receipt["schema"] == "npa.libero.s3-upload-readback.v2"
    assert receipt["status"] == "verified"
    assert receipt["commit_marker"] == "npa_upload_receipt.json"
    assert {item["name"] for item in receipt["artifacts"]} == set(
        module.OUTPUT_UPLOAD_SIZE_LIMITS
    )
    assert all(
        item["object_key"].startswith(receipt["transaction_prefix"])
        and item["object_key"].endswith("/" + item["name"])
        and item["version_id"]
        and item["etag"]
        for item in receipt["artifacts"]
    )
    assert validation_calls == len(puts) + 1


def test_successful_output_inventory_rejects_and_removes_unexpected_entries(
    tmp_path,
):
    module = _load_module()
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    unexpected = output / "fetched-payload.bin"
    unexpected.write_bytes(b"not an output artifact")
    root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(module.BootstrapRefusal, match="exact artifact allowlist"):
            module._output_limits_for_exit(0, root_fd=root_fd)
    finally:
        os.close(root_fd)
    assert not unexpected.exists()


def test_canonical_output_payload_rejects_unknown_fields_and_embedded_bytes():
    module = _load_module()
    base = {
        "schema": "npa.workbench.libero.bc-smoke.v1",
        "status": "failed",
        "exit_status": 1,
        "solution": "libero",
        "capability": "libero_spatial_bc_rnn_train_reload_heldout",
        "source": {},
        "dataset": {},
    }
    with pytest.raises(module.BootstrapRefusal, match="schema is not closed"):
        module._canonical_output_payload(
            "libero-smoke.json", json.dumps({**base, "unknown": "field"}).encode()
        )
    with pytest.raises(module.BootstrapRefusal, match="embedded output payload"):
        module._canonical_output_payload(
            "libero-smoke.json", json.dumps({**base, "payload": "secret"}).encode()
        )


def test_output_upload_allowlist_excludes_checkpoint_and_raw_logs():
    module = _load_module()
    assert "libero-bc-rnn-smoke.pth" not in module.OUTPUT_UPLOAD_SIZE_LIMITS
    assert "solution_smoke_stdout.log" not in module.OUTPUT_UPLOAD_SIZE_LIMITS
    assert "solution_smoke_stderr.log" not in module.OUTPUT_UPLOAD_SIZE_LIMITS
    assert set(module.OUTPUT_UPLOAD_SIZE_LIMITS) == {
        "libero-smoke.json",
        "npa_byof_summary.json",
        "npa_runtime_bootstrap.json",
        "npa_runtime_metadata.json",
    }


def test_failed_conditional_output_put_never_claims_or_deletes_existing_object(
    monkeypatch,
) -> None:
    module = _load_module()
    attempted: dict[str, dict[str, object]] = {}

    def refuse_existing(*_args, **_kwargs):
        raise module.BootstrapRefusal("conditional create refused existing object")

    monkeypatch.setattr(module, "_sigv4_request", refuse_existing)
    with pytest.raises(module.BootstrapRefusal, match="existing object"):
        module._verified_output_put(
            endpoint="https://storage.fixture.invalid",
            bucket="fixture-bucket",
            object_key="byof/run/.npa-transactions/transaction/artifact.json",
            name="artifact.json",
            payload=b"immutable\n",
            digest=hashlib.sha256(b"immutable\n").hexdigest(),
            transaction_token="b" * 32,
            attempted=attempted,
        )

    assert attempted == {}


def test_successful_put_without_identity_headers_refuses_without_claiming_ownership(
    monkeypatch,
):
    module = _load_module()
    attempted: dict[str, dict[str, object]] = {}
    payload = b"immutable\n"
    digest = hashlib.sha256(payload).hexdigest()
    checksum = module.base64.b64encode(bytes.fromhex(digest)).decode()
    calls: list[str] = []
    observed_payload = b""

    def request(
        method,
        _url,
        *,
        payload=b"",
        extra_headers=None,
        expected_statuses=frozenset({200}),
    ):
        calls.append(method)
        nonlocal observed_payload
        if method == "PUT":
            assert payload == b"immutable\n"
            observed_payload = payload
            return 200, {}, b""
        assert method == "HEAD"
        return (
            200,
            {
                "content-length": str(len(observed_payload)),
                "x-amz-checksum-sha256": checksum,
                "x-amz-version-id": "recovered-version",
                "etag": '"recovered-etag"',
            },
            b"",
        )

    monkeypatch.setattr(module, "_sigv4_request", request)
    with pytest.raises(module.BootstrapRefusal, match="ownership identity unavailable"):
        module._verified_output_put(
            endpoint="https://storage.fixture.invalid",
            bucket="fixture-bucket",
            object_key="byof/run/artifact.json",
            name="artifact.json",
            payload=payload,
            digest=digest,
            transaction_token="c" * 32,
            attempted=attempted,
        )
    assert calls == ["PUT", "HEAD"]
    assert attempted["byof/run/artifact.json"]["version_id"] == ""


def test_ambiguous_put_failure_refuses_without_unversioned_recovery(
    monkeypatch,
):
    module = _load_module()
    attempted: dict[str, dict[str, object]] = {}
    object_key = "byof/run/artifact.json"
    calls: list[str] = []

    def request(method, *_args, **_kwargs):
        calls.append(method)
        raise module.BootstrapRefusal("output storage PUT request failed")

    monkeypatch.setattr(module, "_sigv4_request", request)
    key_hash = hashlib.sha256(object_key.encode()).hexdigest()
    with pytest.raises(module.BootstrapRefusal, match=key_hash):
        module._verified_output_put(
            endpoint="https://storage.fixture.invalid",
            bucket="fixture-bucket",
            object_key=object_key,
            name="artifact.json",
            payload=b"immutable\n",
            digest=hashlib.sha256(b"immutable\n").hexdigest(),
            transaction_token="d" * 32,
            attempted=attempted,
        )
    assert calls == ["PUT", "HEAD"]
    assert attempted[object_key]["version_id"] == ""


def test_ambiguous_put_reconciles_only_exact_transaction_metadata(monkeypatch) -> None:
    module = _load_module()
    content = b"reconciled\n"
    digest = hashlib.sha256(content).hexdigest()
    checksum = module.base64.b64encode(bytes.fromhex(digest)).decode()
    token = "e" * 32
    object_key = "byof/run/.npa-transactions/transaction/artifact.json"
    calls: list[str] = []

    def request(
        method,
        _url,
        *,
        payload=b"",
        extra_headers=None,
        expected_statuses=frozenset({200}),
    ):
        del expected_statuses
        calls.append(method)
        if method == "PUT":
            assert payload == content
            raise module.BootstrapRefusal("output storage PUT request failed")
        if method == "HEAD":
            assert extra_headers == {"x-amz-checksum-mode": "ENABLED"}
            return (
                200,
                {
                    "x-amz-meta-npa-transaction-token": token,
                    "x-amz-checksum-sha256": checksum,
                    "x-amz-version-id": "recovered-version",
                    "etag": '"recovered-etag"',
                },
                b"",
            )
        assert method == "GET"
        return (
            200,
            {
                "x-amz-checksum-sha256": checksum,
                "x-amz-version-id": "recovered-version",
                "etag": '"recovered-etag"',
            },
            content,
        )

    monkeypatch.setattr(module, "_sigv4_request", request)
    attempted: dict[str, dict[str, object]] = {}
    record = module._verified_output_put(
        endpoint="https://storage.fixture.invalid",
        bucket="fixture-bucket",
        object_key=object_key,
        name="artifact.json",
        payload=content,
        digest=digest,
        transaction_token=token,
        attempted=attempted,
    )

    assert calls == ["PUT", "HEAD", "GET"]
    assert record["version_id"] == "recovered-version"
    assert attempted[object_key]["version_id"] == "recovered-version"


def test_output_cleanup_preserves_replacement_between_head_and_delete(
    monkeypatch,
) -> None:
    module = _load_module()
    object_key = "byof/run/libero-smoke.json"
    original = b"original transaction bytes"
    replacement = b"replacement transaction bytes"
    checksum = module.base64.b64encode(
        bytes.fromhex(hashlib.sha256(original).hexdigest())
    ).decode()
    state = {
        "payload": original,
        "etag": '"original-etag"',
        "version_id": "original-version",
    }
    delete_attempts = 0

    def request(
        method,
        _url,
        *,
        payload=b"",
        extra_headers=None,
        expected_statuses=frozenset({200}),
    ):
        nonlocal delete_attempts
        assert payload == b""
        if method == "HEAD":
            observed = state.copy()
            state.update(payload=replacement, etag='"replacement-etag"')
            return (
                200,
                {
                    "content-length": str(len(observed["payload"])),
                    "etag": observed["etag"],
                    "x-amz-checksum-sha256": checksum,
                    "x-amz-version-id": observed["version_id"],
                },
                b"",
            )
        assert method == "DELETE"
        delete_attempts += 1
        assert extra_headers == {"if-match": '"original-etag"'}
        assert 412 in expected_statuses
        return 412, {}, b""

    monkeypatch.setattr(module, "_sigv4_request", request)

    with pytest.raises(module.BootstrapRefusal, match="cleanup is incomplete"):
        module._cleanup_output_attempts(
            endpoint="https://storage.fixture.invalid",
            bucket="fixture-bucket",
            attempted={
                object_key: {
                    "size_bytes": len(original),
                    "checksum": checksum,
                    "etag": '"original-etag"',
                    "version_id": "original-version",
                }
            },
        )

    assert delete_attempts == 1
    assert state == {
        "payload": replacement,
        "etag": '"replacement-etag"',
        "version_id": "original-version",
    }


def test_supervisor_evidence_rejects_group_writable_and_symlinked_files(
    tmp_path,
) -> None:
    module, _args, _fixture_data = _fixture(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    evidence.chmod(0o640)
    assert module._immutable_supervisor_bytes(evidence, 1024) == b"{}\n"

    evidence.chmod(0o660)
    with pytest.raises(module.BootstrapRefusal, match="mutable or invalid"):
        module._immutable_supervisor_bytes(evidence, 1024)
    evidence.chmod(0o640)

    link = tmp_path / "evidence-link.json"
    link.symlink_to(evidence)
    with pytest.raises(module.BootstrapRefusal, match="unavailable"):
        module._immutable_supervisor_bytes(link, 1024)

def test_runtime_storage_trust_root_requires_an_immutable_mounted_file(tmp_path, monkeypatch) -> None:
    module = _load_module()
    key = bytes(range(32))
    mounted = tmp_path / 'output-storage-authorization-public-key.b64'
    monkeypatch.setattr(module, 'OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY', mounted)
    monkeypatch.setattr(module, 'OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID', os.getuid())
    monkeypatch.setenv('NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256', 'f' * 64)
    with pytest.raises(module.BootstrapRefusal, match='unavailable'):
        module._trusted_output_storage_authorization_public_key()
    mounted.write_bytes(module.base64.b64encode(key))
    mounted.chmod(0o444)
    assert module._trusted_output_storage_authorization_public_key() == key
    mounted.chmod(0o644)
    with pytest.raises(module.BootstrapRefusal, match='mutable or invalid'):
        module._trusted_output_storage_authorization_public_key()
