"""Fail-closed and atomic tests for the LIBERO runtime materializer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa" / "docker" / "workbench" / "libero" / "runtime-bootstrap.py"


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
                "boundary": boundary,
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
        "runtime_use_decision": dict(module.EXPECTED_RUNTIME_DECISION_METADATA),
        "boundaries": dict(module.EXPECTED_BOUNDARIES),
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = manifest_sha
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
    decision = {
        "schema": module.DECISION_SCHEMA,
        "solution": "libero",
        "decision": "authorized",
        "runtime_fetch_authorized": True,
        "acceptance_id": "libero-fixture-acceptance-0001",
        "candidate_image": "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:"
        + "9" * 64,
        "publication_bundle_sha256": "a" * 64,
        "infrastructure_bundle_sha256": "b" * 64,
        "runtime_manifest_sha256": manifest_sha,
        "upstream_source_revision": "1" * 40,
        "authorized_boundaries": sorted(module.EXPECTED_DECISION_BOUNDARIES),
        "run_id": "libero-runtime-bootstrap-fixture",
        "namespace_sha256": "c" * 64,
        "issuer": "npa-manager",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "nonce": "libero-runtime-bootstrap-fixture-nonce-0001",
    }
    decision_path = tmp_path / "runtime-use-decision.json"
    decision_sha = _write_json(decision_path, decision)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        requirements=str(requirements_path),
        cache_root=str(tmp_path / "cache"),
        decision=str(decision_path),
        decision_sha256=decision_sha,
        output_dir=str(tmp_path / "output"),
    )
    os.environ.update(
        {
            "NPA_LIBERO_EXPECTED_ACCEPTANCE_ID": decision["acceptance_id"],
            "BYOF_IMAGE": decision["candidate_image"],
            "NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256": decision[
                "publication_bundle_sha256"
            ],
            "NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256": decision[
                "infrastructure_bundle_sha256"
            ],
            "NPA_BYOF_RUN_ID": decision["run_id"],
            "NPA_LIBERO_EXPECTED_NAMESPACE_SHA256": decision[
                "namespace_sha256"
            ],
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
    }
    return module, args, fixture


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("unknown_top_level", "top-level schema"),
        ("stale_decision_schema", "decision metadata"),
        ("runtime_python_drift", "Python identity"),
        ("output_boundary_drift", "boundary metadata"),
    ],
)
def test_runtime_manifest_rejects_unknown_or_stale_contract_metadata(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unknown_top_level":
        manifest["unreviewed"] = True
    elif mutation == "stale_decision_schema":
        manifest["runtime_use_decision"]["schema"] = (
            "npa.libero.runtime-use-decision.v1"
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
    monkeypatch.setenv("NPA_LIBERO_RUNTIME_USE_DECISION_B64", "decision-secret")
    monkeypatch.setenv("HF_TOKEN", "provider-secret")
    monkeypatch.setenv("NPA_BYOF_RUN_ID", "libero-runtime-environment")

    environment = module._runtime_execution_environment(Path("/proc/self/fd/7"))

    assert module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
    assert "NPA_LIBERO_RUNTIME_USE_DECISION_B64" not in environment
    assert "HF_TOKEN" not in environment
    assert environment["NPA_BYOF_RUN_ID"] == "libero-runtime-environment"
    assert environment["LIBERO_RUNTIME_ROOT"] == "/proc/self/fd/7"


def test_output_storage_authorization_is_hash_bound_scoped_and_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    run_id = "libero-output-authorization"
    prefix = f"s3://fixture/byof/{run_id}/"
    credentials = {
        "AWS_ACCESS_KEY_ID": "temporary-access",
        "AWS_SECRET_ACCESS_KEY": "temporary-secret",
        "AWS_SESSION_TOKEN": "temporary-session",
    }
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    policy_sha256 = "a" * 64
    authorization = {
        "schema": "npa.libero.output-storage-authorization.v1",
        "issuer": "npa-manager",
        "run_id": run_id,
        "output_prefix": prefix,
        "access_key_id_sha256": _sha(credentials["AWS_ACCESS_KEY_ID"].encode()),
        "secret_access_key_sha256": _sha(
            credentials["AWS_SECRET_ACCESS_KEY"].encode()
        ),
        "session_token_sha256": _sha(credentials["AWS_SESSION_TOKEN"].encode()),
        "policy_sha256": policy_sha256,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "nonce": "libero-output-authorization-nonce-0001",
    }
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

    assert module._storage_authorization(prefix, run_id) == authorization

    monkeypatch.delenv("AWS_SESSION_TOKEN")
    with pytest.raises(module.BootstrapRefusal, match="invalid or expired"):
        module._storage_authorization(prefix, run_id)


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

    headers, body = module._sigv4_request(
        "GET",
        "https://storage.example:8443/bucket/object",
        extra_headers={"x-amz-checksum-mode": "ENABLED"},
    )

    assert body == b"read-back"
    assert headers == {"x-amz-checksum-sha256": "fixture-checksum"}
    assert observed["hostname"] == "storage.example"
    assert observed["port"] == 8443
    assert observed["target"] == "/bucket/object"
    assert observed["body"] is None
    assert "authorization" in observed["headers"]
    assert observed["response_closed"] is True
    assert observed["connection_closed"] is True
    with pytest.raises(module.BootstrapRefusal, match="query-free HTTPS"):
        module._sigv4_request("GET", "https://user@storage.example/bucket/object")


def _install_fake_materializers(
    monkeypatch, module, fixture: dict[str, object], *, python_exit: int = 0
) -> None:
    def fetch_source(root: Path, source: dict[str, object]) -> None:
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
    ) -> None:
        python = root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            '#!/bin/sh\nset -eu\ntest -f "$LIBERO_CONFIG_PATH/config.yaml"\n'
            f"exit {python_exit}\n",
            encoding="utf-8",
        )
        python.chmod(0o755)

    def fetch_inputs(root: Path, manifest: dict[str, object]) -> None:
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
        lambda manifest: module._governing_terms_identity(manifest),
    )


def test_missing_decision_refuses_before_network_or_cache_mutation(
    monkeypatch, tmp_path
) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    Path(args.decision).unlink()
    monkeypatch.setattr(
        module,
        "_fetch_source",
        lambda *_args: pytest.fail("source fetch started before authorization"),
    )

    with pytest.raises(
        module.BootstrapRefusal, match="decision file is unavailable|owner-only"
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
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(
        Path(args.manifest), manifest
    )

    with pytest.raises(module.BootstrapRefusal, match="size/license review"):
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
    module.EXPECTED_RUNTIME_MANIFEST_SHA256 = _write_json(
        Path(args.manifest), manifest
    )

    with pytest.raises(module.BootstrapRefusal, match="cache budget"):
        module.ensure(args)

    assert not Path(args.cache_root).exists()


@pytest.mark.parametrize("mutation", ["hash", "boundary", "acceptance-proxy"])
def test_mismatched_decision_refuses_before_cache_mutation(tmp_path, mutation) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    if mutation == "hash":
        args.decision_sha256 = "0" * 64
    else:
        decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
        if mutation == "boundary":
            decision["authorized_boundaries"].remove("demonstration")
        else:
            decision["ACCEPT_LIBERO_TERMS"] = "YES"
        args.decision_sha256 = _write_json(Path(args.decision), decision)

    with pytest.raises(module.BootstrapRefusal):
        module.ensure(args)
    assert not Path(args.cache_root).exists()


def test_terms_resolution_refuses_before_cache_mutation(monkeypatch, tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest: (_ for _ in ()).throw(
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

    module._install_runtime(tmp_path / "runtime", manifest["runtime_artifacts"], lines)

    installs = [command for command, _environment in commands if "install" in command]
    assert len(installs) == 2
    assert any("--copies" in command for command, _environment in commands)
    assert all("--require-hashes" in command for command in installs)
    assert all("--no-deps" in command for command in installs)
    assert all("--no-index" in command for command in installs)
    assert all("--find-links" in command for command in installs)
    assert all("--no-cache-dir" in command for command in installs)
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


def test_authorized_materialization_is_atomic_and_warm_reusable(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)

    cold = module.ensure(args)
    warm = module.ensure(args)

    final = Path(args.cache_root) / fixture["manifest_sha"]
    assert cold["warm_reuse"] is False
    assert warm["warm_reuse"] is True
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


@pytest.mark.parametrize("python_exit", [0, 23])
def test_smoke_propagates_status_and_never_writes_into_sealed_cache(
    monkeypatch, tmp_path, python_exit
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture, python_exit=python_exit)
    module.ensure(args)
    final = Path(args.cache_root) / fixture["manifest_sha"]
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


def test_manifest_cache_entry_symlink_to_output_refuses_without_network(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    cache_root = Path(args.cache_root)
    cache_root.mkdir()
    output = Path(args.output_dir)
    output.mkdir()
    (cache_root / fixture["manifest_sha"]).symlink_to(output, target_is_directory=True)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest: pytest.fail("symlink refusal must precede network"),
    )

    with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
        module.ensure(args)
    with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
        module.status(args)

    assert not (cache_root / "current").exists()


def test_manifest_cache_entry_swap_is_refused_before_descriptor_validation(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    cache_root = Path(args.cache_root)
    final = cache_root / fixture["manifest_sha"]
    preserved = cache_root / ".preserved-final"
    output = Path(args.output_dir)
    output.mkdir()
    original_identity = module._cache_entry_identity
    first = True

    def swap_after_initial_identity(path: Path):
        nonlocal first
        identity = original_identity(path)
        if first and path == final:
            first = False
            final.rename(preserved)
            final.symlink_to(output, target_is_directory=True)
        return identity

    monkeypatch.setattr(module, "_cache_entry_identity", swap_after_initial_identity)
    monkeypatch.setattr(
        module,
        "_verify_governing_terms",
        lambda _manifest: pytest.fail("warm race refusal must remain offline"),
    )
    try:
        with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
            module.ensure(args)
    finally:
        final.unlink(missing_ok=True)
        preserved.rename(final)

    assert (cache_root / "current").resolve() == final.resolve()


def test_manifest_cache_entry_swap_during_validation_removes_current_link(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    cache_root = Path(args.cache_root)
    final = cache_root / fixture["manifest_sha"]
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
        with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
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
    final = cache_root / fixture["manifest_sha"]
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
        with pytest.raises(module.BootstrapRefusal, match="must be a real directory"):
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
    target = Path(args.cache_root) / fixture["manifest_sha"] / relative
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
        / fixture["manifest_sha"]
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
        / fixture["manifest_sha"]
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
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    with pytest.raises(RuntimeError, match="fixture failure"):
        module.ensure(args)
    cache = Path(args.cache_root)
    assert not (cache / fixture["manifest_sha"]).exists()
    assert not list(cache.glob(".*.partial-*"))
    assert not (cache / "current").exists()


def test_cache_output_overlap_and_cache_symlink_refuse(tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    args.output_dir = str(Path(args.cache_root) / "outputs")
    with pytest.raises(module.BootstrapRefusal, match="boundaries overlap"):
        module.ensure(args)

    args.output_dir = str(tmp_path / "output")
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    Path(args.cache_root).symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(module.BootstrapRefusal, match="may not be a symlink"):
        module.ensure(args)


def test_decision_must_be_owner_private_regular_file(tmp_path) -> None:
    module, args, _fixture_values = _fixture(tmp_path)
    os.chmod(args.decision, 0o644)
    with pytest.raises(module.BootstrapRefusal, match="owner-only regular file"):
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


def test_execute_and_upload_holds_cache_lock_through_readback(
    monkeypatch, tmp_path
) -> None:
    module, args, fixture = _fixture(tmp_path)
    _install_fake_materializers(monkeypatch, module, fixture)
    module.ensure(args)
    output = tmp_path / "run-output"
    output.mkdir()
    (output / "libero-smoke.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "DEFAULT_MANIFEST", Path(args.manifest))
    monkeypatch.setattr(module, "DEFAULT_REQUIREMENTS", Path(args.requirements))
    monkeypatch.setattr(module, "DEFAULT_CACHE", Path(args.cache_root))
    monkeypatch.setattr(module, "_run_output_root", lambda _run_id: output)
    monkeypatch.setenv("NPA_BYOF_RUN_ID", "libero-lock-test-0001")
    monkeypatch.setenv("BYOF_SMOKE_ARTIFACT_NAME", "libero-smoke.json")
    monkeypatch.setenv(
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256", args.decision_sha256
    )

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        assert command == [
            "sudo",
            "--user=npa-libero-exec",
            "/opt/npa/libero/runtime-bootstrap.py",
            "execute",
        ]
        environment = kwargs["env"]
        assert module.STORAGE_SECRET_ENV_NAMES.isdisjoint(environment)
        assert "NPA_LIBERO_RUNTIME_USE_DECISION_B64" not in environment
        assert "HF_TOKEN" not in environment
        assert environment["NPA_BYOF_RUN_ID"] == "libero-lock-test-0001"
        assert environment["HOME"] == "/nonexistent"
        return Completed()

    lock_path = Path(args.cache_root) / ".bootstrap.lock"

    def fake_upload(smoke_exit_code):
        assert smoke_exit_code == 0
        with lock_path.open("rb") as competing:
            with pytest.raises(BlockingIOError):
                module.fcntl.flock(
                    competing,
                    module.fcntl.LOCK_EX | module.fcntl.LOCK_NB,
                )
        return {"status": "verified"}

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "upload_outputs", fake_upload)

    assert module.execute_and_upload() == 0
    assert (output / "solution_smoke_stdout.log").is_file()
    assert (output / "solution_smoke_stderr.log").is_file()
