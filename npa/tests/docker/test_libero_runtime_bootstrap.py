"""Fail-closed and atomic tests for the LIBERO runtime materializer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

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
        "runtime_manifest_sha256": manifest_sha,
        "source_revision": "1" * 40,
        "authorized_boundaries": sorted(module.EXPECTED_DECISION_BOUNDARIES),
        "manager_receipt_sha256": "sha256:" + "5" * 64,
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


def _install_fake_materializers(
    monkeypatch, module, fixture: dict[str, object]
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
        python.write_text("#!/bin/sh\n", encoding="utf-8")

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
    manifest, _ = module._validate_manifest(Path(args.manifest))
    lines, _ = module._validate_requirements(Path(args.requirements), manifest)
    commands: list[list[str]] = []

    def fake_download(destination: Path, **_kwargs) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")

    def fake_run(command: list[str], **_kwargs) -> None:
        commands.append(command)
        if command[1:4] == ["-m", "venv", str(tmp_path / "runtime" / "venv")]:
            python = tmp_path / "runtime" / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")

    site_packages = tmp_path / "runtime" / "venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(module, "_download_verified", fake_download)
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *_args, **_kwargs: str(site_packages),
    )

    module._install_runtime(tmp_path / "runtime", manifest["runtime_artifacts"], lines)

    installs = [command for command in commands if "install" in command]
    assert len(installs) == 2
    assert all("--require-hashes" in command for command in installs)
    assert all("--no-deps" in command for command in installs)
    assert all("--no-index" in command for command in installs)
    assert all("--find-links" in command for command in installs)
    assert all("--no-cache-dir" in command for command in installs)


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
