from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from typing import BinaryIO
import urllib.error
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/docker/workbench/gymnasium-robotics/runtime-bootstrap.py"
LOCK = ROOT / "npa/docker/workbench/gymnasium-robotics/source-lock.json"
REQUIREMENTS = ROOT / "npa/docker/workbench/gymnasium-robotics/requirements.lock"
SPEC = importlib.util.spec_from_file_location("gymnasium_runtime_bootstrap", SCRIPT)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)


@pytest.fixture(autouse=True)
def _one_wheel_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BOOTSTRAP, "EXPECTED_WHEEL_COUNT", 1)


class _Response(io.BytesIO):
    def __init__(self, content: bytes, final_url: str) -> None:
        super().__init__(content)
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


def _source_archive(
    files: dict[str, bytes] | None = None,
    *,
    symlink: tuple[str, str] | None = None,
) -> bytes:
    raw = io.BytesIO()
    prefix = "Gymnasium-Robotics-fixture"
    with tarfile.open(fileobj=raw, mode="w") as archive:
        root = tarfile.TarInfo(prefix)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, content in (files or {"pyproject.toml": b"[build-system]\n"}).items():
            member = tarfile.TarInfo(f"{prefix}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if symlink:
            member = tarfile.TarInfo(f"{prefix}/{symlink[0]}")
            member.type = tarfile.SYMTYPE
            member.linkname = symlink[1]
            archive.addfile(member)
    return gzip.compress(raw.getvalue(), mtime=0)


def _wheel() -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("fixture/__init__.py", "__version__ = '1'\n")
        archive.writestr("fixture-1.dist-info/METADATA", "Name: fixture\nVersion: 1\n")
    return raw.getvalue()


def _write_inputs(
    tmp_path: Path,
    *,
    source: bytes | None = None,
    wheel: bytes | None = None,
    mutate: dict[str, object] | None = None,
) -> tuple[Path, Path, dict[str, bytes]]:
    source = source if source is not None else _source_archive()
    wheel = wheel if wheel is not None else _wheel()
    requirements = tmp_path / "requirements.lock"
    requirements.write_text(
        "# status: complete\nfixture==1 --hash=sha256:"
        + hashlib.sha256(wheel).hexdigest()
        + "\n",
        encoding="utf-8",
    )
    urls = {
        "https://github.com/example/source.tar.gz": source,
        "https://files.pythonhosted.org/packages/fixture.whl": wheel,
    }
    payload: dict[str, object] = {
        "schema": BOOTSTRAP.MANIFEST_SCHEMA,
        "status": "complete",
        "source_commit": BOOTSTRAP.EXPECTED_SOURCE_COMMIT,
        "mujoco_version": BOOTSTRAP.EXPECTED_MUJOCO_VERSION,
        "decision_sha256": BOOTSTRAP.EXPECTED_DECISION_SHA256,
        "requirements_lock_sha256": hashlib.sha256(
            requirements.read_bytes()
        ).hexdigest(),
        "rights_boundary": BOOTSTRAP.RIGHTS_BOUNDARY,
        "artifacts": [
            {
                "name": "gymnasium-robotics-source",
                "role": "solution-source",
                "url": "https://github.com/example/source.tar.gz",
                "final_url": "https://github.com/example/source.tar.gz",
                "sha256": hashlib.sha256(source).hexdigest(),
                "size_bytes": len(source),
                "filename": "source.tar.gz",
                "archive": "tar.gz",
                "strip_prefix": "Gymnasium-Robotics-fixture",
                "max_unpacked_bytes": 1024 * 1024,
            },
            {
                "name": "mujoco-3.12.0-cp312-linux-x86_64",
                "role": "python-wheel",
                "url": "https://files.pythonhosted.org/packages/fixture.whl",
                "final_url": "https://files.pythonhosted.org/packages/fixture.whl",
                "sha256": hashlib.sha256(wheel).hexdigest(),
                "size_bytes": len(wheel),
                "filename": "fixture.whl",
                "archive": "wheel",
                "strip_prefix": None,
                "max_unpacked_bytes": 1024 * 1024,
            },
        ],
        "expected_python_distribution_count": 1,
        "resolved_python_artifact_count": 1,
        "components": {
            "farama_gymnasium_robotics": {
                "archive_sha256": hashlib.sha256(source).hexdigest()
            },
            "mujoco": {"wheel_sha256": hashlib.sha256(wheel).hexdigest()},
        },
    }
    if mutate:
        payload.update(mutate)
    manifest = tmp_path / "source-lock.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, requirements, urls


def _opener(
    content: dict[str, bytes],
    calls: list[str],
    redirects: dict[str, str] | None = None,
):
    def open_url(url: str) -> contextlib.AbstractContextManager[BinaryIO]:
        calls.append(url)
        return _Response(content[url], (redirects or {}).get(url, url))

    return open_url


def _installer(stage: Path, requirements: Path) -> None:
    assert requirements.parent == stage
    assert (stage / "source/pyproject.toml").is_file()
    assert (stage / "wheelhouse/fixture.whl").is_file()
    python = stage / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)


def test_repository_lock_refuses_before_any_network_access(tmp_path: Path) -> None:
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="before network access"):
        BOOTSTRAP.prepare(
            LOCK,
            REQUIREMENTS,
            tmp_path / "cache",
            opener=_opener({}, calls),
            installer=_installer,
        )
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_verified_runtime_is_atomically_published_and_reused(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    calls: list[str] = []
    cache = tmp_path / "cache"
    first = BOOTSTRAP.prepare(
        manifest,
        requirements,
        cache,
        opener=_opener(content, calls),
        installer=_installer,
    )
    target = Path(str(first["runtime_root"]))
    assert first["cache_reused"] is False
    assert target.parent == cache / "versions"
    assert (cache / "current").resolve() == target
    assert json.loads((target / "receipt.json").read_text())["status"] == "ready"
    assert (target / "source/pyproject.toml").read_bytes() == b"[build-system]\n"
    assert list((cache / ".staging").iterdir()) == []
    assert cache.stat().st_uid == os.geteuid()
    assert cache.stat().st_mode & 0o077 == 0
    assert target.stat().st_mode & 0o222 == 0
    tree_raw = (target / "tree-manifest.json").read_bytes()
    tree = json.loads(tree_raw)
    assert tree["schema"] == BOOTSTRAP.TREE_MANIFEST_SCHEMA
    tree_paths = {entry["path"] for entry in tree["entries"]}
    assert {
        "runtime/bin/python",
        "source/pyproject.toml",
        "wheelhouse/fixture.whl",
    } <= tree_paths
    assert all(entry["mode"] & 0o222 == 0 for entry in tree["entries"])
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["tree_manifest_sha256"] == hashlib.sha256(tree_raw).hexdigest()
    assert len(calls) == 2

    second_calls: list[str] = []
    second = BOOTSTRAP.prepare(
        manifest,
        requirements,
        cache,
        opener=_opener({}, second_calls),
        installer=lambda *_: pytest.fail("installer must not run for an exact cache"),
    )
    assert second["cache_reused"] is True
    assert second_calls == []


@pytest.mark.parametrize("mutation", ["content", "extra", "symlink", "receipt"])
def test_cache_reuse_refuses_any_unsealed_or_unmanifested_tree(
    tmp_path: Path, mutation: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    first = BOOTSTRAP.prepare(
        manifest,
        requirements,
        cache,
        opener=_opener(content, []),
        installer=_installer,
    )
    target = Path(str(first["runtime_root"]))
    if mutation == "content":
        candidate = target / "source/pyproject.toml"
        candidate.chmod(0o600)
        candidate.write_text("changed\n", encoding="utf-8")
        candidate.chmod(0o400)
    elif mutation == "extra":
        target.chmod(0o700)
        candidate = target / "extra.bin"
        candidate.write_bytes(b"unexpected")
        candidate.chmod(0o400)
        target.chmod(0o500)
    elif mutation == "symlink":
        target.chmod(0o700)
        (target / "escape").symlink_to("/etc/passwd")
        target.chmod(0o500)
    else:
        candidate = target / "receipt.json"
        receipt = json.loads(candidate.read_text())
        receipt["status"] = "changed"
        candidate.chmod(0o600)
        candidate.write_text(json.dumps(receipt), encoding="utf-8")
        candidate.chmod(0o400)
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="runtime cache"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener({}, []),
            installer=lambda *_: pytest.fail("a mismatched cache must not reinstall"),
        )


def test_unsafe_installer_tree_is_removed_before_publication(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"

    def unsafe_installer(stage: Path, locked_requirements: Path) -> None:
        _installer(stage, locked_requirements)
        (stage / "source/escape").symlink_to("/etc/passwd")

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="symbolic link"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=unsafe_installer,
        )
    assert not (cache / "current").exists()
    assert list((cache / "versions").iterdir()) == []
    assert list((cache / ".staging").iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "incomplete", "before network access"),
        ("source_commit", "0" * 40, "source commit changed"),
        ("mujoco_version", "3.11.0", "MuJoCo version changed"),
        ("decision_sha256", "0" * 64, "manager decision binding changed"),
        ("rights_boundary", "accepted=yes", "rights boundary"),
        ("requirements_lock_sha256", "0" * 64, "requirements lock bytes changed"),
    ],
)
def test_manifest_trust_boundary_refuses_without_fetch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path, mutate={field: value})
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match=message):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/example/source.tar.gz",
        "https://user:secret@github.com/example/source.tar.gz",
        "https://example.invalid/source.tar.gz",
        "https://github.com/example/source.tar.gz?mutable=1",
        "https://github.com/example/source.tar.gz#fragment",
    ],
)
def test_unapproved_or_credential_bearing_url_refuses_before_fetch(
    tmp_path: Path, url: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"][0]["url"] = url
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="credential-free immutable"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


@pytest.mark.parametrize("mismatch", ["sha256", "size"])
def test_mismatched_artifact_leaves_no_published_or_partial_runtime(
    tmp_path: Path, mismatch: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"][0][mismatch if mismatch == "sha256" else "size_bytes"] = (
        "0" * 64 if mismatch == "sha256" else len(next(iter(content.values()))) + 1
    )
    if mismatch == "sha256":
        payload["components"]["farama_gymnasium_robotics"]["archive_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="mismatch"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, []),
            installer=_installer,
        )
    assert not (tmp_path / "cache/current").exists()
    assert list((tmp_path / "cache/versions").iterdir()) == []
    assert list((tmp_path / "cache/.staging").iterdir()) == []


def test_changed_redirect_refuses_and_publishes_nothing(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    source_url = "https://github.com/example/source.tar.gz"
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="redirect target changed"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(
                content,
                [],
                {source_url: "https://codeload.github.com/example/changed.tar.gz"},
            ),
            installer=_installer,
        )
    assert not (tmp_path / "cache/current").exists()


def test_missing_anonymous_access_refuses_without_publishing(tmp_path: Path) -> None:
    manifest, requirements, _ = _write_inputs(tmp_path)
    calls: list[str] = []

    def denied(url: str):
        calls.append(url)
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="access failed"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=denied,
            installer=_installer,
        )
    assert len(calls) == 1
    assert not (tmp_path / "cache/current").exists()
    assert list((tmp_path / "cache/versions").iterdir()) == []


def test_consent_or_credential_field_refuses_before_fetch(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(
        tmp_path, mutate={"accept_terms": True}
    )
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="consent proxy"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_source_archive({"../escape": b"bad"}), "unsafe archive member"),
        (_source_archive(symlink=("link", "/etc/passwd")), "link or special file"),
        (_source_archive({"a": b"x" * 100}), "expansion limit"),
    ],
)
def test_adversarial_source_archive_refuses_without_escape(
    tmp_path: Path, source: bytes, message: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path, source=source)
    if message == "expansion limit":
        payload = json.loads(manifest.read_text())
        payload["artifacts"][0]["max_unpacked_bytes"] = 10
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match=message):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, []),
            installer=_installer,
        )
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "cache/current").exists()


def test_malformed_wheel_refuses_before_install_or_publish(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path, wheel=b"not-a-wheel")
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="wheel is malformed"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, []),
            installer=lambda *_: pytest.fail("malformed wheel reached installer"),
        )
    assert not (tmp_path / "cache/current").exists()


def test_wheel_stream_expansion_limit_refuses_before_install(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"][1]["max_unpacked_bytes"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="expansion limit"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, []),
            installer=lambda *_: pytest.fail("oversized wheel reached installer"),
        )
    assert not (tmp_path / "cache/current").exists()


def test_unowned_or_unsafe_cache_root_refuses(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o777)
    cache.chmod(0o777)
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="group/world writable"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=_installer,
        )
    assert not (cache / "current").exists()


def test_symlinked_cache_root_refuses_before_fetch(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    cache = tmp_path / "cache"
    cache.symlink_to(target, target_is_directory=True)
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="symlink"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


def test_no_consent_proxy_or_implicit_credential_path_exists() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "accept_eula",
        "accept_terms",
        "privacy_consent",
        "authorization:",
        "bearer ",
        "netrc",
    ):
        assert forbidden not in source
    assert "Runtime fetch changes delivery only" in SCRIPT.read_text(encoding="utf-8")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["decision_sha256"] == (
        "758a29a6fae55075dc4ba879907e81f949b7a4e23fa726b790fd4361241697a3"
    )
