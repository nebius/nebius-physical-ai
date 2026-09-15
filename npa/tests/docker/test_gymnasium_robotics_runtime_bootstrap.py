from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import BinaryIO, Iterator
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


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Create test caches below a non-writable owner-controlled parent chain."""

    trusted_tmp = ROOT.parent / "test-tmp"
    trusted_tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-bootstrap-", dir=trusted_tmp) as raw:
        yield Path(raw)


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


def _python_installer(stage: Path, requirements: Path) -> None:
    _installer(stage, requirements)
    python = stage / "runtime/bin/python"
    shutil.copyfile(sys.executable, python)
    python.chmod(0o700)
    base_executable = Path(sys._base_executable).resolve()
    (stage / "runtime/pyvenv.cfg").write_text(
        f"home = {base_executable.parent}\n"
        "include-system-site-packages = false\n"
        f"version = {sys.version_info.major}.{sys.version_info.minor}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("prepare", "runtime preparation"),
        ("exec", "runtime run-smoke execution"),
    ],
)
def test_cli_runtime_paths_refuse_uid_zero_before_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    message: str,
) -> None:
    monkeypatch.setattr(BOOTSTRAP.os, "geteuid", lambda: 0)
    arguments = [
        "--manifest",
        str(tmp_path / "missing-lock"),
        "--requirements",
        str(tmp_path / "missing-requirements"),
        "--cache-root",
        str(tmp_path / "cache"),
        command,
    ]
    if command == "exec":
        arguments.extend(["--", "missing-script"])
    assert BOOTSTRAP.main(arguments) == 65
    assert message in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()


def test_library_prepare_refuses_uid_zero_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BOOTSTRAP.os, "geteuid", lambda: 0)
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="non-root runtime user"):
        BOOTSTRAP.prepare(
            tmp_path / "missing-lock",
            tmp_path / "missing-requirements",
            tmp_path / "cache",
            opener=_opener({}, calls),
            installer=_installer,
        )
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_repository_lock_refuses_before_any_network_access(tmp_path: Path) -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["status"] = "incomplete"
    manifest = tmp_path / "source-lock.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="before network access"):
        BOOTSTRAP.prepare(
            manifest,
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
    assert not any(path.name.startswith(".") for path in (cache / "versions").iterdir())
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


def test_cache_root_replacement_during_lock_never_redirects_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    displaced = tmp_path / "cache-displaced"
    original_open = BOOTSTRAP.os.open
    swapped = False

    def replacing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(path).name == ".bootstrap.lock" and not swapped:
            swapped = True
            cache.rename(displaced)
            cache.mkdir(mode=0o700)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(BOOTSTRAP.os, "open", replacing_open)
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="cache root identity changed"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert swapped is True
    assert calls == []
    assert list(cache.iterdir()) == []


def test_versions_replacement_cannot_supply_a_reused_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    target_name = Path(str(first["runtime_root"])).name
    versions = cache / "versions"
    displaced = cache / "versions-displaced"
    attacker_versions = tmp_path / "attacker-versions"
    shutil.copytree(versions, attacker_versions)
    (cache / "current").unlink()
    original_validate = BOOTSTRAP._validated_existing
    observed: list[Path] = []

    def replacing_validate(target: Path, runtime_lock: object) -> dict[str, object]:
        versions.rename(displaced)
        attacker_versions.rename(versions)
        observed.append(target.resolve())
        return original_validate(target, runtime_lock)

    monkeypatch.setattr(BOOTSTRAP, "_validated_existing", replacing_validate)
    with pytest.raises(
        BOOTSTRAP.BootstrapRefusal, match="versions directory identity changed"
    ):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener({}, []),
            installer=lambda *_: pytest.fail("a replacement cache must not install"),
        )
    assert observed == [displaced / target_name]
    assert not (cache / "current").exists()


def test_versions_replacement_before_staging_never_redirects_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    versions = cache / "versions"
    displaced = cache / "versions-displaced"
    original_mkdtemp = BOOTSTRAP.tempfile.mkdtemp
    swapped = False

    def replacing_mkdtemp(*args: object, **kwargs: object) -> str:
        nonlocal swapped
        if kwargs.get("prefix") == ".staging-runtime-" and not swapped:
            swapped = True
            versions.rename(displaced)
            versions.mkdir(mode=0o700)
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(BOOTSTRAP.tempfile, "mkdtemp", replacing_mkdtemp)
    calls: list[str] = []
    with pytest.raises(
        BOOTSTRAP.BootstrapRefusal, match="versions directory identity changed"
    ):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert swapped is True
    assert calls == []
    assert list(versions.iterdir()) == []
    assert list(displaced.iterdir()) == []


def test_versions_replacement_during_publish_never_redirects_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    versions = cache / "versions"
    displaced = cache / "versions-displaced"
    runtime_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original_replace = BOOTSTRAP.os.replace
    swapped = False

    def replacing_replace(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if Path(destination).name == runtime_digest and not swapped:
            swapped = True
            versions.rename(displaced)
            versions.mkdir(mode=0o700)
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(BOOTSTRAP.os, "replace", replacing_replace)
    with pytest.raises(
        BOOTSTRAP.BootstrapRefusal, match="versions directory identity changed"
    ):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=_installer,
        )
    assert swapped is True
    assert list(versions.iterdir()) == []
    assert list(displaced.iterdir()) == []


def test_cache_root_replacement_during_current_link_never_redirects_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    displaced = tmp_path / "cache-displaced"
    original_symlink = BOOTSTRAP.os.symlink
    swapped = False

    def replacing_symlink(
        source: object,
        destination: object,
        target_is_directory: bool = False,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if Path(destination).name.startswith(".current-") and not swapped:
            swapped = True
            cache.rename(displaced)
            cache.mkdir(mode=0o700)
        original_symlink(
            source,
            destination,
            target_is_directory,
            dir_fd=dir_fd,
        )

    monkeypatch.setattr(BOOTSTRAP.os, "symlink", replacing_symlink)
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="cache root identity changed"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=_installer,
        )
    assert swapped is True
    assert list(cache.iterdir()) == []


def test_stage_cleanup_stays_below_replaced_versions_descriptor(
    tmp_path: Path,
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    versions = cache / "versions"
    displaced = cache / "versions-displaced"
    decoy_marker: Path | None = None
    stage_name = ""

    def replacing_installer(stage: Path, locked_requirements: Path) -> None:
        nonlocal decoy_marker, stage_name
        _installer(stage, locked_requirements)
        stage_name = stage.name
        versions.rename(displaced)
        versions.mkdir(mode=0o700)
        decoy = versions / stage_name
        decoy.mkdir(mode=0o700)
        decoy_marker = decoy / "must-remain"
        decoy_marker.write_text("replacement tree", encoding="utf-8")
        raise BOOTSTRAP.BootstrapRefusal("injected installer failure")

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="injected installer failure"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=replacing_installer,
        )
    assert decoy_marker is not None and decoy_marker.read_text() == "replacement tree"
    assert not (displaced / stage_name).exists()


def test_installer_receives_the_once_read_exact_requirements_bytes(
    tmp_path: Path,
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    expected = requirements.read_bytes()

    def exact_installer(stage: Path, staged_requirements: Path) -> None:
        assert staged_requirements.read_bytes() == expected
        assert stat.S_IMODE(staged_requirements.stat().st_mode) == 0o400
        _installer(stage, staged_requirements)

    BOOTSTRAP.prepare(
        manifest,
        requirements,
        tmp_path / "cache",
        opener=_opener(content, []),
        installer=exact_installer,
    )


def test_requirements_path_replacement_refuses_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    original_open = BOOTSTRAP.os.open
    displaced = tmp_path / "requirements.displaced"
    swapped = False

    def replacing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(path) == requirements and not swapped:
            swapped = True
            requirements.rename(displaced)
            requirements.write_bytes(displaced.read_bytes())
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(BOOTSTRAP.os, "open", replacing_open)
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="trusted stable"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_symlinked_requirements_refuses_before_fetch(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    target = tmp_path / "requirements.target"
    requirements.rename(target)
    requirements.symlink_to(target)
    calls: list[str] = []

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="unsafe or unavailable"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            tmp_path / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []
    assert not (tmp_path / "cache").exists()


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
    assert not any(path.name.startswith(".") for path in (cache / "versions").iterdir())


def test_staging_cleanup_refuses_when_exact_target_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    (stage / "partial").write_bytes(b"runtime payload")
    original_rmtree = BOOTSTRAP.shutil.rmtree

    def leave_exact_stage(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path) == stage:
            return
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(BOOTSTRAP.shutil, "rmtree", leave_exact_stage)
    try:
        with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="target remains"):
            BOOTSTRAP._discard_stage(stage)
        assert stage.is_dir()
    finally:
        monkeypatch.setattr(BOOTSTRAP.shutil, "rmtree", original_rmtree)


def test_staging_cleanup_verifies_exact_target_absent(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    (stage / "partial").write_bytes(b"runtime payload")
    BOOTSTRAP._discard_stage(stage)
    assert not stage.exists()


def test_failed_post_publish_validation_removes_only_the_exact_sealed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    observed_modes: list[int] = []

    def reject_published(target: Path, runtime_lock: object) -> dict[str, object]:
        del runtime_lock
        observed_modes.append(stat.S_IMODE(target.lstat().st_mode))
        raise BOOTSTRAP.BootstrapRefusal("injected post-publish validation failure")

    monkeypatch.setattr(BOOTSTRAP, "_validated_existing", reject_published)
    with pytest.raises(
        BOOTSTRAP.BootstrapRefusal, match="post-publish validation failure"
    ):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            cache,
            opener=_opener(content, []),
            installer=_installer,
        )
    assert observed_modes == [0o500]
    assert not (cache / "current").exists()
    assert list((cache / "versions").iterdir()) == []
    assert not any(path.name.startswith(".") for path in (cache / "versions").iterdir())


def test_exec_handles_remain_bound_when_validated_target_path_is_swapped(
    tmp_path: Path,
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    cache = tmp_path / "cache"
    result = BOOTSTRAP.prepare(
        manifest,
        requirements,
        cache,
        opener=_opener(content, []),
        installer=_installer,
        retain_runtime_handles=True,
    )
    target = Path(str(result["runtime_root"]))
    directory_fd = int(result["_runtime_directory_fd"])
    python_fd = int(result["_runtime_python_fd"])
    monitor_fd = int(result["_runtime_monitor_fd"])
    displaced = target.with_name("displaced")
    target.rename(displaced)
    target.mkdir(mode=0o700)
    replacement = target / "runtime/bin"
    replacement.mkdir(parents=True)
    (replacement / "python").write_text("malicious replacement", encoding="utf-8")
    try:
        bound_root = Path(f"/proc/self/fd/{directory_fd}")
        assert bound_root.resolve() == displaced.resolve()
        assert (bound_root / "runtime/bin/python").read_text() == "#!/bin/sh\nexit 0\n"
        assert os.read(python_fd, 64) == b"#!/bin/sh\nexit 0\n"
        assert (replacement / "python").read_text() == "malicious replacement"
    finally:
        os.close(monitor_fd)
        os.close(python_fd)
        os.close(directory_fd)
        BOOTSTRAP._discard_stage(displaced)
        BOOTSTRAP._discard_stage(target)


def test_exec_path_uses_descriptor_bound_interpreter_and_runtime_root() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'bound_root = Path(f"/proc/self/fd/{directory_fd}")' in source
    assert 'f"/proc/self/fd/{directory_fd}"' in source
    assert (
        'python_name = f"/proc/self/fd/{directory_fd}/runtime/bin/python"' in source
    )
    assert "os.execve(" in source
    assert "python_fd," in source
    assert 'result["_runtime_monitor_fd"]' in source
    assert "_wait_for_runtime_child" in source


def test_descriptor_bound_runtime_executes_unchanged_tree(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    result = BOOTSTRAP.prepare(
        manifest,
        requirements,
        tmp_path / "cache",
        opener=_opener(content, []),
        installer=_python_installer,
        retain_runtime_handles=True,
    )
    script = tmp_path / "exit-cleanly.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "expected = Path(os.environ['NPA_GYMNASIUM_RUNTIME_ROOT']) / 'runtime'\n"
        "assert Path(sys.prefix).samefile(expected)\n",
        encoding="utf-8",
    )
    status = BOOTSTRAP._execute_validated_runtime(
        int(result["_runtime_directory_fd"]),
        int(result["_runtime_python_fd"]),
        int(result["_runtime_monitor_fd"]),
        [str(script)],
    )
    assert status == 0


def _execute_script_from_runtime(tmp_path: Path, source: str) -> int:
    manifest, requirements, content = _write_inputs(tmp_path)
    result = BOOTSTRAP.prepare(
        manifest,
        requirements,
        tmp_path / "cache",
        opener=_opener(content, []),
        installer=_python_installer,
        retain_runtime_handles=True,
    )
    script = tmp_path / "runtime-probe.py"
    script.write_text(source, encoding="utf-8")
    return BOOTSTRAP._execute_validated_runtime(
        int(result["_runtime_directory_fd"]),
        int(result["_runtime_python_fd"]),
        int(result["_runtime_monitor_fd"]),
        [str(script)],
    )


def test_runtime_cannot_read_credential_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "parent-only-v75-secret")
    status = _execute_script_from_runtime(
        tmp_path,
        "import os\n"
        "from pathlib import Path\n"
        "assert 'AWS_SECRET_ACCESS_KEY' not in os.environ\n"
        "try:\n"
        "    Path(f'/proc/{os.getppid()}/environ').read_bytes()\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(71)\n",
    )
    assert status == 0


def test_runtime_cannot_open_metadata_network_or_elevate_with_sudo(
    tmp_path: Path,
) -> None:
    status = _execute_script_from_runtime(
        tmp_path,
        "import os\n"
        "import shutil\n"
        "import socket\n"
        "import subprocess\n"
        "try:\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(72)\n"
        "assert 'NoNewPrivs:\\t1' in open('/proc/self/status').read()\n"
        "sudo = shutil.which('sudo')\n"
        "assert sudo is not None\n"
        "assert subprocess.run([sudo, '-n', 'true']).returncode != 0\n",
    )
    assert status == 0


def test_runtime_refuses_and_kills_surviving_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "survivor.pid"
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="surviving descendant"):
        _execute_script_from_runtime(
            tmp_path,
            "import os\n"
            "import time\n"
            f"pid_path = {str(pid_path)!r}\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "open(pid_path, 'w').write(str(child))\n",
        )
    survivor = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(survivor, 0)


@pytest.mark.parametrize("mutation", ["content", "replacement"])
def test_descriptor_bound_execution_refuses_same_uid_descendant_race(
    tmp_path: Path, mutation: str
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    result = BOOTSTRAP.prepare(
        manifest,
        requirements,
        tmp_path / "cache",
        opener=_opener(content, []),
        installer=_python_installer,
        retain_runtime_handles=True,
    )
    script = tmp_path / "mutate-runtime.py"
    operation = (
        "target.chmod(0o600); target.write_text('changed')"
        if mutation == "content"
        else "target.parent.chmod(0o700); target.replace(target.with_suffix('.old'))"
    )
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "target = Path(os.environ['NPA_GYMNASIUM_RUNTIME_ROOT']) / "
        "'source/pyproject.toml'\n"
        f"{operation}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        BOOTSTRAP.BootstrapRefusal, match="changed during descriptor-bound execution"
    ):
        BOOTSTRAP._execute_validated_runtime(
            int(result["_runtime_directory_fd"]),
            int(result["_runtime_python_fd"]),
            int(result["_runtime_monitor_fd"]),
            [str(script)],
        )


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


def test_only_standard_venv_compatibility_link_is_removed(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "lib").mkdir()
    link = runtime / "lib64"
    link.symlink_to("lib")
    BOOTSTRAP._remove_venv_compatibility_link(runtime)
    assert not link.exists()

    link.symlink_to(tmp_path)
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="unexpected compatibility"):
        BOOTSTRAP._remove_venv_compatibility_link(runtime)


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


def test_group_writable_cache_ancestor_refuses_before_fetch(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    ancestor = tmp_path / "unsafe-parent"
    ancestor.mkdir(mode=0o700)
    ancestor.chmod(0o770)
    calls: list[str] = []

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="group/world writable"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            ancestor / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


def test_symlinked_cache_ancestor_refuses_before_fetch(tmp_path: Path) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    target = tmp_path / "trusted-target"
    target.mkdir(mode=0o700)
    ancestor = tmp_path / "linked-parent"
    ancestor.symlink_to(target, target_is_directory=True)
    calls: list[str] = []

    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="symlink"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            ancestor / "cache",
            opener=_opener(content, calls),
            installer=_installer,
        )
    assert calls == []


def test_untrusted_cache_ancestor_owner_refuses_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, requirements, content = _write_inputs(tmp_path)
    ancestor = tmp_path / "untrusted-owner"
    ancestor.mkdir(mode=0o700)
    real_fstat = BOOTSTRAP.os.fstat
    untrusted_uid = os.geteuid() + 1

    def spoofed_fstat(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        try:
            opened_path = Path(f"/proc/self/fd/{descriptor}").resolve()
        except FileNotFoundError:
            return metadata
        if opened_path != ancestor:
            return metadata
        fields = list(metadata)
        fields[4] = untrusted_uid
        return os.stat_result(fields)

    monkeypatch.setattr(BOOTSTRAP.os, "fstat", spoofed_fstat)
    calls: list[str] = []
    with pytest.raises(BOOTSTRAP.BootstrapRefusal, match="untrusted owner"):
        BOOTSTRAP.prepare(
            manifest,
            requirements,
            ancestor / "cache",
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
