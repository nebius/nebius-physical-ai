"""Native delivery rejects untrusted bytes and paths before privileged installation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import threading
from types import SimpleNamespace

import pytest


PACKAGING = Path(__file__).parents[2] / "docker/workbench/ncore"


@pytest.fixture
def native(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ncore_native_test", PACKAGING / "native_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Exercise the real filesystem checks as the test user in disposable paths.
    # Production has fixed uid/gid zero; there is no CLI/env destination override.
    monkeypatch.setattr(module, "ROOT_UID", os.geteuid())
    monkeypatch.setattr(module, "ROOT_GID", os.getegid())
    return module


def _ar(members):
    result = b"!<arch>\n"
    for name, raw in members:
        header = f"{name:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(raw):<10}`\n"
        assert len(header) == 60
        result += header.encode() + raw + (b"\n" if len(raw) % 2 else b"")
    return result


def _entries(package):
    result = []
    for item in package["files"]:
        entry = tarfile.TarInfo("./" + item["path"])
        entry.mode = item["mode"]
        if "link" in item:
            entry.type = tarfile.SYMTYPE
            entry.linkname = item["link"]
            raw = b""
        else:
            raw = b"\x7fELFsynthetic-library-" + package["name"].encode()
            entry.size = len(raw)
            item.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        result.append((entry, raw))
    return result


def _deb(entries):
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output, mode="w:xz", format=tarfile.USTAR_FORMAT
    ) as archive:
        for entry, raw in entries:
            archive.addfile(entry, io.BytesIO(raw))
    return _ar(
        [
            ("debian-binary", b"2.0\n"),
            ("control.tar.xz", b"unexecuted maintainer scripts"),
            ("data.tar.xz", output.getvalue()),
        ]
    )


def _pin(package, raw):
    package.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


@pytest.fixture
def delivery(native, tmp_path, monkeypatch):
    lock = json.loads((PACKAGING / "native-bootstrap-lock.json").read_text())
    artifacts = [_pin(p, _deb(_entries(p))) for p in lock["debian_binaries"]]
    cache = tmp_path / "cache"
    calls = []

    def download(package):
        calls.append(package["name"])
        return artifacts[lock["debian_binaries"].index(package)]

    monkeypatch.setattr(native, "_download", download)
    return native, lock, artifacts, cache, calls


@pytest.fixture
def installation(delivery, tmp_path, monkeypatch):
    native, lock, artifacts, cache, calls = delivery
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    monkeypatch.setattr(native, "LIBRARY_DIRECTORY", libraries)
    monkeypatch.setattr(native, "INSTALL_STATE", tmp_path / "state")
    selected = native._read_input(lock, io.BytesIO(b"".join(artifacts)))
    return native, lock, selected, libraries


def test_native_lock_pins_exact_original_selection(native):
    raw = (PACKAGING / "native-bootstrap-lock.json").read_bytes()
    assert native.LOCK_SHA256 == hashlib.sha256(raw).hexdigest()
    lock = json.loads(raw)
    native._validate_selection(lock)
    packages = lock["debian_binaries"]
    assert [(p["name"], p["version"], p["sha256"]) for p in packages] == [
        (
            "libgnutls30",
            "3.7.9-2+deb12u7",
            "30abec8c824feb1d2d7e9000a34083cccd19d139625e1b21547e3ac53b922f8e",
        ),
        (
            "libssh2-1",
            "1.10.0-3+deb12u1",
            "fff72a194e493f88e100a2567e22472bb4ab828d429c2956965c6f2f134f1b3a",
        ),
    ]
    assert [f["sha256"] for p in packages for f in p["files"] if "sha256" in f] == [
        "779b25d20249988bea2c1aa6bbeb218f5ae7ea8a9d30ce4f54ea37372965cc4b",
        "e481655791a9b75f4d5957e40101d7d0b5d9c13a18d1ca233731d03365ad0aec",
    ]
    base = json.loads((PACKAGING / "base-source-lock.json").read_text())
    assert not set(native.SELECTION) & {p["name"] for p in base["debian_binaries"]}
    # Provenance retains the original signed input identities. The source lock
    # additionally classifies their delivery outside the published source annex.
    assert [
        {**item, "delivery": "build-only"}
        for item in lock["provenance"]["metadata"]
    ] == [
        a for a in base["artifacts"] if a["path"].startswith("metadata/")
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "etc/sudoers"),
        ("path", "usr/lib/x86_64-linux-gnu/../escape"),
        ("link", "../../escape"),
        ("mode", 0o4777),
    ],
)
def test_selection_cannot_redirect_installation(native, field, value):
    lock = json.loads((PACKAGING / "native-bootstrap-lock.json").read_text())
    lock["debian_binaries"][0]["files"][0][field] = value
    with pytest.raises(ValueError):
        native._validate_selection(lock)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: b"invalid!" + raw[8:],
        lambda raw: raw[:-1],
        lambda raw: raw + b"trailer",
        lambda raw: raw.replace(b"control.tar.xz", b"debian-binary ", 1),
        lambda raw: raw[:56] + b"-1        " + raw[66:],
        lambda raw: raw[:56] + b"9999999999" + raw[66:],
        lambda raw: raw.replace(b"2.0\n", b"3.0\n", 1),
    ],
)
def test_malformed_ar_fails_even_after_artifact_repin(delivery, mutation):
    native, lock, artifacts, _, _ = delivery
    package = lock["debian_binaries"][0]
    raw = _pin(package, mutation(artifacts[0]))
    with pytest.raises(ValueError):
        native._select_libraries(package, raw)


def test_ar_rejects_padding_and_long_name_extensions(native):
    raw = _ar(
        [("debian-binary", b"2.0\n"), ("control.tar.xz", b"x"), ("data.tar.xz", b"y")]
    )
    assert native._ar_members(raw)["data.tar.xz"] == b"y"
    with pytest.raises(ValueError, match="padding"):
        native._ar_members(raw[:-1] + b"x")
    with pytest.raises(ValueError, match="member/order"):
        native._ar_members(raw.replace(b"control.tar.xz  ", b"#1/13           "))


@pytest.mark.parametrize(
    "name,kind",
    [
        ("/absolute", tarfile.REGTYPE),
        ("../escape", tarfile.REGTYPE),
        ("./usr/../escape", tarfile.REGTYPE),
        ("./usr//escape", tarfile.REGTYPE),
        ("./fifo", tarfile.FIFOTYPE),
        ("./device", tarfile.CHRTYPE),
        ("./hardlink", tarfile.LNKTYPE),
    ],
)
def test_unsafe_unselected_tar_members_are_rejected(delivery, name, kind):
    native, lock, _, _, _ = delivery
    package = lock["debian_binaries"][0]
    entries = _entries(package)
    bad = tarfile.TarInfo(name)
    bad.type = kind
    entries.append((bad, b""))
    raw = _pin(package, _deb(entries))
    with pytest.raises(ValueError, match="unsafe"):
        native._select_libraries(package, raw)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "mode", "link", "bytes", "size"]
)
def test_selected_tar_members_must_match_completely(delivery, mutation):
    native, lock, _, _, _ = delivery
    package = lock["debian_binaries"][0]
    entries = _entries(package)
    if mutation == "duplicate":
        entries.append(entries[1])
    elif mutation == "missing":
        entries.pop()
    elif mutation == "mode":
        entries[1][0].mode = 0o755
    elif mutation == "link":
        entries[0][0].linkname = "libwrong.so"
    elif mutation == "bytes":
        entries[1] = (entries[1][0], b"x" * len(entries[1][1]))
    else:
        package["files"][1]["size"] += 1
    raw = _pin(package, _deb(entries))
    with pytest.raises(ValueError):
        native._select_libraries(package, raw)


def test_whole_deb_hash_precedes_archive_parser(delivery, monkeypatch):
    native, lock, artifacts, _, _ = delivery
    monkeypatch.setattr(native, "_ar_members", lambda _: pytest.fail("parser reached"))
    with pytest.raises(ValueError, match="artifact size/SHA256"):
        native._select_libraries(lock["debian_binaries"][0], artifacts[0][:-1])


def test_cold_and_warm_cache_revalidate_complete_artifacts(delivery):
    native, lock, artifacts, cache, calls = delivery
    assert native._cached_artifacts(lock, cache) == artifacts
    assert native._cached_artifacts(lock, cache) == artifacts
    assert len(calls) == 2
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in cache.iterdir())
    target = cache / (lock["debian_binaries"][0]["sha256"] + ".deb")
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        native._cached_artifacts(lock, cache)
    assert len(calls) == 2


def test_failed_download_never_publishes_and_next_start_can_retry(
    delivery, monkeypatch
):
    native, lock, artifacts, cache, calls = delivery
    download = native._download
    monkeypatch.setattr(native, "_download", lambda _: b"interrupted")
    with pytest.raises(ValueError, match="SHA256"):
        native._cached_artifacts(lock, cache)
    assert list(cache.iterdir()) == [cache / "bootstrap.lock"]
    monkeypatch.setattr(native, "_download", download)
    assert native._cached_artifacts(lock, cache) == artifacts


@pytest.mark.parametrize("target", ["bootstrap.lock", "artifact"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "mode"])
def test_cache_rejects_unsafe_files_before_reading(delivery, target, kind):
    native, lock, _, cache, calls = delivery
    native._cached_artifacts(lock, cache)
    name = (
        "bootstrap.lock"
        if target == "bootstrap.lock"
        else lock["debian_binaries"][0]["sha256"] + ".deb"
    )
    path = cache / name
    path.unlink()
    outside = cache.parent / "untouched"
    outside.write_bytes(b"outside")
    outside.chmod(0o600)
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.write_bytes(b"")
        path.chmod(0o644)
    with pytest.raises((ValueError, OSError)):
        native._cached_artifacts(lock, cache)
    assert outside.read_bytes() == b"outside" and len(calls) == 2


@pytest.mark.parametrize("kind", ["symlink", "public", "writable-parent", "relative"])
def test_cache_path_rejects_unsafe_directories(delivery, kind):
    native, lock, _, cache, calls = delivery
    if kind == "symlink":
        outside = cache.parent / "outside"
        outside.mkdir(mode=0o700)
        cache.symlink_to(outside)
    elif kind == "public":
        cache.mkdir(mode=0o755)
        cache.chmod(0o755)
    elif kind == "writable-parent":
        cache.parent.chmod(0o777)
    else:
        cache = Path("relative-cache")
    with pytest.raises((ValueError, OSError)):
        native._cached_artifacts(lock, cache)
    assert calls == []


def test_cache_flock_serializes_concurrent_downloaders(delivery):
    native, lock, artifacts, cache, calls = delivery
    barrier = threading.Barrier(4)

    def ensure(_):
        barrier.wait()
        return native._cached_artifacts(lock, cache)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(ensure, range(4))) == [artifacts] * 4
    assert len(calls) == 2


def test_root_input_is_verified_in_full_before_installation(delivery, monkeypatch):
    native, lock, artifacts, _, _ = delivery
    monkeypatch.setattr(native, "_check_root_entrypoint", lambda: None)
    monkeypatch.setattr(native, "_read_lock", lambda: lock)
    monkeypatch.setattr(
        native, "_install_selected", lambda *_: pytest.fail("install reached")
    )
    monkeypatch.setattr(sys, "argv", ["native_bootstrap.py", "install"])
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(buffer=io.BytesIO(artifacts[0] + b"corrupt"))
    )
    with pytest.raises(ValueError, match="SHA256"):
        native._main()
    with pytest.raises(ValueError, match="trailer"):
        native._read_input(lock, io.BytesIO(b"".join(artifacts) + b"trailer"))


def test_cache_change_after_verified_read_cannot_change_root_input(delivery):
    native, lock, artifacts, cache, _ = delivery
    verified = native._cached_artifacts(lock, cache)
    for path in cache.glob("*.deb"):
        path.write_bytes(b"changed after read")
    assert native._read_input(
        lock, io.BytesIO(b"".join(verified))
    ) == native._read_input(lock, io.BytesIO(b"".join(artifacts)))


def test_installation_is_idempotent_and_selects_only_four_paths(installation):
    native, lock, selected, libraries = installation
    native._install_selected(lock, selected)
    native._install_selected(lock, selected)
    assert {p.name for p in libraries.iterdir()} == {Path(p).name for p in selected}
    with native._directory(libraries, os.geteuid()) as directory:
        assert all(
            native._installed(directory, item)
            for p in lock["debian_binaries"]
            for item in p["files"]
        )


@pytest.mark.parametrize(
    "kind", ["bytes", "mode", "symlink", "hardlink", "fifo", "soname"]
)
def test_installed_tampering_fails_closed(installation, kind):
    native, lock, selected, libraries = installation
    native._install_selected(lock, selected)
    item = lock["debian_binaries"][0]["files"][1]
    path = libraries / Path(item["path"]).name
    if kind == "mode":
        path.chmod(0o666)
    elif kind == "bytes":
        path.write_bytes(b"corrupt")
    elif kind == "soname":
        path = libraries / Path(lock["debian_binaries"][0]["files"][0]["path"]).name
        path.unlink()
        path.symlink_to("../../escape")
    else:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(libraries.parent / "outside")
        elif kind == "hardlink":
            outside = libraries.parent / "outside"
            outside.write_bytes(selected[item["path"]])
            os.link(outside, path)
        else:
            os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        native._install_selected(lock, selected)


def test_selected_bytes_rechecked_before_root_write(installation):
    native, lock, selected, libraries = installation
    first = next(p for p, raw in selected.items() if raw)
    selected[first] = b"tampered"
    with pytest.raises(ValueError, match="selected bytes"):
        native._install_selected(lock, selected)
    assert list(libraries.iterdir()) == []


def test_staged_corruption_fails_before_atomic_install(installation, monkeypatch):
    native, lock, selected, libraries = installation
    read = native._read_file

    def tamper(directory, name, *args):
        if name.startswith(".native-"):
            (libraries / name).write_bytes(b"corrupted staging")
        return read(directory, name, *args)

    monkeypatch.setattr(native, "_read_file", tamper)
    with pytest.raises(ValueError, match="staged bytes"):
        native._install_selected(lock, selected)
    assert list(libraries.iterdir()) == []


def test_interrupted_installation_resumes_verified_files(installation, monkeypatch):
    native, lock, selected, libraries = installation
    write = native._atomic_file
    calls = []

    def interrupt(*args):
        calls.append(args[1])
        if len(calls) == 2:
            raise OSError("interrupted")
        write(*args)

    monkeypatch.setattr(native, "_atomic_file", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        native._install_selected(lock, selected)
    assert len(list(libraries.iterdir())) == 1
    monkeypatch.setattr(native, "_atomic_file", write)
    native._install_selected(lock, selected)
    assert len(list(libraries.iterdir())) == 4


def test_root_flock_serializes_installers(installation):
    native, lock, selected, libraries = installation
    barrier = threading.Barrier(4)

    def install(_):
        barrier.wait()
        native._install_selected(lock, selected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(install, range(4)))
    assert len(list(libraries.iterdir())) == 4


@pytest.mark.parametrize(
    "command", [[], ["install", "--root"], ["ensure", "extra"]]
)
def test_no_cli_path_or_code_overrides(native, monkeypatch, tmp_path, command):
    if "--root" in command:
        command = [*command, str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["native_bootstrap.py", *command])
    with pytest.raises(ValueError, match="expected exactly"):
        native._main()


def test_sudo_uses_fixed_isolated_python_and_stdin(delivery, monkeypatch):
    native, lock, artifacts, cache, _ = delivery
    monkeypatch.setenv("NPA_NCORE_NATIVE_CACHE", str(cache))
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted")
    calls = []
    monkeypatch.setattr(
        native.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    native._ensure(lock)
    assert calls == [
        (
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                "/usr/local/bin/python3.12",
                "-I",
                "-S",
                "-B",
                "/opt/ncore/native/native_bootstrap.py",
                "install",
            ],
            {"input": b"".join(artifacts), "check": True},
        )
    ]


@pytest.mark.parametrize("corrupt", [False, True])
def test_read_lock_requires_pinned_root_owned_bytes(
    native, tmp_path, monkeypatch, corrupt
):
    path = tmp_path / "lock.json"
    raw = (PACKAGING / "native-bootstrap-lock.json").read_bytes()
    path.write_bytes(raw + (b" " if corrupt else b""))
    path.chmod(0o644)
    monkeypatch.setattr(native, "LOCK_PATH", path)
    if corrupt:
        with pytest.raises(ValueError, match="lock SHA256"):
            native._read_lock()
    else:
        assert native._read_lock() == json.loads(raw)
        path.chmod(0o666)
        with pytest.raises(ValueError, match="unsafe native file"):
            native._read_lock()


def test_download_uses_existing_https_transport_and_locked_size(native, monkeypatch):
    calls = []

    def download(url, output, **policy):
        calls.append((url, policy))
        output.write(b"123")

    monkeypatch.setitem(
        sys.modules, "_public_https", SimpleNamespace(download_public_https=download)
    )
    package = {"url": "https://snapshot.debian.org/pinned.deb", "size": 3}
    assert native._download(package) == b"123"
    assert calls == [
        (package["url"], {"allowed_hosts": frozenset({"snapshot.debian.org"})})
    ]
    with pytest.raises(ValueError, match="locked artifact size"):
        native._download({**package, "size": 2})


@pytest.mark.parametrize("status", [0, 7])
def test_bash_env_runs_before_overridden_entrypoint_and_fails_closed(tmp_path, status):
    probe = tmp_path / "probe"
    probe.write_text(f"#!/bin/sh\nprintf 'native\\n'\nexit {status}\n")
    probe.chmod(0o755)
    hook = tmp_path / "hook"
    hook.write_text(
        (PACKAGING / "native-bootstrap.sh")
        .read_text()
        .replace("/usr/local/bin/python3.12", str(probe))
    )
    result = subprocess.run(
        ["/bin/bash", "-c", "printf 'apt bootstrap\\n'"],
        env={"BASH_ENV": str(hook)},
        capture_output=True,
        text=True,
    )
    assert result.stdout == ("native\napt bootstrap\n" if status == 0 else "native\n")
    assert result.returncode == (0 if status == 0 else 1)


def test_only_final_stage_enables_hook_and_build_never_fetches_native_bytes():
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    builder, final = dockerfile.split("FROM scratch AS public-image", 1)
    assert "BASH_ENV" not in builder
    assert "BASH_ENV=/opt/ncore/bin/native-bootstrap.sh" in final
    assert "USER ubuntu" in final
    assert "native_bootstrap.py ensure" not in builder
    assert "native_bootstrap.py verify-lock" in builder
    assert "/opt/ncore/bin/entrypoint.sh /bin/true" not in builder
    assert (
        "source /opt/ncore/bin/native-bootstrap.sh"
        in (PACKAGING / "entrypoint.sh").read_text()
    )
    assert builder.index("--root /public-root") < builder.index(
        "test ! -e /public-root/usr/lib/x86_64-linux-gnu/libgnutls"
    )
    lock = json.loads((PACKAGING / "native-bootstrap-lock.json").read_text())
    for package in lock["debian_binaries"]:
        for item in package["files"]:
            assert "test ! -e /public-root/" + item["path"] in builder
            assert "test ! -L /public-root/" + item["path"] in builder


@pytest.mark.parametrize(
    "field,value",
    [
        ("st_uid", 12345),
        ("st_gid", 12345),
        ("st_nlink", 2),
        ("st_mode", stat.S_IFREG | 0o666),
        ("st_mode", stat.S_IFIFO | 0o600),
    ],
)
def test_file_metadata_rejects_foreign_owner_group_and_special_types(
    native, field, value
):
    fields = dict(
        st_uid=native.ROOT_UID,
        st_gid=native.ROOT_GID,
        st_nlink=1,
        st_mode=stat.S_IFREG | 0o600,
    )
    fields[field] = value
    with pytest.raises(ValueError, match="unsafe native file"):
        native._check_file(SimpleNamespace(**fields), native.ROOT_UID, 0o600)


@pytest.mark.parametrize(
    "kind", ["library-symlink", "library-writable", "state-symlink", "state-public"]
)
def test_installer_rejects_unsafe_directories(installation, tmp_path, kind):
    native, lock, selected, libraries = installation
    if kind == "library-symlink":
        libraries.rmdir()
        libraries.symlink_to(tmp_path)
    elif kind == "library-writable":
        libraries.chmod(0o777)
    elif kind == "state-symlink":
        native.INSTALL_STATE.symlink_to(tmp_path)
    else:
        native.INSTALL_STATE.mkdir()
        native.INSTALL_STATE.chmod(0o755)
    with pytest.raises((OSError, ValueError)):
        native._install_selected(lock, selected)
    assert not list(tmp_path.glob("*.so*"))


def test_replaced_cache_lock_is_rejected_after_flock(delivery, monkeypatch):
    native, lock, _, cache, calls = delivery
    original = native.fcntl.flock

    def replace_lock(descriptor, operation):
        original(descriptor, operation)
        path = cache / "bootstrap.lock"
        path.unlink()
        path.write_bytes(b"")
        path.chmod(0o600)

    monkeypatch.setattr(native.fcntl, "flock", replace_lock)
    with pytest.raises(ValueError, match="lock changed"):
        native._cached_artifacts(lock, cache)
    assert calls == []


def test_read_rejects_same_path_inode_replacement(delivery, monkeypatch):
    native, lock, _, cache, _ = delivery
    native._cached_artifacts(lock, cache)
    name = lock["debian_binaries"][0]["sha256"] + ".deb"
    original = native.os.stat

    def replace_before_stat(path, *args, **kwargs):
        if path == name:
            target = cache / name
            raw = target.read_bytes()
            target.unlink()
            target.write_bytes(raw)
            target.chmod(0o600)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(native.os, "stat", replace_before_stat)
    with pytest.raises(ValueError, match="changed during read"):
        native._cached_artifacts(lock, cache)


def test_runtime_lock_and_root_script_paths_cannot_be_symlinks(
    native, tmp_path, monkeypatch
):
    script = tmp_path / "native_bootstrap.py"
    script.write_bytes((PACKAGING / "native_bootstrap.py").read_bytes())
    script.chmod(0o644)
    interpreter = tmp_path / "python"
    interpreter.write_bytes(b"interpreter")
    interpreter.chmod(0o755)
    monkeypatch.setattr(native, "PYTHON", str(interpreter))
    monkeypatch.setattr(native, "SCRIPT_DIRECTORY", tmp_path)
    native._check_root_entrypoint()
    script.unlink()
    script.symlink_to(interpreter)
    with pytest.raises(OSError):
        native._check_root_entrypoint()
    monkeypatch.setattr(native, "LOCK_PATH", script)
    with pytest.raises(OSError):
        native._read_lock()
