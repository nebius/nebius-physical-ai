from __future__ import annotations

import hashlib
import io
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from npa.workbench.antioch import runtime


WHEEL_BYTES = b"pinned-antioch-wheel-for-tests"
WHEEL_SHA256 = hashlib.sha256(WHEEL_BYTES).hexdigest()


def test_runtime_cache_defaults_to_xdg_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPA_ANTIOCH_RUNTIME_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert runtime.runtime_cache_root() == tmp_path / "npa/antioch"


@pytest.fixture
def runtime_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    calls = {"downloads": 0, "installs": 0}
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")
    monkeypatch.setenv("NPA_ANTIOCH_RUNTIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("NPA_ANTIOCH_RUNTIME_OFFLINE", raising=False)
    monkeypatch.delenv("NPA_ANTIOCH_CLI", raising=False)
    monkeypatch.delenv("NPA_ANTIOCH_CLI_URL", raising=False)
    monkeypatch.delenv("NPA_ANTIOCH_CLI_SHA256", raising=False)
    monkeypatch.setattr(runtime, "ANTIOCH_CLI_SHA256", WHEEL_SHA256)

    def urlopen(url: str):  # noqa: ANN202
        assert url == runtime.ANTIOCH_CLI_URL
        calls["downloads"] += 1
        return io.BytesIO(WHEEL_BYTES)

    def create(_builder, environment: Path) -> None:  # noqa: ANN001
        calls["installs"] += 1
        time.sleep(0.02)
        bin_dir = Path(environment) / "bin"
        bin_dir.mkdir(parents=True)
        for name in ("pip", "antioch", "python"):
            executable = bin_dir / name
            body = (
                "#!/bin/sh\nprintf 'antioch 0.3.63\\n'\n"
                if name == "python"
                else "#!/bin/sh\nexit 0\n"
            )
            executable.write_text(body, encoding="utf-8")
            executable.chmod(0o755)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(runtime.venv.EnvBuilder, "create", create)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""),
    )
    monkeypatch.setattr(
        "npa.workbench.antioch.vendor_cli.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "antioch 0.3.63\n", ""),
    )
    return calls


def test_terms_preflight_is_exact_and_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "yes")
    with pytest.raises(runtime.AntiochRuntimeError, match="=YES"):
        runtime.terms_preflight()
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")
    result = runtime.terms_preflight()
    assert result == {
        "name": "Antioch Terms of Service",
        "url": "https://antioch.com/terms",
        "version": "2026-02-28",
        "scope": "antioch-sim==0.3.63 and Antioch Service use",
        "accepted": True,
    }


def test_ensure_runtime_downloads_pinned_wheel_and_verifies_sha256(
    runtime_harness: dict[str, int],
) -> None:
    executable = runtime.ensure_runtime()
    assert executable.is_file()
    assert runtime_harness == {"downloads": 1, "installs": 1}
    assert executable.parents[2].joinpath(".complete").read_text().strip() == WHEEL_SHA256


def test_ensure_runtime_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")
    monkeypatch.setenv("NPA_ANTIOCH_RUNTIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("NPA_ANTIOCH_CLI_SHA256", raising=False)
    monkeypatch.setattr(runtime, "ANTIOCH_CLI_SHA256", "0" * 64)
    monkeypatch.setattr(
        runtime.urllib.request, "urlopen", lambda _url: io.BytesIO(WHEEL_BYTES)
    )
    with pytest.raises(runtime.AntiochRuntimeError, match="SHA-256"):
        runtime.ensure_runtime()


def test_ensure_runtime_offline_cold_cache_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")
    monkeypatch.setenv("NPA_ANTIOCH_RUNTIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("NPA_ANTIOCH_RUNTIME_OFFLINE", "1")
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda _url: pytest.fail("offline mode attempted a download"),
    )
    with pytest.raises(runtime.AntiochRuntimeError, match="offline mode"):
        runtime.ensure_runtime()


def test_ensure_runtime_concurrent_install_has_one_atomic_publisher(
    runtime_harness: dict[str, int],
) -> None:
    barrier = threading.Barrier(4)

    def ensure() -> Path:
        barrier.wait()
        return runtime.ensure_runtime()

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _item: ensure(), range(4)))
    assert len(set(paths)) == 1
    assert runtime_harness == {"downloads": 1, "installs": 1}


def test_ensure_runtime_reuses_verified_install_without_download(
    runtime_harness: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = runtime.ensure_runtime()
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda _url: pytest.fail("verified install was downloaded again"),
    )
    second = runtime.ensure_runtime()
    assert second == first
    assert runtime_harness == {"downloads": 1, "installs": 1}


def test_ensure_runtime_publishes_relocatable_executable(
    runtime_harness: dict[str, int],
) -> None:
    executable = runtime.ensure_runtime()
    launcher = executable.read_text(encoding="utf-8")
    assert ".antioch-install-" not in launcher
    assert '"$bin_dir/python"' in launcher
    assert 'sys.argv[0]="antioch"' in launcher

    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(  # noqa: S603 - generated test executable
        [str(executable), "--version"],
        stdout=write_fd,
        stderr=subprocess.DEVNULL,
    )
    os.close(write_fd)
    with os.fdopen(read_fd) as output:
        assert output.read().strip() == "antioch 0.3.63"
    assert process.wait() == 0
