"""Inventory RoboCasa's exact source-only FFmpeg adapter without widening trust."""

import importlib
from pathlib import Path
import re

import pytest

SOURCE_URL = (
    "https://files.pythonhosted.org/packages/44/bd/"
    "c3343c721f2a1b0c9fc71c1aebf1966a3b7f08c2eea8ed5437a2865611d6/"
    "imageio_ffmpeg-0.6.0.tar.gz"
    "#sha256=e2556bed8e005564a9f925bb7afa4002d82770d6b08825078b7697ab88ba1755"
)


@pytest.fixture
def dependencies(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("security_dependencies")


@pytest.mark.parametrize("manifest", ["requirements.in", "requirements.lock"])
def test_actual_robocasa_source_manifests_are_inventoried(dependencies, manifest):
    root = Path(__file__).resolve().parents[3]
    text = (root / "npa/docker/workbench/robocasa" / manifest).read_text()
    declaration = next(
        line
        for line in text.replace("\\\n", "").splitlines()
        if line.startswith("imageio-ffmpeg")
    )
    assert declaration.split("--hash", 1)[0].strip() == "imageio-ffmpeg==0.6.0"
    assert re.findall(r"--hash=sha256:([a-f0-9]{64})", declaration) == [
        SOURCE_URL.split("#sha256=", 1)[1]
    ]
    pins = dependencies._exact_pins(text)
    assert pins.count("imageio-ffmpeg==0.6.0") == 1


@pytest.mark.parametrize(
    "declaration",
    [
        "imageio-ffmpeg @ " + SOURCE_URL[:-1] + "0",
        "imageio-ffmpeg @ " + SOURCE_URL.split("#", 1)[0],
        "imageio-ffmpeg @ "
        + SOURCE_URL.replace("files.pythonhosted.org", "example.invalid"),
        "imageio-ffmpeg @ " + SOURCE_URL.replace("0.6.0", "0.6.1"),
        "other-package @ " + SOURCE_URL,
        "imageio-ffmpeg @ " + SOURCE_URL + " --hash=sha256:" + "0" * 64,
    ],
)
def test_changed_source_identity_remains_unrecognized(dependencies, declaration):
    with pytest.raises(ValueError, match="Direct dependency"):
        dependencies._exact_pins(declaration)


def test_source_urls_remain_rejected_even_with_the_reviewed_option_hash(dependencies):
    url, fragment = SOURCE_URL.split("#sha256=", 1)
    with pytest.raises(ValueError, match="Direct dependency"):
        dependencies._exact_pins(f"imageio-ffmpeg @ {url} --hash=sha256:{fragment}")


@pytest.fixture
def source_lock(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "npa/docker/workbench/robocasa"))
    return importlib.import_module("pin_imageio_source")


def test_generated_lock_cannot_retain_wheel_hashes(source_lock):
    requirement = "imageio-ffmpeg==0.6.0"
    source_hash = SOURCE_URL.split("#sha256=", 1)[1]
    source = f"{requirement} --hash=sha256:{source_hash}\n"
    resolved = source.rstrip() + " --hash=sha256:" + "0" * 64 + "\n"
    result = source_lock._restrict_source(source, resolved)
    assert re.findall(r"--hash=sha256:([a-f0-9]{64})", result) == [source_hash]
    assert source_lock._restrict_source(source, result) == result


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "version", "hash", "wheel"]
)
def test_unreviewed_source_input_cannot_be_normalized(source_lock, mutation):
    source_hash = SOURCE_URL.split("#sha256=", 1)[1]
    source = f"imageio-ffmpeg==0.6.0 --hash=sha256:{source_hash}\n"
    bad = {
        "missing": "",
        "duplicate": source + source,
        "version": source.replace("0.6.0", "0.6.1"),
        "hash": source.replace(source_hash, "0" * 64),
        "wheel": source.rstrip() + " --hash=sha256:" + "0" * 64,
    }[mutation]
    with pytest.raises(ValueError):
        source_lock._restrict_source(bad, source)


def test_resolved_lock_missing_reviewed_sdist_is_rejected(source_lock):
    source_hash = SOURCE_URL.split("#sha256=", 1)[1]
    source = f"imageio-ffmpeg==0.6.0 --hash=sha256:{source_hash}\n"
    with pytest.raises(ValueError, match="omitted the reviewed sdist"):
        source_lock._restrict_source(source, source.replace(source_hash, "0" * 64))
