"""Keep the Pillow compatibility repair bound to reviewed release bytes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def compat():
    path = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/common/moviepy_pillow_compat.py"
    )
    spec = importlib.util.spec_from_file_location("moviepy_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_metadata_is_rejected(compat):
    with pytest.raises(RuntimeError, match="reviewed 2.2.1 bytes"):
        compat.patched_metadata(b"Requires-Dist: pillow<12.0,>=9.2.0\n")


def test_unknown_moviepy_version_is_rejected(compat, monkeypatch):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.2")
    )
    with pytest.raises(RuntimeError, match="requires MoviePy 2.2.1"):
        compat.main()


@pytest.mark.parametrize("pillow_version", ["11.3.0", "12.2.0", "13.0.0"])
def test_unpatched_or_unreviewed_pillow_version_is_rejected(
    compat, monkeypatch, pillow_version
):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.1")
    )
    monkeypatch.setattr(compat.metadata, "version", lambda _: pillow_version)
    with pytest.raises(RuntimeError, match="requires patched Pillow"):
        compat.main()
