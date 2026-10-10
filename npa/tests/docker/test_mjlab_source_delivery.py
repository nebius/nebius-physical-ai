"""Reject changed source bytes, unsafe downloads and incomplete package ancestry."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3] / "npa/docker/workbench/mjlab"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signature_metadata_cannot_replace_the_source_version():
    path = ROOT.parent / "habitat-sim/bootstrap_sources.py"
    spec = importlib.util.spec_from_file_location("bootstrap_sources", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    descriptor = "Source: libasyncns\nVersion: 0.8-6\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v2\n"
    assert helper.fields(descriptor)["Version"] == "0.8-6"


@pytest.mark.parametrize("changed", [False, True])
def test_download_is_bound_to_locked_bytes(tmp_path, monkeypatch, changed):
    helper = _load("python_sources")
    expected = b"reviewed source"
    content = b"changed source" if changed else expected
    monkeypatch.setattr(
        helper,
        "download_public_https",
        lambda url, stream, **kwargs: stream.write(content),
    )
    item = {
        "filename": "source.tar.xz",
        "size": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
        "url": "https://ffmpeg.org/source.tar.xz",
    }
    if changed:
        with pytest.raises(ValueError, match="differs"):
            helper._fetch(item, tmp_path)
    else:
        helper._fetch(item, tmp_path)
        assert (tmp_path / item["filename"]).read_bytes() == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://ffmpeg.org/source",
        "https://operator:secret@ffmpeg.org/source",
        "https://localhost/source",
    ],
)
def test_lock_rejects_unapproved_source_origin(url):
    helper = _load("python_sources")
    lock = {
        "schema": "npa.mjlab.python-corresponding-source.v1",
        "artifacts": [
            {"filename": "source.tar.xz", "size": 10, "sha256": "a" * 64, "url": url}
        ],
    }
    with pytest.raises(ValueError, match="origin"):
        helper._validate(lock)


def test_deleted_corresponding_source_fails():
    helper = _load("verify_sources")
    with pytest.raises(ValueError, match="absent, deleted or changed"):
        helper._require_file({}, "source.tar.xz", 10, "a" * 64)


def test_superseded_package_version_also_requires_source():
    helper = _load("verify_sources")
    inventory = {
        "layers": [
            {"debian_packages": [{"source": "libsample", "source_version": version}]}
            for version in ("1.0", "2.0")
        ]
    }
    with pytest.raises(ValueError, match="all-layer package versions"):
        helper._verify_debian(
            inventory, [{"source": "libsample", "version": "2.0", "artifacts": []}]
        )
