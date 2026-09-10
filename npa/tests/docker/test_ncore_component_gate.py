"""Bind supplemental component evaluation to actual source and image inputs."""

import io
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import components, provenance  # noqa: E402

SHA = "a" * 40


def _archive(path, rows):
    with tarfile.open(path, "w") as stream:
        for name, value in rows:
            member = tarfile.TarInfo(name)
            if value is None:
                member.type = tarfile.DIRTYPE
                stream.addfile(member)
            else:
                member.size = len(value)
                stream.addfile(member, io.BytesIO(value))


@pytest.mark.parametrize("change", [None, "source", "notice", "revision", "hook", "license-copy", "duplicate"])
def test_component_source_map_authenticates_notices_generated_bytes_and_copies(tmp_path, monkeypatch, change):
    paths = {
        "source": "opt/npa/src/npa/__init__.py",
        "notice": "usr/share/doc/npa-ncore/notices/NPA-LICENSE",
        "revision": "usr/share/doc/npa-ncore/npa-source-sha",
        "hook": "opt/venv/lib/python3.12/site-packages/npa-source.pth",
        "license-copy": "usr/share/doc/npa-ncore/LICENSE-APACHE-2.0",
    }
    values = [b"source", b"notice", (SHA + "\n").encode(), b"/opt/npa/src\n", b"upstream"]
    files = [(paths[key], value) for key, value in zip(paths, values, strict=True)]
    if change in paths:
        files = [(name, b"modified" if name == paths[change] else raw) for name, raw in files]
    if change == "duplicate":
        files.append(files[0])
    files += [("opt/npa/src/npa/workflows", None), ("opt/ncore/src/ncore/LICENSE", b"upstream")]
    _archive(tmp_path / "rootfs.tar", files)
    expected = {ROOT / "npa/src/npa/__init__.py": W.sha(b"source"),
                ROOT / "npa/docker/workbench/ncore/notices/NPA-LICENSE": W.sha(b"notice")}
    monkeypatch.setattr(provenance, "_committed_digest", lambda path, sha: expected[path])
    monkeypatch.setattr(provenance, "shipped_source", lambda *_: 43)
    if change is None:
        actual = components._committed_files(tmp_path / "rootfs.tar", SHA)
        assert actual == dict(zip(paths.values(), map(W.sha, values), strict=True))
    else:
        with pytest.raises(ValueError):
            components._committed_files(tmp_path / "rootfs.tar", SHA)


def test_component_evaluation_receives_exact_verified_graph_and_committed_locks(tmp_path, monkeypatch):
    from npa.deploy import ncore_component_scan

    calls = []
    monkeypatch.setattr(components, "_committed_files", lambda path, sha: {"source": "b" * 64})
    monkeypatch.setattr(ncore_component_scan, "scan_archive", lambda *args, **kwargs: calls.append((args, kwargs)))
    graph = {"image_manifest_digest": "sha256:" + "c" * 64, "image_config_digest": "sha256:" + "d" * 64}
    components.verify(tmp_path, "sha256:" + "e" * 64, graph, SHA)
    args, kwargs = calls[0]
    assert args == (tmp_path / "rootfs.tar", tmp_path / "components")
    assert kwargs["image_digest"] == "sha256:" + "e" * 64
    assert kwargs["platform_digest"] == graph["image_manifest_digest"]
    assert kwargs["config_digest"] == graph["image_config_digest"]
    assert kwargs["source_sha"] == SHA and kwargs["committed_files"] == {"source": "b" * 64}
    for name, file in (("base_lock", "base-source-lock.json"), ("source_lock", "source-lock.json")):
        assert kwargs[name] == (ROOT / "npa/docker/workbench/ncore" / file).read_bytes()
