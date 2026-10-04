"""Pinned installer build inputs cannot silently change or omit components."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2] / "docker/workbench/common/secure_pip"
SPEC = importlib.util.spec_from_file_location("secure_pip_builder", ROOT / "build.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def manifest():
    return json.loads((ROOT / "inputs.json").read_text())


def upstream(manifest):
    return (
        "\n".join(
            row["name"]
            + "=="
            + manifest["replacements"]
            .get(row["name"].lower(), {})
            .get("from", row["version"])
            for row in manifest["vendors"]
        )
        + "\n"
    )


def test_complete_upstream_population_has_only_three_version_changes(manifest):
    requirements = builder.vendor_requirements(upstream(manifest), manifest)
    assert len(requirements.splitlines()) == len(manifest["vendors"]) == 18
    assert set(manifest["replacements"]) == {"urllib3", "msgpack", "setuptools"}
    for row in manifest["vendors"]:
        assert (
            f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}"
            in requirements.splitlines()
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "version", "extra", "replacement"]
)
def test_unexpected_vendor_input_is_rejected(manifest, mutation):
    original = upstream(manifest)
    altered = copy.deepcopy(manifest)
    if mutation == "missing":
        original = "\n".join(original.splitlines()[1:]) + "\n"
    elif mutation == "duplicate":
        original += original.splitlines()[0] + "\n"
    elif mutation == "version":
        original = original.replace("msgpack==1.1.2", "msgpack==0.0.0")
    elif mutation == "extra":
        altered["vendors"].append(dict(altered["vendors"][0]))
    else:
        altered["replacements"]["msgpack"]["to"] = "0.0.0"
    with pytest.raises(ValueError):
        builder.vendor_requirements(original, altered)


def test_public_download_is_verified_before_write(tmp_path, monkeypatch):
    data = b"authentic build input"
    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda _: io.BytesIO(data))
    row = {
        "url": "https://files.pythonhosted.org/packages/input.whl",
        "filename": "input.whl",
        "sha256": builder.digest(data),
    }
    assert builder.fetch(row, tmp_path).read_bytes() == data
    row["filename"] = "modified.whl"
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        builder.fetch(row, tmp_path)
    assert not (tmp_path / "modified.whl").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/x",
        "https://example.invalid/x",
        "file:///etc/passwd",
    ],
)
def test_unapproved_download_origin_rejected_before_network(tmp_path, monkeypatch, url):
    monkeypatch.setattr(
        builder.urllib.request, "urlopen", lambda _: pytest.fail("network attempted")
    )
    with pytest.raises(ValueError, match="origin"):
        builder.fetch({"url": url, "filename": "input", "sha256": "0" * 64}, tmp_path)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "..", ""])
def test_download_path_escape_rejected_before_network(tmp_path, monkeypatch, name):
    monkeypatch.setattr(
        builder.urllib.request, "urlopen", lambda _: pytest.fail("network attempted")
    )
    with pytest.raises(ValueError, match="filename"):
        builder.fetch(
            {
                "url": "https://files.pythonhosted.org/x",
                "filename": name,
                "sha256": "0" * 64,
            },
            tmp_path,
        )


@pytest.mark.parametrize(
    "name",
    ["../outside", "/absolute", "different-root/member", "pip-abc/../../outside"],
)
def test_archive_escape_rejected_before_extraction(tmp_path, name):
    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(name)
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="archive member"):
        builder.extract_source(archive, tmp_path / "dest", "abc")
    assert not (tmp_path / "dest").exists()


def test_archive_positive_preserves_source_bytes(tmp_path):
    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("pip-abc/source.py")
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    target = builder.extract_source(archive, tmp_path / "dest", "abc")
    assert (target / "source.py").read_bytes() == b"x"


def test_existing_work_directory_is_not_reused(tmp_path):
    marker = tmp_path / "retained"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        builder.build(tmp_path, tmp_path / "out")
    assert marker.read_text() == "keep"


def test_build_lock_preserves_marker_dependency():
    lock = (ROOT / "build-tools.lock").read_text()
    assert "typing-extensions==4.16.0" in lock
    assert "vendoring==1.4.0" in lock
    assert "flit-core==3.12.0" in lock
    assert "setuptools==80.9.0" in lock


@pytest.mark.parametrize("outer_mask", [0o022, 0o077])
def test_child_build_mode_is_fixed_without_changing_parent(tmp_path, outer_mask):
    previous = os.umask(outer_mask)
    try:
        builder.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('member').write_bytes(b'fixed')",
            ],
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"]},
        )
        assert stat.S_IMODE((tmp_path / "member").stat().st_mode) == 0o644
        assert (tmp_path / "member").read_bytes() == b"fixed"
        observed = os.umask(outer_mask)
        assert observed == outer_mask
    finally:
        os.umask(previous)
