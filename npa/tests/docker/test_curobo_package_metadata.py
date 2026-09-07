"""A pinned metadata repair preserves validation and rejects source drift."""

import hashlib
import importlib.util
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa/docker/workbench/curobo"


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location(
        "curobo_metadata_correction", IMAGE_DIR / "correct_package_metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reviewed_source(tmp_path, helper, monkeypatch):
    # Small artificial metadata exercises the transformation without vendoring
    # the upstream project. Exact release hashes are independently checked below
    # and during the real package-backend qualification.
    original = (
        b"# SPDX-License-Identifier: Apache-2.0\n"
        b"[build-system]\nrequires = ['setuptools>=45', 'wheel']\n"
        b"[project]\nname = 'nvidia-curobo'\nversion = '0.8.0'\n"
        b'license = {text = "Apache-2.0"}\n'
        b'classifiers = ["Topic :: Scientific/Engineering :: Robotics"]\n'
        b"dependencies = ['warp-lang>=0.10.0', 'numpy']\n"
    )
    (tmp_path / "pyproject.toml").write_bytes(original)
    (tmp_path / "NPA_SOURCE_REVISION").write_text(helper.SOURCE_REVISION + "\n")
    (tmp_path / "solver.py").write_text("# Runtime source must remain unchanged.\n")
    corrected = helper.CHANGE_NOTICE + original.replace(
        helper.ORIGINAL_CLASSIFIER, helper.CORRECTED_CLASSIFIER
    )
    monkeypatch.setattr(helper, "UPSTREAM_METADATA_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(helper, "CORRECTED_METADATA_SHA256", hashlib.sha256(corrected).hexdigest())
    return tmp_path, original, corrected


def test_single_classifier_correction_preserves_every_other_metadata_field(helper, reviewed_source):
    root, original, expected = reviewed_source
    original_files = {p.name: p.read_bytes() for p in root.iterdir()}
    receipt = helper.correct_package_metadata(root)
    actual = (root / "pyproject.toml").read_bytes()
    assert actual == expected
    before = tomllib.loads(original.decode())
    after = tomllib.loads(actual.decode())
    assert after["project"].pop("classifiers") == ["Topic :: Scientific/Engineering"]
    before["project"].pop("classifiers")
    assert before == after
    assert actual.startswith(helper.CHANGE_NOTICE + b"# SPDX-License-Identifier: Apache-2.0\n")
    assert receipt["before_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["after_sha256"] == hashlib.sha256(actual).hexdigest()
    assert receipt["source_archive_sha256"] == helper.SOURCE_ARCHIVE_SHA256
    assert receipt["source_revision"] == helper.SOURCE_REVISION
    assert receipt["changed_file"] == "pyproject.toml"
    assert set(p.name for p in root.iterdir()) == set(original_files)
    for name in ("NPA_SOURCE_REVISION", "solver.py"):
        assert (root / name).read_bytes() == original_files[name]


@pytest.mark.parametrize("failure", ["changed_metadata", "revision", "missing_revision", "second_apply"])
def test_unreviewed_source_fails_without_further_mutation(helper, reviewed_source, failure):
    root, _, _ = reviewed_source
    if failure == "changed_metadata":
        with (root / "pyproject.toml").open("ab") as stream:
            stream.write(b"# Unreviewed source change\n")
    elif failure == "revision":
        (root / "NPA_SOURCE_REVISION").write_text("0" * 40 + "\n")
    elif failure == "missing_revision":
        (root / "NPA_SOURCE_REVISION").unlink()
    else:
        helper.correct_package_metadata(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError):
        helper.correct_package_metadata(root)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize("count", [0, 2])
def test_reviewed_classifier_must_appear_exactly_once(helper, reviewed_source, monkeypatch, count):
    root, original, _ = reviewed_source
    changed = original.replace(helper.ORIGINAL_CLASSIFIER, b", ".join([helper.ORIGINAL_CLASSIFIER] * count))
    (root / "pyproject.toml").write_bytes(changed)
    monkeypatch.setattr(helper, "UPSTREAM_METADATA_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="exactly one"):
        helper.correct_package_metadata(root)
    assert (root / "pyproject.toml").read_bytes() == changed


def test_changed_transform_output_fails_before_write(helper, reviewed_source, monkeypatch):
    root, original, _ = reviewed_source
    monkeypatch.setattr(helper, "CORRECTED_METADATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="corrected metadata hash"):
        helper.correct_package_metadata(root)
    assert (root / "pyproject.toml").read_bytes() == original


@pytest.mark.parametrize("name", ["pyproject.toml", "NPA_SOURCE_REVISION"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_linked_input_is_rejected_without_modifying_target(helper, reviewed_source, name, kind):
    root, _, _ = reviewed_source
    path = root / name
    target = root / (name + ".outside")
    path.rename(target)
    original = target.read_bytes()
    if kind == "symlink":
        path.symlink_to(target)
    else:
        path.hardlink_to(target)
    with pytest.raises(ValueError, match="regular, unlinked"):
        helper.correct_package_metadata(root)
    assert target.read_bytes() == original


def test_docker_build_applies_hash_guard_before_normal_pinned_backend(helper):
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    correction = "python /opt/correct_package_metadata.py /opt/curobo"
    install = "SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 pip install --no-deps --no-build-isolation --no-cache-dir /opt/curobo"
    assert f"{helper.SOURCE_ARCHIVE_SHA256}  /tmp/curobo.tar.gz" in dockerfile
    assert f"{helper.SOURCE_REVISION} > /opt/curobo/NPA_SOURCE_REVISION" in dockerfile
    assert dockerfile.index("sha256sum -c -", dockerfile.index("/tmp/curobo.tar.gz")) < dockerfile.index(correction)
    assert dockerfile.index("/opt/curobo/NPA_SOURCE_REVISION") < dockerfile.index(correction) < dockerfile.index(install)
    assert correction + " > /usr/share/doc/npa-curobo/metadata-correction.json" in dockerfile
    lock = (IMAGE_DIR / "requirements.lock").read_text()
    assert "setuptools==84.0.0" in lock
    assert "trove-classifiers==2026.6.1.19" in lock
    assert helper.UPSTREAM_METADATA_SHA256 == "4c93ee00a80dbc46e45fb6a1dd9486b57c8547e59bd948c30978a8ac7ed03a44"
    assert helper.CORRECTED_METADATA_SHA256 == "ca1967835fbf45a89617a70d5cbf596cd4368623b84a8013dfca2b6bedae32b9"
