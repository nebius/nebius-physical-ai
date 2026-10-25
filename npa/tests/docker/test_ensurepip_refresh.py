"""Keep the standard-library installer repair bound to real replacement bytes."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2] / "docker/workbench/common/refresh_ensurepip.py"
)
SPEC = importlib.util.spec_from_file_location("ensurepip_refresh", SCRIPT)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def test_valid_bundle_replacement_updates_both_versions_and_real_wheel_bytes(tmp_path):
    source = '_PIP_VERSION = "23.0.1"\n_SETUPTOOLS_VERSION = "79.0.1"\n'
    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    (bundled / "pip-23.0.1-py3-none-any.whl").write_bytes(b"previous pip")
    (bundled / "setuptools-79.0.1-py3-none-any.whl").write_bytes(b"previous setuptools")
    wheels = {
        "pip-26.2.1+npa.1-py3-none-any.whl": b"replacement pip bytes",
        "setuptools-84.0.0-py3-none-any.whl": b"replacement setuptools bytes",
    }
    result = repair._replace_bundle(tmp_path, source, wheels)
    assert repair._versions(result) == {"pip": "26.2.1+npa.1", "setuptools": "84.0.0"}
    assert {path.name: path.read_bytes() for path in bundled.iterdir()} == wheels
    assert repair._replace_bundle(tmp_path, result, wheels) == result


@pytest.mark.parametrize(
    "source",
    [
        "",
        '_PIP_VERSION = "1"\n',
        '_PIP_VERSION = "1"\n_PIP_VERSION = "2"\n_SETUPTOOLS_VERSION = "3"\n',
    ],
)
def test_unrecognized_stdlib_layout_is_rejected(source):
    with pytest.raises(ValueError, match="version declarations"):
        repair._versions(source)


def test_unexpected_bootstrap_wheel_is_not_deleted_or_ignored(tmp_path):
    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    surprise = bundled / "unreviewed.whl"
    surprise.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="wheel population"):
        repair._replace_bundle(
            tmp_path, '_PIP_VERSION = "1"\n_SETUPTOOLS_VERSION = "2"\n', {}
        )
    assert surprise.read_bytes() == b"preserve"


@pytest.mark.parametrize("version,corrupt", [("84.0.0", True), ("0.0.0", False)])
def test_changed_or_mislabeled_wheel_is_rejected(tmp_path, version, corrupt):
    wheel = tmp_path / "setuptools-84.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "setuptools-84.0.0.dist-info/METADATA",
            f"Name: setuptools\nVersion: {version}\n",
        )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if corrupt:
        wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes|metadata"):
        repair._wheel(wheel, "setuptools", "84.0.0", digest)
