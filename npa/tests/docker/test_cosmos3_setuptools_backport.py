"""Fail-closed checks for the runtime's reviewed manifest-fix verifier."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docker/workbench/cosmos3-serving"


@pytest.fixture
def backport():
    spec = importlib.util.spec_from_file_location(
        "manifest_backport", IMAGE.parent / "common/backport_setuptools_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path", ["setuptools/command/egg_info.py", "setuptools/unicode_utils.py"]
)
@pytest.mark.parametrize("version", ["80.10.2", "81.0.0", "84.1.0"])
def test_unreviewed_installed_source_is_rejected(backport, path, version):
    with pytest.raises(RuntimeError, match="differs from the reviewed"):
        backport.patched_source(
            path, b"unexpected package replacement\n", version=version
        )


def test_verifier_does_not_accept_an_unreviewed_distribution(backport, monkeypatch):
    class UnsupportedDistribution:
        version = "84.0.0"

    monkeypatch.setattr(
        backport.metadata, "distribution", lambda _: UnsupportedDistribution()
    )
    with pytest.raises(RuntimeError, match="reviewed Setuptools"):
        backport.main()


def test_secure_isaac_setuptools_is_reviewed_native_fix(backport):
    hashes = backport.SOURCE_HASHES["84.1.0"]
    assert set(hashes) == {
        "setuptools/command/egg_info.py",
        "setuptools/unicode_utils.py",
    }
    assert all(original == patched for original, patched in hashes.values())


def test_fresh_and_reused_runtime_apply_backport_before_execution():
    script = (IMAGE / "runtime_bootstrap.sh").read_text()
    call = (
        '"${VENV}/bin/python" /opt/npa-cosmos3-serving/backport_setuptools_manifest.py'
    )
    reuse = script[script.index('if [ -f "${MARKER}"') : script.index("work=")]
    assert reuse.index(call) < reuse.index('exec "$@"')
    install = script[script.index("work=") :]
    assert install.index('--no-deps "${work}/source"') < install.index(call)
    assert install.index(call) < install.index('"${VENV}/bin/python" -m pip check')
    assert install.index(call) < install.index('touch "${MARKER}"')
    assert (
        "COPY --chmod=0644 docker/workbench/common/backport_setuptools_manifest.py"
        in (IMAGE / "Dockerfile").read_text()
    )
