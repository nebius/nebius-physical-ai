"""Fail-closed checks for the constrained runtime's upstream manifest backport."""

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
@pytest.mark.parametrize("version", ["80.10.2", "81.0.0", "84.2.1"])
def test_unreviewed_installed_source_is_rejected(backport, path, version):
    with pytest.raises(RuntimeError, match="differs from the reviewed"):
        backport.patched_source(
            path, b"unexpected package replacement\n", version=version
        )


def test_exact_already_fixed_security_closure_is_a_verified_noop(backport):
    expected = {
        "setuptools/command/egg_info.py": "37bbdddc1a29864675b4a9bf97d131c336907ac83ed63d91e899006662a77134",
        "setuptools/unicode_utils.py": "0e400678e34254f1f7e8ef29a85515473e9df98699ea4d4314098ee414721d52",
    }
    assert {
        name: original
        for name, (original, patched) in backport.SOURCE_HASHES["84.2.1"].items()
        if original == patched
    } == expected


def test_current_isaac_python312_setuptools_closure_is_reviewed(backport):
    requirements = (IMAGE.parent / "common/isaac3-oss-deps.txt").read_text()
    selected = [
        line.partition("==")[2].strip()
        for line in requirements.splitlines()
        if line.startswith("setuptools==")
    ]
    assert selected == ["84.2.1"]
    assert all(
        original == patched
        for original, patched in backport.SOURCE_HASHES[selected[0]].values()
    )


def test_backport_rejects_unknown_dependency_closure(backport, monkeypatch):
    class UnsupportedDistribution:
        version = "84.0.0"

    monkeypatch.setattr(
        backport.metadata, "distribution", lambda _: UnsupportedDistribution()
    )
    with pytest.raises(RuntimeError, match="exact Setuptools 80.10.2"):
        backport.main()


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
