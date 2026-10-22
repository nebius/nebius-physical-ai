"""Keep the Python 3.11 package floor compatible with image runtime boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "npa" / "docker" / "workbench"


def test_genesis_and_cosmos2_install_npa_in_python_311_or_newer() -> None:
    """The Sim2Real lineage must not install the package into a Python 3.10 venv."""

    genesis = (DOCKER / "genesis" / "Dockerfile").read_text(encoding="utf-8")
    cosmos2 = (DOCKER / "cosmos2-transfer" / "Dockerfile").read_text(encoding="utf-8")

    assert "/opt/npa/venv/lib/python3.11/site-packages" in genesis
    assert '"pillow>=13.0.0,<14.0"' in genesis
    assert "NPA_CONTROL_PYTHON=/opt/npa/control-venv/bin/python" in cosmos2
    assert "NPA_BAKED_PYTHON=/opt/npa/control-venv/bin/python" in cosmos2
    assert "uv venv --python /usr/bin/python3 /opt/npa/control-venv" in cosmos2
    assert "assert sys.version_info >= (3, 11)" in cosmos2


def test_groot_keeps_vendor_python_310_separate_from_python_311_npa_control() -> None:
    """Do not force Pillow 13 into GR00T's constrained upstream ABI."""

    text = (DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --python 3.10" in text
    assert "NPA_BAKED_PYTHON=/opt/isaac-lab/venv/bin/python" in text
    assert '"pillow>=13.0.0,<14.0"' in text
    assert 'uv pip install --python "${NPA_BAKED_PYTHON}" --no-deps -e /opt/npa' in text
    assert (
        'uv pip install --python "${GROOT_VENV}/bin/python" --no-deps -e /opt/npa'
        not in text
    )
    assert 'env -u PYTHONPATH "${NPA_BAKED_PYTHON}" -c' in text
    assert "assert sys.version_info >= (3, 11)" in text
    assert 'os.environ["GROOT_VENV"] + "/bin/python"' in text


def test_cosmos_uses_npa_only_as_build_time_ast_source() -> None:
    """Cosmos' CPython 3.10 vendor environment must not install the NPA package."""

    text = (DOCKER / "cosmos" / "Dockerfile").read_text(encoding="utf-8")

    assert "python /opt/npa/docker/workbench/cosmos/build_server.py" in text
    assert "python -m pip install --no-deps -e /opt/npa" not in text
    assert "PYTHONPATH=/opt/npa/src" not in text
    assert "rm -rf /opt/npa/src /opt/npa/pyproject.toml" in text
