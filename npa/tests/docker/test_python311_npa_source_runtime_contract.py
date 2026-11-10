"""Keep the Python >=3.11 package floor compatible with image runtime boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "npa" / "docker" / "workbench"


def test_genesis_and_cosmos2_use_supported_python_families() -> None:
    """The Sim2Real lineage must not keep an unpatchable Python 3.10 runtime."""

    genesis = (DOCKER / "genesis" / "Dockerfile").read_text(encoding="utf-8")
    cosmos2 = (DOCKER / "cosmos2-transfer" / "Dockerfile").read_text(encoding="utf-8")

    assert "/opt/genesis/venv/lib/python3.12/site-packages" in genesis
    assert "NPA_BAKED_PYTHON=/opt/genesis/venv/bin/python" in genesis
    assert "python3.12 -m venv /opt/genesis/venv" in genesis
    assert '"pillow>=13.0.0,<14.0"' in genesis
    assert "NPA_CONTROL_PYTHON=/opt/npa/control-venv/bin/python" in cosmos2
    assert "NPA_BAKED_PYTHON=/opt/npa/control-venv/bin/python" in cosmos2
    assert "uv venv --python /usr/bin/python3 /opt/npa/control-venv" in cosmos2
    assert "assert sys.version_info >= (3, 11)" in cosmos2
    assert "COSMOS_PYTHON_VERSION=3.11.16" in cosmos2
    assert ".venv/lib/python3.11/site-packages" in cosmos2
    assert ".venv/lib/python3.10/site-packages" not in cosmos2


def test_groot_uses_vendor_python312_separate_from_python311_npa_control() -> None:
    """Keep the GR00T vendor runtime separate from NPA's Isaac control runtime."""

    text = (DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --python 3.12" in text
    assert "NPA_BAKED_PYTHON=/opt/isaac-lab/venv/bin/python" in text
    assert text.count('"pillow==13.0.0"') == 1
    assert '"pillow>=13.0.0,<14.0"' not in text
    assert text.count('"cffi>=2.0.0"') == 2
    assert text.count("from cryptography.hazmat.bindings._rust import exceptions") == 2
    isaac_control_overlay = text.split(
        "RUN /opt/isaac-lab/venv/bin/python -m pip install --no-deps --upgrade",
        maxsplit=1,
    )[1]
    for requirement in (
        '"bcrypt==5.0.0"',
        '"annotated-types==0.8.0"',
        '"invoke==3.0.3"',
        '"paramiko==5.0.0"',
        '"pydantic==2.13.5"',
        '"pydantic-core==2.46.5"',
        '"pyarrow==25.0.1"',
        '"PyNaCl==1.6.2"',
        '"typing-inspection==0.4.4"',
        '"zstandard==0.25.0"',
    ):
        assert requirement in isaac_control_overlay
    assert 'uv pip install --python "${NPA_BAKED_PYTHON}" --no-deps -e /opt/npa' in text
    assert (
        'uv pip install --python "${GROOT_VENV}/bin/python" --no-deps -e /opt/npa'
        not in text
    )
    assert 'env -u PYTHONPATH "${NPA_BAKED_PYTHON}" -c' in text
    assert "assert sys.version_info >= (3, 11)" in text
    assert 'os.environ["GROOT_VENV"] + "/bin/python"' in text


def test_groot_does_not_commit_package_manager_caches_to_scanned_layers() -> None:
    """Keep install-time package caches out of every GR00T image layer."""

    text = (DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")

    assert "PIP_NO_CACHE_DIR=1" in text
    assert "UV_NO_CACHE=1" in text


def test_groot_vendor_runtime_pins_the_certifi_security_fix() -> None:
    """Keep GR00T's upstream dependency closure at the reviewed CA-bundle fix."""

    text = (DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")
    overlay = text.split(
        '&& uv pip install --python "${GROOT_VENV}/bin/python" --no-deps --upgrade',
        maxsplit=1,
    )[1].split("&& (uv pip uninstall", maxsplit=1)[0]

    assert '"certifi==2026.10.19"' in overlay


def test_groot_corrects_the_upstream_scikit_image_recipe_before_commit() -> None:
    """Keep the inert signed documentation recipe out of scanned image layers."""

    text = (DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY --chown=ubuntu:ubuntu docker/workbench/curobo/remove_scikit_image_recipe.py"
        in text
    )
    correction = text.split(
        '&& "${GROOT_VENV}/bin/python" /opt/groot/remove-scikit-image-recipe.py', 1
    )[1]
    assert '--site-packages "${GROOT_VENV}/lib/python3.12/site-packages"' in correction
    assert "--capability groot" in correction
    assert "> /opt/groot/dependency-source-correction.json" in correction


def test_cosmos_uses_npa_only_as_build_time_ast_source() -> None:
    """Cosmos' CPython 3.10 vendor environment must not install the NPA package."""

    text = (DOCKER / "cosmos" / "Dockerfile").read_text(encoding="utf-8")

    assert "python /opt/npa/docker/workbench/cosmos/build_server.py" in text
    assert "python -m pip install --no-deps -e /opt/npa" not in text
    assert "PYTHONPATH=/opt/npa/src" not in text
    assert "rm -rf /opt/npa/src /opt/npa/pyproject.toml" in text
    cleanup = text.split(
        "COPY --chown=ubuntu:ubuntu docker/workbench/cosmos/smoke_functional.py", 1
    )[1]
    assert cleanup.index("USER root") < cleanup.index(
        "RUN python /opt/npa/docker/workbench/cosmos/build_server.py"
    )
    assert cleanup.index("USER ubuntu") < cleanup.index("RUN python -m py_compile")
