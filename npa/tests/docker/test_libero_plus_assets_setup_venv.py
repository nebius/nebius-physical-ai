"""Executable dependency-closure contract for the private LIBERO assets image."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile


PY_YAML_VERSION = "6.0.3"


def _local_pyyaml_wheel(tmp_path: Path) -> Path:
    """Make an offline wheel from the test environment's pinned PyYAML bytes."""

    distribution = metadata.distribution("PyYAML")
    assert distribution.version == PY_YAML_VERSION

    wheel_metadata = next(
        (
            Path(distribution.locate_file(member))
            for member in distribution.files or ()
            if str(member).endswith(".dist-info/WHEEL")
        ),
        None,
    )
    assert wheel_metadata is not None
    tags = [
        line.removeprefix("Tag: ")
        for line in wheel_metadata.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag: ")
    ]
    assert tags

    wheel = tmp_path / f"pyyaml-{PY_YAML_VERSION}-{tags[0]}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(distribution.files or (), key=str):
            relative = PurePosixPath(str(member))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source = Path(distribution.locate_file(member))
            if source.is_file():
                archive.write(source, relative.as_posix())
    return wheel


def test_setup_venv_bootstrap_installs_pyyaml_offline(tmp_path: Path) -> None:
    """Exercise the NPA_SETUP_PYTHON PyYAML bootstrap without a Docker build.

    The image recipe is statically bound to the same pin.  This isolated venv is
    deliberately not allowed to inherit the test runner's site packages: a host
    PyYAML installation must not make the image bootstrap appear healthy.
    """

    wheel = _local_pyyaml_wheel(tmp_path)
    venv_root = tmp_path / "npa-setup-venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_root)], check=True)
    setup_python = venv_root / "bin" / "python"
    environment = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(tmp_path / "empty-pip-cache"),
    }

    installed = subprocess.run(
        [
            setup_python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert installed.returncode == 0, installed.stderr
    assert "Successfully installed" in installed.stdout

    imported = subprocess.run(
        [
            setup_python,
            "-c",
            "import yaml; assert yaml.__version__ == '6.0.3'; print(yaml.__file__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert imported.returncode == 0, imported.stderr
    assert Path(imported.stdout.strip()).is_relative_to(venv_root)
