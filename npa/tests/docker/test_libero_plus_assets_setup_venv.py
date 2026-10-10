"""Executable dependency-closure contract for the private LIBERO assets image."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


PY_YAML_VERSION = "6.0.3"
ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "npa/docker/workbench/libero-plus-assets/setup-venv.sh"


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


def test_setup_venv_bootstrap_executes_production_script_offline(
    tmp_path: Path,
) -> None:
    """Execute the Dockerfile's real bootstrap against an offline local wheel."""

    _local_pyyaml_wheel(tmp_path)
    base_runtime = tmp_path / "base-runtime"
    subprocess.run([sys.executable, "-m", "venv", str(base_runtime)], check=True)
    base_python = base_runtime / "bin" / "python"
    missing = subprocess.run(
        [base_python, "-c", "import yaml"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0

    venv_root = tmp_path / "npa-setup-venv"
    setup_python = venv_root / "bin" / "python"
    environment = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(tmp_path / "empty-pip-cache"),
        "PIP_FIND_LINKS": str(tmp_path),
    }

    # Docker installs this executable at a fixed path, then invokes it directly.
    # Retain the production execution contract here: copy, permission, shebang,
    # and interpreter selection must all work without a test-side `bash` wrapper.
    installed_bootstrap = tmp_path / "npa-libero-plus-assets-setup-venv"
    shutil.copy2(BOOTSTRAP, installed_bootstrap)
    installed_bootstrap.chmod(0o555)

    bootstrapped = subprocess.run(
        [str(installed_bootstrap), str(base_python), str(venv_root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert bootstrapped.returncode == 0, bootstrapped.stderr
    assert "Successfully installed" in bootstrapped.stdout
    assert "include-system-site-packages = true" in (
        venv_root / "pyvenv.cfg"
    ).read_text(encoding="utf-8")

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
