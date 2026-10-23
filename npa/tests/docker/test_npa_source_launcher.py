"""Exercise source entrypoints independently of the full client Python floor."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "npa/docker/workbench/common/install_npa_source_launcher.py"
SPEC = importlib.util.spec_from_file_location("npa_source_launcher", INSTALLER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_launcher_preserves_arguments_and_selects_owned_source(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source with a ' quote"
    entry = source / "npa/cli/entry.py"
    entry.parent.mkdir(parents=True)
    (source / "npa/__init__.py").touch()
    (source / "npa/cli/__init__.py").touch()
    entry.write_text(
        "import json, sys\ndef main():\n    print(json.dumps(sys.argv[1:]))\n"
    )
    environment = tmp_path / "vendor"
    (environment / "bin").mkdir(parents=True)
    monkeypatch.setattr(MODULE.sys, "prefix", str(environment))
    launcher = MODULE.install_launcher(source)
    arguments = ["workbench", "vendor", "a b", "$(never-execute)"]
    result = subprocess.run(
        [str(launcher), *arguments], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == arguments
    assert launcher.stat().st_mode & 0o777 == 0o755
    assert not list(environment.glob("**/*.dist-info"))


def test_source_launcher_rejects_missing_entrypoint(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE.sys, "prefix", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="source entrypoint"):
        MODULE.install_launcher(tmp_path)
    assert not (tmp_path / "bin/npa").exists()


@pytest.mark.parametrize("tool", ["genesis", "cosmos", "groot"])
def test_legacy_vendor_images_use_source_entrypoint(tool: str) -> None:
    dockerfile = (ROOT / f"npa/docker/workbench/{tool}/Dockerfile").read_text()
    assert "common/install_npa_source_launcher.py" in dockerfile
    assert "install_npa_source_launcher.py /opt/npa/src" in dockerfile
    assert "--no-deps -e /opt/npa" not in dockerfile
    assert "--ignore-requires-python" not in dockerfile
