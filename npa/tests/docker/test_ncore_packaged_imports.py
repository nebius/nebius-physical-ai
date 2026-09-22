"""Load qualification modules from the container's actual curated source set."""

from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig


NPA = Path(__file__).resolve().parents[2]


def test_packaged_source_supports_qualification_commands(tmp_path):
    dockerfile = (NPA / "docker/workbench/ncore/Dockerfile").read_text()
    package_root = tmp_path / "source"
    # Stage the real COPY instructions, including the narrow entrypoint aliases.
    # This deliberately does not expose the checkout or its editable install.
    for line in dockerfile.replace("\\\n", " ").splitlines():
        arguments = shlex.split(line, comments=True)
        if not arguments or arguments[0] != "COPY":
            continue
        sources, destination = arguments[1:-1], arguments[-1]
        prefix = "/opt/npa/src/"
        if not destination.startswith(prefix):
            continue
        target = package_root / destination.removeprefix(prefix)
        for source in sources:
            output = target / Path(source).name if destination.endswith("/") else target
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(NPA / source, output)

    script = """
import importlib
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), *sys.argv[2:]]
for name in (
    'npa.cli.entry',
    'npa.clients.storage',
    'npa.workbench.nurec.source_control',
    'npa.workbench.nurec.colmap',
    'npa.workbench.nurec.ncore_audit',
    'npa.workbench.nurec.ncore_rig',
):
    importlib.import_module(name)
from npa.cli.entry import application
application()
for name, module in tuple(sys.modules.items()):
    if name == 'npa' or name.startswith('npa.'):
        assert Path(module.__file__).resolve().is_relative_to(root), name
"""
    # -S prevents .pth files from registering an editable-checkout finder; only
    # dependency directories are restored, without running their startup hooks.
    dependencies = sorted(
        {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(package_root), *dependencies],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
