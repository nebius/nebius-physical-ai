"""Exercise the publisher's real committed import boundary in clean subprocesses."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def committed_publisher(tmp_path):
    root = tmp_path / "source"
    paths = (
        "npa/scripts/ncore_publication",
        "npa/scripts/image_byte_scan",
        "npa/src/npa",
        "npa/docker/workbench/ncore",
    )
    for relative in paths:
        shutil.copytree(
            ROOT / relative,
            root / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    entry = Path("npa/scripts/publish_ncore_oci.py")
    shutil.copyfile(ROOT / entry, root / entry)
    root.chmod(0o700)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Committed publisher fixture",
    )
    return root, _git(root, "rev-parse", "HEAD")


def _startup(root, script):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "PYTEST"))
    }
    paths = [str(root / "npa/scripts"), str(root / "npa/src")]
    setup = f"import sys; sys.path[:0] = {paths!r}; "
    return subprocess.run(
        [sys.executable, "-B", "-c", setup + script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_prepare_reaches_inputs_after_committed_authorization(
    committed_publisher, tmp_path
):
    root, sha = committed_publisher
    (tmp_path / "analysis").mkdir(mode=0o700)
    argv = [
        str(root / "npa/scripts/publish_ncore_oci.py"),
        "prepare",
        "--source-sha",
        sha,
        "--analysis-root",
        str(tmp_path / "analysis"),
        "--policy-mode",
        "exact-literals",
    ]
    # Missing inventory stops in inputs, before any download/build/registry call.
    script = f"import runpy; sys.argv = {argv!r}; runpy.run_path(sys.argv[0], run_name='__main__')"
    result = _startup(root, script)
    assert result.returncode == 1
    assert "phase=inputs status=begin" in result.stderr
    assert "phase=inputs status=failure" in result.stderr
    assert "phase=prepare-keyring" not in result.stderr


@pytest.mark.parametrize("preload", [False, True])
def test_real_import_guard_accepts_committed_lazy_load_and_rejects_preload(
    committed_publisher, preload
):
    root, sha = committed_publisher
    script = ("import npa; " if preload else "") + (
        "from ncore_publication import cli, process; import json\n"
        "try:\n"
        f" with process.committed_npa_imports({sha!r}):\n"
        "  from ncore_publication import acceptance\n"
        "  from npa.deploy import images\n"
        " print(json.dumps({'authorized': True}))\n"
        "except Exception as error:\n"
        " print(json.dumps({'authorized': False, 'error': str(error)}))\n"
    )
    result = _startup(root, script)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    if preload:
        assert observed == {
            "authorized": False,
            "error": "host_npa_import_before_source_authorization",
        }
    else:
        assert observed == {"authorized": True}
