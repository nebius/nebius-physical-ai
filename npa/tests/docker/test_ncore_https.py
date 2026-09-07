"""NCore call sites keep HTTPS host policy, hashes and bootstrap independence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from npa import _public_https as transport
from npa.workflows import ncore_runtime


ROOT = Path(__file__).parents[3]
RECIPES = ROOT / "npa/docker/workbench/ncore"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def response_bytes(monkeypatch):
    """Use the real shared transport, substituting only the network boundary."""
    payloads = []

    class Response(io.BytesIO):
        status = 200

    class Connection:
        def __init__(self, host, **kwargs):
            self.response = Response(payloads.pop(0))

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            pass

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", Connection)
    return payloads


@pytest.mark.parametrize("valid", [True, False])
def test_runtime_download_checks_actual_bytes_and_removes_corruption(
    response_bytes, tmp_path, valid
):
    data = b"complete pinned runtime artifact"
    response_bytes.append(data if valid else b"truncated")
    destination = tmp_path / "artifact.whl"
    item = {
        "url": "https://files.pythonhosted.org/artifact.whl",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if valid:
        ncore_runtime.download_artifact(item, destination)
        assert destination.read_bytes() == data
    else:
        with pytest.raises(ncore_runtime.NcoreRuntimeError, match="SHA256"):
            ncore_runtime.download_artifact(item, destination)
        assert not destination.exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://files.pythonhosted.org/artifact",
        "https://unexpected.example/artifact",
    ],
)
def test_runtime_direct_download_rejects_urls_without_relying_on_read_lock(
    response_bytes, tmp_path, url
):
    destination = tmp_path / "artifact.whl"
    with pytest.raises(transport.PublicDownloadError):
        ncore_runtime.download_artifact({"url": url, "sha256": "0" * 64}, destination)
    assert not destination.exists()


@pytest.mark.parametrize("valid", [True, False])
def test_source_assembly_retains_hash_verification(response_bytes, tmp_path, valid):
    data = b"complete pinned source"
    response_bytes.append(data if valid else b"corrupt")
    module = load(RECIPES / "base_sources.py")
    lock = {
        "artifacts": [
            {
                "path": "sources/source.tar",
                "filename": "source.tar",
                "url": "https://snapshot.debian.org/source.tar",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ]
    }
    output = tmp_path / "output"
    if valid:
        module.assemble(lock, output, tmp_path / "native", [])
        assert (output / "sources/source.tar").read_bytes() == data
    else:
        with pytest.raises(ValueError, match="SHA256"):
            module.assemble(lock, output, tmp_path / "native", [])
        assert not (output / "sources/source.tar").exists()
    assert not list(output.rglob("*.partial"))


@pytest.mark.parametrize("script", ["stage_upstream.py", "base_sources.py"])
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://snapshot.debian.org/source.tar",
        "https://unexpected.example/source.tar",
    ],
)
def test_source_recipes_reject_unapproved_urls(response_bytes, tmp_path, script, url):
    module = load(RECIPES / script)
    item = {
        "url": url,
        "sha256": "0" * 64,
        "path": "source.tar",
        "filename": "source.tar",
    }
    with pytest.raises(transport.PublicDownloadError):
        if script == "stage_upstream.py":
            module.stage({"ncore": item}, tmp_path / "output")
        else:
            module.assemble(
                {"artifacts": [item]}, tmp_path / "output", tmp_path / "native", []
            )
    assert not list(tmp_path.rglob("*.partial"))


def test_upstream_download_hash_fails_before_extraction(response_bytes, tmp_path):
    response_bytes.append(b"corrupt archive")
    module = load(RECIPES / "stage_upstream.py")
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        module.stage(
            {
                "ncore": {
                    "url": "https://codeload.github.com/NVIDIA/ncore/tar.gz/pin",
                    "sha256": "0" * 64,
                }
            },
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("script", ["stage_upstream.py", "base_sources.py"])
def test_bootstrap_recipes_import_without_installed_packages(tmp_path, script):
    shutil.copyfile(RECIPES / script, tmp_path / script)
    shutil.copyfile(Path(transport.__file__), tmp_path / "_public_https.py")
    # -S excludes site-packages and the editable NPA install. The recipe uses
    # only its delivered sibling helper and the interpreter's standard library.
    result = subprocess.run(
        [sys.executable, "-E", "-S", "-B", str(tmp_path / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


@pytest.mark.parametrize("fails", [False, True])
def test_runtime_schema_help_uses_private_temporary_directory(
    monkeypatch, tmp_path, fails
):
    paths = []

    def run(argv):
        if "--output-dir" in argv:
            path = Path(argv[argv.index("--output-dir") + 1])
            assert path.is_dir()
            assert path.stat().st_mode & 0o777 == 0o700
            paths.append(path)
            if fails:
                raise ncore_runtime.NcoreRuntimeError("help failed")

    monkeypatch.setattr(ncore_runtime, "_run", run)
    if fails:
        with pytest.raises(ncore_runtime.NcoreRuntimeError, match="help failed"):
            ncore_runtime.verify_runtime(tmp_path)
    else:
        ncore_runtime.verify_runtime(tmp_path)
        ncore_runtime.verify_runtime(tmp_path)
        assert paths[0] != paths[1]
    assert paths and all(not path.exists() for path in paths)


def test_committed_source_locks_use_explicit_approved_hosts():
    from urllib.parse import urlsplit

    module = load(RECIPES / "base_sources.py")
    lock = json.loads((RECIPES / "base-source-lock.json").read_bytes())
    hosts = {
        urlsplit(item["url"]).hostname for item in lock["artifacts"] if "url" in item
    }
    assert hosts <= module.SOURCE_DOWNLOAD_HOSTS
