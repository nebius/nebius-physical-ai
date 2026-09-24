"""Prove the SAM image scan rejects vendor bytes while allowing its bootstrap."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/scripts/scan_image_sam3_payload.py"
spec = importlib.util.spec_from_file_location("scan_image_sam3_payload", SCRIPT)
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)


def _tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


@pytest.mark.parametrize(
    "path",
    [
        "opt/source/sam3/model/model.py",
        "usr/local/lib/python3.12/site-packages/sam3/__init__.py",
        "opt/source/sam3/assets/bpe_simple_vocab.txt.gz",
        "workspace/weights/sam3.1_multiplex.pt",
        "opt/source/.git/objects/pack/a.pack",
        "usr/local/lib/python3.12/site-packages/torch/__init__.py",
        "workspace/.cache/huggingface/token",
    ],
)
def test_forbidden_payload_is_detected_even_in_earlier_layer(tmp_path, path):
    first = _tar(tmp_path / "one.tar", {path: b"payload"})
    second = _tar(tmp_path / "two.tar", {"opt/.wh.source": b""})
    assert scanner._scan([first, second], {})


def test_actual_bootstrap_files_pass(tmp_path):
    directory = ROOT / "npa/docker/workbench/sam3"
    files = {
        f"opt/npa/sam3/{path.name}": path.read_bytes()
        for path in directory.iterdir()
        if path.is_file()
    }
    archive = _tar(tmp_path / "clean.tar", files)
    assert scanner._scan([archive], {}) == []


def test_build_time_model_fetch_is_rejected(tmp_path):
    archive = _tar(tmp_path / "clean.tar", {})
    config = {"history": [{"created_by": "RUN sam3-runtime ensure"}]}
    assert any(f.kind == "sam_fetch_at_build" for f in scanner._scan([archive], config))


def test_publication_scans_both_local_and_pushed_bytes():
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    assert workflow.count("npa/scripts/scan_image_sam3_payload.py") == 2
    assert '"$RUNNER_TEMP/${TOOL}-sam3-payload.json"' in workflow
    assert '"$RUNNER_TEMP/${TOOL}-pushed-sam3-payload.json"' in workflow


def test_libssh2_identity_matches_reviewed_debian_package():
    lock = json.loads(
        (ROOT / "npa/docker/workbench/ncore/native-bootstrap-lock.json").read_text()
    )
    package = next(
        item for item in lock["debian_binaries"] if item["name"] == "libssh2-1"
    )
    library = package["elf_files"][0]
    assert scanner.AUDITED_SECRET_FILES[library["path"]] == library["sha256"]


def test_libssh2_exception_rejects_other_bytes(tmp_path):
    path = "usr/lib/x86_64-linux-gnu/libssh2.so.1.0.1"
    archive = _tar(tmp_path / "changed.tar", {path: b"changed library bytes"})
    assert any(
        item.kind == "audited_literal_byte_drift"
        for item in scanner._scan([archive], {})
    )
