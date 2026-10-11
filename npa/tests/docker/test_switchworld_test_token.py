"""Verify test-credential removal occurs before final image-layer delivery."""

import ast
import base64
import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

NPA_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = NPA_ROOT / "docker/workbench/lingbot-world"
SPEC = importlib.util.spec_from_file_location(
    "sanitize_transformers_test_token",
    RECIPE_ROOT / "sanitize_transformers_test_token.py",
)
assert SPEC is not None and SPEC.loader is not None
SANITIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SANITIZER)


def _package(
    tmp_path,
    monkeypatch,
    source=b'# Apache-2.0 license remains\nTOKEN = "fixture-value"\nANSWER = 42\n',
):
    target = tmp_path / SANITIZER.SOURCE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    monkeypatch.setattr(SANITIZER, "SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    record = tmp_path / "transformers-5.5.0.dist-info/RECORD"
    record.parent.mkdir()
    record.write_text(
        SANITIZER.SOURCE_PATH
        + ",old,0\ntransformers/__pycache__/testing_utils.cpython-310.pyc,,\n"
    )
    bytecode = target.parent / "__pycache__/testing_utils.cpython-310.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"fixture-value")
    return target, record, bytecode


def test_removes_credential_and_bytecode_preserving_license_and_other_code(
    tmp_path, monkeypatch
):
    source, record, bytecode = _package(tmp_path, monkeypatch)
    SANITIZER.sanitize(tmp_path)
    changed = source.read_bytes()
    assert b"fixture-value" not in changed
    assert b"Apache-2.0 license remains" in changed
    assert SANITIZER.MODIFICATION_NOTICE.encode() in changed
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(changed).body
        if isinstance(node, ast.Assign)
    }
    assert assignments == {"TOKEN": "", "ANSWER": 42}
    assert not bytecode.exists()
    rows = list(csv.reader(record.read_text().splitlines()))
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(changed).digest()).rstrip(b"=").decode()
    )
    assert rows == [[SANITIZER.SOURCE_PATH, "sha256=" + expected, str(len(changed))]]


@pytest.mark.parametrize("mutation", ["source", "record"])
def test_unreviewed_inputs_refuse_before_writing(tmp_path, monkeypatch, mutation):
    source, record, bytecode = _package(tmp_path, monkeypatch)
    if mutation == "source":
        source.write_bytes(source.read_bytes() + b"# changed\n")
    else:
        record.write_text("unrelated.py,,\n")
    before = {path: path.read_bytes() for path in (source, record, bytecode)}
    with pytest.raises(ValueError):
        SANITIZER.sanitize(tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "source",
    [
        b'TOKEN = "first"\nTOKEN = "second"\n',
        b"TOKEN = read_token()\n",
        b'TOKEN = "fixture-value"\nOTHER = "fixture-value"\n',
    ],
)
def test_unsupported_assignments_refuse_without_mutation(tmp_path, monkeypatch, source):
    target, record, bytecode = _package(tmp_path, monkeypatch, source)
    before = [path.read_bytes() for path in (target, record, bytecode)]
    with pytest.raises(ValueError):
        SANITIZER.sanitize(tmp_path)
    assert [path.read_bytes() for path in (target, record, bytecode)] == before


def test_final_runtime_copies_only_the_sanitized_python_stage():
    dockerfile = (RECIPE_ROOT / "Dockerfile.switchworld-private").read_text()
    sanitized, final = dockerfile.split(
        "FROM python-runtime AS wan-python-sanitized\n", 1
    )[1].split("FROM ${DEBIAN_RUNTIME_BASE}\n", 1)
    assert "COPY --from=wan-source /opt/wan-base /opt/wan-base" in sanitized
    assert "RUN python3 -I -B /tmp/sanitize_transformers_test_token.py" in sanitized
    assert "COPY --from=wan-python-sanitized /opt/wan-base /opt/wan-base" in final
    assert "COPY --from=wan-source /opt/wan-base" not in final
    ast.parse((RECIPE_ROOT / "sanitize_transformers_test_token.py").read_text())
