"""Remove one inert dependency recipe before layer creation, preserving behavior."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.util
import io
import py_compile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/curobo"


@pytest.fixture
def installed(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("curobo_dependency_correction", IMAGE / "remove_scikit_image_recipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (b"# Copyright and BSD notice preserved.\n" * 508
              + b'def grass():\n    """Primary public documentation."""\n    """\n'
              + b"    synthetic noncredential historical recipe\n" * 23
              + b'    """\n    return _load("data/grass.png")\n')
    sanitized = b"".join(source.splitlines(keepends=True)[:510] + source.splitlines(keepends=True)[535:])
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(module, "SANITIZED_SHA256", hashlib.sha256(sanitized).hexdigest())
    path = tmp_path / module.MODULE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    py_compile.compile(str(path), doraise=True)
    py_compile.compile(str(path), doraise=True, optimize=1)
    unrelated = path.parent / "__pycache__/unrelated.pyc"
    unrelated.write_bytes(b"unrelated bytecode must stay identical")
    dist = tmp_path / "scikit_image-0.26.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: scikit-image\nVersion: 0.26.0\n")
    (dist / "LICENSE").write_bytes(b"complete original dependency license\n")
    rows = [[module.MODULE, "", str(len(source))],
            [str((dist / "LICENSE").relative_to(tmp_path)), "", ""],
            [str((dist / "RECORD").relative_to(tmp_path)), "", ""]]
    for pyc in path.parent.glob("__pycache__/_fetchers.*.pyc"):
        rows.append([str(pyc.relative_to(tmp_path)), "", ""])
    with (dist / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return module, tmp_path, path, dist, source, sanitized, unrelated


def test_exact_correction_preserves_loader_and_notices_and_updates_record(installed):
    module, root, path, dist, source, sanitized, unrelated = installed
    report = module.sanitize_installation(root)
    assert path.read_bytes() == sanitized
    assert (dist / "LICENSE").read_bytes() == b"complete original dependency license\n"
    assert unrelated.read_bytes() == b"unrelated bytecode must stay identical"
    assert report["executable_ast_preserved"] and report["primary_docstring_preserved"]
    assert report["record_sha256_before"] != report["record_sha256_after"]
    rows = list(csv.reader(io.StringIO((dist / "RECORD").read_text())))
    source_row = next(row for row in rows if row[0] == module.MODULE)
    assert source_row == [module.MODULE, "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(sanitized).digest()).decode().rstrip("="), str(len(sanitized))]
    caches = list(path.parent.glob("__pycache__/_fetchers.*.pyc"))
    assert len(caches) == 1
    assert hashlib.sha256(caches[0].read_bytes()).hexdigest() == report["bytecode_sha256"]
    assert not any("opt-1" in row[0] for row in rows)
    calls = []
    outputs = []
    for body in (source, sanitized):
        namespace = {"_load": lambda value: calls.append(value) or "same result"}
        exec(compile(ast.parse(body), "synthetic-example", "exec"), namespace)
        outputs.append(namespace["grass"]())
        assert namespace["grass"].__doc__ == "Primary public documentation."
    assert outputs == ["same result", "same result"]
    assert calls == ["data/grass.png", "data/grass.png"]


@pytest.mark.parametrize("mutation", ["version", "source", "record_duplicate", "record_missing",
    "record_malformed", "source_symlink", "cache_symlink", "distribution_duplicate"])
def test_unrecognized_installation_refuses_before_source_mutation(installed, mutation, tmp_path):
    module, root, path, dist, source, _sanitized, _unrelated = installed
    if mutation == "version":
        (dist / "METADATA").write_text("Name: scikit-image\nVersion: 0.26.1\n")
    elif mutation == "source":
        path.write_bytes(source + b"# unreviewed source change\n")
    elif mutation == "record_duplicate":
        with (dist / "RECORD").open("a") as stream:
            stream.write(module.MODULE + ",,\n")
    elif mutation == "record_missing":
        (dist / "RECORD").write_text("other.py,,\n")
    elif mutation == "record_malformed":
        (dist / "RECORD").write_text(module.MODULE + ",\n")
    elif mutation == "source_symlink":
        target = tmp_path / "outside.py"
        target.write_bytes(source)
        path.unlink()
        path.symlink_to(target)
    elif mutation == "cache_symlink":
        pyc = next(path.parent.glob("__pycache__/_fetchers.*.pyc"))
        pyc.unlink()
        pyc.symlink_to(tmp_path / "elsewhere")
    else:
        duplicate = root / "scikit_image-duplicate.dist-info"
        duplicate.mkdir()
        (duplicate / "METADATA").write_text((dist / "METADATA").read_text())
    before = path.read_bytes()
    with pytest.raises(ValueError):
        module.sanitize_installation(root)
    assert path.read_bytes() == before


def test_already_corrected_source_is_not_silently_reaccepted(installed):
    module, root, path, _dist, _source, sanitized, _unrelated = installed
    module.sanitize_installation(root)
    with pytest.raises(ValueError, match="unexpected scikit-image"):
        module.sanitize_installation(root)
    assert path.read_bytes() == sanitized


def test_runtime_statements_cannot_be_removed_as_a_recipe(installed, monkeypatch):
    module, _root, _path, _dist, source, _sanitized, _unrelated = installed
    changed = source.replace(b'    """\n    return', b'    """\n    execute_work()\n    return')
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="unexpected image loader body"):
        module.sanitize_source(changed, "0.26.0")


@pytest.mark.parametrize("failure", ["compiler_error", "missing_cache", "linked_cache"])
def test_regeneration_failure_cannot_emit_a_success_receipt(installed, monkeypatch, failure):
    module, _root, path, dist, _source, _sanitized, unrelated = installed
    original_record = (dist / "RECORD").read_bytes()

    def broken_compile(*args, **kwargs):
        if failure == "compiler_error":
            raise RuntimeError("synthetic compiler failure")
        if failure == "linked_cache":
            Path(importlib.util.cache_from_source(str(path))).symlink_to(unrelated)

    monkeypatch.setattr(module.py_compile, "compile", broken_compile)
    # The enclosing installation RUN must fail, so none of its files can become
    # a successful ancestor layer. RECORD is not updated to claim completion.
    with pytest.raises((RuntimeError, ValueError)):
        module.sanitize_installation(path.parents[2])
    assert (dist / "RECORD").read_bytes() == original_record
    assert unrelated.read_bytes() == b"unrelated bytecode must stay identical"


def test_correction_runs_before_installation_layer_commits():
    dockerfile = (IMAGE / "Dockerfile").read_text()
    start = dockerfile.index("RUN pip install --no-cache-dir --require-hashes")
    end = dockerfile.index("\n\n", start)
    instruction = dockerfile[start:end]
    assert "python /opt/remove_scikit_image_recipe.py" in instruction
    assert instruction.index("remove_scikit_image_recipe.py") < instruction.index("pip check")
    source = (IMAGE / "remove_scikit_image_recipe.py").read_text()
    assert "50e6234fa2170820eaf8d0f8f42b51905822afc3680a4f09113fa11d435f7fb4" in source
    assert "7f505612106adcc880746de642ceb91c9cbb74a6bd0c8100689c0da539c96abf" in source
