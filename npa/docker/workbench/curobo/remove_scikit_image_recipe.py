#!/usr/bin/env python3
"""Remove a token-bearing historical recipe from the exact locked source.

Call in the same image RUN that installs scikit-image. The primary documentation,
image loader and all other source stay intact; an unrecognized input fails.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import py_compile
from pathlib import Path

EXPECTED_VERSION = "0.26.0"
SOURCE_SHA256 = "50e6234fa2170820eaf8d0f8f42b51905822afc3680a4f09113fa11d435f7fb4"
SANITIZED_SHA256 = "7f505612106adcc880746de642ceb91c9cbb74a6bd0c8100689c0da539c96abf"
MODULE = "skimage/data/_fetchers.py"


def sanitize_source(source: bytes, version: str) -> bytes:
    if version != EXPECTED_VERSION or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected scikit-image version or source bytes")
    tree = ast.parse(source)
    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grass"]
    if len(candidates) != 1 or len(candidates[0].body) != 3:
        raise ValueError("unexpected image loader body")
    function = candidates[0]
    recipe = function.body[1]
    if not (isinstance(recipe, ast.Expr) and isinstance(recipe.value, ast.Constant)
            and isinstance(recipe.value.value, str) and recipe.lineno == 511
            and recipe.end_lineno == 535):
        raise ValueError("unexpected inert recipe structure")
    primary_docstring = ast.get_docstring(function, clean=False)
    del function.body[1]
    lines = source.splitlines(keepends=True)
    sanitized = b"".join(lines[:recipe.lineno - 1] + lines[recipe.end_lineno:])
    transformed = ast.parse(sanitized)
    new_function = next(n for n in transformed.body if isinstance(n, ast.FunctionDef)
                        and n.name == "grass")
    if (hashlib.sha256(sanitized).hexdigest() != SANITIZED_SHA256
            or ast.dump(tree, include_attributes=False) != ast.dump(transformed, include_attributes=False)
            or ast.get_docstring(new_function, clean=False) != primary_docstring):
        raise ValueError("unexpected source behavior change")
    return sanitized


def sanitize_installation(site_packages: Path) -> dict:
    site_packages = site_packages.resolve(strict=True)
    distributions = [d for d in importlib.metadata.distributions(path=[str(site_packages)])
                     if d.metadata.get("Name", "").lower().replace("_", "-") == "scikit-image"]
    if len(distributions) != 1:
        raise ValueError("expected one installed scikit-image distribution")
    distribution = distributions[0]
    source_path = site_packages / MODULE
    if source_path.resolve(strict=True) != source_path or not source_path.is_file():
        raise ValueError("unexpected image loader file")
    sanitized = sanitize_source(source_path.read_bytes(), distribution.version)
    record_entries = [p for p in distribution.files or [] if str(p).endswith(".dist-info/RECORD")]
    if len(record_entries) != 1:
        raise ValueError("expected installed package RECORD")
    record_path = Path(distribution.locate_file(record_entries[0]))
    record_path.resolve(strict=True).relative_to(site_packages)
    if record_path.is_symlink():
        raise ValueError("unexpected package RECORD link")
    old_record = record_path.read_bytes()
    rows = list(csv.reader(io.StringIO(old_record.decode())))
    if any(len(row) != 3 for row in rows) or sum(row[0] == MODULE for row in rows) != 1:
        raise ValueError("unexpected package RECORD entries")
    cache = source_path.parent / "__pycache__"
    cache_paths = list(cache.glob("_fetchers.*.pyc")) if cache.is_dir() else []
    if cache.is_symlink() or any(p.is_symlink() or not p.is_file() for p in cache_paths):
        raise ValueError("unexpected loader bytecode paths")
    source_path.write_bytes(sanitized)
    for path in cache_paths:
        path.unlink()
    py_compile.compile(str(source_path), doraise=True)
    pyc = Path(importlib.util.cache_from_source(str(source_path)))
    if not pyc.is_file() or pyc.is_symlink():
        raise ValueError("missing regenerated loader bytecode")
    source_digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(sanitized).digest()).decode().rstrip("=")
    new_rows = []
    for row in rows:
        path = Path(row[0])
        if path.parent == Path(MODULE).parent / "__pycache__" and path.name.startswith("_fetchers.") and path.suffix == ".pyc":
            continue
        new_rows.append([MODULE, source_digest, str(len(sanitized))] if row[0] == MODULE else row)
    new_rows.append([str(pyc.relative_to(site_packages)), "", ""])
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(new_rows)
    record_path.write_text(stream.getvalue())
    return {"schema_version": "npa.curobo.dependency-source-correction.v1",
            "distribution": "scikit-image", "version": distribution.version,
            "module": MODULE, "source_sha256": SOURCE_SHA256,
            "sanitized_sha256": SANITIZED_SHA256,
            "record_sha256_before": hashlib.sha256(old_record).hexdigest(),
            "record_sha256_after": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "bytecode_sha256": hashlib.sha256(pyc.read_bytes()).hexdigest(),
            "executable_ast_preserved": True, "primary_docstring_preserved": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize_installation(args.site_packages), sort_keys=True))


if __name__ == "__main__":
    main()
