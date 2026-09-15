"""Authenticate NCore's five CPython KaRaMeL runtime headers and their upstream query identity."""

from __future__ import annotations

import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

from npa._public_https import download_public_https

_CPYTHON_COMMIT = "2abcf904b8dac8c999d2b3aac76681abb333798a"
_HACL_COMMIT = "bb3d0dc8d9d15a5cd51094d5b69e70aa09005ff0"
_KARAMEL_COMMIT = "95968326f0ca1d6f9056347496482d285e6a9f1e"
_SOURCES = {
    "cpython": {
        "url": "https://codeload.github.com/python/cpython/tar.gz/" + _CPYTHON_COMMIT,
        "sha256": "72b4d8791b053a808f247e0de247a8b976868e705204e8fcab0812c823873766",
        "root": "cpython-" + _CPYTHON_COMMIT,
    },
    "hacl": {
        "url": "https://codeload.github.com/hacl-star/hacl-star/tar.gz/" + _HACL_COMMIT,
        "sha256": "cc3ed28cce79ef34e49c3f342fe1f9db7398d55dd56ff64026c35e2e3646cc85",
        "root": "hacl-star-" + _HACL_COMMIT,
    },
    "karamel": {
        "url": "https://codeload.github.com/FStarLang/karamel/tar.gz/" + _KARAMEL_COMMIT,
        "sha256": "0bb643da72c42177cc0a7ee080a4921ec6138cb85d1d6fe29ff19adceda4998d",
        "root": "karamel-" + _KARAMEL_COMMIT,
    },
}
_HEADERS = {
    "lowstar_endianness.h": "include/krml/lowstar_endianness.h",
    "internal/target.h": "include/krml/internal/target.h",
    "FStar_UInt_8_16_32_64.h": "krmllib/dist/minimal/FStar_UInt_8_16_32_64.h",
    "fstar_uint128_struct_endianness.h": "krmllib/dist/minimal/fstar_uint128_struct_endianness.h",
    "FStar_UInt128_Verified.h": "krmllib/dist/minimal/FStar_UInt128_Verified.h",
}
_CPYTHON_PREFIX = "Modules/_hacl/include/krml/"
_HEADER_MAPPING_SHA256 = "ed447e61b43ba3e6dbae927af8155d6e3fd6da4b384b0a582a069474bee674f3"
_HACL_TREE_SHA256 = "0dc78bcc5c9548f8346d05e5a8cc1115e5b05cf8e2f4ac5c4510cb1a47a1c204"
_CPYTHON_TREE_SHA256 = "86c3d1e77fd7cb544f092bdc43344c878f51ca2510c618042c0060747fbc1d3f"
_RECORDS = {
    "hacl": {
        "dist/gcc-compatible/INFO.txt": "1807190c4274ede6c957afd1b127956e5e53b9a5f87f3b0687955984c6bfaa11",
        "Makefile": "a3507c968e6edcef3ddd15ea59edc49a5096c7d15de8ab44df43faef112c7ea5",
    },
    "cpython": {
        "Modules/_hacl/refresh.sh": "d8b79058056e8ed823d3191ab98a45196379989c51aafa918d4005f4bfeec7ab",
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError("NCore KaRaMeL source: " + message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _source_archive(name, directory, raw=None):
    source = _SOURCES[name]
    if raw is None:
        output = io.BytesIO()
        download_public_https(source["url"], output,
                              allowed_hosts=frozenset({"codeload.github.com"}))
        raw = output.getvalue()
    (directory / (name + ".karamel-source.tar.gz")).write_bytes(raw)
    _require(_sha(raw) == source["sha256"], "archive hash differs: " + name)
    return _archive_files(raw, source["root"])


def _archive_files(raw, root):
    files, seen = {}, set()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            path = member.name.removesuffix("/")
            parts = path.split("/")
            _require(parts[0] == root and all(p not in {"", ".", ".."} for p in parts)
                     and "\\" not in path, "invalid archive path")
            _require(path not in seen, "duplicate archive member")
            seen.add(path)
            if member.isdir():
                continue
            # Unrelated upstream symlinks are data only; no archive is extracted.
            if not member.isfile():
                continue
            _require(len(parts) > 1, "archive file has no root directory")
            files["/".join(parts[1:])] = archive.extractfile(member).read()
    _require(files, "empty source archive")
    return files


def _source_records(cpython, hacl):
    records = {}
    for name, files in (("cpython", cpython), ("hacl", hacl)):
        records[name] = {}
        for path, expected in _RECORDS[name].items():
            _require(path in files and _sha(files[path]) == expected,
                     "source record differs: " + path)
            records[name][path] = expected
    _require(b"Karamel version: " + _KARAMEL_COMMIT.encode() + b"\n"
             in hacl["dist/gcc-compatible/INFO.txt"], "recorded KaRaMeL revision differs")
    return records


def _vendor_tree(hacl, karamel):
    prefix = "dist/karamel/"
    tree = {p.removeprefix(prefix): _sha(raw) for p, raw in hacl.items() if p.startswith(prefix)}
    _require(len(tree) == 21 and _sha(_canonical(tree)) == _HACL_TREE_SHA256,
             "HACL vendored population differs")
    for path, expected in tree.items():
        _require(path in karamel and _sha(karamel[path]) == expected,
                 "HACL file differs from recorded KaRaMeL revision: " + path)
    return tree


def _header_mapping(cpython, karamel):
    tree = {p: _sha(raw) for p, raw in cpython.items() if p.startswith(_CPYTHON_PREFIX)}
    expected = {_CPYTHON_PREFIX + path for path in _HEADERS} | {_CPYTHON_PREFIX + "types.h"}
    _require(tree.keys() == expected and _sha(_canonical(tree)) == _CPYTHON_TREE_SHA256,
             "CPython runtime header population differs")
    files = {}
    for destination, origin in sorted(_HEADERS.items()):
        path = _CPYTHON_PREFIX + destination
        _require(origin in karamel, "selected upstream header missing: " + origin)
        upstream, vendored = karamel[origin], cpython[path]
        patch = "".join(difflib.unified_diff(
            upstream.decode().splitlines(True), vendored.decode().splitlines(True),
            fromfile=origin, tofile=path,
        )).encode()
        files[path] = {"upstream_path": origin, "hacl_path": "dist/karamel/" + origin,
                       "upstream_sha256": _sha(upstream), "vendored_sha256": _sha(vendored),
                       "patch_sha256": _sha(patch)}
    # These exact patches were reproduced with the pinned refresh.sh sed rules.
    # This binds deletions and include rewrites without executing upstream scripts.
    _require(_sha(_canonical(files)) == _HEADER_MAPPING_SHA256,
             "reviewed five-header source/transform population differs")
    return {"files": files, "cpython_tree": tree}


def verify_karamel_source(directory: Path, *, cpython_archive: bytes | None = None) -> dict:
    """Verify the complete selected runtime source and return its bounded OSV identity.

    Args:
        directory: Existing private directory retaining authenticated source downloads.
        cpython_archive: Optional downloaded archive; the same fixed SHA256 is required.
    Returns:
        Source proof, exact commit query, file/patch hashes and required parent scopes.
        This does not evaluate advisories or accept an image or its installed inventory.
    Raises:
        ValueError: Changed archive, source record, population or reviewed transformation.
        OSError, RuntimeError, tarfile.TarError: Download, storage or archive failure.
    """
    cpython = _source_archive("cpython", directory, cpython_archive)
    hacl = _source_archive("hacl", directory)
    karamel = _source_archive("karamel", directory)
    records = _source_records(cpython, hacl)
    return {
        "component": "karamel-runtime", "method": "verified-vendored-source",
        "query": {"commit": _KARAMEL_COMMIT}, "sources": _SOURCES,
        "source_records": records, "hacl_vendored_tree": _vendor_tree(hacl, karamel),
        **_header_mapping(cpython, karamel),
        "parent_advisory_scope": {"component": "cpython", "version": "3.12.14",
                                  "query": {"cpe": "cpe:2.3:a:python:python:3.12.14:*:*:*:*:*:*:*"}},
        "hacl_advisory_scope": {"component": "hacl", "query": {"commit": _HACL_COMMIT}},
        "limit": "Five runtime headers plus fixed CPython refresh reductions; types.h is a local wrapper. "
                 "The recorded upstream revision has identical vendored bytes, not a unique introduction "
                 "date. Compiler/translator packages are not installed. Parent CPython and HACL "
                 "evaluations remain required; empty OSV results do not prove absence of vulnerabilities.",
    }
