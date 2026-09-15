"""Exercise refusal boundaries for the finite KaRaMeL runtime source proof."""

import difflib
import io
import tarfile

import pytest

from npa.deploy import ncore_karamel_source as source


def _archive(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, raw in entries:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


@pytest.mark.parametrize("entries", [
    [("root/header.h", b"first"), ("root/header.h", b"second")],
    [("root/../header.h", b"data")],
    [("/root/header.h", b"data")],
    [("other/header.h", b"data")],
    [("root/header.h", b"data"), ("other/header.h", b"data")],
    [("root//header.h", b"data")],
    [("root/./header.h", b"data")],
    [("root/header\\file.h", b"data")],
    [("root", b"data")],
    [],
])
def test_archive_refuses_ambiguous_population(entries):
    with pytest.raises(ValueError, match="archive"):
        source._archive_files(_archive(entries), "root")


def test_archive_is_read_as_data():
    raw = _archive([("root/header.h", b"source")])
    assert source._archive_files(raw, "root") == {"header.h": b"source"}


@pytest.mark.parametrize("name", ["cpython", "hacl", "karamel"])
def test_download_hash_checked_before_parsing(monkeypatch, tmp_path, name):
    def download(url, output, **kwargs):
        assert url == source._SOURCES[name]["url"]
        assert kwargs == {"allowed_hosts": frozenset({"codeload.github.com"})}
        output.write(b"substituted source")

    monkeypatch.setattr(source, "download_public_https", download)
    monkeypatch.setattr(source, "_archive_files", lambda *_: pytest.fail("parsed unauthenticated bytes"))
    with pytest.raises(ValueError, match="archive hash"):
        source._source_archive(name, tmp_path)
    assert (tmp_path / (name + ".karamel-source.tar.gz")).read_bytes() == b"substituted source"


def test_supplied_cpython_archive_has_same_hash_requirement(monkeypatch, tmp_path):
    monkeypatch.setattr(source, "download_public_https", lambda *_: pytest.fail("downloaded"))
    with pytest.raises(ValueError, match="archive hash"):
        source.verify_karamel_source(tmp_path, cpython_archive=b"another interpreter")


@pytest.fixture
def reviewed_headers(monkeypatch):
    # Synthetic reviewed input establishes a baseline for deliberate later tampering.
    cpython = {source._CPYTHON_PREFIX + "types.h": b"local wrapper\n"}
    karamel, mapping = {}, {}
    for destination, origin in source._HEADERS.items():
        path = source._CPYTHON_PREFIX + destination
        karamel[origin] = b"/* synthetic upstream */\n#include \"krml/internal/types.h\"\n"
        cpython[path] = b"/* synthetic upstream */\n#include \"krml/types.h\"\n"
        patch = "".join(difflib.unified_diff(
            karamel[origin].decode().splitlines(True), cpython[path].decode().splitlines(True),
            fromfile=origin, tofile=path,
        )).encode()
        mapping[path] = {"upstream_path": origin, "hacl_path": "dist/karamel/" + origin,
                         "upstream_sha256": source._sha(karamel[origin]),
                         "vendored_sha256": source._sha(cpython[path]),
                         "patch_sha256": source._sha(patch)}
    tree = {p: source._sha(raw) for p, raw in cpython.items()}
    monkeypatch.setattr(source, "_CPYTHON_TREE_SHA256", source._sha(source._canonical(tree)))
    monkeypatch.setattr(source, "_HEADER_MAPPING_SHA256", source._sha(source._canonical(mapping)))
    assert len(source._header_mapping(cpython, karamel)["files"]) == 5
    return cpython, karamel


@pytest.mark.parametrize("header", source._HEADERS)
@pytest.mark.parametrize("side", ["upstream", "vendored"])
def test_every_selected_header_is_bound(reviewed_headers, header, side):
    cpython, karamel = reviewed_headers
    if side == "upstream":
        karamel[source._HEADERS[header]] += b"changed\n"
    else:
        cpython[source._CPYTHON_PREFIX + header] += b"changed\n"
    with pytest.raises(ValueError, match="population differs"):
        source._header_mapping(cpython, karamel)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrapper", "omitted-selection"])
def test_complete_header_population_and_wrapper_are_bound(reviewed_headers, monkeypatch, mutation):
    cpython, karamel = reviewed_headers
    if mutation == "missing":
        del cpython[source._CPYTHON_PREFIX + "internal/target.h"]
    elif mutation == "extra":
        cpython[source._CPYTHON_PREFIX + "extra.h"] = b"extra\n"
    elif mutation == "wrapper":
        cpython[source._CPYTHON_PREFIX + "types.h"] += b"changed\n"
    else:
        monkeypatch.setattr(source, "_HEADERS", {
            path: origin for path, origin in source._HEADERS.items() if path != "internal/target.h"
        })
    with pytest.raises(ValueError, match="population differs"):
        source._header_mapping(cpython, karamel)


def test_changed_transform_hash_refuses(reviewed_headers, monkeypatch):
    cpython, karamel = reviewed_headers
    monkeypatch.setattr(source, "_HEADER_MAPPING_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="transform population"):
        source._header_mapping(cpython, karamel)


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra", "upstream"])
def test_entire_hacl_vendor_population_is_bound(monkeypatch, mutation):
    karamel = {f"header-{i}.h": b"synthetic source\n" for i in range(21)}
    hacl = {"dist/karamel/" + p: raw for p, raw in karamel.items()}
    tree = {p: source._sha(raw) for p, raw in karamel.items()}
    monkeypatch.setattr(source, "_HACL_TREE_SHA256", source._sha(source._canonical(tree)))
    assert source._vendor_tree(hacl, karamel) == tree
    if mutation == "missing":
        del hacl["dist/karamel/header-0.h"]
    elif mutation == "extra":
        hacl["dist/karamel/extra.h"] = b"extra\n"
    elif mutation == "changed":
        hacl["dist/karamel/header-0.h"] += b"changed\n"
    else:
        karamel["header-0.h"] += b"changed\n"
    with pytest.raises(ValueError, match="differs"):
        source._vendor_tree(hacl, karamel)


def test_source_record_cannot_be_a_moving_branch(monkeypatch):
    cpython = {"Modules/_hacl/refresh.sh": b"pinned refresh\n"}
    hacl = {"Makefile": b"copy source\n", "dist/gcc-compatible/INFO.txt": b"Karamel version: origin/master\n"}
    monkeypatch.setattr(source, "_RECORDS", {
        "cpython": {p: source._sha(raw) for p, raw in cpython.items()},
        "hacl": {p: source._sha(raw) for p, raw in hacl.items()},
    })
    with pytest.raises(ValueError, match="recorded KaRaMeL revision"):
        source._source_records(cpython, hacl)
