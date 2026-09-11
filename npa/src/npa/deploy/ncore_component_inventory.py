"""Bind NCore's non-Debian component population to a read-only merged image tar."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import tarfile

from npa.deploy import ncore_karamel_source as karamel
from npa.deploy import ncore_selected_sbom as selected

_CPYTHON_COMMIT = "2abcf904b8dac8c999d2b3aac76681abb333798a"
_CPYTHON_VERSION = "3.12.14"
_CPYTHON_SOURCE = {
    "url": "https://codeload.github.com/python/cpython/tar.gz/" + _CPYTHON_COMMIT,
    "sha256": "72b4d8791b053a808f247e0de247a8b976868e705204e8fcab0812c823873766",
}
_LIBB2_COMMIT = "620681a3b15c4d7239b9323b9da5ea208a959d3d"
# The mapping hashes cover every vendored file, its upstream counterpart and
# exact diff. Wrapper-tree hashes keep CPython adaptations in the parent scope.
_BUNDLED_SOURCE_PROFILES = {
    "mpdecimal": {
        "upstream": {
            "url": "https://www.bytereef.org/software/mpdecimal/releases/mpdecimal-2.5.1.tar.gz",
            "sha256": "9f9cd4c041f99b5c49ffb7b59d9f12d95b683d88585608aa56a6307667b2b21f",
        },
        "vendor_prefix": "Modules/_decimal/libmpdec/", "upstream_prefix": "libmpdec/",
        "renames": {"mpdecimal.h": "mpdecimal.h.in"},
        "files_sha256": "f897eb5792e5c2db314bf8084494f5f8215ccbd86f60d9aabb212d1e4abfa764",
        "cpython_tree_sha256": "f68b36a60e4e2e2e7662fbdeded2a7dbcc2a6c2b65434dc03d98570a5c871652",
        "query": {"cpe": "cpe:2.3:a:bytereef:mpdecimal:2.5.1:*:*:*:*:*:*:*"},
        "identity_source": "Misc/sbom.spdx.json#SPDXRef-PACKAGE-mpdecimal",
    },
    "blake2": {
        "upstream": {
            "url": "https://codeload.github.com/BLAKE2/libb2/tar.gz/" + _LIBB2_COMMIT,
            "sha256": "92e0566634e3fd89407e5d959e3022b04f97d23c9e2e1d253604b6a0b7bfde1f",
        },
        "vendor_prefix": "Modules/_blake2/impl/", "upstream_prefix": "src/", "renames": {},
        "files_sha256": "843e06bc12707e25330553f8cb8a5725a51f69f2b2c40d30b2464ba3d7d1d64a",
        "cpython_tree_sha256": "fdfff2e1a9e1970f3f554cfa945b634e04a2f0ac7b94f73178902a879e192708",
        "query": {"commit": _LIBB2_COMMIT},
        "identity_source": "https://github.com/python/cpython/pull/6286",
    },
}
_HACL_COMMIT = "bb3d0dc8d9d15a5cd51094d5b69e70aa09005ff0"
_KARAMEL_SOURCE_PROFILE = {
    "sources": copy.deepcopy(karamel._SOURCES),
    "source_records": copy.deepcopy(karamel._RECORDS),
    "files_sha256": karamel._HEADER_MAPPING_SHA256,
    "hacl_tree_sha256": karamel._HACL_TREE_SHA256,
    "cpython_tree_sha256": karamel._CPYTHON_TREE_SHA256,
    "query": {"commit": karamel._KARAMEL_COMMIT},
    "parent_advisory_scope": {"component": "cpython", "version": _CPYTHON_VERSION,
                              "query": {"cpe": "cpe:2.3:a:python:python:"
                                        + _CPYTHON_VERSION + ":*:*:*:*:*:*:*"}},
    "hacl_advisory_scope": {"component": "hacl", "query": {"commit": _HACL_COMMIT}},
    # Exact retained JSON produced by the standalone source verifier, before
    # any advisory evaluation. Its bytes are independent of the image receipt.
    "source_proof_sha256": "362fa6f8dfa800e7145d8b9c3efe0b74cc0ad018f5a1bce842afa54ef8669c98",
}
_CPYTHON_ELF_INVENTORY = "fd9c96ddcf94b6e8cc74d04e0d555bf6964c47f0117514def0cc8d7ac180e698"
_SOURCE_INVENTORY = "opt/ncore/src/source-inventory.json"
_SOURCE_LOCK = "usr/share/doc/npa-ncore/source-lock.json"
_LICENSE_ROOT = "usr/share/doc/npa-ncore/cpython/"
_COMPONENT_NAMES = frozenset({
    "cpython", "expat", "mpdecimal", "hacl", "karamel-runtime", "blake2",
    "ncore", "pycolmap", "npa",
})

# These are assembled OS configuration/links, not additional software products.
# New paths require classification, even when another scanner finds nothing.
_ASSEMBLY_FILES = frozenset("""
bin lib lib64 sbin usr/bin/awk usr/bin/nc usr/bin/sh usr/bin/which
etc/apt/apt.conf.d/90npa-bootstrap etc/apt/apt.conf.d/99snapshot
etc/apt/sources.list.d/debian.sources etc/default/ssh etc/group etc/gshadow
etc/hosts etc/ld.so.cache etc/ld.so.conf etc/localtime etc/nsswitch.conf
etc/os-release etc/pam.d/common-account etc/pam.d/common-auth
etc/pam.d/common-password etc/pam.d/common-session
etc/pam.d/common-session-noninteractive etc/passwd etc/shadow
etc/ssh/sshd_config etc/ssh/sshd_config.d/npa.conf
etc/ssl/certs/ca-certificates.crt etc/sudoers etc/sudoers.d/ubuntu
var/cache/ldconfig/aux-cache var/lib/dpkg/status var/lock var/run
""".split())
_VENV_FILES = frozenset("""
opt/venv/bin/Activate.ps1 opt/venv/bin/activate opt/venv/bin/activate.csh
opt/venv/bin/activate.fish opt/venv/bin/python opt/venv/bin/python3
opt/venv/bin/python3.12 opt/venv/lib64 opt/venv/pyvenv.cfg
""".split())

# Exact upstream notice bytes, not inferred SPDX grants. C/H entries are the
# first complete comment plus one LF from CPython v3.12.14's pinned commit.
_NOTICES = {
    "cpython": ("usr/local/lib/python3.12/LICENSE.txt",
                "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf"),
    "cpython-incorporated": (_LICENSE_ROOT + "LICENSE.third-party",
                            "341832873fd316a37927e79385093fbbfd40a467428480835fe435a80cadf4e5"),
    "expat": (_LICENSE_ROOT + "LICENSE.expat",
              "31b15de82aa19a845156169a17a5488bf597e561b2c318d159ed583139b25e87"),
    "mpdecimal": (_LICENSE_ROOT + "LICENSE.mpdecimal",
                  "b07528d8b1dbf1e2d2741052996f0876e23342ce2d30d0effa39c5457716c25a"),
    "hacl": (_LICENSE_ROOT + "LICENSE.hacl",
             "998ce04fb8ad9dedb0bc1b44938f8c3dcf1089780fa105ce1c8c30fb5554d78c"),
    "hacl-krml": (_LICENSE_ROOT + "LICENSE.hacl-krml",
                  "6b7cb4bd5441f287eeb8b745073c11bc5d2974252bec3b6d8c101ff67f918a26"),
    "hacl-fstar": (_LICENSE_ROOT + "LICENSE.hacl-fstar",
                   "54079ca9757dbaa2a54053fbd9be60f77c5e06695fa997191fc141cd20d41d59"),
    "blake2-python": (_LICENSE_ROOT + "LICENSE.blake2-python",
                      "74baf34496c290dd3b44efca41d040893117c9c1a901c2831b629c9cf235ceb1"),
    "blake2": (_LICENSE_ROOT + "LICENSE.blake2",
               "a27d7169aa4b7db350c5e46da87184c4b7a8ac8e4d86dce38845fdffd789d7b5"),
}


def _require(condition, reason):
    if not condition:
        raise ValueError("NCore component scan: " + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read(archive, members, name):
    _require(name in members and members[name].isfile(), "regular file required: " + name)
    return archive.extractfile(members[name]).read()


def _files(archive, members):
    files = {}
    for name, member in members.items():
        _require(not any(part.endswith((".dist-info", ".egg-info", ".whl", ".egg"))
                         for part in Path(name).parts),
                 "unexpected installed/archive Python distribution: " + name)
        if member.isdir():
            continue
        if member.issym():
            files[name] = {"link": member.linkname}
            continue
        _require(member.isfile(), "unsupported member kind: " + name)
        with archive.extractfile(member) as stream:
            prefix = stream.read(4)
            digest = hashlib.sha256(prefix)
            while block := stream.read(1024 * 1024):
                digest.update(block)
        files[name] = {"sha256": digest.hexdigest(), "elf": prefix == b"\x7fELF"}
    return files


def _match(files, expected):
    for name, identity in expected.items():
        kind = "link" if "link" in identity else "sha256"
        _require(files.get(name, {}).get(kind) == identity[kind], "file identity differs: " + name)


def _python_profile(lock):
    components = [item for item in lock["components"] if item["kind"] != "debian-source"]
    _require(len(components) == 1 and all(components[0][key] == value for key, value in {
        "id": "cpython:" + _CPYTHON_VERSION, "kind": "cpython", "name": "cpython",
        "version": _CPYTHON_VERSION,
    }.items()),
             "unreviewed non-Debian base component/version")
    _require(not lock["python_distributions"] and not lock["python_wheels"],
             "installed Python dependencies need component coverage")
    _require(_sha(_canonical(lock["cpython_elf_files"])) == _CPYTHON_ELF_INVENTORY,
             "CPython embedded component profile needs review")
    _notice_profile(lock)
    files = {item["path"]: item for item in lock["cpython_files"]}
    _match(files, {item["path"]: item for item in lock["cpython_elf_files"]})
    return files


def _verify_python_elf(files, python_files):
    observed = [{"path": path, "sha256": files[path]["sha256"]} for path in sorted(python_files)
                if files.get(path, {}).get("elf")]
    _require(_sha(_canonical(observed)) == _CPYTHON_ELF_INVENTORY,
             "actual CPython ELF population differs from reviewed source profile")


def _notice_profile(lock):
    provenance = lock["cpython_provenance"]
    _require(provenance["source_commit"] == _CPYTHON_COMMIT
             and provenance["version"] == "3.12.14"
             and provenance["embedded_components"] == {
                 "expat": "2.8.3", "mpdecimal": "2.5.1",
                 "hacl": _HACL_COMMIT, "blake2": _CPYTHON_COMMIT,
             }, "CPython source/component provenance differs")
    expected = {path: digest for path, digest in _NOTICES.values() if path.startswith(_LICENSE_ROOT)}
    rows = provenance["notices"]
    _require(len(rows) == len(expected)
             and {row["path"]: row["sha256"] for row in rows} == expected,
             "CPython bundled notice profile differs")


def _source_files(archive, members, source_lock):
    _require(_read(archive, members, _SOURCE_LOCK) == source_lock, "source lock differs")
    inventory = json.loads(_read(archive, members, _SOURCE_INVENTORY))
    sources = json.loads(source_lock)
    _require(inventory["sources"] == sources, "source revisions differ")
    _require(set(sources) == {"format", "ncore", "pycolmap"}, "source population differs")
    expected = {}
    for name, digest in inventory["files"].items():
        name = selected._path(name)
        _require(name.startswith(("ncore/", "pycolmap/")), "unexpected upstream source")
        _require(re.fullmatch(r"[0-9a-f]{64}", digest), "invalid source file hash")
        expected["opt/ncore/src/" + name] = {"sha256": digest}
    _require(expected, "empty upstream source inventory")
    return sources, expected


def _classify(files, base_files, python_files, upstream, committed_files):
    _match(files, base_files)
    _match(files, python_files)
    _match(files, upstream)
    _match(files, {name: {"sha256": digest} for name, digest in committed_files.items()})
    groups = {"debian": [], "cpython": [], "ncore": [], "pycolmap": [], "npa": [], "assembly": []}
    notices = {path for path, _ in _NOTICES.values()}
    for name, identity in files.items():
        group = _file_group(name, base_files, python_files, upstream, committed_files, notices)
        _require(group is not None, "unclassified shipped file: " + name)
        _require(not identity.get("elf") or name in base_files or name in python_files,
                 "unclassified native component: " + name)
        groups[group].append(name)
    _require(all(groups[group] for group in ("cpython", "ncore", "pycolmap", "npa")),
             "required component population missing")
    return groups


def _file_group(name, base, python, upstream, committed, notices):
    if name in python or name in _VENV_FILES or name in notices:
        return "cpython"
    if name in base:
        return "debian"
    if name in upstream:
        return name.split("/")[3]
    if name in committed:
        return "npa"
    if name in _ASSEMBLY_FILES or name in (selected.LOCK_PATH, _SOURCE_INVENTORY):
        return "assembly"
    return None


def _component(name, version, paths, files, query, license_path, license_hash=None):
    _require(paths, "component has no shipped files: " + name)
    return {"name": name, "version": version, "files": sorted(paths),
            "files_sha256": _sha(_canonical({p: files[p] for p in sorted(paths)})),
            "query": query, "license_path": license_path,
            "license_sha256": license_hash or files.get(license_path, {}).get("sha256")}


def _components(groups, files, sources, source_sha):
    cp_query = {"cpe": "cpe:2.3:a:python:python:" + _CPYTHON_VERSION + ":*:*:*:*:*:*:*"}
    components = [_component("cpython", _CPYTHON_VERSION, groups["cpython"], files,
                             cp_query, *_NOTICES["cpython"])]
    bundled = {
        "expat": ("2.8.3", ("/pyexpat.",), {"cpe": "cpe:2.3:a:libexpat:expat:2.8.3:*:*:*:*:*:*:*"}),
        "mpdecimal": ("2.5.1", ("/_decimal.",), _BUNDLED_SOURCE_PROFILES["mpdecimal"]["query"]),
        "hacl": (_HACL_COMMIT, ("/_md5.", "/_sha1.", "/_sha2.", "/_sha3."), {"commit": _HACL_COMMIT}),
        "blake2": (_CPYTHON_COMMIT, ("/_blake2.",), _BUNDLED_SOURCE_PROFILES["blake2"]["query"]),
    }
    for name, (version, markers, query) in bundled.items():
        paths = [p for p in groups["cpython"] if any(marker in p for marker in markers)]
        components.append(_component(name, version, paths, files, query, *_NOTICES[name]))
        if name in _BUNDLED_SOURCE_PROFILES:
            components[-1]["source_mapping"] = copy.deepcopy(_BUNDLED_SOURCE_PROFILES[name])
    # The installed snapshot describes HACL's copy; the separately verified
    # KaRaMeL revision identifies the upstream advisory scope of these helpers.
    paths = next(row["files"] for row in components if row["name"] == "hacl")
    components.append(_component("karamel-runtime", _HACL_COMMIT, paths, files,
                                 copy.deepcopy(_KARAMEL_SOURCE_PROFILE["query"]), *_NOTICES["hacl-krml"]))
    components[-1]["source_mapping"] = copy.deepcopy(_KARAMEL_SOURCE_PROFILE)
    for name in ("ncore", "pycolmap"):
        version = sources[name]["revision"]
        _require(re.fullmatch(r"[0-9a-f]{40}", version), "full source revision required")
        notice = "opt/ncore/src/" + name + ("/LICENSE" if name == "ncore" else "/LICENSE.txt")
        components.append(_component(name, version, groups[name], files, {"commit": version}, notice))
    notice = "usr/share/doc/npa-ncore/notices/NPA-LICENSE"
    components.append(_component("npa", source_sha, groups["npa"], files, {"commit": source_sha}, notice))
    return components


def inventory_archive(path: Path, *, base_lock: bytes, source_lock: bytes,
                      source_sha: str, committed_files: dict[str, str]) -> dict:
    """Partition every shipped file and bind nine non-Debian component identities.

    Args:
        path: Merged rootfs tar; the caller must prove its OCI graph derivation.
        base_lock: Reviewed committed base lock, compared byte-for-byte to delivery.
        source_lock: Reviewed committed NCore/pycolmap lock.
        source_sha: Full reviewed NPA commit.
        committed_files: Image path to committed SHA256 for every NPA-owned file;
            the caller's source-closure gate must derive/authenticate this mapping.
    Returns:
        File/component inventory; missing notices are findings, never a pass.
    Raises:
        ValueError: Unknown components, changed files, or unsupported CPython profile.
        OSError, KeyError, TypeError: Missing or malformed required inputs.
    """
    _require(re.fullmatch(r"[0-9a-f]{40}", source_sha) and committed_files,
             "committed source closure required")
    with tarfile.open(path) as archive:
        members = selected._archive_members(archive)
        _require(_read(archive, members, selected.LOCK_PATH) == base_lock, "base lock differs")
        lock, _, base_files = selected._inventory(base_lock)
        python_files = _python_profile(lock)
        sources, upstream = _source_files(archive, members, source_lock)
        files = _files(archive, members)
    groups = _classify(files, base_files, python_files, upstream, committed_files)
    _verify_python_elf(files, python_files)
    components = _components(groups, files, sources, source_sha)
    return {"format": "npa_ncore_component_inventory_v1", "components": components,
            "files": files, "groups": groups, "base_lock_sha256": _sha(base_lock),
            "source_lock_sha256": _sha(source_lock), "source_sha": source_sha,
            "population_sha256": _sha(_canonical(components)),
            "required_notices": {name: {"path": path, "sha256": digest}
                                 for name, (path, digest) in _NOTICES.items()}}
