"""Authenticate the finite upstream source and release-review scopes of NCore's vendored libraries."""

from __future__ import annotations

import difflib
import io
import json
from pathlib import Path
import tarfile

from npa._public_https import download_public_https
from npa.deploy import ncore_component_inventory as inventory
from npa.deploy.ncore_karamel_source import verify_karamel_source

_SOURCE_HOSTS = frozenset({"codeload.github.com", "www.bytereef.org"})
_MPDECIMAL_REVIEW = {
    "url": "https://www.bytereef.org/mpdecimal/changelog.html",
    "sha256": "00a5a2d83efd91425faea4fb14975acac94a3330ff6218b1462e0aa61d43b6d8",
    "installed_version": "2.5.1",
    "reviewed_through": "4.0.1",
    "scope": "Official upstream release notes; this is not an exhaustive vulnerability database",
    "dispositions": [
        {"versions": "1.2 through 2.5.1", "status": "included-in-installed-release",
         "reason": "The reviewed release includes earlier upstream fixes; CPython deltas are separately bound"},
        {"versions": "4.0.0", "status": "upstream-nonsecurity-change",
         "reason": "The z format and transcendental Subnormal/Underflow flag changes are described as features"},
        {"versions": "4.0.0", "status": "upstream-nonsecurity-change",
         "reason": "Nonzero-status handling in mpd_qset_string_exact/i64_exact/u64_exact is a reliability fix"},
        {"versions": "4.0.0", "status": "outside-vendored-file-scope",
         "reason": "Decimal.shiftl/shiftr/ln10 validation changes libmpdec++, which CPython does not vendor"},
        {"versions": "4.0.0 and 4.0.1", "status": "outside-vendored-file-scope",
         "reason": "Remaining changes concern upstream build/install machinery, tests and C++ wrappers"},
    ],
}


def _fetch(source, directory, stem):
    output = io.BytesIO()
    download_public_https(source["url"], output, allowed_hosts=_SOURCE_HOSTS)
    raw = output.getvalue()
    (directory / (stem + ".download")).write_bytes(raw)
    inventory._require(inventory._sha(raw) == source["sha256"],
                       "upstream source or release review changed: " + stem)
    return raw


def _archive_files(raw):
    files, roots = {}, set()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            name = inventory.selected._path(member.name)
            roots.add(name.split("/")[0])
            if member.isdir():
                continue
            # Symlinks outside the vendored scope are not imported or executed.
            if not member.isfile():
                continue
            inventory._require("/" in name, "source archive has no root directory")
            relative = name.split("/", 1)[1]
            inventory._require(relative not in files, "duplicate source file")
            files[relative] = archive.extractfile(member).read()
    inventory._require(len(roots) == 1 and files, "source archive root differs")
    return files


def _file_mapping(profile, cpython, upstream):
    prefix = profile["vendor_prefix"]
    files = {}
    for path, raw in sorted(cpython.items()):
        if not path.startswith(prefix):
            continue
        relative = path.removeprefix(prefix)
        origin = profile["upstream_prefix"] + profile["renames"].get(relative, relative)
        inventory._require(origin in upstream, "vendored file has no upstream source: " + path)
        original = upstream[origin]
        patch = "".join(difflib.unified_diff(
            original.decode().splitlines(True), raw.decode().splitlines(True),
            fromfile=origin, tofile=path,
        )).encode()
        files[path] = {"upstream_path": origin, "upstream_sha256": inventory._sha(original),
                       "vendored_sha256": inventory._sha(raw), "patch_sha256": inventory._sha(patch)}
    tree_prefix = prefix.rsplit("/", 2)[0] + "/"
    tree = {path: inventory._sha(raw) for path, raw in cpython.items() if path.startswith(tree_prefix)}
    inventory._require(inventory._sha(inventory._canonical(files)) == profile["files_sha256"],
                       "vendored upstream file/patch population differs")
    inventory._require(inventory._sha(inventory._canonical(tree)) == profile["cpython_tree_sha256"],
                       "CPython wrapper/source population differs")
    return {"files": files, "cpython_tree": tree}


def _mpdecimal_identity(cpython, profile):
    document = json.loads(cpython["Misc/sbom.spdx.json"])
    packages = [row for row in document["packages"] if row["SPDXID"] == "SPDXRef-PACKAGE-mpdecimal"]
    inventory._require(len(packages) == 1, "CPython mpdecimal identity missing or duplicated")
    package = packages[0]
    inventory._require(package["name"] == "mpdecimal" and package["versionInfo"] == "2.5.1"
                       and package["downloadLocation"] == profile["upstream"]["url"]
                       and {"algorithm": "SHA256", "checksumValue": profile["upstream"]["sha256"]}
                       in package["checksums"]
                       and {"referenceCategory": "SECURITY", "referenceType": "cpe23Type",
                            "referenceLocator": profile["query"]["cpe"]} in package["externalRefs"],
                       "CPython mpdecimal source/advisory identity differs")
    return {"path": "Misc/sbom.spdx.json", "sha256": inventory._sha(cpython["Misc/sbom.spdx.json"]),
            "package": package,
            "limit": "PSF-declared CPE; independent NVD dictionary recognition is not established"}


def _component_proof(component, cpython, directory):
    name = component["name"]
    profile = inventory._BUNDLED_SOURCE_PROFILES[name]
    inventory._require(component.get("source_mapping") == profile
                       and component["query"] == profile["query"], "unreviewed bundled source mapping")
    upstream = _archive_files(_fetch(profile["upstream"], directory, name + ".upstream"))
    proof = {"profile": profile, "cpython_source": inventory._CPYTHON_SOURCE,
             "cpython_commit": inventory._CPYTHON_COMMIT,
             "parent_advisory_scope": {"component": "cpython", "version": inventory._CPYTHON_VERSION,
                                       "query": {"cpe": "cpe:2.3:a:python:python:"
                                                 + inventory._CPYTHON_VERSION + ":*:*:*:*:*:*:*"}},
             **_file_mapping(profile, cpython, upstream)}
    if name == "mpdecimal":
        proof["identity_evidence"] = _mpdecimal_identity(cpython, profile)
        _fetch(_MPDECIMAL_REVIEW, directory, "mpdecimal.release-review")
        proof["upstream_review"] = _MPDECIMAL_REVIEW
    return proof


def _karamel_proof(component, cpython_archive, directory):
    profile = inventory._KARAMEL_SOURCE_PROFILE
    inventory._require(component.get("source_mapping") == profile
                       and component["query"] == profile["query"]
                       and component["version"] == inventory._HACL_COMMIT,
                       "unreviewed KaRaMeL source mapping")
    proof = verify_karamel_source(directory, cpython_archive=cpython_archive)
    inventory._require(proof["sources"] == profile["sources"]
                       and proof["source_records"] == profile["source_records"]
                       and all(proof[key] == profile[key] for key in (
                           "query", "parent_advisory_scope", "hacl_advisory_scope"))
                       and all(inventory._sha(inventory._canonical(proof[key])) == profile[digest]
                               for key, digest in (("files", "files_sha256"),
                                                   ("hacl_vendored_tree", "hacl_tree_sha256"),
                                                   ("cpython_tree", "cpython_tree_sha256")))
                       and inventory._sha(json.dumps(proof, indent=2).encode() + b"\n")
                       == profile["source_proof_sha256"], "KaRaMeL source proof differs")
    return proof


def verify_bundled_sources(components: list[dict], directory: Path) -> dict:
    """Reproduce reviewed upstream identities and finite patch scopes from public sources.

    Args:
        components: Inventory bound to the reviewed interpreter's actual ELF hashes.
        directory: Existing private evidence directory; downloads are retained as data.
    Returns:
        Per-component proof with every source-file hash, patch hash and parent scope.
    Raises:
        ValueError: Changed source population, mapping, identity or upstream release notes.
        OSError, KeyError, TypeError, RuntimeError: Download, archive or schema failure.
    """
    expected = set(inventory._BUNDLED_SOURCE_PROFILES) | {"karamel-runtime"}
    bundled = [row for row in components if row["name"] in expected]
    inventory._require(len(bundled) == len(expected)
                       and {row["name"] for row in bundled} == expected,
                       "bundled source population differs")
    raw = _fetch(inventory._CPYTHON_SOURCE, directory, "cpython.upstream")
    cpython = _archive_files(raw)
    return {row["name"]: (_karamel_proof(row, raw, directory) if row["name"] == "karamel-runtime"
                           else _component_proof(row, cpython, directory)) for row in bundled}
