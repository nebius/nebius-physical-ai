"""Evaluate the authenticated non-Debian population of the selected NCore image."""

import tarfile

from image_byte_scan import core as W
from . import provenance
from .process import ROOT


def verify(directory, digest, graph, source_sha):
    """Require real component vulnerability and delivered-license coverage.

    Args:
        directory: Private directory with the authenticated merged rootfs.
        digest: Original verified OCI index digest.
        graph: Verified platform, config and ordered layers.
        source_sha: Exact committed source used for the image.
    Returns:
        Fresh component scan receipt after every coverage and policy check.
    Raises:
        ValueError, OSError: Source, component, scanner or coverage failure.
    """
    from npa.deploy.ncore_component_scan import scan_archive

    path = directory / "rootfs.tar"
    return scan_archive(
        path, directory / "components",
        base_lock=(ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes(),
        source_lock=(ROOT / "npa/docker/workbench/ncore/source-lock.json").read_bytes(),
        source_sha=source_sha, committed_files=_committed_files(path, source_sha),
        image_digest=digest, platform_digest=graph["image_manifest_digest"],
        config_digest=graph["image_config_digest"],
    )


def _committed_files(path, source_sha):
    from npa.deploy.ncore_selected_sbom import _archive_members

    result = {}
    # The primary shipped-source gate also enforces its mandatory population.
    provenance.shipped_source(path, source_sha)
    with tarfile.open(path) as archive:
        members = _archive_members(archive)
        for name, member in members.items():
            if member.isdir():
                continue
            expected = _expected_digest(name, source_sha, archive, members)
            if expected is None:
                continue
            W.require(member.isfile(), "component_source_must_be_regular")
            W.require(W.sha(archive.extractfile(member).read()) == expected,
                      "component_source_changed")
            result[name] = expected
    return result


def _expected_digest(name, sha, archive, members):
    source = provenance._source_path(name)
    notices = "usr/share/doc/npa-ncore/notices/"
    if name.startswith(notices):
        source = ROOT / "npa/docker/workbench/ncore/notices" / name.removeprefix(notices)
    if source is not None:
        return provenance._committed_digest(source, sha)
    if name == "usr/share/doc/npa-ncore/npa-source-sha":
        return W.sha((sha + "\n").encode())
    if name == "opt/venv/lib/python3.12/site-packages/npa-source.pth":
        return W.sha(b"/opt/npa/src\n")
    if name == "usr/share/doc/npa-ncore/LICENSE-APACHE-2.0":
        # This duplicate is copied from the staged, hash-verified NCore source.
        # Its upstream archive/patch authentication remains a separate gate.
        member = members.get("opt/ncore/src/ncore/LICENSE")
        W.require(member is not None and member.isfile(), "upstream_license_copy_source_required")
        return W.sha(archive.extractfile(member).read())
    return None
