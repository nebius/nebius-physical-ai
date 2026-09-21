"""Static contract checks for the CPU Open3D registration and reconstruction image.

The version floor here exists because a scan of the built image found four fixable
CRITICAL findings -- CVE-2026-58016 in libglib2.0-0t64, and CVE-2026-13221, CVE-2026-42496
and CVE-2026-8376 in perl-base -- against a snapshot that predated Debian 13.7.

Asserting the snapshot timestamp alone would prove nothing: a timestamp is not a version,
and the next person to move it has no way to tell which packages it was moved for. So the
floor is asserted as versions, and the versions are compared the way dpkg compares them.
"""

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa" / "docker" / "workbench" / "open3d"
DOCKERFILE = IMAGE / "Dockerfile"

#: What the image security policy reported installed, and what it reported as fixed.
#: `libglib2.0-0t64` is the t64 package that provides the `libglib2.0-0` the Dockerfile
#: installs by name, and it is the name the scanner and dpkg both use.
SCANNED = {
    "libglib2.0-0t64": {"vulnerable": "2.84.4-3~deb13u3", "fixed": "2.84.4-3~deb13u4"},
    "perl-base": {"vulnerable": "5.40.1-6", "fixed": "5.40.1-6+deb13u1"},
}

#: What trixie actually carries at the pinned snapshot. glib's is *above* the version the
#: scanner named, which is the case a floor has to admit and an equality check would not.
AT_THE_PINNED_SNAPSHOT = {
    "libglib2.0-0t64": "2.84.4-3~deb13u5",
    "perl-base": "5.40.1-6+deb13u1",
}

#: The snapshot the two versions above were read out of, and the identity of the index
#: they were read from. Retained so a reviewer can re-derive the claim instead of taking
#: it, and so that moving the pin fails here: these hashes describe one snapshot and
#: nothing else, and the versions stop being evidence the moment it changes.
PINNED_SNAPSHOT = "20260913T000000Z"
SNAPSHOT_INDEXES = {
    "debian trixie Release": {
        "sha256": "ed56aac47e7911e65ee63aae8d67e29f20e023f840e49f8ed037043a33de138a",
        "bytes": 138612,
        "says": "Version 13.7, dated 2026-09-12, which is the point release the fixes shipped in",
    },
    "debian trixie main/binary-amd64/Packages.xz": {
        "sha256": "7778d3e3f303b7ddb8ce0fe7c8d57473a076c6bf2e8f241f75421d2396352498",
        "bytes": 9678380,
        "says": "named by that Release under SHA256, and lists both versions above",
    },
    "debian-security trixie-security Release": {
        "sha256": "5faa5f143a2cfd1dce82502bd15438fd5a4f24c78696432285bea920999caca6",
        "bytes": 41759,
        "says": "the security suite at the same snapshot, dated 2026-09-12",
    },
}


def _segment_rank(text: str, index: int) -> int:
    """Rank one character of a version, as dpkg orders them.

    `~` sorts before everything including the end of the string, letters sort before
    everything else, and a digit ends the non-digit run rather than taking part in it.
    """

    if index >= len(text):
        return 0
    char = text[index]
    if char.isdigit():
        return 0
    if char == "~":
        return -1
    if char.isalpha():
        return ord(char)
    return ord(char) + 256


def _compare_part(left: str, right: str) -> int:
    i = j = 0
    while i < len(left) or j < len(right):
        while (i < len(left) and not left[i].isdigit()) or (
            j < len(right) and not right[j].isdigit()
        ):
            rank = _segment_rank(left, i) - _segment_rank(right, j)
            if rank:
                return rank
            i += 1
            j += 1
        while i < len(left) and left[i] == "0":
            i += 1
        while j < len(right) and right[j] == "0":
            j += 1
        digits = 0
        while (
            i < len(left)
            and left[i].isdigit()
            and j < len(right)
            and right[j].isdigit()
        ):
            digits = digits or ord(left[i]) - ord(right[j])
            i += 1
            j += 1
        if i < len(left) and left[i].isdigit():
            return 1
        if j < len(right) and right[j].isdigit():
            return -1
        if digits:
            return digits
    return 0


def compare_debian_versions(left: str, right: str) -> int:
    """Negative, zero or positive, as `dpkg --compare-versions` would order them."""

    def split(version: str) -> tuple[int, str, str]:
        epoch, _, rest = (
            version.rpartition(":") if ":" in version else ("0", "", version)
        )
        upstream, _, revision = rest.rpartition("-") if "-" in rest else (rest, "", "")
        return int(epoch), upstream, revision

    left_epoch, left_upstream, left_revision = split(left)
    right_epoch, right_upstream, right_revision = split(right)
    if left_epoch != right_epoch:
        return left_epoch - right_epoch
    return _compare_part(left_upstream, right_upstream) or _compare_part(
        left_revision, right_revision
    )


def _build_arg(text: str, name: str) -> str:
    match = re.search(rf"^ARG {re.escape(name)}=(.+)$", text, re.MULTILINE)
    assert match, f"{name} is not declared as a build arg"
    return match.group(1).strip()


def test_version_ordering_matches_dpkg_where_string_ordering_does_not() -> None:
    # The cases that make a string comparison wrong. Without these the comparator below
    # could be `<` and every other assertion in this module would still pass.
    assert compare_debian_versions("2.84.4-3~deb13u10", "2.84.4-3~deb13u5") > 0
    assert compare_debian_versions("2.84.4-3~deb13u5", "2.84.4-3") < 0
    assert compare_debian_versions("1.0", "1.0-0") == 0
    assert compare_debian_versions("1:1.0", "2.0") > 0


def test_pinned_snapshot_clears_the_versions_the_scan_failed_on() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    floors = {
        "libglib2.0-0t64": _build_arg(text, "MIN_LIBGLIB_VERSION"),
        "perl-base": _build_arg(text, "MIN_PERL_BASE_VERSION"),
    }

    for package, scanned in SCANNED.items():
        floor = floors[package]
        assert floor == scanned["fixed"], (
            f"{package}'s floor must stay at the version the scan named as fixed"
        )
        assert compare_debian_versions(scanned["vulnerable"], floor) < 0, (
            f"{package} {scanned['vulnerable']} must not satisfy the floor"
        )
        assert compare_debian_versions(AT_THE_PINNED_SNAPSHOT[package], floor) >= 0, (
            f"{package} at the pinned snapshot must satisfy the floor"
        )


def test_the_versions_stay_tied_to_the_snapshot_they_were_read_from() -> None:
    # `AT_THE_PINNED_SNAPSHOT` is a claim about one archive state, and the hashes above
    # are its identity. Moving the pin without re-reading the index would leave the
    # claim standing over an archive nobody checked, so it fails here first.
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert _build_arg(text, "DEBIAN_SNAPSHOT") == PINNED_SNAPSHOT, (
        "the snapshot moved; re-read the index and refresh SNAPSHOT_INDEXES and "
        "AT_THE_PINNED_SNAPSHOT from the new signed Release"
    )
    for index in SNAPSHOT_INDEXES.values():
        assert re.fullmatch(r"[0-9a-f]{64}", index["sha256"])
        assert index["bytes"] > 0


def test_the_floor_is_enforced_at_build_time_by_dpkg() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG DEBIAN_SNAPSHOT=" in text
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in text
    assert "snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" in text
    assert "apt-get upgrade -y --no-install-recommends" in text
    # The check has to run against what was installed, using dpkg's ordering. A grep of
    # the snapshot timestamp, or an equality check, would pass while shipping the CVEs.
    assert "dpkg-query --show --showformat=" in text
    assert 'dpkg --compare-versions "${installed}" ge "$2"' in text
    assert '"libglib2.0-0t64 ${MIN_LIBGLIB_VERSION}"' in text
    assert '"perl-base ${MIN_PERL_BASE_VERSION}"' in text
    # No policy escape hatch was added in place of the fix.
    assert ".trivyignore" not in text
    assert "--ignore-unfixed" not in text


def test_mcap_notice_is_delivered_and_bound_to_the_installed_package() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    notice = IMAGE / "notices" / "mcap-LICENSE.txt"

    # mcap is MIT but ships no licence file in its wheel or its sdist, so the image is
    # where the grant has to come from.
    assert notice.is_file()
    raw = notice.read_bytes()
    # The exact upstream grant, byte for byte. A reformatted or re-wrapped copy of a
    # licence is a different document from the one the project published.
    assert len(raw) == 1077
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d"
    assert digest == _build_arg(text, "MCAP_NOTICE_SHA256"), (
        "the retained notice no longer hashes to what the Dockerfile checks"
    )
    assert "MIT License" in notice.read_text(encoding="utf-8")

    assert (
        "COPY --chmod=0444 docker/workbench/open3d/notices/mcap-LICENSE.txt "
        "\\\n    /usr/share/doc/npa-open3d/notices/mcap-LICENSE.txt"
    ) in text
    assert "/usr/share/doc/npa-open3d/THIRD_PARTY_NOTICES.md" in text
    # Both notice paths have to exist with a traversable mode before anything is copied
    # into them, or `--chmod` sets it on the directories too and the runtime user cannot
    # read what it was given. Proved against real builds in test_open3d_notice_delivery.
    assert "/usr/share/doc/npa-open3d /usr/share/doc/npa-open3d/notices" in text
    assert "sha256sum --check --status" in text
    # Bound to the package, not just present: a notice for a version the image no longer
    # installs reads as though someone checked.
    assert "m.version('mcap')" in text
    assert "'${MCAP_NOTICE_VERSION}'" in text
    assert _build_arg(text, "MCAP_NOTICE_VERSION") == "1.4.0"

    index = (IMAGE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "notices/mcap-LICENSE.txt" in index
    assert "b33fa682a5c517b1d213faeabd118e0b4f9d9d93" in index
    assert digest in index


def test_open3d_image_keeps_its_cpu_and_non_root_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    # Open3D's registration and geometry APIs have no CUDA path, so this image must not
    # derive from a base that would have a stage asking for an accelerator it cannot use.
    bases = re.findall(r"^FROM (\S+)", text, re.MULTILINE)
    assert bases == [
        "python:3.11-slim-trixie@sha256:"
        "a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0"
    ]
    # This runtime compiles no bytecode of its own, which is how a `cpython-312.pyc`
    # found in a layer was identifiable as copied from the build host rather than
    # produced here. The exclusion that keeps it out lives in `npa/.dockerignore`.
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "USER ubuntu" in text
    assert "useradd -m -s /bin/bash -u 1000 ubuntu" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert "python -m pip check" in text
    assert 'ENTRYPOINT ["/opt/npa/docker/workbench/open3d/entrypoint.sh"]' in text
