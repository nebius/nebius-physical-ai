"""Validate notice bytes and directory traversal as the image's non-root user.

A retained image build failed to read its MCAP grant because a parent directory
was not traversable. Root could still read it. The positive controls use the
Dockerfile's actual notice-delivery instructions and retained licence bytes.

BuildKit versions differ in the parent-directory modes created by COPY --chmod.
The negative controls therefore remove traversal permission explicitly, rather
than requiring every builder to reproduce one historical COPY implementation.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa" / "docker" / "workbench" / "open3d"
DOCKERFILE = IMAGE / "Dockerfile"

NOTICE_DIR = Path("/usr/share/doc/npa-open3d")
MCAP_NOTICE = NOTICE_DIR / "notices" / "mcap-LICENSE.txt"
INDEX = NOTICE_DIR / "THIRD_PARTY_NOTICES.md"
MCAP_SHA256 = "da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="a file mode only exists once Docker has copied something; there is nothing to ask without a daemon",
)


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _base_image() -> str:
    match = re.search(r"^FROM (\S+)", _dockerfile(), re.MULTILINE)
    assert match, "no FROM line"
    return match.group(1)


def _notice_copies() -> list[str]:
    """The Dockerfile's own notice COPY instructions, line continuations joined."""

    joined = _dockerfile().replace("\\\n", " ")
    return [
        " ".join(line.split())
        for line in joined.splitlines()
        if line.startswith("COPY") and "/usr/share/doc/npa-open3d" in line
    ]


def _run_as_ubuntu(*commands: str) -> str:
    """Read the notices as uid 1000, letting a failed read fail the build.

    Deliberately not `... || echo DENIED`: Docker echoes each instruction into the build
    output, so a sentinel in the command text is present whether or not it ran. The exit
    status is the only signal here that cannot be faked by the log, and it is also what
    the real build did.
    """

    return "\n".join(["USER ubuntu", *(f"RUN {command}" for command in commands)])


#: The same two reads the image performs, in the same form: `/bin/sh` here is dash, and
#: the Dockerfile pipes into `sha256sum` rather than using a bash herestring.
READS = (
    f"echo '{MCAP_SHA256}  {MCAP_NOTICE}' | sha256sum --check --status",
    f"cat {INDEX} > /dev/null",
)


def _build(dockerfile_body: str) -> tuple[int, str]:
    """Build a context holding only the real notice bytes; return exit status and output."""

    with tempfile.TemporaryDirectory() as workspace:
        context = Path(workspace)
        staged = context / "docker" / "workbench" / "open3d"
        (staged / "notices").mkdir(parents=True)
        shutil.copy2(
            IMAGE / "THIRD_PARTY_NOTICES.md", staged / "THIRD_PARTY_NOTICES.md"
        )
        shutil.copy2(
            IMAGE / "notices" / "mcap-LICENSE.txt",
            staged / "notices" / "mcap-LICENSE.txt",
        )
        (context / "Dockerfile").write_text(dockerfile_body)
        finished = subprocess.run(
            [
                "docker",
                "build",
                "--no-cache",
                "--progress=plain",
                "--output",
                "type=cacheonly",
                ".",
            ],
            cwd=context,
            capture_output=True,
            text=True,
            timeout=900,
        )
        return finished.returncode, finished.stdout + finished.stderr


def _prepare() -> str:
    """Replay the Dockerfile's own directory creation, not a convenient copy of it.

    The mode is only correct because the Dockerfile makes these directories before
    anything is copied into them. Hardcoding that here would leave the tests passing
    if it were dropped from the source, which is the half of the fix that is easy to
    lose.
    """

    joined = _dockerfile().replace("\\\n", " ")
    match = re.search(r"install -d -m 0755((?:\s+/\S+)+)", joined)
    assert match, (
        "the Dockerfile no longer creates its directories with an explicit mode"
    )
    directories = match.group(1).split()
    return (
        "RUN useradd -m -s /bin/bash -u 1000 ubuntu"
        f" && install -d -m 0755 {' '.join(directories)}"
    )


@requires_docker
@pytest.mark.parametrize("read", READS, ids=["mcap-grant-hash", "notices-index"])
def test_notices_are_readable_by_the_user_the_image_runs_as(read: str):
    # The Dockerfile's own COPY lines, not a paraphrase of them, so editing those lines
    # changes what this test exercises.
    status, output = _build(
        "\n".join(
            [
                f"FROM {_base_image()}",
                _prepare(),
                *_notice_copies(),
                _run_as_ubuntu(read),
            ]
        )
    )
    assert status == 0, output[-3000:]


@requires_docker
@pytest.mark.parametrize(
    ("directory", "read"),
    [
        (NOTICE_DIR, READS[0]),
        (NOTICE_DIR, READS[1]),
        (NOTICE_DIR / "notices", READS[0]),
    ],
    ids=["mcap-parent", "index-parent", "mcap-nested-parent"],
)
def test_nontraversable_notice_directory_blocks_runtime_user(
    directory: Path, read: str
):
    # Start from valid delivery so a failed read is attributable to the one bad mode.
    status, output = _build(
        "\n".join(
            [
                f"FROM {_base_image()}",
                _prepare(),
                *_notice_copies(),
                f"RUN chmod 0444 {directory}",
                _run_as_ubuntu(read),
            ]
        )
    )
    assert status != 0, "the runtime user read through a nontraversable directory"
    assert "Permission denied" in output, output[-3000:]


@requires_docker
def test_the_notice_directories_are_traversable_and_the_files_stay_read_only():
    status, output = _build(
        "\n".join(
            [
                f"FROM {_base_image()}",
                _prepare(),
                *_notice_copies(),
                f'RUN stat -c "MODE %A %n" {NOTICE_DIR} {NOTICE_DIR / "notices"}'
                f" {INDEX} {MCAP_NOTICE}",
            ]
        )
    )
    assert status == 0, output[-3000:]
    assert f"MODE drwxr-xr-x {NOTICE_DIR}" in output, output[-3000:]
    assert f"MODE drwxr-xr-x {NOTICE_DIR / 'notices'}" in output
    # Read-only files: the grant must not be rewritable by the runtime user either.
    assert f"MODE -r--r--r-- {INDEX}" in output
    assert f"MODE -r--r--r-- {MCAP_NOTICE}" in output


def test_every_retained_notice_has_its_own_copy_instruction():
    # One file per COPY is what keeps the directory mode correct, and it is also how a
    # newly retained notice gets silently left out of the image. This is the check that
    # stops that, and it needs no daemon.
    copied = " ".join(_notice_copies())
    for notice in sorted((IMAGE / "notices").iterdir()):
        relative = f"docker/workbench/open3d/notices/{notice.name}"
        assert relative in copied, (
            f"{notice.name} is retained but never copied into the image"
        )
    assert "docker/workbench/open3d/notices/ " not in copied, (
        "the notices directory is being copied as a directory again; "
        "--chmod can make its parents untraversable on some builders"
    )


def test_the_hash_the_image_checks_is_still_the_retained_grant():
    raw = (IMAGE / "notices" / "mcap-LICENSE.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MCAP_SHA256
    assert len(raw) == 1077
    # The in-image check is the read that failed, so it must stay in the build.
    assert "sha256sum --check --status" in _dockerfile()
