"""Hold `verify_rerun_recording` to the promise in its own docstring.

The function says it decodes "with Rerun's own verifier, not by checking a suffix".
Both of its guards were unreachable from any unit test: the only test that named it
replaced it with `lambda _: None`. It is called on the real path and in the functional
smoke, but no workflow under `.github/` runs that smoke, so nothing CI executes on a
pull request touched either branch.

Writing these found that the CLI check did not actually keep that promise. Measured
against rerun-sdk 0.38.1:

    file                        rrd verify   decodes   entity paths
    a real recording            exit 0       yes       /__properties, /points
    opened, nothing logged      exit 0       yes       /__properties
    zero bytes                  exit 0       no        --
    truncated to its header     exit 0       no        --
    wrong bytes, .rrd suffix    exit 1       no        --
    truncated to 75-99%         exit 1       yes       /__properties, /points

The third column is the one that matters: Rerun writes `/__properties` whether or not
the caller logged anything, so a recording that was opened and closed carrying no data
passes both a suffix check and `rrd verify`. Three checks now run, and each is the only
one that rejects some row above -- which is why the last row is in the table, since it
is the case that stops the CLI check from being redundant once decoding is added.
`test_each_check_is_the_only_one_that_catches_its_shape` re-measures all three and
fails if that stops being true.

That last row's band is not an artifact of a small recording, which was the obvious
objection to keeping the CLI check. It widens with size: 75-99% for the 11 kB
two-chunk recording above, 70-95% for a 1.8 MB one, and 50-95% for an 18 MB one. A
small recording is the worst case for that check's value, not the best. The reviewer
reproduced that independently on rerun 0.31.4 across a 775x size ratio and supplied
the mechanism: the reader's fallback recovers a caller entity as soon as one complete
chunk precedes the cut, so a larger recording crosses that threshold at a smaller
percentage.

Because which layer rejects a given cut depends on that chunk layout, the property
worth pinning is the disjunction rather than any one band:
`test_no_truncation_of_a_recording_passes_every_check` asserts no truncation is
accepted by every check, measured over 147 cuts at three sizes.
"""

from __future__ import annotations

import builtins
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from npa.workbench.open3d.artifacts import (
    RERUN_INTERNAL,
    Open3dError,
    verify_rerun_recording,
)

HEADER_BYTES = 40
"""Enough of a real recording for `rrd verify` to exit 0 on the remainder."""


def rerun_cli() -> str | None:
    """Resolve the CLI the way the function under test resolves it."""

    sibling = Path(sys.executable).with_name("rerun")
    return str(sibling) if sibling.is_file() else shutil.which("rerun")


needs_cli = pytest.mark.skipif(
    rerun_cli() is None, reason="the Rerun CLI is not installed on this machine"
)


def test_the_cli_is_resolvable_wherever_the_sdk_is_installed() -> None:
    """Stop the tests below from silently skipping in CI.

    `rerun-sdk` is in `npa/ci/requirements.txt`, and its console script installs next
    to `sys.executable`, which is the first place the function looks. If the SDK
    imports but the CLI does not resolve, every real-decode test here turns into a
    skip and this file goes back to proving nothing -- which is the shape of the
    finding it exists to close. So that combination is a failure, not a skip.
    """

    # Imported, not `importorskip`. `rerun-sdk` is a core dependency, so an import
    # failure is a packaging or ABI fault and not a reason to stand down -- and with
    # `importorskip` here the guard skipped in precisely the case it exists to catch,
    # taking `needs_cli` and the whole file with it into a green suite of skips.
    import rerun  # noqa: F401

    assert rerun_cli() is not None, (
        "rerun imports but its CLI is not resolvable, so the verifier tests would skip"
    )


def test_the_reserved_prefix_still_covers_rerun_own_metadata_path() -> None:
    """Fail loudly if Rerun moves its internal path out from under the prefix.

    `RERUN_INTERNAL` is how the verifier tells "Rerun wrote this" from "the caller
    logged this", and it is the prefix Rerun documents as reserved for itself rather
    than a pattern I inferred from one observation. If the SDK's own properties path
    stopped matching it, the verifier would start counting metadata as logged data
    and the hollow-recording check would quietly stop working.
    """

    rr = pytest.importorskip("rerun")
    assert rr.RECORDING_PROPERTIES_PATH.startswith(RERUN_INTERNAL), (
        f"Rerun's own metadata path {rr.RECORDING_PROPERTIES_PATH!r} no longer starts "
        f"with {RERUN_INTERNAL!r}, so the logged-data check needs re-deriving"
    )


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """A real recording, written by Rerun, with points actually logged."""

    rr = pytest.importorskip("rerun")
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "real.rrd"
    rr.init("npa-open3d-verifier-test", spawn=False)
    rr.save(str(path))
    rr.log(
        "points",
        rr.Points3D(numpy.random.default_rng(0).normal(size=(256, 3))),
    )
    rr.disconnect()
    if not path.is_file() or not path.stat().st_size:
        pytest.skip("Rerun did not write a recording on this machine")
    return path


@needs_cli
def test_a_real_recording_is_accepted(recording: Path) -> None:
    """The positive control. Without it the rejections below prove only strictness."""

    verify_rerun_recording(recording)


@needs_cli
def test_a_file_that_is_merely_named_rrd_is_refused(tmp_path: Path) -> None:
    """The claim the docstring makes in its own favour.

    Right suffix, wrong bytes. This is the case the function names, and it was the
    one with no test.
    """

    path = tmp_path / "pretend.rrd"
    path.write_text("this is not a recording, it is a sentence")
    with pytest.raises(Open3dError, match="rejected|could not decode"):
        verify_rerun_recording(path)


@needs_cli
def test_an_empty_recording_is_refused(tmp_path: Path) -> None:
    """`rrd verify` exits 0 on zero bytes, so the CLI check passed this.

    A stage that created the path but whose writer died before Rerun flushed anything
    leaves exactly this file, and it was being published as verified.
    """

    path = tmp_path / "empty.rrd"
    path.write_bytes(b"")
    # Which layer catches it is not this test's claim. If `rrd verify` ever learns to
    # reject a zero-byte file, the function's behaviour is unchanged and better, and
    # pinning the decode layer here would turn that improvement into a red required
    # gate. Layer attribution is the matrix test's job, and only its.
    with pytest.raises(Open3dError, match="rejected|could not decode"):
        verify_rerun_recording(path)


@needs_cli
def test_a_recording_truncated_to_its_header_is_refused(
    tmp_path: Path, recording: Path
) -> None:
    """Also exit 0 from the CLI: a valid header followed by no messages."""

    path = tmp_path / "header-only.rrd"
    path.write_bytes(recording.read_bytes()[:HEADER_BYTES])
    with pytest.raises(Open3dError, match="rejected|could not decode|no data"):
        verify_rerun_recording(path)


@needs_cli
def test_a_recording_that_logged_nothing_is_refused(tmp_path: Path) -> None:
    """The shape a suffix check and `rrd verify` both wave through.

    `rr.save` then `rr.disconnect` with no `rr.log` between them writes 3.2 kB that
    the CLI accepts and that decodes cleanly to one chunk, because Rerun records its
    own `/__properties` regardless. Nothing the caller logged is in it.
    """

    rr = pytest.importorskip("rerun")
    path = tmp_path / "nothing-logged.rrd"
    rr.init("npa-open3d-verifier-test-empty", spawn=False)
    rr.save(str(path))
    rr.disconnect()
    if not path.is_file():
        pytest.skip("Rerun did not write a recording on this machine")
    assert path.stat().st_size > 0, "this case is about a file with bytes in it"
    with pytest.raises(Open3dError, match="no logged data"):
        verify_rerun_recording(path)


@needs_cli
def test_a_mostly_complete_recording_is_refused(
    tmp_path: Path, recording: Path
) -> None:
    """The case that keeps the CLI check from being redundant.

    At this truncation the reader's whole-file fallback still returns `/points`, so
    both decode checks pass and only `rrd verify` rejects it.
    """

    payload = recording.read_bytes()
    path = tmp_path / "mostly.rrd"
    path.write_bytes(payload[: len(payload) * 90 // 100])
    # Same reason as above, in the other direction: today only the CLI rejects a 90
    # percent cut, but a reader that tightened its fallback would reject it too, and
    # that is not a regression.
    with pytest.raises(Open3dError, match="rejected|could not decode"):
        verify_rerun_recording(path)


@needs_cli
def test_each_check_is_the_only_one_that_catches_its_shape(
    tmp_path: Path, recording: Path
) -> None:
    """Fail if any of the three guards becomes droppable.

    Measures the CLI and the decode independently on one shape per guard, so the
    table in this file's docstring stays a measurement rather than a remembered
    claim. If a Rerun upgrade makes one check subsume another, this fails and asks
    for a re-measurement instead of silently leaving dead code in place.
    """

    rr = pytest.importorskip("rerun")
    payload = recording.read_bytes()
    nothing = tmp_path / "nothing.rrd"
    rr.init("npa-open3d-verifier-test-matrix", spawn=False)
    rr.save(str(nothing))
    rr.disconnect()
    cases = {
        "header_only": payload[:HEADER_BYTES],
        "mostly_complete": payload[: len(payload) * 90 // 100],
        "nothing_logged": nothing.read_bytes(),
    }
    cli = rerun_cli()
    assert cli is not None
    verdicts = {}
    for name, body in cases.items():
        path = tmp_path / f"{name}.rrd"
        path.write_bytes(body)
        checked = subprocess.run(
            [cli, "rrd", "verify", str(path)], capture_output=True, check=False
        )
        try:
            from npa.viz.recordings import load_recordings

            decoded = load_recordings(path)
            logged = [
                str(chunk.entity_path) for one in decoded for chunk in one.chunks()
            ]
            decodes, substantive = (
                True,
                [name for name in logged if not name.startswith("/__")],
            )
        except (RuntimeError, ValueError):
            decodes, substantive = False, []
        verdicts[name] = (checked.returncode == 0, decodes, bool(substantive))

    stale = "re-measure the table in this file's docstring before simplifying"
    # Only the decode rejects this one.
    assert verdicts["header_only"] == (True, False, False), f"{stale}: header_only"
    # Only the CLI rejects this one.
    assert verdicts["mostly_complete"] == (False, True, True), (
        f"{stale}: mostly_complete"
    )
    # Only the logged-data check rejects this one.
    assert verdicts["nothing_logged"] == (True, True, False), f"{stale}: nothing_logged"


def test_an_unavailable_cli_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other guard with no test.

    Defeats both resolution paths: the interpreter's sibling and `PATH`. Without the
    second, a machine with `rerun` on `PATH` would never reach the raise.
    """

    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    path = tmp_path / "unreachable.rrd"
    path.write_bytes(b"")
    with pytest.raises(Open3dError, match="Rerun CLI is unavailable"):
        verify_rerun_recording(path)


@needs_cli
def test_the_cli_is_not_redundant_at_a_realistic_recording_size(
    tmp_path: Path,
) -> None:
    """Keep the CLI check honest on something bigger than a toy.

    The case for keeping it rests on one band, and the obvious objection is that the
    band is an artifact of an 11 kB two-chunk recording. It is not: it widens with
    size. On a 1.8 MB recording every truncation from 70 to 95 percent is rejected
    only by the CLI, and on an 18 MB one so is every truncation from 50 percent up.
    A small recording is the *worst* case for this check's value, not the best.

    This builds the 1.8 MB shape rather than the 18 MB one, because the finding
    reproduces at both sizes and the smaller one keeps the test cheap.
    """

    rr = pytest.importorskip("rerun")
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "many-frames.rrd"
    rng = numpy.random.default_rng(0)
    rr.init("npa-open3d-verifier-test-many", spawn=False)
    rr.save(str(path))
    for frame in range(50):
        rr.set_time("frame", sequence=frame)
        rr.log("points", rr.Points3D(rng.normal(size=(2000, 3))))
        rr.log("mesh/verts", rr.Points3D(rng.normal(size=(1000, 3))))
    rr.disconnect()
    if not path.is_file() or path.stat().st_size < 100_000:
        pytest.skip("Rerun did not write a multi-chunk recording on this machine")

    payload = path.read_bytes()
    truncated = tmp_path / "many-frames-cut.rrd"
    truncated.write_bytes(payload[: len(payload) * 90 // 100])

    from npa.viz.recordings import load_recordings

    decoded = load_recordings(truncated)
    substantive = [
        str(chunk.entity_path)
        for one in decoded
        for chunk in one.chunks()
        if not str(chunk.entity_path).startswith("/__")
    ]
    assert substantive, (
        "a 90% truncation of a realistic recording no longer decodes to any logged "
        "entity path, so re-measure the band before trusting the CLI-only claim"
    )
    with pytest.raises(Open3dError, match="rejected"):
        verify_rerun_recording(truncated)


@needs_cli
@pytest.mark.parametrize(
    ("frames", "points"),
    [(1, 500), (50, 2000)],
    ids=["one_chunk", "many_chunks"],
)
def test_no_truncation_of_a_recording_passes_every_check(
    tmp_path: Path, frames: int, points: int
) -> None:
    """The property a consumer actually needs, stated over all truncations at once.

    The per-shape tests above each pin one example, which leaves the obvious question
    of whether some other cut slips through the gap between the layers. This sweeps
    2% to 98% and asserts no truncation is accepted by every check -- so the claim is
    "a truncated recording does not verify", not "these four truncations do not".

    Measured at three sizes over 147 cuts with no exception, including the 18 MB case
    that is too slow to build here. Which layer does the rejecting varies with the
    cut and with the recording's chunk layout, which is exactly why the sweep asserts
    the disjunction rather than naming a layer.
    """

    rr = pytest.importorskip("rerun")
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "sweep-source.rrd"
    rng = numpy.random.default_rng(0)
    rr.init("npa-open3d-verifier-test-sweep", spawn=False)
    rr.save(str(path))
    for frame in range(frames):
        rr.set_time("frame", sequence=frame)
        rr.log("points", rr.Points3D(rng.normal(size=(points, 3))))
    rr.disconnect()
    if not path.is_file() or not path.stat().st_size:
        pytest.skip("Rerun did not write a recording on this machine")

    verify_rerun_recording(path)  # the untruncated file must still pass

    payload = path.read_bytes()
    cut = tmp_path / "cut.rrd"
    accepted = []
    for percent in range(2, 100, 8):
        cut.write_bytes(payload[: len(payload) * percent // 100])
        try:
            verify_rerun_recording(cut)
            accepted.append(percent)
        except Open3dError:
            pass
    assert not accepted, (
        f"truncations at {accepted}% of a {len(payload)}-byte recording passed every "
        "check, so a cut recording can be published as verified"
    )


@needs_cli
def test_an_unimportable_sdk_is_reported_as_an_open3d_error(
    recording: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with the Rerun binary but not the SDK.

    The decode runs after the CLI check, so "no Rerun at all" still reports the
    unavailable CLI. This is the narrower combination, where the import itself fails
    and `ImportError` would escape a caller that only handles `Open3dError`.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args, **kwargs):
        if name == "npa.viz.recordings":
            raise ImportError("no module named rerun.chunk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(Open3dError, match="Rerun SDK is unavailable"):
        verify_rerun_recording(recording)


def test_a_missing_file_does_not_pass_verification(
    tmp_path: Path,
) -> None:
    """A path that was never written is not a verified recording."""

    if rerun_cli() is None:
        pytest.skip("the Rerun CLI is not installed on this machine")
    with pytest.raises(Open3dError):
        verify_rerun_recording(tmp_path / "never-written.rrd")


@needs_cli
def test_an_io_error_during_the_decode_is_reported_as_an_open3d_error(
    recording: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decode is a second read of the file, a subprocess later.

    `rrd verify` can pass and then `RrdReader` can fail on I/O rather than on
    content: the path removed, chmod'd, or replaced by a directory in between, or an
    unreadable mount. pyo3 surfaces `std::io::Error` as `OSError`, which was in
    neither except clause, so `verify_rerun_recording` raised something `_call` does
    not catch -- a traceback and a non-contract exit instead of the stderr message
    and exit 1 this function exists to produce.
    """

    import npa.viz.recordings as recordings_module

    def unreadable(path: Path):
        raise OSError(13, "Permission denied", str(path))

    monkeypatch.setattr(recordings_module, "load_recordings", unreadable)
    with pytest.raises(Open3dError, match="could not decode"):
        verify_rerun_recording(recording)


@needs_cli
def test_an_unreadable_file_on_disk_is_reported_as_an_open3d_error(
    tmp_path: Path, recording: Path
) -> None:
    """The same thing without a monkeypatch: a real unreadable file.

    `rrd verify` is lenient enough to be worth checking rather than assuming -- if it
    rejects the file first this passes through the CLI branch, which is why the
    monkeypatched test above exists alongside it.
    """

    path = tmp_path / "unreadable.rrd"
    path.write_bytes(recording.read_bytes())
    path.chmod(0o000)
    try:
        with pytest.raises(Open3dError):
            verify_rerun_recording(path)
    finally:
        path.chmod(0o644)


@needs_cli
@pytest.mark.parametrize("escaping", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_is_not_swallowed_by_the_decode_guard(
    recording: Path, monkeypatch: pytest.MonkeyPatch, escaping: type[BaseException]
) -> None:
    """The guard widened to `OSError`, deliberately not to `BaseException`.

    Catching `BaseException` would also catch a pyo3 `PanicException`, which was the
    other escape the review named -- but it would take `KeyboardInterrupt` and
    `SystemExit` with it, and a panic is a decoder bug that should keep its traceback
    rather than a condition this function can report on. So both have to pass
    straight through.
    """

    import npa.viz.recordings as recordings_module

    def interrupted(path: Path):
        raise escaping()

    monkeypatch.setattr(recordings_module, "load_recordings", interrupted)
    with pytest.raises(escaping):
        verify_rerun_recording(recording)
