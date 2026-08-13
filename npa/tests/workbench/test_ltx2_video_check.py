"""Decode validation for generated LTX-2.5 video.

These run real ffmpeg over real files. A test that mocked the decode would prove
only that the mock returns what the test told it to, and the whole point of this
module is that a file can exist, probe cleanly, and still be worthless output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from npa.workbench.ltx2.video_check import (
    ARTIFACT_SCHEMA,
    VideoCheckError,
    validate_video,
)

_FFMPEG_MISSING = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None

if _FFMPEG_MISSING and os.environ.get("NPA_REQUIRE_FFMPEG") == "1":
    # This module decides whether a GPU run produced anything worth keeping. A
    # silent skip on a runner without ffmpeg means that decision goes untested
    # while the suite still reports green, so CI sets NPA_REQUIRE_FFMPEG=1 and
    # gets a hard failure instead of a quiet hole.
    raise RuntimeError(
        "NPA_REQUIRE_FFMPEG=1 but ffmpeg/ffprobe are absent: the LTX-2.5 output "
        "validator cannot be exercised, and skipping it would hide that."
    )

pytestmark = pytest.mark.skipif(
    _FFMPEG_MISSING, reason="ffmpeg/ffprobe are required to decode the fixtures"
)


def _synthesize(path: Path, source: str, *, seconds: str = "2") -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            seconds,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def moving_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ltx2-video")
    return _synthesize(root / "moving.mp4", "life=size=64x64:rate=24")


class TestAcceptsRealVideo:
    def test_a_moving_clip_passes_and_reports_what_it_measured(
        self, moving_clip: Path
    ) -> None:
        result = validate_video(moving_clip, min_frames=24, capability="probe")

        assert result.frame_count >= 24
        assert result.codec == "h264"
        assert result.width == 64 and result.height == 64
        assert result.max_frame_delta > 0
        assert len(result.sha256) == 64

    def test_the_artifact_is_json_serialisable_and_schema_tagged(
        self, moving_clip: Path
    ) -> None:
        payload = validate_video(
            moving_clip, capability="ltx2_5_text_to_video"
        ).as_dict()

        assert payload["schema"] == ARTIFACT_SCHEMA
        assert payload["capability"] == "ltx2_5_text_to_video"
        assert json.loads(json.dumps(payload)) == payload


class TestRejectsPlausibleLookingFailures:
    """Each of these leaves a file that exists and mostly probes fine."""

    def test_a_flat_single_colour_render_is_not_a_generation(
        self, tmp_path: Path
    ) -> None:
        clip = _synthesize(tmp_path / "flat.mp4", "color=c=black:size=64x64:rate=24")

        with pytest.raises(VideoCheckError, match="flat frames"):
            validate_video(clip)

    def test_one_still_repeated_is_not_a_video(self, tmp_path: Path) -> None:
        still = _synthesize(
            tmp_path / "still-source.mp4", "testsrc=size=64x64:rate=24", seconds="1"
        )
        frozen = tmp_path / "frozen.mp4"
        # Freeze frame 0 for two seconds: real picture content, zero motion.
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(still),
                "-vf",
                "select=eq(n\\,0),loop=loop=-1:size=1:start=0",
                "-t",
                "2",
                "-pix_fmt",
                "yuv420p",
                str(frozen),
            ],
            check=True,
            capture_output=True,
        )

        with pytest.raises(VideoCheckError, match="no motion"):
            validate_video(frozen)

    def test_a_clip_shorter_than_required_fails(self, moving_clip: Path) -> None:
        with pytest.raises(VideoCheckError, match="below the required"):
            validate_video(moving_clip, min_frames=10_000)

    def test_a_truncated_file_fails_rather_than_probing_clean(
        self, tmp_path: Path, moving_clip: Path
    ) -> None:
        broken = tmp_path / "truncated.mp4"
        broken.write_bytes(moving_clip.read_bytes()[:200])

        with pytest.raises(VideoCheckError):
            validate_video(broken)

    def test_an_absent_output_fails(self, tmp_path: Path) -> None:
        with pytest.raises(VideoCheckError, match="no video at"):
            validate_video(tmp_path / "never-written.mp4")

    def test_an_empty_file_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")

        with pytest.raises(VideoCheckError, match="is empty"):
            validate_video(empty)


class TestInImageEntryPointStaysInSync:
    """The container runs a copy of this module, so the copy must be the module."""

    def test_the_dockerfile_copies_the_tested_module_verbatim(self) -> None:
        dockerfile = (
            Path(__file__).resolve().parents[2]
            / "docker"
            / "workbench"
            / "ltx2"
            / "Dockerfile"
        ).read_text(encoding="utf-8")

        assert (
            "COPY --chmod=0444 src/npa/workbench/ltx2/video_check.py "
            "/opt/npa/ltx2/video_check.py" in dockerfile
        )
        assert "validate_video.py --video /tmp/npa-ltx-moving.mp4" in dockerfile
        assert (
            "! /opt/npa/ltx2/validate_video.py --video /tmp/npa-ltx-flat.mp4"
            in dockerfile
        )

    def test_the_wrapper_imports_the_copied_module_not_the_package(self) -> None:
        wrapper = (
            Path(__file__).resolve().parents[2]
            / "docker"
            / "workbench"
            / "ltx2"
            / "validate_video.py"
        ).read_text(encoding="utf-8")

        assert "import video_check" in wrapper
        assert "npa.workbench" not in wrapper, (
            "the image has no npa package installed; the wrapper must import the copy"
        )


class TestPartialDegeneration:
    """A clip that starts real and then freezes is the failure worth catching.

    The whole-clip checks use `max`, so before the fraction criteria they only
    asserted "at least one frame is not flat" and "at least one pair differs".
    One real frame followed by a hundred frozen ones satisfied both, and a
    two-stage video model producing exactly that is a plausible, plausible-looking
    failure — the kind a human skimming an MP4 thumbnail would not notice.
    """

    def _concat(self, path: Path, parts: list[Path]) -> Path:
        listing = path.parent / "concat.txt"
        listing.write_text(
            "".join(f"file '{part}'\n" for part in parts), encoding="utf-8"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path

    def test_a_clip_that_moves_briefly_then_freezes_is_rejected(
        self, tmp_path: Path
    ) -> None:
        moving = _synthesize(
            tmp_path / "moving.mp4", "life=size=64x64:rate=24", seconds="0.5"
        )
        frozen = _synthesize(
            tmp_path / "frozen.mp4", "testsrc=size=64x64:rate=24", seconds="4"
        )
        # Freeze the test pattern by taking one frame and holding it.
        held = tmp_path / "held.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(frozen),
                "-vf",
                "select=eq(n\\,0),loop=loop=95:size=1:start=0",
                "-r",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(held),
            ],
            check=True,
            capture_output=True,
        )
        clip = self._concat(tmp_path / "partial.mp4", [moving, held])

        with pytest.raises(VideoCheckError) as excinfo:
            validate_video(clip, min_frames=24)

        assert "mostly frozen" in str(excinfo.value)

    def test_a_genuinely_moving_clip_still_passes(self, tmp_path: Path) -> None:
        """The control: the stricter criterion must not reject real output."""

        clip = _synthesize(
            tmp_path / "real.mp4", "life=size=64x64:rate=24", seconds="3"
        )

        result = validate_video(clip, min_frames=24)

        # Assert against the dataclass defaults' opposite: a real clip moves in
        # essentially every pair, so `>= 0.5` would have passed even when these
        # fields were never assigned and kept their 1.0 defaults.
        assert 0.5 <= result.moving_pair_fraction <= 1.0
        assert 0.5 <= result.textured_frame_fraction <= 1.0
        # And the evidence has to survive serialization, since the artifact is
        # what a reviewer reads, not the in-process object.
        decode = result.as_dict()["decode"]
        assert decode["moving_pair_fraction"] == result.moving_pair_fraction
        assert decode["textured_frame_fraction"] == result.textured_frame_fraction

    def test_the_recorded_fraction_is_measured_not_defaulted(
        self, tmp_path: Path
    ) -> None:
        """A clip that passes with a frozen tail must record a fraction below 1.

        `moving_pair_fraction` and `textured_frame_fraction` defaulted to 1.0 and
        were never assigned from the measurement, so every `>= 0.5` assertion
        held vacuously against the default. A clip that is mostly-but-not-all
        motion is the case that can only pass if the value is real.
        """

        moving = _synthesize(tmp_path / "m.mp4", "life=size=64x64:rate=24", seconds="3")
        frozen = _synthesize(
            tmp_path / "f.mp4", "testsrc=size=64x64:rate=24", seconds="2"
        )
        held = tmp_path / "held.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(frozen),
                "-vf",
                "select=eq(n\\,0),loop=loop=23:size=1:start=0",
                "-r",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(held),
            ],
            check=True,
            capture_output=True,
        )
        clip = self._concat(tmp_path / "mostly-moving.mp4", [moving, held])

        result = validate_video(clip, min_frames=24)

        assert 0.5 <= result.moving_pair_fraction < 1.0, (
            "a clip with a frozen tail should record a measured fraction strictly "
            "below the 1.0 dataclass default"
        )
        assert result.as_dict()["decode"]["moving_pair_fraction"] == (
            result.moving_pair_fraction
        )


class TestBoundaries:
    def test_min_frames_below_two_is_a_video_check_error(self, tmp_path: Path) -> None:
        """It raised ValueError from `max(())` before, which callers do not catch."""

        clip = _synthesize(tmp_path / "clip.mp4", "life=size=64x64:rate=24")

        with pytest.raises(VideoCheckError) as excinfo:
            validate_video(clip, min_frames=1)

        assert "at least 2" in str(excinfo.value)

    def test_probing_fewer_frames_than_required_is_refused_up_front(
        self, tmp_path: Path
    ) -> None:
        """That combination can never pass, so say so instead of decoding first."""

        clip = _synthesize(tmp_path / "clip.mp4", "life=size=64x64:rate=24")

        with pytest.raises(VideoCheckError) as excinfo:
            validate_video(clip, min_frames=24, max_probe_frames=8)

        assert "could never pass" in str(excinfo.value)
