"""Decode validation for generated LTX-2.5 video.

These run real ffmpeg over real files. A test that mocked the decode would prove
only that the mock returns what the test told it to, and the whole point of this
module is that a file can exist, probe cleanly, and still be worthless output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from npa.workbench.ltx2.video_check import (
    ARTIFACT_SCHEMA,
    VideoCheckError,
    validate_video,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required to decode the fixtures",
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
        payload = validate_video(moving_clip, capability="ltx2_5_text_to_video").as_dict()

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
        assert "! /opt/npa/ltx2/validate_video.py --video /tmp/npa-ltx-flat.mp4" in dockerfile

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
