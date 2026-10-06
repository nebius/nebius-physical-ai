"""Contract tests for the real-media SwitchWorld workflow adapter."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import inspect
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow.spec import load_spec
from npa.workflows import switchworld


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/switchworld-lingbot-viewpoint-switch.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")


def _video(path: Path, filter_graph: str) -> None:
    """Encode a four-frame H.264 fixture with changing, decodable pixels."""

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            filter_graph,
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _frame_timestamps(path: Path) -> list[float]:
    """Read decoded presentation timestamps through ffprobe's frame parser."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return [
        float(line.split(",", 1)[0])
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _controls() -> dict[str, object]:
    """Return a valid minimal native-control sidecar fixture."""

    return {
        "schema": "npa.switchworld.controls.v1",
        "prompt": "A robot moves between two camera viewpoints.",
        "view_ids": [1, 1, 2, 2],
        "switch_frame": 2,
        "camera_poses": [[0.0] * 6 for _ in range(4)],
        "intrinsics": [[900.0, 900.0, 32.0, 24.0] for _ in range(4)],
    }


def test_controls_require_an_actual_viewpoint_transition(tmp_path: Path) -> None:
    """Reject a sidecar that labels frames but does not contain a switch."""

    path = tmp_path / "controls.json"
    invalid = _controls()
    invalid["view_ids"] = [1, 1, 1, 1]
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(switchworld.SwitchWorldError, match="transition"):
        switchworld._load_controls(path, target_frames=4)


def test_native_condition_is_bound_to_the_real_control_sidecar(tmp_path: Path) -> None:
    """Reject a native tensor whose camera/view schedule differs from controls."""

    torch = pytest.importorskip("torch")
    views = [1] * 7 + [2] * 8
    controls = {
        "schema": "npa.switchworld.controls.v1",
        "prompt": "A robot moves between two camera viewpoints.",
        "view_ids": views,
        "switch_frame": 7,
        "camera_poses": [
            [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
            for _ in views
        ],
        "intrinsics": [[900.0, 900.0, 320.0, 176.0] for _ in views],
    }
    condition = {
        "meta": {
            "condition_schema": "physicalworld_lingbot_v3_causal_multihot",
            "switch_frames": [7],
        },
        "view_id": torch.tensor(views),
        "switch_mask": torch.tensor([0] * 7 + [1] + [0] * 7),
        "camera_c2w": torch.eye(4).repeat(15, 1, 1),
        "camera_intrinsics": torch.tensor(controls["intrinsics"]),
    }
    path = tmp_path / "condition.pt"
    torch.save(condition, path)

    switchworld._validate_native_condition(path, controls)

    controls["intrinsics"][0][0] = 901.0
    with pytest.raises(switchworld.SwitchWorldError, match="intrinsics"):
        switchworld._validate_native_condition(path, controls)


def test_native_tensor_loads_are_weights_only() -> None:
    """Keep untrusted staged tensors on PyTorch's restricted deserialization path."""

    source = Path(switchworld.__file__).read_text(encoding="utf-8")

    assert "weights_only=False" not in source
    assert source.count("weights_only=True") == 4


def test_strict_sdpa_fallback_preserves_lengths_scale_and_local_causal_mask() -> None:
    """Exercise the numeric fallback used by the real adapter call site."""

    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    torch.manual_seed(7)
    q = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(2, 6, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(2, 6, 2, 3, dtype=torch.bfloat16)
    q_lens = torch.tensor([3, 5])
    k_lens = torch.tensor([4, 6])

    observed = switchworld._lingbot_sdpa_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        q_scale=0.75,
        softmax_scale=0.4,
        causal=True,
        window_size=(2, 0),
        dtype=torch.bfloat16,
    )
    expected = torch.zeros_like(observed)
    for index, (query_length, key_length) in enumerate(zip(q_lens, k_lens)):
        query_length, key_length = int(query_length), int(key_length)
        query = q[index, :query_length].transpose(0, 1).unsqueeze(0) * 0.75
        keys = k[index, :key_length].transpose(0, 1).unsqueeze(0)
        values = v[index, :key_length].transpose(0, 1).unsqueeze(0)
        mask = switchworld._local_attention_mask(
            query_length=query_length,
            key_length=key_length,
            causal=True,
            window_size=(2, 0),
            device=q.device,
        )
        expected[index, :query_length] = (
            functional.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=0.4,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    torch.testing.assert_close(observed, expected, rtol=0.03, atol=0.03)
    assert torch.count_nonzero(observed[0, 3:]) == 0


def test_adapter_attention_source_overlay_is_scoped_and_recorded(
    tmp_path: Path,
) -> None:
    """Replace only the reviewed direct adapter call with the strict shim."""

    source = tmp_path / "SwitchWorld" / "models"
    source.mkdir(parents=True)
    target = source / "lingbot_perspective_molora.py"
    target.write_text(
        "from wan.modules.attention import flash_attention\n"
        "output = flash_attention(q=q, k=k, v=v)\n",
        encoding="utf-8",
    )

    modification = switchworld._source_overlay_attention_fallback(
        tmp_path / "SwitchWorld"
    )
    patched = target.read_text(encoding="utf-8")
    assert "from wan.modules.attention import flash_attention" not in patched
    assert "from npa.workflows.switchworld import lingbot_attention" in patched
    assert "output = lingbot_attention(" in patched
    assert modification["upstream_revision"] == switchworld.SWITCHWORLD_REVISION
    assert modification["before_sha256"] != modification["after_sha256"]


def _source_archive(*, symlink: bool = False) -> bytes:
    """Build a tiny archive with the same root shape as the pinned upstream source."""

    root = f"SwitchWorld-{switchworld.SWITCHWORLD_REVISION}"
    payloads = {
        "pyproject.toml": b"[build-system]\nrequires = []\n",
        "models/lingbot_perspective_molora.py": b"# pinned source fixture\n",
    }
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        directory = tarfile.TarInfo(f"{root}/models")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, payload in payloads.items():
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
        if symlink:
            link = tarfile.TarInfo(f"{root}/unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside"
            archive.addfile(link)
    return output.getvalue()


def test_checkout_source_uses_verified_archive_without_runtime_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use only an immutable archive when the final image has no Git runtime."""

    payload = _source_archive()
    monkeypatch.setattr(switchworld, "SWITCHWORLD_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        switchworld,
        "SWITCHWORLD_ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    requests: list[tuple[str, frozenset[str], frozenset[str]]] = []

    def download(url: str, output: BytesIO, **kwargs: object) -> None:
        requests.append(
            (
                url,
                kwargs["allowed_hosts"],  # type: ignore[index]
                kwargs.get("redirect_hosts", frozenset()),  # type: ignore[union-attr]
            )
        )
        output.write(payload)

    monkeypatch.setattr(switchworld, "download_public_https", download)
    commands: list[list[str]] = []

    def install(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(switchworld.subprocess, "run", install)

    checkout = switchworld._checkout_source(tmp_path)

    assert (checkout / "pyproject.toml").is_file()
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(checkout),
        ]
    ]
    assert requests == [
        (
            switchworld.SWITCHWORLD_ARCHIVE_URL,
            frozenset({"codeload.github.com"}),
            frozenset(),
        )
    ]
    assert not (tmp_path / "SwitchWorld.tar.gz").exists()


def test_checkout_source_rejects_tampered_or_unsafe_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject altered bytes and archive links before an editable install runs."""

    payload = _source_archive()
    monkeypatch.setattr(switchworld, "SWITCHWORLD_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        switchworld,
        "SWITCHWORLD_ARCHIVE_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        switchworld,
        "download_public_https",
        lambda _url, output, **_kwargs: output.write(payload),
    )

    with pytest.raises(switchworld.SwitchWorldError, match="digest mismatch"):
        switchworld._checkout_source(tmp_path)
    assert not (tmp_path / "SwitchWorld").exists()
    assert not (tmp_path / "SwitchWorld.tar.gz").exists()

    unsafe_payload = _source_archive(symlink=True)
    monkeypatch.setattr(switchworld, "SWITCHWORLD_ARCHIVE_BYTES", len(unsafe_payload))
    monkeypatch.setattr(
        switchworld,
        "SWITCHWORLD_ARCHIVE_SHA256",
        hashlib.sha256(unsafe_payload).hexdigest(),
    )
    monkeypatch.setattr(
        switchworld,
        "download_public_https",
        lambda _url, output, **_kwargs: output.write(unsafe_payload),
    )
    monkeypatch.setattr(
        switchworld.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("editable install must not run"),
    )

    with pytest.raises(switchworld.SwitchWorldError, match="non-regular"):
        switchworld._checkout_source(tmp_path)
    assert not (tmp_path / "SwitchWorld").exists()
    assert not (tmp_path / "SwitchWorld.tar.gz").exists()


def test_adapter_command_explicitly_requires_anchor_only_visual_history(
    tmp_path: Path,
) -> None:
    """Never permit later target frames as canonical-adapter conditioning."""

    (tmp_path / "controls.json").write_text(
        json.dumps({"sampling_steps": 70}), encoding="utf-8"
    )
    adapters = {
        "joint_high_rank128.pt": tmp_path / "high.pt",
        "joint_low_rank128.pt": tmp_path / "low.pt",
    }
    command = switchworld._native_command(
        tmp_path / "SwitchWorld",
        tmp_path,
        tmp_path / "model",
        tmp_path / "out.mp4",
        1,
        adapters,
    )
    assert command[-2:] == ["--visual-history-mode", "anchors_only"]


def test_causal_transition_timing_uses_native_video_frames(tmp_path: Path) -> None:
    """Never linearly project a latent transition onto the real target timeline."""

    controls = {
        "schema": "npa.switchworld.controls.v1",
        "prompt": "A real simulated object changes camera viewpoints.",
        "view_ids": [2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2],
        "switch_frame": 4,
        "switch_frame_unit": "latent_frame_index",
        "source_video_switch_frames": [13, 21, 33, 41],
        "camera_poses": [
            [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
            for _ in range(15)
        ],
        "intrinsics": [[900.0, 900.0, 320.0, 176.0] for _ in range(15)],
    }
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls), encoding="utf-8")

    resolved = switchworld._load_controls(path, target_frames=57)
    assert resolved["_npa_primary_video_switch_frame"] == 13
    assert resolved["_npa_transition_video_frames"] == [13, 21, 33, 41]

    report = {
        "per_frame": [{"psnr_db": 20.0, "ssim": 0.5, "mae": 10.0} for _ in range(57)]
    }
    window = switchworld._switch_window(report, resolved)
    assert (window["first_frame"], window["last_frame"]) == (12, 14)


@pytest.mark.parametrize("source_rate", [4, 16])
def test_pair_and_rrd_use_decoded_media_not_manifests(
    tmp_path: Path, source_rate: int
) -> None:
    """Pair real source frames and independently validate the emitted RRD."""

    import numpy as np

    baseline = tmp_path / "baseline.mp4"
    adapted = tmp_path / "adapted.mp4"
    target = tmp_path / "target.mp4"
    _video(baseline, f"testsrc=size=64x48:rate={source_rate}")
    _video(adapted, f"testsrc2=size=64x48:rate={source_rate}")
    _video(target, f"smptebars=size=64x48:rate={source_rate}")

    baseline_frames, _ = switchworld._decoded_frames(baseline)
    adapted_frames, _ = switchworld._decoded_frames(adapted)
    paired_path = tmp_path / "paired.mp4"
    paired = switchworld._pair_videos(baseline, adapted, paired_path)
    paired_frames, paired_fps = switchworld._decoded_frames(paired_path)
    rrd = switchworld._build_rrd(
        baseline,
        adapted,
        target,
        {"schema": "npa.switchworld.real_frame_metrics.v1"},
        tmp_path / "switchworld.rrd",
        "switchworld-fixture",
    )

    assert paired["frame_count"] == 4
    assert paired["fps"] == switchworld.PAIR_FRAME_RATE
    assert paired_fps == switchworld.PAIR_FRAME_RATE
    assert _frame_timestamps(paired_path) == pytest.approx(
        [index / switchworld.PAIR_FRAME_RATE for index in range(4)]
    )
    for index, frame in enumerate(paired_frames):
        baseline_error = np.abs(
            frame[:, :64].astype(int) - baseline_frames[index].astype(int)
        )
        adapted_error = np.abs(
            frame[:, 64:].astype(int) - adapted_frames[index].astype(int)
        )
        assert float(baseline_error.mean()) < 5
        assert float(adapted_error.mean()) < 5
    assert paired["width"] == 128
    assert paired["mean_temporal_abs_delta"] > 0
    assert rrd["frame_count"] == 4
    assert rrd["rerun_verify"] == "passed"
    assert rrd["rerun_inspection"] == "passed"
    rrd_source = inspect.getsource(switchworld._build_rrd)
    assert '"-vv"' not in rrd_source
    assert '"--entity"' in rrd_source


def test_pair_videos_aligns_unequal_rates_by_decoded_frame_index(
    tmp_path: Path,
) -> None:
    """Keep each real target/model frame pair once when their source rates differ."""

    import numpy as np

    target = tmp_path / "target-40fps.mp4"
    adapted = tmp_path / "adapted-16fps.mp4"
    paired_path = tmp_path / "paired.mp4"
    _video(target, "testsrc=size=64x48:rate=40")
    _video(adapted, "testsrc2=size=64x48:rate=16")

    target_frames, _ = switchworld._decoded_frames(target)
    adapted_frames, _ = switchworld._decoded_frames(adapted)
    paired = switchworld._pair_videos(target, adapted, paired_path)
    paired_frames, paired_fps = switchworld._decoded_frames(paired_path)

    assert paired["frame_count"] == 4
    assert paired_fps == switchworld.PAIR_FRAME_RATE == 16
    assert len(paired_frames) == len(target_frames) == len(adapted_frames)
    for index, frame in enumerate(paired_frames):
        target_error = np.abs(
            frame[:, :64].astype(int) - target_frames[index].astype(int)
        )
        adapted_error = np.abs(
            frame[:, 64:].astype(int) - adapted_frames[index].astype(int)
        )
        assert float(target_error.mean()) < 5
        assert float(adapted_error.mean()) < 5


def test_pair_videos_retains_large_failing_encoder_diagnostics_without_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drain a large failing-child diagnostic while raw input pressures stdin."""

    baseline = tmp_path / "baseline.mp4"
    adapted = tmp_path / "adapted.mp4"
    output = tmp_path / "paired.mp4"
    _video(baseline, "testsrc=size=640x480:rate=16")
    _video(adapted, "testsrc2=size=640x480:rate=16")
    failing_child = tmp_path / "failing_encoder.py"
    failing_child.write_text(
        """import sys
sys.stderr.buffer.write(b'failing-child-diagnostic\\n' * 65536)
sys.stderr.buffer.flush()
payload = sys.stdin.buffer.read()
sys.stderr.buffer.write(f'received_bytes={len(payload)}\\n'.encode())
sys.stderr.buffer.flush()
raise SystemExit(23)
""",
        encoding="utf-8",
    )
    original_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def failing_encoder(
        _command: list[str], **kwargs: object
    ) -> subprocess.Popen[bytes]:
        child = original_popen([sys.executable, str(failing_child)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(switchworld.subprocess, "Popen", failing_encoder)
    raw_input_bytes = 4 * 640 * 480 * 3 * 2
    assert raw_input_bytes > 64 * 1024
    previous_handler = signal.getsignal(signal.SIGALRM)

    def test_harness_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("test harness detected a raw-stdin/stderr deadlock")

    signal.signal(signal.SIGALRM, test_harness_timeout)
    signal.setitimer(signal.ITIMER_REAL, 15)
    try:
        with pytest.raises(subprocess.CalledProcessError) as error:
            switchworld._pair_videos(baseline, adapted, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert error.value.returncode == 23
    assert b"failing-child-diagnostic" in error.value.stderr
    assert b"received_bytes=" in error.value.stderr
    assert len(children) == 1
    assert children[0].returncode == 23
    assert children[0].stdin is not None and children[0].stdin.closed


def test_real_pixel_cross_check_rejects_disconnected_metrics(tmp_path: Path) -> None:
    """Require metric reports to agree with an independent decode of real MP4 frames."""

    target = tmp_path / "target.mp4"
    prediction = tmp_path / "prediction.mp4"
    _video(target, "testsrc=size=64x48:rate=4")
    _video(prediction, "testsrc2=size=64x48:rate=4")

    independent = switchworld._independent_pixel_measurement(target, prediction)
    upstream = {
        "target_frames": independent["evaluated_frames"],
        "prediction_frames": independent["evaluated_frames"],
        "evaluated_frames": independent["evaluated_frames"],
        "per_frame": independent["per_frame"],
        "all_frames": independent["all_frames"],
        "future_frames_excluding_reference": independent[
            "future_frames_excluding_reference"
        ],
    }
    verified = switchworld._verify_pixel_measurement(upstream, independent)
    assert verified["status"] == "passed"
    assert verified["checked_metrics"] == ["psnr_db", "mae"]
    decoded = switchworld._decoded_pixel_evidence(independent)
    assert decoded["status"] == "passed"
    assert decoded["engine"] == "npa.switchworld.independent_decoded_pixels.v1"

    disconnected = deepcopy(upstream)
    disconnected["per_frame"][0]["mae"] += 1.0
    with pytest.raises(switchworld.SwitchWorldError, match="decoded pixels"):
        switchworld._verify_pixel_measurement(disconnected, independent)


def test_cv2_cross_check_and_pyav_evidence_are_distinct_decode_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the OpenCV reader contract without requiring dev-env OpenCV.

    The workflow image supplies OpenCV for SwitchWorld's upstream evaluator and
    the runtime integration proof uses that real decoder. The CPU CI environment
    intentionally does not install the optional package, so this test supplies
    a minimal, independently owned VideoCapture implementation to test our
    decode loop and metric computation. The PyAV branch below still decodes the
    actual MP4 fixtures rather than a synthetic frame manifest.
    """

    import numpy as np

    target = tmp_path / "target.mp4"
    prediction = tmp_path / "prediction.mp4"
    _video(target, "testsrc=size=64x48:rate=4")
    _video(prediction, "testsrc2=size=64x48:rate=4")

    target_frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(4)]
    prediction_frames = [
        np.full((2, 2, 3), value + 10, dtype=np.uint8) for value in range(4)
    ]

    class Capture:
        """Small OpenCV-compatible reader that is separate from upstream code."""

        def __init__(self, path: str) -> None:
            self.frames = target_frames if path == str(target) else prediction_frames
            self.index = 0

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API spelling
            return True

        def read(self) -> tuple[bool, object]:
            if self.index == len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=Capture, resize=lambda frame, _: frame),
    )

    upstream_decoder = switchworld._independent_cv2_pixel_measurement(
        target, prediction
    )
    upstream = {
        "target_frames": upstream_decoder["evaluated_frames"],
        "prediction_frames": upstream_decoder["evaluated_frames"],
        "evaluated_frames": upstream_decoder["evaluated_frames"],
        "per_frame": upstream_decoder["per_frame"],
        "all_frames": upstream_decoder["all_frames"],
        "future_frames_excluding_reference": upstream_decoder[
            "future_frames_excluding_reference"
        ],
    }
    assert (
        switchworld._verify_pixel_measurement(upstream, upstream_decoder)["status"]
        == "passed"
    )
    pyav_evidence = switchworld._decoded_pixel_evidence(
        switchworld._independent_pixel_measurement(target, prediction)
    )
    assert pyav_evidence["status"] == "passed"
    assert pyav_evidence["evaluated_frames"] == 4
    assert upstream_decoder["evaluated_frames"] == 4


def test_measure_serializes_actual_upstream_cv2_cross_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the measured report bound to both actual OpenCV cross-checks."""

    cross_check = {
        "engine": "npa.switchworld.upstream_cv2_pixels.v1",
        "status": "passed",
        "evaluated_frames": 57,
    }
    decoded_check = {
        "engine": "npa.switchworld.independent_decoded_pixels.v1",
        "status": "passed",
        "evaluated_frames": 57,
    }
    pair_report = {
        "all_frames": {"frames": 57},
        "future_frames_excluding_reference": {"frames": 56},
        "npa_upstream_cv2_pixel_cross_check": cross_check,
        "npa_independent_decoded_pixel_check": decoded_check,
    }

    monkeypatch.setattr(
        switchworld.StorageClient,
        "from_environment",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        switchworld,
        "_download_prepared",
        lambda _storage, _uri, root: {
            "target.mp4": root / "target.mp4",
            "controls.json": root / "controls.json",
        },
    )
    monkeypatch.setattr(switchworld, "_download", lambda *_: None)
    monkeypatch.setattr(switchworld, "_checkout_source", lambda root: root)
    monkeypatch.setattr(
        switchworld,
        "_load_controls",
        lambda *_: {"switch_frame": 13, "source_video_switch_frames": [13]},
    )
    monkeypatch.setattr(
        switchworld,
        "_video_evidence",
        lambda _: {"frame_count": 57, "sha256": "actual-video"},
    )
    monkeypatch.setattr(
        switchworld,
        "_switch_window",
        lambda *_: {"first_frame": 12, "last_frame": 14, "frames": 3},
    )
    monkeypatch.setattr(switchworld, "_upload_tree", lambda *_: None)

    def evaluate(*args: object) -> dict[str, object]:
        output = args[3]
        assert isinstance(output, Path)
        output.mkdir(parents=True)
        (output / "comparison.mp4").write_bytes(b"real-comparison-placeholder")
        return deepcopy(pair_report)

    monkeypatch.setattr(switchworld, "_run_pair_evaluation", evaluate)

    report = switchworld.measure(
        prepared_uri="s3://test/prepared/",
        baseline_uri="s3://test/baseline.mp4",
        adapted_uri="s3://test/adapted.mp4",
        output_uri="s3://test/metrics/",
    )

    assert report["target_vs_adapter"]["upstream_cv2_pixel_cross_check"] == cross_check
    assert (
        report["baseline_vs_adapter"]["upstream_cv2_pixel_cross_check"] == cross_check
    )
    assert (
        report["target_vs_adapter"]["independent_decoded_pixel_check"] == decoded_check
    )


def test_workflow_has_five_connected_real_stages() -> None:
    """Require the workflow's substantive source, inference, metric, and viz path."""

    spec = load_spec(WORKFLOW)
    raw = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    states = raw["states"]

    assert raw["config"]["source_overlay"] is True
    assert raw["config"]["gpu_type"] == "B200"
    assert raw["config"]["gpu_count"] == "1"
    assert raw["config"]["runtime_image"] == "tool://lingbot-world"
    assert raw["resources"]["gpu"]["accelerators"] == (
        "{{config.gpu_type}}:{{config.gpu_count}}"
    )
    assert raw["resources"]["cpu"]["image"] == "{{config.runtime_image}}"
    assert raw["resources"]["gpu"]["image"] == "{{config.runtime_image}}"
    assert len(states) == 5
    assert spec.initial == "prepare-real-case"
    assert states["prepare-real-case"]["next"] == "generate-lingbot-baseline"
    assert states["generate-lingbot-baseline"]["next"] == "generate-adapter-switch"
    assert states["generate-adapter-switch"]["next"] == "measure-real-frames"
    assert states["measure-real-frames"]["next"] == "emit-paired-artifacts"
    assert states["emit-paired-artifacts"]["terminal"] is True

    for state_name, state in states.items():
        argv = state["run"]["argv"]
        command = " ".join(argv)
        runtime = (
            "switchworld-visualize-runtime"
            if state_name == "emit-paired-artifacts"
            else "wan-runtime"
        )
        assert argv[:3] == [runtime, "exec", "python3"]
        assert "npa.workflows.switchworld" in command
        assert "echo" not in command
        assert "mock" not in command.lower()

    assert str(states["generate-lingbot-baseline"]["inputs"][0]["uri"]).endswith(
        "prepared_uri}}"
    )
    assert (
        states["generate-lingbot-baseline"]["inputs"][0]["schema"]
        == "npa.switchworld.prepared_bundle.v1"
    )
    assert "baseline.mp4" in str(states["measure-real-frames"]["inputs"])
    assert "adapted.mp4" in str(states["emit-paired-artifacts"]["inputs"])
    assert "switchworld.rrd" in str(states["emit-paired-artifacts"]["outputs"])


@pytest.mark.parametrize(
    ("state", "action"),
    (
        ("prepare-real-case", "prepare"),
        ("generate-lingbot-baseline", "baseline"),
        ("generate-adapter-switch", "adapter"),
        ("measure-real-frames", "measure"),
        ("emit-paired-artifacts", "visualize"),
    ),
)
def test_all_raw_stage_argv_parse_against_switchworld_parser(
    state: str, action: str
) -> None:
    """Pin every executable YAML state to the module's real argparse surface."""

    raw = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    argv = raw["states"][state]["run"]["argv"]
    runtime = (
        "switchworld-visualize-runtime"
        if state == "emit-paired-artifacts"
        else "wan-runtime"
    )
    assert argv[:5] == [
        runtime,
        "exec",
        "python3",
        "-m",
        "npa.workflows.switchworld",
    ]
    parser_argv = ["1" if value == "{{config.seed}}" else value for value in argv[5:]]
    parsed = switchworld._parser().parse_args(parser_argv)
    assert parsed.action == action


def test_readiness_record_is_bound_to_configurable_runtime_workflow() -> None:
    """Keep the readiness decision bound to the exact executable YAML bytes."""

    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["prerequisites"]["worker_input"]["status"] == "verified"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"


def test_canonical_adapter_selection_is_hash_pinned() -> None:
    """Keep the known identical alias from silently replacing canonical provenance."""

    assert switchworld.CANONICAL_ADAPTER_REPOSITORY == "PencilHu/SwitchWorld"
    assert (
        switchworld.CANONICAL_ADAPTER_REVISION
        == "5a01361ae1f9c9b1cfe115bef1c4d0377d922c1f"
    )
    assert set(switchworld.ADAPTER_FILES) == {
        "joint_high_rank128.pt",
        "joint_low_rank128.pt",
    }
    for digest, size in switchworld.ADAPTER_FILES.values():
        assert len(digest) == 64
        assert size > 6_000_000_000


def test_output_lineage_credits_lingbot_and_wan() -> None:
    """Retain the native runtime and checkpoint lineage in generated evidence."""

    lineage = switchworld._runtime_lineage()
    assert lineage["switchworld"]["revision"] == switchworld.SWITCHWORLD_REVISION
    assert lineage["lingbot_world"]["repository"].endswith("lingbot-world.git")
    assert lineage["lingbot_base_checkpoint"]["repository"] == (
        "robbyant/lingbot-world-base-cam"
    )
    assert lineage["umt5_tokenizer"]["repository"] == "google/umt5-xxl"
    assert lineage["wan"]["revision"] == switchworld.WAN_REVISION
