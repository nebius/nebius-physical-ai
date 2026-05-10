"""Rerun web-viewer MP4 renderer for LeRobot trajectory visualizations."""

from __future__ import annotations

import base64
import functools
import http.server
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


APPLICATION_ID = "npa_viz_lerobot_rerun"
TIMELINE = "frame_time"
BACKGROUND_COLOR = (26, 26, 26)
INPUT_COLOR = (0, 217, 255, 255)
PREDICTION_COLOR = (255, 136, 0, 190)
PREDICTION_SIDE_BY_SIDE_COLOR = (255, 136, 0, 255)
VIEWER_CHROME_TOP_PX = 104
VIEWER_CHROME_BOTTOM_PX = 96
REPRESENTATIVE_JOINT_INDICES = (6, 7, 18, 32)


class RerunRenderError(Exception):
    """Raised when the Rerun backend cannot render an MP4."""


@dataclass(frozen=True)
class _RuntimeTools:
    rerun_cli: str
    chrome: str
    ffmpeg: str


def render(
    skeleton_data,
    predictions_data,
    layout: str,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    duration_s: float,
    title: str,
    joint_connections: list[tuple[int, int]],
) -> None:
    """Render skeleton trajectory data to an MP4 with Rerun-rendered frames.

    Rerun 0.31.x can save ``.rrd`` recordings and its CLI documents
    ``--screenshot-to``, but native screenshots on macOS captured viewer chrome
    and hit WGPU surface-size validation during this run. This backend instead
    lets the Rerun web viewer render each frame, captures only the Spatial3D /
    TimeSeries viewport via Chrome DevTools Protocol, then uses ffmpeg only to
    encode those Rerun-rendered PNG frames to MP4.
    """
    skeleton = np.asarray(skeleton_data, dtype=np.float32)
    predictions = None if predictions_data is None else np.asarray(predictions_data, dtype=np.float32)
    _validate_inputs(skeleton, predictions, layout, output_path, resolution, fps, duration_s, joint_connections)

    tools = _ensure_runtime_tools()
    with tempfile.TemporaryDirectory(prefix="npa-rerun-render-") as tmp:
        work_dir = Path(tmp)
        viewer_dir = work_dir / "viewer"
        frames_dir = work_dir / "frames"

        _prepare_web_viewer_assets(tools.rerun_cli, viewer_dir)
        recordings = _write_frame_recordings(
            skeleton,
            predictions,
            layout,
            viewer_dir / "recordings",
            fps,
            duration_s,
            title or "LeRobot trajectory",
            joint_connections,
        )
        _capture_rerun_frames(
            tools.chrome,
            viewer_dir,
            recordings,
            frames_dir,
            resolution,
        )
        _encode_png_sequence(tools.ffmpeg, frames_dir, fps, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RerunRenderError(f"Rerun backend did not create MP4 output: {output_path}")


def _validate_inputs(
    skeleton: np.ndarray,
    predictions: np.ndarray | None,
    layout: str,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    duration_s: float,
    joint_connections: list[tuple[int, int]],
) -> None:
    if layout not in {"single", "side-by-side", "overlay"}:
        raise RerunRenderError(f"Unsupported layout '{layout}'")
    if layout in {"side-by-side", "overlay"} and predictions is None:
        raise RerunRenderError(f"predictions_data is required for layout '{layout}'")
    if skeleton.ndim != 3 or skeleton.shape[-1] != 3:
        raise RerunRenderError(f"skeleton_data must have shape [T, J, 3], got {skeleton.shape}")
    if skeleton.shape[0] == 0 or skeleton.shape[1] == 0:
        raise RerunRenderError("skeleton_data must contain at least one frame and one joint")
    if predictions is not None and predictions.shape != skeleton.shape:
        raise RerunRenderError(
            f"predictions_data shape {predictions.shape} must match skeleton_data shape {skeleton.shape}"
        )
    if not joint_connections:
        raise RerunRenderError("joint_connections must not be empty")
    max_joint = skeleton.shape[1] - 1
    for start, end in joint_connections:
        if start < 0 or end < 0 or start > max_joint or end > max_joint:
            raise RerunRenderError(f"joint connection {(start, end)} is outside skeleton joint range 0..{max_joint}")
    if output_path.suffix.lower() != ".mp4":
        raise RerunRenderError(f"Rerun backend writes MP4 only, got: {output_path}")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise RerunRenderError(f"resolution dimensions must be positive, got: {resolution}")
    if fps <= 0:
        raise RerunRenderError(f"fps must be positive, got {fps}")
    if duration_s <= 0:
        raise RerunRenderError(f"duration_s must be positive, got {duration_s}")


def _ensure_runtime_tools() -> _RuntimeTools:
    return _RuntimeTools(
        rerun_cli=_resolve_rerun_cli(),
        chrome=_resolve_chrome_executable(),
        ffmpeg=_resolve_required_executable(
            env_var="NPA_RERUN_FFMPEG",
            candidates=("ffmpeg",),
            description="ffmpeg",
            install_hint="Install ffmpeg and ensure it is on PATH, or set NPA_RERUN_FFMPEG.",
        ),
    )


def _resolve_rerun_cli() -> str:
    env_path = os.environ.get("NPA_RERUN_CLI")
    candidates: list[str | Path] = []
    if env_path:
        candidates.append(env_path)
    candidates.append(Path(sys.executable).with_name("rerun"))
    candidates.append("rerun")
    return _resolve_required_executable(
        env_var="NPA_RERUN_CLI",
        candidates=tuple(candidates),
        description="Rerun CLI",
        install_hint="Install rerun-sdk==0.31.4 and ensure the rerun CLI is on PATH.",
    )


def _resolve_chrome_executable() -> str:
    env_path = os.environ.get("NPA_RERUN_CHROME")
    candidates: list[str | Path] = []
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "chrome",
        ]
    )
    return _resolve_required_executable(
        env_var="NPA_RERUN_CHROME",
        candidates=tuple(candidates),
        description="Chrome or Chromium",
        install_hint="Install Chrome/Chromium or set NPA_RERUN_CHROME to the executable path.",
    )


def _resolve_required_executable(
    *,
    env_var: str,
    candidates: tuple[str | Path, ...],
    description: str,
    install_hint: str,
) -> str:
    for candidate in candidates:
        candidate_s = os.fspath(candidate)
        path_candidate = Path(candidate_s)
        if path_candidate.exists() and os.access(path_candidate, os.X_OK):
            return str(path_candidate)
        resolved = shutil.which(candidate_s)
        if resolved:
            return resolved
    raise RerunRenderError(f"{description} is required for the Rerun backend. {install_hint} ({env_var})")


def _prepare_web_viewer_assets(rerun_cli: str, viewer_dir: Path) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    logs_dir = viewer_dir.parent / "rerun-web-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("RERUN_ANALYTICS_ENABLED", "false")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [
                rerun_cli,
                "--serve-web",
                "--web-viewer-port",
                str(port),
                "--bind",
                "127.0.0.1",
                "--renderer",
                "webgl",
            ],
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            index = _download_when_ready(f"{base_url}/", timeout_s=30.0)
            (viewer_dir / "index.html").write_bytes(index)
            assets = _extract_asset_names(index.decode("utf-8", errors="replace"))
            assets.update({"re_viewer_bg.wasm", "favicon.ico", "favicon.svg"})
            for asset in sorted(assets):
                target = viewer_dir / asset
                if "/" in asset or "\\" in asset or asset.startswith("."):
                    continue
                try:
                    target.write_bytes(_download_url(f"{base_url}/{asset}", timeout_s=60.0))
                except Exception as exc:
                    if asset in {"re_viewer.js", "re_viewer_bg.wasm"}:
                        raise RerunRenderError(f"Failed to download Rerun web asset {asset}: {exc}") from exc
        finally:
            _terminate_process(proc)
    if not (viewer_dir / "index.html").exists() or not (viewer_dir / "re_viewer.js").exists():
        raise RerunRenderError(
            "Rerun web viewer assets were not prepared. "
            f"Rerun stderr tail: {_tail_text(stderr_path)}"
        )


def _extract_asset_names(index_html: str) -> set[str]:
    assets = set(re.findall(r"""(?:src|href)=["']([^"']+)["']""", index_html))
    return {
        asset
        for asset in assets
        if asset and not asset.startswith(("http://", "https://", "data:")) and "://" not in asset
    }


def _write_frame_recordings(
    skeleton: np.ndarray,
    predictions: np.ndarray | None,
    layout: str,
    recordings_dir: Path,
    fps: int,
    duration_s: float,
    title: str,
    joint_connections: list[tuple[int, int]],
) -> list[Path]:
    rr, rrb = _import_rerun()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_offset = _side_by_side_offset(skeleton, predictions) if layout == "side-by-side" else 0.0
    recordings = []
    for frame_idx in range(int(skeleton.shape[0])):
        path = recordings_dir / f"frame_{frame_idx:06d}.rrd"
        _write_single_frame_recording(
            rr,
            rrb,
            skeleton,
            predictions,
            layout,
            frame_idx,
            path,
            fps,
            duration_s,
            title,
            joint_connections,
            side_by_side_offset,
        )
        recordings.append(path)
    return recordings


def _write_single_frame_recording(
    rr: Any,
    rrb: Any,
    skeleton: np.ndarray,
    predictions: np.ndarray | None,
    layout: str,
    frame_idx: int,
    output_rrd: Path,
    fps: int,
    duration_s: float,
    title: str,
    joint_connections: list[tuple[int, int]],
    side_by_side_offset: float,
) -> None:
    blueprint = _build_blueprint(rrb, title, layout)
    recording = rr.RecordingStream(APPLICATION_ID)
    recording.save(output_rrd, default_blueprint=blueprint)
    rr.send_blueprint(blueprint, recording=recording)

    input_frame = skeleton[frame_idx]
    prediction_frame = None if predictions is None else predictions[frame_idx]
    if layout == "side-by-side" and prediction_frame is not None:
        input_frame = input_frame + np.array([-side_by_side_offset / 2.0, 0.0, 0.0], dtype=np.float32)
        prediction_frame = prediction_frame + np.array([side_by_side_offset / 2.0, 0.0, 0.0], dtype=np.float32)

    _log_skeleton(rr, recording, "world/input", input_frame, joint_connections, INPUT_COLOR)
    if prediction_frame is not None and layout in {"overlay", "side-by-side"}:
        prediction_color = PREDICTION_SIDE_BY_SIDE_COLOR if layout == "side-by-side" else PREDICTION_COLOR
        _log_skeleton(rr, recording, "world/predictions", prediction_frame, joint_connections, prediction_color)

    if hasattr(rr, "TextDocument"):
        rr.log("world/title", rr.TextDocument(title, media_type="text/markdown"), static=True, recording=recording)
    _log_motion_series(rr, recording, skeleton, predictions, fps=fps, duration_s=duration_s)
    _disconnect_recording(rr, recording)

    if not output_rrd.exists() or output_rrd.stat().st_size == 0:
        raise RerunRenderError(f"Rerun recording was not written: {output_rrd}")


def _build_blueprint(rrb: Any, title: str, layout: str) -> Any:
    spatial_name = "GR00T predictions overlay" if layout == "overlay" else "Isaac Lab trajectory"
    if layout == "side-by-side":
        spatial_name = "Input and predictions"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents="world/**",
                name=spatial_name,
                background=rrb.Background(color=BACKGROUND_COLOR),
                line_grid=False,
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(2.8, -3.8, 2.4),
                    look_target=(0.0, 0.0, 0.75),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            rrb.TimeSeriesView(
                origin="world",
                contents="world/**/angles/**",
                name="Representative joint motion",
            ),
            column_shares=[3.0, 1.0],
            name=title,
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Hidden, timeline=TIMELINE, fps=30.0),
        auto_layout=False,
        collapse_panels=True,
    )


def _log_skeleton(
    rr: Any,
    recording: Any,
    entity_root: str,
    positions: np.ndarray,
    joint_connections: list[tuple[int, int]],
    color: tuple[int, int, int, int],
) -> None:
    rr.log(
        f"{entity_root}/joints",
        rr.Points3D(
            positions,
            colors=[color] * int(positions.shape[0]),
            radii=0.045,
        ),
        static=True,
        recording=recording,
    )
    segments = np.asarray(
        [[positions[start], positions[end]] for start, end in joint_connections],
        dtype=np.float32,
    )
    rr.log(
        f"{entity_root}/bones",
        rr.LineStrips3D(
            segments,
            colors=[color] * len(joint_connections),
            radii=0.018,
        ),
        static=True,
        recording=recording,
    )


def _log_motion_series(
    rr: Any,
    recording: Any,
    skeleton: np.ndarray,
    predictions: np.ndarray | None,
    *,
    fps: int,
    duration_s: float,
) -> None:
    _log_series_styles(rr, recording, "world/input/angles", INPUT_COLOR)
    if predictions is not None:
        _log_series_styles(rr, recording, "world/predictions/angles", PREDICTION_SIDE_BY_SIDE_COLOR)
    frame_count = int(skeleton.shape[0])
    for frame_idx in range(frame_count):
        seconds = min(frame_idx / float(fps), duration_s)
        _set_time_seconds(rr, recording, seconds)
        for label, value in _representative_values(skeleton[frame_idx]):
            rr.log(f"world/input/angles/{label}", rr.Scalars(float(value)), recording=recording)
        if predictions is not None:
            for label, value in _representative_values(predictions[frame_idx]):
                rr.log(f"world/predictions/angles/{label}", rr.Scalars(float(value)), recording=recording)
    if hasattr(rr, "reset_time"):
        rr.reset_time(recording=recording)


def _log_series_styles(
    rr: Any,
    recording: Any,
    entity_root: str,
    color: tuple[int, int, int, int],
) -> None:
    if not hasattr(rr, "SeriesLines"):
        return
    for label in _representative_labels():
        rr.log(
            f"{entity_root}/{label}",
            rr.SeriesLines(colors=[color], names=[label]),
            static=True,
            recording=recording,
        )


def _representative_values(frame: np.ndarray) -> list[tuple[str, float]]:
    values = []
    for index, label in zip(_representative_indices(int(frame.shape[0])), _representative_labels(), strict=True):
        values.append((label, float(frame[index, 2])))
    return values


def _representative_indices(joint_count: int) -> list[int]:
    return [min(index, joint_count - 1) for index in REPRESENTATIVE_JOINT_INDICES]


def _representative_labels() -> list[str]:
    return [f"joint_{index}" for index in REPRESENTATIVE_JOINT_INDICES]


def _set_time_seconds(rr: Any, recording: Any, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(TIMELINE, seconds, recording=recording)
    else:
        rr.set_time(TIMELINE, duration=seconds, recording=recording)


def _disconnect_recording(rr: Any, recording: Any) -> None:
    try:
        rr.disconnect(recording=recording)
    except Exception:
        disconnect = getattr(recording, "disconnect", None)
        if callable(disconnect):
            disconnect()


def _side_by_side_offset(skeleton: np.ndarray, predictions: np.ndarray | None) -> float:
    arrays = [skeleton]
    if predictions is not None:
        arrays.append(predictions)
    combined = np.concatenate([arr.reshape(-1, 3) for arr in arrays], axis=0)
    span = float(np.max(combined[:, 0]) - np.min(combined[:, 0]))
    return max(1.2, span * 1.8)


def _capture_rerun_frames(
    chrome_executable: str,
    viewer_dir: Path,
    recordings: list[Path],
    frames_dir: Path,
    resolution: tuple[int, int],
) -> None:
    if not recordings:
        raise RerunRenderError("No Rerun recordings were produced")
    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = resolution
    crop_top = _viewer_chrome_top_px()
    crop_bottom = _viewer_chrome_bottom_px()
    chrome_log = frames_dir.parent / "chrome.log"

    with _StaticAssetServer(viewer_dir) as server_url:
        debug_port = _free_port()
        first_url = _viewer_url(server_url, viewer_dir, recordings[0])
        command = [
            chrome_executable,
            "--headless=new",
            "--disable-first-run-ui",
            "--no-first-run",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--disable-features=Translate,OptimizationGuideModelDownloading",
            "--force-device-scale-factor=1",
            f"--user-data-dir={frames_dir.parent / 'chrome-profile'}",
            f"--window-size={width},{height + crop_top + crop_bottom}",
            f"--remote-debugging-port={debug_port}",
            "--remote-allow-origins=*",
            first_url,
        ]
        with chrome_log.open("wb") as stderr:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr)
            client: _CDPClient | None = None
            try:
                ws_url = _wait_for_cdp_target(debug_port, timeout_s=30.0)
                client = _CDPClient(ws_url)
                client.call("Page.enable")
                client.call("Runtime.enable")
                for frame_idx, recording in enumerate(recordings):
                    if frame_idx > 0:
                        client.call("Page.navigate", {"url": _viewer_url(server_url, viewer_dir, recording)})
                    _wait_for_rerun_canvas(client, timeout_s=_frame_wait_timeout_s(), minimum_wait_s=_frame_min_wait_s(frame_idx))
                    screenshot = client.call(
                        "Page.captureScreenshot",
                        {
                            "format": "png",
                            "clip": {
                                "x": 0,
                                "y": crop_top,
                                "width": width,
                                "height": height,
                                "scale": 1,
                            },
                        },
                        timeout_s=30.0,
                    )
                    data = screenshot.get("data")
                    if not isinstance(data, str) or not data:
                        raise RerunRenderError(f"Chrome did not return screenshot data for frame {frame_idx}")
                    (frames_dir / f"frame_{frame_idx:06d}.png").write_bytes(base64.b64decode(data))
            finally:
                if client is not None:
                    client.close()
                _terminate_process(proc)

    expected = len(recordings)
    captured = len(list(frames_dir.glob("frame_*.png")))
    if captured != expected:
        raise RerunRenderError(f"Captured {captured} Rerun frames, expected {expected}. Chrome log: {_tail_text(chrome_log)}")


def _encode_png_sequence(ffmpeg: str, frames_dir: Path, fps: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RerunRenderError(f"ffmpeg failed to encode Rerun frames: {stderr or result.returncode}")


def _viewer_url(server_url: str, viewer_dir: Path, recording: Path) -> str:
    relative = recording.relative_to(viewer_dir).as_posix()
    recording_url = f"{server_url}/{relative}"
    return f"{server_url}/?url={recording_url}&renderer=webgl&hide_welcome_screen=true&persist=false"


class _StaticAssetServer:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._server: _ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> str:
        handler = functools.partial(_CORSRequestHandler, directory=str(self._directory))
        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="npa-rerun-http", daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class _CDPClient:
    def __init__(self, websocket_url: str) -> None:
        parsed = urlparse(websocket_url)
        if parsed.scheme != "ws":
            raise RerunRenderError(f"Unsupported CDP websocket URL: {websocket_url}")
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._path = parsed.path
        if parsed.query:
            self._path += f"?{parsed.query}"
        self._socket = socket.create_connection((self._host, self._port), timeout=10.0)
        self._socket.settimeout(30.0)
        self._next_id = 0
        self._handshake()

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float = 30.0) -> dict[str, Any]:
        self._next_id += 1
        message_id = self._next_id
        payload: dict[str, Any] = {"id": message_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send_text(json.dumps(payload, separators=(",", ":")))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._receive_message(deadline - time.monotonic())
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RerunRenderError(f"Chrome DevTools command {method} failed: {message['error']}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}
        raise RerunRenderError(f"Timed out waiting for Chrome DevTools command {method}")

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def _handshake(self) -> None:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._socket.recv(4096)
            if not chunk:
                break
            response += chunk
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RerunRenderError(f"Chrome DevTools websocket handshake failed: {response[:200]!r}")

    def _send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = secrets.token_bytes(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 2**16:
            header.extend([0x80 | 126, *struct.pack("!H", length)])
        else:
            header.extend([0x80 | 127, *struct.pack("!Q", length)])
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + mask + masked)

    def _receive_message(self, timeout_s: float) -> dict[str, Any]:
        self._socket.settimeout(max(0.1, timeout_s))
        while True:
            first = self._read_exact(2)
            opcode = first[0] & 0x0F
            masked = bool(first[1] & 0x80)
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length) if length else b""
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                raise RerunRenderError("Chrome DevTools websocket closed")
            if opcode == 0x9:
                self._send_pong(payload)
                continue
            if opcode != 0x1:
                continue
            return json.loads(payload.decode("utf-8"))

    def _send_pong(self, payload: bytes) -> None:
        mask = secrets.token_bytes(4)
        header = bytes([0x8A, 0x80 | len(payload)])
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(header + mask + masked)

    def _read_exact(self, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = self._socket.recv(length - len(data))
            if not chunk:
                raise RerunRenderError("Chrome DevTools websocket closed while reading")
            data += chunk
        return data


def _wait_for_cdp_target(port: int, *, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/json/list"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            targets = json.loads(_download_url(url, timeout_s=1.0).decode("utf-8"))
            for target in targets:
                websocket_url = target.get("webSocketDebuggerUrl")
                if target.get("type") == "page" and websocket_url:
                    return str(websocket_url)
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    raise RerunRenderError(f"Timed out waiting for Chrome DevTools target on port {port}: {last_error}")


def _wait_for_rerun_canvas(client: _CDPClient, *, timeout_s: float, minimum_wait_s: float) -> None:
    time.sleep(minimum_wait_s)
    deadline = time.monotonic() + timeout_s
    expression = """
(() => {
  const canvas = document.getElementById('the_canvas_id');
  const center = document.getElementById('center_text');
  return !!canvas && canvas.classList.contains('visible') &&
         canvas.width > 0 && canvas.height > 0 &&
         (!center || center.classList.contains('hidden'));
})()
"""
    while time.monotonic() < deadline:
        result = client.call("Runtime.evaluate", {"expression": expression, "returnByValue": True}, timeout_s=5.0)
        value = result.get("result", {}).get("value")
        if value is True:
            time.sleep(_post_canvas_ready_wait_s())
            return
        time.sleep(0.25)
    raise RerunRenderError("Timed out waiting for Rerun web viewer canvas to become visible")


def _import_rerun() -> tuple[Any, Any]:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RerunRenderError("rerun-sdk==0.31.4 is required for the Rerun backend") from exc
    return rr, rrb


def _download_when_ready(url: str, *, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _download_url(url, timeout_s=2.0)
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RerunRenderError(f"Timed out waiting for {url}: {last_error}")


def _download_url(url: str, *, timeout_s: float) -> bytes:
    with urlopen(url, timeout=timeout_s) as response:
        return response.read()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _tail_text(path: Path, *, max_bytes: int = 4096) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace").strip()


def _viewer_chrome_top_px() -> int:
    return int(os.environ.get("NPA_RERUN_VIEWER_TOP_CROP_PX", str(VIEWER_CHROME_TOP_PX)))


def _viewer_chrome_bottom_px() -> int:
    return int(os.environ.get("NPA_RERUN_VIEWER_BOTTOM_CROP_PX", str(VIEWER_CHROME_BOTTOM_PX)))


def _frame_wait_timeout_s() -> float:
    return float(os.environ.get("NPA_RERUN_FRAME_WAIT_TIMEOUT_S", "45.0"))


def _frame_min_wait_s(frame_idx: int) -> float:
    if frame_idx == 0:
        return float(os.environ.get("NPA_RERUN_FIRST_FRAME_WAIT_S", "6.0"))
    return float(os.environ.get("NPA_RERUN_NEXT_FRAME_WAIT_S", "1.75"))


def _post_canvas_ready_wait_s() -> float:
    return float(os.environ.get("NPA_RERUN_POST_CANVAS_WAIT_S", "0.5"))
