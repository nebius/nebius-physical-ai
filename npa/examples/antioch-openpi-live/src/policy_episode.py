"""Measure physical pickup progress and retain exact policy/control evidence."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _PickupProgress:
    initial_distance: float | None = None
    minimum_distance: float = math.inf
    maximum_lift: float = 0.0
    maximum_contact_force: float = 0.0
    contact_samples: int = 0
    hold_started: float | None = None
    hold_seconds: float = 0.0
    success: bool = False
    previous_sim_seconds: float | None = None

    def observe(self, *, sim_seconds, distance, lift, contact_force, closed):
        values = (sim_seconds, distance, lift, contact_force)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite physical pickup evidence")
        if self.previous_sim_seconds is not None:
            if sim_seconds < self.previous_sim_seconds:
                raise ValueError("simulation clock moved backwards")
            if sim_seconds == self.previous_sim_seconds:
                return
        self.previous_sim_seconds = sim_seconds
        if self.initial_distance is None:
            self.initial_distance = distance
        self.minimum_distance = min(self.minimum_distance, distance)
        self.maximum_lift = max(self.maximum_lift, lift)
        self.maximum_contact_force = max(self.maximum_contact_force, contact_force)
        contact = contact_force >= 0.1
        self.contact_samples += int(contact)
        if lift >= 0.05 and contact and closed:
            if self.hold_started is None:
                self.hold_started = sim_seconds
            self.hold_seconds = sim_seconds - self.hold_started
            self.success = self.hold_seconds >= 1.0
        else:
            self.hold_started = None
            self.hold_seconds = 0.0
            self.success = False

    @property
    def approach(self):
        if self.initial_distance is None:
            return 0.0
        return max(0.0, self.initial_distance - self.minimum_distance)


def _termination_reason(
    *,
    objective,
    round_trips,
    completed_chunks,
    pickup,
    applied,
    control_steps,
    sim_seconds,
    last_apply_sim_seconds,
):
    if (
        objective == "pickup"
        and pickup.success
        and pickup.approach >= 0.05
        and completed_chunks >= 2
        and round_trips >= 2
    ):
        return "pickup_complete"
    # Communication and exhaustion need a full interval of physics after the
    # last target. Pickup already has a continuously measured physical hold.
    if sim_seconds - last_apply_sim_seconds < 1.0 / 15.0 - 1e-9:
        return ""
    if objective == "communication":
        return (
            "communication_complete"
            if round_trips >= 2 and completed_chunks >= 2
            else ""
        )
    if applied >= control_steps:
        return "control_steps_exhausted"
    return ""


def _readiness_failure(
    *, now, camera_ready, camera_unavailable_since, last_control_at, deadline
):
    if not camera_ready and now - camera_unavailable_since >= deadline:
        return "camera_unavailable"
    if last_control_at is not None and now - last_control_at >= deadline:
        return "control_stalled"
    return ""


@dataclass
class _CameraStartup:
    """Separate cold renderer startup from loss of an operating camera pair."""

    started_at: float
    startup_deadline: float
    runtime_deadline: float
    completed_at: float | None = None
    unavailable_since: float | None = None

    def observe(self, *, now, produced_pair, policy_eligible):
        first_pair = (self.completed_at is None and produced_pair
                      and now - self.started_at < self.startup_deadline)
        if first_pair:
            self.completed_at = now
            self.unavailable_since = now
        if policy_eligible:
            self.unavailable_since = now
        return first_pair

    def failure(self, *, now, camera_ready, last_control_at):
        if self.completed_at is None:
            if now - self.started_at >= self.startup_deadline:
                return "camera_startup_unavailable"
            return ""
        return _readiness_failure(
            now=now, camera_ready=camera_ready,
            camera_unavailable_since=self.unavailable_since,
            last_control_at=last_control_at, deadline=self.runtime_deadline,
        )

    def record(self, run):
        complete = self.completed_at is not None
        elapsed = self.completed_at - self.started_at if complete else None
        run.add_result("camera_startup_seconds", elapsed)
        run.add_result("camera_startup_deadline_seconds", self.startup_deadline)
        run.check("camera_startup_completed", complete,
                  detail="Both native policy buffers and producer clocks became available")


class _PolicyEvidence:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(tempfile.mkdtemp(prefix="openpi-evidence-"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.trace = (self.root / "control.jsonl").open("x", encoding="utf-8")
        self.requests = 0
        self.responses = 0
        self.applied = 0

    def event(self, kind, **values):
        self.trace.write(json.dumps({"kind": kind, **values}, allow_nan=False) + "\n")
        self.trace.flush()

    def request(self, request, *, sim_seconds, producer_markers):
        import openpi_protocol

        payload = openpi_protocol.Packer().pack(request.observation)
        stem = f"request-{request.camera_pair_id:06d}"
        (self.root / f"{stem}.msgpack").write_bytes(payload)
        images = {}
        for view, key in (
            ("exterior", "exterior_image_1_left"),
            ("wrist", "wrist_image_left"),
        ):
            rgb = request.observation[f"observation/{key}"]
            pixels = rgb.tobytes(order="C")
            (self.root / f"{stem}-{view}.ppm").write_bytes(
                b"P6\n224 224\n255\n" + pixels
            )
            images[view] = hashlib.sha256(pixels).hexdigest()
        self.requests += 1
        self.event(
            "request",
            camera_pair_id=request.camera_pair_id,
            render_sequence=request.render_sequence,
            sim_seconds=sim_seconds,
            producer_markers=producer_markers,
            image_sha256=images,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
        return payload

    def response(self, camera_pair_id, response, *, sim_seconds):
        import numpy as np

        raw = np.asarray(response.get("actions"))
        # Never enable pickle for an untrusted policy payload. Malformed numeric
        # tensors remain inspectable, including NaNs rejected by the controller.
        numeric = np.issubdtype(raw.dtype, np.number) and not raw.dtype.hasobject
        if numeric:
            np.save(
                self.root / f"response-{camera_pair_id:06d}.npy",
                raw,
                allow_pickle=False,
            )
        self.responses += 1
        self.event(
            "response",
            camera_pair_id=camera_pair_id,
            sim_seconds=sim_seconds,
            shape=list(raw.shape),
            numeric=numeric,
            finite=bool(numeric and np.isfinite(raw).all()),
        )

    def target(self, *, camera_pair_id, row, target, before, sim_seconds):
        self.applied += 1
        self.event(
            "applied",
            camera_pair_id=camera_pair_id,
            row=row,
            target=target.tolist(),
            measured_before=before.tolist(),
            sim_seconds=sim_seconds,
        )

    def rejected_pair(self, pair, *, render_sequence):
        # Keep the most recent rejected image of each view without growing the
        # artifact on every warm-up frame. Successful requests are all retained.
        for view in ("exterior", "wrist"):
            frame = getattr(pair, view)
            if frame.rgb is not None:
                pixels = frame.rgb.tobytes(order="C")
                (self.root / f"rejected-{view}.ppm").write_bytes(
                    b"P6\n224 224\n255\n" + pixels
                )
        self.event(
            "camera_rejected",
            render_sequence=render_sequence,
            view=pair.rejected_view,
            reason=pair.reason,
        )

    def finish(self, run):
        self.trace.close()
        files = sorted(self.root.iterdir())
        manifest = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        }
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "npa.openpi.policy-evidence.v1",
                    "files": manifest,
                    "requests": self.requests,
                    "responses": self.responses,
                    "applied": self.applied,
                    "image_encoding": "lossless RGB PPM; request msgpack is exactly the client payload",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        archive = self.root / "policy-evidence.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
            for path in [*files, self.root / "manifest.json"]:
                output.write(path, arcname=path.name)
        run.add_artifact(
            archive,
            name="policy-evidence.zip",
            description="Exact policy inputs, raw actions, applied targets and measured physics",
        )
        return hashlib.sha256(archive.read_bytes()).hexdigest()


class _ShowcaseRecording:
    """Archive genuine HD sensor frames with their simulation timestamps."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(tempfile.mkdtemp(prefix="openpi-showcase-"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.frames: list[dict] = []
        self.last_marker: tuple[int, int] | None = None
        self.rejected = 0

    def capture(self, pixels, *, sim_seconds, render_sequence, producer_marker):
        """Save each advancing native frame once, with JPEG compression only."""
        import numpy as np
        from PIL import Image

        rgb = np.asarray(pixels)
        if rgb.dtype != np.uint8 or rgb.shape != (720, 1280, 3):
            raise ValueError("showcase camera requires native 1280x720 uint8 RGB")
        if not math.isfinite(sim_seconds):
            raise ValueError("showcase simulation timestamp must be finite")
        marker = producer_marker
        if marker is None or marker[1] <= 0:
            self.rejected += 1
            return False
        if self.last_marker and marker[0] * self.last_marker[1] <= self.last_marker[0] * marker[1]:
            self.rejected += 1
            return False
        if self.frames and sim_seconds <= self.frames[-1]["sim_seconds"]:
            raise ValueError("showcase simulation timestamp did not advance")
        self.last_marker = marker
        path = self.root / f"frame-{len(self.frames):06d}.jpg"
        Image.fromarray(rgb).save(path, format="JPEG", quality=92, subsampling=0)
        self.frames.append({
            "file": path.name, "sim_seconds": sim_seconds,
            "render_sequence": render_sequence, "producer_marker": list(marker),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mean_rgb": float(rgb.mean()), "variance": float(rgb.var()),
        })
        return True

    def finish(self, run):
        """Publish a checksummed frame archive independent of the policy inputs."""
        manifest = {
            "schema": "npa.openpi.showcase-recording.v1", "resolution": [1280, 720],
            "source": "native Isaac Sim RTX spectator camera",
            "treatment": "JPEG encoding only; no generated or interpolated frames",
            "policy_input": False, "frames": self.frames, "rejected": self.rejected,
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        archive = self.root / "showcase-frames.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as output:
            for name in ["manifest.json", *(frame["file"] for frame in self.frames)]:
                output.write(self.root / name, arcname=name)
        run.add_artifact(archive, name="showcase-frames.zip",
                         description="Native 1280x720 robot video frames and exact simulation times")
        run.add_result("showcase_frame_count", len(self.frames))
        run.add_result("showcase_recording_sha256", hashlib.sha256(archive.read_bytes()).hexdigest())
        usable = sum(5 < frame["mean_rgb"] <= 220 and frame["variance"] > 25
                     for frame in self.frames)
        run.add_result("showcase_usable_frame_count", usable)
        run.check("showcase_recording_available", usable >= 2,
                  detail=f"{usable} advancing nonblack native HD camera frames")
