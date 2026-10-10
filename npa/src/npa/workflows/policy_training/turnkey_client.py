"""Apply served neural actions in native LIBERO physics and capture NVIDIA-rendered proof."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import time

from .public_vla_data import write_json
from .public_vla_export import file_sha256


def benchmark_client(
    service: str, session_uri: str, output: Path, recipe: dict, promotion: dict
) -> None:
    """Run the untouched third initial-state set against the independent model server.

    Args:
        service: Private gang-local model-server address.
        session_uri: Private authenticated rendezvous artifact.
        output: Native rollout, telemetry and camera artifacts.
        recipe: Immutable evaluation settings.
        promotion: Exact checkpoint approved by the second gate.
    Returns:
        None.
    Raises:
        ValueError: Model, action, renderer or observation identity is invalid.
        RuntimeError: Native benchmark or serving fails.
    """
    session, server = _session(service, session_uri, promotion)
    environment = _environment(recipe)
    episodes, steps, completed = [], [], False
    try:
        for episode in range(recipe["evaluation_episodes"]):
            initial = recipe["deployment_state_offset"] + episode
            row, recorded = _episode(
                environment, session, service, output, recipe, episode, initial
            )
            episodes.append(row)
            steps.extend(recorded)
        _summary(output, recipe, server, episodes, steps)
        completed = True
    finally:
        _close(environment, session_uri)
        _finish(session, service, completed)


def _session(service, session_uri, promotion):
    import requests
    from botocore.exceptions import ClientError
    from npa.workbench.dataset.storage import read_json_uri

    while True:
        try:
            authentication = read_json_uri(session_uri)
            break
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                raise
            time.sleep(3)
    if authentication["checkpoint_sha256"] != promotion["checkpoint_sha256"]:
        raise ValueError("serving rendezvous checkpoint differs from promotion")
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + authentication["token"]
    while True:
        try:
            response = session.get(service + "/health", timeout=10)
            response.raise_for_status()
            runtime = response.json()
            if runtime["checkpoint_sha256"] != promotion["checkpoint_sha256"]:
                raise ValueError("live server loaded a different checkpoint")
            return session, runtime
        except requests.ConnectionError:
            time.sleep(3)


def _episode(environment, session, service, output, recipe, episode, initial):
    environment.init_state_id = initial
    observation, _ = environment.reset(seed=recipe["seed"] + episode)
    _renderer(output)
    response = session.post(service + "/reset/" + str(episode), timeout=30)
    response.raise_for_status()
    video = output / f"episode-{episode:02d}.mp4"
    writer = _writer(video)
    steps, success = [], False
    try:
        for step in range(environment._max_episode_steps):
            observation, row, frame = _step(
                environment, session, service, observation, episode, step
            )
            writer.append_data(frame)
            _camera_snapshots(output, frame, episode, step)
            steps.append(row)
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if row["terminated"]:
                success = row["success"]
                writer.append_data(
                    _terminal_frame(environment, observation, row, episode, step)
                )
                break
    finally:
        writer.close()
    return {
        "episode": episode,
        "initial_state": initial,
        "success": success,
        "steps": len(steps),
        "video": video.name,
        "video_sha256": file_sha256(video),
    }, steps


def _close(environment, session_uri):
    from .diagnostics import _cleanup

    _cleanup(environment.close, session_uri)


def _terminal_frame(environment, observation, row, episode, step):
    import numpy as np

    scene = environment._env.env.sim.render(
        width=960, height=720, camera_name="agentview"
    )[::-1, ::-1].copy()
    images, _ = _observation(observation)
    return _visual(
        scene,
        images,
        np.asarray(row["action"]),
        row,
        row["round_trip_ms"],
        episode,
        step,
    )


def _step(environment, session, service, observation, episode, step):
    import numpy as np

    images, state = _observation(observation)
    request = {
        "episode": episode,
        "step": step,
        "task": environment.task_description,
        "state": state.tolist(),
        "image": _encode(images[0]),
        "wrist": _encode(images[1]),
    }
    result, elapsed = _request(session, service, request)
    action = np.asarray(result["action"], dtype=np.float32)
    if (
        action.shape != (7,)
        or not np.isfinite(action).all()
        or (result["episode"], result["step"]) != (episode, step)
    ):
        raise ValueError("served action is invalid or out of sequence")
    scene = environment._env.env.sim.render(
        width=960, height=720, camera_name="agentview"
    )[::-1, ::-1].copy()
    frame = _visual(scene, images, action, result, elapsed, episode, step)
    raw, reward, done, _ = environment._env.step(action)
    success = bool(environment._env.check_success())
    record = result | {
        "round_trip_ms": elapsed,
        "state": state.tolist(),
        "success": success,
        "reward": float(reward),
        "terminated": bool(done or success),
        "observation_sha256": hashlib.sha256(
            images[0].tobytes() + images[1].tobytes() + state.tobytes()
        ).hexdigest(),
    }
    return environment._format_raw_obs(raw), record, frame


def _observation(observation):
    import numpy as np
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep

    def batch(value):
        if isinstance(value, dict):
            return {key: batch(child) for key, child in value.items()}
        return value[None] if isinstance(value, np.ndarray) else value

    processed = LiberoProcessorStep().observation(
        preprocess_observation(batch(observation))
    )
    images = [
        processed["observation.images." + name][0]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
        for name in ("image", "image2")
    ]
    return images, processed["observation.state"][0].numpy()


def _encode(image):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _renderer(output):
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

    if (output / "renderer.json").exists():
        return
    renderer = {
        key: glGetString(value).decode()
        for key, value in (
            ("renderer", GL_RENDERER),
            ("vendor", GL_VENDOR),
            ("version", GL_VERSION),
        )
    }
    if "nvidia" not in (renderer["renderer"] + renderer["vendor"]).lower():
        raise ValueError("native rollout was not rendered on an NVIDIA GPU")
    write_json(output / "renderer.json", renderer)


def _visual(scene, images, action, result, elapsed, episode, step):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    frame = Image.new("RGB", (1280, 720), "#0d1723")
    frame.paste(Image.fromarray(scene), (0, 0))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=17)
    draw.rectangle((0, 0, 959, 44), fill="#0d1723")
    draw.text(
        (18, 12),
        "WORKBENCH / DEPLOYED SMOLVLA / NATIVE LIBERO",
        font=font,
        fill="#e5f3ff",
    )
    for index, image in enumerate(images):
        frame.paste(Image.fromarray(image).resize((128, 128)), (980 + 140 * index, 60))
    labels = [
        f"Episode {episode + 1:02d} / Step {step:03d}",
        "WORKSPACE + WRIST OBSERVATIONS",
        f"Server response {result['server_ms']:.1f} ms",
        f"Round trip {elapsed:.1f} ms",
        "New chunk" if result["new_action_chunk"] else "Queued chunk action",
        "ACTION APPLIED TO PHYSICS",
    ]
    for index, label in enumerate(labels):
        draw.text(
            (980, 218 + index * 30),
            label,
            font=font,
            fill="#68e7cd" if index == 4 else "#d1deeb",
        )
    _action_bars(draw, font, action)
    return np.asarray(frame)


def _camera_snapshots(output, frame, episode, step):
    from PIL import Image

    if step == 0:
        Image.fromarray(frame[:720, :960]).save(
            output / f"episode-{episode:02d}-scene.png"
        )
        for index, name in enumerate(("observation", "wrist")):
            Image.fromarray(frame[60:188, 980 + 140 * index : 1108 + 140 * index]).save(
                output / f"episode-{episode:02d}-{name}.png"
            )


def _summary(output, recipe, server, episodes, steps):
    import numpy as np

    def timing(key, selected):
        values = [row[key] for row in selected]
        return {
            "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }

    summary = {
        "engine": "native-libero-http-policy",
        "model": server,
        "episodes": episodes,
        "successes": sum(row["success"] for row in episodes),
        "trials": len(episodes),
        "steps": len(steps),
        "new_action_chunks": sum(row["new_action_chunk"] for row in steps),
        "round_trip_ms": timing("round_trip_ms", steps),
        "server_ms": timing("server_ms", steps),
        "new_chunk_server_ms": timing(
            "server_ms", [r for r in steps if r["new_action_chunk"]]
        ),
        "minimum_success": recipe["minimum_success"],
        "initial_state_offset": recipe["deployment_state_offset"],
        "physical_robot_tested": False,
        "simulation_physics": "MuJoCo CPU",
        "simulation_rendering": "NVIDIA EGL",
        "task": recipe["suite"] + "/" + str(recipe["task_id"]),
        "public_data_only": True,
    }
    write_json(output / "summary.json", summary)


def _finish(session, service, completed):
    try:
        response = session.post(
            service + "/finish",
            params={"completed": str(completed).lower()},
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        if completed:
            raise


def _environment(recipe):
    import os
    from libero import libero as benchmark_runtime
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    assets = Path(os.environ["NPA_VLA_ASSETS"])
    if not (assets / "scenes").is_dir():
        raise ValueError("pinned LIBERO assets are missing")
    benchmark_runtime._assets_path_cache = str(assets)
    suite = benchmark.get_benchmark_dict()[recipe["suite"]]()
    return LiberoEnv(
        suite,
        task_id=recipe["task_id"],
        task_suite_name=recipe["suite"],
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
    )


def _request(session, service, request):
    started = time.perf_counter()
    response = session.post(service + "/infer", json=request, timeout=180)
    response.raise_for_status()
    result = response.json()
    elapsed = (time.perf_counter() - started) * 1000
    return result, elapsed


def _writer(video):
    import imageio.v2 as imageio

    return imageio.get_writer(
        video,
        fps=20,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )


def _action_bars(draw, font, action):
    import numpy as np

    for index, (label, value) in enumerate(
        zip(("x", "y", "z", "rx", "ry", "rz", "grip"), action, strict=True)
    ):
        y = 422 + index * 31
        draw.text((980, y), label, font=font, fill="#93abc0")
        draw.rectangle((1034, y + 4, 1174, y + 12), fill="#24364a")
        end = 1104 + float(np.clip(value, -1, 1)) * 70
        draw.rectangle((min(1104, end), y + 4, max(1104, end), y + 12), fill="#68e7cd")
        draw.text((1190, y), f"{value:+.2f}", font=font, fill="white")
