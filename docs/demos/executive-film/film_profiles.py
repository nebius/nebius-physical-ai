"""Define preview and final composition sizes and their cache dependencies."""

import subprocess
import sys
from pathlib import Path

import numpy
import PIL
from film_cache import _hash

_ROOT = Path(__file__).parent
_PROFILES = {
    "preview": {"width": 960, "height": 540, "preset": "ultrafast", "crf": 27},
    "final": {"width": 1920, "height": 1080, "preset": "fast", "crf": 18},
}


def _environment():
    return {"python": sys.version.split()[0], "pillow": PIL.__version__,
            "numpy": numpy.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}


def _scene_inputs(scene, index, total, assets, profile, environment):
    return {
        "scene": {key: value for key, value in scene.items() if key != "narration"},
        "index": index, "total": total, "profile": profile, "environment": environment,
        "assets": {role: {key: value for key, value in assets[role].items() if key != "path"}
                   for role in scene["assets"]},
        "code": {name: _hash(_ROOT / name) for name in
                 ["render.py", "graphics.py", "film_profiles.py", "fonts/Manrope.ttf"]},
    }


def _audio_inputs(scenes, voice_dir, environment):
    return {"scenes": [{"duration": scene["duration"],
                        "audio": _hash(voice_dir / f"{scene['id']}.mp3")}
                       for scene in scenes],
            "code": _hash(_ROOT / "soundtrack.py"), "environment": environment}


def _scaled_rectangle(rectangle, profile):
    scale = profile["width"] / 1920
    x, y, width, height = rectangle
    return (round(x * scale), round(y * scale),
            round(width * scale / 2) * 2, round(height * scale / 2) * 2)
