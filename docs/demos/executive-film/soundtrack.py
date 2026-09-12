"""Synthesize an original instrumental bed and align supplied narration clips."""

import subprocess
import wave
from pathlib import Path

import numpy as np
from film_cache import _build_cached, _hash

_RATE = 48000


def _tone(frequency, seconds, decay=0):
    time = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    signal = np.sin(2 * np.pi * frequency * time)
    signal += 0.18 * np.sin(2 * np.pi * frequency * 2 * time)
    if decay:
        signal *= np.exp(-time * decay)
    else:
        signal *= np.minimum(time / 0.7, 1) * np.minimum((seconds - time) / 1.4, 1)
    return signal


def _add(track, sound, start, gain, pan=0):
    begin = round(start * _RATE)
    end = min(len(track), begin + len(sound))
    if end <= begin:
        return
    track[begin:end, 0] += sound[:end - begin] * gain * (1 - pan * 0.35)
    track[begin:end, 1] += sound[:end - begin] * gain * (1 + pan * 0.35)


def _score(path, duration):
    track = np.zeros((round(duration * _RATE), 2), dtype=np.float32)
    chords = [(164.81, 196, 246.94), (130.81, 164.81, 196),
              (146.83, 185, 220), (110, 130.81, 164.81)]
    random = np.random.default_rng(20260912)
    for section, start in enumerate(np.arange(0, duration, 8)):
        chord = chords[section % len(chords)]
        for index, frequency in enumerate(chord):
            _add(track, _tone(frequency, 9), start, 0.015, index - 1)
        _add(track, _tone(chord[0] / 2, 8), start, 0.027)
    for beat, start in enumerate(np.arange(0, duration - 3, 0.5)):
        chord = chords[int(start // 8) % len(chords)]
        _add(track, _tone(chord[beat % 3] * 2, 1.2, 6), start, 0.026, (-1) ** beat)
        if start > 7:
            _add(track, _tone(52, 0.28, 15), start, 0.073 if beat % 2 == 0 else 0.03)
            noise = random.normal(0, 1, int(_RATE * 0.07)).astype(np.float32)
            noise = np.diff(noise, prepend=noise[0])
            noise *= np.exp(-np.arange(len(noise)) / (_RATE * 0.015))
            _add(track, noise, start + 0.25, 0.004)
    time = np.arange(len(track), dtype=np.float32) / _RATE
    track *= (np.minimum(time / 3, 1) * np.clip((duration - time) / 4, 0, 1))[:, None]
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, _RATE, 0, "NONE", "not compressed"))
        output.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())


def _duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True).strip())


def _normalize_voice(scene, source, target):
    available = scene["duration"] - 0.85
    speed = max(1, _duration(source) / available)
    if speed > 1.15:
        raise ValueError(f"Narration too long for {scene['id']}; shorten its text")
    filters = (f"atempo={speed},loudnorm=I=-18:TP=-2:LRA=9,"
               f"adelay=350:all=1,apad,atrim=duration={scene['duration']}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-af", filters, "-ar", str(_RATE), "-ac", "2", str(target)], check=True)


def _narration(scenes, voice_dir, output_dir, cache_root=None):
    parts = []
    for index, scene in enumerate(scenes):
        source = voice_dir / f"{scene['id']}.mp3"
        target = output_dir / f"voice-{index:02d}.wav"
        if cache_root is None:
            _normalize_voice(scene, source, target)
        else:
            inputs = {"audio": _hash(source), "duration": scene["duration"],
                      "code": _hash(Path(__file__)),
                      "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
            cached, _ = _build_cached(cache_root, "voices", inputs,
                lambda staging, scene=scene, source=source: _normalize_voice(scene, source, staging / "voice.wav"))
            target = cached / "voice.wav"
        parts.append(target)
    target = output_dir / "narration.wav"
    with wave.open(str(target), "wb") as output:
        output.setparams((2, 2, _RATE, 0, "NONE", "not compressed"))
        for part in parts:
            with wave.open(str(part), "rb") as source:
                output.writeframes(source.readframes(source.getnframes()))
    return target


def _mix(scenes, voice_dir, output_dir, cache_root=None):
    music = output_dir / "original-score.wav"
    _score(music, sum(s["duration"] for s in scenes))
    narration = _narration(scenes, Path(voice_dir), output_dir, cache_root)
    result = output_dir / "mix.m4a"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(narration), "-i", str(music),
        "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=9",
        "-ar", str(_RATE), "-c:a", "aac", "-b:a", "256k", str(result),
    ], check=True)
    return result
