"""Keep recorded narration identity aligned with the current scene wording."""

import hashlib
import json

from film_cache import _hash


def _voice_manifest(directory):
    path = directory / "voice-manifest.json"
    if not path.exists():
        return {}
    return {record["scene"]: record for record in json.loads(path.read_text())}


def _voice_record(scene, directory, voice):
    return {"scene": scene["id"], "voice": voice,
            "text_sha256": hashlib.sha256(scene["narration"].encode()).hexdigest(),
            "audio_sha256": _hash(directory / f"{scene['id']}.mp3"),
            "captions_sha256": _hash(directory / f"{scene['id']}.srt")}


def _voice_matches(scene, directory, record, voice=None):
    if not record or (voice is not None and record["voice"] != voice):
        return False
    try:
        current = _voice_record(scene, directory, record["voice"])
    except FileNotFoundError:
        return False
    fields = ["text_sha256", "audio_sha256"]
    if "captions_sha256" in record:
        fields.append("captions_sha256")
    return all(current[field] == record[field] for field in fields)


def _verify_narration(scenes, directory):
    records = _voice_manifest(directory)
    changed = [scene["id"] for scene in scenes
               if not _voice_matches(scene, directory, records.get(scene["id"]))]
    if changed:
        raise ValueError(f"Narration is missing or stale for {', '.join(changed)}. "
                         "Run narrate.py for changed speech, or use --recorded to register supplied recordings.")
