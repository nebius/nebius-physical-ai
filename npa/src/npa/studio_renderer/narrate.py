"""Generate replaceable narration and sentence captions from the film storyboard."""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from film_cache import _render_lock
from film_voice import _voice_manifest, _voice_matches, _voice_record


async def _scene(scene, output_dir, voice):
    import edge_tts

    media = output_dir / f"{scene['id']}.mp3"
    subtitles = edge_tts.SubMaker()
    communication = edge_tts.Communicate(scene["narration"], voice, rate="+0%")
    with media.open("wb") as output:
        async for event in communication.stream():
            if event["type"] == "audio":
                output.write(event["data"])
            elif event["type"] == "SentenceBoundary":
                subtitles.feed(event)
    media.with_suffix(".srt").write_text(subtitles.get_srt(), encoding="utf-8")
    print(f"Narrated {scene['id']}", flush=True)
    return _voice_record(scene, output_dir, voice)


async def _update_scene(scene, args, previous):
    if args.recorded:
        return _voice_record(scene, args.output_dir, "recorded")
    voice = None if previous and previous["voice"] == "recorded" else args.voice
    if not args.force and _voice_matches(scene, args.output_dir, previous, voice):
        print(f"Reused narration {scene['id']}", flush=True)
        return _voice_record(scene, args.output_dir, previous["voice"])
    with TemporaryDirectory(prefix=".speech-", dir=args.output_dir) as temporary:
        staging = Path(temporary)
        record = await _scene(scene, staging, args.voice)
        for extension in ["mp3", "srt"]:
            (staging / f"{scene['id']}.{extension}").replace(args.output_dir / f"{scene['id']}.{extension}")
    return record


async def _run(args):
    storyboard = json.loads(args.storyboard.read_text())
    from render import _validate_storyboard

    _validate_storyboard(storyboard)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with _render_lock(args.output_dir):
        records = _voice_manifest(args.output_dir)
        for scene in storyboard["scenes"]:
            records[scene["id"]] = await _update_scene(scene, args, records.get(scene["id"]))
            temporary = args.output_dir / ".voice-manifest.pending"
            temporary.write_text(json.dumps(list(records.values()), indent=2) + "\n")
            temporary.replace(args.output_dir / "voice-manifest.json")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", type=Path, default=Path(__file__).with_name("storyboard.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
    parser.add_argument("--force", action="store_true", help="Regenerate every narration clip.")
    parser.add_argument("--recorded", action="store_true", help="Register supplied MP3/SRT files without contacting a speech service.")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    _main()
