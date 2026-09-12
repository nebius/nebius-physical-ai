"""Generate replaceable narration and sentence captions from the film storyboard."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import edge_tts


async def _scene(scene, output_dir, voice):
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
    return {"scene": scene["id"], "voice": voice,
            "text_sha256": hashlib.sha256(scene["narration"].encode()).hexdigest(),
            "audio_sha256": hashlib.sha256(media.read_bytes()).hexdigest()}


async def _run(args):
    storyboard = json.loads(args.storyboard.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in storyboard["scenes"]:
        results.append(await _scene(scene, args.output_dir, args.voice))
    (args.output_dir / "voice-manifest.json").write_text(json.dumps(results, indent=2) + "\n")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", type=Path, default=Path(__file__).with_name("storyboard.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    _main()
