"""Exercise native GPU capabilities in the curated public job images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[
        'mochi-1', 'cogvideox-2b', 'wan2.1-14b', 'lingbot-world',
        'depth-anything-v2', 'sam2.1',
    ])
    parser.add_argument('--input-path', type=Path)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--prompt', default='A warehouse robot carries a blue crate along a sunlit aisle. Locked camera.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--box', type=float, nargs=4)
    args = parser.parse_args()
    if args.model in {'lingbot-world', 'depth-anything-v2', 'sam2.1'} and args.input_path is None:
        parser.error('--input-path is required for camera and perception capabilities')
    if args.model == 'sam2.1' and args.box is None:
        parser.error('--box X1 Y1 X2 Y2 is required for prompted segmentation')
    return args


def _run(args):
    if args.model == 'lingbot-world':
        from npa.solutions.lingbot_camera import generate_camera_video

        return generate_camera_video(args.input_path, args.prompt, args.seed, args.output_path, args.gpus)
    if args.model in {'depth-anything-v2', 'sam2.1'}:
        from npa.solutions.perception_video import run_perception

        kind = 'depth' if args.model == 'depth-anything-v2' else 'sam'
        return run_perception(kind, args.input_path, args.output_path, args.box)
    from npa.solutions.video_generation import generate_video

    return generate_video(args.model, args.prompt, args.seed, args.output_path)


if __name__ == '__main__':
    arguments = _arguments()
    evidence = _run(arguments)
    (arguments.output_path / 'capability.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))
