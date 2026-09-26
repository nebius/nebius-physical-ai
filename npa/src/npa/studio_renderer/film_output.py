"""Expose optional S3 delivery for rendered videos while preserving local editing."""

import json


def _output_arguments(parser):
    parser.add_argument(
        "--output-path",
        help="Also deliver the completed MP4 to this exact s3://bucket/key.mp4 URI.",
    )
    parser.add_argument(
        "--storage-project",
        help="NPA project alias for S3 credentials; defaults to external NPA configuration.",
    )


def _validate_output(args, parser):
    if args.storage_project is not None and not args.storage_project.strip():
        parser.error("--storage-project must name a configured NPA project alias")
    if args.storage_project is not None and args.output_path is None:
        parser.error("--storage-project requires --output-path")
    if args.output_path is None:
        return
    from npa.video_output import validate_video_output

    try:
        validate_video_output(args.output_path)
    except ValueError:
        parser.error(
            "--output-path must name an s3://bucket/key.mp4 object without URL credentials, query or fragment"
        )


def _deliver_output(args, video):
    if args.output_path is None:
        return
    from npa.video_output import write_video_output

    result = write_video_output(video, args.output_path, project=args.storage_project)
    print(json.dumps(result), flush=True)
