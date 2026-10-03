"""Expose internal single-stage argv entrypoints for the workflow renderer."""

import argparse

from npa.workflows.field_failure.stages import run_stage


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "validate",
            "reconstruct",
            "train",
            "baseline-evaluate",
            "candidate-evaluate",
            "compare",
        ],
    )
    parser.add_argument("--bundle-uri", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--runtime-image", default="")
    args = parser.parse_args()
    run_stage(
        args.stage,
        args.bundle_uri,
        args.bundle_sha256,
        args.output_root,
        args.run_id,
        args.adapter,
        args.runtime_image,
    )


if __name__ == "__main__":
    _main()
