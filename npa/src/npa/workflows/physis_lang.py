"""Execute stateless paper-inspired physical prompting stages through the NPA workflow runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from npa.workflows.physis_lang_artifacts import materialize, publish, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--seeds", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--output-path", required=True)
    generate = stages.add_parser("generate")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--input-path", required=True)
    generate.add_argument("--output-path", required=True)
    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--input-path", action="append", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output-path", required=True)
    return parser


def _generate(prepared: Path, output: Path, seed: int) -> None:
    subprocess.run(["model-runtime", "ensure"], check=True)
    cache = Path(os.environ["NPA_WAN_RUNTIME_CACHE"])
    interpreter = cache / "current/venv/bin/python"
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    argv = [
        str(interpreter),
        "-m",
        "npa.workflows.physis_lang_generate",
        "--input-path",
        str(prepared),
        "--output-path",
        str(output),
        "--seed",
        str(seed),
    ]
    with (output / "generation.log").open("w") as log:
        subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _run(args, temporary: Path, output: Path) -> None:
    if args.stage == "prepare":
        from npa.workflows.physis_lang_contract import prepare

        prepare(
            output, args.run_id, [int(s) for s in args.seeds.split(",")], args.model
        )
    elif args.stage == "generate":
        prepared = materialize(args.input_path, temporary / "input")
        _generate(prepared, output, args.seed)
    else:
        from npa.workflows.physis_lang_evaluate import evaluate

        roots = [
            materialize(uri, temporary / f"input-{i}")
            for i, uri in enumerate(args.input_path)
        ]
        evaluate(roots, output, args.model)


def main(argv: list[str] | None = None) -> None:
    """Run a stage, preserving failure evidence without publishing success.

    Args:
        argv: Optional command arguments; defaults to the process command line.
    Returns:
        None.
    Raises:
        ValueError: Invalid inputs or artifact contracts.
        RuntimeError: Model inference, validation or publication fails.
    """
    args = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="npa-physis-") as temporary:
        root = Path(temporary)
        output = root / "output"
        output.mkdir()
        try:
            _run(args, root, output)
        except Exception as error:
            write_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "stage": args.stage,
                    "error_type": type(error).__name__,
                },
            )
            publish(output, args.output_path.rstrip("/") + "-failed/")
            raise
        publish(output, args.output_path)
    print(f"Physis {args.stage} completed and artifacts verified", flush=True)


if __name__ == "__main__":
    main()
