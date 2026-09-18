"""Operator-side XR1/Antioch data, source, and checkpoint exchange through Nebius S3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from npa.clients.antioch import antioch_environment
from npa.clients.project_credentials import storage_client_for_project, storage_env_for_project

from .assets import SOURCE_REVISION, fetch_checkpoint, fetch_processor
from .storage import _Store, _sha256
from .transport import fetch_inputs, publish_artifacts


def _assets(args) -> dict:
    root = args.output_path
    root.mkdir(parents=True, exist_ok=False)
    metadata = {"source_revision": SOURCE_REVISION,
                "model": fetch_checkpoint(root / "base/model_states.pt"),
                "processor": fetch_processor(root / "processor")}
    (root / "assets.json").write_text(json.dumps(metadata, indent=2))
    hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in root.rglob("*") if path.is_file()}
    (root / "checksums.json").write_text(json.dumps(hashes, indent=2))
    return _Store(args.s3_uri).publish(root)


def _source(args, storage) -> dict:
    root = args.output_path
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(root / "runtime.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, "npa/workflows/xr1_antioch/" + path.name)
        archive.write(args.split_manifest, "recipe/split-manifest.json")
    manifest = _Store(args.s3_uri).publish(root)
    return fetch_inputs(args.antioch_project, args.remote_root, args.s3_uri,
                        manifest, storage, source_bundle="runtime.zip")


def _runtime_recipe(args) -> None:
    import yaml

    root = args.output_path.resolve()
    source = root / "source"
    source.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).with_name("runtime.py"), source / "runtime.py")
    spec = {"name": "xr1-sm120-runtime", "workdir": str(source), "resources": {
        "cloud": "kubernetes", "accelerators": "RTXPRO6000:1", "cpus": 32, "memory": "256+",
        "image_id": "docker:pytorch/pytorch@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"},
        "envs": {"AWS_ENDPOINT_URL": os.environ["AWS_ENDPOINT_URL"], "XR1_RUNTIME_URI": args.s3_uri,
                 "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1", "MAX_JOBS": "16",
                 "NPA_EXECUTION_OUTPUTS": json.dumps([{"uri": args.s3_uri, "kind": "directory"}])},
        "run": 'python runtime.py --work-path /var/lib/npa/xr1-runtime --output-path "$XR1_RUNTIME_URI"'}
    (root / "runtime.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    print(json.dumps({"recipe": str(root / "runtime.yaml")}), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Configured NPA project alias; required for S3 operations")
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare")
    for option in ("baseline", "candidate", "split-manifest", "output-path"):
        comparison.add_argument(f"--{option}", type=Path, required=True)
    _antioch_commands(commands)
    for name in ("assets", "runtime", "source", "collect", "fetch", "publish"):
        command = commands.add_parser(name)
        command.add_argument("--s3-uri", required=True)
        command.add_argument("--output-path", type=Path, required=True)
        if name not in ("assets", "runtime"):
            command.add_argument("--antioch-project", type=Path, required=True)
            command.add_argument("--remote-root", required=True)
        if name in ("source", "collect"):
            command.add_argument("--split-manifest", type=Path, required=True)
        if name == "collect":
            command.add_argument("--source-root", required=True)
        if name == "fetch":
            command.add_argument("--manifest", default="manifest.json")
            command.add_argument("--include", action="append", help="Exact file names; default is every manifest entry")
    return parser


def _antioch_commands(commands) -> None:
    for name in ("antioch", "exec"):
        command = commands.add_parser(name, help="Run the Antioch CLI or attached remote argv with NPA credentials")
        command.add_argument("--antioch-project", type=Path, required=True)
        if name == "exec":
            command.add_argument("--antioch-python", type=Path, required=True,
                                 help="Interpreter with antioch-sim==0.4.236")
        command.add_argument("argv", nargs=argparse.REMAINDER, help="Literal arguments after --")


def _run_antioch(args) -> int:
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        raise ValueError("Antioch command arguments are required after --")
    if args.command == "exec":
        launcher = [str(args.antioch_python.resolve()),
                    str(Path(__file__).with_name("attached_exec.py")), "--"]
    else:
        launcher = ["antioch"]
    try:
        return subprocess.run(launcher + command, cwd=args.antioch_project,
                              env=antioch_environment()).returncode
    except KeyboardInterrupt:
        return 130


def _compare(args) -> None:
    from .dataset import validate_splits
    from .report import compare_rollouts

    manifest = json.loads(args.split_manifest.read_text())
    validate_splits(manifest)
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if baseline.get("complete") is not True or candidate.get("complete") is not True:
        raise ValueError("Both evaluation cohorts must have finished")
    report = compare_rollouts(baseline["rollouts"], candidate["rollouts"],
                              [row["seed"] for row in manifest["test"]],
                              baseline["checkpoint_sha256"], candidate["checkpoint_sha256"])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def _main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "compare":
        _compare(args)
        return
    if args.command in ("antioch", "exec"):
        raise SystemExit(_run_antioch(args))
    if not args.project:
        parser.error("--project is required for S3 operations")
    storage = storage_client_for_project(args.project, allow_host_creds=True)
    os.environ.update(storage_env_for_project(args.project, allow_host_creds=True))
    if args.command == "runtime":
        _runtime_recipe(args)
        return
    if args.command == "collect":
        from .operator_data import _collect

        _collect(args, storage)
        return
    if args.command == "assets":
        result = _assets(args)
    elif args.command == "source":
        result = _source(args, storage)
    elif args.command == "publish":
        result = publish_artifacts(args.antioch_project, args.remote_root, args.s3_uri, storage)
    else:
        manifest = _Store(args.s3_uri).read_json(args.manifest)
        if args.include:
            manifest = {name: manifest[name] for name in args.include}
        result = fetch_inputs(args.antioch_project, args.remote_root, args.s3_uri, manifest, storage)
    _write_result(args, result)


def _write_result(args, result: dict) -> None:
    target = args.output_path / "receipt.json" if args.command in ("assets", "source") else args.output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({"operation": args.command, "verified": True}), flush=True)


if __name__ == "__main__":
    _main()
