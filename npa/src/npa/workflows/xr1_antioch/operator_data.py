"""Collect the sealed demonstration cohort and verify every native artifact in S3."""

from __future__ import annotations

import json
import subprocess

from npa.clients.antioch import antioch_environment

from .dataset import validate_episode, validate_splits, verify_video
from .storage import _relative, _sha256
from .transport import _destination, _readback, publish_artifacts


def _seal(client, bucket: str, prefix: str, manifest: dict) -> None:
    from botocore.exceptions import ClientError

    key = prefix + "/split-manifest.json"
    body = json.dumps(manifest, sort_keys=True).encode()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
            raise
        with client.get_object(Bucket=bucket, Key=key)["Body"] as stream:
            if json.load(stream) != manifest:
                raise ValueError("S3 already contains a different sealed split") from None


def _collect_one(args, entry: dict, split: str, storage, bucket: str, prefix: str) -> dict:
    name = entry["episode_id"]
    if name != f"{split}-{int(entry['seed'])}":
        raise ValueError("Episode identity must follow its sealed split and seed")
    remote = args.remote_root.rstrip("/") + "/" + name
    local = args.output_path / name
    local.mkdir(parents=True, exist_ok=True)
    receipt = local / "receipt.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if (result["episode_id"], result["split"], result["seed"]) != (name, split, entry["seed"]):
            raise ValueError("Local receipt differs from the sealed cohort")
        _readback(storage.s3, bucket, f"{prefix}/episodes/{name}", result["transfer"]["files"])
        _publish_receipt(storage, bucket, prefix, result)
        return result
    base = ["antioch", "service", "exec", "--no-tty", "--no-stream", "--"]
    probe = base + ["python", "-c", "from pathlib import Path; print(Path(" + repr(remote + "/episode.json") + ").is_file())"]
    environment = antioch_environment()
    exists = subprocess.check_output(probe, cwd=args.antioch_project, stdin=subprocess.DEVNULL,
                                     text=True, env=environment).strip()
    if exists != "True":
        command = base + ["env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                          f"PYTHONPATH={args.source_root}", "python", "-u", "-m",
                          "npa.workflows.xr1_antioch.collection", "--output-path", remote,
                          "--seed", str(entry["seed"]), "--split", split]
        with (local / "collection.log").open("w") as log:
            subprocess.run(command, cwd=args.antioch_project, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    transfer = publish_artifacts(args.antioch_project, remote,
                                 f"s3://{bucket}/{prefix}/episodes/{name}", storage)
    return _receipt(local, entry, split, storage, bucket, prefix, transfer)


def _receipt(local, entry, split, storage, bucket, prefix, transfer) -> dict:
    name = entry["episode_id"]
    for filename, identity in transfer["files"].items():
        target = local / _relative(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        storage.s3.download_file(bucket, f"{prefix}/episodes/{name}/{filename}", str(target))
        if _sha256(target) != identity["sha256"]:
            raise ValueError("Local recording differs from the verified S3 artifact")
    episode = json.loads((local / "episode.json").read_text())
    validate_episode(episode)
    if (episode["episode_id"], episode["split"], episode["seed"]) != (name, split, entry["seed"]):
        raise ValueError("Recorded episode differs from its sealed assignment")
    videos = {camera: verify_video(local / filename, episode["num_frames"], episode["control_hz"])
              for camera, filename in episode["videos"].items()}
    result = {**entry, "split": split, "success": episode["success"], "videos": videos, "transfer": transfer}
    receipt = local / "receipt.json"
    receipt.write_text(json.dumps(result, indent=2))
    _publish_receipt(storage, bucket, prefix, result)
    return result


def _publish_receipt(storage, bucket: str, prefix: str, result: dict) -> None:
    from botocore.exceptions import ClientError

    key = f"{prefix}/receipts/{result['episode_id']}.json"
    try:
        storage.s3.put_object(Bucket=bucket, Key=key,
                              Body=json.dumps(result, indent=2).encode(), IfNoneMatch="*")
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
            raise
    with storage.s3.get_object(Bucket=bucket, Key=key)["Body"] as stream:
        if json.load(stream) != result:
            raise ValueError("Published episode receipt differs from the verified recording")


def _collect(args, storage) -> None:
    manifest = json.loads(args.split_manifest.read_text())
    validate_splits(manifest)
    bucket, prefix = _destination(args.s3_uri)
    _seal(storage.s3, bucket, prefix, manifest)
    args.output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        for entry in manifest[split]:
            result = _collect_one(args, entry, split, storage, bucket, prefix)
            print(json.dumps({key: result[key] for key in ("episode_id", "success")}), flush=True)
