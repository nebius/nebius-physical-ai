"""S3-backed retrigger for the sim-to-real pipeline.

The trigger watches an S3-compatible bucket/prefix for LeRobot-format dataset
objects, records a cursor, and launches the current sim-to-real pipeline once
for each batch of newly observed data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from npa.workflows.sim_to_real import DEFAULT_GPU_FAILOVER, DEFAULT_GPU_TYPE, new_run_id


DEFAULT_TRIGGER_WATERMARK_NAME = ".npa/sim-to-real-retrigger-watermark.json"
DEFAULT_TRIGGER_POLL_INTERVAL = 60
DEFAULT_TRIGGER_SUBMIT_TIMEOUT = 1800
LEROBOT_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


class SimToRealTriggerError(ValueError):
    """Raised when the sim-to-real trigger cannot run safely."""


@dataclass(frozen=True)
class TriggerObject:
    """Cursor data for an object under the watched S3 prefix."""

    bucket: str
    key: str
    last_modified: str
    etag: str = ""
    size_bytes: int = 0

    @property
    def uri(self) -> str:
        """Return the object's S3 URI."""

        return f"s3://{self.bucket}/{self.key}"

    @property
    def signature(self) -> str:
        """Return a stable signature used for same-timestamp idempotency."""

        return f"{self.key}|{self.etag}|{self.size_bytes}"


@dataclass(frozen=True)
class TriggerWatermark:
    """Persisted cursor for a watched LeRobot dataset prefix."""

    cursor_last_modified: str = ""
    cursor_signatures: tuple[str, ...] = ()
    last_run_id: str = ""
    launches: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "TriggerWatermark":
        """Build a watermark from persisted JSON."""

        if not payload:
            return cls()
        signatures = payload.get("cursor_signatures", ())
        if isinstance(signatures, (list, tuple)):
            normalized = tuple(str(item) for item in signatures)
        else:
            normalized = ()
        return cls(
            cursor_last_modified=str(payload.get("cursor_last_modified") or ""),
            cursor_signatures=normalized,
            last_run_id=str(payload.get("last_run_id") or ""),
            launches=int(payload.get("launches") or 0),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-serializable cursor state."""

        return {
            "cursor_last_modified": self.cursor_last_modified,
            "cursor_signatures": list(self.cursor_signatures),
            "last_run_id": self.last_run_id,
            "launches": self.launches,
        }


@dataclass(frozen=True)
class TriggerConfig:
    """Configuration for one trigger watch/run invocation."""

    s3_endpoint: str
    s3_bucket: str
    s3_prefix: str
    watermark_uri: str = ""
    pipeline_yaml: str = ""
    pipeline_bucket: str = ""
    pipeline_s3_prefix: str = ""
    pipeline_input_data_uri: str = ""
    pipeline_render_only: bool = False
    task_cloud: str = "kubernetes"
    controller_backend: str = "kubernetes"
    sky_bin: str = ""
    gpu: str = DEFAULT_GPU_TYPE
    gpu_failover: str = DEFAULT_GPU_FAILOVER
    submit_timeout: int = DEFAULT_TRIGGER_SUBMIT_TIMEOUT

    def validate(self) -> None:
        """Validate trigger configuration before touching S3 or launching."""

        if not self.s3_endpoint:
            raise SimToRealTriggerError("s3_endpoint is required for the trigger")
        if not self.s3_bucket:
            raise SimToRealTriggerError("s3_bucket is required for the trigger")
        if self.task_cloud not in {"kubernetes", "nebius"}:
            raise SimToRealTriggerError("task_cloud must be 'kubernetes' or 'nebius'")
        if self.controller_backend not in {"kubernetes", "nebius"}:
            raise SimToRealTriggerError("controller_backend must be 'kubernetes' or 'nebius'")
        if self.submit_timeout <= 0:
            raise SimToRealTriggerError(f"submit_timeout must be positive, got {self.submit_timeout}")

    @property
    def normalized_prefix(self) -> str:
        """Return the watched prefix without a leading slash."""

        return self.s3_prefix.strip("/")

    @property
    def input_data_uri(self) -> str:
        """Return the LeRobot dataset URI passed to the next pipeline run."""

        if self.pipeline_input_data_uri:
            return self.pipeline_input_data_uri
        prefix = self.normalized_prefix
        suffix = f"/{prefix.rstrip('/')}/" if prefix else "/"
        return f"s3://{self.s3_bucket}{suffix}"

    @property
    def effective_watermark_uri(self) -> str:
        """Return an explicit or default S3 watermark URI."""

        if self.watermark_uri:
            return self.watermark_uri
        prefix = self.normalized_prefix
        key = "/".join(part for part in (prefix, DEFAULT_TRIGGER_WATERMARK_NAME) if part)
        return f"s3://{self.s3_bucket}/{key}"

    @property
    def effective_pipeline_bucket(self) -> str:
        """Return the bucket used for the launched pipeline outputs."""

        return self.pipeline_bucket or self.s3_bucket


@dataclass(frozen=True)
class PipelineLaunch:
    """Result from launching or rendering one pipeline run."""

    run_id: str
    status: str
    input_data_uri: str
    command: tuple[str, ...] = ()
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class TriggerResult:
    """Result from one trigger polling pass."""

    status: str
    watched_uri: str
    watermark_uri: str
    new_object_count: int
    new_objects: tuple[TriggerObject, ...]
    launch: PipelineLaunch | None
    watermark: TriggerWatermark
    generated_at: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""

        payload = asdict(self)
        payload["new_objects"] = [asdict(item) | {"uri": item.uri} for item in self.new_objects]
        payload["watermark"] = self.watermark.to_payload()
        return payload


class WatermarkStore(Protocol):
    """Minimal storage contract for trigger watermarks."""

    def load(self) -> TriggerWatermark:
        """Load the current watermark or return an empty one."""

    def save(self, watermark: TriggerWatermark) -> None:
        """Persist a new watermark."""


class PipelineLauncher(Protocol):
    """Minimal launch contract for the sim-to-real pipeline."""

    def launch(self, config: TriggerConfig, objects: tuple[TriggerObject, ...]) -> PipelineLaunch:
        """Launch a pipeline run for newly observed objects."""


class LocalWatermarkStore:
    """Watermark store backed by a local JSON file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> TriggerWatermark:
        """Load a local watermark file."""

        if not self.path.exists():
            return TriggerWatermark()
        return TriggerWatermark.from_payload(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, watermark: TriggerWatermark) -> None:
        """Persist a local watermark file."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(watermark.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class S3WatermarkStore:
    """Watermark store backed by an S3 object."""

    def __init__(self, *, uri: str, s3_client: Any) -> None:
        self.uri = uri
        self.s3_client = s3_client
        self.bucket, self.key = _split_s3_uri(uri)

    def load(self) -> TriggerWatermark:
        """Load an S3 watermark object if it exists."""

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
        except Exception as exc:
            if _is_missing_s3_object(exc):
                return TriggerWatermark()
            raise
        body = response["Body"].read()
        payload = json.loads(body.decode("utf-8"))
        return TriggerWatermark.from_payload(payload)

    def save(self, watermark: TriggerWatermark) -> None:
        """Persist the watermark as JSON in S3."""

        body = json.dumps(watermark.to_payload(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
        self.s3_client.put_object(Bucket=self.bucket, Key=self.key, Body=body)


class SubprocessPipelineLauncher:
    """Launch the current sim-to-real SkyPilot runner as a subprocess."""

    def launch(self, config: TriggerConfig, objects: tuple[TriggerObject, ...]) -> PipelineLaunch:
        """Launch or render one pipeline run for a batch of new LeRobot data."""

        del objects
        run_id = new_run_id("sim-to-real")
        command = _pipeline_command(config, run_id=run_id)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        status = "rendered" if config.pipeline_render_only and completed.returncode == 0 else "launched"
        if completed.returncode != 0:
            raise SimToRealTriggerError(
                "pipeline launch failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return PipelineLaunch(
            run_id=run_id,
            status=status,
            input_data_uri=config.input_data_uri,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def build_config_from_env(**overrides: Any) -> TriggerConfig:
    """Build trigger config from explicit overrides and environment variables."""

    def value(name: str, *env_names: str, default: Any = "") -> Any:
        if name in overrides and overrides[name] not in (None, ""):
            return overrides[name]
        for env_name in env_names:
            env_value = os.environ.get(env_name)
            if env_value not in (None, ""):
                return env_value
        return default

    return TriggerConfig(
        s3_endpoint=str(
            value(
                "s3_endpoint",
                "NPA_TRIGGER_S3_ENDPOINT",
                "S3_ENDPOINT_URL",
                "AWS_ENDPOINT_URL",
                "NEBIUS_S3_ENDPOINT",
            )
        ),
        s3_bucket=str(value("s3_bucket", "NPA_TRIGGER_S3_BUCKET", "S3_BUCKET", "NPA_S3_BUCKET")),
        s3_prefix=str(value("s3_prefix", "NPA_TRIGGER_S3_PREFIX", "S3_PREFIX", "LEROBOT_DATASET_PREFIX")),
        watermark_uri=str(value("watermark_uri", "NPA_TRIGGER_WATERMARK_URI")),
        pipeline_yaml=str(value("pipeline_yaml", "NPA_TRIGGER_PIPELINE_YAML")),
        pipeline_bucket=str(value("pipeline_bucket", "NPA_TRIGGER_PIPELINE_BUCKET")),
        pipeline_s3_prefix=str(value("pipeline_s3_prefix", "NPA_TRIGGER_PIPELINE_S3_PREFIX")),
        pipeline_input_data_uri=str(value("pipeline_input_data_uri", "NPA_TRIGGER_PIPELINE_INPUT_DATA_URI")),
        pipeline_render_only=_bool_value(value("pipeline_render_only", "NPA_TRIGGER_PIPELINE_RENDER_ONLY", default=False)),
        task_cloud=str(value("task_cloud", "NPA_TRIGGER_TASK_CLOUD", default="kubernetes")),
        controller_backend=str(value("controller_backend", "NPA_TRIGGER_CONTROLLER_BACKEND", default="kubernetes")),
        sky_bin=str(value("sky_bin", "NPA_SKYPILOT_BIN")),
        gpu=str(value("gpu", "NPA_GPU_TYPE", default=DEFAULT_GPU_TYPE)),
        gpu_failover=str(value("gpu_failover", "NPA_GPU_FAILOVER", default=DEFAULT_GPU_FAILOVER)),
        submit_timeout=int(value("submit_timeout", "NPA_TRIGGER_SUBMIT_TIMEOUT", default=DEFAULT_TRIGGER_SUBMIT_TIMEOUT)),
    )


def run_once(
    config: TriggerConfig | None = None,
    *,
    s3_client: Any | None = None,
    watermark_store: WatermarkStore | None = None,
    launcher: PipelineLauncher | None = None,
) -> TriggerResult:
    """Poll once, launch at most one pipeline run, and update the watermark."""

    config = config or build_config_from_env()
    config.validate()
    client = s3_client or _s3_client_from_config(config)
    store = watermark_store or watermark_store_for_uri(config.effective_watermark_uri, s3_client=client)
    previous = store.load()
    objects = tuple(list_lerobot_objects(config, s3_client=client))
    new_objects = tuple(new_lerobot_objects(objects, previous))
    launch: PipelineLaunch | None = None
    watermark = previous
    status = "idle"
    if new_objects:
        launch = (launcher or SubprocessPipelineLauncher()).launch(config, new_objects)
        watermark = advance_watermark(previous, new_objects, run_id=launch.run_id)
        store.save(watermark)
        status = "triggered"

    return TriggerResult(
        status=status,
        watched_uri=_watched_uri(config),
        watermark_uri=config.effective_watermark_uri,
        new_object_count=len(new_objects),
        new_objects=new_objects,
        launch=launch,
        watermark=watermark,
        generated_at=_utc_now(),
    )


def watch(
    config: TriggerConfig | None = None,
    *,
    poll_interval: int = DEFAULT_TRIGGER_POLL_INTERVAL,
    max_polls: int = 0,
    max_launches: int = 0,
    s3_client: Any | None = None,
    watermark_store: WatermarkStore | None = None,
    launcher: PipelineLauncher | None = None,
) -> list[TriggerResult]:
    """Poll repeatedly until a bounded watch condition is reached."""

    if poll_interval < 0:
        raise SimToRealTriggerError(f"poll_interval must be non-negative, got {poll_interval}")
    results: list[TriggerResult] = []
    launches = 0
    polls = 0
    while True:
        result = run_once(config, s3_client=s3_client, watermark_store=watermark_store, launcher=launcher)
        results.append(result)
        polls += 1
        if result.launch is not None:
            launches += 1
        if max_polls and polls >= max_polls:
            break
        if max_launches and launches >= max_launches:
            break
        time.sleep(poll_interval)
    return results


def list_lerobot_objects(config: TriggerConfig, *, s3_client: Any) -> list[TriggerObject]:
    """List LeRobot-format objects under the configured S3 prefix."""

    objects: list[TriggerObject] = []
    for item in _iter_s3_objects(s3_client, bucket=config.s3_bucket, prefix=config.normalized_prefix):
        key = str(item.get("Key") or "")
        if not key or not is_lerobot_key(key, config.normalized_prefix):
            continue
        objects.append(
            TriggerObject(
                bucket=config.s3_bucket,
                key=key,
                last_modified=_last_modified_iso(item.get("LastModified")),
                etag=str(item.get("ETag") or "").strip('"'),
                size_bytes=int(item.get("Size") or 0),
            )
        )
    return sorted(objects, key=lambda obj: (_timestamp(obj.last_modified), obj.key))


def is_lerobot_key(key: str, prefix: str = "") -> bool:
    """Return whether an object key looks like part of a LeRobot dataset."""

    relative = _relative_key(key, prefix)
    if relative in {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/stats.json",
    }:
        return True
    if relative.startswith("data/") and relative.endswith(".parquet"):
        return True
    if relative.startswith("videos/") and relative.lower().endswith(LEROBOT_VIDEO_SUFFIXES):
        return True
    return False


def new_lerobot_objects(
    objects: tuple[TriggerObject, ...] | list[TriggerObject],
    watermark: TriggerWatermark,
) -> list[TriggerObject]:
    """Return objects newer than the current watermark cursor."""

    if not watermark.cursor_last_modified:
        return list(objects)
    cursor_time = _timestamp(watermark.cursor_last_modified)
    cursor_signatures = set(watermark.cursor_signatures)
    new_objects: list[TriggerObject] = []
    for obj in objects:
        object_time = _timestamp(obj.last_modified)
        if object_time > cursor_time:
            new_objects.append(obj)
        elif object_time == cursor_time and obj.signature not in cursor_signatures:
            new_objects.append(obj)
    return new_objects


def advance_watermark(
    previous: TriggerWatermark,
    objects: tuple[TriggerObject, ...] | list[TriggerObject],
    *,
    run_id: str,
) -> TriggerWatermark:
    """Advance a watermark after a successful launch."""

    if not objects:
        return previous
    max_time = max(_timestamp(obj.last_modified) for obj in objects)
    cursor_last_modified = _format_timestamp(max_time)
    signatures = {
        obj.signature
        for obj in objects
        if _timestamp(obj.last_modified) == max_time
    }
    if cursor_last_modified == previous.cursor_last_modified:
        signatures.update(previous.cursor_signatures)
    return TriggerWatermark(
        cursor_last_modified=cursor_last_modified,
        cursor_signatures=tuple(sorted(signatures)),
        last_run_id=run_id,
        launches=previous.launches + 1,
    )


def watermark_store_for_uri(uri: str, *, s3_client: Any) -> WatermarkStore:
    """Return a local or S3-backed watermark store for a URI."""

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3WatermarkStore(uri=uri, s3_client=s3_client)
    if parsed.scheme == "file":
        return LocalWatermarkStore(parsed.path)
    return LocalWatermarkStore(uri)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for raw module invocation."""

    args = _parse_args(argv)
    config = build_config_from_env(**vars(args))
    try:
        if args.command == "run":
            result = run_once(config)
            print(json.dumps(result.to_payload(), indent=2, sort_keys=True))
            return 0
        results = watch(
            config,
            poll_interval=args.poll_interval,
            max_polls=args.max_polls,
            max_launches=args.max_launches,
        )
        for result in results:
            print(json.dumps(result.to_payload(), sort_keys=True), flush=True)
        return 0
    except SimToRealTriggerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _pipeline_command(config: TriggerConfig, *, run_id: str) -> list[str]:
    script = _pipeline_runner_script()
    yaml_path = config.pipeline_yaml or str(_default_pipeline_yaml())
    command = [
        sys.executable,
        str(script),
        "--yaml",
        yaml_path,
        "--run-id",
        run_id,
        "--bucket",
        config.effective_pipeline_bucket,
        "--s3-endpoint",
        config.s3_endpoint,
        "--input-data-uri",
        config.input_data_uri,
        "--gpu",
        config.gpu,
        "--gpu-failover",
        config.gpu_failover,
        "--task-cloud",
        config.task_cloud,
        "--controller-backend",
        config.controller_backend,
        "--submit-timeout",
        str(config.submit_timeout),
    ]
    pipeline_prefix = _pipeline_prefix(config, run_id=run_id)
    if pipeline_prefix:
        command.extend(["--s3-prefix", pipeline_prefix])
    if config.sky_bin:
        command.extend(["--sky-bin", config.sky_bin])
    if config.pipeline_render_only:
        command.append("--render-only")
    return command


def _pipeline_prefix(config: TriggerConfig, *, run_id: str) -> str:
    if not config.pipeline_s3_prefix:
        return ""
    return config.pipeline_s3_prefix.format(run_id=run_id).strip("/")


def _pipeline_runner_script() -> Path:
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "run_sim_to_real_pipeline.py"
    if not script.exists():
        raise SimToRealTriggerError(f"sim-to-real pipeline runner not found: {script}")
    return script


def _default_pipeline_yaml() -> Path:
    root = Path(__file__).resolve().parents[3]
    path = root / "src" / "npa" / "workflows" / "skypilot" / "sim-to-real-pipeline.yaml"
    if not path.exists():
        raise SimToRealTriggerError(f"sim-to-real pipeline YAML not found: {path}")
    return path


def _s3_client_from_config(config: TriggerConfig) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise SimToRealTriggerError("boto3 is required for S3-compatible trigger polling") from exc
    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
    )


def _iter_s3_objects(s3_client: Any, *, bucket: str, prefix: str):
    if hasattr(s3_client, "get_paginator"):
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            yield from page.get("Contents", [])
        return

    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3_client.list_objects_v2(**kwargs)
        yield from page.get("Contents", [])
        if not page.get("IsTruncated"):
            return
        token = page.get("NextContinuationToken")
        if not token:
            return


def _relative_key(key: str, prefix: str) -> str:
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix:
        return key.lstrip("/")
    prefix_dir = normalized_prefix.rstrip("/") + "/"
    if key.startswith(prefix_dir):
        return key[len(prefix_dir):]
    return key


def _watched_uri(config: TriggerConfig) -> str:
    suffix = f"/{config.normalized_prefix.rstrip('/')}/" if config.normalized_prefix else "/"
    return f"s3://{config.s3_bucket}{suffix}"


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SimToRealTriggerError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _is_missing_s3_object(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    code = ""
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code") or "")
    return code in {"NoSuchKey", "NoSuchBucket", "404", "NotFound"} or exc.__class__.__name__ in {
        "NoSuchKey",
        "NoSuchBucket",
    }


def _last_modified_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if value:
        return _format_timestamp(_timestamp(str(value)))
    return _utc_now()


def _timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return _format_timestamp(datetime.now(timezone.utc))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "watch"):
        sub = subparsers.add_parser(name)
        _add_common_args(sub)
        if name == "watch":
            sub.add_argument("--poll-interval", type=int, default=int(os.environ.get("NPA_TRIGGER_POLL_INTERVAL", "60")))
            sub.add_argument("--max-polls", type=int, default=int(os.environ.get("NPA_TRIGGER_MAX_POLLS", "0")))
            sub.add_argument("--max-launches", type=int, default=int(os.environ.get("NPA_TRIGGER_MAX_LAUNCHES", "0")))
    return parser.parse_args(argv)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s3-endpoint", default="")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--watermark-uri", default="")
    parser.add_argument("--pipeline-yaml", default="")
    parser.add_argument("--pipeline-bucket", default="")
    parser.add_argument("--pipeline-s3-prefix", default="")
    parser.add_argument("--pipeline-input-data-uri", default="")
    parser.add_argument("--pipeline-render-only", action="store_true")
    parser.add_argument("--task-cloud", choices=("kubernetes", "nebius"), default="kubernetes")
    parser.add_argument("--controller-backend", choices=("kubernetes", "nebius"), default="kubernetes")
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--gpu", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-failover", default=DEFAULT_GPU_FAILOVER)
    parser.add_argument("--submit-timeout", type=int, default=DEFAULT_TRIGGER_SUBMIT_TIMEOUT)


if __name__ == "__main__":
    sys.exit(main())
