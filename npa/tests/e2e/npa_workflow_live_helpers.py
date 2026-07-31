"""Shared helpers for live npa.workflow infra tests."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pytest
from typer.testing import Result

from npa.clients.config import resolve_project_storage
from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec
from npa.orchestration.npa_workflow.submit_matrix import (
    SUBMIT_LIVE_MATRIX,
    SubmitLiveCase,
    selected_submit_cases,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS_DIR = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def resolve_spec_path(name: str) -> Path:
    """Resolve a live-submit spec by name across every blueprint root."""

    path = resolve_npa_workflow_spec(name)
    if path is None:
        pytest.fail(f"spec {name!r} not found in any npa.workflow spec root")
    return path

ALL_GOLDEN_SPECS = sorted(
    [
        "vlm-eval-single.yaml",
        "tokenfactory-rollout-judge.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "sim2real-vlm-rl.yaml",
        "bdd100k-pipeline.yaml",
    ]
)

DYNAMIC_SPECS = frozenset(
    {
        "sim2real-vlm-rl.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "rl-policy-training-sim-success.yaml",
        "physical-ai-data-factory.yaml",
        "token-factory-gate-loop.yaml",
    }
)

__all__ = [
    "ALL_GOLDEN_SPECS",
    "DYNAMIC_SPECS",
    "SPECS_DIR",
    "SUBMIT_LIVE_MATRIX",
    "SubmitLiveCase",
    "assume_decision_for",
    "assert_cli_ok",
    "assert_no_credential_leakage",
    "concurrency_overlaps",
    "live_bucket",
    "live_credential_markers",
    "materialize_live_spec",
    "parse_json_output",
    "parse_json_payload",
    "parse_runtime_json",
    "resolve_spec_path",
    "seed_live_workflow_inputs",
    "selected_submit_cases",
    "write_runtime_evidence",
]

_LEAK_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}"),
    re.compile(r"(?i)nebius_api_key\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)hf_[a-z0-9]{20,}"),
)


def assume_decision_for(name: str, *, mode: str = "promote") -> str:
    if name in DYNAMIC_SPECS:
        return "loop_back" if mode == "loop" else "promote_checkpoint"
    return ""


def live_bucket(e2e_project: str | None) -> str:
    storage = resolve_project_storage(e2e_project)
    raw = storage.checkpoint_bucket or ""
    if not raw:
        pytest.fail("checkpoint_bucket is not configured for live npa.workflow tests")
    parsed = urlparse(raw if "://" in raw else f"s3://{raw}")
    bucket = parsed.netloc if parsed.scheme == "s3" else raw.split("/")[0]
    if not bucket:
        pytest.fail(f"could not resolve live bucket from {raw!r}")
    return bucket


def seed_live_workflow_inputs(
    *,
    spec_name: str,
    bucket: str,
    run_id: str,
    e2e_project: str | None = None,
) -> None:
    """Upload minimal S3 fixtures so Token Factory twins have real inputs."""

    from io import BytesIO

    from npa.clients.project_credentials import s3_client_for_project

    marker = f"npa-workflow-e2e/{run_id}/{spec_name.replace('.yaml', '')}"
    client = s3_client_for_project(e2e_project, allow_host_creds=True)

    if spec_name == "token-factory-caption.yaml":
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover
            pytest.fail(f"Pillow required to seed caption fixtures: {exc}")
        image = Image.new("RGB", (320, 240), (200, 200, 200))
        draw = ImageDraw.Draw(image)
        draw.rectangle([40, 80, 160, 200], fill=(180, 40, 40))
        buf = BytesIO()
        image.save(buf, format="PNG")
        client.put_object(
            Bucket=bucket,
            Key=f"{marker}/images/fixture.png",
            Body=buf.getvalue(),
            ContentType="image/png",
        )
        return

    if spec_name == "token-factory-parallel-fanout.yaml":
        # One small image per shard prefix so all three fan-out members have real
        # Token Factory work to do concurrently.
        for shard in ("shard-a", "shard-b", "shard-c"):
            _seed_images(client, bucket=bucket, prefix=f"{marker}/images/{shard}/", count=2)
        return

    if spec_name == "token-factory-trigger-watch.yaml":
        # Deliberately NOT seeded here: the whole point of the trigger pattern is that
        # the run waits for data that is not there yet. Use seed_trigger_inbox_later().
        return

    if spec_name == "token-factory-gate-loop.yaml":
        # The loop captions and scores the same small batch every iteration.
        _seed_images(client, bucket=bucket, prefix=f"{marker}/images/", count=3)
        return

    if spec_name == "token-factory-generate.yaml":
        body = b'{"id": "e2e-1", "prompt": "Reply with the single word: ready"}\n'
        client.put_object(
            Bucket=bucket,
            Key=f"{marker}/prompts.jsonl",
            Body=body,
            ContentType="application/x-ndjson",
        )
        return

    if spec_name in ("token-factory-cosmos-reason.yaml", "tokenfactory-cosmos-gate.yaml"):
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover
            pytest.fail(f"Pillow required to seed scene fixtures: {exc}")
        # The cosmos-gate loop reasons over several scene frames before it can
        # gate; seed a small batch so the reason-scene stage has real inputs.
        frame_count = 1 if spec_name == "token-factory-cosmos-reason.yaml" else 3
        for index in range(frame_count):
            image = Image.new("RGB", (320, 240), (200, 200, 200))
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 180, 320, 240], fill=(120, 90, 60))
            draw.rectangle([120 + index * 10, 100, 200 + index * 10, 180], fill=(180, 40, 40))
            buf = BytesIO()
            image.save(buf, format="PNG")
            client.put_object(
                Bucket=bucket,
                Key=f"{marker}/scene/frame_{index:03d}.png",
                Body=buf.getvalue(),
                ContentType="image/png",
            )
        return

    if spec_name == "sonic-locomotion-finetuning.yaml":
        # SONIC retargeting needs a real G1 motion dataset (SOMA/G1 CSV clips,
        # each a directory with joint_pos.csv/body_pos.csv/body_quat.csv). We do
        # not vendor the dual-licensed upstream data; the operator points
        # NPA_E2E_SONIC_MOTION_SRC at a staged real dataset (an ``s3://`` prefix
        # or local directory, e.g. NVlabs/GR00T-WholeBodyControl
        # gear_sonic_deploy/reference/example after ``git lfs pull``).
        src = os.environ.get("NPA_E2E_SONIC_MOTION_SRC", "").strip()
        if not src:
            pytest.skip(
                "NPA_E2E_SONIC_MOTION_SRC not set; stage a real SOMA/G1 motion "
                "dataset (soma-csv clips) and point this at it."
            )
        _seed_prefix_from_source(src, bucket, f"{marker}/source/", client)
        return

    # VLM-eval GPU twins score a rollout: seed a short RGB frame sequence under
    # the rollouts prefix so the self-hosted VLM has real frames to evaluate.
    if spec_name in {
        "vlm-eval-single.yaml",
        "vlm-eval-benchmark.yaml",
        "tokenfactory-rollout-judge.yaml",
    }:
        _seed_rollout_frames(client, bucket=bucket, marker=marker)
        if spec_name == "tokenfactory-rollout-judge.yaml":
            # This twin also reasons over a captured scene before judging.
            _seed_scene_frame(client, bucket=bucket, marker=marker)
        return


def _seed_images(client, *, bucket: str, prefix: str, count: int = 2) -> None:
    """Upload ``count`` small deterministic PNGs under ``prefix``."""

    from io import BytesIO

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        pytest.fail(f"Pillow required to seed image fixtures: {exc}")
    for index in range(max(1, count)):
        image = Image.new("RGB", (320, 240), (30, 30, 30))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 190, 320, 240], fill=(90, 70, 50))
        draw.rectangle([40 + index * 30, 120, 120 + index * 30, 190], fill=(200, 60, 60))
        draw.rectangle([240, 120, 300, 190], fill=(40, 200, 40))
        buf = BytesIO()
        image.save(buf, format="PNG")
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}frame_{index:03d}.png",
            Body=buf.getvalue(),
            ContentType="image/png",
        )


def seed_trigger_inbox_later(
    *,
    bucket: str,
    run_id: str,
    spec_name: str,
    delay_seconds: float = 45.0,
    e2e_project: str | None = None,
    count: int = 2,
):
    """Drop frames into a trigger's inbox after ``delay_seconds``.

    Returns the started timer. Seeding late is what makes the live trigger test
    meaningful: the driver must poll an empty prefix, wait, and only then submit.
    """

    import threading

    from npa.clients.project_credentials import s3_client_for_project

    marker = f"npa-workflow-e2e/{run_id}/{spec_name.replace('.yaml', '')}"

    def _seed() -> None:
        client = s3_client_for_project(e2e_project, allow_host_creds=True)
        _seed_images(client, bucket=bucket, prefix=f"{marker}/inbox/", count=count)

    timer = threading.Timer(delay_seconds, _seed)
    timer.daemon = True
    timer.start()
    return timer


def concurrency_overlaps(tasks: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return task-name pairs whose [start_at, end_at] intervals overlap.

    This is the live proof that a ``parallel:`` group really ran concurrently:
    SkyPilot records per-task ``start_at`` / ``end_at`` for every member of a
    JobGroup, so overlapping intervals cannot be produced by a serial chain.
    """

    # SkyPilot leaves ``start_at`` unset for JobGroup members on some versions;
    # ``submitted_at`` is always recorded, so use it as the interval start.
    rows = [
        (
            str(task.get("task_name") or task.get("task_id")),
            float(task.get("start_at") or task.get("submitted_at") or 0.0),
            float(task.get("end_at") or 0.0),
        )
        for task in tasks
        if task.get("start_at") or task.get("submitted_at")
    ]
    overlaps: list[tuple[str, str]] = []
    for index, (name_a, start_a, end_a) in enumerate(rows):
        for name_b, start_b, end_b in rows[index + 1 :]:
            latest_start = max(start_a, start_b)
            earliest_end = min(end_a or float("inf"), end_b or float("inf"))
            if earliest_end > latest_start:
                overlaps.append((name_a, name_b))
    return overlaps


def write_runtime_evidence(name: str, payload: Any) -> Path:
    """Persist a runtime run's JSON summary for EVIDENCE.md (never contains secrets)."""

    log_dir = Path(
        os.environ.get("NPA_LIVE_E2E_LOG_DIR", "")
        or (Path.home() / "npa-live-e2e-logs")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"runtime-{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _seed_prefix_from_source(source: str, bucket: str, dest_prefix: str, client) -> None:
    """Copy a real dataset (``s3://`` prefix or local dir) into ``dest_prefix``."""

    source = source.strip()
    if source.startswith("s3://"):
        without = source[len("s3://") :]
        src_bucket, _, src_prefix = without.partition("/")
        src_prefix = src_prefix.lstrip("/")
        paginator = client.get_paginator("list_objects_v2")
        copied = 0
        for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(src_prefix) :].lstrip("/")
                if not rel:
                    continue
                client.copy_object(
                    Bucket=bucket,
                    Key=f"{dest_prefix}{rel}",
                    CopySource={"Bucket": src_bucket, "Key": key},
                )
                copied += 1
        if copied == 0:
            pytest.fail(f"NPA_E2E_SONIC_MOTION_SRC {source!r} had no objects to seed")
        return

    local_root = Path(source.replace("file://", ""))
    if not local_root.is_dir():
        pytest.fail(f"NPA_E2E_SONIC_MOTION_SRC {source!r} is not an s3:// URI or a directory")
    uploaded = 0
    for item in sorted(local_root.rglob("*")):
        if item.is_file():
            rel = item.relative_to(local_root).as_posix()
            client.upload_file(str(item), bucket, f"{dest_prefix}{rel}")
            uploaded += 1
    if uploaded == 0:
        pytest.fail(f"NPA_E2E_SONIC_MOTION_SRC {source!r} contained no files to seed")


def _seed_scene_frame(client, *, bucket: str, marker: str) -> None:
    from io import BytesIO

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        pytest.fail(f"Pillow required to seed scene fixtures: {exc}")
    image = Image.new("RGB", (320, 240), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 180, 320, 240], fill=(120, 90, 60))
    draw.rectangle([120, 100, 200, 180], fill=(180, 40, 40))
    buf = BytesIO()
    image.save(buf, format="PNG")
    client.put_object(
        Bucket=bucket,
        Key=f"{marker}/scene/frame_000.png",
        Body=buf.getvalue(),
        ContentType="image/png",
    )


def _seed_rollout_frames(
    client, *, bucket: str, marker: str, episodes: int = 1, frames: int = 4
) -> None:
    """Upload a short RGB rollout (a cube moving toward a target) to `rollouts/`.

    The VLM-eval tool discovers image frames recursively under the rollouts URI
    prefix, so a small deterministic sequence is enough real input for a live
    self-hosted VLM score.
    """
    from io import BytesIO

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        pytest.fail(f"Pillow required to seed rollout fixtures: {exc}")
    for episode in range(max(1, episodes)):
        for frame in range(max(1, frames)):
            image = Image.new("RGB", (320, 240), (30, 30, 30))
            draw = ImageDraw.Draw(image)
            # Static green target.
            draw.rectangle([250, 150, 300, 200], fill=(40, 200, 40))
            # Red cube advancing left→right across frames toward the target.
            x = 40 + frame * 50
            draw.rectangle([x, 150, x + 40, 200], fill=(200, 40, 40))
            draw.rectangle([0, 200, 320, 240], fill=(90, 70, 50))  # table
            buf = BytesIO()
            image.save(buf, format="PNG")
            client.put_object(
                Bucket=bucket,
                Key=f"{marker}/rollouts/episode_{episode:03d}/frame_{frame:03d}.png",
                Body=buf.getvalue(),
                ContentType="image/png",
            )


def materialize_live_spec(
    tmp_path: Path,
    name: str,
    *,
    bucket: str,
    run_id: str,
) -> Path:
    """Copy a golden spec with the live bucket and a unique e2e prefix."""

    text = resolve_spec_path(name).read_text(encoding="utf-8")
    text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
    marker = f"npa-workflow-e2e/{run_id}"
    # Keep per-spec prefix tokens but anchor runs under a shared e2e root.
    text = re.sub(
        r'(prefix:\s*")([^"]*)(")',
        lambda m: f'{m.group(1)}{marker}/{name.replace(".yaml", "")}{m.group(3)}',
        text,
        count=1,
    )
    # Optional bdd100k smoke knobs: synthesize rows so the pipeline runs without
    # a real BDD100K dataset, and shrink training epochs to keep the live run
    # bounded. Both are pure config toggles (synthetic_rows=0 -> real source).
    bdd_synth = os.environ.get("NPA_E2E_BDD100K_SYNTHETIC_ROWS", "").strip()
    if bdd_synth:
        text = re.sub(
            r'(synthetic_rows:\s*")[^"]*(")',
            lambda m: f"{m.group(1)}{bdd_synth}{m.group(2)}",
            text,
        )
    bdd_epochs = os.environ.get("NPA_E2E_BDD100K_EPOCHS", "").strip()
    if bdd_epochs:
        text = re.sub(
            r'(train_epochs:\s*")[^"]*(")',
            lambda m: f"{m.group(1)}{bdd_epochs}{m.group(2)}",
            text,
        )
    # Optional live remap, e.g. NPA_E2E_ACCELERATOR_REMAP=H100:1=RTXPRO6000:1,H200:1=L40S:1
    remap = os.environ.get("NPA_E2E_ACCELERATOR_REMAP", "").strip()
    if remap:
        for pair in remap.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            src, dst = pair.split("=", 1)
            src, dst = src.strip(), dst.strip()
            if src and dst:
                text = text.replace(f"accelerators: {src}", f"accelerators: {dst}")
    # Optional cloud remap for live capacity, e.g. NPA_E2E_CLOUD_REMAP=kubernetes=nebius
    cloud_remap = os.environ.get("NPA_E2E_CLOUD_REMAP", "").strip()
    if cloud_remap:
        for pair in cloud_remap.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            src, dst = pair.split("=", 1)
            src, dst = src.strip(), dst.strip()
            if src and dst:
                text = text.replace(f"cloud: {src}", f"cloud: {dst}")
    # Optional: inject accelerators into CPU-only resource profiles (Nebius CPU
    # docker images currently fail apt setup; L40S/H100 VMs are healthy).
    # Example: NPA_E2E_FORCE_ACCELERATORS=L40S:1
    force_accel = os.environ.get("NPA_E2E_FORCE_ACCELERATORS", "").strip()
    if force_accel:
        text = _force_accelerators_on_cpu_profiles(text, force_accel)
    # When remapping onto denser GPU nodes (e.g. RTXPRO), high cpu/mem floors
    # from H100-shaped twins fail prechecks. Optionally clamp to a smaller floor.
    if os.environ.get("NPA_E2E_RELAX_CPU_MEM", "").strip() in {"1", "true", "yes"} or (
        force_accel or remap
    ):
        text = _relax_all_cpu_mem_floors(
            text,
            cpus=os.environ.get("NPA_E2E_RELAX_CPUS", "4+"),
            memory=os.environ.get("NPA_E2E_RELAX_MEMORY", "16+"),
        )
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _relax_all_cpu_mem_floors(text: str, *, cpus: str, memory: str) -> str:
    """Rewrite every ``cpus`` / ``memory`` resource line to a smaller floor."""

    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if re.match(r"^(\s*)cpus:\s*\S+", line):
            indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
            out.append(f"{indent}cpus: {cpus}\n" if line.endswith("\n") else f"{indent}cpus: {cpus}")
            continue
        if re.match(r"^(\s*)memory:\s*\S+", line):
            indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
            suffix = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}memory: {memory}{suffix}")
            continue
        out.append(line)
    return "".join(out)

def _force_accelerators_on_cpu_profiles(text: str, accelerators: str) -> str:
    """Add ``accelerators`` to named resource profiles that lack them.

    Also relax exact ``cpus`` / ``memory`` to ``N+`` so GPU instance shapes
    (e.g. L40S) can satisfy the request — Nebius has no ``cpus=4,mem=16,L40S:1``.
    """

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_resources = False
    profile_lines: list[str] = []
    profile_has_accel = False

    def _relax_cpu_mem(line: str) -> str:
        match = re.match(r"^(\s*(?:cpus|memory):\s*)(\S+)(\s*)$", line)
        if not match:
            return line
        prefix, value, suffix = match.groups()
        raw = value.strip()
        if raw.endswith("+") or raw.endswith("*"):
            return line
        if raw.lower().endswith("gi"):
            raw = raw[:-2]
        elif raw.lower().endswith("g"):
            raw = raw[:-1]
        # Keep the original line ending; `suffix` already carries it when present.
        return f"{prefix}{raw}+{suffix}"

    def flush_profile() -> None:
        nonlocal profile_lines, profile_has_accel
        if not profile_lines:
            return
        if not profile_has_accel:
            inserted = False
            rebuilt: list[str] = []
            for pl in profile_lines:
                rebuilt.append(_relax_cpu_mem(pl))
                if not inserted and re.match(r"^    cloud:\s*", pl):
                    rebuilt.append(f"    accelerators: {accelerators}\n")
                    inserted = True
            if not inserted:
                rebuilt = [profile_lines[0], f"    accelerators: {accelerators}\n"] + [
                    _relax_cpu_mem(pl) for pl in profile_lines[1:]
                ]
            profile_lines = rebuilt
        out.extend(profile_lines)
        profile_lines = []
        profile_has_accel = False

    for line in lines:
        if re.match(r"^resources:\s*$", line):
            flush_profile()
            in_resources = True
            out.append(line)
            continue
        if in_resources:
            if re.match(r"^\S", line):
                flush_profile()
                in_resources = False
                out.append(line)
                continue
            if re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
                flush_profile()
                profile_lines = [line]
                profile_has_accel = False
                continue
            if profile_lines:
                if re.search(r"^\s*accelerators:\s*", line):
                    profile_has_accel = True
                profile_lines.append(line)
                continue
        out.append(line)
    flush_profile()
    return "".join(out)


def live_credential_markers() -> list[str]:
    """Collect credential substrings that must never appear in CLI output."""

    markers: list[str] = []
    try:
        from npa.clients.credentials import load_credentials

        storage = load_credentials().get("storage") or {}
        for key in ("aws_access_key_id", "aws_secret_access_key"):
            value = storage.get(key)
            if isinstance(value, str) and len(value) >= 8:
                markers.append(value)
    except Exception:
        pass
    for env_key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NEBIUS_AI_CLOUD_KEY",
        "NEBIUS_TOKEN_FACTORY_KEY",
    ):
        value = os.environ.get(env_key, "")
        if value and len(value) >= 8:
            markers.append(value)
    return markers


def assert_no_credential_leakage(
    text: str,
    *,
    extra_forbidden: Iterable[str] | None = None,
) -> None:
    """Fail when CLI output contains secrets or live credential material."""

    for pattern in _LEAK_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"credential pattern leaked: {match.group(0)[:32]!r}"
    for marker in extra_forbidden or ():
        if marker and len(marker) >= 8 and marker in text:
            raise AssertionError("live credential marker leaked in CLI output")


def assert_cli_ok(result: Result, *, forbidden: Iterable[str] | None = None) -> None:
    assert result.exit_code == 0, _cli_failure_detail(result)
    assert_no_credential_leakage(result.output, extra_forbidden=forbidden)


def _cli_failure_detail(result: Result) -> str:
    """Include the swallowed CliRunner exception; output alone hides real crashes."""

    parts = [f"exit_code={result.exit_code}", result.output or "(no output)"]
    exception = getattr(result, "exception", None)
    if exception is not None and not isinstance(exception, SystemExit):
        import traceback

        parts.append(
            "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        )
    stderr = getattr(result, "stderr", None)
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


def parse_json_output(result: Result, *, forbidden: Iterable[str] | None = None) -> Any:
    assert_cli_ok(result, forbidden=forbidden)
    return json.loads(result.output)


def parse_json_payload(result: Result, forbidden: Iterable[str]) -> dict[str, Any]:
    payload = parse_json_output(result, forbidden=forbidden)
    assert isinstance(payload, dict)
    return payload


def parse_runtime_json(result: Result, forbidden: Iterable[str]) -> dict[str, Any]:
    """Parse the JSON summary of a ``submit --runtime`` run.

    The runtime driver streams ``[runtime] ...`` progress to stderr, and
    ``CliRunner`` merges stderr into ``result.output`` on click < 8.2, so the JSON
    document has to be sliced out of the combined stream.
    """

    assert_cli_ok(result, forbidden=forbidden)
    text = result.output or ""
    start = text.find("\n{")
    start = 0 if text.lstrip().startswith("{") else (start + 1 if start >= 0 else -1)
    assert start >= 0, f"no JSON summary in runtime output:\n{text[-4000:]}"
    payload = json.loads(text[start:])
    assert isinstance(payload, dict)
    return payload
