"""Committed generations must not overwrite another reader's verified cache."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcap.writer import Writer

from npa.agent_backend.canonical_mcap import prepare_canonical_mcap
from npa.workflows.sim2real import publication
from .test_canonical_mcap import _S3, _journal_present, _safe_key


ROOT = "s3://bucket/runs/run-1"


def _mcap_bytes(value: int) -> bytes:
    stream = io.BytesIO()
    writer = Writer(stream)
    writer.start()
    channel = writer.register_channel(
        topic="/control", message_encoding="json", schema_id=0
    )
    writer.add_message(
        channel_id=channel,
        log_time=value,
        publish_time=value,
        data=f'{{"value":{value}}}'.encode(),
    )
    writer.finish()
    return stream.getvalue()


def _select_generation(store, generation: str, payload: bytes) -> None:
    targets = []
    for suffix in (
        "reports/sim2real-report.json",
        "reports/sim2real.rrd",
        "reports/sim2real.mcap",
    ):
        immutable = f"{ROOT}/reports/generations/{generation}/{Path(suffix).name}"
        body = payload if suffix.endswith(".mcap") else b"transport control"
        store.objects[immutable.removeprefix("s3://bucket/")] = body
        targets.append(_journal_present(f"{ROOT}/{suffix}", immutable, body))
    component = b"stage14 transport control"
    history = f"{ROOT}/components/history/stage_14/{hashlib.sha256(component).hexdigest()}.json"
    store.objects[history.removeprefix("s3://bucket/")] = component
    targets.append(
        _journal_present(f"{ROOT}/components/stage_14.json", history, component)
    )
    store.objects["runs/run-1/reports/.sim2real-publication.json"] = (
        publication._journal_bytes(
            transaction_id=generation,
            attempt_id="a" * 32,
            state="committed",
            objects=targets,
        )
    )


def _prepare(store, directory: Path):
    from npa.sdk.workbench.foxglove import inspect_mcap

    def find(*_args, **_kwargs):
        return "bucket", [
            SimpleNamespace(key=key, s3_uri=f"s3://bucket/{key}")
            for key in store.objects
        ]

    return prepare_canonical_mcap(
        run_id="run-1",
        fps=10.0,
        max_frames=10,
        validate_run_id=lambda value: value,
        s3_client=lambda: (store, {"prefix": "runs"}),
        list_buckets=lambda *_args: ["bucket"],
        find_artifacts=find,
        safe_key=_safe_key,
        download=lambda *_a, **_kw: None,
        convert=lambda **_kwargs: None,
        summarize=inspect_mcap,
        invalidate_cache=lambda: None,
        now_iso=lambda: "2026-10-03T00:00:00Z",
        recordings_dir=directory,
    )


def test_committed_mcap_generations_have_distinct_verified_local_paths(
    tmp_path,
) -> None:
    store = _S3()
    first_bytes, second_bytes = _mcap_bytes(1), _mcap_bytes(2)
    _select_generation(store, "a" * 64, first_bytes)
    first = _prepare(store, tmp_path)
    assert Path(first["local_path"]).read_bytes() == first_bytes
    _select_generation(store, "b" * 64, second_bytes)
    second = _prepare(store, tmp_path)

    assert first["local_path"] != second["local_path"]
    assert Path(first["local_path"]).read_bytes() == first_bytes
    assert Path(second["local_path"]).read_bytes() == second_bytes
    assert first["summary"]["output"] == first["local_path"]
    assert second["summary"]["output"] == second["local_path"]


def test_rejected_mcap_replacement_cannot_mutate_verified_cache(tmp_path) -> None:
    store = _S3()
    first_bytes = _mcap_bytes(1)
    generation = "a" * 64
    _select_generation(store, generation, first_bytes)
    first = _prepare(store, tmp_path)
    immutable_key = f"runs/run-1/reports/generations/{generation}/sim2real.mcap"
    store.objects[immutable_key] = _mcap_bytes(2)
    remote_before = dict(store.objects)

    with pytest.raises(RuntimeError, match="bytes disagree"):
        _prepare(store, tmp_path)

    assert store.objects == remote_before
    assert Path(first["local_path"]).read_bytes() == first_bytes
