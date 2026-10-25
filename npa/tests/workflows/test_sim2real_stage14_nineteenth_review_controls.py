from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workflows import data_factory_viz
from npa.workflows.rerun_serve import RerunServeConfig, verify_rrd_exists_on_s3
from npa.workflows.sim2real import (
    checkpoint_selection,
    legacy_heldout,
    publication,
    stage_execution,
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig


pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")

_ROOT = "s3://demo-bucket/sim2real/run-a"
_LOCK_URI = f"{_ROOT}/reports/.sim2real-publication.json"
_CANONICAL_REPORT = f"{_ROOT}/reports/sim2real-report.json"
_CANONICAL_RRD = f"{_ROOT}/reports/sim2real.rrd"
_CANONICAL_MCAP = f"{_ROOT}/reports/sim2real.mcap"
_CANONICAL_STAGE14 = f"{_ROOT}/components/stage_14.json"
_TRANSACTION_ID = "generation-a"
_IMMUTABLE_REPORT = (
    f"{_ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real-report.json"
)
_IMMUTABLE_RRD = f"{_ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real.rrd"
_STAGE14_BYTES = b'{"stage":14}\n'
_IMMUTABLE_STAGE14 = (
    f"{_ROOT}/components/history/stage_14/"
    f"{hashlib.sha256(_STAGE14_BYTES).hexdigest()}.json"
)


def _present(uri: str, immutable_uri: str, payload: bytes) -> dict[str, object]:
    return {
        "uri": uri,
        "state": "present",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "immutable_uri": immutable_uri,
    }


def _complete_journal() -> bytes:
    return publication._journal_bytes(
        transaction_id=_TRANSACTION_ID,
        attempt_id="a" * 32,
        state="committed",
        objects=[
            _present(
                _CANONICAL_REPORT,
                _IMMUTABLE_REPORT,
                b'{"visualization":{"source":"committed"}}\n',
            ),
            _present(_CANONICAL_RRD, _IMMUTABLE_RRD, b"committed-rrd"),
            {"uri": _CANONICAL_MCAP, "state": "absent"},
            _present(
                _CANONICAL_STAGE14,
                _IMMUTABLE_STAGE14,
                _STAGE14_BYTES,
            ),
        ],
    )


def test_data_factory_existing_rrd_resolves_through_publication_journal() -> None:
    class Store:
        def __init__(self) -> None:
            self.objects = {
                _LOCK_URI: _complete_journal(),
                _CANONICAL_RRD: b"stale-alias",
                _IMMUTABLE_RRD: b"committed-rrd",
            }

        def read_small_bytes_with_etag(
            self, uri: str, *, max_bytes: int = 1024 * 1024
        ) -> tuple[bytes, str] | None:
            payload = self.objects.get(uri)
            if payload is None:
                return None
            assert len(payload) <= max_bytes
            return payload, hashlib.sha256(payload).hexdigest()

    assert hasattr(data_factory_viz, "_resolve_existing_output_publication")
    resolved, publication = data_factory_viz._resolve_existing_output_publication(
        Store(),
        _CANONICAL_RRD,
        inventory_keys={
            _LOCK_URI.removeprefix("s3://demo-bucket/"),
            _CANONICAL_RRD.removeprefix("s3://demo-bucket/"),
        },
    )
    assert resolved == _IMMUTABLE_RRD
    assert publication is not None and publication.journaled


def test_data_factory_probes_journal_when_inventory_omits_it() -> None:
    class Store:
        calls: list[str] = []

        def read_small_bytes_with_etag(
            self, uri: str, *, max_bytes: int = 1024 * 1024
        ) -> tuple[bytes, str] | None:
            self.calls.append(uri)
            return None

    store = Store()
    resolved, publication = data_factory_viz._resolve_existing_output_publication(
        store,
        _CANONICAL_RRD,
        inventory_keys=set(),
    )

    assert resolved == ""
    assert publication is not None and not publication.journaled
    assert store.calls == [_LOCK_URI]


def test_diagnostic_heldout_reruns_use_unique_component_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identities: list[str] = []
    tokens = iter(("1" * 32, "2" * 32))
    monkeypatch.setattr(legacy_heldout.secrets, "token_hex", lambda _size: next(tokens))
    monkeypatch.setattr(
        legacy_heldout,
        "_resolve_env_records_s3_uri",
        lambda uri: f"{uri.rstrip('/')}/envs.jsonl",
    )
    monkeypatch.setattr(
        legacy_heldout, "_component_env", lambda *_args, extra, **_kwargs: dict(extra)
    )
    monkeypatch.setattr(legacy_heldout, "_byo_robot_env", lambda _config: {})
    monkeypatch.setattr(
        legacy_heldout, "_heldout_k8s_image_ready", lambda _config: True
    )
    monkeypatch.setattr(
        legacy_heldout,
        "_component_attempt_id",
        lambda _config, _component, identity: (
            identities.append(identity),
            hashlib.sha256(identity.encode()).hexdigest()[:32],
        )[1],
    )
    monkeypatch.setattr(
        legacy_heldout,
        "_upload_component_file",
        lambda *_args, **kwargs: (
            f"s3://bucket/input/{kwargs['attempt_id']}/{kwargs['name']}"
        ),
    )
    monkeypatch.setattr(
        legacy_heldout,
        "_component_output_uri",
        lambda *_args, **kwargs: (
            f"s3://bucket/output/{kwargs['attempt_id']}/report.json"
        ),
    )

    def run_image(
        _image: str,
        *,
        output_json: Path,
        output_uri: str,
        **_kwargs: Any,
    ) -> dict[str, str]:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.heldout_eval.v1",
                    "per_env": [
                        {
                            "env_id": "validation-0001",
                            "score": 1.0,
                            "success": True,
                            "details": {},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"mode": "kubernetes_job", "output_uri": output_uri}

    monkeypatch.setattr(legacy_heldout, "_run_image_component", run_image)
    monkeypatch.setattr(
        legacy_heldout,
        "_read_component_json",
        lambda path, _invocation: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        legacy_heldout,
        "_ensure_heldout_renders_for_viz",
        lambda _config, _local_dir, report, **_kwargs: report,
    )
    config = Sim2RealLoopConfig(
        run_id="diagnostic-rerun",
        output_dir=tmp_path,
        s3_bucket="bucket",
        s3_prefix="sim2real",
        validation_envs_uri="s3://bucket/sim2real/diagnostic-rerun/envs/validation",
        validation_env_count=1,
        heldout_env_count=1,
        eval_image="registry/eval@sha256:" + "a" * 64,
    )
    for _ in range(2):
        legacy_heldout.run_heldout_eval(
            config,
            local_dir=tmp_path,
            inner_evidence={},
            outer_iteration=1,
            evaluation_split="validation",
            inner_iteration=1,
            checkpoint_iteration=1,
        )

    assert len(identities) == 2
    assert identities[0] != identities[1]
    assert identities == [
        "validation-outer-01-iter-01-checkpoint-0001-attempt-" + "1" * 32,
        "validation-outer-01-iter-01-checkpoint-0001-attempt-" + "2" * 32,
    ]


def test_legacy_validation_resume_reruns_incomplete_checkpoint_identity() -> None:
    checkpoint_uri = "s3://bucket/run/model.pt"
    legacy_report = {
        "report_uri": "s3://bucket/run/validation/legacy.json",
        "success_rate": 1.0,
        "per_env": [{"env_id": "validation-0001", "success": True}],
    }
    digest = "b" * 64
    replacement_report = {
        **legacy_report,
        "report_uri": "s3://bucket/run/validation/replacement.json",
        "policy_checkpoint_sha256": digest,
        "policy_checkpoint_size_bytes": 256,
        "policy_inference_provenance": {"generator_policy_sha256": digest},
    }

    class Durable:
        def __init__(self) -> None:
            self.committed: list[dict[str, object]] = []

        def load_unit(
            self, _unit: str, _inputs: dict[str, object]
        ) -> dict[str, object]:
            return {"report": legacy_report}

        def commit_unit(
            self,
            unit: str,
            inputs: dict[str, object],
            payload: dict[str, object],
        ) -> None:
            self.committed.append({"unit": unit, "inputs": inputs, "payload": payload})

    durable = Durable()
    calls: list[str] = []
    assert hasattr(stage_execution, "_resume_or_run_validation_candidate")
    report, candidate = stage_execution._resume_or_run_validation_candidate(
        durable,
        unit="validation-unit",
        validation_input={"checkpoint_uri": checkpoint_uri},
        run_evaluation=lambda: (calls.append("rerun"), replacement_report)[1],
        checkpoint_uri=checkpoint_uri,
        outer_iteration=1,
        inner_iteration=1,
        training_iteration=1,
    )

    assert calls == ["rerun"]
    assert report["report_uri"].endswith("/replacement.json")
    assert durable.committed
    assert checkpoint_selection.checkpoint_candidate_has_complete_identity(candidate)


def test_rerun_head_only_bound_client_cannot_bypass_publication_journal() -> None:
    class Store:
        def __init__(self) -> None:
            self.objects = {
                _LOCK_URI: _complete_journal(),
                _CANONICAL_RRD: b"stale-alias",
                _IMMUTABLE_RRD: b"committed-rrd",
            }

        @staticmethod
        def _uri(bucket: str, key: str) -> str:
            return f"s3://{bucket}/{key}"

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            payload = self.objects[self._uri(Bucket, Key)]
            return {
                "ContentLength": len(payload),
                "Metadata": {"npa-sha256": hashlib.sha256(payload).hexdigest()},
            }

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            payload = self.objects[self._uri(Bucket, Key)]
            return {
                "Body": io.BytesIO(payload),
                "ETag": hashlib.sha256(payload).hexdigest(),
            }

    storage = Store()
    resolved, digest, size = verify_rrd_exists_on_s3(
        RerunServeConfig(
            run_id="run-a",
            s3_bucket="demo-bucket",
            s3_prefix="sim2real",
        ),
        head_object=storage.head_object,
        include_identity=True,
    )

    assert resolved == _IMMUTABLE_RRD
    assert digest == hashlib.sha256(b"committed-rrd").hexdigest()
    assert size == len(b"committed-rrd")
