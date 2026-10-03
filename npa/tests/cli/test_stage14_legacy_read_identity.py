"""Legacy publication aliases still need strong authorization-to-read identity."""

from __future__ import annotations

import io
import sys

import pytest
from fastapi import HTTPException

from tests.cli.test_stage14_twentieth_agent_controls import _import_rendered_backend


def test_legacy_publication_read_rejects_missing_strong_identity(
    monkeypatch, tmp_path
) -> None:
    name = "npa_rendered_legacy_missing_identity"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=name)
    alias = "sim2real/run-a/reports/sim2real.rrd"
    reads = []

    class Store:
        def get_object(self, **kwargs):
            if kwargs["Key"].endswith(".sim2real-publication.json"):
                raise KeyError(kwargs["Key"])
            reads.append(kwargs)
            return {"Body": io.BytesIO(b"EVIL!")}

    artifact = module.Artifact(
        run_id="run-a",
        key=alias,
        s3_uri=f"s3://demo-bucket/{alias}",
        size=5,
        last_modified="",
        render="rerun",
        inline=False,
        namespace="sim2real",
        relative_key="reports/sim2real.rrd",
    )
    try:
        with pytest.raises(HTTPException) as caught:
            module._verified_publication_artifact_body(Store(), "demo-bucket", artifact)
        assert caught.value.status_code == 409
        assert not reads
    finally:
        sys.modules.pop(name, None)
