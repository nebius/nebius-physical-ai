from __future__ import annotations

import pytest

from npa.orchestration.npa_workflow.artifacts import require_input_artifacts, s3_object_exists
from npa.orchestration.npa_workflow.errors import NpaWorkflowError


def test_s3_object_exists_uses_checker() -> None:
    seen: list[str] = []

    def checker(_bucket: str, key: str) -> bool:
        seen.append(key)
        return key.endswith("manifest.json")

    assert s3_object_exists("s3://bucket/prefix/manifest.json", checker=checker)
    assert seen == ["prefix/manifest.json"]


def test_require_input_artifacts_raises_when_missing() -> None:
    with pytest.raises(NpaWorkflowError, match="missing required input"):
        require_input_artifacts(
            ["s3://bucket/missing.json"],
            checker=lambda _bucket, _key: False,
        )
