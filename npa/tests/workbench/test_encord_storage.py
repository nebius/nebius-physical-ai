"""S3-only artifact writer for receipts, manifests, and reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from encord_fakes import FakeStorage
from npa.workbench.encord.schemas import EncordToolError
from npa.workbench.encord.storage import write_json


def test_write_json_rejects_non_s3_destinations(tmp_path: Path) -> None:
    with pytest.raises(EncordToolError, match="expected an s3:// URI"):
        write_json(
            {"a": 1},
            result_uri=str(tmp_path / "receipt.json"),
            filename="receipt.json",
            storage_client=FakeStorage(),
        )


