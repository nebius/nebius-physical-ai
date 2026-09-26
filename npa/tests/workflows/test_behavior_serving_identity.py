"""Bind standalone evaluator wire dependencies to the frozen serving artifact."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from npa.workflows.behavior_challenge import serving_identity


@pytest.mark.parametrize("kind", ["comet12", "rlc"])
@pytest.mark.parametrize(
    "helper",
    ["evaluator_versions.py", "evaluator_wire.py", "nonreporting_train.py"],
)
def test_changed_wire_helper_rejects_frozen_policy(tmp_path, monkeypatch, kind, helper):
    source = Path(serving_identity.__file__).parent
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    names = set(serving_identity._SERVING_FILES) | {
        "evaluator_versions.py",
        "evaluator_wire.py",
    }
    for name in names:
        shutil.copyfile(source / name, runtime / name)
    monkeypatch.setattr(
        serving_identity, "__file__", str(runtime / "serving_identity.py")
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"frozen test checkpoint")
    args = SimpleNamespace(
        policy_archive=checkpoint,
        policy_kind=kind,
        policy_execution_variant="native",
    )
    policy = {
        "artifacts": {
            "checkpoint": {
                "sha256": serving_identity.file_digest(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            "serving": serving_identity.serving_artifact(args),
        }
    }
    serving_identity.verify_serving_identity(args, policy)

    with (runtime / helper).open("ab") as stream:
        stream.write(b"\n# Changed after the policy was frozen.\n")

    with pytest.raises(ValueError, match="serving code or configuration differs"):
        serving_identity.verify_serving_identity(args, policy)
