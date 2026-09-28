"""Embed actual scored native rollout frames while excluding later observer-only motion."""

import json
from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _read
from npa.workflows.field_failure.native_artifacts import _bundle
from npa.workflows.navigation.artifacts import materialize


def preview_groups(args, final):
    """Load verified evaluation artifacts and prepare self-contained scored-frame previews.

    Args:
        args: Run artifact root and identifier.
        final: Final comparison record, or None when development blocked final evaluation.
    Returns:
        Embedded preview groups with explicit scored intervals and focal outcomes.
    Raises:
        ValueError: Checkpoint, frame chronology, or scored interval is inconsistent.
        OSError: Required measured frames or native evidence are missing.
    """
    with tempfile.TemporaryDirectory(prefix="npa-rl-preview-") as temporary:
        root = Path(temporary)
        groups = []
        for arm in ("baseline", "candidate"):
            source = (
                materialize(args.output_root + "/development/" + arm, root / arm)
                if final is None
                else _final_artifacts(args, final, arm, root)
            )
            groups.append(
                _preview(source, arm, "Development" if final is None else "Final")
            )
        return groups


def _final_artifacts(args, final, arm, root):
    prefix = args.output_root + "/loop/" + args.run_id
    record, _ = _read(
        prefix + "/" + arm + "-evaluation.json",
        final["record_sha256"][arm + "_evaluation"],
    )
    evidence_prefix = prefix + "/" + arm + "-evaluation/attempt-" + record["attempt_id"]
    native, _ = _read(evidence_prefix + "/native-evaluation-0.json")
    if native["policy"] != final[arm]:
        raise ValueError("native preview belongs to another checkpoint")
    source = _bundle(native["artifacts"], root, arm)
    report = json.loads((source / "evaluation.json").read_text())
    if report["checkpoint_sha256"] != final[arm]["checkpoint"]["sha256"]:
        raise ValueError("native preview checkpoint differs from the final comparison")
    return source


def _preview(source, arm, cohort):
    from npa.workflows.navigation.preview import scored_rollout_group

    report = json.loads((source / "evaluation.json").read_text())
    return scored_rollout_group(source, report, title=cohort + " · " + arm.title())
