import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "npa/workflows/workbench/cosmos3-wam-slurm/paired_replay_report.py"
EPISODES = (
    ROOT / "docs/workbench/evidence/cosmos3-wam-rtx-libero-phase-sampler/episodes"
)


def pair(task):
    return [
        EPISODES / str(task) / arm / file
        for arm in ("control", "sampler")
        for file in ("summary.json", "stages.jsonl")
    ]


def run(args):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True)


def test_public_cohort():
    result = run([arg for task in (3, 6, 9) for arg in ("--pair", *pair(task))])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["total"] == {
        "control_success": 113,
        "sampler_success": 111,
        "gain": 25,
        "regression": 27,
        "pairs": 150,
        "both_success": 86,
        "both_failure": 12,
    }
    for task in report["tasks"].values():
        for outcome in ("gain", "regression"):
            failed = {
                state
                for states in task["unmet_terminal_goals"][outcome].values()
                for state in states
            }
            assert len(failed) == task["counts"][outcome]
    failures = {
        k: v["losing_arm_states"]["regression"] for k, v in report["tasks"].items()
    }
    assert len(failures["3"]["bowl_inside_drawer_not_closed"]) == 12
    assert failures["6"]["two_pad_contact_pudding_never_placed"] == [6, 7, 9, 36, 48]


@pytest.mark.parametrize(
    "fault", ["state_hash", "steps", "duplicate", "missing", "boolean"]
)
def test_refuses_unmatched_evidence(tmp_path, fault):
    paths = pair(3)
    rows = [json.loads(line) for line in paths[3].read_text().splitlines()]
    if fault == "state_hash":
        rows[0]["init_sha256"] = "a" * 64
    elif fault == "steps":
        rows[0]["replay_steps"] += 1
    elif fault == "duplicate":
        rows.append(rows[1])
    elif fault == "missing":
        rows.pop()
    else:
        rows[0]["replay_success"] = "false"
    paths[3] = tmp_path / "stages.jsonl"
    paths[3].write_text("\n".join(map(json.dumps, rows)))
    result = run(["--pair", *paths])
    assert result.returncode != 0 and not result.stdout
    assert "invalid comparison:" in result.stderr
