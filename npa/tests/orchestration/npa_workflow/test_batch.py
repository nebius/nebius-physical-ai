"""Exercise batch expansion, actual child concurrency, and uncertain-run recovery."""

import json
from pathlib import Path
import sys
import subprocess

import pytest
import yaml

from npa.orchestration.npa_workflow import batch
from npa.orchestration.npa_workflow.batch_plan import plan_batch

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def manifest(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("""
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: batch-test}
config:
  bucket: test-bucket
  prefix: "batch-test/{{run.id}}"
initial: process
states:
  process:
    run: {argv: [echo, hello]}
    outputs:
      - uri: "s3://{{config.bucket}}/{{config.prefix}}/result.json"
    terminal: true
""")
    document = {
        "apiVersion": "npa.workflow.batch/v0.0.1",
        "batch_id": "test-batch",
        "workflow": str(workflow),
        "project": "test",
        "entries": [{"id": f"item-{index}"} for index in range(6)],
    }
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, document


def _save(manifest):
    path, document = manifest
    path.write_text(yaml.safe_dump(document))
    return path


def test_four_datasets_eighty_episodes(manifest):
    path, document = manifest
    document["workflow"] = str(ROOT / "workflows/main/paidf-cosmos3.yaml")
    document["entries"] = [
        {
            "id": f"dataset-{index}",
            "lerobot_uri": f"s3://test-bucket/dataset-{index}/",
            "lerobot_camera": "observation.images.front",
            "episodes": {"start": 0, "stop": 80},
        }
        for index in range(4)
    ]
    plan = plan_batch(_save(manifest))
    assert len(plan["runs"]) == 320
    assert len({run["run_id"] for run in plan["runs"]}) == 320
    assert plan["runs"][-1]["episode"] == 79
    assert "--require-explicit-lerobot-selection" in plan["runs"][0]["input_args"]


def test_two_hundred_episodes_twenty_variants(manifest):
    _, document = manifest
    document["workflow"] = str(ROOT / "workflows/main/paidf-cosmos3.yaml")
    document["vars"] = {"variant_count": 20}
    document["entries"] = [
        {
            "id": "dataset",
            "lerobot_uri": "s3://test-bucket/dataset/",
            "lerobot_camera": "observation.images.front",
            "episodes": {"start": 0, "stop": 200},
        }
    ]
    runs = plan_batch(_save(manifest))["runs"]
    assert sum(int(run["vars"]["variant_count"]) for run in runs) == 4000


@pytest.mark.parametrize(
    "change,match",
    [
        ({"entries": [{"id": "same"}, {"id": "same"}]}, "duplicate"),
        ({"vars": {"prefix": "shared"}}, "run.id"),
        ({"entries": [{"id": "x", "episodes": [0]}]}, "require lerobot_uri"),
        ({"unknown": True}, "Extra inputs"),
    ],
)
def test_invalid_manifest_fails_before_launch(manifest, change, match):
    manifest[1].update(change)
    with pytest.raises(ValueError, match=match):
        plan_batch(_save(manifest))


def test_rejects_shared_declared_output(manifest):
    workflow = Path(manifest[1]["workflow"])
    workflow.write_text(
        workflow.read_text().replace("{{config.prefix}}/result", "shared/result")
    )
    with pytest.raises(ValueError, match="collision"):
        plan_batch(manifest[0])


def test_duplicate_yaml_keys_rejected(manifest):
    with manifest[0].open("a") as stream:
        stream.write("batch_id: replacement\n")
    with pytest.raises(ValueError, match="duplicate YAML"):
        plan_batch(manifest[0])


@pytest.fixture
def children(tmp_path, monkeypatch):
    script = tmp_path / "submit.py"
    journal = tmp_path / "children.json"
    journal.write_text(json.dumps({"active": 0, "maximum": 0, "calls": []}))
    script.write_text("""
import fcntl, json, os, sys, time
from pathlib import Path
args = sys.argv[2:]
resume = '--resume-run' in args
run_id = args[args.index('--resume-run' if resume else '--run-id') + 1]
path = Path(sys.argv[1])
def update(delta):
    with path.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        state['active'] += delta
        state['maximum'] = max(state['maximum'], state['active'])
        if delta > 0:
            state['calls'].append({'run_id': run_id, 'resume': resume, 'args': args})
        handle.seek(0)
        json.dump(state, handle)
        handle.truncate()
update(1)
minimum_started = int(os.getenv('BATCH_TEST_START_BARRIER', '0'))
while minimum_started:
    with path.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        started = len(json.load(handle)['calls'])
    if started >= minimum_started:
        break
    time.sleep(0.01)
time.sleep(0.3)
update(-1)
if os.getenv('BATCH_TEST_FAILURE') == run_id and not resume:
    print('lost response')
    sys.exit(1)
print(json.dumps({'status': 'succeeded', 'run_id': run_id}))
""")
    monkeypatch.setattr(
        batch,
        "internal_cli_argv",
        lambda args: [sys.executable, str(script), str(journal), *args],
    )
    return journal


def test_real_children_obey_limit_and_completed_resume_reverifies(
    manifest, children, tmp_path, monkeypatch
):
    # Startup and durable writes may outlast the synthetic child workload.
    # Hold the first wave until every slot has started to measure real overlap.
    monkeypatch.setenv("BATCH_TEST_START_BARRIER", "3")
    result = batch.run_batch(
        manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=3
    )
    assert all(record["status"] == "succeeded" for record in result["runs"].values())
    evidence = json.loads(children.read_text())
    assert evidence["maximum"] == 3
    assert len(evidence["calls"]) == 6
    assert all("--runtime" in call["args"] for call in evidence["calls"])
    assert all(
        call["args"][call["args"].index("--max-wait-seconds") + 1] == "0"
        for call in evidence["calls"]
    )
    monkeypatch.setenv("BATCH_TEST_START_BARRIER", "9")
    batch.run_batch(
        manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=3, resume=True
    )
    replay = json.loads(children.read_text())
    assert len(replay["calls"]) == 12
    assert all(call["resume"] for call in replay["calls"][6:])
    assert replay["maximum"] == 3


def test_failure_stops_admission_and_resume_reconciles_first(
    manifest, children, tmp_path, monkeypatch
):
    monkeypatch.setenv("BATCH_TEST_FAILURE", "test-batch-item-0")
    result = batch.run_batch(
        manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=2
    )
    assert result["runs"]["test-batch-item-0"]["status"] == "reconcile-required"
    assert result["runs"]["test-batch-item-2"]["status"] == "pending"
    assert len(json.loads(children.read_text())["calls"]) == 2
    result = batch.run_batch(
        manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=2, resume=True
    )
    calls = json.loads(children.read_text())["calls"]
    assert calls[2]["run_id"] == "test-batch-item-0"
    assert calls[2]["resume"] is True
    assert all(record["status"] == "succeeded" for record in result["runs"].values())
    assert len(calls) == 8


def test_changed_identity_refuses_resume(manifest, children, tmp_path):
    batch.run_batch(manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=2)
    manifest[1]["vars"] = {"bucket": "another-test-bucket"}
    with pytest.raises(ValueError, match="identity changed"):
        batch.run_batch(
            _save(manifest),
            state_dir=tmp_path / "state",
            max_concurrent_runs=2,
            resume=True,
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


def test_lock_rejects_second_driver(manifest, tmp_path):
    directory = tmp_path / "state/test-batch"
    with batch._batch_lock(directory):
        with pytest.raises(ValueError, match="active local driver"):
            batch.run_batch(
                manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=1
            )


def test_child_retains_lock_after_parent_releases_it(tmp_path):
    directory = tmp_path / "batch"
    with batch._batch_lock(directory) as descriptor:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            pass_fds=(descriptor,),
        )
    try:
        with pytest.raises(ValueError, match="active local driver"):
            with batch._batch_lock(directory):
                pytest.fail("a live child must retain the lock")
    finally:
        child.communicate()
    with batch._batch_lock(directory):
        assert child.returncode == 0


def test_corrupted_state_cannot_relaunch(manifest, children, tmp_path):
    state_dir = tmp_path / "state"
    batch.run_batch(manifest[0], state_dir=state_dir, max_concurrent_runs=3)
    path = state_dir / "test-batch/state.json"
    state = json.loads(path.read_text())
    state["runs"].pop("test-batch-item-0")
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="run identities"):
        batch.run_batch(
            manifest[0], state_dir=state_dir, max_concurrent_runs=3, resume=True
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


def test_missing_state_cannot_reset_completed_runs(manifest, children, tmp_path):
    state_dir = tmp_path / "state"
    batch.run_batch(manifest[0], state_dir=state_dir, max_concurrent_runs=3)
    (state_dir / "test-batch/state.json").unlink()
    with pytest.raises(ValueError, match="absence cannot be proven"):
        batch.run_batch(
            manifest[0],
            state_dir=state_dir,
            max_concurrent_runs=3,
            resume=True,
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


def test_cli_plan_and_missing_input_are_json(manifest):
    from typer.testing import CliRunner
    from npa.cli.workbench.workflow import app

    runner = CliRunner()
    result = runner.invoke(app, ["batch", "plan", str(manifest[0])])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["runs"]) == 6
    result = runner.invoke(app, ["check-input"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"


def test_batch_argv_parses_against_real_submit_command(manifest):
    from typer.main import get_command
    from npa.cli.workbench.workflow import app

    plan = plan_batch(manifest[0])
    arguments = batch._argv(plan, plan["runs"][0], False)
    submit = get_command(app).commands["submit"]
    with submit.make_context("submit", arguments[6:]) as context:
        assert context.params["runtime"] is True
        assert context.params["deploy_if_absent"] is False
        assert context.params["max_wait_seconds"] == 0


def test_private_stage_documentation_validates_and_preserves_command(tmp_path):
    import re
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import render_skypilot_yaml

    guide = (ROOT / "docs/workbench/guides/paidf-dataset-batches.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)```", guide, re.DOTALL)
    path = tmp_path / "private-processing.yaml"
    path.write_text(blocks[1])
    spec = load_spec(path)
    plan = build_plan(spec, run_id="private-stage-test")
    assert plan.steps[0].argv[0] == "/opt/company/bin/process"
    assert plan.steps[0].argv[-1] == "private-stage-test"
    assert plan.steps[0].outputs[0]["uri"].endswith("/private-stage-test/result.json")
    rendered = render_skypilot_yaml(spec, plan, run_id="private-stage-test")
    assert "/opt/company/bin/process" in rendered
    assert "registry.example.com/team/private-adapter" in rendered


def test_source_change_refuses_resume(manifest, children, tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    batch.run_batch(manifest[0], state_dir=state_dir, max_concurrent_runs=3)
    from npa.orchestration.npa_workflow import batch_plan

    monkeypatch.setattr(batch_plan, "source_fingerprint", lambda _: "changed-source")
    with pytest.raises(ValueError, match="identity changed"):
        batch.run_batch(
            manifest[0], state_dir=state_dir, max_concurrent_runs=3, resume=True
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


def test_config_change_refuses_resume(manifest, children, tmp_path):
    config = tmp_path / "sky.yaml"
    config.write_text("kubernetes: {namespace: test-namespace}\n")
    manifest[1]["config_path"] = str(config)
    state_dir = tmp_path / "state"
    batch.run_batch(_save(manifest), state_dir=state_dir, max_concurrent_runs=3)
    assert (
        state_dir / "test-batch/skypilot-config.yaml"
    ).read_text() == config.read_text()
    config.write_text("kubernetes: {namespace: changed-namespace}\n")
    with pytest.raises(ValueError, match="identity changed"):
        batch.run_batch(
            manifest[0], state_dir=state_dir, max_concurrent_runs=3, resume=True
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


@pytest.mark.parametrize("filename", ["batch.lock", "plan.tmp", "workflow.yaml"])
def test_ledger_symlink_cannot_write_outside(manifest, tmp_path, filename):
    directory = tmp_path / "state/test-batch"
    directory.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside"
    outside.write_text("retain original")
    (directory / filename).symlink_to(outside)
    with pytest.raises(OSError):
        batch.run_batch(
            manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=1
        )
    assert outside.read_text() == "retain original"


def test_world_readable_ledger_is_rejected(manifest, tmp_path):
    directory = tmp_path / "state/test-batch"
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        batch.run_batch(
            manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=1
        )


def test_log_traversal_refuses_reconciliation(manifest, children, tmp_path):
    state_dir = tmp_path / "state"
    batch.run_batch(manifest[0], state_dir=state_dir, max_concurrent_runs=3)
    path = state_dir / "test-batch/state.json"
    state = json.loads(path.read_text())
    state["runs"]["test-batch-item-0"]["stdout"] = "../different-run.json"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="logs do not match"):
        batch.run_batch(
            manifest[0], state_dir=state_dir, max_concurrent_runs=3, resume=True
        )
    assert len(json.loads(children.read_text())["calls"]) == 6


def test_recorded_success_is_not_trusted_when_recheck_fails(
    manifest, children, tmp_path, monkeypatch
):
    state_dir = tmp_path / "state"
    batch.run_batch(manifest[0], state_dir=state_dir, max_concurrent_runs=3)
    original = batch._finish

    def failed_recheck(process, run_id, directory, state):
        result = original(process, run_id, directory, state)
        if run_id == "test-batch-item-0":
            state["runs"][run_id]["status"] = "reconcile-required"
            batch._write_json(directory / "state.json", state)
            return False
        return result

    monkeypatch.setattr(batch, "_finish", failed_recheck)
    result = batch.run_batch(
        manifest[0], state_dir=state_dir, max_concurrent_runs=3, resume=True
    )
    assert result["runs"]["test-batch-item-0"]["status"] == "reconcile-required"
    assert all(call["resume"] for call in json.loads(children.read_text())["calls"][6:])


def test_ledger_hardlink_does_not_truncate_external_file(manifest, tmp_path):
    import os

    directory = tmp_path / "state/test-batch"
    directory.mkdir(parents=True, mode=0o700)
    external = tmp_path / "outside"
    external.write_text("retain original")
    os.link(external, directory / "workflow.yaml")
    with pytest.raises(ValueError, match="owned regular"):
        batch.run_batch(
            manifest[0], state_dir=tmp_path / "state", max_concurrent_runs=1
        )
    assert external.read_text() == "retain original"


def test_planned_storage_is_pinned_against_ambient_submit_defaults(
    manifest, monkeypatch
):
    monkeypatch.setenv("NPA_S3_BUCKET", "ambient-other-bucket")
    monkeypatch.setenv("NPA_S3_PREFIX", "ambient-shared-prefix")
    plan = plan_batch(manifest[0])
    run = plan["runs"][0]
    assert run["vars"]["bucket"] == "test-bucket"
    assert run["vars"]["prefix"] == "batch-test/{{run.id}}"
    args = batch._argv(plan, run, False)
    assert "bucket=test-bucket" in args
    assert "prefix=batch-test/{{run.id}}" in args
    assert all("ambient-" not in value for value in args)


def test_yaml_object_construction_is_rejected_without_execution(manifest, tmp_path):
    marker = tmp_path / "unexpected-execution"
    manifest[0].write_text(f"!!python/object/apply:os.system ['touch {marker}']\n")
    with pytest.raises(yaml.YAMLError):
        plan_batch(manifest[0])
    assert not marker.exists()
