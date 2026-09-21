"""Original evidence, real subprocess locks, and hostile provider boundaries."""

import copy
from datetime import datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import yaml

from npa.cluster.absent_evidence import AbsenceRecoveryError, load_legacy_evidence
from npa.cluster.absent_journal import recovery_lease
from npa.cluster.absent_provider import authority_environment
from npa.provisioning_journal import load_operation

NATIVE_IDENTITIES = [
    (
        "nebius_mk8s_v1_cluster",
        "cluster-example",
        "project-example",
        "owned-example",
        None,
    ),
    (
        "nebius_mk8s_v1_node_group",
        "group-example",
        "cluster-example",
        "owned-example-ng-cpu",
        None,
    ),
    (
        "nebius_applications_v1alpha1_k8s_release",
        "release-example",
        "project-example",
        None,
        "cluster-example",
    ),
]

CREATE_OPERATION = """
import json
from npa.provisioning_journal import ProvisioningOperation,operation_context
op=ProvisioningOperation.prepare(command="npa cluster up",project_alias="example",
project_id="project-example",tenant_id="tenant-example",region="example-region",
resource_type="cluster",requested_name="owned-example",resume_command="npa cluster up")
with operation_context(op):
 op.transition("mutating")
 op.record_failure(RuntimeError("retained synthetic failure"))
 op.transition("recovery-required")
print(json.dumps({"path":str(op.path),"journal":op.read()}))
"""

FAKE_PROVIDER = """
import json,os,sys
from pathlib import Path
root=Path(os.environ["ABSENCE_FAKE_ROOT"])
args=sys.argv[1:]
with (root/"provider-calls.jsonl").open("a") as stream:
 stream.write(json.dumps(args)+"\\n")
mode=os.environ.get("ABSENCE_FAKE_MODE", "absent")
if mode=="race":
 path=Path(os.environ["ABSENCE_FAKE_JOURNAL"])
 x=json.loads(path.read_text());x["concurrent_writer"]=True;path.write_text(json.dumps(x))
if "project" in args:
 print(json.dumps({"metadata":{"id":"project-example","parent_id":"tenant-example"},
                   "status":{"state":"ACTIVE"}}));sys.exit(0)
if "get" in args:
 if mode=="live":print("{}");sys.exit(0)
 print("rpc error: code = " + ("PermissionDenied" if mode=="denied" else
       "Unavailable" if mode=="unknown" else "NotFound"),file=sys.stderr);sys.exit(5)
if "list" in args:
 rows=[]
 if mode=="same-name" and "cluster" in args:
  rows=[{"metadata":{"id":"another-cluster","parent_id":"project-example", "name":"owned-example"}}]
 token="again" if mode=="page-loop" else "second" if "--page-token" not in args else ""
 print(json.dumps({"items":rows,"next_page_token":token}));sys.exit(0)
print("unexpected command",file=sys.stderr);sys.exit(9)
"""


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _pin(path):
    return {"path": str(path), "sha256": _hash(path.read_bytes())}


def _put(root, name, value):
    path = root / (name + ".json")
    path.write_text(json.dumps(value))
    return _pin(path)


def _authority(root):
    key = root / "synthetic-key.json"
    key.write_text('{"synthetic": "not-a-credential"}')
    config = root / "provider.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "example": {
                        "endpoint": "api.example.invalid",
                        "auth-type": "service account",
                        "parent-id": "project-example",
                        "tenant-id": "tenant-example",
                        "service-account-credentials-file-path": str(key),
                    }
                }
            }
        )
    )
    return {
        "profile": "example",
        "provider_config_path": str(config),
        "provider_config_sha256": _hash(config.read_bytes()),
        "credential_file_path": str(key),
        "credential_file_sha256": _hash(key.read_bytes()),
        "key_backed": True,
        "attached_metadata_selected": False,
        "project_id": "project-example",
        "tenant_id": "tenant-example",
        "region": "example-region",
    }


def _native_state():
    return {
        "lineage": "synthetic-lineage",
        "resources": [
            {
                "mode": "managed",
                "type": kind,
                "instances": [
                    {
                        "attributes": {
                            "id": identity,
                            "name": name,
                            "parent_id": parent,
                            "cluster_id": cluster,
                        }
                    }
                ],
            }
            for kind, identity, parent, name, cluster in NATIVE_IDENTITIES
        ],
    }


def _native_archive(root, journal):
    state = json.dumps(_native_state()).encode()
    sidecar = {k: journal[k] for k in ("project_id", "tenant_id", "region")}
    sidecar.update(backend="mk8s", cluster_name=journal["requested_name"])
    path = root / "original-native.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in (
            (".npa-fleet-env.json", json.dumps(sidecar).encode()),
            ("k8s-training/terraform.tfstate", state),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return _pin(path), _hash(state)


def _producer_files(root, journal):
    runner = root / "original-runner.py"
    runner.write_text("# inert original producer fixture\n")
    stdout, stderr = root / "stdout", root / "stderr"
    stdout.write_text("synthetic failed attempt\n")
    stderr.write_text("Provisioning operation: " + journal["operation_id"] + "\n")
    return runner, stdout, stderr


def _producer_documents(root, journal, repository, authority):
    runner, stdout, stderr = _producer_files(root, journal)
    source = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    finished = datetime.fromisoformat(
        journal["updated_at"].replace("Z", "+00:00")
    ) + timedelta(microseconds=500000)
    start = {
        "started_at": journal["created_at"],
        "source_revision": source,
        "argv": [
            "npa",
            "cluster",
            "up",
            "--project",
            "example",
            "--context",
            "owned-example",
        ],
        "profile_binding_sha256": authority["sha256"],
        "runner_sha256": _hash(runner.read_bytes()),
    }
    result = {
        "started_at": start["started_at"],
        "finished_at": finished.isoformat(),
        "exit": 130,
        "primary_invoked_process_joined": True,
        "candidate_job_submitted": False,
        "stdout_sha256": _hash(stdout.read_bytes()),
        "stderr_sha256": _hash(stderr.read_bytes()),
    }
    return {
        "producer_start": _put(root, "start", start),
        "producer_result": _put(root, "result", result),
        "producer_runner": _pin(runner),
        "stdout": _pin(stdout),
        "stderr": _pin(stderr),
    }, source


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PYTHONPATH", str(repository / "npa/src"))
    monkeypatch.delenv("NPA_PARENT_LIFECYCLE_OPERATION", raising=False)
    child = subprocess.check_output([sys.executable, "-c", CREATE_OPERATION], text=True)
    original = json.loads(child)
    journal = original["journal"]
    snapshot = tmp_path / "original-journal.json"
    snapshot.write_bytes(Path(original["path"]).read_bytes())
    authority = _put(tmp_path, "authority", _authority(tmp_path))
    docs, source = _producer_documents(tmp_path, journal, repository, authority)
    archive, state_hash = _native_archive(tmp_path, journal)
    intent = {
        "original_journal_sha256": _hash(snapshot.read_bytes()),
        "source_revision": source,
        "backend_archive_sha256": archive["sha256"],
        "original_state_sha256": state_hash,
    }
    manifest = {
        "schema_version": 1,
        "operation_id": journal["operation_id"],
        "producer_repository": str(repository),
        "original_journal": _pin(snapshot),
        "profile_binding": authority,
        "cleanup_intent": _put(tmp_path, "intent", intent),
        "backend_archive": archive,
        **docs,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    binary = tmp_path / "nebius"
    binary.write_text("#!" + sys.executable + "\n" + FAKE_PROVIDER)
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("ABSENCE_FAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("ABSENCE_FAKE_JOURNAL", original["path"])
    return path, manifest, load_operation(journal["operation_id"])


def _run(path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "npa",
            "cluster",
            "reconcile-absent",
            "--evidence-file",
            str(path),
        ],
        capture_output=True,
        text=True,
    )


def test_fresh_process_recovers_then_repeats_without_cloud_reads(recovery):
    path, manifest, operation = recovery
    first = _run(path)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["status"] == "reconciled-destroyed"
    assert operation.read()["phase"] == "destroyed"
    assert (
        json.loads(Path(manifest["original_journal"]["path"]).read_text())["phase"]
        == "recovery-required"
    )
    calls = (path.parent / "provider-calls.jsonl").read_bytes()
    assert b"--page-token" in calls and b"release-example" in calls
    second = _run(path)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["fresh_provider_verification"] is False
    assert (path.parent / "provider-calls.jsonl").read_bytes() == calls


@pytest.mark.parametrize(
    "mode", ["live", "denied", "unknown", "same-name", "page-loop", "race"]
)
def test_fresh_process_refuses_hostile_provider_or_journal(recovery, monkeypatch, mode):
    path, _manifest, operation = recovery
    monkeypatch.setenv("ABSENCE_FAKE_MODE", mode)
    result = _run(path)
    assert result.returncode != 0
    assert operation.read()["phase"] == "recovery-required"
    assert not operation.read().get("absence_recovery")


def test_execution_lock_refuses_second_process(recovery):
    path, manifest, operation = recovery
    with recovery_lease(operation, manifest["original_journal"]["sha256"]):
        result = _run(path)
    assert result.returncode != 0
    assert not (path.parent / "provider-calls.jsonl").exists()


@pytest.mark.parametrize(
    "field", ["producer_runner", "stdout", "stderr", "backend_archive"]
)
def test_changed_original_bytes_refused_before_provider(recovery, field):
    path, manifest, operation = recovery
    Path(manifest[field]["path"]).write_bytes(b"changed")
    assert _run(path).returncode != 0
    assert operation.read()["phase"] == "recovery-required"
    assert not (path.parent / "provider-calls.jsonl").exists()


def test_wrong_operation_is_not_rebound(recovery):
    _path, manifest, _operation = recovery
    candidate = copy.deepcopy(manifest)
    candidate["operation_id"] += "-r2"
    with pytest.raises(AbsenceRecoveryError, match="Wrong operation"):
        load_legacy_evidence(candidate)


def test_original_authority_scrubs_ambient_selectors(recovery, monkeypatch):
    _path, manifest, _operation = recovery
    evidence = load_legacy_evidence(manifest)
    for key in (
        "NEBIUS_ENDPOINT",
        "NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_CONFIG",
        "NPA_REUSE_IAM_TOKEN",
    ):
        monkeypatch.setenv(key, "hostile")
    argv, env = authority_environment(evidence["authority"])
    assert "--config" in argv and "--profile" in argv
    assert "hostile" not in env.values()


@pytest.mark.parametrize(
    "field,key,value",
    [
        ("producer_result", "started_at", "other-attempt"),
        ("producer_start", "source_revision", "0" * 40),
        ("producer_result", "candidate_job_submitted", True),
        ("profile_binding", "profile", "foreign-profile"),
    ],
)
def test_rehashed_new_summary_cannot_replace_original_authority(
    recovery, field, key, value
):
    path, manifest, operation = recovery
    original = Path(manifest[field]["path"])
    document = json.loads(original.read_text())
    document[key] = value
    original.write_text(json.dumps(document))
    manifest[field] = _pin(original)
    path.write_text(json.dumps(manifest))
    assert _run(path).returncode != 0
    assert operation.read()["phase"] == "recovery-required"
    assert not (path.parent / "provider-calls.jsonl").exists()


@pytest.mark.parametrize("field", ["provider_config_path", "credential_file_path"])
def test_original_authority_bytes_must_still_match(recovery, field):
    path, manifest, operation = recovery
    authority = json.loads(Path(manifest["profile_binding"]["path"]).read_text())
    Path(authority[field]).write_text("changed")
    assert _run(path).returncode != 0
    assert operation.read()["phase"] == "recovery-required"
    assert not (path.parent / "provider-calls.jsonl").exists()


def test_terminal_recovery_refuses_changed_audit(recovery):
    path, _manifest, operation = recovery
    assert _run(path).returncode == 0
    audit = operation.read()["absence_recovery"]
    Path(audit["path"]).write_text("changed")
    assert _run(path).returncode != 0


def test_compare_and_swap_rejects_changed_project_lease(recovery):
    from npa.cluster.absent_journal import complete_absence

    path, manifest, operation = recovery
    with recovery_lease(operation, manifest["original_journal"]["sha256"]) as lease:
        value = json.loads(lease["path"].read_text())
        value["operation_id"] = "foreign-operation"
        lease["path"].write_text(json.dumps(value))
        with pytest.raises(AbsenceRecoveryError, match="lease generation changed"):
            complete_absence(
                operation, manifest["original_journal"]["sha256"], {}, lease
            )
    assert operation.read()["phase"] == "recovery-required"


@pytest.mark.parametrize(
    "code,stderr",
    [
        (5, "code = PermissionDenied; wrapped code = NotFound"),
        (-15, "code = NotFound"),
        (0, "code = NotFound"),
    ],
)
def test_mixed_or_interrupted_errors_do_not_prove_absence(code, stderr):
    from npa.cluster.absent_provider import require_absent

    with pytest.raises(AbsenceRecoveryError):
        require_absent(subprocess.CompletedProcess([], code, stdout="", stderr=stderr))


@pytest.mark.parametrize("symlink", [True, False])
def test_audit_directory_never_follows_links_or_changes_existing_permissions(
    recovery, symlink
):
    path, _manifest, operation = recovery
    directory = operation.path.parent / "absence-recovery"
    foreign = path.parent / "unowned-directory"
    foreign.mkdir(mode=0o755)
    if symlink:
        directory.symlink_to(foreign, target_is_directory=True)
    else:
        directory.mkdir(mode=0o755)
    before = foreign.stat().st_mode if symlink else directory.stat().st_mode
    result = _run(path)
    assert result.returncode != 0
    assert operation.read()["phase"] == "recovery-required"
    assert (foreign.stat().st_mode if symlink else directory.stat().st_mode) == before
    assert not (path.parent / "provider-calls.jsonl").exists()
