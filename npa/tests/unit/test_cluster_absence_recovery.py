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
                   "status":{"container_state":"ACTIVE","suspension_state":"NONE","region":"example-region"}}));sys.exit(0)
if "get" in args:
 if mode=="live":print("{}");sys.exit(0)
 print("rpc error: code = " + ("PermissionDenied" if mode=="denied" else
       "Unavailable" if mode=="unknown" else "NotFound"),file=sys.stderr);sys.exit(5)
if "list" in args:
 if "k8s-release" in args:
  if "--cluster-id" not in args or args[args.index("--cluster-id")+1] != "cluster-example":
   print("code = InvalidArgument",file=sys.stderr);sys.exit(3)
  if mode=="application-live":
   print(json.dumps({"items":[{"unexpected":"live or unbound row"}]}));sys.exit(0)
 if mode=="empty-protobuf":print("{}");sys.exit(0)
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


def _crash_after_terminal_journal(path, mode="crash"):
    code = """
import os,sys
from pathlib import Path
from npa import provisioning_journal as journals
from npa.cluster.reconcile_absent import reconcile_absent
original=journals._write_atomic
def crash(path,payload):
 if sys.argv[2]=='write-error' and path.name=='lease.json' and payload.get('phase')=='destroyed':
  raise OSError('synthetic durable lease write failure')
 original(path,payload)
 if sys.argv[2]=='crash' and path.name=='journal.json' and payload.get('phase')=='destroyed':
  os._exit(73)
journals._write_atomic=crash
reconcile_absent(Path(sys.argv[1]))
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(path), mode], capture_output=True, text=True
    )


@pytest.mark.parametrize("mode", ["crash", "write-error"])
def test_fresh_process_finishes_lease_after_terminal_journal_crash(recovery, mode):
    from npa import provisioning_journal as journals

    path, _manifest, operation = recovery
    crash = _crash_after_terminal_journal(path, mode)
    assert crash.returncode == (73 if mode == "crash" else 1), crash.stderr
    assert operation.read()["phase"] == "destroyed"
    lease = journals._project_lease_directory("project-example") / "lease.json"
    assert json.loads(lease.read_text())["phase"] != "destroyed"
    calls = (path.parent / "provider-calls.jsonl").read_bytes()
    retried = _run(path)
    assert retried.returncode == 0, retried.stderr
    released = json.loads(lease.read_text())
    assert released["phase"] == "destroyed"
    assert released["released_at"] == operation.read()["updated_at"]
    assert (path.parent / "provider-calls.jsonl").read_bytes() == calls
    assert _run(path).returncode == 0


@pytest.mark.parametrize("change", ["foreign", "raced", "released", "audit", "nested"])
def test_terminal_retry_refuses_changed_lease_or_original_binding(
    recovery, monkeypatch, change
):
    from npa import provisioning_journal as journals

    path, _manifest, operation = recovery
    assert _crash_after_terminal_journal(path).returncode == 73
    lease = journals._project_lease_directory("project-example") / "lease.json"
    payload = json.loads(lease.read_text())
    if change == "foreign":
        payload["operation_id"] = "foreign-operation"
    elif change == "raced":
        payload["owner_pid"] = os.getpid()
    elif change == "released":
        payload.update(phase="destroyed", released_at="wrong-generation")
    elif change == "audit":
        receipt = operation.read()["absence_recovery"]
        (Path(receipt["path"]).parent / "original-project-lease.json").write_text(
            "changed"
        )
    else:
        monkeypatch.setenv("NPA_PARENT_LIFECYCLE_OPERATION", operation.operation_id)
    if change in {"foreign", "raced", "released"}:
        lease.write_text(json.dumps(payload))
    before = lease.read_bytes()
    calls = (path.parent / "provider-calls.jsonl").read_bytes()
    assert _run(path).returncode != 0
    assert lease.read_bytes() == before
    assert (path.parent / "provider-calls.jsonl").read_bytes() == calls


@pytest.mark.parametrize("lock_name", ["project", "execution"])
def test_terminal_retry_refuses_active_lifecycle_lock(recovery, lock_name):
    from npa import provisioning_journal as journals
    from npa.cluster.absent_journal import _exclusive_existing

    path, _manifest, operation = recovery
    assert _crash_after_terminal_journal(path).returncode == 73
    project = journals._project_lease_directory("project-example")
    lock = (
        project / ".lock"
        if lock_name == "project"
        else operation.path.parent / ".execution.lock"
    )
    before = (project / "lease.json").read_bytes()
    with _exclusive_existing(lock):
        assert _run(path).returncode != 0
    assert (project / "lease.json").read_bytes() == before
    assert _run(path).returncode == 0


@pytest.mark.parametrize(
    "status",
    [
        {},
        {"state": "ACTIVE"},
        {
            "container_state": "DELETING",
            "suspension_state": "NONE",
            "region": "example-region",
        },
        {
            "container_state": "ACTIVE",
            "suspension_state": "SUSPENDED",
            "region": "example-region",
        },
        {
            "container_state": "ACTIVE",
            "suspension_state": "NONE",
            "region": "foreign-region",
        },
        {"container_state": "ACTIVE", "region": "example-region"},
    ],
)
def test_actual_project_status_contract_rejects_unknown_or_wrong_scope(
    recovery, monkeypatch, status
):
    from npa.cluster.absent_provider import AbsenceProvider

    path, manifest, _operation = recovery
    evidence = load_legacy_evidence(manifest)
    provider = AbsenceProvider(evidence["authority"], path.parent)
    response = {
        "metadata": {"id": "project-example", "parent_id": "tenant-example"},
        "status": status,
    }
    monkeypatch.setattr(
        provider,
        "query",
        lambda args: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(response), stderr=""
        ),
    )
    with pytest.raises(AbsenceRecoveryError, match="Wrong project authority"):
        provider.project()


@pytest.mark.parametrize("mode", ["absent", "empty-protobuf"])
def test_application_inventory_binds_cluster_on_every_page(recovery, monkeypatch, mode):
    path, _manifest, operation = recovery
    monkeypatch.setenv("ABSENCE_FAKE_MODE", mode)
    result = _run(path)
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in (path.parent / "provider-calls.jsonl").read_text().splitlines()
    ]
    pages = [args for args in calls if "k8s-release" in args and "list" in args]
    assert len(pages) == (2 if mode == "absent" else 1)
    for args in pages:
        assert args[args.index("--cluster-id") + 1] == "cluster-example"
        assert args[args.index("--parent-id") + 1] == "project-example"
    assert operation.read()["phase"] == "destroyed"


def test_any_filtered_application_row_refuses_terminal_transition(
    recovery, monkeypatch
):
    path, _manifest, operation = recovery
    monkeypatch.setenv("ABSENCE_FAKE_MODE", "application-live")
    result = _run(path)
    assert result.returncode != 0
    assert operation.read()["phase"] == "recovery-required"


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "not an empty list response"},
        {"items": None},
        {"items": {}},
        {"next_page_token": None},
        {"next_page_token": 7},
    ],
)
def test_unknown_or_malformed_inventory_response_is_not_absence(
    recovery, monkeypatch, payload
):
    from npa.cluster.absent_provider import AbsenceProvider

    path, manifest, _operation = recovery
    evidence = load_legacy_evidence(manifest)
    provider = AbsenceProvider(evidence["authority"], path.parent)
    monkeypatch.setattr(
        provider,
        "query",
        lambda args: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(payload), stderr=""
        ),
    )
    with pytest.raises(AbsenceRecoveryError):
        provider.inventory(
            ["applications", "v1alpha1", "k8s-release"],
            "project-example",
            extra_args=("--cluster-id", "cluster-example"),
        )
