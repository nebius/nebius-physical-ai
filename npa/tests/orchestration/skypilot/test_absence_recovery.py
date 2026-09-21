"""Exercise original producer binding, actual HTTP reads and journal race refusals."""

import base64
import copy
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import yaml

from npa.cluster.absent_evidence import AbsenceRecoveryError, digest
from npa.orchestration.skypilot import absence_evidence as evidence
from npa.orchestration.skypilot import absence_reader as reader
from npa.orchestration.skypilot import absence_recovery as recovery
from npa.orchestration.skypilot.absence_target import reader_environment
from npa.provisioning_journal import load_operation


def pin(path):
    path.chmod(0o600)
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return pin(path)


def operation_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.delenv("NPA_PARENT_LIFECYCLE_OPERATION", raising=False)
    command = """
import json
from npa.provisioning_journal import ProvisioningOperation, operation_context
op = ProvisioningOperation.prepare(command='npa workbench workflow submit',
 project_alias='example',project_id='project-example',tenant_id='tenant-example',
 region='example-region',resource_type='workflow-submit',requested_name='example-run',
 resume_command='npa workbench workflow submit')
with operation_context(op):
 op.transition('mutating')
 op.record_failure(RuntimeError("exact managed-job name 'example-run' maps to multiple immutable IDs: 1, 2, 3"))
 op.transition('recovery-required')
print(json.dumps({'path':str(op.path),'journal':op.read()}))
"""
    produced = json.loads(
        subprocess.check_output([sys.executable, "-c", command], text=True)
    )
    return load_operation(produced["journal"]["operation_id"])


def native_config():
    return {
        "allowed_clouds": ["kubernetes"],
        "kubernetes": {"allowed_contexts": ["example-context"]},
        "jobs": {
            "controller": {
                "resources": {
                    "cloud": "kubernetes",
                    "region": "example-context",
                    "cpus": 2,
                    "memory": 8,
                    "autostop": False,
                }
            }
        },
    }


def native_fixture(tmp_path, scope):
    root, user = scope["root"], scope["user"]
    config_entry = put(tmp_path / "native-config.yaml", native_config())
    wrappers = []
    for identity in (1, 2, 3):
        wrapper = {
            "run": f"echo \"export SKYPILOT_USER_ID='{user}'\"\njob_ids_array=({identity} )\n",
            "file_mounts": {
                "~/.sky/managed_jobs/job.config_yaml": config_entry["path"]
            },
        }
        wrappers.append(
            put(
                root / "home/.sky/jobs_controller" / f"example-run-{identity:04x}.yaml",
                wrapper,
            )
        )
    return {
        "native_configs": [config_entry],
        "native_wrappers": wrappers,
        "native_state": native_database(scope),
    }


def native_database(scope):
    database = scope["root"] / "sky-runtime/.sky/state.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cluster_yaml(cluster_name TEXT, yaml TEXT)")
        native = {
            "cluster_name": scope["controller"],
            "provider": {
                "type": "external",
                "module": "sky.provision.kubernetes",
                "context": "example-context",
                "namespace": "default",
                "services": [{"metadata": {"name": scope["controller"] + "-head-ssh"}}],
            },
        }
        connection.execute(
            "INSERT INTO cluster_yaml VALUES (?, ?)",
            (scope["logical"], yaml.safe_dump(native)),
        )
    return pin(database)


def target_authority(tmp_path):
    return {
        "profile": "example",
        "config": put(
            tmp_path / "config.yaml",
            {
                "profiles": {
                    "example": {
                        "auth-type": "service account",
                        "parent-id": "different-default-project",
                        "private-key-file-path": str(tmp_path / "dummy-key"),
                    }
                }
            },
        ),
        "key": put(tmp_path / "dummy-key", {"synthetic": True}),
        "binary": put(tmp_path / "nebius", {"inert": True}),
    }


def target_config(authority):
    tls = {
        "server": "https://example.invalid",
        "certificate-authority-data": base64.b64encode(b"synthetic-ca").decode(),
    }
    return {
        "contexts": [
            {
                "name": "example-context",
                "context": {"cluster": "example", "user": "example"},
            }
        ],
        "clusters": [{"name": "example", "cluster": tls}],
        "users": [
            {
                "name": "example",
                "user": {
                    "exec": {
                        "command": authority["binary"]["path"],
                        "args": [
                            "mk8s",
                            "v1",
                            "cluster",
                            "get-token",
                            "--profile",
                            "example",
                            "--format",
                            "json",
                        ],
                    }
                },
            }
        ],
    }


def target_fixture(tmp_path, context_path):
    authority = target_authority(tmp_path)
    config = target_config(authority)
    return {
        "authority": authority,
        "original_kubeconfig": put(context_path, config),
        "reader_kubeconfig": put(tmp_path / "reader-kubeconfig", config),
        "cluster_registration": put(
            context_path.parent / "cluster.json",
            {
                "name": "example-context",
                "cluster_id": "cluster-example",
                "project_id": "project-example",
                "endpoint": config["clusters"][0]["cluster"]["server"],
            },
        ),
    }


def trace_fixture(tmp_path, operation, scope):
    journal = operation.read()
    config_dir = tmp_path / "config"
    kubeconfig = config_dir / "clusters/example-context/kubeconfig"
    response_path = tmp_path / "original-response.json"
    controller = {"name": scope["logical"], "selected_context": "example-context"}
    launch = {
        "state": "indeterminate",
        "reconciliation_error": journal["last_error"],
        "controller": controller,
    }
    response = put(response_path, {"status": "failed", "launch_transaction": launch})
    profile = task_profile()
    ledger = {
        "schema_version": "npa.workflow.submission.v1",
        "project": "example",
        "run_id": "example-run",
        "launch": launch,
        "workflow": {"steps": [{"resources_profile": profile}]},
    }
    ledger_entry = put(
        config_dir / "workflow-submissions/example/example-run.json", ledger
    )
    command = f"L={tmp_path} && ISO=$L/isolated && K={kubeconfig} && env NPA_CONFIG_DIR={config_dir} KUBECONFIG=$K npa/.venv/bin/python -m npa.cli.main workbench workflow submit /tmp/example.yaml --project example --infra k8s/example-context --isolated-config-dir $ISO --controller-backend kubernetes --resume-run example-run > {response_path}"
    return {
        "producer_transcript": put(
            tmp_path / "original-transcript.jsonl", trace_record(command, journal)
        ),
        "producer_line": 1,
        "producer_response": response,
        "submission_ledger": ledger_entry,
    }, kubeconfig


def task_profile():
    return {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": "12Gi",
        "kubernetes": {
            "pod_config": {"spec": {"imagePullSecrets": [{"name": "example-pull"}]}}
        },
    }


def trace_record(command, journal):
    timestamp = datetime.fromisoformat(
        journal["updated_at"].replace("Z", "+00:00")
    ).timestamp()
    return {
        "type": "tool_call",
        "subtype": "completed",
        "timestamp_ms": int(timestamp * 1000),
        "tool_call": {
            "shellToolCall": {
                "args": {"command": command},
                "result": {"success": {"command": command}},
            }
        },
    }


@pytest.fixture
def sample(tmp_path, monkeypatch):
    operation = operation_fixture(tmp_path, monkeypatch)
    root = tmp_path / "isolated"
    user = "npa-" + digest(str(root).encode())[:12]
    scope = {
        "root": root,
        "user": user,
        "logical": "sky-jobs-controller-" + user,
        "controller": "controller-" + user,
    }
    trace, kubeconfig = trace_fixture(tmp_path, operation, scope)
    original_journal = tmp_path / "original-journal.json"
    original_journal.write_bytes(operation.path.read_bytes())
    manifest = {
        "schema": "npa.sky.absence.v1",
        "operation_id": operation.operation_id,
        "original_journal": pin(original_journal),
        "isolated_root": str(root),
        "sky_version": "0.12.2",
        "naming_source": {},
        **trace,
        **native_fixture(tmp_path, scope),
        **target_fixture(tmp_path, kubeconfig),
    }
    # Separate tests exercise the fixed source digest predicate; synthetic native
    # state here intentionally avoids vendoring the upstream package.
    monkeypatch.setattr(evidence, "_naming_source", lambda _: None)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, operation


def test_original_shape_binds_three_jobs_without_invented_uid(sample):
    _, manifest, _ = sample
    result = evidence.load_evidence(manifest)
    assert result["scope"]["managed_ids"] == [1, 2, 3]
    assert result["historical_workload_outcome"] == "unknown"
    assert "uid" not in result["scope"]["controller"]


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "missing-job",
        "duplicate-job",
        "extra-job",
        "config",
        "response",
        "foreign-project",
        "profile-override",
        "target",
        "summary",
    ],
)
def test_incomplete_or_foreign_producer_refuses_without_writes(sample, change):
    _, original, operation = sample
    manifest = copy.deepcopy(original)
    before = operation.path.read_bytes()
    if change == "root":
        manifest["isolated_root"] += "-different"
    elif change == "missing-job":
        manifest["native_wrappers"].pop()
    elif change == "duplicate-job":
        manifest["native_wrappers"].append(manifest["native_wrappers"][0])
    elif change == "extra-job":
        entry = manifest["native_wrappers"][0]
        value = json.loads(Path(entry["path"]).read_text())
        value["run"] = value["run"].replace("(1 )", "(9 )")
        manifest["native_wrappers"][0] = put(Path(entry["path"]), value)
    elif change == "config":
        manifest["native_configs"] = []
    elif change == "response":
        manifest["producer_response"]["path"] += "-invented"
    elif change in {"foreign-project", "profile-override"}:
        entry = manifest["submission_ledger"]
        value = json.loads(Path(entry["path"]).read_text())
        if change == "foreign-project":
            value["project"] = "foreign"
        else:
            value["workflow"]["steps"][0]["resources_profile"]["kubernetes"][
                "custom_metadata"
            ] = {"name": "hidden"}
        manifest["submission_ledger"] = put(Path(entry["path"]), value)
    elif change == "target":
        manifest["reader_kubeconfig"]["sha256"] = "0" * 64
    else:
        manifest["producer_transcript"] = put(
            Path(manifest["producer_transcript"]["path"]), {"claimed_owner": True}
        )
    with pytest.raises((ValueError, KeyError, OSError)):
        evidence.load_evidence(manifest)
    assert operation.path.read_bytes() == before


def test_existing_reader_default_project_is_not_compute_ownership(sample, monkeypatch):
    _, manifest, _ = sample
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/unreadable-metadata-token")
    monkeypatch.setenv("NPA_NEBIUS_IAM_TOKEN", "synthetic-disallowed")
    environment = reader_environment(manifest["authority"])
    assert "NEBIUS_IAM_TOKEN_FILE" not in environment
    assert "NPA_NEBIUS_IAM_TOKEN" not in environment
    assert environment["NEBIUS_PROFILE"] == "example"


def test_pinned_other_file_cannot_select_unpinned_default_config(sample):
    _, manifest, _ = sample
    authority = manifest["authority"]
    default = Path(authority["config"]["path"])
    other = default.with_name("other.yaml")
    other.write_bytes(default.read_bytes())
    authority["config"] = pin(other)
    default.write_text("profiles:\n  example:\n    auth-type: metadata\n")
    with pytest.raises(ValueError, match="config.yaml basename"):
        reader_environment(authority)
    with pytest.raises(ValueError, match="config.yaml basename"):
        evidence.load_evidence(manifest)


@pytest.mark.parametrize(
    "change", ["remedy", "state", "controller", "empty-endpoint", "foreign-endpoint"]
)
def test_real_original_variants_preserve_structured_identity_checks(sample, change):
    _, manifest, _ = sample
    key = "cluster_registration" if "endpoint" in change else "submission_ledger"
    entry = manifest[key]
    value = json.loads(Path(entry["path"]).read_bytes())
    if change == "remedy":
        value["launch"]["operator_remedy"] = (
            "Retained ledger guidance differs from CLI prose"
        )
    elif change == "state":
        value["launch"]["state"] = "committed"
    elif change == "controller":
        value["launch"]["controller"]["name"] = "foreign-controller"
    else:
        value["endpoint"] = (
            "" if change == "empty-endpoint" else "https://foreign.invalid"
        )
    manifest[key] = put(Path(entry["path"]), value)
    if change in {"remedy", "empty-endpoint"}:
        assert (
            evidence.load_evidence(manifest)["historical_workload_outcome"] == "unknown"
        )
    else:
        with pytest.raises(ValueError):
            evidence.load_evidence(manifest)


class HttpMetadata:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def call_api(self, path, method, **kwargs):
        self.calls.append((path, method, kwargs.get("query_params")))
        status, payload = self.pages.pop(0)
        return SimpleNamespace(
            status=status, data=json.dumps(payload).encode(), release_conn=lambda: None
        )


def page(items=(), token=""):
    return {
        "kind": "PartialObjectMetadataList",
        "metadata": {"continue": token, "resourceVersion": "123"},
        "items": list(items),
    }


@pytest.mark.parametrize("kind", ["pods", "services"])
@pytest.mark.parametrize(
    "mode",
    [
        "live",
        "terminating",
        "recreated",
        "label",
        "annotation",
        "denied",
        "partial",
        "loop",
    ],
)
def test_live_ambiguous_and_incomplete_reads_refuse(sample, tmp_path, kind, mode):
    _, manifest, operation = sample
    scope = evidence.load_evidence(manifest)["scope"]
    before = operation.path.read_bytes()
    metadata = {
        "name": "task-" + scope["user"] + "-head",
        "namespace": "default",
        "uid": "new-uid",
    }
    if mode == "terminating":
        metadata["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    if mode in {"label", "annotation"}:
        metadata["name"] = "foreign-name"
        metadata[mode + "s"] = {"identity": scope["user"]}
    pages = [(200, page([{"metadata": metadata}]))]
    if mode == "denied":
        pages = [(403, {"kind": "Status", "reason": "Forbidden"})]
    elif mode == "partial":
        pages = [(200, {"kind": "PartialObjectMetadataList", "metadata": {}})]
    elif mode == "loop":
        pages = [(200, page(token="same")), (200, page(token="same"))]
    with pytest.raises(ValueError):
        reader._inventory(HttpMetadata(pages), kind, scope, tmp_path, [])
    assert operation.path.read_bytes() == before


@contextmanager
def metadata_server(foreign):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            payload = (
                page([foreign], "")
                if "continue=" in self.path
                else page(token="second")
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port, calls
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_kubernetes_http_client_reads_all_pages_and_preserves_foreign(
    sample, tmp_path
):
    from kubernetes.client import ApiClient, Configuration

    scope = evidence.load_evidence(sample[1])["scope"]
    foreign = {
        "metadata": {
            "name": "foreign-controller",
            "namespace": "default",
            "uid": "foreign-original",
        }
    }
    with metadata_server(foreign) as (port, calls):
        config = Configuration(host=f"http://127.0.0.1:{port}")
        records = []
        with ApiClient(config) as client:
            reader._inventory(client, "pods", scope, tmp_path, records)
        assert calls == ["/api/v1/pods", "/api/v1/pods?continue=second"]
        assert len(records) == 2
        assert all(record["method"] == "GET" for record in records)


def test_new_naming_source_bytes_cannot_extend_the_contract():
    with pytest.raises(AbsenceRecoveryError, match="Incomplete naming"):
        evidence._naming_source({"sky_version": "0.12.2", "naming_source": {}})


def retained_reads(_path, _manifest, output):
    records = []
    for index in range(4):
        path = output / f"response-{index}.json"
        path.write_text('{"synthetic_unit_reader": true}')
        records.append({"file": path.name, "sha256": digest(path.read_bytes())})
    return records


def lease_path(operation):
    from npa.provisioning_journal import _project_lease_directory

    return _project_lease_directory(operation.read()["project_id"]) / "lease.json"


def test_preview_retains_lease_then_apply_preserves_failure_and_repeats(
    sample, monkeypatch
):
    path, _, operation = sample
    before = operation.path.read_bytes()
    lease = lease_path(operation)
    lease_before = lease.read_bytes()
    monkeypatch.setattr(recovery, "_read", retained_reads)
    preview = recovery.reconcile_absent(path)
    assert preview["status"] == "absence-verified-not-applied"
    assert operation.path.read_bytes() == before
    assert lease.read_bytes() == lease_before
    applied = recovery.reconcile_absent(path, apply=True)
    assert applied["status"] == "reconciled-absent"
    current = operation.read()
    assert current["last_error"] == json.loads(before)["last_error"]
    assert current["events"][:-1] == json.loads(before)["events"]
    audit = json.loads(Path(current["absence_recovery"]["path"]).read_text())
    assert audit["historical_workload_outcome"] == "unknown"
    assert (
        Path(current["absence_recovery"]["path"])
        .parent.joinpath("original-journal.json")
        .read_bytes()
        == before
    )
    monkeypatch.setattr(
        recovery, "_read", lambda *_: pytest.fail("repeat must not invent new reads")
    )
    assert recovery.reconcile_absent(path, apply=True)["status"] == "already-reconciled"


@pytest.mark.parametrize("change", ["journal", "lease", "manifest", "evidence"])
def test_concurrent_change_prevents_terminal_cas(sample, monkeypatch, change):
    path, manifest, operation = sample
    before, lease = operation.path.read_bytes(), lease_path(operation)
    lease_before = lease.read_bytes()
    reads = []

    def raced_read(*args):
        reads.append(True)
        records = retained_reads(*args)
        target = {
            "journal": operation.path,
            "lease": lease,
            "manifest": path,
            "evidence": Path(manifest["submission_ledger"]["path"]),
        }[change]
        value = json.loads(target.read_text())
        value["concurrent_change"] = True
        target.write_text(json.dumps(value))
        return records

    monkeypatch.setattr(recovery, "_read", raced_read)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.reconcile_absent(path, apply=True)
    assert reads == [True]
    assert "absence_recovery" not in operation.read()
    if change != "journal":
        assert operation.path.read_bytes() == before
    if change != "lease":
        assert lease.read_bytes() == lease_before


@pytest.mark.parametrize("kind", ["live-owner", "foreign-lease"])
def test_owner_or_foreign_lease_refuses_before_reader(sample, monkeypatch, kind):
    path, manifest, operation = sample
    if kind == "live-owner":
        journal = operation.read()
        journal["owner_pid"] = os.getpid()
        operation.path.write_text(json.dumps(journal))
        manifest["original_journal"] = put(
            Path(manifest["original_journal"]["path"]), journal
        )
        path.write_text(json.dumps(manifest))
    else:
        lease = lease_path(operation)
        payload = json.loads(lease.read_text())
        payload["operation_id"] = "foreign-operation"
        lease.write_text(json.dumps(payload))
    monkeypatch.setattr(
        recovery, "_read", lambda *_: pytest.fail("reader must not run")
    )
    before = operation.path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        recovery.reconcile_absent(path, apply=True)
    assert operation.path.read_bytes() == before


@pytest.mark.parametrize(
    "status,reason",
    [(200, ""), (403, "Forbidden"), (404, "Unauthorized"), (500, "Unknown")],
)
def test_exact_get_requires_actual_named_notfound(sample, tmp_path, status, reason):
    scope = evidence.load_evidence(sample[1])["scope"]
    payload = {
        "kind": "Status",
        "reason": reason,
        "details": {"name": scope["controller"]["name"] + "-head"},
    }
    with pytest.raises(ValueError):
        reader._exact_controller(HttpMetadata([(status, payload)]), scope, tmp_path, [])


def test_wal_is_required_and_original_bytes_are_unchanged(sample):
    from npa.orchestration.skypilot.absence_native import _controller_rows

    _, manifest, _ = sample
    path = Path(manifest["native_state"]["path"])
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE original_extra(value TEXT)")
        writer.commit()
        manifest["native_state"] = pin(path)
        wal = path.with_name(path.name + "-wal")
        before = {path: path.read_bytes(), wal: wal.read_bytes()}
        with pytest.raises(ValueError, match="WAL is unbound"):
            _controller_rows(manifest, "missing")
        manifest["native_wal"] = pin(wal)
        assert _controller_rows(manifest, "missing") == []
        assert all(item.read_bytes() == content for item, content in before.items())


@pytest.mark.parametrize("tamper", [False, True])
def test_crash_after_journal_cas_requires_retained_raw_bodies(
    sample, monkeypatch, tamper
):
    from npa.cluster import absent_journal

    path, _, operation = sample
    lease = lease_path(operation)
    lease_before = lease.read_bytes()
    monkeypatch.setattr(recovery, "_read", retained_reads)
    release = absent_journal._release_bound_lease
    monkeypatch.setattr(
        absent_journal,
        "_release_bound_lease",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic write failure")),
    )
    with pytest.raises(OSError):
        recovery.reconcile_absent(path, apply=True)
    assert operation.read()["phase"] == "destroyed"
    assert lease.read_bytes() == lease_before
    monkeypatch.setattr(absent_journal, "_release_bound_lease", release)
    receipt = operation.read()["absence_recovery"]
    raw = Path(receipt["path"]).parent / "response-0.json"
    if tamper:
        raw.write_text("modified after crash")
        with pytest.raises(ValueError, match="Reader audit changed"):
            recovery.reconcile_absent(path, apply=True)
        assert lease.read_bytes() == lease_before
    else:
        assert (
            recovery.reconcile_absent(path, apply=True)["status"]
            == "already-reconciled"
        )
        assert json.loads(lease.read_text())["phase"] == "destroyed"


@pytest.mark.parametrize("change", ["none", "wrong-name"])
def test_exact_named_get_positive_and_wrong_resource(sample, tmp_path, change):
    scope = evidence.load_evidence(sample[1])["scope"]
    controller = scope["controller"]
    names = [controller["name"] + "-head", *controller["services"]]
    pages = [
        (404, {"kind": "Status", "reason": "NotFound", "details": {"name": name}})
        for name in names
    ]
    client = HttpMetadata(pages)
    if change == "wrong-name":
        pages[0][1]["details"]["name"] = "different-resource"
        with pytest.raises(ValueError):
            reader._exact_controller(client, scope, tmp_path, [])
    else:
        reader._exact_controller(client, scope, tmp_path, [])
        assert len(client.calls) == len(names)
        assert all(call[1] == "GET" for call in client.calls)


@pytest.mark.parametrize(
    "change", ["revision", "missing-revision", "remaining", "nonstring-token"]
)
def test_inventory_snapshot_and_completion_are_required(sample, tmp_path, change):
    scope = evidence.load_evidence(sample[1])["scope"]
    final = page()
    if change == "revision":
        final["metadata"]["resourceVersion"] = "124"
    elif change == "missing-revision":
        final["metadata"].pop("resourceVersion")
    elif change == "remaining":
        final["metadata"]["remainingItemCount"] = 1
    else:
        final["metadata"]["continue"] = 1
    client = HttpMetadata([(200, page(token="next")), (200, final)])
    with pytest.raises(ValueError):
        reader._inventory(client, "pods", scope, tmp_path, [])


def test_provider_project_endpoint_and_ca_are_actual_bound_values(sample):
    bound = evidence.load_evidence(sample[1])
    valid = {
        "metadata": {"id": "cluster-example", "parent_id": "project-example"},
        "status": {
            "state": "RUNNING",
            "control_plane": {
                "endpoints": {"public_endpoint": "https://example.invalid"},
                "auth": {"cluster_ca_certificate": "synthetic-ca"},
            },
        },
    }
    reader._provider_identity(valid, bound)
    for selector in ("project", "endpoint", "ca", "status"):
        candidate = copy.deepcopy(valid)
        if selector == "project":
            candidate["metadata"]["parent_id"] = "different-default-project"
        elif selector == "endpoint":
            candidate["status"]["control_plane"]["endpoints"]["public_endpoint"] += (
                "-other"
            )
        elif selector == "ca":
            candidate["status"]["control_plane"]["auth"]["cluster_ca_certificate"] += (
                "-other"
            )
        else:
            candidate["status"]["state"] = "DELETING"
        with pytest.raises(ValueError):
            reader._provider_identity(candidate, bound)
