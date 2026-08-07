from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import status as status_mod
from npa.cluster.api import ClusterInfo, NodeGroupInfo
from npa.cluster.state import ClusterState


runner = CliRunner()


def _state() -> ClusterState:
    return ClusterState(
        name="cluster-a",
        cluster_id="mk8scluster-a",
        project_id="project-a",
        region="eu-north1",
        node_count=1,
        node_platform="cpu-e2",
        node_preset="2vcpu-8gb",
        k8s_version="1.33",
        subnet_id="vpcsubnet-a",
        created_at="2026-05-14T21:46:00Z",
        last_seen_state="PROVISIONING",
        last_seen_at="2026-05-14T21:50:00Z",
        kubeconfig_path="/etc/hosts",
    )


def test_status_for_named_local_cluster(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name="cluster-a",
                project_id=project_id,
                status="READY",
                created_at="2026-05-14T21:46:00Z",
                endpoint="https://api.example.invalid",
            )

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="mk8snodegroup-a",
                    name="cluster-a-cpu",
                    cluster_id=cluster_id,
                    status="READY",
                    node_count=1,
                )
            ]

    saved: list[ClusterState] = []
    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", saved.append)

    result = runner.invoke(app, ["status", "--name", "cluster-a"])

    assert result.exit_code == 0
    assert "cluster-a" in result.output
    assert "READY" in result.output
    assert saved[0].last_seen_state == "READY"


def test_list_json_merges_remote_and_local(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def list_clusters(self, project_id):
            return [
                ClusterInfo(
                    id="mk8scluster-a",
                    name="cluster-a",
                    project_id=project_id,
                    status="READY",
                    created_at="2026-05-14T21:46:00Z",
                )
            ]

        def get_cluster(self, name, *, project_id=""):
            return self.list_clusters(project_id)[0]

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="mk8snodegroup-a",
                    name="cluster-a-cpu",
                    cluster_id=cluster_id,
                    status="READY",
                    node_count=1,
                )
            ]

    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-a")
    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["name"] == "cluster-a"
    assert payload[0]["state"] == "READY"
    assert payload[0]["node_count"] == 1
    assert payload[0]["region"] == "eu-north1"


def test_remote_only_cluster_uses_inventory_region_then_explicit_unknown(
    monkeypatch,
) -> None:
    remote = ClusterInfo(
        id="mk8scluster-remote",
        name="remote",
        project_id="project-a",
        status="READY",
        raw={"spec": {"region_id": "me-west1"}},
    )

    class FakeClient:
        def list_node_groups(self, _cluster_id):  # noqa: ANN201
            return []

    row = status_mod._row_for_cluster(FakeClient(), "remote", None, remote)
    unknown = status_mod._row_for_cluster(
        FakeClient(),
        "unknown",
        None,
        ClusterInfo(id="id", name="unknown", project_id="project-a", raw={}),
    )

    assert row["region"] == "me-west1"
    assert row["region_source"] == "provider_inventory"
    assert unknown["region"] is None
    assert unknown["region_source"] == "unknown"


def test_terraform_status_never_invents_default_region(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        status_mod,
        "terraform_status",
        lambda path: {
            "kube_cluster": {"value": {"id": "cluster-a", "name": "cluster-a"}}
        },
    )
    monkeypatch.setattr(status_mod, "_read_tfvars", lambda path: {})

    row = status_mod._terraform_row(tmp_path)

    assert row is not None
    assert row["region"] is None
    assert row["region_source"] == "unknown"


def test_terraform_merge_never_erases_authoritative_provider_region() -> None:
    merged = status_mod._merge_terraform_row(
        {
            "name": "cluster-a",
            "region": "me-west1",
            "region_source": "provider_inventory",
        },
        {"name": "cluster-a", "region": None, "region_source": "unknown"},
    )

    assert merged["region"] == "me-west1"
    assert merged["region_source"] == "provider_inventory"


def test_list_resolves_project_alias_like_up_and_down(monkeypatch) -> None:
    """`cluster status`/`list` accept `--project <alias>` (not just `--project-id`).

    Regression: the audit flagged `npa cluster status --project` as rejected while
    `npa cluster up`/`down` accept the alias.
    """
    from types import SimpleNamespace

    import npa.clients.config as config_mod

    seen: dict[str, str] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def list_clusters(self, project_id):
            seen["project_id"] = project_id
            return []

        def get_cluster(self, name, *, project_id=""):  # pragma: no cover - not hit
            raise AssertionError("get_cluster should not be called here")

        def list_node_groups(self, cluster_id):  # pragma: no cover - no clusters
            return []

    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [])
    monkeypatch.setattr(
        config_mod,
        "resolve_environment",
        lambda project=None: SimpleNamespace(
            project_id="project-from-alias" if project == "test-rtx" else ""
        ),
    )

    result = runner.invoke(app, ["list", "--project", "test-rtx", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert seen["project_id"] == "project-from-alias"
    # And `status --project` parses the same option (no "No such option").
    status_result = runner.invoke(
        app, ["status", "--project", "test-rtx", "--format", "json"]
    )
    assert status_result.exit_code == 0, status_result.output


def test_project_scoped_status_never_uses_an_unrelated_local_cluster(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import npa.clients.config as config_mod

    unrelated = _state()
    unrelated.project_id = "project-other"

    class ExactProjectClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def list_clusters(self, project_id):
            assert project_id == "project-selected"
            return []

        def get_cluster(self, name, *, project_id=""):
            assert name == unrelated.name
            assert project_id == "project-selected"
            raise status_mod.ClusterNotFoundError("absent from selected project")

    monkeypatch.setattr(status_mod, "MK8sClient", ExactProjectClient)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [unrelated])
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda _name: unrelated)
    monkeypatch.setattr(
        config_mod,
        "resolve_environment",
        lambda _project=None: SimpleNamespace(project_id="project-selected", region="r"),
    )

    listed = runner.invoke(
        app, ["list", "--project", "selected", "--format", "json"]
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output) == []

    named = runner.invoke(
        app,
        ["status", "--name", unrelated.name, "--project", "selected", "--format", "json"],
    )
    assert named.exit_code == 0, named.output
    assert json.loads(named.output)[0]["project_id"] != "project-other"


def test_missing_project_alias_never_falls_back_to_default(monkeypatch) -> None:
    from types import SimpleNamespace

    import npa.clients.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "resolve_environment",
        lambda project=None: (
            SimpleNamespace(project_id="default-project", region="r")
            if project is None
            else None
        ),
    )

    result = runner.invoke(app, ["status", "--project", "missing"])

    assert result.exit_code == 1
    assert "refusing to fall back" in result.output


def test_status_json_includes_terraform_outputs(monkeypatch, tmp_path) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def list_clusters(self, project_id):
            return []

        def get_cluster(self, name, *, project_id=""):
            raise status_mod.ClusterNotFoundError("not found")

        def list_node_groups(self, cluster_id):
            return []

    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [])
    monkeypatch.setattr(
        status_mod,
        "terraform_status",
        lambda terraform_dir: {
            "kube_cluster": {
                "value": {
                    "id": "mk8scluster-tf",
                    "name": "cluster-tf",
                    "endpoints": {"public_endpoint": "https://cluster.example"},
                }
            },
            "k8s_training_ref": {"value": "main-v2026-05-25"},
            "shared_filesystem": {"value": {"id": "computefilesystem-a"}},
            "filesystem_csi": {
                "value": {
                    "status": "deployed",
                    "storage_class_name": "csi-mounted-fs-path-sc",
                }
            },
        },
    )

    result = runner.invoke(
        app,
        ["status", "--terraform-dir", str(tmp_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["name"] == "cluster-tf"
    assert payload[0]["cluster_id"] == "mk8scluster-tf"
    assert payload[0]["k8s_training_ref"] == "main-v2026-05-25"
    assert payload[0]["filesystem_csi_storage_class"] == "csi-mounted-fs-path-sc"


def test_status_surfaces_a_node_group_that_never_came_up(monkeypatch) -> None:
    """A cluster is RUNNING as soon as its control plane is.

    Regression: a GPU node group the platform refused (no quota, no capacity) was
    invisible — the row said RUNNING with a quietly smaller node count.
    """

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name="cluster-a",
                project_id=project_id,
                status="RUNNING",
                created_at="2026-05-14T21:46:00Z",
                endpoint="https://api.example.invalid",
            )

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="mk8snodegroup-cpu",
                    name="cluster-a-cpu",
                    cluster_id=cluster_id,
                    status="RUNNING",
                    node_count=1,
                    platform="cpu-d3",
                    preset="4vcpu-16gb",
                ),
                NodeGroupInfo(
                    id="mk8snodegroup-gpu",
                    name="cluster-a-gpu-0",
                    cluster_id=cluster_id,
                    status="PROVISIONING",
                    node_count=0,
                    platform="gpu-rtx6000",
                    preset="1gpu-24vcpu-218gb",
                ),
            ]

    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["status", "--name", "cluster-a"])

    assert result.exit_code == 3, result.output
    assert "not RUNNING" in result.output
    assert "cluster-a-gpu-0: PROVISIONING" in result.output
    assert "gpu-rtx6000" in result.output
    assert "npa cluster down --force" in result.output
    # The healthy group is not listed as a problem.
    assert "cluster-a-cpu: RUNNING" not in result.output

    payload = json.loads(
        runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"]).output
    )
    groups = {group["name"]: group["state"] for group in payload[0]["node_groups"]}
    assert groups == {"cluster-a-cpu": "RUNNING", "cluster-a-gpu-0": "PROVISIONING"}


def test_status_stays_quiet_when_every_node_group_is_running(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name="cluster-a",
                project_id=project_id,
                status="RUNNING",
            )

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="mk8snodegroup-cpu",
                    name="cluster-a-cpu",
                    cluster_id=cluster_id,
                    status="RUNNING",
                    node_count=1,
                )
            ]

    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["status", "--name", "cluster-a"])

    assert result.exit_code == 0, result.output
    assert "not RUNNING" not in result.output


def test_provider_running_with_zero_workers_and_no_kubeconfig_is_partial(
    monkeypatch,
) -> None:
    class ControlPlaneOnly:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name=name,
                project_id=project_id,
                status="RUNNING",
            )

        def list_node_groups(self, cluster_id):
            return []

    local = _state()
    local.kubeconfig_path = ""
    local.node_count = 0
    monkeypatch.setattr(status_mod, "MK8sClient", ControlPlaneOnly)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: local)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [local])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"])

    assert result.exit_code == 3
    row = json.loads(result.output)[0]
    assert row["provider_state"] == "RUNNING"
    assert row["state"] == "PARTIAL"
    assert row["node_count"] == 0
    assert row["kubeconfig_available"] is False
    assert "provider reports no worker node groups" in row["failure_reasons"]
    assert "kubeconfig is unavailable" in row["failure_reasons"]


def test_expected_second_node_group_missing_is_degraded(monkeypatch) -> None:
    class OneGroup:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a", name=name, project_id=project_id, status="RUNNING"
            )

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="cpu-a",
                    name="cluster-a-cpu",
                    cluster_id=cluster_id,
                    status="RUNNING",
                    node_count=1,
                )
            ]

    local = _state()
    local.node_count = 2
    monkeypatch.setattr(status_mod, "MK8sClient", OneGroup)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: local)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [local])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"])

    assert result.exit_code == 3
    row = json.loads(result.output)[0]
    assert row["state"] == "DEGRADED"
    assert row["node_count"] == 1
    assert "expected 2 worker nodes, provider reports 1" in row["failure_reasons"]


@pytest.mark.parametrize(
    ("error", "code", "category"),
    [
        (
            "dial tcp: lookup api.synthetic.invalid: no such host",
            "DNS_RESOLUTION_FAILED",
            "DNS",
        ),
        ("clusters is forbidden by RBAC", "LIVE_QUERY_FORBIDDEN", "RBAC"),
        ("401 Unauthorized", "LIVE_QUERY_AUTHENTICATION_FAILED", "AUTHENTICATION"),
    ],
)
def test_status_live_failures_are_unavailable_nonzero_with_last_known(
    monkeypatch, error: str, code: str, category: str
) -> None:
    class FailingClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            raise RuntimeError(error)

    monkeypatch.setattr(status_mod, "MK8sClient", FailingClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])

    result = runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"])

    assert result.exit_code == 2, result.output
    row = json.loads(result.output)[0]
    assert row["state"] == "VERIFICATION_UNAVAILABLE"
    assert row["live_verified"] is False
    assert row["automation_may_trust_state"] is False
    assert row["last_known"]["state"] == "PROVISIONING"
    assert row["last_known"]["observed_at"] == "2026-05-14T21:50:00Z"
    assert row["live_verification"]["error_code"] == code
    assert row["live_verification"]["category"] == category
    assert row["live_verification"]["retry_command"].startswith("npa cluster status")

    human = runner.invoke(app, ["status", "--name", "cluster-a"])
    assert human.exit_code == 2
    assert human.output.startswith("VERIFICATION_UNAVAILABLE")
    assert "last-known state: PROVISIONING" in human.output


def test_cached_cluster_status_is_explicit_untrusted_and_does_not_query(
    monkeypatch,
) -> None:
    class NoQueryClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, *args, **kwargs):
            raise AssertionError("--cached must not query the provider")

    monkeypatch.setattr(status_mod, "MK8sClient", NoQueryClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])

    result = runner.invoke(
        app,
        ["status", "--name", "cluster-a", "--cached", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(result.output)[0]
    assert row["state"] == "CACHED"
    assert row["verification_status"] == "CACHED"
    assert row["last_known"]["state"] == "PROVISIONING"
    assert row["live_verified"] is False


def test_authoritative_absence_and_no_configuration_are_distinct(monkeypatch) -> None:
    class AbsentClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, *args, **kwargs):
            raise status_mod.ClusterNotFoundError("not found")

    monkeypatch.setattr(status_mod, "MK8sClient", AbsentClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    absent = runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"])
    assert absent.exit_code == 0
    absent_row = json.loads(absent.output)[0]
    assert absent_row["state"] == "ABSENT"
    assert absent_row["live_verified"] is True

    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: None)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [])
    monkeypatch.setattr(
        status_mod, "_resolve_environment_for_status", lambda *args: ("", "")
    )
    unconfigured = runner.invoke(
        app, ["status", "--name", "never-configured", "--format", "json"]
    )
    assert unconfigured.exit_code == 0
    unconfigured_row = json.loads(unconfigured.output)[0]
    assert unconfigured_row["state"] == "NOT_CONFIGURED"
    assert unconfigured_row["last_known"]["source"] == "configuration"

    unconfigured_list = runner.invoke(app, ["list", "--format", "json"])
    assert unconfigured_list.exit_code == 0
    list_row = json.loads(unconfigured_list.output)[0]
    assert list_row["name"] == "<not-configured>"
    assert list_row["state"] == "NOT_CONFIGURED"


def test_partial_node_group_provisioning_is_verified_degraded(monkeypatch) -> None:
    class DegradedClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name="cluster-a",
                project_id=project_id,
                status="RUNNING",
            )

        def list_node_groups(self, cluster_id):
            return [
                NodeGroupInfo(
                    id="gpu-a",
                    name="cluster-a-gpu",
                    cluster_id=cluster_id,
                    status="PROVISIONING",
                    node_count=0,
                )
            ]

    monkeypatch.setattr(status_mod, "MK8sClient", DegradedClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["status", "--name", "cluster-a", "--format", "json"])

    assert result.exit_code == 3, result.output
    row = json.loads(result.output)[0]
    assert row["state"] == "DEGRADED"
    assert row["provider_state"] == "RUNNING"
    assert row["live_verified"] is True
