from __future__ import annotations

from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import deploy as deploy_mod
from npa.cluster.api import ClusterInfo
from npa.cluster.config import ClusterConfig


runner = CliRunner()


def test_deploy_public_ip_flag_is_opt_in(monkeypatch) -> None:
    seen_configs: list[ClusterConfig] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def create_cluster(self, config: ClusterConfig):
            seen_configs.append(config)
            return ClusterInfo(
                id="mk8scluster-a",
                name=config.name,
                project_id=config.project_id,
                status="RUNNING",
                node_group_id="mk8snodegroup-a",
            )

        def wait_for_ready(self, cluster_id, **kwargs):
            return ClusterInfo(
                id=cluster_id,
                name="cluster-a",
                project_id="project-a",
                status="RUNNING",
                node_count=1,
                node_group_id="mk8snodegroup-a",
            )

        def get_kubeconfig(self, cluster_id, kubeconfig_path, **kwargs):
            return kubeconfig_path

    monkeypatch.setattr(deploy_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(deploy_mod, "resolve_project_id", lambda project_id="": "project-a")
    monkeypatch.setattr(deploy_mod, "resolve_subnet", lambda project_id: "vpcsubnet-a")
    monkeypatch.setattr(deploy_mod, "save_cluster_state", lambda *args, **kwargs: None)

    result = runner.invoke(app, ["deploy", "--name", "cluster-a", "--no-wait"])
    assert result.exit_code == 0
    assert seen_configs[-1].public_node_ip is False

    result = runner.invoke(app, ["deploy", "--name", "cluster-a", "--no-wait", "--public-ip"])
    assert result.exit_code == 0
    assert seen_configs[-1].public_node_ip is True
