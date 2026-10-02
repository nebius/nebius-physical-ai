"""Reject ambiguous controller rewrites and verify safe NFS peer export ordering."""

import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "wam_add_worker", RECIPE / "add_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    etc = tmp_path / "etc"
    (etc / "slurm").mkdir(parents=True)
    config = (RECIPE / "slurm.conf.in").read_text()
    for key, value in {
        "NODE": "synthetic-controller",
        "ADDRESS": "10.0.0.1",
        "SHAPE": "CPUs=160",
        "MEMORY": "1700000",
        "CLUSTER": "fixture",
    }.items():
        config = config.replace(f"@{key}@", value)
    (etc / "slurm/slurm.conf").write_text(config)

    def path(*parts):
        value = Path(*parts)
        return (
            etc / value.relative_to("/etc") if value.is_relative_to("/etc") else value
        )

    monkeypatch.setattr(module, "Path", path)
    monkeypatch.setattr(
        module, "socket", SimpleNamespace(gethostname=lambda: "synthetic-controller")
    )
    monkeypatch.setattr(module, "_run", Mock())
    return module, etc


def _args():
    return SimpleNamespace(worker_name="synthetic-worker", worker_address="10.0.0.2")


@pytest.mark.parametrize("partition_suffix", [" Default=YES", "\tDefault=YES", ""])
def test_configuration_updates_only_intended_controller_fields(
    controller, partition_suffix
):
    module, etc = controller
    path = etc / "slurm/slurm.conf"
    original = path.read_text().replace(
        " Default=YES MaxTime=INFINITE State=UP", partition_suffix
    )
    path.write_text(original)
    updated, peers = module._configuration(_args())
    assert "NodeName=synthetic-worker NodeAddr=10.0.0.2 CPUs=160" in updated
    assert f"Nodes=synthetic-controller,synthetic-worker{partition_suffix}\n" in updated
    assert "AccountingStorageHost=10.0.0.1\n" in updated
    assert peers == {
        "controller_name": "synthetic-controller",
        "controller_address": "10.0.0.1",
    }
    assert path.read_text() == original
    module._run.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        ("Nodes=synthetic-controller", "Nodes=other"),
        (
            "Nodes=synthetic-controller",
            "Nodes=synthetic-controller Nodes=synthetic-controller",
        ),
        ("PartitionName=gpu", "#PartitionName=gpu"),
        ("AccountingStorageHost=localhost", "AccountingStorageHost=other"),
        ("AccountingStorageHost=localhost", "#AccountingStorageHost=localhost"),
        (
            "AccountingStorageHost=localhost",
            "AccountingStorageHost=localhost\nAccountingStorageHost=other",
        ),
        ("NodeName=synthetic-controller", "NodeName=other"),
    ],
)
def test_ambiguous_or_missing_targets_fail_before_mutation(controller, change):
    module, etc = controller
    path = etc / "slurm/slurm.conf"
    original = path.read_text().replace(*change)
    path.write_text(original)
    with pytest.raises(ValueError):
        module._configuration(_args())
    assert path.read_text() == original
    module._run.assert_not_called()


def test_export_preserves_root_squash_and_persists_peer_rule(controller, tmp_path):
    module, etc = controller
    module._export(tmp_path / "shared", {"worker_address": "10.0.0.2"})
    path = etc / "exports.d/wam-slurm.exports"
    assert (
        path.read_text()
        == f"{tmp_path}/shared 10.0.0.2(rw,sync,no_subtree_check,root_squash)\n"
    )
    assert module._run.call_args_list == [
        call("systemctl", "enable", "--now", "nfs-server"),
        call("exportfs", "-ra"),
        call("iptables", "-I", "WAM_INPUT", "1", "-s", "10.0.0.2", "-j", "ACCEPT"),
        call("bash", str(RECIPE / "persist-firewall.sh")),
    ]


def test_existing_export_is_not_overwritten(controller, tmp_path):
    module, etc = controller
    path = etc / "exports.d/wam-slurm.exports"
    path.parent.mkdir()
    path.write_text("retained peer export\n")
    with pytest.raises(FileExistsError):
        module._export(tmp_path / "shared", {"worker_address": "10.0.0.2"})
    assert path.read_text() == "retained peer export\n"
    module._run.assert_not_called()


@pytest.mark.parametrize(
    "failed_command", ["systemctl", "exportfs", "iptables", "bash"]
)
def test_export_failure_stops_later_operations(controller, tmp_path, failed_command):
    module, _ = controller

    def execute(*command):
        if command[0] == failed_command:
            raise subprocess.CalledProcessError(1, command)

    module._run.side_effect = execute
    with pytest.raises(subprocess.CalledProcessError):
        module._export(tmp_path / "shared", {"worker_address": "10.0.0.2"})
    assert module._run.call_args.args[0] == failed_command
