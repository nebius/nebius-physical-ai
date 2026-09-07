"""Regression coverage for authenticated local access to the operator App."""

import ast
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli import fiftyone
from npa.cli.main import app
from npa.clients.endpoint import EndpointError


runner = CliRunner()


@pytest.mark.parametrize("command", ["deploy", "launch"])
def test_non_loopback_address_is_rejected_before_remote_operations(command, mocker):
    provision = mocker.patch.object(fiftyone.provisioner, "apply")
    ssh = mocker.patch.object(fiftyone, "SSHClient")
    result = runner.invoke(app, ["workbench", "fiftyone", command, "--address", "0.0.0.0"])
    assert result.exit_code == 1
    assert "requires loopback binding" in result.output
    provision.assert_not_called()
    ssh.assert_not_called()


def test_kubernetes_public_request_fails_before_cluster_access(mocker):
    resolve = mocker.patch.object(fiftyone, "_resolve_required_kubeconfig")
    kubectl = mocker.patch.object(fiftyone, "_kubectl")
    result = runner.invoke(app, ["workbench", "fiftyone", "deploy", "--public-ip"])
    assert result.exit_code == 1
    resolve.assert_not_called()
    kubectl.assert_not_called()


def test_kubernetes_redeploy_reconciles_app_listener_and_loopback_probe(mocker):
    mocker.patch.object(fiftyone, "_resolve_required_kubeconfig", return_value="operator-kubeconfig")
    mocker.patch.object(fiftyone, "_save_k8s_workbench_state")
    kubectl = mocker.patch.object(fiftyone, "_kubectl")
    image = "registry.example/fiftyone@sha256:" + "a" * 64
    fiftyone._deploy_kubernetes_fiftyone(
        cluster_name="review", kubeconfig="operator-kubeconfig", image=image,
        name="curation", namespace="workbench", port=6161, address="127.0.0.1",
        image_pull_secret="", public_ip=False, destroy=False, dry_run=False,
        output=fiftyone.OutputFormat.json,
    )
    applied = kubectl.call_args_list[0]
    assert applied.args == (["apply", "-f", "-"],)
    manifest = json.loads(applied.kwargs["stdin"])
    deployment, service = manifest["items"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == image
    assert 'address = "127.0.0.1"' in container["command"][-1]
    assert "httpGet" not in container["readinessProbe"]
    assert "HTTPConnection('127.0.0.1',6161" in container["readinessProbe"]["exec"]["command"][-1]
    assert service["spec"]["type"] == "ClusterIP"


def test_vm_open_keeps_verified_tunnel_until_interrupt_and_cleans_up(mocker):
    cfg = SimpleNamespace(service_port=6161, endpoint="http://legacy.example:5151")
    mocker.patch.object(fiftyone, "_try_get_ssh_config", return_value=cfg)
    process = mocker.MagicMock()
    process.poll.return_value = None
    forward = mocker.patch("npa.clients.endpoint._open_ssh_forward", return_value=process)
    ready = mocker.patch("npa.clients.endpoint._wait_for_ssh_forward")
    close = mocker.patch("npa.clients.endpoint._close_process")
    browser = mocker.patch.object(fiftyone.webbrowser, "open")
    mocker.patch.object(fiftyone.time, "sleep", side_effect=KeyboardInterrupt)
    result = runner.invoke(app, ["workbench", "fiftyone", "open", "--local-port", "15151"])
    assert result.exit_code == 0
    forward.assert_called_once_with(cfg, 15151, 6161)
    ready.assert_called_once_with(process, 15151, 6161)
    browser.assert_called_once_with("http://127.0.0.1:15151?polling=true")
    close.assert_called_once_with(process)


def test_vm_open_does_not_open_browser_when_ssh_is_not_ready(mocker):
    cfg = SimpleNamespace(service_port=5151, endpoint="http://legacy.example:5151")
    mocker.patch.object(fiftyone, "_try_get_ssh_config", return_value=cfg)
    process = mocker.MagicMock()
    mocker.patch("npa.clients.endpoint._open_ssh_forward", return_value=process)
    mocker.patch("npa.clients.endpoint._wait_for_ssh_forward", side_effect=EndpointError("SSH not ready"))
    close = mocker.patch("npa.clients.endpoint._close_process")
    browser = mocker.patch.object(fiftyone.webbrowser, "open")
    result = runner.invoke(app, ["workbench", "fiftyone", "open"])
    assert result.exit_code == 1
    browser.assert_not_called()
    close.assert_called_once_with(process)


def test_sdk_launch_defaults_to_loopback(mocker):
    from npa.workbench import fiftyone as sdk

    callback = mocker.patch.object(sdk, "call_cli_callback")
    sdk.launch()
    assert callback.call_args.kwargs["address"] == "127.0.0.1"


def test_load_dataset_migrates_native_listener_before_import(mocker):
    cfg = SimpleNamespace(service_port=6161, runtime="vm", ssh=SimpleNamespace())
    mocker.patch.object(fiftyone, "_get_ssh_config", return_value=cfg)
    mocker.patch.object(fiftyone, "SSHClient")
    run = mocker.patch.object(fiftyone, "_run_fiftyone_command", return_value=(0, "{}", ""))
    result = runner.invoke(app, ["workbench", "fiftyone", "load-dataset", "--name", "review", "--input-path", "s3://example-bucket/images", "--output", "json"])
    assert result.exit_code == 0
    assert len(run.call_args_list) == 2
    assert "FIFTYONE_DEFAULT_APP_ADDRESS=127.0.0.1" in run.call_args_list[0].args[1]
    assert "FIFTYONE_DEFAULT_APP_PORT=6161" in run.call_args_list[0].args[1]
    assert "download_s3(SOURCE)" in run.call_args_list[1].args[1]


def test_kubernetes_readiness_requires_its_process_to_bind_the_local_port():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        script = (
            "import socket; s=socket.socket(); "
            f"s.bind(('127.0.0.1',{port})); "
            f"print('Forwarding from 127.0.0.1:{port} -> 5151',flush=True)"
        )
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            with pytest.raises(EndpointError, match="before confirming"):
                fiftyone._wait_for_kubernetes_forward(proc, port, 5151)
        finally:
            from npa.clients.endpoint import _close_process

            _close_process(proc)


def test_kubernetes_readiness_accepts_matching_confirmation_from_live_process():
    script = (
        "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
        "print(f'Forwarding from 127.0.0.1:{s.getsockname()[1]} -> 5151',flush=True); "
        "sys.stdin.read()"
    )
    from npa.clients.endpoint import _close_process

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    script = script.replace("('127.0.0.1',0)", f"('127.0.0.1',{port})")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        fiftyone._wait_for_kubernetes_forward(proc, port, 5151)
        assert proc.poll() is None
    finally:
        _close_process(proc)
        proc.stdin.close()


@pytest.mark.parametrize("container", [False, True])
def test_lerobot_importer_loads_from_private_directory_and_cleans_up(container, tmp_path):
    builder = fiftyone._build_container_load_dataset_command if container else fiftyone._build_load_dataset_command
    shell = shlex.split(builder("review", "s3://example-bucket/dataset", fiftyone.DatasetFormat.lerobot))[-1]
    if container:
        exec_start = shell.index("sudo docker exec -i ")
        shell = shlex.split(shell[exec_start:])[7]
    script = shell.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    function = next(node for node in ast.parse(script).body if isinstance(node, ast.FunctionDef) and node.name == "load_lerobot")
    importer = fiftyone._lerobot_importer_source() + '''
def import_lerobot_dataset(name, source, directory):
    path = Path(__file__)
    sample = LeRobotSampleSpec(filepath="frame.png", fields={}, tags=[])
    return {"path": str(path), "parent_mode": path.parent.stat().st_mode & 0o777,
            "name": name, "sample": sample.filepath}
'''
    namespace = {
        "Path": Path, "sys": sys, "LEROBOT_IMPORTER_SOURCE": importer,
        "NAME": "review", "SOURCE": "s3://example-bucket/dataset", "DATASETS_DIR": tmp_path,
    }
    prior_path = list(sys.path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "generated_loader", "exec"), namespace)
    result = namespace["load_lerobot"]()
    assert result["parent_mode"] == 0o700
    assert result["name"] == "review"
    assert result["sample"] == "frame.png"
    assert not Path(result["path"]).parent.exists()
    assert sys.path == prior_path
    assert "_npa_fiftyone_lerobot_importer" not in sys.modules
