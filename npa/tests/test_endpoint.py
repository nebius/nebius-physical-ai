from __future__ import annotations

from types import SimpleNamespace

import yaml
import pytest

from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.endpoint import service_endpoint


def _cfg(
    *,
    strategy: str = "public",
    endpoint: str = "http://vm:8080",
    strategy_configured: bool = True,
    runtime: str = "vm",
) -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint=endpoint,
        ssh=SSHConfig(host="vm", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        endpoint_strategy=strategy,
        service_port=8080,
        endpoint_strategy_configured=strategy_configured,
        service_port_configured=True,
        runtime=runtime,
    )


def test_service_endpoint_defaults_to_public() -> None:
    with service_endpoint(_cfg()) as active:
        assert active.url == "http://vm:8080"
        assert active.strategy == "public"


def test_service_endpoint_serverless_uses_saved_public_url(mocker) -> None:
    cfg = _cfg(strategy="ssh_fallback", endpoint="https://cosmos.example", runtime="serverless")
    popen = mocker.patch("npa.clients.endpoint.subprocess.Popen")

    with service_endpoint(cfg) as active:
        assert active.url == "https://cosmos.example"
        assert active.strategy == "serverless"

    popen.assert_not_called()


def test_service_endpoint_keeps_existing_loopback_tunnel() -> None:
    cfg = _cfg(strategy="ssh_fallback", endpoint="http://127.0.0.1:18081")

    with service_endpoint(cfg) as active:
        assert active.url == "http://127.0.0.1:18081"
        assert active.strategy == "ssh_fallback"


def test_service_endpoint_opens_transient_ssh_forward(mocker) -> None:
    cfg = _cfg(strategy="ssh_fallback")
    popen = mocker.patch("npa.clients.endpoint.subprocess.Popen")
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=mocker.MagicMock(),
        wait=mocker.MagicMock(),
        stderr=SimpleNamespace(read=lambda: ""),
    )
    popen.return_value = process
    mocker.patch("npa.clients.endpoint._free_local_port", return_value=19090)
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=False)
    wait = mocker.patch("npa.clients.endpoint._wait_for_local_port")

    with service_endpoint(cfg) as active:
        assert active.url == "http://127.0.0.1:19090"
        assert active.local_port == 19090

    wait.assert_called_once_with(19090)
    process.terminate.assert_called_once()
    popen.assert_called_once()
    assert "127.0.0.1:19090:127.0.0.1:8080" in popen.call_args.args[0]


def test_service_endpoint_accepts_legacy_ssh_strategy_name(mocker) -> None:
    cfg = _cfg(strategy="ssh", strategy_configured=True, runtime="byovm")
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=True)
    public_probe = mocker.patch("npa.clients.endpoint._public_endpoint_open")

    with service_endpoint(cfg) as active:
        assert active.strategy == "ssh_fallback"

    public_probe.assert_not_called()


@pytest.mark.parametrize("strategy_configured", [False, True])
def test_service_endpoint_self_heals_blocked_byovm_public_alias_to_ssh(
    strategy_configured: bool,
    mocker,
) -> None:
    cfg = _cfg(
        strategy="public",
        strategy_configured=strategy_configured,
        runtime="byovm",
        endpoint="http://vm:5151",
    )
    cfg.service_port = 0
    cfg.project = "proj"
    cfg.name = "fiftyone"
    popen = mocker.patch("npa.clients.endpoint.subprocess.Popen")
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=mocker.MagicMock(),
        wait=mocker.MagicMock(),
        stderr=SimpleNamespace(read=lambda: ""),
    )
    popen.return_value = process
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=False)
    mocker.patch("npa.clients.endpoint._free_local_port", return_value=15151)
    mocker.patch("npa.clients.endpoint._wait_for_local_port")
    persist = mocker.patch("npa.clients.endpoint.update_workbench_endpoint_strategy")

    with service_endpoint(cfg, default_port=5151) as active:
        assert active.url == "http://127.0.0.1:15151"
        assert active.strategy == "ssh_fallback"

    persist.assert_called_once_with("proj", "fiftyone", "ssh_fallback", 5151)
    assert cfg.endpoint_strategy == "ssh_fallback"
    assert cfg.service_port == 5151
    assert "127.0.0.1:15151:127.0.0.1:5151" in popen.call_args.args[0]


def test_service_endpoint_persists_self_healed_strategy_to_config(
    tmp_path,
    monkeypatch,
    mocker,
) -> None:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({
        "projects": {
            "proj": {
                "workbenches": {
                    "fiftyone": {
                        "endpoint": "http://vm:5151",
                        "runtime": "byovm",
                        "ssh": {
                            "host": "vm",
                            "user": "ubuntu",
                            "key_path": "~/.ssh/id",
                        },
                    },
                },
            },
        },
    }))
    cfg = config_module.resolve_ssh_config(project="proj", name="fiftyone")
    assert cfg.endpoint_strategy_configured is False
    assert cfg.service_port_configured is False

    process = SimpleNamespace(
        poll=lambda: None,
        terminate=mocker.MagicMock(),
        wait=mocker.MagicMock(),
        stderr=SimpleNamespace(read=lambda: ""),
    )
    mocker.patch("npa.clients.endpoint.subprocess.Popen", return_value=process)
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=False)
    mocker.patch("npa.clients.endpoint._free_local_port", return_value=15151)
    mocker.patch("npa.clients.endpoint._wait_for_local_port")

    with service_endpoint(cfg, default_port=5151):
        pass

    saved = yaml.safe_load(cfg_path.read_text())
    wb = saved["projects"]["proj"]["workbenches"]["fiftyone"]
    assert wb["endpoint_strategy"] == "ssh_fallback"
    assert wb["service_port"] == 5151


def test_service_endpoint_stored_strategy_skips_legacy_probe(mocker) -> None:
    cfg = _cfg(strategy="ssh_fallback", strategy_configured=True, runtime="byovm")
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=True)
    public_probe = mocker.patch("npa.clients.endpoint._public_endpoint_open")

    with service_endpoint(cfg) as active:
        assert active.url == "http://127.0.0.1:8080"
        assert active.strategy == "ssh_fallback"

    public_probe.assert_not_called()


def test_service_endpoint_stored_ssh_strategy_persists_missing_service_port(mocker) -> None:
    cfg = _cfg(strategy="ssh_fallback", strategy_configured=True, runtime="byovm")
    cfg.service_port_configured = False
    cfg.project = "proj"
    cfg.name = "cosmos"
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=True)
    public_probe = mocker.patch("npa.clients.endpoint._public_endpoint_open")
    persist = mocker.patch("npa.clients.endpoint.update_workbench_endpoint_strategy")

    with service_endpoint(cfg) as active:
        assert active.url == "http://127.0.0.1:8080"

    public_probe.assert_not_called()
    persist.assert_called_once_with("proj", "cosmos", "ssh_fallback", 8080)
