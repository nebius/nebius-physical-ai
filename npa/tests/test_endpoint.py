from __future__ import annotations

from types import SimpleNamespace

import yaml
import pytest

from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.endpoint import EndpointError, service_endpoint


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


@pytest.mark.parametrize("endpoint", ["http://vm:8080", "http://127.0.0.1:8080"])
@pytest.mark.parametrize("strategy", ["public", "ssh_fallback"])
def test_required_ssh_always_creates_fresh_forward(endpoint, strategy, mocker):
    cfg = _cfg(endpoint=endpoint, strategy=strategy)
    process = mocker.MagicMock()
    process.poll.return_value = None
    forward = mocker.patch("npa.clients.endpoint._open_ssh_forward", return_value=process)
    mocker.patch("npa.clients.endpoint._free_local_port", return_value=19090)
    tcp = mocker.patch("npa.clients.endpoint._tcp_open", return_value=True)
    mocker.patch("npa.clients.endpoint._wait_for_ssh_forward")
    mocker.patch("npa.clients.endpoint.time.sleep")
    public = mocker.patch("npa.clients.endpoint._public_endpoint_open", return_value=True)
    with service_endpoint(cfg, require_ssh=True) as active:
        assert active.url == "http://127.0.0.1:19090"
    forward.assert_called_once_with(cfg, 19090, 8080)
    tcp.assert_not_called()
    public.assert_not_called()
    process.terminate.assert_called_once()


@pytest.mark.parametrize("known_hosts_source", ["operator", "provider", "standard"])
def test_ssh_forward_requires_verified_host_keys(known_hosts_source, tmp_path, monkeypatch, mocker):
    from npa.clients.endpoint import _open_ssh_forward

    operator = tmp_path / "operator-known-hosts"
    provider = tmp_path / "provider-known-hosts"
    monkeypatch.delenv("NPA_SSH_KNOWN_HOSTS", raising=False)
    if known_hosts_source == "operator":
        operator.touch()
        monkeypatch.setenv("NPA_SSH_KNOWN_HOSTS", str(operator))
    if known_hosts_source in {"operator", "provider"}:
        provider.touch()
    mocker.patch("npa.deploy.ssh_trust.known_hosts_path", return_value=provider)
    popen = mocker.patch("npa.clients.endpoint.subprocess.Popen")
    process = _open_ssh_forward(_cfg(), 19090, 8080)
    from npa.clients.endpoint import _close_process
    _close_process(process)
    argv = popen.call_args.args[0]
    assert "StrictHostKeyChecking=yes" in argv
    assert "StrictHostKeyChecking=accept-new" not in argv
    assert "ExitOnForwardFailure=yes" in argv
    assert "-M" in argv
    assert "-S" in argv
    selected = [arg for arg in argv if arg.startswith("UserKnownHostsFile=")]
    if known_hosts_source == "standard":
        assert selected == []
    else:
        expected = operator if known_hosts_source == "operator" else provider
        assert selected == [f"UserKnownHostsFile={expected}"]


def test_required_ssh_failure_does_not_yield_endpoint(mocker):
    process = mocker.MagicMock()
    process.poll.return_value = 255
    process.stderr.read.return_value = "Host key verification failed"
    mocker.patch("npa.clients.endpoint._open_ssh_forward", return_value=process)
    mocker.patch("npa.clients.endpoint.time.sleep")
    mocker.patch("npa.clients.endpoint.os.path.exists", return_value=False)
    with pytest.raises(EndpointError, match="host verification failed"):
        with service_endpoint(_cfg(), require_ssh=True):
            pytest.fail("Failed SSH must not produce an active route")


def test_required_ssh_needs_credentials_even_for_loopback_endpoint():
    cfg = _cfg(endpoint="http://127.0.0.1:8080")
    cfg.ssh.host = ""
    with pytest.raises(EndpointError, match="requires ssh host"):
        with service_endpoint(cfg, require_ssh=True):
            pytest.fail("An existing local address is not a verified SSH route")


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
    wait = mocker.patch("npa.clients.endpoint._wait_for_ssh_forward")

    with service_endpoint(cfg) as active:
        assert active.url == "http://127.0.0.1:19090"
        assert active.local_port == 19090

    wait.assert_called_once_with(process, 19090, 8080)
    process.terminate.assert_called_once()
    popen.assert_called_once()
    assert "-M" in popen.call_args.args[0]


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
    ready = mocker.patch("npa.clients.endpoint._wait_for_ssh_forward")
    persist = mocker.patch("npa.clients.endpoint.update_workbench_endpoint_strategy")

    with service_endpoint(cfg, default_port=5151) as active:
        assert active.url == "http://127.0.0.1:15151"
        assert active.strategy == "ssh_fallback"

    persist.assert_called_once_with("proj", "fiftyone", "ssh_fallback", 5151)
    assert cfg.endpoint_strategy == "ssh_fallback"
    assert cfg.service_port == 5151
    ready.assert_called_once_with(process, 15151, 5151)


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
    mocker.patch("npa.clients.endpoint._wait_for_ssh_forward")

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
