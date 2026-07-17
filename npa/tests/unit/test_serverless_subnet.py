from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from npa.serverless_common.subnet import SubnetResolutionError, resolve_subnet


@pytest.fixture(autouse=True)
def _clear_subnet_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_SERVERLESS_SUBNET_ID", raising=False)
    monkeypatch.delenv("NPA_SERVERLESS_SUBNET_ID", raising=False)
    monkeypatch.delenv("NPA_NEBIUS_PROFILE", raising=False)
    monkeypatch.delenv("NEBIUS_PROFILE", raising=False)


def _resource(
    resource_id: str,
    name: str,
    *,
    state: str = "READY",
    network_id: str = "",
) -> dict:
    resource = {
        "metadata": {"id": resource_id, "name": name},
        "status": {"state": state},
    }
    if network_id:
        resource["spec"] = {"network_id": network_id}
    return resource


def _result(items: list[dict], returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        returncode=returncode,
        stdout=json.dumps({"items": items}),
        stderr=stderr,
    )


def _vpc_resource(args: list[str]) -> str:
    """Return vpc resource kind from ``nebius [--profile X] vpc <kind> ...``."""
    assert args and args[0] == "nebius"
    idx = 1
    if idx < len(args) and args[idx] == "--profile":
        idx += 2
    assert args[idx : idx + 1] == ["vpc"], args
    return str(args[idx + 1])


def _mock_nebius(mocker, *, networks: list[dict], subnets: list[dict]):
    def run(args, **kwargs):
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        kind = _vpc_resource(list(args))
        if kind == "network":
            return _result(networks)
        if kind == "subnet":
            return _result(subnets)
        raise AssertionError(f"unexpected command: {args}")

    return mocker.patch("npa.serverless_common.subnet.subprocess.run", side_effect=run)


def test_explicit_override_wins_without_cli_calls(mocker) -> None:
    run = mocker.patch("npa.serverless_common.subnet.subprocess.run")

    assert resolve_subnet("project-1", explicit_subnet_id=" vpcsubnet-foo ") == "vpcsubnet-foo"
    run.assert_not_called()


def test_env_subnet_override_wins_without_cli_calls(mocker, monkeypatch) -> None:
    run = mocker.patch("npa.serverless_common.subnet.subprocess.run")
    monkeypatch.setenv("NPA_E2E_SERVERLESS_SUBNET_ID", " vpcsubnet-e2e ")
    monkeypatch.delenv("NPA_SERVERLESS_SUBNET_ID", raising=False)

    assert resolve_subnet("project-1") == "vpcsubnet-e2e"
    run.assert_not_called()


def test_nebius_profile_passed_to_vpc_list(mocker, monkeypatch) -> None:
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "npa-mk8s")
    monkeypatch.delenv("NEBIUS_PROFILE", raising=False)
    seen: list[list[str]] = []

    def run(args, **kwargs):
        seen.append(list(args))
        kind = _vpc_resource(list(args))
        if kind == "network":
            return _result([_resource("vpcnetwork-default", "default-network")])
        if kind == "subnet":
            return _result(
                [
                    _resource(
                        "vpcsubnet-default",
                        "default-subnet-xyz",
                        network_id="vpcnetwork-default",
                    )
                ]
            )
        raise AssertionError(f"unexpected command: {args}")

    mocker.patch("npa.serverless_common.subnet.subprocess.run", side_effect=run)
    assert resolve_subnet("project-1") == "vpcsubnet-default"
    assert seen
    assert seen[0][:4] == ["nebius", "--profile", "npa-mk8s", "vpc"]


def test_default_network_default_subnet_selection(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-default", "default-network")],
        subnets=[
            _resource("vpcsubnet-other", "tool-subnet", network_id="vpcnetwork-other"),
            _resource("vpcsubnet-default", "default-subnet-xyz", network_id="vpcnetwork-default"),
        ],
    )

    assert resolve_subnet("project-1") == "vpcsubnet-default"


def test_single_subnet_fallback(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-tool", "tool-network")],
        subnets=[_resource("vpcsubnet-only", "tool-subnet", network_id="vpcnetwork-tool")],
    )

    assert resolve_subnet("project-1") == "vpcsubnet-only"


def test_multi_subnet_no_default_raises(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-a", "network-a")],
        subnets=[
            _resource("vpcsubnet-a", "subnet-a", network_id="vpcnetwork-a"),
            _resource("vpcsubnet-b", "subnet-b", network_id="vpcnetwork-b"),
        ],
    )

    with pytest.raises(SubnetResolutionError, match="specify --subnet-id"):
        resolve_subnet("project-1")


def test_no_subnets_in_project_raises(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-default", "default-network")],
        subnets=[],
    )

    with pytest.raises(SubnetResolutionError, match="No READY subnets"):
        resolve_subnet("project-1")


def test_cli_subprocess_error_raises_with_stderr(mocker) -> None:
    mocker.patch(
        "npa.serverless_common.subnet.subprocess.run",
        return_value=_result([], returncode=1, stderr="permission denied"),
    )

    with pytest.raises(SubnetResolutionError, match="permission denied"):
        resolve_subnet("project-1")


def test_non_ready_subnets_ignored(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-default", "default-network")],
        subnets=[
            _resource(
                "vpcsubnet-default",
                "default-subnet-xyz",
                state="PROVISIONING",
                network_id="vpcnetwork-default",
            ),
            _resource("vpcsubnet-ready", "tool-subnet", network_id="vpcnetwork-tool"),
        ],
    )

    assert resolve_subnet("project-1") == "vpcsubnet-ready"


def test_multiple_default_networks_raise(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[
            _resource("vpcnetwork-default-a", "default-network"),
            _resource("vpcnetwork-default-b", "default-network"),
        ],
        subnets=[_resource("vpcsubnet-only", "only-subnet", network_id="vpcnetwork-default-a")],
    )

    with pytest.raises(SubnetResolutionError, match="multiple READY networks"):
        resolve_subnet("project-1")


def test_multiple_default_subnets_under_default_network_raise(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-default", "default-network")],
        subnets=[
            _resource("vpcsubnet-a", "default-subnet-a", network_id="vpcnetwork-default"),
            _resource("vpcsubnet-b", "default-subnet-b", network_id="vpcnetwork-default"),
        ],
    )

    with pytest.raises(SubnetResolutionError, match="multiple READY subnets"):
        resolve_subnet("project-1")


def test_custom_network_and_subnet_prefix_params(mocker) -> None:
    _mock_nebius(
        mocker,
        networks=[_resource("vpcnetwork-prod", "prod-network")],
        subnets=[
            _resource("vpcsubnet-prod", "prod-subnet-main", network_id="vpcnetwork-prod"),
            _resource("vpcsubnet-other", "default-subnet-a", network_id="vpcnetwork-other"),
        ],
    )

    assert (
        resolve_subnet(
            "project-1",
            default_network_name="prod-network",
            default_subnet_prefix="prod-subnet",
        )
        == "vpcsubnet-prod"
    )
