from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.nebius import NebiusError


runner = CliRunner()


def _instance(
    *,
    vm_id: str = "computeinstance-test",
    project_id: str = "project-test",
    public_ip: str = "203.0.113.10/32",
    security_groups: list[str] | None = None,
) -> dict[str, Any]:
    groups = security_groups if security_groups is not None else ["sg-test"]
    return {
        "metadata": {"id": vm_id, "parent_id": project_id, "name": "vm"},
        "spec": {
            "network_interfaces": [
                {
                    "name": "eth0",
                    "subnet_id": "subnet-test",
                    "security_groups": [{"id": group} for group in groups],
                }
            ]
        },
        "status": {
            "state": "RUNNING",
            "network_interfaces": [
                {
                    "name": "eth0",
                    "public_ip_address": {"address": public_ip},
                }
            ],
        },
    }


def _security_group(group_id: str = "sg-test") -> dict[str, Any]:
    return {
        "metadata": {"id": group_id, "parent_id": "project-test", "name": "demo-sg"},
        "spec": {"network_id": "network-test"},
        "status": {"state": "READY"},
    }


def _ingress_rule(
    *,
    rule_id: str = "rule-existing",
    name: str = "allow-existing",
    ports: list[int],
    source: str = "0.0.0.0/0",
    protocol: str = "TCP",
    access: str = "ALLOW",
) -> dict[str, Any]:
    return {
        "metadata": {"id": rule_id, "parent_id": "sg-test", "name": name},
        "spec": {
            "access": access,
            "priority": 500,
            "protocol": protocol,
            "ingress": {
                "source_cidrs": [source],
                "destination_ports": ports,
            },
            "type": "STATEFUL",
        },
        "status": {"state": "READY", "direction": "INGRESS"},
    }


def _mock_nebius(
    mocker,
    *,
    instance: dict[str, Any] | None = None,
    instances: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    rules_by_sg: dict[str, list[dict[str, Any]]] | None = None,
    get_instance_error: str | None = None,
    create_error: str | None = None,
):
    calls: list[list[str]] = []
    instance = instance if instance is not None else _instance()
    instances = instances if instances is not None else [instance]
    rules = rules if rules is not None else []

    def fake_run_json(args: list[str], **_kwargs):
        calls.append(args)
        if args[:3] == ["compute", "instance", "get"]:
            if get_instance_error:
                raise NebiusError(get_instance_error)
            return instance
        if args[:3] == ["compute", "instance", "list"]:
            return {"items": instances}
        if args[:3] == ["vpc", "security-group", "get"]:
            return _security_group(args[3])
        if args[:3] == ["vpc", "security-rule", "list"]:
            group_id = args[args.index("--parent-id") + 1]
            return {"items": (rules_by_sg or {}).get(group_id, rules)}
        if args[:3] == ["vpc", "security-rule", "create"]:
            if create_error:
                raise NebiusError(create_error)
            name = args[args.index("--name") + 1]
            group_id = args[args.index("--parent-id") + 1]
            return {"metadata": {"id": f"rule-created-{group_id}", "name": name}}
        raise AssertionError(f"unexpected nebius args: {args}")

    mocker.patch("npa.clients.nebius._run_json", side_effect=fake_run_json)
    return calls


def _create_calls(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[:3] == ["vpc", "security-rule", "create"]]


def test_network_ensure_ingress_success_with_vm(mocker) -> None:
    calls = _mock_nebius(mocker)

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
            "--tool",
            "cosmos",
        ],
    )

    assert result.exit_code == 0
    assert "created_rule: rule-created-sg-test" in result.output
    assert "created_rule_name: allow-npa-cosmos-8081" in result.output
    assert len(_create_calls(calls)) == 1


def test_network_ensure_ingress_success_with_ip_project(mocker) -> None:
    calls = _mock_nebius(mocker)

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--ip",
            "203.0.113.10",
            "--project",
            "project-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code == 0
    assert "vm: computeinstance-test" in result.output
    assert any(call[:3] == ["compute", "instance", "list"] for call in calls)
    assert len(_create_calls(calls)) == 1


def test_network_ensure_ingress_matching_spec_is_noop(mocker) -> None:
    calls = _mock_nebius(
        mocker,
        rules=[
            _ingress_rule(
                name="allow-demo-services-5151-8081-8082",
                ports=[5151, 8081, 8082],
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "5151,8081,8082",
        ],
    )

    assert result.exit_code == 0
    assert "matching spec already covered, no rule changes" in result.output
    assert _create_calls(calls) == []


def test_network_ensure_ingress_same_name_different_spec_warns(mocker) -> None:
    calls = _mock_nebius(
        mocker,
        rules=[
            _ingress_rule(
                name="allow-npa-cosmos-8081",
                ports=[8081],
                source="10.0.0.0/8",
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
            "--tool",
            "cosmos",
        ],
    )

    assert result.exit_code == 0
    assert (
        "already uses name 'allow-npa-cosmos-8081' but does not match" in result.output
    )
    assert len(_create_calls(calls)) == 1


def test_network_ensure_ingress_partial_coverage_creates_missing_ports(mocker) -> None:
    calls = _mock_nebius(
        mocker,
        rules=[
            _ingress_rule(name="allow-npa-cosmos-8081", ports=[8081]),
        ],
    )

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081,8082",
            "--tool",
            "cosmos",
        ],
    )

    assert result.exit_code == 0
    create = _create_calls(calls)[0]
    assert create[create.index("--name") + 1] == "allow-npa-cosmos-8082"
    assert create.count("--ingress-destination-ports") == 1
    assert "8082" in create
    assert "8081" not in create


def test_network_ensure_ingress_permission_failure_is_clean(mocker) -> None:
    _mock_nebius(mocker, create_error="permission denied")

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code == 1
    assert "Could not create ingress rule" in result.output
    assert "permission denied" in result.output


def test_network_ensure_ingress_missing_instance_is_clean(mocker) -> None:
    _mock_nebius(mocker, get_instance_error="not found")

    result = runner.invoke(
        app,
        ["network", "ensure-ingress", "--vm", "missing-vm", "--ports", "8081"],
    )

    assert result.exit_code == 1
    assert "Could not fetch VM missing-vm" in result.output
    assert "not found" in result.output


def test_network_ensure_ingress_missing_security_group_is_clean(mocker) -> None:
    _mock_nebius(mocker, instance=_instance(security_groups=[]))

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code == 1
    assert "has no security group references" in result.output


def test_network_ensure_ingress_multiple_ports_collapsed_into_single_rule(
    mocker,
) -> None:
    calls = _mock_nebius(mocker)

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8082,5151,8081",
            "--tool",
            "fiftyone",
        ],
    )

    assert result.exit_code == 0
    create_calls = _create_calls(calls)
    assert len(create_calls) == 1
    create = create_calls[0]
    assert create[create.index("--name") + 1] == "allow-npa-fiftyone-5151-8081-8082"
    assert create.count("--ingress-destination-ports") == 3
    assert create[-6:] == [
        "--ingress-destination-ports",
        "5151",
        "--ingress-destination-ports",
        "8081",
        "--ingress-destination-ports",
        "8082",
    ]


def test_network_ensure_ingress_multi_sg_coverage_in_second_group_is_noop(
    mocker,
) -> None:
    calls = _mock_nebius(
        mocker,
        instance=_instance(security_groups=["sg-one", "sg-two"]),
        rules_by_sg={
            "sg-one": [],
            "sg-two": [_ingress_rule(name="allow-existing", ports=[8081])],
        },
    )

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code == 0
    assert "matching spec already covered, no rule changes" in result.output
    assert _create_calls(calls) == []


def test_network_ensure_ingress_multi_sg_missing_ports_create_on_first_group_only(
    mocker,
) -> None:
    calls = _mock_nebius(
        mocker,
        instance=_instance(security_groups=["sg-one", "sg-two"]),
        rules_by_sg={"sg-one": [], "sg-two": []},
    )

    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code == 0
    create_calls = _create_calls(calls)
    assert len(create_calls) == 1
    assert create_calls[0][create_calls[0].index("--parent-id") + 1] == "sg-one"


def test_network_ensure_ingress_rejects_vm_and_ip_combination() -> None:
    result = runner.invoke(
        app,
        [
            "network",
            "ensure-ingress",
            "--vm",
            "computeinstance-test",
            "--ip",
            "203.0.113.10",
            "--project",
            "project-test",
            "--ports",
            "8081",
        ],
    )

    assert result.exit_code != 0
    assert "pass exactly one of --vm or (--ip and --project)" in result.output


def test_network_ensure_ingress_rejects_missing_target() -> None:
    result = runner.invoke(app, ["network", "ensure-ingress", "--ports", "8081"])

    assert result.exit_code != 0
    assert "pass exactly one of --vm or (--ip and --project)" in result.output


def test_remove_npa_ingress_rules_deletes_allow_npa_rules(mocker) -> None:
    from npa.clients import network as network_client

    mocker.patch(
        "npa.clients.network._list_security_rules",
        return_value=[
            {
                "metadata": {"id": "rule-npa", "name": "allow-npa-agent-443"},
                "spec": {"ingress": {"destination_ports": [443]}},
            },
            {
                "metadata": {"id": "rule-tf", "name": "allow-server"},
                "spec": {"ingress": {"destination_ports": [8088]}},
            },
        ],
    )
    run = mocker.patch("npa.clients.network.nebius._run")

    deleted = network_client.remove_npa_ingress_rules(("sg-test",))

    assert deleted == ["rule-npa"]
    run.assert_called_once_with(["vpc", "security-rule", "delete", "--id", "rule-npa"])


@pytest.mark.parametrize(
    "message",
    [
        "rpc error: code = FailedPrecondition desc = cannot delete default security group",
        "Error: default security group cannot be deleted",
        "operation failed: CannotDeleteDefaultSecurityGroup",
    ],
)
def test_default_security_group_refusal_is_narrowly_classified(message: str) -> None:
    from npa.clients.network import is_default_security_group_delete_refusal

    assert is_default_security_group_delete_refusal(message)


@pytest.mark.parametrize(
    "message",
    [
        "FailedPrecondition: security group is assigned to an instance",
        "PermissionDenied deleting non-default security group",
        "FailedPrecondition: non-default security group cannot be deleted while in use",
        "connection reset while deleting security group",
        "default network route is still in use",
    ],
)
def test_non_default_security_group_failures_are_not_reclassified(message: str) -> None:
    from npa.clients.network import is_default_security_group_delete_refusal

    assert not is_default_security_group_delete_refusal(message)


def test_owned_network_recovers_default_security_group_refusal(mocker) -> None:
    from npa.clients.network import recover_default_security_group_delete

    run = mocker.patch("npa.clients.network.nebius._run")
    status: list[str] = []

    recovered = recover_default_security_group_delete(
        error="FailedPrecondition: cannot delete default security group",
        parent_network_id="vpcnetwork-owned",
        parent_network_owned=True,
        cleanup_action="`npa agent destroy --project prod --yes`",
        on_status=status.append,
    )

    assert recovered is True
    run.assert_called_once_with(
        ["vpc", "network", "delete", "--id", "vpcnetwork-owned"]
    )
    joined = "\n".join(status)
    assert "cannot be deleted directly" in joined
    assert "NPA-owned" in joined
    assert "parent network" in joined


def test_unowned_network_is_preserved_after_default_security_group_refusal(
    mocker,
) -> None:
    from npa.clients.network import (
        NetworkCleanupError,
        recover_default_security_group_delete,
    )

    run = mocker.patch("npa.clients.network.nebius._run")

    with pytest.raises(NetworkCleanupError) as caught:
        recover_default_security_group_delete(
            error="FailedPrecondition: cannot delete default security group",
            parent_network_id="vpcnetwork-shared",
            parent_network_owned=False,
            cleanup_action="`npa agent destroy --project prod --yes`",
        )

    run.assert_not_called()
    message = str(caught.value)
    assert "reused/shared network" in message
    assert "proof that NPA owns" in message
    assert "parent network" in message
    assert "npa agent destroy" in message


def test_owned_parent_network_already_absent_is_idempotent(mocker) -> None:
    from npa.clients.network import recover_default_security_group_delete

    mocker.patch(
        "npa.clients.network.nebius._run",
        side_effect=NebiusError("NotFound: network is already absent"),
    )
    status: list[str] = []

    assert recover_default_security_group_delete(
        error="FailedPrecondition: default security group cannot be deleted",
        parent_network_id="vpcnetwork-gone",
        parent_network_owned=True,
        cleanup_action="`npa agent destroy --project prod --yes`",
        on_status=status.append,
    )
    assert any("already absent" in line for line in status)


def test_genuine_security_group_failure_never_deletes_parent_network(mocker) -> None:
    from npa.clients.network import recover_default_security_group_delete

    run = mocker.patch("npa.clients.network.nebius._run")

    assert not recover_default_security_group_delete(
        error="FailedPrecondition: non-default security group is still in use",
        parent_network_id="vpcnetwork-owned",
        parent_network_owned=True,
        cleanup_action="`npa agent destroy --project prod --yes`",
    )
    run.assert_not_called()


def test_delete_project_default_requires_durable_project_ownership(mocker) -> None:
    from npa.clients import nebius

    mocker.patch(
        "npa.clients.nebius.get_project_identity",
        return_value=nebius.ProjectIdentity(
            "project-a", "demo", "tenant-a", "us-central1", "test"
        ),
    )
    mocker.patch("npa.project_destroy._project_ownership_operation", return_value=None)
    delete = mocker.patch("npa.clients.nebius.delete_project_default_network")

    result = runner.invoke(
        app,
        [
            "network",
            "delete-project-default",
            "--project",
            "demo",
            "--project-id",
            "project-a",
            "--tenant-id",
            "tenant-a",
            "--yes",
        ],
    )

    assert result.exit_code == 1
    assert "durable NPA project-creation proof" in result.output
    delete.assert_not_called()


def test_delete_project_default_verifies_exact_absence(mocker) -> None:
    from npa.clients import nebius

    identity = nebius.ProjectDefaultNetworkIdentity(
        "network-a",
        "default-network",
        "subnet-a",
        "default-subnet-a",
        "sg-a",
        "default-security-group-a",
        "project-a",
        "test",
    )
    mocker.patch(
        "npa.clients.nebius.get_project_identity",
        return_value=nebius.ProjectIdentity(
            "project-a", "demo", "tenant-a", "us-central1", "test"
        ),
    )
    owner = mocker.Mock(operation_id="project-create-a")
    mocker.patch("npa.project_destroy._project_ownership_operation", return_value=owner)
    get_topology = mocker.patch(
        "npa.clients.nebius.get_project_default_network_identity",
        side_effect=[identity, None],
    )
    delete = mocker.patch("npa.clients.nebius.delete_project_default_network")
    record = mocker.patch("npa.teardown_receipts.record_teardown_event")

    result = runner.invoke(
        app,
        [
            "network",
            "delete-project-default",
            "--project",
            "demo",
            "--project-id",
            "project-a",
            "--tenant-id",
            "tenant-a",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"outcome": "verified_deleted"' in result.output
    delete.assert_called_once_with(identity, profile=None)
    assert get_topology.call_count == 2
    assert record.call_count == 2
    for call in record.call_args_list:
        assert "outcome" not in call.kwargs["identity"]
