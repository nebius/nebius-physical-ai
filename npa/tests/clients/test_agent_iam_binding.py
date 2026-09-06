"""Exact-project grants must be both scope checked and independently removable."""

from copy import deepcopy

import pytest

from npa.clients import agent_iam_binding as binding
from npa.clients import nebius


def resource(identity, parent, name="", **spec):
    return {
        "metadata": {"id": identity, "parent_id": parent, "name": name},
        "spec": spec,
    }


@pytest.fixture
def provider(monkeypatch):
    class Provider:
        def __init__(self):
            self.resources = {
                "project": {"project-test": resource("project-test", "tenant-test")},
                "service-account": {
                    "account-test": resource(
                        "account-test", "project-test", "npa-agent"
                    )
                },
                "group": {},
                "access-permit": {},
                "group-membership": {},
            }
            self.calls = []
            self.created = []
            self.removed = []
            self.transform = lambda argv, response: response
            self.before = lambda argv: None

        def run(self, argv):
            self.calls.append(list(argv))
            self.before(argv)
            assert argv[0] == "iam"
            kind, operation = argv[1:3]
            opts = dict(zip(argv[3::2], argv[4::2]))
            rows = self.resources[kind]
            if operation == "get":
                if opts["--id"] not in rows:
                    raise nebius.NebiusError("NotFound")
                response = rows[opts["--id"]]
            elif operation == "get-by-name":
                matches = [
                    r
                    for r in rows.values()
                    if r["metadata"]["parent_id"] == opts["--parent-id"]
                    and r["metadata"]["name"] == opts["--name"]
                ]
                if not matches:
                    raise nebius.NebiusError("NotFound")
                assert len(matches) == 1
                response = matches[0]
            elif operation in {"list", "list-members"}:
                assert opts["--page-size"] == "1000"
                field = "memberships" if operation == "list-members" else "items"
                response = {
                    field: [
                        r
                        for r in rows.values()
                        if r["metadata"]["parent_id"] == opts["--parent-id"]
                    ]
                }
            elif operation == "create":
                identity = kind + "-test"
                assert identity not in rows
                spec = {
                    key.removeprefix("--").replace("-", "_"): value
                    for key, value in opts.items()
                    if key not in {"--parent-id", "--name"}
                }
                response = resource(
                    identity, opts["--parent-id"], opts.get("--name", ""), **spec
                )
                rows[identity] = response
            elif operation == "delete":
                del rows[opts["--id"]]
                response = {}
            else:
                raise AssertionError(argv)
            return self.transform(argv, deepcopy(response))

        def ensure(self):
            return binding.ensure_agent_project_binding(
                project_id="project-test",
                tenant_id="tenant-test",
                service_account_id="account-test",
                on_resource_created=lambda kind, metadata: self.created.append(
                    (kind, metadata)
                ),
            )

        def owned(self):
            owned = {kind: {} for kind in binding.BINDING_KINDS}
            for kind, metadata in self.created:
                owned[kind][metadata["id"]] = metadata
            return owned

        def cleanup(self):
            return binding.cleanup_agent_project_binding(
                "project-test",
                self.owned(),
                on_removed=lambda kind, identity: self.removed.append((kind, identity)),
            )

        def mutations(self):
            return [argv for argv in self.calls if argv[2] in {"create", "delete"}]

    instance = Provider()
    monkeypatch.setattr(nebius, "_run_json", instance.run)
    monkeypatch.setattr(nebius, "_run", instance.run)
    return instance


def test_exact_project_creation_reuse_and_verified_cleanup(provider):
    result = provider.ensure()
    assert result["agent_iam_scope_id"] == "project-test"
    assert result["agent_iam_role"] == "editor"
    assert result["agent_iam_state"] == "created"
    assert [kind for kind, _ in provider.created] == list(binding.BINDING_KINDS)
    assert all(
        record["ownership_source"] == "provider-create-response"
        for _, record in provider.created
    )
    assert all("tenant-test" not in argv for argv in provider.mutations())
    count = len(provider.mutations())
    assert provider.ensure()["agent_iam_state"] == "existing"
    assert len(provider.mutations()) == count
    assert provider.cleanup() == ["agent_membership", "agent_permit", "agent_group"]
    assert not provider.resources["group"]
    assert len(provider.removed) == 3
    assert provider.cleanup() == ["agent_membership", "agent_permit", "agent_group"]


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("project", "parent_id", "tenant-other"),
        ("project", "id", "project-other"),
        ("service-account", "parent_id", "project-other"),
        ("service-account", "name", "another-account"),
        ("service-account", "id", "another-account"),
    ],
)
def test_identity_mismatch_blocks_every_mutation(provider, kind, field, value):
    row = next(iter(provider.resources[kind].values()))
    row["metadata"][field] = value
    with pytest.raises(nebius.NebiusError):
        provider.ensure()
    assert provider.mutations() == []


@pytest.mark.parametrize(
    "error", ["PermissionDenied", "PermissionDenied NotFound", "transport unavailable"]
)
def test_unreadable_group_never_falls_back_to_tenant(provider, error):
    def before(argv):
        if argv[2] == "get-by-name":
            raise nebius.NebiusError(error)

    provider.before = before
    with pytest.raises(nebius.NebiusError, match="no broader grant"):
        provider.ensure()
    assert provider.mutations() == []


@pytest.mark.parametrize("kind", ["access-permit", "group-membership"])
@pytest.mark.parametrize(
    "response",
    [
        {"error": "provider failure"},
        {"items": {}},
        {"items": [], "memberships": [], "next_page_token": 7},
    ],
)
def test_malformed_inventory_blocks_further_mutations(provider, kind, response):
    provider.ensure()
    provider.calls.clear()
    provider.transform = lambda argv, actual: (
        response if argv[1] == kind and argv[2].startswith("list") else actual
    )
    with pytest.raises(nebius.NebiusError):
        provider.ensure()
    assert provider.mutations() == []


def test_all_membership_pages_are_checked_before_reuse(provider):
    provider.ensure()
    first = deepcopy(provider.resources["group-membership"]["group-membership-test"])
    duplicate = deepcopy(first)
    duplicate["metadata"]["id"] = "membership-duplicate"

    def transform(argv, actual):
        if argv[2] == "list-members":
            if "--page-token" in argv:
                return {"memberships": [duplicate]}
            return {"memberships": [first], "next_page_token": "page-two"}
        return actual

    provider.transform = transform
    provider.calls.clear()
    with pytest.raises(nebius.NebiusError, match="duplicate memberships"):
        provider.ensure()
    assert provider.mutations() == []
    assert any("page-two" in argv for argv in provider.calls)


def test_repeating_page_token_fails_closed(provider):
    provider.ensure()
    provider.transform = lambda argv, actual: (
        {"memberships": [], "next_page_token": "repeat"}
        if argv[2] == "list-members"
        else actual
    )
    with pytest.raises(nebius.NebiusError, match="repeats a pagination"):
        provider.ensure()


@pytest.mark.parametrize(
    "spec",
    [
        {"resource_id": "tenant-test", "role": "editor"},
        {"resource_id": "project-test", "role": "admin"},
    ],
)
def test_unexpected_group_permission_blocks_membership(provider, spec):
    provider.ensure()
    provider.resources["access-permit"]["access-permit-test"]["spec"] = spec
    provider.resources["group-membership"].clear()
    provider.calls.clear()
    with pytest.raises(nebius.NebiusError, match="unexpected permission"):
        provider.ensure()
    assert provider.mutations() == []


def test_missing_permit_cannot_grant_other_group_members(provider):
    provider.ensure()
    provider.resources["access-permit"].clear()
    provider.resources["group-membership"]["group-membership-test"]["spec"][
        "member_id"
    ] = "account-unrelated"
    provider.calls.clear()
    with pytest.raises(nebius.NebiusError, match="other accounts"):
        provider.ensure()
    assert provider.mutations() == []


def test_creation_id_is_journaled_before_failed_scope_validation(provider):
    def transform(argv, response):
        if argv[1:3] == ["access-permit", "create"]:
            response["spec"]["resource_id"] = "project-other"
        return response

    provider.transform = transform
    with pytest.raises(nebius.NebiusError, match="different resource"):
        provider.ensure()
    assert [kind for kind, _ in provider.created] == ["agent_group", "agent_permit"]
    assert provider.resources["access-permit"]


def test_cleanup_never_removes_reused_objects(provider):
    provider.ensure()
    provider.created.clear()
    provider.ensure()
    provider.calls.clear()
    assert provider.cleanup() == []
    assert provider.mutations() == []


@pytest.mark.parametrize("extra_kind", ["group-membership", "access-permit"])
def test_cleanup_preserves_shared_group_and_permissions(provider, extra_kind):
    provider.ensure()
    spec = (
        {"member_id": "another-account"}
        if extra_kind == "group-membership"
        else {"resource_id": "project-test", "role": "viewer"}
    )
    provider.resources[extra_kind]["unowned-resource"] = resource(
        "unowned-resource", "group-test", **spec
    )
    with pytest.raises(nebius.NebiusError, match="partial"):
        provider.cleanup()
    assert provider.resources["group"]
    assert provider.resources["access-permit"]
    assert provider.removed == [("agent_membership", "group-membership-test")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "other-project"),
        ("ownership_source", "name-lookup"),
        ("created_by", "operator"),
        ("role", "admin"),
    ],
)
def test_cleanup_rejects_unproven_creation(provider, field, value):
    provider.ensure()
    provider.created[0][1][field] = value
    provider.calls.clear()
    with pytest.raises(nebius.NebiusError, match="provenance"):
        provider.cleanup()
    assert provider.mutations() == []


def test_changed_membership_identity_is_never_deleted(provider):
    provider.ensure()
    provider.resources["group-membership"]["group-membership-test"]["spec"][
        "member_id"
    ] = "another-account"
    provider.calls.clear()
    with pytest.raises(nebius.NebiusError, match="partial"):
        provider.cleanup()
    assert provider.mutations() == []
    assert provider.removed == []


def test_delete_requires_read_after_delete_absence(provider):
    provider.ensure()
    previous = deepcopy(provider.resources["group-membership"]["group-membership-test"])

    def transform(argv, actual):
        if argv[1:3] == ["group-membership", "delete"]:
            provider.resources["group-membership"]["group-membership-test"] = previous
        return actual

    provider.transform = transform
    with pytest.raises(nebius.NebiusError, match="partial"):
        provider.cleanup()
    assert not provider.removed


def test_bootstrap_wrong_tenant_never_creates_an_account(provider):
    provider.resources["service-account"].clear()
    with pytest.raises(nebius.NebiusError):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-other", "region-test", reuse_storage_credentials={}
        )
    assert provider.mutations() == []


def test_bootstrap_wrong_scope_account_is_retained_in_partial_journal(provider):
    from npa.cli import agent_iam

    provider.resources["service-account"].clear()

    def transform(argv, response):
        if argv[1:3] == ["service-account", "create"]:
            provider.resources["service-account"]["service-account-test"]["metadata"][
                "parent_id"
            ] = "project-other"
            response["metadata"]["parent_id"] = "project-other"
        return response

    provider.transform = transform
    with pytest.raises(nebius.NebiusError):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    assert not any(argv[2] == "delete" for argv in provider.mutations())
    assert agent_iam.agent_iam_owned("project-test", "service-account-test")
    data, _ = agent_iam._agent_iam_records()
    assert data["agent_iam"]["projects"]["project-test"]["status"] == "partial"


def test_bootstrap_account_delete_acknowledgement_cannot_erase_evidence(
    provider, monkeypatch
):
    from npa.cli import agent_iam

    provider.resources["service-account"].clear()

    def transform(argv, response):
        if argv[1:3] == ["service-account", "delete"]:
            provider.resources["service-account"]["service-account-test"] = resource(
                "service-account-test", "project-test", "npa-agent"
            )
        return response

    provider.transform = transform
    monkeypatch.setattr(
        nebius,
        "get_iam_token",
        lambda: (_ for _ in ()).throw(RuntimeError("injected post-grant failure")),
    )
    with pytest.raises(RuntimeError, match="post-grant"):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    assert agent_iam.agent_iam_owned("project-test", "service-account-test")
    data, _ = agent_iam._agent_iam_records()
    assert data["agent_iam"]["projects"]["project-test"]["status"] == "partial"
    assert not agent_iam.agent_iam_binding_resources("project-test")


@pytest.mark.parametrize(
    "document",
    [
        "agent_iam: malformed\n",
        "agent_iam: {projects: []}\n",
        "agent_iam: {projects: {project-test: {resources: malformed}}}\n",
    ],
)
def test_nested_malformed_journal_prevents_account_creation(provider, document):
    from npa.clients import credentials

    credentials.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials.CREDENTIALS_PATH.write_text(document)
    provider.resources["service-account"].clear()
    with pytest.raises(RuntimeError, match="malformed"):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    assert not provider.mutations()


def test_owned_absent_account_and_bindings_reconcile_entire_journal(
    provider, monkeypatch
):
    from npa.cli import agent_iam

    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    provider.ensure()
    for kind, record in provider.created:
        agent_iam.record_agent_iam_resource("project-test", kind, record)
    provider.resources["service-account"].clear()

    def query(argv):
        if argv[:3] == ["compute", "instance", "list"]:
            return {"items": []}
        return provider.run(argv)

    monkeypatch.setattr(nebius, "_run_json", query)
    removed = agent_iam.report_agent_iam(
        project_id="project-test",
        remaining_agents=0,
        purge=True,
        on_status=lambda message: None,
        strict=True,
    )
    assert len(removed) == 3
    data, _ = agent_iam._agent_iam_records()
    assert "agent_iam" not in data
    assert (
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
        == []
    )


def test_absent_named_account_does_not_prove_recorded_identity_absent(
    provider, monkeypatch
):
    from npa.cli import agent_iam

    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    monkeypatch.setattr(
        nebius, "get_service_account_id_by_name", lambda *args, **kwargs: None
    )
    with pytest.raises(agent_iam.AgentIAMCleanupError, match="unresolved"):
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
    assert agent_iam.agent_iam_owned("project-test", "account-test")
    assert not provider.mutations()


@pytest.mark.parametrize("fault", ["wrong-parent", "unverified-delete"])
def test_normal_teardown_requires_exact_account_scope_and_absence(
    provider, monkeypatch, fault
):
    from npa.cli import agent_iam
    from npa.lifecycle_intent import OperationIntent, operation_intent

    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    monkeypatch.setattr(
        nebius, "get_service_account_id_by_name", lambda *args, **kwargs: "account-test"
    )
    monkeypatch.setattr(
        nebius, "list_access_keys_for_service_account", lambda *args, **kwargs: []
    )

    def query(argv):
        if argv[:3] == ["compute", "instance", "list"]:
            return {"items": []}
        return provider.run(argv)

    monkeypatch.setattr(nebius, "_run_json", query)
    if fault == "wrong-parent":
        provider.resources["service-account"]["account-test"]["metadata"][
            "parent_id"
        ] = "project-other"
    else:

        def transform(argv, response):
            if argv[1:3] == ["service-account", "delete"]:
                provider.resources["service-account"]["account-test"] = resource(
                    "account-test", "project-test", "npa-agent"
                )
            return response

        provider.transform = transform
    with (
        operation_intent(OperationIntent.DESTROY),
        pytest.raises(agent_iam.AgentIAMCleanupError),
    ):
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
    assert agent_iam.agent_iam_owned("project-test", "account-test")
    if fault == "wrong-parent":
        assert not provider.mutations()


@pytest.mark.parametrize("key_absent", [False, True])
def test_absent_account_never_erases_unverified_project_key_receipts(
    provider, monkeypatch, key_absent
):
    from npa.cli import agent_iam

    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    agent_iam.record_agent_iam_resource(
        "project-test", "access_key", {"id": "key-test", "name": "npa-agent-key"}
    )
    provider.resources["service-account"].clear()

    def query(argv):
        if argv[:3] == ["compute", "instance", "list"]:
            return {"items": []}
        return provider.run(argv)

    monkeypatch.setattr(nebius, "_run_json", query)

    def key_scalar(key_id, field, **kwargs):
        assert key_id == "key-test" and field == "id"
        if key_absent:
            raise nebius.NebiusError("rpc error: code = NotFound")
        return key_id

    monkeypatch.setattr(nebius, "_access_key_metadata_scalar", key_scalar)
    if key_absent:
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
        assert not agent_iam.agent_iam_owned("project-test", "account-test")
        assert not agent_iam._recorded_access_keys("project-test")
    else:
        with pytest.raises(agent_iam.AgentIAMCleanupError, match="access-key receipts"):
            agent_iam.report_agent_iam(
                project_id="project-test",
                remaining_agents=0,
                purge=True,
                on_status=lambda message: None,
                strict=True,
            )
        assert agent_iam.agent_iam_owned("project-test", "account-test")
        assert agent_iam._recorded_access_keys("project-test")
    assert not provider.mutations()


def test_cleanup_allows_destroy_intent_but_refuses_observe(provider):
    from npa.lifecycle_intent import (
        OperationIntent,
        OperationIntentError,
        operation_intent,
    )

    provider.ensure()
    provider.calls.clear()
    with operation_intent(OperationIntent.OBSERVE), pytest.raises(OperationIntentError):
        provider.cleanup()
    assert not provider.mutations()
    with operation_intent(OperationIntent.DESTROY):
        assert len(provider.cleanup()) == 3


def test_provider_omitted_empty_repeated_fields_complete_entire_binding_lifecycle(
    provider,
):
    """ProtoJSON's successful empty response is {}, not a malformed inventory."""

    def omit_empty(argv, response):
        if argv[2] in {"list", "list-members"}:
            return {key: value for key, value in response.items() if value != []}
        return response

    provider.transform = omit_empty
    assert provider.ensure()["agent_iam_state"] == "created"
    assert provider.ensure()["agent_iam_state"] == "existing"
    assert provider.cleanup() == ["agent_membership", "agent_permit", "agent_group"]
    assert not any(
        provider.resources[kind]
        for kind in ("group", "access-permit", "group-membership")
    )


@pytest.mark.parametrize(
    "resource_type,field",
    [("access-permit", "items"), ("group-membership", "memberships")],
)
def test_omitted_empty_iam_page_still_follows_nonempty_token(
    provider, resource_type, field
):
    seen = []

    def response(argv):
        seen.append(argv)
        if "--page-token" not in argv:
            return {"next_page_token": "second-page"}
        assert argv[-1] == "second-page"
        return {}

    provider.before = lambda argv: None
    from unittest.mock import patch

    with patch.object(nebius, "_run_json", side_effect=response):
        assert binding._inventory(resource_type, "group-test") == []
    assert len(seen) == 2


@pytest.mark.parametrize(
    "resource_type,field",
    [("access-permit", "items"), ("group-membership", "memberships")],
)
@pytest.mark.parametrize(
    "shape", ["null-array", "null-token", "unknown", "error", "wrong-array-key"]
)
def test_omitted_empty_compatibility_keeps_malformed_iam_responses_rejected(
    provider, resource_type, field, shape
):
    responses = {
        "null-array": {field: None},
        "null-token": {"next_page_token": None},
        "unknown": {"unexpected": []},
        "error": {"error": "denied"},
        "wrong-array-key": {"items" if field == "memberships" else "memberships": []},
    }
    from unittest.mock import patch

    with (
        patch.object(nebius, "_run_json", return_value=responses[shape]),
        pytest.raises(nebius.NebiusError),
    ):
        binding._inventory(resource_type, "group-test")
