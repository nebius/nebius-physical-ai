"""Prove interrupted and concurrent submissions retain one private retry identity."""

import json
import threading
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from npa.workbench.team import submission_receipts
from npa.workbench.team.errors import AuthenticationError, TeamError
from npa.workbench.team.models import SubmitRequest
from npa.workbench.team.sessions import SavedSession, SessionStore
from npa.workbench.team.submission_receipts import submit_saved


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "team")


@pytest.fixture
def session():
    return SavedSession(
        endpoint="https://team.example.test", verified_subject="person-a"
    )


@pytest.fixture
def submission(workflow):
    return SubmitRequest(
        workspace="robotics",
        cluster="east",
        workflow=workflow,
        idempotency_key="automatic-placeholder",
    )


class Client:
    def __init__(self, *, subject="person-a", on_submit=None):
        self.subject = subject
        self.on_submit = on_submit
        self.calls = []

    def whoami(self):
        return {"subject": self.subject, "workspaces": []}

    def submit(self, submission):
        self.calls.append(submission)
        if self.on_submit:
            return self.on_submit(submission)
        return {"id": "run-" + "1" * 32, "status": "accepted"}


def _records(store):
    return [
        json.loads(path.read_text())
        for path in (store.root / "requests").glob("*.json")
    ]


def test_key_is_durable_before_first_network_submission(store, session, submission):
    def submit(actual):
        assert _records(store) == [
            {"key": actual.idempotency_key, "acknowledged": False}
        ]
        assert actual.idempotency_key != submission.idempotency_key
        return {"id": "run-" + "1" * 32, "status": "accepted"}

    result = submit_saved(Client(on_submit=submit), session, submission, store=store)
    assert result["submission"]["reused"] is False
    assert _records(store) == [
        {"key": result["submission"]["idempotency_key"], "acknowledged": True}
    ]
    assert submission.idempotency_key == "automatic-placeholder"


def test_failure_retry_and_successful_repeat_use_same_key(store, session, submission):
    def fail(actual):
        raise TeamError("synthetic transport interruption")

    failed = Client(on_submit=fail)
    with pytest.raises(TeamError, match="transport interruption"):
        submit_saved(failed, session, submission, store=store)
    key = failed.calls[0].idempotency_key
    assert _records(store) == [{"key": key, "acknowledged": False}]
    successful = Client()
    first = submit_saved(successful, session, submission, store=store)
    second = submit_saved(successful, session, submission, store=store)
    assert (
        first["submission"]
        == second["submission"]
        == {"idempotency_key": key, "reused": True}
    )
    assert [item.idempotency_key for item in successful.calls] == [key, key]


def test_new_run_replaces_only_acknowledged_previous_identity(
    store, session, submission
):
    client = Client()
    first = submit_saved(client, session, submission, store=store)
    second = submit_saved(client, session, submission, store=store, new_run=True)
    assert (
        first["submission"]["idempotency_key"]
        != second["submission"]["idempotency_key"]
    )
    assert second["submission"]["reused"] is False
    third = submit_saved(client, session, submission, store=store)
    assert (
        third["submission"]["idempotency_key"]
        == second["submission"]["idempotency_key"]
    )
    assert len(_records(store)) == 1


def test_unconfirmed_submission_refuses_new_run_without_post(
    store, session, submission
):
    def fail(actual):
        raise TeamError("synthetic interruption")

    client = Client(on_submit=fail)
    with pytest.raises(TeamError, match="interruption"):
        submit_saved(client, session, submission, store=store)
    before = _records(store)
    with pytest.raises(TeamError, match="unconfirmed"):
        submit_saved(client, session, submission, store=store, new_run=True)
    assert len(client.calls) == 1
    assert _records(store) == before


@pytest.mark.parametrize(
    "changed", ["endpoint", "person", "workspace", "cluster", "workflow"]
)
def test_receipt_scope_includes_exact_document_person_and_endpoint(
    store, session, submission, changed
):
    first = submit_saved(Client(), session, submission, store=store)
    client = Client()
    if changed == "endpoint":
        session = session.model_copy(update={"endpoint": "https://other.example.test"})
    elif changed == "person":
        session = session.model_copy(update={"verified_subject": "person-b"})
        client = Client(subject="person-b")
    elif changed == "workflow":
        submission = submission.model_copy(
            update={
                "workflow": {**submission.workflow, "metadata": {"name": "changed"}}
            }
        )
    else:
        submission = submission.model_copy(update={changed: "alternative"})
    second = submit_saved(client, session, submission, store=store)
    assert (
        first["submission"]["idempotency_key"]
        != second["submission"]["idempotency_key"]
    )
    assert len(_records(store)) == 2


def test_mapping_order_and_placeholder_do_not_change_retry_identity(
    store, session, submission
):
    first = submit_saved(Client(), session, submission, store=store)
    reordered = dict(reversed(list(submission.workflow.items())))
    changed = submission.model_copy(
        update={"workflow": reordered, "idempotency_key": "other-placeholder"}
    )
    second = submit_saved(Client(), session, changed, store=store)
    assert second["submission"] == {**first["submission"], "reused": True}


def test_explicit_key_bypasses_automatic_receipts(store, session, submission):
    class ExplicitClient(Client):
        def whoami(self):
            pytest.fail("explicit submission must retain its previous API contract")

    client = ExplicitClient()
    explicit = submission.model_copy(update={"idempotency_key": "caller-retained-key"})
    result = submit_saved(client, session, explicit, store=store, automatic=False)
    assert client.calls == [explicit]
    assert "submission" not in result
    assert not store.root.exists()
    with pytest.raises(TeamError, match="not both"):
        submit_saved(
            client, session, explicit, store=store, automatic=False, new_run=True
        )
    assert len(client.calls) == 1


def test_write_failure_prevents_network_submission(
    store, session, submission, monkeypatch
):
    def fail(*args):
        raise TeamError("synthetic persistence failure")

    monkeypatch.setattr(submission_receipts, "write_private_json", fail)
    client = Client()
    with pytest.raises(TeamError, match="persistence failure"):
        submit_saved(client, session, submission, store=store)
    assert client.calls == []


def test_acknowledgement_write_failure_still_retries_original_key(
    store, session, submission, monkeypatch
):
    original = submission_receipts.write_private_json

    def fail_acknowledgement(root, name, document):
        if document["acknowledged"]:
            raise TeamError("synthetic acknowledgement failure")
        return original(root, name, document)

    monkeypatch.setattr(submission_receipts, "write_private_json", fail_acknowledgement)
    client = Client()
    with pytest.raises(TeamError, match="acknowledgement failure"):
        submit_saved(client, session, submission, store=store)
    key = client.calls[0].idempotency_key
    assert _records(store) == [{"key": key, "acknowledged": False}]
    monkeypatch.setattr(submission_receipts, "write_private_json", original)
    result = submit_saved(client, session, submission, store=store)
    assert result["submission"] == {"idempotency_key": key, "reused": True}


def test_identity_change_prevents_receipts_or_posts(store, session, submission):
    client = Client(subject="different-person")
    with pytest.raises(AuthenticationError, match="account changed"):
        submit_saved(client, session, submission, store=store)
    assert client.calls == []
    assert not store.root.exists()


def test_receipts_do_not_persist_workflow_secrets_or_account_details(
    store, session, submission
):
    workflow = {
        **submission.workflow,
        "config": {"private_example": "synthetic-sensitive-workflow-value"},
    }
    document = submission.model_copy(update={"workflow": workflow})
    submit_saved(Client(), session, document, store=store)
    files = list((store.root / "requests").iterdir())
    assert files
    for path in files:
        content = path.read_text()
        for value in (
            "synthetic-sensitive-workflow-value",
            session.endpoint,
            session.verified_subject,
        ):
            assert value not in content and value not in path.name
    assert set(_records(store)[0]) == {"key", "acknowledged"}


def _notifying_client(authenticated, posted):
    class SecondClient(Client):
        def whoami(self):
            authenticated.set()
            return super().whoami()

        def submit(self, actual):
            posted.set()
            return super().submit(actual)

    return SecondClient()


def test_concurrent_retries_wait_for_first_post_and_reuse_its_key(
    store, session, submission
):
    posting, release = threading.Event(), threading.Event()
    second_authenticated, second_posted = threading.Event(), threading.Event()

    def first_submit(actual):
        posting.set()
        assert release.wait(5)
        return {"id": "run-" + "1" * 32}

    second_client = _notifying_client(second_authenticated, second_posted)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            submit_saved,
            Client(on_submit=first_submit),
            session,
            submission,
            store=store,
        )
        assert posting.wait(5)
        second = executor.submit(
            submit_saved, second_client, session, submission, store=store
        )
        try:
            assert second_authenticated.wait(5)
            assert not second_posted.wait(0.1)
        finally:
            release.set()
        first_result, second_result = first.result(timeout=5), second.result(timeout=5)
    assert (
        first_result["submission"]["idempotency_key"]
        == second_result["submission"]["idempotency_key"]
    )
    assert second_result["submission"]["reused"] is True


@pytest.mark.parametrize(
    "content",
    [{}, {"key": "", "acknowledged": True}, {"key": "key", "acknowledged": "yes"}],
)
def test_corrupt_receipt_does_not_silently_start_another_run(
    store, session, submission, content
):
    client = Client()
    submit_saved(client, session, submission, store=store)
    path = next((store.root / "requests").glob("*.json"))
    path.write_text(json.dumps(content))
    with pytest.raises(TeamError, match="invalid submission receipt"):
        submit_saved(client, session, submission, store=store)
    assert len(client.calls) == 1


def test_explicit_submission_before_login_leaves_private_usable_session_directory(
    store, session, submission
):
    submit_saved(Client(), session, submission, store=store)
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    store.save("default", session, token="personal-key")
    assert store.load() == session
    assert store.token() == "personal-key"
