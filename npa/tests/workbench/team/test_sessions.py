"""Prove saved login isolation, online verification, and private atomic persistence."""

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from npa.workbench.team import session_storage, sessions
from npa.workbench.team.errors import AuthenticationError, TeamError
from npa.workbench.team.sessions import (
    SavedSession,
    SessionStore,
    login_session,
    resolve_session,
    verify_session_identity,
)


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "connections")


@pytest.fixture
def session():
    return SavedSession(endpoint="https://team.example.test")


@pytest.fixture
def access():
    return {
        "subject": "local-account",
        "workspaces": [
            {
                "name": "robotics",
                "role": "runner",
                "allocated": True,
                "clusters": {"east": 1},
            }
        ],
    }


def _transport(access, status=200):
    def respond(request):
        assert request.url.path == "/v1/me"
        assert request.headers["authorization"].startswith("Bearer ")
        return httpx.Response(status, json=access)

    return httpx.MockTransport(respond)


def _saved(session):
    return session.model_copy(update={"verified_subject": "local-account"})


def test_local_login_verifies_and_saves_unique_placement(store, session, access):
    saved, actual = login_session(
        session, "local-test-key", store=store, transport=_transport(access)
    )
    assert actual == access
    assert (saved.workspace, saved.cluster, saved.verified_subject) == (
        "robotics",
        "east",
        "local-account",
    )
    assert store.load() == saved
    assert store.token() == "local-test-key"
    assert "local-test-key" not in repr(saved)
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in store.root.iterdir()
    )


def test_failed_authentication_never_creates_or_replaces_saved_key(
    store, session, access
):
    with pytest.raises(AuthenticationError):
        login_session(
            session, "rejected-key", store=store, transport=_transport({}, 401)
        )
    assert not store.root.exists()
    login_session(session, "working-key", store=store, transport=_transport(access))
    before = (store.root / "default.json").read_bytes()
    with pytest.raises(AuthenticationError):
        login_session(
            session, "rejected-key", store=store, transport=_transport({}, 401)
        )
    assert (store.root / "default.json").read_bytes() == before


def test_native_login_never_saves_token_and_refreshes_selected_profile(store, access):
    session = SavedSession(
        endpoint="https://team.example.test",
        auth_mode="nebius",
        nebius_profile="human-work",
    )
    saved, _ = login_session(
        session, "native-private-token", store=store, transport=_transport(access)
    )
    assert not any(
        "native-private-token" in path.read_text() for path in store.root.iterdir()
    )
    assert "token" not in json.loads((store.root / "default.json").read_text())
    calls = []

    def fresh(profile):
        calls.append(profile)
        return f"fresh-{len(calls)}"

    assert resolve_session(store=store, token_supplier=fresh) == (saved, "fresh-1")
    assert resolve_session(store=store, token_supplier=fresh) == (saved, "fresh-2")
    assert calls == ["human-work", "human-work"]
    with pytest.raises(TeamError, match="fresh token"):
        store.token()
    with pytest.raises(TeamError, match="never be saved"):
        store.save("bad", saved, token="native-private-token")


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"workspace": "unknown"}, "workspace is not authorized"),
        ({"cluster": "unknown"}, "select a workspace"),
        ({"workspace": "robotics", "cluster": "west"}, "cluster is not allocated"),
    ],
)
def test_unauthorized_defaults_do_not_persist(
    store, session, access, changes, expected
):
    with pytest.raises(TeamError, match=expected):
        login_session(
            session.model_copy(update=changes),
            "test-key",
            store=store,
            transport=_transport(access),
        )
    assert not store.root.exists()


def test_multiple_allocations_need_selection_without_blocking_login(
    store, session, access
):
    access["workspaces"].append(
        {
            "name": "perception",
            "role": "runner",
            "allocated": True,
            "clusters": {"west": 1},
        }
    )
    saved, _ = login_session(session, "key", store=store, transport=_transport(access))
    assert saved.workspace is None and saved.cluster is None
    selected = session.model_copy(update={"workspace": "perception"})
    saved, _ = login_session(selected, "key", store=store, transport=_transport(access))
    assert (saved.workspace, saved.cluster) == ("perception", "west")


def test_single_workspace_multiple_clusters_does_not_guess_cluster(
    store, session, access
):
    access["workspaces"][0]["clusters"]["west"] = 1
    saved, _ = login_session(session, "key", store=store, transport=_transport(access))
    assert saved.workspace == "robotics" and saved.cluster is None


@pytest.mark.parametrize(
    "workspaces",
    [[], [{"name": "robotics", "role": "reader", "allocated": False, "clusters": {}}]],
)
def test_readers_and_unassigned_people_can_login(store, session, access, workspaces):
    access["workspaces"] = workspaces
    saved, _ = login_session(session, "key", store=store, transport=_transport(access))
    assert saved.verified_subject == "local-account"
    assert saved.workspace is None and saved.cluster is None


def test_explicit_cluster_selects_only_unambiguous_workspace(store, session, access):
    access["workspaces"].append(
        {
            "name": "perception",
            "role": "runner",
            "allocated": True,
            "clusters": {"west": 1},
        }
    )
    selected = session.model_copy(update={"cluster": "west"})
    saved, _ = login_session(selected, "key", store=store, transport=_transport(access))
    assert saved.workspace == "perception" and saved.cluster == "west"
    access["workspaces"][0]["clusters"]["west"] = 1
    with pytest.raises(TeamError, match="select a workspace"):
        login_session(selected, "key", store=store, transport=_transport(access))


def test_endpoint_override_cannot_forward_saved_credentials(store, session):
    store.save("default", _saved(session), token="private-key")
    with pytest.raises(TeamError, match="endpoint differs"):
        resolve_session(endpoint="https://other.example.test", store=store)
    assert (
        resolve_session(endpoint=session.endpoint + "/", store=store)[1]
        == "private-key"
    )
    explicit, token = resolve_session(
        endpoint="https://other.example.test", token="other-key", store=store
    )
    assert explicit.endpoint == "https://other.example.test" and token == "other-key"
    assert explicit.workspace is None and explicit.verified_subject is None


def test_endpoint_override_rejects_before_native_token_supplier(store):
    session = SavedSession(
        endpoint="https://team.example.test",
        auth_mode="nebius",
        nebius_profile="human",
        verified_subject="account",
    )
    store.save("default", session)

    def forbidden(profile):
        pytest.fail("must not request a token for another endpoint")

    with pytest.raises(TeamError, match="endpoint differs"):
        resolve_session(
            endpoint="https://other.example.test", token_supplier=forbidden, store=store
        )


def test_multiple_profiles_and_logout_are_isolated(store, session):
    store.save("research", _saved(session), token="research-key")
    other = _saved(session).model_copy(
        update={"endpoint": "https://other.example.test"}
    )
    store.save("other", other, token="other-key")
    assert resolve_session(store=store) == (other, "other-key")
    assert resolve_session(profile="research", store=store) == (
        _saved(session),
        "research-key",
    )
    assert store.remove("research") is True
    assert store.token() == "other-key"
    assert store.remove() is True
    assert store.exists() is False
    assert store.remove("research") is False


def test_connection_and_key_are_read_from_one_profile_snapshot(
    store, session, monkeypatch
):
    store.save("alpha", _saved(session), token="alpha-key")
    other = _saved(session).model_copy(
        update={"endpoint": "https://other.example.test"}
    )
    store.save("beta", other, token="beta-key", make_default=False)
    original = sessions.read_private_json

    def switch_after_read(root, name):
        result = original(root, name)
        if name == "current.json":
            session_storage.write_private_json(root, name, {"profile": "beta"})
        return result

    monkeypatch.setattr(sessions, "read_private_json", switch_after_read)
    assert resolve_session(store=store) == (_saved(session), "alpha-key")
    assert resolve_session(store=store) == (other, "beta-key")


def test_atomic_write_failure_keeps_previous_connection(store, session, monkeypatch):
    store.save("default", _saved(session), token="old-key")
    original = session_storage.os.replace

    def fail_profile(source, destination, **kwargs):
        if destination == "default.json":
            raise OSError("synthetic disk failure")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(session_storage.os, "replace", fail_profile)
    with pytest.raises(TeamError, match="saved safely"):
        store.save("default", _saved(session), token="new-key")
    assert store.token() == "old-key"
    assert sorted(path.name for path in store.root.iterdir()) == [
        "current.json",
        "default.json",
        "profiles.lock",
    ]


@pytest.mark.parametrize("target", ["directory", "record"])
def test_session_rejects_insecure_permissions(store, session, target):
    store.save("default", _saved(session), token="key")
    path = store.root if target == "directory" else store.root / "default.json"
    path.chmod(0o755 if target == "directory" else 0o640)
    with pytest.raises(TeamError, match="mode"):
        store.exists()


@pytest.mark.parametrize("target", ["directory", "record"])
def test_session_rejects_symlink_paths(store, session, tmp_path, target):
    store.save("default", _saved(session), token="key")
    path = store.root if target == "directory" else store.root / "default.json"
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=target == "directory")
    with pytest.raises(TeamError, match="read safely"):
        store.load()


def test_hardlinked_credential_is_rejected(store, session, tmp_path):
    store.save("default", _saved(session), token="key")
    os.link(store.root / "default.json", tmp_path / "copy")
    with pytest.raises(TeamError, match="private mode-0600"):
        store.load()


def test_fifo_record_is_rejected_without_blocking(store, session):
    store.save("default", _saved(session), token="key")
    path = store.root / "default.json"
    path.unlink()
    os.mkfifo(path, 0o600)
    with pytest.raises(TeamError, match="private mode-0600"):
        store.load()


def test_atomic_save_does_not_write_through_destination_symlink(
    store, session, tmp_path
):
    store.save("default", _saved(session), token="key")
    outside = tmp_path / "untouched"
    outside.write_text("original")
    destination = store.root / "default.json"
    destination.unlink()
    destination.symlink_to(outside)
    store.save("default", _saved(session), token="replacement")
    assert outside.read_text() == "original"
    assert not destination.is_symlink()
    assert store.token() == "replacement"


@pytest.mark.parametrize(
    "profile",
    ["../outside", "/absolute", "has.dot", "", "current", "Current", "--option"],
)
def test_invalid_profile_cannot_create_files(store, session, profile):
    with pytest.raises(TeamError, match="profile"):
        store.save(profile, _saved(session), token="key")
    assert not store.root.exists()


@pytest.mark.parametrize(
    "payload", [[], {}, {"subject": ""}, {"subject": "account", "workspaces": [None]}]
)
def test_malformed_identity_response_is_not_saved(store, session, payload):
    with pytest.raises(TeamError):
        login_session(session, "key", store=store, transport=_transport(payload))
    assert not store.root.exists()


def test_missing_connection_differs_from_corrupt_saved_state(store, session):
    assert store.exists() is False
    store.save("default", _saved(session), token="key")
    (store.root / "default.json").write_text("not json")
    with pytest.raises(TeamError, match="read safely"):
        store.exists()


def test_refreshed_identity_must_match_saved_account(session):
    verify_session_identity(_saved(session), {"subject": "local-account"})
    with pytest.raises(AuthenticationError, match="account changed"):
        verify_session_identity(_saved(session), {"subject": "other-account"})


def test_store_respects_configuration_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "custom"))
    assert SessionStore().root == tmp_path / "custom" / "team"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://team.example.test",
        "https://user:secret@team.example.test",
        "https://team.example.test?token=secret",
        "https://team.example.test/#fragment",
        "https://team.example.test\\@other.example.test",
        "https://team.example.test\n",
    ],
)
def test_invalid_endpoints_are_rejected(endpoint):
    with pytest.raises(ValueError):
        SavedSession(endpoint=endpoint)


def test_ca_file_is_saved_as_absolute_path(session, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configured = SavedSession(endpoint=session.endpoint, ca_file="gateway-ca.pem")
    assert configured.ca_file == str(tmp_path / "gateway-ca.pem")


def test_native_session_refresh_uses_its_isolated_configuration(store, tmp_path):
    path = tmp_path / "human-config.yaml"
    session = SavedSession(
        endpoint="https://team.example.test",
        auth_mode="nebius",
        nebius_profile="human",
        nebius_config=str(path),
        verified_subject="account",
    )
    store.save("human", session)
    calls = []

    def fresh(profile, *, config_file):
        calls.append((profile, config_file))
        return "fresh-token"

    assert resolve_session(store=store, token_supplier=fresh) == (
        session,
        "fresh-token",
    )
    assert calls == [("human", path)]


@pytest.mark.parametrize(
    "settings",
    [
        {"auth_mode": "nebius"},
        {"nebius_profile": "human"},
        {"nebius_config": "human-config.yaml"},
        {"auth_mode": "nebius", "nebius_profile": "--option"},
        {"auth_mode": "nebius", "nebius_profile": "human\nother"},
    ],
)
def test_native_settings_are_explicit_and_cannot_change_local_auth(settings):
    with pytest.raises(ValueError):
        SavedSession(endpoint="https://team.example.test", **settings)


def test_private_ca_is_forwarded_to_authenticated_client(
    store, session, access, tmp_path, monkeypatch
):
    certificate = str(tmp_path / "ca.pem")
    closed = []

    class Client:
        def __init__(self, endpoint, token, *, ca_file, transport):
            assert endpoint == session.endpoint and token == "key"
            assert ca_file == certificate

        def whoami(self):
            return access

        def close(self):
            closed.append(True)

    monkeypatch.setattr(sessions, "TeamClient", Client)
    requested = SavedSession(endpoint=session.endpoint, ca_file=certificate)
    login_session(requested, "key", store=store)
    assert store.load().ca_file == certificate
    assert closed == [True]


def test_logout_does_not_silently_select_an_older_default_account(store, session):
    store.save("default", _saved(session), token="old-key")
    other = _saved(session).model_copy(update={"verified_subject": "other-account"})
    store.save("research", other, token="research-key")
    assert store.remove() is True
    assert store.exists() is False
    assert store.remove() is False
    with pytest.raises(AuthenticationError, match="signed out"):
        resolve_session(store=store)
    assert store.load("default") == _saved(session)
    assert store.selected_profile() == "default"
    store.save("default", _saved(session), token="new-key")
    assert resolve_session(store=store) == (_saved(session), "new-key")


def test_concurrent_login_and_logout_preserve_the_later_login(
    store, session, monkeypatch
):
    store.save("default", _saved(session), token="old-key")
    closing, release, later_written = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = sessions.write_private_json

    def intercept(root, name, document):
        if name == "current.json" and document == {"profile": None}:
            closing.set()
            assert release.wait(5)
        if name == "later.json":
            later_written.set()
        return original(root, name, document)

    monkeypatch.setattr(sessions, "write_private_json", intercept)
    with ThreadPoolExecutor(max_workers=2) as executor:
        logout = executor.submit(store.remove)
        assert closing.wait(5)
        login = executor.submit(store.save, "later", _saved(session), token="later-key")
        try:
            assert not later_written.wait(0.1)
        finally:
            release.set()
        assert logout.result(timeout=5) is True
        login.result(timeout=5)
    assert store.token() == "later-key"
    assert store.selected_profile() == "later"
