"""Exercise first-time and returning Workbench login through the actual CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import certifi
import httpx
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.team import nebius_login, sessions
from npa.workbench.team.client import TeamClient
from npa.workbench.team.errors import AuthenticationError
from npa.workbench.team.sessions import SessionStore

runner = CliRunner()
_ENDPOINT = "https://team.example.test"
_KEY = "private-personal-key"
_NATIVE_TOKEN = "private-native-human-token"


@pytest.fixture
def connection(monkeypatch):
    state = SimpleNamespace(requests=[], clients=[], status=200)
    state.access = {
        "subject": "workbench-user",
        "workspaces": [
            {
                "name": "robotics",
                "role": "runner",
                "allocated": True,
                "clusters": {"east": 1},
                "gpu_limit": 1,
            }
        ],
    }

    def respond(request):
        state.requests.append(request)
        assert request.method == "GET" and request.url.path == "/v1/me"
        return httpx.Response(state.status, json=state.access)

    def client(endpoint, token, **options):
        state.clients.append((endpoint, options.get("ca_file")))
        options["transport"] = httpx.MockTransport(respond)
        return TeamClient(endpoint, token, **options)

    monkeypatch.setattr(sessions, "TeamClient", client)
    return state


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "personal.key"
    path.write_text(_KEY + "\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def terminal(monkeypatch):
    testing_module = sys.modules[CliRunner.isolation.__module__]
    monkeypatch.setattr(testing_module._NamedTextIOWrapper, "isatty", lambda self: True)


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(
        calls=[],
        created=[],
        profiles=[
            nebius_login.NebiusProfile("human-work", "federation"),
            nebius_login.NebiusProfile("operator", "service account", True),
        ],
    )

    def login(profile, **options):
        state.calls.append((profile, options))
        return _NATIVE_TOKEN

    def create(profile, config_file):
        state.created.append((profile, config_file))
        config_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        config_file.write_text("official-profile-configuration")
        config_file.chmod(0o600)

    monkeypatch.setattr(nebius_login, "list_profiles", lambda **kwargs: state.profiles)
    monkeypatch.setattr(nebius_login, "login_token", login)
    monkeypatch.setattr(nebius_login, "create_human_profile", create)
    return state


def _assert_private_session(profile="default"):
    store = SessionStore()
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert (store.root / f"{profile}.json").stat().st_mode & 0o777 == 0o600
    return store


def _key_login(key_file, *options):
    return runner.invoke(
        app,
        [
            "login",
            "--endpoint",
            _ENDPOINT,
            "--token-file",
            str(key_file),
            *options,
        ],
    )


def test_top_level_login_and_logout_help_are_discoverable():
    result = runner.invoke(app, ["login", "--help"])
    assert result.exit_code == 0
    assert "--nebius" in result.output and "--token-file" in result.output
    assert runner.invoke(app, ["logout", "--help"]).exit_code == 0


def test_local_key_login_verifies_before_saving_private_connection(
    connection, key_file
):
    result = _key_login(key_file, "--output-format", "json")
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert (document["profile"], document["authentication"]) == ("default", "local")
    assert (document["workspace"], document["cluster"]) == ("robotics", "east")
    assert _KEY not in result.output
    store = _assert_private_session()
    assert store.token() == _KEY
    assert len(connection.requests) == 1
    assert connection.requests[0].headers["Authorization"] == f"Bearer {_KEY}"


def test_no_arguments_guides_first_local_key_login(connection, key_file, terminal):
    result = runner.invoke(app, ["login"], input=f"{_ENDPOINT}\nkey\n{key_file}\n")
    assert result.exit_code == 0, result.output
    assert "Workbench HTTPS endpoint" in result.output
    assert "Sign-in method (key or nebius)" in result.output
    assert "Personal key file" in result.output
    assert _assert_private_session().token() == _KEY
    assert _KEY not in result.output


def test_no_arguments_guides_nebius_login_without_key_files(
    connection, native, terminal
):
    result = runner.invoke(app, ["login"], input=f"{_ENDPOINT}\nnebius\n")
    assert result.exit_code == 0, result.output
    assert native.calls[0][0] == "human-work"
    assert native.calls[0][1]["no_browser"] is False
    saved = _assert_private_session().load()
    assert saved.auth_mode == "nebius" and saved.nebius_profile == "human-work"
    assert _NATIVE_TOKEN not in result.output
    assert all(
        _NATIVE_TOKEN not in p.read_text() for p in SessionStore().root.iterdir()
    )


def test_first_nebius_login_creates_isolated_profile_and_keeps_browser_default(
    connection, native
):
    native.profiles = [nebius_login.NebiusProfile("operator", "service account", True)]
    result = runner.invoke(app, ["login", "--endpoint", _ENDPOINT, "--nebius"])
    assert result.exit_code == 0, result.output
    assert len(native.created) == 1
    name, config_file = native.created[0]
    assert name == "workbench" and config_file.parent == SessionStore().root
    assert native.calls[0][0] == "workbench"
    assert native.calls[0][1]["config_file"] == config_file
    assert native.calls[0][1]["no_browser"] is False
    assert native.profiles[0].active
    assert _assert_private_session().load().nebius_config == str(config_file)


def test_new_explicit_nebius_config_is_used(connection, native, tmp_path):
    native.profiles = []
    config_file = tmp_path / "private" / "nebius.yaml"
    result = runner.invoke(
        app,
        ["login", "--endpoint", _ENDPOINT, "--nebius-config", str(config_file)],
    )
    assert result.exit_code == 0, result.output
    assert native.created == [("workbench", config_file)]
    assert native.calls[0][1]["config_file"] == config_file
    assert _assert_private_session().load().nebius_config == str(config_file)


def test_nebius_remote_options_reach_official_provider(connection, native):
    result = runner.invoke(
        app,
        [
            "login",
            "--endpoint",
            _ENDPOINT,
            "--nebius-profile",
            "human-work",
            "--no-browser",
            "--ssh-host",
            "dev-vm",
        ],
    )
    assert result.exit_code == 0, result.output
    assert native.calls[0][1]["no_browser"] is True
    assert native.calls[0][1]["ssh_host"] == "dev-vm"


def test_multiple_human_profiles_prompt_for_explicit_choice(
    connection, native, terminal
):
    native.profiles.append(nebius_login.NebiusProfile("human-second", "federation"))
    result = runner.invoke(
        app, ["login", "--endpoint", _ENDPOINT, "--nebius"], input="human-second\n"
    )
    assert result.exit_code == 0, result.output
    assert native.calls[0][0] == "human-second"
    assert not native.created


def test_no_argument_relogin_retains_named_connection_and_ca(connection, key_file):
    ca_file = certifi.where()
    result = _key_login(key_file, "--profile", "research", "--ca-file", ca_file)
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["login", "--output-format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["profile"] == "research"
    store = _assert_private_session("research")
    assert store.selected_profile() == "research"
    assert store.load().ca_file == str(Path(ca_file).resolve())
    assert connection.clients[-1] == (_ENDPOINT, str(Path(ca_file).resolve()))
    assert not (store.root / "default.json").exists()


def test_no_argument_nebius_relogin_retains_provider_config_and_ca(connection, native):
    native.profiles = []
    result = runner.invoke(
        app,
        [
            "login",
            "--endpoint",
            _ENDPOINT,
            "--nebius",
            "--profile",
            "research",
            "--ca-file",
            certifi.where(),
        ],
    )
    assert result.exit_code == 0, result.output
    before = SessionStore().load()
    result = runner.invoke(app, ["login", "--output-format", "json"])
    assert result.exit_code == 0, result.output
    assert SessionStore().load() == before
    assert len(native.created) == 1
    assert native.calls[-1][0] == before.nebius_profile
    assert str(native.calls[-1][1]["config_file"]) == before.nebius_config


def test_logout_forgets_only_selected_connection_and_keeps_jobs_untouched(
    connection, key_file
):
    assert _key_login(key_file, "--profile", "first").exit_code == 0
    assert _key_login(key_file, "--profile", "second").exit_code == 0
    result = runner.invoke(app, ["logout", "--output-format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"logged_out": True}
    assert SessionStore().exists("first")
    assert not SessionStore().exists("second")
    assert key_file.read_text().strip() == _KEY
    assert len(connection.requests) == 2


def test_denied_login_never_persists_or_replaces_working_credentials(
    connection, key_file
):
    connection.status = 401
    result = _key_login(key_file, "--output-format", "json")
    assert result.exit_code != 0
    assert "npa login" in str(result.exception)
    assert not SessionStore().root.exists()
    connection.status = 200
    assert _key_login(key_file).exit_code == 0
    before = (SessionStore().root / "default.json").read_bytes()
    connection.status = 401
    result = _key_login(key_file, "--output-format", "json")
    assert result.exit_code != 0
    assert (SessionStore().root / "default.json").read_bytes() == before
    assert _KEY not in result.output


def test_failed_human_sign_in_never_saves_connection(connection, native, monkeypatch):
    def failed(*args, **kwargs):
        raise AuthenticationError("Human sign-in was cancelled")

    monkeypatch.setattr(nebius_login, "login_token", failed)
    result = runner.invoke(
        app,
        [
            "login",
            "--endpoint",
            _ENDPOINT,
            "--nebius",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code != 0
    assert not SessionStore().root.exists()
    assert not connection.requests


def test_noninteractive_missing_options_fail_with_actionable_instruction(connection):
    result = runner.invoke(app, ["login", "--output-format", "json"])
    assert result.exit_code != 0
    assert "--endpoint" in str(result.exception)
    assert not connection.requests


def test_conflicting_authentication_and_unauthorized_placement_do_not_save(
    connection, key_file, native
):
    result = _key_login(key_file, "--nebius", "--output-format", "json")
    assert result.exit_code != 0
    assert not connection.requests and not native.calls
    result = _key_login(
        key_file, "--workspace", "not-authorized", "--output-format", "json"
    )
    assert result.exit_code != 0
    assert "not authorized" in str(result.exception)
    assert not SessionStore().root.exists()


def _multiple_workspaces(connection):
    connection.access["workspaces"].append(
        {
            "name": "perception",
            "role": "runner",
            "allocated": True,
            "clusters": {"west": 1},
            "gpu_limit": 1,
        }
    )


def test_relogin_preserves_explicit_placement_when_multiple_workspaces_exist(
    connection, key_file
):
    _multiple_workspaces(connection)
    result = _key_login(key_file, "--workspace", "perception", "--cluster", "west")
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["login", "--output-format", "json"])
    assert result.exit_code == 0, result.output
    assert (SessionStore().load().workspace, SessionStore().load().cluster) == (
        "perception",
        "west",
    )


@pytest.mark.parametrize("change", ["endpoint", "person"])
def test_other_endpoint_or_person_never_inherits_prior_placement(
    connection, key_file, change
):
    _multiple_workspaces(connection)
    assert _key_login(key_file, "--workspace", "perception").exit_code == 0
    options = []
    if change == "endpoint":
        options = [
            "--endpoint",
            "https://other-team.example.test",
            "--token-file",
            str(key_file),
        ]
    else:
        connection.access["subject"] = "another-workbench-user"
    result = runner.invoke(app, ["login", *options, "--output-format", "json"])
    assert result.exit_code == 0, result.output
    assert SessionStore().load().workspace is None
    assert SessionStore().load().cluster is None


def test_explicit_workspace_does_not_inherit_previous_cluster(connection, key_file):
    _multiple_workspaces(connection)
    assert _key_login(key_file, "--workspace", "robotics").exit_code == 0
    result = runner.invoke(
        app,
        [
            "login",
            "--workspace",
            "perception",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (SessionStore().load().workspace, SessionStore().load().cluster) == (
        "perception",
        "west",
    )


def test_explicit_cluster_resolves_its_workspace_without_inheriting_old_one(
    connection, key_file
):
    _multiple_workspaces(connection)
    assert _key_login(key_file, "--workspace", "robotics").exit_code == 0
    result = runner.invoke(
        app, ["login", "--cluster", "west", "--output-format", "json"]
    )
    assert result.exit_code == 0, result.output
    assert (SessionStore().load().workspace, SessionStore().load().cluster) == (
        "perception",
        "west",
    )


def test_relogin_revalidates_saved_placement_against_current_permissions(
    connection, key_file
):
    _multiple_workspaces(connection)
    assert _key_login(key_file, "--workspace", "perception").exit_code == 0
    before = (SessionStore().root / "default.json").read_bytes()
    connection.access["workspaces"].pop()
    result = runner.invoke(app, ["login", "--output-format", "json"])
    assert result.exit_code != 0
    assert "not authorized" in str(result.exception)
    assert (SessionStore().root / "default.json").read_bytes() == before


def test_same_account_keeps_placement_when_switching_from_key_to_nebius(
    connection, key_file, native
):
    _multiple_workspaces(connection)
    assert _key_login(key_file, "--workspace", "perception").exit_code == 0
    result = runner.invoke(app, ["login", "--nebius", "--output-format", "json"])
    assert result.exit_code == 0, result.output
    saved = SessionStore().load()
    assert (saved.auth_mode, saved.workspace, saved.cluster) == (
        "nebius",
        "perception",
        "west",
    )
    assert _KEY not in (SessionStore().root / "default.json").read_text()
