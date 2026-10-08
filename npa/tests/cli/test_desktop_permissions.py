"""Verify explicit desktop policies without changing unrelated thread permissions."""

from unittest.mock import Mock

import pytest

from npa.tools.desktop.chat_models import model_selection
from npa.tools.desktop.chat_permissions import permission_settings, settings_metadata
from npa.tools.desktop.chat_server import _update_settings
from .test_desktop_chat import chat as chat, _MODEL


@pytest.mark.parametrize(
    "mode,sandbox,approval",
    [
        ("read-only", "readOnly", "on-request"),
        ("workspace", "workspaceWrite", "on-request"),
        ("full", "dangerFullAccess", "never"),
    ],
)
def test_explicit_permission_change_targets_existing_thread(
    chat, mode, sandbox, approval
):
    client, rpc = chat
    thread = {"id": "existing", "status": {"type": "idle"}, "model": "example-model"}

    def upstream(method, params):
        if method == "model/list":
            return {"data": [_MODEL]}
        if method in {"thread/read", "thread/resume"}:
            return {"thread": thread}
        return {}

    rpc.call.side_effect = upstream
    response = client.post(
        "/chat/api/settings", json={"id": "existing", "permissionMode": mode}
    )
    assert response.status_code == 200
    rpc.call.assert_called_with(
        "thread/settings/update",
        {
            "threadId": "existing",
            "model": "example-model",
            "effort": "medium",
            "approvalPolicy": approval,
            "sandboxPolicy": {"type": sandbox},
        },
    )
    assert (
        client.get("/chat/api/thread?id=existing").json()["thread"]["permissionMode"]
        == mode
    )
    assert not {"thread/start", "turn/start"} & {
        c.args[0] for c in rpc.call.call_args_list
    }


@pytest.mark.parametrize("mode", [None, {}, [], "unknown", "custom", "root"])
def test_unknown_permission_choices_are_rejected(mode):
    with pytest.raises(ValueError, match="permissions"):
        permission_settings(mode)


def test_model_only_change_does_not_replace_custom_permissions():
    rpc = Mock()
    rpc.call.return_value = {"data": [_MODEL]}
    result = model_selection(rpc, {"id": "existing", "model": "example-model"}, {})
    assert "approvalPolicy" not in result
    assert "sandboxPolicy" not in result


def test_metadata_does_not_claim_custom_policies_are_presets():
    assert "permissionMode" not in settings_metadata({})
    for mode in ("read-only", "workspace", "full"):
        preset = permission_settings(mode)
        assert settings_metadata(preset)["permissionMode"] == mode
        preset["sandboxPolicy"]["extraPolicy"] = True
        assert settings_metadata(preset)["permissionMode"] == "custom"


def test_runtime_notification_updates_cached_policy():
    server = Mock(settings={"existing": {"permissionMode": "full"}})
    _update_settings(
        server,
        {
            "method": "thread/settings/updated",
            "params": {
                "threadId": "existing",
                "threadSettings": permission_settings("read-only"),
            },
        },
    )
    assert server.settings["existing"]["permissionMode"] == "read-only"


@pytest.mark.parametrize(
    "mode,defaults",
    [
        ("read-only", {"networkAccess": False}),
        (
            "workspace",
            {
                "writableRoots": [],
                "networkAccess": False,
                "excludeTmpdirEnvVar": False,
                "excludeSlashTmp": False,
            },
        ),
    ],
)
def test_runtime_serialized_defaults_preserve_the_selected_preset(mode, defaults):
    preset = permission_settings(mode)
    preset["sandboxPolicy"].update(defaults)
    assert settings_metadata(preset)["permissionMode"] == mode
    preset["sandboxPolicy"]["networkAccess"] = True
    assert settings_metadata(preset)["permissionMode"] == "custom"


def test_permission_change_requires_authenticated_same_origin(chat):
    client, rpc = chat
    body = {"id": "existing", "permissionMode": "full"}
    assert client.post("/chat/api/settings", json=body, auth=None).status_code == 401
    assert (
        client.post(
            "/chat/api/settings",
            json=body,
            headers={"Origin": "https://foreign.example.test"},
        ).status_code
        == 403
    )
    rpc.call.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"approvalPolicy": "never"},
        {"sandboxPolicy": {"type": "dangerFullAccess"}},
        {"permissionMode": "full", "approvalPolicy": "never"},
    ],
)
def test_browser_cannot_inject_raw_permission_fields(body):
    rpc = Mock()
    with pytest.raises(ValueError, match="supported permission"):
        model_selection(rpc, {"id": "existing", **body}, {})
    rpc.call.assert_not_called()
