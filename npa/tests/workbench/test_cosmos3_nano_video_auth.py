"""Administrative Ray access must not inherit inference-client privileges."""

import builtins
import os
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from npa.workbench.cosmos import nano_video_auth as auth
from npa.workbench.cosmos import nano_video_server as server


@pytest.fixture(autouse=True)
def clean_auth(monkeypatch):
    for name in ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_TOKEN", "synthetic-inference-token")


@pytest.mark.parametrize("mode,token", [(None, None), ("disabled", "management"), ("token", "")])
def test_builder_refuses_unprotected_ray_before_import(monkeypatch, mode, token):
    if mode is not None:
        monkeypatch.setenv("RAY_AUTH_MODE", mode)
    if token is not None:
        monkeypatch.setenv("RAY_AUTH_TOKEN", token)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ray":
            pytest.fail("Unprotected Ray was imported before admission failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="management"):
        server.app()


def test_inference_token_cannot_become_management_token(monkeypatch):
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN", os.environ["NPA_COSMOS3_VIDEO_TOKEN"])
    with pytest.raises(ValueError, match="different"):
        auth.require_management_auth()


def test_kuberay_explicit_separate_secret_is_accepted(monkeypatch):
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN", "synthetic-management-secret")
    auth.require_management_auth()


@pytest.mark.parametrize("kind", ["public", "symlink", "fifo", "empty"])
def test_unsafe_or_empty_management_file_is_rejected(tmp_path, monkeypatch, kind):
    path = tmp_path / "token"
    if kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_text("management")
        target.chmod(0o600)
        path.symlink_to(target)
    else:
        path.write_text("" if kind == "empty" else "management")
        path.chmod(0o644 if kind == "public" else 0o600)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN_PATH", str(path))
    with pytest.raises((ValueError, OSError)):
        auth.require_management_auth()


def test_fresh_private_credential_overrides_ambient_and_cleans_up(monkeypatch):
    monkeypatch.setenv("RAY_AUTH_TOKEN", "ambient-management")
    monkeypatch.setenv("RAY_AUTH_TOKEN_PATH", "/synthetic/ambient-token")
    paths = []
    values = []
    for fail in (False, True):
        try:
            with auth.local_management_auth():
                assert "RAY_AUTH_TOKEN" not in os.environ
                path = Path(os.environ["RAY_AUTH_TOKEN_PATH"])
                paths.append(path)
                values.append(path.read_text())
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
                assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
                assert values[-1] != os.environ["NPA_COSMOS3_VIDEO_TOKEN"]
                auth.require_management_auth()
                if fail:
                    raise RuntimeError("startup failed")
        except RuntimeError:
            assert fail
        assert not path.parent.exists()
        assert os.environ["RAY_AUTH_TOKEN"] == "ambient-management"
        assert os.environ["RAY_AUTH_TOKEN_PATH"] == "/synthetic/ambient-token"
        assert "RAY_AUTH_MODE" not in os.environ
    assert paths[0] != paths[1] and values[0] != values[1]


@pytest.mark.parametrize("failure", ["init", "run", None])
def test_direct_launcher_authenticates_before_ray_and_cleans_up(monkeypatch, failure):
    events = []
    token_paths = []

    def init(**kwargs):
        auth.require_management_auth()
        token_paths.append(Path(os.environ["RAY_AUTH_TOKEN_PATH"]))
        assert kwargs == {
            "address": "local", "include_dashboard": False,
            "_node_ip_address": "127.0.0.1",
        }
        events.append("init")
        if failure == "init":
            raise RuntimeError("init failure")

    def run(application, **kwargs):
        assert application == "synthetic-application" and kwargs == {"blocking": True}
        events.append("run")
        if failure == "run":
            raise RuntimeError("run failure")

    ray = ModuleType("ray")
    ray.init = init
    ray.shutdown = lambda: events.append("shutdown")
    ray.serve = SimpleNamespace(start=lambda **kw: events.append("serve-start"), run=run)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(server, "app", lambda: "synthetic-application")
    if failure:
        with pytest.raises(RuntimeError, match="failure"):
            auth.serve_local()
    else:
        auth.serve_local()
    assert events[-1] == "shutdown"
    assert not token_paths[0].parent.exists()


def test_cluster_uses_operator_auth_and_standalone_image_uses_secure_launcher():
    root = Path(__file__).resolve().parents[2]
    service = yaml.safe_load((root / "deploy/cosmos3-nano-video/rayservice.yaml").read_text())
    cluster = service["spec"]["rayClusterConfig"]
    assert cluster["authOptions"] == {"mode": "token"}
    for group in [cluster["headGroupSpec"], *cluster["workerGroupSpecs"]]:
        environment = group["template"]["spec"]["containers"][0]["env"]
        assert not any(item["name"].startswith("RAY_AUTH_") for item in environment)
        api_token = next(item for item in environment if item["name"] == "NPA_COSMOS3_VIDEO_TOKEN")
        assert api_token["valueFrom"]["secretKeyRef"]["name"] == "cosmos3-nano-video-auth"
    docker = (root / "docker/workbench/cosmos3-nano-video/Dockerfile").read_text()
    assert 'CMD ["python", "-m", "npa.workbench.cosmos.nano_video_server", "--serve"]' in docker
    assert "src/npa/workbench/cosmos/nano_video_auth.py" in docker
