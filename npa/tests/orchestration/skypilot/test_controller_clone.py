"""A controller copy must preserve remote identity without sharing local state."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shlex
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from npa.orchestration.skypilot import controller_clone as clone


def proxy(key, script, *, context="fixture-context", namespace="fixture", pod="fixture-head",
          user="skypilot:ssh_user"):
    nested = shlex.join([str(script), "-c", context, "-n", namespace, pod])
    return shlex.join(["ssh", "-tt", "-i", str(key), "-o", "IdentitiesOnly=yes", "-W",
                       "[%h]:%p", user + "@127.0.0.1", "-o", "ProxyCommand=" + nested])


@pytest.fixture
def controller(tmp_path, monkeypatch):
    source = tmp_path / "source-home"
    source.mkdir()
    root = tmp_path / "transaction"
    home = root / "home"
    runtime = root / "sky-runtime"
    home.mkdir(parents=True)
    (runtime / ".sky").mkdir(parents=True)
    (runtime / ".sky/state.db").touch()
    config_path = root / "transaction-config.yaml"
    config_path.write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", str(config_path))
    monkeypatch.delenv("SKYPILOT_DB_CONNECTION_URI", raising=False)
    name = "sky-jobs-controller-fixture"
    relative = Path(".sky/clients/fixture-client/ssh/sky-key")
    key = home / relative
    key.parent.mkdir(parents=True)
    key.write_text("fixture-private-key-data")
    Path(str(key) + ".pub").write_text("fixture-public-key-data")
    script_relative = Path(".sky/kubernetes-port-forward-proxy-command-v2.sh")
    config = {"cluster_name": name, "provider": {"type": "external", "module": "sky.provision.kubernetes",
              "context": "fixture-context", "namespace": "fixture"},
              "auth": {"ssh_private_key": str(source / relative), "ssh_user": "sky",
                       "ssh_proxy_command": proxy(source / relative, source / script_relative,
                                                  pod=name + "-head")},
              "setup_commands": ["operator-owned-command"], "file_mounts": {"target": "operator-input"}}
    original = copy.deepcopy(config)
    writes = []
    tunnels = {name: (12345, 54321)}

    class Kubernetes:
        pass

    class Handle:
        def __init__(self):
            self.cluster_name = name
            self.launched_resources = SimpleNamespace(cloud=Kubernetes())
            self._cluster_yaml = "~/.sky/generated/" + name + ".yml"
            self.cached_cluster_info = object()

        @property
        def cluster_yaml(self):
            return os.path.expanduser(self._cluster_yaml)

        @cluster_yaml.setter
        def cluster_yaml(self, value):
            self._cluster_yaml = value

    handle = Handle()
    records = {name: {"handle": handle}}
    yamls = {name: copy.deepcopy(config)}

    def read_yaml(path):
        assert Path(path).is_relative_to(home), "must not use Sky's source-file fallback"
        return copy.deepcopy(yamls[Path(path).stem])

    def save_yaml(selected, body):
        writes.append(("yaml", selected))
        yamls[selected] = yaml.safe_load(body)

    state = SimpleNamespace(
        get_cluster_from_name=lambda selected, **kwargs: records.get(selected),
        get_cluster_yaml_dict=read_yaml,
        get_ssh_keys=lambda user: ("fixture-public-key-data", "fixture-private-key-data", user == "fixture-client"),
        set_cluster_yaml=save_yaml,
        update_cluster_handle=lambda selected, value: writes.append(("handle", selected, value)),
        set_cluster_skylet_ssh_tunnel_metadata=lambda selected, value: tunnels.__setitem__(selected, value),
    )
    modules = {}
    for name_ in ["sky", "sky.backends", "sky.provision", "sky.provision.kubernetes", "sky.utils"]:
        modules[name_] = ModuleType(name_)
        modules[name_].__path__ = []
        monkeypatch.setitem(sys.modules, name_, modules[name_])
    modules["sky"].__version__ = "0.12.2"
    modules["sky"].backends = modules["sky.backends"]
    modules["sky"].clouds = SimpleNamespace(Kubernetes=Kubernetes)
    modules["sky"].global_user_state = state
    modules["sky.backends"].CloudVmRayResourceHandle = Handle
    modules["sky.backends"].backend_utils = SimpleNamespace(
        ssh_credential_from_yaml=lambda path: read_yaml(path)["auth"],
    )
    modules["sky.provision.kubernetes"].utils = SimpleNamespace(
        PORT_FORWARD_PROXY_CMD_PATH="~/" + str(script_relative),
        get_ssh_proxy_command=lambda **kw: proxy(kw["private_key_path"], home / script_relative,
                                               context=kw["context"], namespace=kw["namespace"], pod=kw["pod_name"]),
    )
    modules["sky.utils"].auth_utils = SimpleNamespace(
        create_ssh_key_files_from_db=lambda path: Path(path) == key,
    )
    return SimpleNamespace(manifest={"clone_root": str(root), "source_home": str(source),
                                    "controller_names": [name], "context": "fixture-context"},
                           name=name, home=home, source=source, key=key, handle=handle,
                           config=config, original=original, yamls=yamls, writes=writes,
                           tunnels=tunnels, records=records, config_path=config_path)


@pytest.mark.parametrize("resolved_user", [False, True])
def test_relocation_preserves_remote_fields_and_clears_only_copied_tunnel(controller, resolved_user):
    c = controller
    if resolved_user:
        c.yamls[c.name]["auth"]["ssh_proxy_command"] = c.yamls[c.name]["auth"]["ssh_proxy_command"].replace(
            "skypilot:ssh_user@127.0.0.1", "sky@127.0.0.1")
        c.yamls[c.name]["auth"]["ssh_user"] = " sky "
    cached = c.handle.cached_cluster_info
    c.tunnels["unrelated-controller"] = (22222, 33333)
    clone.prepare(c.manifest)
    actual = c.yamls[c.name]
    assert c.handle.cached_cluster_info is cached
    assert c.handle.cluster_name == c.name
    assert actual["provider"] == c.original["provider"]
    assert actual["setup_commands"] == c.original["setup_commands"]
    assert actual["file_mounts"] == c.original["file_mounts"]
    assert actual["auth"]["ssh_private_key"] == str(c.key)
    destination = ("sky" if resolved_user else "skypilot:ssh_user") + "@127.0.0.1"
    assert destination in shlex.split(actual["auth"]["ssh_proxy_command"])
    assert c.tunnels == {c.name: None, "unrelated-controller": (22222, 33333)}
    target = Path(c.handle.cluster_yaml)
    assert target.is_relative_to(c.home) and target.stat().st_mode & 0o777 == 0o600
    assert yaml.safe_load(target.read_text()) == actual
    assert c.config == c.original


@pytest.mark.parametrize("mutation", ["key_bytes", "linked_key", "foreign_key", "context", "provider",
                                     "proxy_target", "proxy_flags", "proxy_user", "proxy_user_injection",
                                     "yaml_path", "missing_target", "external_db"])
def test_invalid_controller_never_changes_copied_records(controller, mutation, monkeypatch):
    c = controller
    config = c.yamls[c.name]
    if mutation == "key_bytes":
        c.key.write_text("different-key")
    elif mutation == "linked_key":
        other = c.home / "other-key"
        c.key.rename(other)
        c.key.symlink_to(other)
    elif mutation == "foreign_key":
        config["auth"]["ssh_private_key"] = str(c.source.parent / "foreign/sky-key")
    elif mutation == "context":
        config["provider"]["context"] = "other-context"
    elif mutation == "provider":
        config["provider"]["module"] = "other-provider"
    elif mutation == "proxy_target":
        config["auth"]["ssh_proxy_command"] = config["auth"]["ssh_proxy_command"].replace("fixture-context", "other-context")
    elif mutation == "proxy_flags":
        config["auth"]["ssh_proxy_command"] += " -o LocalCommand=unexpected"
    elif mutation == "proxy_user":
        config["auth"]["ssh_proxy_command"] = config["auth"]["ssh_proxy_command"].replace(
            "skypilot:ssh_user@127.0.0.1", "another-user@127.0.0.1")
    elif mutation == "proxy_user_injection":
        config["auth"]["ssh_user"] = "sky -o LocalCommand=unexpected"
    elif mutation == "yaml_path":
        c.handle._cluster_yaml = str(c.source.parent / "foreign/controller.yml")
    elif mutation == "missing_target":
        c.manifest["controller_names"].append("sky-jobs-controller-missing")
    elif mutation == "external_db":
        monkeypatch.setenv("SKYPILOT_DB_CONNECTION_URI", "fixture-external-db")
    with pytest.raises(ValueError):
        clone.prepare(c.manifest)
    assert not c.writes
    assert c.tunnels[c.name] == (12345, 54321)


def test_legacy_absolute_yaml_is_relocated_before_fallback(controller):
    c = controller
    c.handle._cluster_yaml = str(c.source / ".sky/generated" / (c.name + ".yml"))
    clone.prepare(c.manifest)
    assert Path(c.handle.cluster_yaml).is_relative_to(c.home)
