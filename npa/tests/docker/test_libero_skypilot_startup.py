# npa: publication-enforcement=libero
"""The startup adapter preserves runtime code and refuses unreviewed inputs."""

import base64
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def startup(monkeypatch):
    path = ROOT / "npa/docker/workbench/libero/skypilot-startup.py"
    spec = importlib.util.spec_from_file_location("libero_startup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithm = b"ssh-ed25519"
    wire = (
        len(algorithm).to_bytes(4, "big")
        + algorithm
        + (32).to_bytes(4, "big")
        + bytes(range(32))
    )
    key = "ssh-ed25519 " + base64.b64encode(wire).decode() + " synthetic"
    apt = "# STEP 1:\ncat <<'SKYPILOT_SSH_KEY_EOF'\n" + key + "\nSKYPILOT_SSH_KEY_EOF\n"
    environment = "# STEP 3:\noriginal-environment\n"
    runtime = "# STEP 2:\n" + module.INSTALL + "\noriginal-ray-start\n"
    source = (
        module.HOOK
        + "\n"
        + apt
        + runtime
        + environment
        + "function mylsof\noriginal-keepalive\n"
    )
    monkeypatch.setattr(
        module,
        "BLOCKS",
        (
            (
                "apt",
                "# STEP 1:",
                "# STEP 2:",
                hashlib.sha256(apt.replace(key, "PUBLIC_KEY").encode()).hexdigest(),
            ),
            (
                "env",
                "# STEP 3:",
                "function mylsof",
                hashlib.sha256(environment.encode()).hexdigest(),
            ),
        ),
    )
    return module, source, runtime, key


def test_adapter_preserves_runtime_and_initializes_real_ssh(startup):
    module, source, runtime, key = startup
    adapted = module.adapt(source)
    assert runtime.replace(module.INSTALL, module.INSTALL + "==0.12.2") in adapted
    assert "original-keepalive" in adapted
    assert module.HOOK not in adapted
    assert key in adapted
    assert "sudo -n /usr/local/bin/ssh-keygen -A" in adapted
    assert "sudo -n /usr/sbin/service ssh start" in adapted
    assert "npa-skypilot-bootstrap-guard verify" in adapted
    assert '. "$HOME/container_env_var.sh"' in adapted
    assert "apt-get" not in adapted
    assert "/etc/profile.d" not in adapted


@pytest.mark.parametrize(
    "mutation",
    ["apt", "env", "key", "duplicate", "late_hook", "runtime", "runtime_version"],
)
def test_adapter_refuses_changed_bootstrap(startup, mutation):
    module, source, _, key = startup
    if mutation == "apt":
        source = source.replace("# STEP 1:\n", "# STEP 1:\necho changed\n")
    elif mutation == "env":
        source = source.replace("original-environment", "changed-environment")
    elif mutation == "key":
        source = source.replace(key, key + "; unexpected-command")
    elif mutation == "duplicate":
        source += module.HOOK
    elif mutation == "late_hook":
        source = source.replace(module.HOOK, "") + module.HOOK
    elif mutation == "runtime_version":
        source = source.replace(module.INSTALL, module.INSTALL + "==0.13.0")
    else:
        source = source.replace(module.INSTALL, "changed-runtime-install")
    with pytest.raises(ValueError):
        module.adapt(source)
