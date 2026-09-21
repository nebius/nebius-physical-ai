"""BYOF image must not retain build-generated SSH host keys in any layer."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "npa/scripts/run_byof_repo.py"

_LAYER_START = re.compile(
    r"^(?:RUN|FROM|USER|WORKDIR|ARG|COPY|ENV|ENTRYPOINT|CMD)\b", re.MULTILINE
)


def _dockerfile() -> str:
    spec = importlib.util.spec_from_file_location("byof_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._dockerfile_text()


def _run_layers(dockerfile: str) -> list[str]:
    """Split into instructions, honouring backslash line continuations."""
    starts = [m.start() for m in _LAYER_START.finditer(dockerfile)]
    bounds = starts + [len(dockerfile)]
    instructions = [
        dockerfile[bounds[index] : bounds[index + 1]] for index in range(len(starts))
    ]
    return [text for text in instructions if text.startswith("RUN")]


def test_openssh_install_and_host_key_removal_share_one_layer() -> None:
    """The removal must happen in the layer that created the keys.

    openssh-server's postinst generates /etc/ssh/ssh_host_* at install time.
    Removing them in a later RUN leaves the private keys readable in the
    earlier layer even though the merged rootfs is clean, which is what an
    image scanner reports as a HIGH private-key finding.
    """
    layers = _run_layers(_dockerfile())
    installing = {
        index for index, text in enumerate(layers) if "openssh-server" in text
    }
    removing = {
        index
        for index, text in enumerate(layers)
        if "rm -f /etc/ssh/ssh_host_*" in text
    }

    assert installing, "no layer installs openssh-server"
    assert removing, "no layer removes the generated host keys"
    assert installing <= removing, (
        "openssh-server is installed in a layer that does not also remove "
        f"/etc/ssh/ssh_host_*: install={sorted(installing)} "
        f"remove={sorted(removing)}"
    )


def test_runtime_host_key_generation_is_preserved() -> None:
    """Removing build-time keys must not leave the container without keys."""
    dockerfile = _dockerfile()
    assert "ssh-keygen -A" in dockerfile, (
        "runtime host-key generation was dropped; containers would start sshd "
        "with no host keys"
    )
