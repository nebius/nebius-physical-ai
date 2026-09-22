#!/usr/local/bin/python
"""Adapt only the reviewed SkyPilot 0.12.2 initialization for a read-only image."""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sys

HOOK = 'exec /usr/local/bin/python /opt/npa/libero/skypilot-startup.py "$BASH_EXECUTION_STRING"'
KEY_PATTERN = r"(<<'SKYPILOT_SSH_KEY_EOF'\n)([^\n]+)(\nSKYPILOT_SSH_KEY_EOF)"
BLOCKS = (
    ("apt", "# STEP 1:", "# STEP 2:", "ade317c8c16b3ea7506bc9c95adb7318dc02ffaacc4a329753c54913ba3e931a"),
    ("env", "# STEP 3:", "function mylsof", "d6a8dfa214375341d886df5a389afa535d98f19c28affbf7254d2e78fd952f9c"),
)
INSTALL = "pip install skypilot[kubernetes,remote]"


def adapt(source: str) -> str:
    """Keep the rendered runtime stage; refuse different initialization bytes."""
    if len(source.encode()) > 65536 or source.count(HOOK) != 1:
        raise ValueError("unexpected SkyPilot startup hook")
    keys = list(re.finditer(KEY_PATTERN, source))
    if len(keys) != 1:
        raise ValueError("unexpected SkyPilot SSH key inventory")
    key = keys[0][2]
    if re.fullmatch(r"(?:ssh-rsa|ssh-ed25519) [A-Za-z0-9+/]+={0,2}(?: [A-Za-z0-9@._-]+)?", key) is None:
        raise ValueError("unexpected SkyPilot SSH key format")
    algorithm, encoded, *_ = key.split()
    decoded = base64.b64decode(encoded, validate=True)
    length = int.from_bytes(decoded[:4], "big")
    if not 32 <= len(decoded) <= 8192 or decoded[4:4 + length] != algorithm.encode():
        raise ValueError("unexpected SkyPilot SSH key wire format")
    blocks = {}
    for name, start, end, expected in BLOCKS:
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError("unexpected SkyPilot initialization boundaries")
        begin, finish = source.index(start), source.index(end)
        if not source.index(HOOK) < begin < finish:
            raise ValueError("SkyPilot hook must precede initialization")
        block = source[begin:finish]
        normalized = re.sub(KEY_PATTERN, lambda match: match[1] + "PUBLIC_KEY" + match[3], block)
        if hashlib.sha256(normalized.encode()).hexdigest() != expected:
            raise ValueError("SkyPilot initialization differs from reviewed 0.12.2 bytes")
        blocks[name] = block
    apt = '''# STEP 1: Verify installed packages and initialize actual SSH.
(
  set -e
  install -d -m 0750 "$HOME"
  /usr/local/sbin/npa-skypilot-bootstrap-guard verify
  sudo -n /usr/local/bin/ssh-keygen -A
  install -d -m 0700 "$HOME/.ssh"
  cat > "$HOME/.ssh/authorized_keys" <<'NPA_KEY_EOF'
''' + key + '''
NPA_KEY_EOF
  chmod 0600 "$HOME/.ssh/authorized_keys"
  sudo -n /usr/sbin/service ssh start
  nc -z 127.0.0.1 22
  touch /tmp/apt_ssh_setup_complete
) > /tmp/apt-ssh-setup.log 2>&1 || { cat /tmp/apt-ssh-setup.log; touch /tmp/apt-ssh-setup.failed; exit 1; }

'''
    environment = '''# STEP 3: Expose the actual environment to user SSH sessions.
(
  set -e
  umask 077
  export -p > "$HOME/container_env_var.sh"
  printf '%s\\n' '. "$HOME/container_env_var.sh"' >> "$HOME/.bashrc"
  printf '%s\\n' '. "$HOME/.bashrc"' >> "$HOME/.profile"
  touch /tmp/env_setup_complete
) > /tmp/env-setup.log 2>&1 || { cat /tmp/env-setup.log; touch /tmp/env-setup.failed; exit 1; }

'''
    adapted = source.replace(HOOK, "# Reviewed image startup adapter consumed once")
    adapted = adapted.replace(blocks["apt"], apt).replace(blocks["env"], environment)
    if adapted.count(INSTALL) != 1 or adapted.count(INSTALL + "\n") != 1:
        raise ValueError("unexpected SkyPilot runtime installation")
    return adapted.replace(INSTALL, INSTALL + "==0.12.2")


def main() -> None:
    if len(sys.argv) != 2 or os.geteuid() != 1000 or os.environ.get("HOME") != "/home/ubuntu":
        raise ValueError("SkyPilot startup requires the fixed nonroot supervisor")
    adapted = adapt(sys.argv[1])
    # The three original setup stages run concurrently and all use this home.
    os.makedirs("/home/ubuntu", mode=0o750, exist_ok=True)
    os.execv("/bin/bash", ["/bin/bash", "-c", adapted])


if __name__ == "__main__":
    main()
