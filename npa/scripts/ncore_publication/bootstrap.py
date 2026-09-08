"""Exercise the exact NCore image in disposable CPU bootstrap containers."""

import ast
import json
import re
import shlex
import textwrap

from image_byte_scan import core as W
from . import artifact
from .process import ROOT, file_sha, public_environment, run, write_json


def verify(directory, config_digest, upstream):
    """Run offline packaging, entrypoint and Bash-override cold/warm probes.

    Args:
        directory: Private directory containing the derived inspection.tar.
        config_digest: Exact verified image configuration identity.
        upstream: Directory containing the two locked SkyPilot source files.
    Returns:
        None; original image bytes are never modified.
    Raises:
        ValueError, OSError: Pinned input or an actual bootstrap command fails.
    """
    env = public_environment()
    local_id = _load_verified(directory, config_digest, env)
    verifier = ["/opt/venv/bin/python", "/opt/ncore/bin/verify-packaging.py"]
    run(["docker", "run", "--rm", "--pull=never", "--network=none",
         "--entrypoint", verifier[0], local_id, *verifier[1:], "--image-only"],
        directory / "image-only.log", env=env)
    # Each process is a fresh disposable container. The second verification in
    # each container proves the warm path without copying any cache to the image.
    apt = _apt_script(upstream)
    script = _diagnostic_trap(apt) + _capabilities() + "\n" + apt + "\n" + (
        "/opt/venv/bin/python /opt/ncore/bin/verify-packaging.py\n" * 2)
    run(["docker", "run", "--rm", "--pull=never", "-i", local_id, "/bin/bash", "-es"],
        directory / "entrypoint-bootstrap.log", env=env, input_bytes=script.encode())
    run(["docker", "run", "--rm", "--pull=never", "-i", "--entrypoint", "/bin/bash",
         local_id, "-es"], directory / "bash-bootstrap.log", env=env, input_bytes=script.encode())


def _diagnostic_trap(apt):
    # SkyPilot keeps APT update failures in this container-local log. Preserve
    # those diagnostics in the private process stderr before --rm removes it.
    locations = set(re.findall(r"^\s*local log=(\S+)$", apt, re.MULTILINE))
    W.require(len(locations) == 1, "upstream_bootstrap_diagnostic_location_required")
    location = locations.pop()
    W.require(location.startswith("/"), "upstream_bootstrap_log_must_be_literal")
    # Read the actual pinned upstream log location, without changing its body
    # or keeping a second hard-coded path that could drift from the template.
    return "npa_apt_diagnostic_log=" + shlex.quote(location) + "\n" + """trap 'rc=$?; set +e
if [ "$rc" -ne 0 ] && [ -r "$npa_apt_diagnostic_log" ]; then
  cat "$npa_apt_diagnostic_log" >&2
fi
exit "$rc"' EXIT
"""


def _load_verified(directory, config_digest, env):
    expected = W.json_object((directory / "inspection.json").read_bytes())
    W.require(expected["config_digest"] == config_digest, "inspection_config_identity")
    image = directory / "inspection.tar"
    W.require(file_sha(image) == expected["inspection_sha256"], "inspection_archive_changed")
    run(["docker", "load", "--input", str(image)], directory / "load.log", env=env)
    identities = [line for line in (directory / "load.log").read_text().splitlines()
                  if line.startswith("Loaded image")]
    W.require(len(identities) == 1, "ambiguous_loaded_identity")
    match = re.fullmatch(r"Loaded image ID: (sha256:[0-9a-f]{64})", identities[0])
    W.require(match is not None, "unknown_loaded_identity")
    loaded_id = match.group(1)
    run(["docker", "image", "inspect", loaded_id], directory / "loaded-image.json", env=env)
    inspected = W.json_object((directory / "loaded-image.json").read_bytes())
    W.require(isinstance(inspected, list) and len(inspected) == 1 and isinstance(inspected[0], dict),
              "loaded_inspect_population")
    W.require(inspected[0].get("Id") == loaded_id, "loaded_inspect_identity")
    exported = directory / "loaded-image.tar"
    W.require(not exported.exists(), "loaded_export_must_be_new")
    run(["docker", "image", "save", "--output", str(exported), loaded_id], directory / "save.log", env=env)
    relationship = artifact.verify_local_export(exported, inspected[0], expected)
    W.require(file_sha(image) == expected["inspection_sha256"], "inspection_archive_changed")
    write_json(directory / "local-image-binding.json", relationship)
    return loaded_id


def _capabilities():
    return """set -euo pipefail
test "$(id -u)" != 0
test -w /tmp && test -w "$HOME"
sudo -n ssh-keygen -A
sudo -n /usr/sbin/sshd -t
sudo -n service ssh restart
test -s /run/sshd.pid
sudo -n kill -0 "$(cat /run/sshd.pid)"
sudo -n service ssh stop
rsync --version >/dev/null
"""


def _apt_script(upstream):
    lock = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    contents = {}
    for item in lock["bootstrap"]["upstream"]:
        name = item["url"].rsplit("/", 1)[1]
        path = upstream / name
        W.require(file_sha(path) == item["sha256"], "bootstrap_upstream_sha256")
        contents[name] = path.read_text()
    template = contents["kubernetes-ray.yml.j2"]
    prefix = next(line.strip() for line in template.splitlines() if line.strip().startswith("prefix_cmd() {"))
    start = template.index("                APT_ACQUIRE_TIMEOUT=")
    end = template.index("                {% if k8s_enable_docker_all", start)
    apt = textwrap.dedent(template[start:end])
    W.require("{{" not in apt and "{%" not in apt, "unrendered_bootstrap_template")
    provisioner = ast.parse(contents["instance.py"])
    assignment = next(node for node in ast.walk(provisioner) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "install_ssh_k8s_cmd" for target in node.targets))
    ssh = ast.literal_eval(assignment.value).split("$(prefix_cmd) mkdir -p /var/run/sshd;", 1)[0]
    packages = ("curl", "patch", "openssh-server", "rsync")
    predicates = "\n".join(f'dpkg -l | grep -q "^ii  {name} "' for name in packages)
    return prefix + "\n" + apt + "\n" + ssh + "\n" + predicates + "\nsudo -n apt-get check\nsudo -n dpkg --audit\n"
