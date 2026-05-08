"""SSH-based application deployment to the VM."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, FileSystemLoader

from npa.clients.config import SSHConfig
from npa.clients.ssh import SSHClient, SSHError

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEPLOY_DIR = Path(__file__).parent.parent.parent.parent / "deploy"
_NPA_PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent


class ConfiguratorError(Exception):
    pass


def _step(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg}", flush=True)


def _step_ok(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg} done", flush=True)


def install_lerobot(ssh: SSHClient) -> bool:
    """Check if LeRobot is installed; return True if already present."""
    code, out, _ = ssh.run(
        "/opt/lerobot/venv/bin/python -c 'import lerobot; print(lerobot.__version__)' 2>/dev/null"
    )
    if code == 0 and out.strip():
        return True
    return False


def install_container_runtime(
    ssh: SSHClient,
    *,
    ssh_user: str = "ubuntu",
    gpu: bool = True,
) -> None:
    """Install Docker and, for GPU workbenches, NVIDIA Container Toolkit."""
    gpu_install = ""
    if gpu:
        gpu_install = """
if ! dpkg-query -W nvidia-container-toolkit >/dev/null 2>&1; then
  sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
"""

    install_cmd = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  fi
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

{gpu_install}
sudo systemctl restart docker
sudo usermod -aG docker {shlex.quote(ssh_user)} || true
"""
    ssh.run_or_raise(f"bash -lc {shlex.quote(install_cmd)}")


def write_remote_env_file(
    ssh: SSHClient,
    remote_path: str,
    env: dict[str, Any],
    *,
    owner: str = "ubuntu",
) -> None:
    """Write an env file on the VM using SFTP, then secure it with sudo."""
    lines = [f"{key}={value}" for key, value in env.items() if value is not None]
    env_content = "\n".join(lines) + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as tmp:
        tmp.write(env_content)
        local_path = tmp.name

    tmp_remote = f"/tmp/{Path(remote_path).name}.{int(time.time() * 1000)}"
    try:
        _sftp_upload(ssh, local_path, tmp_remote)
        ssh.run_or_raise(
            f"sudo mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
            f"sudo mv {shlex.quote(tmp_remote)} {shlex.quote(remote_path)} && "
            f"sudo chown {shlex.quote(owner)}:{shlex.quote(owner)} {shlex.quote(remote_path)} && "
            f"sudo chmod 600 {shlex.quote(remote_path)}"
        )
    finally:
        os.unlink(local_path)


def write_remote_text_file(
    ssh: SSHClient,
    remote_path: str,
    content: str,
    *,
    owner: str = "ubuntu",
    mode: str = "0644",
) -> None:
    """Write a text file on the VM using SFTP, then move it into place."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        local_path = tmp.name

    tmp_remote = f"/tmp/{Path(remote_path).name}.{int(time.time() * 1000)}"
    try:
        _sftp_upload(ssh, local_path, tmp_remote)
        ssh.run_or_raise(
            f"sudo mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
            f"sudo mv {shlex.quote(tmp_remote)} {shlex.quote(remote_path)} && "
            f"sudo chown {shlex.quote(owner)}:{shlex.quote(owner)} {shlex.quote(remote_path)} && "
            f"sudo chmod {shlex.quote(mode)} {shlex.quote(remote_path)}"
        )
    finally:
        os.unlink(local_path)


def docker_exec_cmd(container_name: str, command: str) -> str:
    """Wrap a shell command for execution inside a Workbench container."""
    return f"sudo docker exec {shlex.quote(container_name)} bash -lc {shlex.quote(command)}"


def deploy_workbench_container(
    ssh: SSHClient,
    *,
    image_ref: str,
    container_name: str,
    env_file: str | None = None,
    volumes: Sequence[str] = (),
    work_dirs: Sequence[str] = (),
    command: str = "-lc 'tail -f /dev/null'",
    ssh_user: str = "ubuntu",
    gpu: bool = True,
    registry_token: str = "",
) -> None:
    """Install Docker and run a Workbench image as a long-lived container."""
    install_container_runtime(ssh, ssh_user=ssh_user, gpu=gpu)

    if work_dirs:
        dirs = " ".join(shlex.quote(path) for path in work_dirs)
        ssh.run_or_raise(
            f"sudo mkdir -p {dirs} && sudo chown -R "
            f"{shlex.quote(ssh_user)}:{shlex.quote(ssh_user)} {dirs}"
        )

    registry = image_ref.split("/", 1)[0]
    if registry_token:
        login_cmd = (
            f"printf %s {shlex.quote(registry_token)} | "
            f"sudo docker login {shlex.quote(registry)} -u iam --password-stdin || true"
        )
        ssh.run_or_raise(f"bash -lc {shlex.quote(login_cmd)}")

    ssh.run_or_raise(f"sudo docker pull {shlex.quote(image_ref)}")

    gpu_flag = "--gpus all " if gpu else ""
    env_flag = f"--env-file {shlex.quote(env_file)} " if env_file else ""
    volume_flags = " ".join(f"-v {shlex.quote(volume)}" for volume in volumes)
    run_cmd = (
        f"sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true\n"
        f"sudo docker run -d {gpu_flag}--ipc=host --network host "
        f"--name {shlex.quote(container_name)} --restart unless-stopped "
        f"{env_flag}{volume_flags} {shlex.quote(image_ref)} {command}"
    )
    ssh.run_or_raise(run_cmd)


def deploy_server(
    ssh: SSHClient,
    server_config: dict[str, Any],
) -> None:
    """Copy the npa package to the VM, render server config, install systemd unit."""
    # 1. Package and upload the npa source
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        archive_path = tmp.name

    try:
        subprocess.run(
            ["tar", "-czf", archive_path, "-C", str(_NPA_PACKAGE_ROOT), "."],
            check=True,
            capture_output=True,
        )
        # Upload via SSH (paramiko sftp)
        ssh.run_or_raise(f"mkdir -p /tmp/npa-deploy")
        _sftp_upload(ssh, archive_path, "/tmp/npa-deploy/npa.tgz")
    finally:
        os.unlink(archive_path)

    # 2. Extract and install on the VM
    ssh.run_or_raise(
        "rm -rf /tmp/npa-src && mkdir /tmp/npa-src && "
        "tar -xzf /tmp/npa-deploy/npa.tgz -C /tmp/npa-src 2>/dev/null; "
        '/opt/lerobot/venv/bin/pip install -q "/tmp/npa-src[server]"'
    )

    # 3. Render and upload server.yaml
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
    template = env.get_template("server.yaml.j2")
    rendered = template.render(**server_config)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(rendered)
        yaml_path = tmp.name

    try:
        ssh.run_or_raise("sudo mkdir -p /etc/npa")
        _sftp_upload(ssh, yaml_path, "/tmp/npa-server.yaml")
        ssh.run_or_raise("sudo mv /tmp/npa-server.yaml /etc/npa/server.yaml && sudo chmod 644 /etc/npa/server.yaml")
    finally:
        os.unlink(yaml_path)

    # 4. Write env file for systemd from the server config
    env_lines = [
        f"NPA_SERVER_HOST={server_config.get('server_host', '0.0.0.0')}",
        f"NPA_SERVER_PORT={server_config.get('server_port', 8080)}",
        f"NPA_CHECKPOINT_DIR={server_config.get('checkpoint_dir', '/opt/lerobot/checkpoints')}",
        f"NPA_CHECKPOINT_BUCKET={server_config.get('checkpoint_bucket', '')}",
        f"NPA_JOB_STATUS_DIR={server_config.get('job_status_dir', '/opt/lerobot/job_status')}",
        f"NPA_LOG_DIR={server_config.get('log_dir', '/var/log/npa-lerobot')}",
        f"AWS_ENDPOINT_URL={server_config.get('storage_endpoint', '')}",
    ]
    env_content = "\n".join(env_lines) + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as tmp:
        tmp.write(env_content)
        env_path = tmp.name

    try:
        ssh.run_or_raise("sudo mkdir -p /etc/npa-lerobot-server")
        _sftp_upload(ssh, env_path, "/tmp/npa-server.env")
        # Merge S3 credentials from existing .env on VM
        ssh.run_or_raise(
            "sudo mv /tmp/npa-server.env /etc/npa-lerobot-server/env && "
            "sudo chmod 600 /etc/npa-lerobot-server/env && "
            "if [ -f /opt/lerobot/.env ]; then "
            "  grep -E '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)=' /opt/lerobot/.env "
            "    | while IFS='=' read -r k v; do "
            "        sudo sed -i \"s|^${k}=.*|${k}=${v}|\" /etc/npa-lerobot-server/env 2>/dev/null || "
            "        echo \"${k}=${v}\" | sudo tee -a /etc/npa-lerobot-server/env >/dev/null; "
            "      done; "
            "fi"
        )
    finally:
        os.unlink(env_path)

    # 5. Upload and enable systemd unit
    service_src = _DEPLOY_DIR / "npa-lerobot-server.service"
    if service_src.exists():
        _sftp_upload(ssh, str(service_src), "/tmp/npa-lerobot-server.service")
        ssh.run_or_raise(
            "sudo mv /tmp/npa-lerobot-server.service /etc/systemd/system/ && "
            "sudo systemctl daemon-reload && "
            "sudo systemctl enable npa-lerobot-server"
        )

    # 6. Create required directories
    ssh.run_or_raise(
        "sudo mkdir -p /var/log/npa-lerobot /opt/lerobot/checkpoints /opt/lerobot/job_status && "
        "sudo chown ubuntu:ubuntu /var/log/npa-lerobot /opt/lerobot/checkpoints /opt/lerobot/job_status"
    )

    # 7. Restart service
    ssh.run_or_raise("sudo systemctl restart npa-lerobot-server")


def deploy_lerobot_container(
    ssh: SSHClient,
    *,
    image_ref: str,
    server_config: dict[str, Any],
    ssh_user: str = "ubuntu",
    container_name: str = "npa-lerobot",
    registry_token: str = "",
) -> None:
    """Install Docker/NVIDIA runtime and run the LeRobot server container."""
    install_cmd = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

sudo install -d -m 0755 -o {shlex.quote(ssh_user)} -g {shlex.quote(ssh_user)} \
  /opt/lerobot \
  /opt/lerobot/checkpoints \
  /opt/lerobot/job_status \
  /opt/lerobot/dataset_cache \
  /opt/lerobot/checkpoint_cache \
  /opt/lerobot/benchmarks \
  /var/log/npa-lerobot
sudo touch /opt/lerobot/.env
sudo chown {shlex.quote(ssh_user)}:{shlex.quote(ssh_user)} /opt/lerobot/.env
sudo chmod 600 /opt/lerobot/.env

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  fi
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! dpkg-query -W nvidia-container-toolkit >/dev/null 2>&1; then
  sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker {shlex.quote(ssh_user)} || true
"""
    ssh.run_or_raise(f"bash -lc {shlex.quote(install_cmd)}")

    if registry_token:
        registry = image_ref.split("/", 1)[0]
        login_cmd = (
            f"printf %s {shlex.quote(registry_token)} | "
            f"sudo docker login {shlex.quote(registry)} -u iam --password-stdin || true"
        )
        ssh.run_or_raise(f"bash -lc {shlex.quote(login_cmd)}")

    ssh.run_or_raise(f"sudo docker pull {shlex.quote(image_ref)}")

    env_args = {
        "NPA_SERVER_HOST": server_config.get("server_host", "0.0.0.0"),
        "NPA_SERVER_PORT": "8080",
        "NPA_CHECKPOINT_DIR": server_config.get("checkpoint_dir", "/opt/lerobot/checkpoints"),
        "NPA_CHECKPOINT_BUCKET": server_config.get("checkpoint_bucket", ""),
        "NPA_JOB_STATUS_DIR": server_config.get("job_status_dir", "/opt/lerobot/job_status"),
        "NPA_LOG_DIR": server_config.get("log_dir", "/var/log/npa-lerobot"),
        "AWS_ENDPOINT_URL": server_config.get("storage_endpoint", ""),
        "HF_LEROBOT_HOME": server_config.get("hf_cache_dir", "/opt/lerobot/hf_cache"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
    }
    if server_config.get("cuda_visible_devices"):
        env_args["CUDA_VISIBLE_DEVICES"] = server_config["cuda_visible_devices"]
    if server_config.get("gpu_count"):
        env_args["NPA_GPU_COUNT"] = str(server_config["gpu_count"])
    env_flags = " ".join(
        f"--env {shlex.quote(key + '=' + str(value))}"
        for key, value in env_args.items()
    )
    volume_flags = " ".join(
        [
            "-v /opt/lerobot/.env:/opt/lerobot/.env:ro",
            "-v /opt/lerobot/checkpoints:/opt/lerobot/checkpoints",
            "-v /opt/lerobot/job_status:/opt/lerobot/job_status",
            "-v /opt/lerobot/dataset_cache:/opt/lerobot/dataset_cache",
            "-v /opt/lerobot/checkpoint_cache:/opt/lerobot/checkpoint_cache",
            "-v /opt/lerobot/benchmarks:/opt/lerobot/benchmarks",
            "-v /var/log/npa-lerobot:/var/log/npa-lerobot",
        ]
    )
    run_cmd = (
        "sudo systemctl stop npa-lerobot-server >/dev/null 2>&1 || true\n"
        f"sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true\n"
        f"sudo docker run -d --gpus all --ipc=host --network host "
        f"--name {shlex.quote(container_name)} --restart unless-stopped "
        f"--env-file /opt/lerobot/.env {env_flags} {volume_flags} "
        f"{shlex.quote(image_ref)}"
    )
    ssh.run_or_raise(run_cmd)


def health_check(endpoint: str, *, retries: int = 10, backoff: float = 3.0) -> bool:
    """Poll the /health endpoint until success or timeout."""
    import httpx

    url = f"{endpoint}/health"
    for attempt in range(retries):
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(backoff)
    return False


def write_manifest(
    ssh: SSHClient,
    tool: str,
    version: str,
    deployed_by: str,
) -> None:
    """Write /etc/npa/manifest.json on the VM."""
    manifest = json.dumps(
        {
            "tool": tool,
            "version": version,
            "deployed_by": deployed_by,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        indent=2,
    )
    ssh.run_or_raise(
        f"sudo mkdir -p /etc/npa && "
        f"echo '{manifest}' | sudo tee /etc/npa/manifest.json >/dev/null"
    )


def _sftp_upload(ssh: SSHClient, local_path: str, remote_path: str) -> None:
    """Upload a file via SFTP using paramiko."""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = os.path.expanduser(ssh._config.key_path)
    try:
        client.connect(
            hostname=ssh._config.host,
            username=ssh._config.user,
            key_filename=key_path,
            timeout=15,
            look_for_keys=False,
        )
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
    finally:
        client.close()
