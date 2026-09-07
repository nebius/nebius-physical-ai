"""Opt-in native VM security boundaries using real Docker, systemd and nginx.

Supply NPA_VM_SECURITY_CONFIG as an owner-only JSON file containing host, user,
key_path, known_hosts, evidence_dir, caddy_image (an immutable reviewed image),
and artifact_local (an actual workload MCAP). The VM must be task-owned and
already provisioned through the normal authenticated deployment path. An
explicit isolated-pod transport instead requires kubeconfig, context, namespace,
pod and exact pod_uid, with no host namespaces or mounts. These
checks do not provision infrastructure or run model inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest
from npa.clients.config import SSHConfig
from npa.clients.env import load_env_file_script, render_shell_env_file
from npa.clients.ssh import SSHClient
from npa.deploy.configurator import write_remote_docker_env_file

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or os.environ.get("NPA_VM_SECURITY_LIVE") != "1",
        reason="requires an explicitly configured task-owned native VM or isolated CPU Pod",
    ),
]



class _PodCommandClient:
    """Test-only transport into one UID-verified, isolated task-owned CPU Pod."""

    def __init__(self, cfg):
        self.prefix = ["kubectl", "--kubeconfig", cfg["kubeconfig"], "--context", cfg["context"], "-n", cfg["namespace"]]
        self.pod = cfg["pod"]
        probe = subprocess.run([*self.prefix, "get", "pod", self.pod, "-o", "json"], capture_output=True, check=True)
        actual = json.loads(probe.stdout)
        assert actual["metadata"]["uid"] == cfg["pod_uid"]
        assert not any(actual["spec"].get(key, False) for key in ("hostPID", "hostIPC", "hostNetwork"))
        assert not any("hostPath" in item for item in actual["spec"].get("volumes", []))

    def run(self, command, **_kwargs):
        result = subprocess.run([*self.prefix, "exec", self.pod, "--", "bash", "-c", command], capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr

    def run_or_raise(self, command, **kwargs):
        result = self.run(command, **kwargs)
        assert result[0] == 0, "private Pod command failed"
        return result

    def _upload(self, content, remote_path):
        parent = shlex.quote(str(Path(remote_path).parent))
        target = shlex.quote(remote_path)
        command = ("set -eu; umask 077; "
                   + f"stage=$(mktemp -d {parent}/.npa-stage.XXXXXXXX); "
                   + "trap 'rm -rf -- \"$stage\"' EXIT; "
                   + "install -m 0600 /dev/null \"$stage/payload\"; "
                   + "cat > \"$stage/payload\"; "
                   + f"mv -fT -- \"$stage/payload\" {target}")
        result = subprocess.run([*self.prefix, "exec", "-i", self.pod, "--", "bash", "-c", command], input=content, capture_output=True, check=False)
        assert result.returncode == 0, "private Pod upload failed"
        return remote_path

    def upload_private_text(self, content, remote_path):
        return self._upload(content.encode(), remote_path)

    def upload_file(self, local_path, remote_path):
        return self._upload(Path(local_path).read_bytes(), remote_path)

    @contextmanager
    def temporary_directory(self):
        code, out, _ = self.run("umask 077; mktemp -d /tmp/npa-security.XXXXXXXX")
        assert code == 0
        path = out.strip()
        try:
            yield path
        finally:
            code, _, _ = self.run("rm -rf -- " + shlex.quote(path))
            assert code == 0


@pytest.fixture(scope="module")
def vm():
    config_path = Path(os.environ["NPA_VM_SECURITY_CONFIG"])
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    cfg = json.loads(config_path.read_text())
    evidence = Path(cfg["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    if cfg.get("transport") == "isolated-pod":
        ssh = _PodCommandClient(cfg)
    else:
        ssh = SSHClient(SSHConfig(
            host=cfg["host"], user=cfg["user"], key_path=cfg["key_path"],
        ), known_hosts=cfg["known_hosts"])
    with ssh.temporary_directory() as remote:
        yield ssh, cfg, evidence, remote


def _record(vm, name, value):
    path = vm[2] / (name + ".json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
    return value


def _private_run(vm, name, script, *, interpreter="bash", root=False):
    ssh, _, _, remote = vm
    path = f"{remote}/{name}"
    ssh.upload_private_text(script, path)
    command = ("sudo " if root else "") + shlex.join([interpreter, path])
    code, stdout, stderr = ssh.run(command)
    _record(vm, name + "-execution", {
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "invocation_contains_script_bytes": script in command,
    })
    return code, stdout, stderr


def _python(vm, name, source):
    code, out, _ = _private_run(
        vm, name, textwrap.dedent(source), interpreter="python3", root=True,
    )
    assert code == 0, f"{name}: private execution receipt records failure"
    result = json.loads(out)
    _record(vm, name + "-result", result)
    return result


def test_agent_bcrypt_private_staging_and_actual_nginx(vm):
    from npa.cli.agent import _agent_auth_setup_script

    ssh, _, _, remote = vm
    password = secrets.token_urlsafe(24) + " ' $ \\"
    username = "security-validation"
    script_path = f"{remote}/auth-install.sh"
    ssh.upload_private_text(_agent_auth_setup_script(username, password), script_path)
    result = _python(vm, "agent-auth", f'''
        import base64, hashlib, json, os, pathlib, socket, stat, subprocess, tempfile, threading, time
        import urllib.error, urllib.request
        password = {password!r}
        username = {username!r}
        remote = pathlib.Path({remote!r})
        target = pathlib.Path('/etc/nginx/.npa-agent-htpasswd')
        old = target.read_bytes() if target.exists() else None
        old_stat = target.stat() if old is not None else None
        result = {{'password_in_process_arguments': False, 'htpasswd_observed': False,
                   'private_staging_observed': False, 'staging_mode_violation': False}}
        stop = threading.Event()
        def observe():
            while not stop.is_set():
                for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        value = path.read_bytes()
                        if password.encode() in value:
                            result['password_in_process_arguments'] = True
                        if b'htpasswd' in value and b'-iBc' in value:
                            result['htpasswd_observed'] = True
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        pass
                for path in pathlib.Path('/etc/nginx').glob('.npa-auth.*/auth'):
                    try:
                        mode = stat.S_IMODE(path.stat().st_mode)
                        parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
                        if parent_mode != 0o700 or mode not in (0o600, 0o640):
                            result['staging_mode_violation'] = True
                        result['private_staging_observed'] = True
                    except FileNotFoundError:
                        pass
        def checked_observe():
            try:
                observe()
            except Exception as error:
                result['observer_error'] = type(error).__name__
        worker = threading.Thread(target=checked_observe, daemon=True)
        nginx = None
        try:
            worker.start()
            installed = subprocess.run(['bash', {script_path!r}], capture_output=True)
            stop.set(); worker.join()
            assert installed.returncode == 0
            metadata = target.stat()
            result['final_mode'] = oct(stat.S_IMODE(metadata.st_mode))
            result['root_owned'] = metadata.st_uid == 0
            import grp
            result['web_group'] = metadata.st_gid == grp.getgrnam('www-data').gr_gid
            value = target.read_text().split(':', 1)[1]
            result['bcrypt'] = value.startswith(('$2y$12$', '$2b$12$', '$2a$12$'))
            with socket.socket() as available:
                available.bind(('127.0.0.1', 0)); port = available.getsockname()[1]
            config = remote / 'nginx.conf'
            config.write_text('user www-data;\\npid ' + str(remote/'nginx.pid') + ';\\nevents {{}}\\nhttp {{ server {{ listen 127.0.0.1:' + str(port) + '; auth_basic "validation"; auth_basic_user_file /etc/nginx/.npa-agent-htpasswd; location / {{ root /usr/share/nginx/html; }} }} }}')
            nginx = subprocess.Popen(['nginx', '-c', str(config), '-g', 'daemon off;'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            def request(credential):
                headers = {{}}
                if credential is not None:
                    headers['Authorization'] = 'Basic ' + base64.b64encode((username+':'+credential).encode()).decode()
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+str(port)+'/', headers=headers), timeout=5) as response:
                        return response.status, len(response.read())
                except urllib.error.HTTPError as error:
                    return error.code, len(error.read())
            while True:
                assert nginx.poll() is None
                try:
                    result['anonymous_status'], _ = request(None); break
                except urllib.error.URLError:
                    time.sleep(.05)
            result['incorrect_status'], _ = request(password+'incorrect')
            result['correct_status'], result['response_bytes'] = request(password)
            result['staging_clean'] = not list(pathlib.Path('/etc/nginx').glob('.npa-auth.*'))
        finally:
            try:
                stop.set()
                if worker.ident is not None:
                    worker.join()
                if nginx is not None:
                    nginx.terminate(); nginx.wait()
            finally:
                if old is not None:
                    with tempfile.TemporaryDirectory(prefix='.restore-', dir=target.parent) as directory:
                        restored = pathlib.Path(directory) / 'auth'
                        fd = os.open(restored, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, 'wb') as handle:
                            handle.write(old)
                        os.chmod(restored, stat.S_IMODE(old_stat.st_mode))
                        os.chown(restored, old_stat.st_uid, old_stat.st_gid)
                        os.replace(restored, target)
                else:
                    target.unlink(missing_ok=True)
        print(json.dumps(result))
    ''')
    assert result["htpasswd_observed"] and result["private_staging_observed"]
    assert "observer_error" not in result
    assert not result["password_in_process_arguments"] and not result["staging_mode_violation"]
    assert result["final_mode"] == "0o640" and result["root_owned"] and result["web_group"]
    assert result["bcrypt"] and result["staging_clean"]
    assert [result[k] for k in ("anonymous_status", "incorrect_status", "correct_status")] == [401, 401, 200]
    assert result["response_bytes"] > 0


def test_cosmos_actual_systemd_environment_and_failed_atomic_update(vm):
    from npa.cli.cosmos import _build_service_env_script

    ssh, _, _, remote = vm
    token = secrets.token_urlsafe(24) + " dollar$ quotes'\" slash\\ unicode-é"
    model = "validation/model quote'\" dollar$ slash\\ é"
    for name, value in (("valid", token), ("invalid", token + "\ninvalid")):
        script = "set -euo pipefail\n" + render_shell_env_file({"HF_TOKEN": value}, export=True)
        script += _build_service_env_script(model, 8123)
        ssh.upload_private_text(script, f"{remote}/cosmos-{name}.sh")
    expected = hashlib.sha256(token.encode()).hexdigest()
    result = _python(vm, "cosmos-env", f'''
        import hashlib, json, os, pathlib, stat, subprocess, tempfile, uuid
        target = pathlib.Path('/etc/npa-cosmos-server/env')
        remote = pathlib.Path({remote!r})
        old = target.read_bytes() if target.exists() else None
        old_stat = target.stat() if old is not None else None
        try:
            valid = subprocess.run(['bash', str(remote/'cosmos-valid.sh')], capture_output=True)
            assert valid.returncode == 0
            before = target.read_bytes()
            probe = remote/'systemd-proof.py'
            proof = remote/'systemd-proof.json'
            probe.write_text('import os,json,hashlib,pathlib\\npathlib.Path('+repr(str(proof))+').write_text(json.dumps({{"token_sha256":hashlib.sha256(os.environ["HF_TOKEN"].encode()).hexdigest(),"model_sha256":hashlib.sha256(os.environ["COSMOS_MODEL_ID"].encode()).hexdigest()}}))')
            unit = 'npa-security-env-' + uuid.uuid4().hex
            parsed = subprocess.run(['systemd-run', '--quiet', '--wait', '--collect', '--unit='+unit, '--property=EnvironmentFile='+str(target), '/usr/bin/python3', str(probe)], capture_output=True)
            assert parsed.returncode == 0
            result = json.loads(proof.read_text())
            result['final_mode'] = oct(stat.S_IMODE(target.stat().st_mode))
            invalid = subprocess.run(['bash', str(remote/'cosmos-invalid.sh')], capture_output=True)
            result['invalid_exit_code'] = invalid.returncode
            result['old_bytes_preserved'] = target.read_bytes() == before
            result['staging_clean'] = not list(target.parent.glob('.env.*'))
        finally:
            if old is not None:
                with tempfile.TemporaryDirectory(prefix='.restore-', dir=target.parent) as directory:
                    restore = pathlib.Path(directory) / 'env'
                    fd = os.open(restore, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'wb') as handle:
                        handle.write(old)
                    os.chmod(restore, stat.S_IMODE(old_stat.st_mode))
                    os.chown(restore, old_stat.st_uid, old_stat.st_gid)
                    os.replace(restore, target)
            else:
                target.unlink(missing_ok=True)
        print(json.dumps(result))
    ''')
    assert result["token_sha256"] == expected
    assert result["model_sha256"] == hashlib.sha256(model.encode()).hexdigest()
    assert result["final_mode"] == "0o600"
    assert result["invalid_exit_code"] != 0 and result["old_bytes_preserved"] and result["staging_clean"]


def test_literal_shared_env_matches_real_docker_and_private_override(vm):
    ssh, cfg, _, remote = vm
    marker = f"{remote}/must-not-execute"
    value = "spaces ' \" \\ $HOME `touch " + marker + "` $(touch " + marker + ") café"
    first = {"VALIDATION_LITERAL": value, "VALIDATION_OVERRIDE": "first"}
    second = {"VALIDATION_OVERRIDE": "second"}
    base, override = f"{remote}/shared.env", f"{remote}/container.env"
    write_remote_docker_env_file(ssh, base, first, owner=cfg["user"])
    write_remote_docker_env_file(ssh, override, second, owner=cfg["user"])
    script = load_env_file_script(base) + "\n" + load_env_file_script(override)
    script += "\nprintf '%s' \"$VALIDATION_LITERAL\" | sha256sum\nprintf '%s' \"$VALIDATION_OVERRIDE\" | sha256sum\n"
    code, out, _ = _private_run(vm, "literal-shell", script, interpreter="/bin/sh")
    assert code == 0
    shell_hashes = [line.split()[0] for line in out.splitlines()]
    docker_script = f"{remote}/docker-proof.sh"
    ssh.upload_private_text("printf '%s' \"$VALIDATION_LITERAL\" | sha256sum\nprintf '%s' \"$VALIDATION_OVERRIDE\" | sha256sum\n", docker_script)
    image = cfg["caddy_image"]
    assert "@sha256:" in image
    argv = ["sudo", "docker", "run", "--rm", "--network", "none", "--env-file", base, "--env-file", override, "-v", f"{docker_script}:/proof.sh:ro", "--entrypoint", "/bin/sh", image, "/proof.sh"]
    assert value not in shlex.join(argv)
    code, out, _ = ssh.run(shlex.join(argv))
    assert code == 0
    docker_hashes = [line.split()[0] for line in out.splitlines()]
    expected = [hashlib.sha256(value.encode()).hexdigest(), hashlib.sha256(b"second").hexdigest()]
    assert shell_hashes == docker_hashes == expected
    code, _, _ = ssh.run("test ! -e " + shlex.quote(marker))
    assert code == 0
    result = _python(vm, "literal-env-modes", f'''
        import json,pathlib,stat
        files = [pathlib.Path({base!r}), pathlib.Path({override!r})]
        print(json.dumps({{'all_private': all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in files), 'staging_clean': not list(files[0].parent.glob('.npa-install.*'))}}))
    ''')
    assert result["all_private"] and result["staging_clean"]
    _record(vm, "literal-env-parity", {"literal_shell_and_docker_hashes_match": True, "override_order_preserved": True, "marker_absent": True, "credential_not_in_docker_argv": True})


def test_lichtblick_real_mcap_range_and_loopback_binding(vm):
    from npa.workbench.lichtblick import build_launch_plan, launch_viewer

    ssh, cfg, _, remote = vm
    artifact = Path(cfg["artifact_local"])
    data = artifact.read_bytes()
    assert data.startswith(b"\x89MCAP0\r\n") and data.endswith(b"\x89MCAP0\r\n")
    remote_artifact = f"{remote}/recording.mcap"
    # Binary data uses the same private transfer helper's underlying SFTP API.
    ssh.upload_file(str(artifact), remote_artifact)
    code, out, _ = ssh.run("python3 -c 'import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()'")
    assert code == 0
    port = int(out.strip())
    plan = build_launch_plan(input_path=remote_artifact, port=port, image=cfg["caddy_image"])
    assert plan.host == "127.0.0.1"
    container = "security-viewer-" + secrets.token_hex(8)
    def runner(argv):
        # The reviewed base is the production Caddy transport, with its exact
        # production command supplied explicitly; this does not claim UI build coverage.
        code, _, _ = ssh.run(shlex.join(["sudo", *argv, "caddy", "file-server", "--root", "/srv", "--listen", ":8080"]))
        assert code == 0
    try:
        # Caddy's nobody user needs read access to the already staged artifact.
        code, _, _ = ssh.run("chmod 644 " + shlex.quote(remote_artifact))
        assert code == 0
        launch_viewer(plan, local_artifact=remote_artifact, runner=runner, container_name=container)
        result = _python(vm, "lichtblick-http", f'''
            import hashlib,json,socket,subprocess,time,urllib.error,urllib.request
            url = 'http://127.0.0.1:{port}/data/recording.mcap'
            while True:
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        payload=response.read(); status=response.status
                    break
                except urllib.error.HTTPError:
                    raise
                except urllib.error.URLError:
                    state = json.loads(subprocess.check_output(['docker', 'inspect', {container!r}]))[0]['State']
                    assert state['Running'], 'viewer process exited during readiness'
                    time.sleep(.05)
            with urllib.request.urlopen(urllib.request.Request(url,headers={{'Range':'bytes=8-71'}}),timeout=5) as response:
                part=response.read(); range_status=response.status
            binding=json.loads(subprocess.check_output(['docker','inspect',{container!r}]))[0]['HostConfig']['PortBindings']['8080/tcp']
            interfaces = [ip for ip in subprocess.check_output(['hostname','-I']).decode().split() if ':' not in ip and not ip.startswith('127.')]
            assert interfaces
            external_closed = True
            for address in interfaces:
                try:
                    with socket.create_connection((address, {port}), timeout=2):
                        external_closed = False
                except OSError:
                    pass
            print(json.dumps({{'external_interface_closed':external_closed,'status':status,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'range_status':range_status,'range_sha256':hashlib.sha256(part).hexdigest(),'loopback_only':all(row['HostIp']=='127.0.0.1' for row in binding)}}))
        ''')
        assert result["status"] == 200 and result["sha256"] == hashlib.sha256(data).hexdigest()
        assert result["range_status"] == 206 and result["range_sha256"] == hashlib.sha256(data[8:72]).hexdigest()
        assert result["loopback_only"] and result["external_interface_closed"]
    finally:
        code, _, _ = ssh.run(shlex.join(["sudo", "docker", "rm", "-f", container]))
        assert code == 0
