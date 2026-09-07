"""Exercise OpenSSH control forwarding with a real local SSH protocol server."""

import http.client
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
from types import SimpleNamespace
from urllib.parse import urlparse

import paramiko
import pytest

from npa.clients import endpoint


@pytest.fixture
def local_ssh(tmp_path, monkeypatch):
    ssh_binary = shutil.which("ssh")
    if not ssh_binary:
        pytest.fail("OpenSSH client is required for endpoint forwarding validation")
    host_key = paramiko.RSAKey.generate(2048)
    client_key = paramiko.RSAKey.generate(2048)
    private_key = tmp_path / "client-key"
    client_key.write_private_key_file(str(private_key))
    private_key.chmod(0o600)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    ssh_port = listener.getsockname()[1]
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(f"[127.0.0.1]:{ssh_port} {host_key.get_name()} {host_key.get_base64()}\n")
    known_hosts.chmod(0o600)
    monkeypatch.setenv("NPA_SSH_KNOWN_HOSTS", str(known_hosts))
    stop = threading.Event()
    transports = []

    class Server(paramiko.ServerInterface):
        def check_auth_publickey(self, username, key):
            return paramiko.AUTH_SUCCESSFUL if username == "operator" and key == client_key else paramiko.AUTH_FAILED

        def get_allowed_auths(self, username):
            return "publickey"

        def check_channel_direct_tcpip_request(self, chanid, origin, destination):
            return paramiko.OPEN_SUCCEEDED if destination == ("127.0.0.1", 5151) else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def serve():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            transport = paramiko.Transport(conn)
            transports.append(transport)
            transport.add_server_key(host_key)
            transport.start_server(server=Server())
            while transport.is_active() and not stop.is_set():
                channel = transport.accept(0.1)
                if channel is not None:
                    channel.recv(4096)
                    channel.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\nConnection: close\r\n\r\nverified route")
                    try:
                        channel.close()
                    except EOFError:
                        logging.getLogger(__name__).debug("SSH client closed after receiving the response")

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    original_popen = subprocess.Popen
    processes = []

    def local_popen(argv, *args, **kwargs):
        if argv[0] == "ssh":
            argv = [ssh_binary, "-F", "/dev/null", "-p", str(ssh_port), *argv[1:]]
        proc = original_popen(argv, *args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(endpoint.subprocess, "Popen", local_popen)
    cfg = SimpleNamespace(
        ssh=SimpleNamespace(host="127.0.0.1", user="operator", key_path=str(private_key)),
        endpoint="http://legacy.example:5151", endpoint_strategy="public", service_port=5151,
        service_port_configured=True,
    )
    yield cfg, processes
    stop.set()
    for transport in transports:
        transport.close()
    worker.join(5)
    listener.close()


def test_authenticated_master_confirms_forward_and_removes_private_control_socket(local_ssh):
    cfg, processes = local_ssh
    with endpoint.service_endpoint(cfg, require_ssh=True) as active:
        master = processes[0]
        control_dir = Path(master._npa_ssh_control_directory.name)
        assert os.stat(control_dir).st_mode & 0o777 == 0o700
        assert Path(master._npa_ssh_control_path).is_socket()
        parsed = urlparse(active.url)
        client = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        client.request("GET", "/")
        response = client.getresponse()
        assert response.status == 200
        assert response.read() == b"verified route"
        client.close()
    assert not control_dir.exists()
    assert master.poll() is not None


def test_occupied_port_cannot_satisfy_ssh_readiness(local_ssh, monkeypatch):
    cfg, processes = local_ssh
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        monkeypatch.setattr(endpoint, "_free_local_port", lambda: occupied.getsockname()[1])
        with pytest.raises(endpoint.EndpointError, match="could not bind"):
            with endpoint.service_endpoint(cfg, require_ssh=True):
                pytest.fail("A busy local port must not be presented as a forwarded endpoint")
    master = processes[0]
    assert master.poll() is not None
    assert not Path(master._npa_ssh_control_directory.name).exists()
