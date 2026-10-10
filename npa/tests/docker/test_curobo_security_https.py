"""The locked Ubuntu package fetch cannot follow redirects."""

import importlib.util
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "docker/workbench/curobo"
SPEC = importlib.util.spec_from_file_location(
    "apt_https", ROOT / "install_security_apt.py"
)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


@pytest.mark.parametrize("status", [200, 301, 302, 307, 308, 404, 500])
def test_exact_package_https_rejects_redirect_before_read(monkeypatch, status):
    calls = []

    class Response(io.BytesIO):
        def read(self, *args):
            calls.append("read")
            return super().read(*args)

    response = Response(b"exact package bytes")
    response.status = status
    response.headers = {"Location": "http://unapproved.invalid/redirect"}

    class Connection:
        def __init__(self, host):
            calls.append(("connect", host))

        def request(self, method, path):
            calls.append((method, path))

        def getresponse(self):
            return response

        def close(self):
            calls.append("close")

    monkeypatch.setattr(fetcher.http.client, "HTTPSConnection", Connection)
    url = "https://snapshot.ubuntu.com/ubuntu/pinned.deb"
    if status == 200:
        assert fetcher._download_https(url) == b"exact package bytes"
    else:
        with pytest.raises(ValueError, match="not 200"):
            fetcher._download_https(url)
        assert "read" not in calls
    assert calls[0:2] == [
        ("connect", "snapshot.ubuntu.com"),
        ("GET", "/ubuntu/pinned.deb"),
    ]
    assert sum(isinstance(c, tuple) and c[0] == "connect" for c in calls) == 1
    assert calls[-1] == "close" and response.closed


def test_package_transport_error_still_closes_connection(monkeypatch):
    closed = []

    class Connection:
        def __init__(self, host):
            pass

        def request(self, *args):
            raise OSError("original TLS failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(fetcher.http.client, "HTTPSConnection", Connection)
    with pytest.raises(OSError, match="original TLS failure"):
        fetcher._download_https("https://snapshot.ubuntu.com/ubuntu/pinned.deb")
    assert closed == [True]
