"""Build fetches retain TLS and cannot follow attacker-selected redirects."""

import importlib.util
import io
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2] / "docker/workbench"


@pytest.fixture
def fetcher():
    spec = importlib.util.spec_from_file_location(
        "build_https", ROOT / "common/secure_pip/build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, True


@pytest.mark.parametrize("status", [200, 301, 302, 307, 308, 404, 500])
def test_one_https_connection_rejects_redirect_and_error_without_read(
    fetcher, monkeypatch, status
):
    module, parsed_input = fetcher
    calls = []

    class Response(io.BytesIO):
        def read(self, *args):
            calls.append("read")
            return super().read(*args)

    response = Response(b"exact source bytes")
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

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    url = "https://files.pythonhosted.org/pinned-source"
    if status == 200:
        assert (
            module._download_https(urlsplit(url) if parsed_input else url)
            == b"exact source bytes"
        )
    else:
        with pytest.raises(ValueError, match="not 200"):
            module._download_https(urlsplit(url) if parsed_input else url)
        assert "read" not in calls
    assert calls[0:2] == [
        ("connect", "files.pythonhosted.org"),
        ("GET", "/pinned-source"),
    ]
    assert sum(isinstance(c, tuple) and c[0] == "connect" for c in calls) == 1
    assert calls[-1] == "close" and response.closed


def test_transport_error_still_closes_connection(fetcher, monkeypatch):
    module, parsed_input = fetcher
    closed = []

    class Connection:
        def __init__(self, host):
            pass

        def request(self, *args):
            raise OSError("original TLS failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    url = "https://files.pythonhosted.org/pinned-source"
    with pytest.raises(OSError, match="original TLS failure"):
        module._download_https(urlsplit(url) if parsed_input else url)
    assert closed == [True]
