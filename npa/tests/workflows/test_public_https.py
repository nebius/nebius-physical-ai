"""Offline transport boundary tests for anonymous bootstrap downloads."""

import http.client
import io
import ssl
import traceback

import pytest

from npa import _public_https as transport


ORIGIN = "https://huggingface.co/public.zip"
HOSTS = frozenset({"huggingface.co"})
CDNS = frozenset({"cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"})


class Response(io.BytesIO):
    def __init__(self, body=b"", *, status=200, location=None):
        super().__init__(body)
        self.status = status
        self.location = location

    def getheader(self, name):
        assert name == "Location"
        return self.location


@pytest.fixture
def network(monkeypatch):
    responses, connections, requests = [], [], []

    class Connection:
        def __init__(self, host, *, port, context):
            assert responses, "unexpected network request"
            self.host = host
            self.response = responses.pop(0)
            self.closed = False
            assert port == 443
            assert context.check_hostname is True
            assert context.verify_mode == ssl.CERT_REQUIRED
            connections.append(self)

        def request(self, method, target, body=None, headers=None):
            requests.append((self.host, method, target, body, headers))

        def getresponse(self):
            return self.response

        def close(self):
            self.closed = True

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", Connection)
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    return responses, connections, requests


def fetch(url=ORIGIN):
    output = io.BytesIO()
    transport.download_public_https(
        url, output, allowed_hosts=HOSTS, redirect_hosts=CDNS
    )
    return output.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/public.zip",
        "file:///etc/passwd",
        "ftp://huggingface.co/public.zip",
        "https://unexpected.example/public.zip",
        "https://huggingface.co.attacker.example/public.zip",
        "https://cas-bridge.xethub.hf.co/public.zip",  # redirect-only host
        "https://user:password@huggingface.co/public.zip",
        "https://huggingface.co@unexpected.example/public.zip",
        "https://huggingface.co:444/public.zip",
        "https://huggingface.co:/public.zip",
        "https://huggingface.co:invalid/public.zip",
        "https://huggingface.co./public.zip",
        "https://127.0.0.1/public.zip",
        "https://[::1]/public.zip",
        "https://[invalid/public.zip",
        "https://huggingface.co/public.zip#fragment",
        " https://huggingface.co/public.zip",
        "https://huggingface.co/pub\nlic.zip",
        "https://huggingface.co/pub\tlic.zip",
        "https://huggingface.co/é.zip",
    ],
)
def test_initial_url_rejected_before_connection(network, url):
    with pytest.raises(transport.PublicDownloadError):
        fetch(url)
    assert network[1:] == ([], [])


@pytest.mark.parametrize(
    "location",
    [
        "http://huggingface.co/public.zip",
        "file:///etc/passwd",
        "ftp://huggingface.co/public.zip",
        "https://unexpected.example/public.zip",
        "//unexpected.example/public.zip",
        "https://cas-bridge.xethub.hf.co.attacker.example/public.zip",
        "https://user:password@cas-bridge.xethub.hf.co/public.zip",
        "https://cas-bridge.xethub.hf.co:8443/public.zip",
        "https://[invalid/public.zip",
        "https://cas-bridge.xethub.hf.co/pub\r\nlic.zip",
        "\thttps://cas-bridge.xethub.hf.co/public.zip",
        None,
        "",
    ],
)
def test_redirect_rejected_before_second_connection(network, location):
    responses, connections, requests = network
    response = Response(status=302, location=location)
    responses.append(response)
    with pytest.raises(transport.PublicDownloadError):
        fetch()
    assert len(requests) == 1
    assert response.closed and connections[0].closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_public_redirects_stream_exact_bytes_without_credentials(
    network, monkeypatch, tmp_path, status
):
    # No proxy, netrc, bearer token, response cookie or previous signed query
    # may become credentials on the next host.
    monkeypatch.setenv("HF_TOKEN", "unit-test-token")
    monkeypatch.setenv("HTTPS_PROXY", "https://unit-user:unit-password@proxy.example")
    netrc = tmp_path / "netrc"
    netrc.write_text("machine huggingface.co login unit-user password unit-password\n")
    monkeypatch.setenv("NETRC", str(netrc))
    payload = bytes(range(256)) * 8193  # more than two read chunks
    responses, connections, requests = network
    replies = [
        Response(status=status, location="/resolve/public.zip"),
        Response(
            status=302,
            location="https://cas-bridge.xethub.hf.co/blob?Signature=unit-signed",
        ),
        Response(
            status=307,
            location="https://us.aws.cdn.hf.co/final?Signature=unit-cdn-signed",
        ),
        Response(payload),
    ]
    responses.extend(replies)
    assert fetch() == payload
    assert [r[:3] for r in requests] == [
        ("huggingface.co", "GET", "/public.zip"),
        ("huggingface.co", "GET", "/resolve/public.zip"),
        ("cas-bridge.xethub.hf.co", "GET", "/blob?Signature=unit-signed"),
        ("us.aws.cdn.hf.co", "GET", "/final?Signature=unit-cdn-signed"),
    ]
    assert all(r[3:] == (None, None) for r in requests)
    assert all(c.closed for c in connections)
    assert all(r.closed for r in replies)


def test_https_default_path_and_explicit_standard_port(network):
    network[0].append(Response(b"data"))
    assert fetch("https://huggingface.co:443") == b"data"
    assert network[2][0][:3] == ("huggingface.co", "GET", "/")


@pytest.mark.parametrize(
    "status", [201, 204, 206, 300, 304, 305, 400, 401, 403, 404, 500]
)
def test_non_download_status_never_returns_server_body(network, status):
    network[0].extend(Response(b"server secret", status=status)
                      for _ in range(4 if status == 500 else 1))
    with pytest.raises(transport.PublicDownloadError, match="HTTP status") as caught:
        fetch()
    assert "server secret" not in str(caught.value)
    assert network[1][0].closed


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_retries_without_delivering_error_bytes(network, monkeypatch, status):
    delays = []
    monkeypatch.setattr(transport.time, "sleep", delays.append)
    network[0].extend([Response(b"private server body", status=status), Response(b"verified input")])
    assert fetch() == b"verified input"
    assert delays == [1]
    assert len(network[2]) == 2
    assert all(request[3:] == (None, None) for request in network[2])
    assert all(connection.closed for connection in network[1])


def test_transient_status_exhaustion_stays_a_failure(network, monkeypatch):
    delays = []
    monkeypatch.setattr(transport.time, "sleep", delays.append)
    network[0].extend(Response(b"private server body", status=503) for _ in range(4))
    with pytest.raises(transport.PublicDownloadError, match="HTTP status") as caught:
        fetch()
    assert delays == [1, 2, 4]
    assert len(network[2]) == 4
    assert "private server body" not in str(caught.value)
    assert all(connection.closed for connection in network[1])


def test_retry_does_not_relax_redirect_policy(network):
    network[0].extend([Response(status=503), Response(status=302, location="http://huggingface.co/public.zip")])
    with pytest.raises(transport.PublicDownloadError, match="not permitted"):
        fetch()
    assert len(network[2]) == 2


def test_redirect_loop_closes_response(network):
    network[0].append(Response(status=302, location=ORIGIN))
    with pytest.raises(transport.PublicDownloadError, match="redirect loop"):
        fetch()
    assert len(network[1]) == 1 and network[1][0].closed


def test_changing_redirect_cycle_is_bounded(network):
    network[0].extend(Response(status=302, location=f"/next/{i}") for i in range(11))
    with pytest.raises(transport.PublicDownloadError, match="too many redirects"):
        fetch()
    assert all(c.closed for c in network[1])


@pytest.mark.parametrize("phase", ["request", "getresponse", "read", "close"])
@pytest.mark.parametrize(
    "exception",
    [OSError, http.client.HTTPException, ValueError, ssl.SSLCertVerificationError],
)
def test_transport_failures_never_expose_signed_urls(
    network, monkeypatch, phase, exception
):
    secret = "unit-sensitive-signature"
    url = f"https://huggingface.co/public.zip?Signature={secret}"
    response = Response(b"payload")
    network[0].append(response)

    def fail(*args, **kwargs):
        raise exception(url)

    if phase == "read":
        monkeypatch.setattr(response, phase, fail)
    else:
        monkeypatch.setattr(transport.http.client.HTTPSConnection, phase, fail)
    with pytest.raises(
        transport.PublicDownloadError, match="HTTPS download failed"
    ) as caught:
        fetch(url)
    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in rendered
    assert url not in rendered
    assert caught.value.__suppress_context__
    assert len(network[1]) == 1  # no HTTP downgrade, unverified TLS retry, or retry


def test_invalid_signed_redirect_error_is_sanitized(network):
    network[0].append(
        Response(
            status=302,
            location="https://unexpected.example/blob?Signature=unit-sensitive-signature",
        )
    )
    with pytest.raises(transport.PublicDownloadError) as caught:
        fetch()
    rendered = "".join(traceback.format_exception(caught.value))
    assert "unit-sensitive-signature" not in rendered
    assert "unexpected.example" not in rendered


def test_real_http_client_does_not_forward_cookies_auth_or_signed_origin_query(
    monkeypatch, tmp_path
):
    """Exercise stdlib request serialization and response parsing without sockets."""
    monkeypatch.setenv("HF_TOKEN", "unit-bearer-secret")
    monkeypatch.setenv(
        "HTTPS_PROXY", "https://unit-user:unit-proxy-secret@proxy.example"
    )
    netrc = tmp_path / "netrc"
    netrc.write_text(
        "machine huggingface.co login unit-user password unit-netrc-secret\n"
    )
    monkeypatch.setenv("NETRC", str(netrc))
    replies = [
        b"HTTP/1.1 302 Found\r\n"
        b"Location: https://cas-bridge.xethub.hf.co/blob?Signature=unit-cdn-signature\r\n"
        b"Set-Cookie: unit-session=unit-cookie-secret\r\n"
        b"Content-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npayload",
    ]
    sockets = []

    class Socket:
        def __init__(self, host):
            self.host = host
            self.reply = replies.pop(0)
            self.sent = bytearray()
            self.closed = False

        def sendall(self, data):
            self.sent.extend(data)

        def makefile(self, mode):
            assert mode == "rb"
            return io.BytesIO(self.reply)

        def close(self):
            self.closed = True

    def connect(connection):
        assert connection.port == 443
        assert connection._context.check_hostname is True
        assert connection._context.verify_mode == ssl.CERT_REQUIRED
        connection.sock = Socket(connection.host)
        sockets.append(connection.sock)

    monkeypatch.setattr(transport.http.client.HTTPSConnection, "connect", connect)
    assert fetch(ORIGIN + "?Signature=unit-origin-signature") == b"payload"
    assert [s.host for s in sockets] == ["huggingface.co", "cas-bridge.xethub.hf.co"]
    assert all(s.closed for s in sockets)
    assert sockets[1].sent.startswith(
        b"GET /blob?Signature=unit-cdn-signature HTTP/1.1\r\n"
    )
    assert b"unit-origin-signature" not in sockets[1].sent
    for socket in sockets:
        wire = socket.sent.lower()
        assert b"authorization:" not in wire
        assert b"cookie:" not in wire
        assert b"referer:" not in wire
        assert b"secret" not in wire
