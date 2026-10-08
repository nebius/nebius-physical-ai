"""Exercise real JWT validation across browser login, session, and access boundaries."""

import base64
import hashlib
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.browser_sessions import SESSION_COOKIE, STATE_COOKIE
from npa.workbench.team.client import TeamClient
from npa.workbench.team.errors import AuthenticationError
from npa.workbench.team.models import BrowserLogin, TeamConfig
from npa.workbench.team.oidc import OidcProvider
from npa.workbench.team.service import TeamService


@pytest.fixture
def browser(config, tokens, monkeypatch):
    settings = BrowserLogin(public_url="https://testserver", client_id="workbench")
    policy = [config.model_copy(update={"browser_login": settings})]
    exchange = {"claims": {}, "requests": []}
    metadata = {
        "issuer": config.identity.issuer,
        "jwks_uri": config.identity.jwks_url,
        "authorization_endpoint": config.identity.issuer + "/authorize",
        "token_endpoint": config.identity.issuer + "/token",
        "code_challenge_methods_supported": ["S256"],
    }

    def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata)
        exchange["requests"].append(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"id_token": tokens.sign(**exchange["claims"])})

    provider = OidcProvider(
        config.identity,
        settings,
        http=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    monkeypatch.setattr(
        "npa.workbench.team.browser.OidcProvider", lambda *args: provider
    )
    service = TeamService(lambda: policy[0])
    app = create_app(None, service=service, verifier=tokens.verifier)
    with TestClient(app, base_url="https://testserver") as client:
        yield SimpleNamespace(
            client=client,
            exchange=exchange,
            policy=policy,
            provider=provider,
            app=app,
            service=service,
        )


def _begin(browser, **claims):
    response = browser.client.get("/auth/login", follow_redirects=False)
    assert response.status_code == 303
    parameters = parse_qs(urlsplit(response.headers["location"]).query)
    browser.exchange["claims"] = {
        "nonce": parameters["nonce"][0],
        "preferred_username": "Alice",
        **claims,
    }
    return parameters


def _finish(browser, parameters):
    return browser.client.get(
        "/auth/callback",
        params={"state": parameters["state"][0], "code": "one-use-code"},
        follow_redirects=False,
    )


def test_verified_login_pkce_private_cookies_and_live_permissions(browser):
    parameters = _begin(browser)
    assert parameters["code_challenge_method"] == ["S256"]
    assert parameters["redirect_uri"] == ["https://testserver/auth/callback"]
    response = _finish(browser, parameters)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    cookie = response.headers.get_list("set-cookie")[-1]
    assert all(
        value in cookie for value in ("HttpOnly", "Secure", "SameSite=lax", "Path=/")
    )
    request = browser.exchange["requests"][0]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(request["code_verifier"][0].encode()).digest()
    )
    assert challenge.rstrip(b"=").decode() == parameters["code_challenge"][0]
    profile = browser.client.get("/v1/me")
    assert profile.headers["cache-control"] == "no-store"
    assert profile.json()["subject"] == "alice"
    assert profile.json()["display_name"] == "Alice"
    assert profile.json()["workspaces"][0]["role"] == "runner"
    assert profile.json()["workspaces"][0]["allocated"] is True
    assert "token" not in profile.text
    assert browser.client.get("/v1/runs?workspace=robotics").status_code == 200
    assert browser.client.get("/v1/runs?workspace=ungranted").status_code == 403


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "attacker"},
        {"azp": "other-client"},
        {"aud": ["workbench", "other-client"]},
        {"aud": "other-client"},
        {"iss": "https://other.example.test"},
        {"exp": 1},
        {"groups": "researchers"},
    ],
)
def test_rejects_unbound_or_invalid_id_tokens(browser, claims):
    response = _finish(browser, _begin(browser, **claims))
    assert response.status_code == 401
    assert browser.client.get("/v1/me").status_code == 401


def test_state_requires_browser_cookie_and_cannot_be_replayed(browser):
    parameters = _begin(browser)
    state = parameters["state"][0]
    browser.client.cookies.clear()
    assert _finish(browser, parameters).status_code == 401
    browser.client.cookies.set(STATE_COOKIE, state)
    assert _finish(browser, parameters).status_code == 303
    browser.client.cookies.set(STATE_COOKIE, state)
    assert _finish(browser, parameters).status_code == 401
    assert len(browser.exchange["requests"]) == 1


def test_rejects_duplicate_callback_parameters_without_exchange(browser):
    parameters = _begin(browser)
    response = browser.client.get(
        "/auth/callback",
        params=[
            ("state", parameters["state"][0]),
            ("state", "other"),
            ("code", "code"),
        ],
    )
    assert response.status_code == 401
    assert browser.exchange["requests"] == []


@pytest.mark.parametrize(
    "origin,proof",
    [(None, True), ("https://other.test", True), ("https://testserver", False)],
)
def test_cookie_authenticated_writes_require_origin_and_csrf(browser, origin, proof):
    _finish(browser, _begin(browser))
    profile = browser.client.get("/v1/me").json()
    headers = {}
    if origin:
        headers["Origin"] = origin
    if proof:
        headers["X-Workbench-CSRF"] = profile["csrf"]
    assert browser.client.post("/auth/logout", headers=headers).status_code == 403
    assert (
        browser.client.post(
            "/v1/runs/run-" + "a" * 32 + "/cancel", headers=headers
        ).status_code
        == 403
    )
    assert browser.client.get("/v1/me").status_code == 200


def test_logout_invalidates_stolen_copy_of_session_cookie(browser):
    _finish(browser, _begin(browser))
    cookie = browser.client.cookies.get(SESSION_COOKIE)
    csrf = browser.client.get("/v1/me").json()["csrf"]
    response = browser.client.post(
        "/auth/logout",
        headers={"Origin": "https://testserver", "X-Workbench-CSRF": csrf},
    )
    assert response.status_code == 204
    browser.client.cookies.set(SESSION_COOKIE, cookie)
    assert browser.client.get("/v1/me").status_code == 401


def test_current_grants_and_disabled_subjects_apply_to_live_session(browser):
    _finish(browser, _begin(browser))
    original = browser.policy[0]
    browser.policy[0] = original.model_copy(update={"workspaces": {}})
    assert browser.client.get("/v1/me").json()["workspaces"] == []
    browser.policy[0] = original.model_copy(update={"disabled_subjects": ("alice",)})
    assert browser.client.get("/v1/me").status_code == 403
    assert browser.client.get("/v1/runs?workspace=robotics").status_code == 403


def test_authenticated_person_without_grant_gets_no_workspace(browser):
    _finish(browser, _begin(browser, subject="eve", groups=[]))
    assert browser.client.get("/v1/me").json()["workspaces"] == []
    assert browser.client.get("/v1/runs?workspace=robotics").status_code == 403


def test_no_cookie_authentication_fallback_from_invalid_bearer(browser):
    _finish(browser, _begin(browser))
    assert (
        browser.client.get(
            "/v1/me", headers={"Authorization": "Bearer invalid"}
        ).status_code
        == 401
    )


def test_portal_is_public_but_private_identity_requires_authentication(browser):
    response = browser.client.get("/")
    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert browser.client.get("/portal.js").status_code == 200
    assert browser.client.get("/portal.css").status_code == 200
    assert browser.client.get("/v1/me").status_code == 401


@pytest.mark.parametrize(
    "changes",
    [
        {"issuer": "https://attacker.test"},
        {"jwks_uri": "https://attacker.test/keys"},
        {"token_endpoint": "http://issuer.test/token"},
        {"code_challenge_methods_supported": ["plain"]},
    ],
)
def test_discovery_must_match_pinned_identity_and_support_pkce(browser, changes):
    document = {**browser.provider.metadata, **changes}
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=document)
        )
    )
    with pytest.raises(AuthenticationError):
        OidcProvider(
            browser.policy[0].identity, browser.policy[0].browser_login, http=http
        )
    http.close()


def test_browser_client_must_use_expected_audience(config):
    data = config.model_dump()
    data["browser_login"] = {
        "public_url": "https://workbench.example.test",
        "client_id": "other",
    }
    with pytest.raises(ValueError, match="audience"):
        TeamConfig.model_validate(data)


def test_sdk_whoami_uses_bearer_auth_without_cloud_credentials():
    def transport(request):
        assert request.url.path == "/v1/me"
        assert request.headers["authorization"] == "Bearer external-token"
        return httpx.Response(200, json={"subject": "verified", "workspaces": []})

    client = TeamClient(
        "https://workbench.example.test",
        "external-token",
        transport=httpx.MockTransport(transport),
    )
    assert client.whoami()["subject"] == "verified"
    client.close()


def test_browser_session_and_login_challenge_expire(browser, monkeypatch):
    import time

    _finish(browser, _begin(browser))
    parameters = _begin(browser)
    future = time.time() + 3600
    monkeypatch.setattr(
        "npa.workbench.team.browser_sessions.time", SimpleNamespace(time=lambda: future)
    )
    assert browser.client.get("/v1/me").status_code == 401
    assert _finish(browser, parameters).status_code == 401


@pytest.mark.parametrize(
    "value",
    [
        "https://:password@workbench.test",
        "http://workbench.test",
        "https://workbench.test/prefix",
    ],
)
def test_browser_origin_rejects_credentials_and_non_origin_urls(value):
    with pytest.raises(ValueError):
        BrowserLogin(public_url=value, client_id="workbench")


def test_unicode_state_is_rejected_without_server_error(browser):
    parameters = _begin(browser)
    parameters["state"] = ["invalid-\u00e9"]
    assert _finish(browser, parameters).status_code == 401
