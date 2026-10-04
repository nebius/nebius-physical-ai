"""Prove preference credentials and private images cannot cross implicit routes."""

import ast
from collections import Counter
from dataclasses import replace
import inspect
import json

from PIL import Image
import pytest

from npa.clients import credentials, token_factory
from npa.workbench import vlm_eval


@pytest.fixture(autouse=True)
def _isolated_preference_credentials(monkeypatch):
    for name in (
        "VLM_EVAL_API_KEY",
        "NEBIUS_TOKEN_FACTORY_KEY",
        "OPENAI_API_KEY",
        "VLM_EVAL_API_BASE_URL",
        "OPENAI_BASE_URL",
        *token_factory.BASE_URL_ENV_KEYS,
        "PREFERENCE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _request(tmp_path, **options):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(first)
    Image.new("RGB", (8, 8), "blue").save(second)
    request = vlm_eval.VlmPreferenceComparisonRequest(
        baseline_path=str(first),
        candidate_path=str(second),
        output_path=str(tmp_path / "evidence"),
        task="Compare visible detail.",
        rubric="Use only visible pixels.",
    )
    return replace(request, **options)


def _assert_no_transport(monkeypatch, request):
    calls = []
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(vlm_eval.VlmEvalError) as error:
        vlm_eval.compare_vlm_preference(request)
    assert calls == []
    assert "private-route.invalid" not in str(error.value)
    assert "synthetic-openai-key" not in str(error.value)
    assert "synthetic-nebius-key" not in str(error.value)
    return str(error.value)


@pytest.mark.parametrize(
    "endpoint_env",
    ["VLM_EVAL_API_BASE_URL", "OPENAI_BASE_URL", *token_factory.BASE_URL_ENV_KEYS],
)
@pytest.mark.parametrize("named_key", [None, "VLM_EVAL_API_KEY"])
def test_ambient_endpoint_requires_explicit_destination(
    monkeypatch, tmp_path, endpoint_env, named_key
):
    monkeypatch.setenv(endpoint_env, "https://private-route.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    if named_key:
        monkeypatch.setenv(named_key, "synthetic-nebius-key")
    error = _assert_no_transport(monkeypatch, _request(tmp_path))
    assert "explicit_endpoint" in error


def test_nebius_default_never_falls_back_to_openai_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    error = _assert_no_transport(monkeypatch, _request(tmp_path))
    assert "credential" in error


@pytest.mark.parametrize("api_key_env", ["VLM_EVAL_API_KEY", "PREFERENCE_KEY"])
def test_custom_endpoint_never_falls_back_to_another_key(
    monkeypatch, tmp_path, api_key_env
):
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-nebius-key")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    request = _request(
        tmp_path,
        endpoint_url="https://private-route.invalid/v1",
        api_key_env=api_key_env,
    )
    assert "credential" in _assert_no_transport(monkeypatch, request)


def test_custom_key_requires_explicit_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("PREFERENCE_KEY", "synthetic-openai-key")
    request = _request(tmp_path, api_key_env="PREFERENCE_KEY")
    assert "explicit_endpoint" in _assert_no_transport(monkeypatch, request)


def test_named_key_typo_never_falls_back_on_explicit_nebius_route(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-nebius-key")
    request = _request(
        tmp_path,
        endpoint_url=token_factory.DEFAULT_BASE_URL,
        api_key_env="PREFERENCE_KEY",
    )
    assert "credential" in _assert_no_transport(monkeypatch, request)


def test_nebius_environment_key_precedes_file_and_unrelated_provider_key(monkeypatch):
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-nebius-key")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    monkeypatch.setattr(
        token_factory, "resolve_config", lambda **_: pytest.fail("must not read store")
    )
    assert (
        vlm_eval._preference_api_key(endpoint_url="", api_key_env="VLM_EVAL_API_KEY")
        == "synthetic-nebius-key"
    )


def test_credential_store_failure_is_sanitized_before_transport(monkeypatch, tmp_path):
    def broken_store(**_):
        raise ValueError("synthetic-openai-key at private-route.invalid")

    monkeypatch.setattr(token_factory, "resolve_config", broken_store)
    error = _assert_no_transport(monkeypatch, _request(tmp_path))
    assert error == "preference_credential_configuration_invalid"


def _completion(request):
    return {
        "model": request["model"],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "preference": "tie",
                            "confidence": "high",
                            "observable_support": ["visible pixels in both images"],
                            "critical_defects": {
                                "A": ["No visible defect."],
                                "B": ["No visible defect."],
                            },
                            "uncertainty": "No hidden state is known.",
                        }
                    )
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("endpoint", "key_name", "stored", "expected_url"),
    [
        ("", "VLM_EVAL_API_KEY", False, token_factory.DEFAULT_BASE_URL),
        ("", "NEBIUS_TOKEN_FACTORY_KEY", False, token_factory.DEFAULT_BASE_URL),
        ("", "VLM_EVAL_API_KEY", True, token_factory.DEFAULT_BASE_URL),
        (
            token_factory.DEFAULT_BASE_URL,
            "VLM_EVAL_API_KEY",
            True,
            token_factory.DEFAULT_BASE_URL,
        ),
        (
            "https://custom.invalid/v1",
            "OPENAI_API_KEY",
            False,
            "https://custom.invalid/v1",
        ),
        (
            "https://custom.invalid/v1/chat/completions",
            "PREFERENCE_KEY",
            False,
            "https://custom.invalid/v1",
        ),
        (
            token_factory.DEFAULT_BASE_URL,
            "PREFERENCE_KEY",
            False,
            token_factory.DEFAULT_BASE_URL,
        ),
    ],
)
def test_explicit_supported_routes_keep_exact_key_and_both_requests(
    monkeypatch, tmp_path, endpoint, key_name, stored, expected_url
):
    if endpoint:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
    if stored:
        credentials.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        credentials.CREDENTIALS_PATH.write_text(
            "tokens:\n  NEBIUS_TOKEN_FACTORY_KEY: synthetic-selected-key\n"
        )
        credentials.CREDENTIALS_PATH.chmod(0o600)
    else:
        monkeypatch.setenv(key_name, "synthetic-selected-key")
    request = _request(tmp_path, endpoint_url=endpoint, api_key_env=key_name)
    calls = []

    def post(**kwargs):
        calls.append(kwargs)
        return _completion(kwargs["request"])

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    report = vlm_eval.compare_vlm_preference(request)
    assert report.status == "consistent_tie"
    assert len(calls) == 2
    assert {call["url"] for call in calls} == {
        vlm_eval._chat_completions_url(expected_url)
    }
    assert all(
        call["headers"]["Authorization"] == "Bearer synthetic-selected-key"
        for call in calls
    )
    assert report.requests_counterbalanced is True


def test_vlm_module_has_no_duplicate_top_level_assignment_targets():
    tree = ast.parse(inspect.getsource(vlm_eval))
    targets = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert [name for name, count in Counter(targets).items() if count > 1] == []
