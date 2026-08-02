"""A self-hosted VLM stage must start the server it is about to call.

``vlm_backend: self-hosted`` makes the tool POST to an OpenAI-compatible endpoint on
localhost, and nothing in a spec started one — so ``vlm-eval-single.yaml`` failed live
with ``VLM backend request failed: [Errno 111] Connection refused`` (EVIDENCE §5.2b). The
retired ``vlm-eval.yaml`` template did the serve/wait/teardown in its ``run:`` block,
which is exactly the kind of bash a ``toolRef`` cannot carry.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    DEFAULT_VLM_SERVE_PORT,
    NpaWorkflowRenderError,
    DEFAULT_VLM_SERVER_READY_SECONDS,
    VLM_SERVER_POLL_INTERVAL_SECONDS,
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_run_preamble_for_tool,
    render_self_hosted_vlm_preamble,
    render_skypilot_yaml,
    render_task_run_script,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workbench.vlm_eval import DEFAULT_ENDPOINT_URL, DEFAULT_MODEL

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def test_preamble_serves_the_model_the_tool_will_ask_for() -> None:
    """Defaults must line up with the tool's, so a spec needs no extra config."""

    preamble = render_self_hosted_vlm_preamble({})

    assert f"\nnpa_vlm_model={shlex.quote(DEFAULT_MODEL)}\n" in preamble
    assert f"\nnpa_vlm_port={shlex.quote(str(DEFAULT_VLM_SERVE_PORT))}\n" in preamble
    # DEFAULT_ENDPOINT_URL is what the tool posts to when --endpoint-url is empty.
    assert f":{DEFAULT_VLM_SERVE_PORT}/" in DEFAULT_ENDPOINT_URL


def test_preamble_starts_waits_and_tears_down() -> None:
    preamble = render_self_hosted_vlm_preamble({})

    assert "vllm.entrypoints.openai.api_server" in preamble
    # Backgrounded, then trapped so neither success nor failure leaks a GPU-resident server.
    assert preamble.count("&\n") >= 1
    assert "trap 'kill \"$npa_vlm_pid\" 2>/dev/null || true' EXIT" in preamble
    assert "/health" in preamble
    expected_attempts = DEFAULT_VLM_SERVER_READY_SECONDS // VLM_SERVER_POLL_INTERVAL_SECONDS
    assert f"attempts, interval = {expected_attempts}, {VLM_SERVER_POLL_INTERVAL_SECONDS}" in preamble


def test_ready_window_is_configurable_and_validated() -> None:
    """A cold multi-GB checkpoint download is the slow part; specs may need longer."""

    preamble = render_self_hosted_vlm_preamble({"vlm_serve_ready_seconds": 60})

    assert f"attempts, interval = {60 // VLM_SERVER_POLL_INTERVAL_SECONDS}" in preamble

    with pytest.raises(NpaWorkflowRenderError, match="at least"):
        render_self_hosted_vlm_preamble({"vlm_serve_ready_seconds": 1})
    with pytest.raises(NpaWorkflowRenderError, match="integer number of seconds"):
        render_self_hosted_vlm_preamble({"vlm_serve_ready_seconds": "soon"})


def test_preamble_fails_fast_when_the_server_dies() -> None:
    """A dead server must surface its log, not time out silently for ten minutes."""

    preamble = render_self_hosted_vlm_preamble({})

    assert "exited before becoming ready" in preamble
    assert "readlines()[-200:]" in preamble
    assert "SystemExit(1)" in preamble


def test_preamble_puts_the_interpreters_script_dir_on_path() -> None:
    """vLLM JIT-compiles with `ninja`; pip installs it beside the interpreter (job 214)."""

    preamble = render_self_hosted_vlm_preamble({})

    assert 'sysconfig.get_path("scripts")' in preamble
    assert "export PATH=$npa_vlm_scripts:$PATH" in preamble
    # Must happen before the server starts, or the JIT still cannot find ninja.
    assert preamble.index("export PATH=") < preamble.index("api_server")


def test_preamble_survives_an_image_with_no_cuda_toolchain() -> None:
    """Job 217: ninja ran, then failed with '/usr/local/cuda/bin/nvcc: not found'."""

    preamble = render_self_hosted_vlm_preamble({})

    # Falls back to the nvcc wheel vLLM already depends on ...
    assert "if [ ! -x /usr/local/cuda/bin/nvcc ]; then" in preamble
    assert "'nvidia' / 'cuda_nvcc'" in preamble
    assert "export CUDA_HOME=$npa_cuda_home" in preamble
    # ... and disables the sampler that wants the JIT, so a compiler-less image cannot
    # break startup at all.
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in preamble
    assert preamble.index("VLLM_USE_FLASHINFER_SAMPLER") < preamble.index("api_server")


def test_flashinfer_sampler_can_be_opted_back_in() -> None:
    preamble = render_self_hosted_vlm_preamble({"vlm_use_flashinfer_sampler": "1"})

    assert "export VLLM_USE_FLASHINFER_SAMPLER=1" in preamble


def test_setup_installs_ninja_for_the_jit_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="render-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    setup = [doc for doc in yaml.safe_load_all(text) if doc][1]["setup"]
    # #238's installer resolves with uv first and falls back to pip; both forms name the pin.
    assert "vllm>=0.8.5" in setup
    assert "pip_install('ninja')" in setup
    # Only when it is missing: workbench images may already ship a toolchain.
    assert "shutil.which('ninja') is None" in setup


def test_preamble_uses_python_not_curl_for_the_health_wait() -> None:
    """curl is not guaranteed in every task image; python3 is (setup records one)."""

    preamble = render_self_hosted_vlm_preamble({})

    assert "curl" not in preamble
    assert "urllib.request" in preamble


def test_preamble_has_no_braced_expansion() -> None:
    """A `${var}` would trip the rendered-YAML placeholder guard."""

    assert_no_unresolved_placeholders(render_self_hosted_vlm_preamble({}))
    assert "${" not in render_self_hosted_vlm_preamble({})


def test_config_overrides_model_port_and_trust() -> None:
    preamble = render_self_hosted_vlm_preamble(
        {"vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct", "vlm_serve_port": "9001"}
    )

    assert "\nnpa_vlm_model=Qwen/Qwen2.5-VL-7B-Instruct\n" in preamble
    assert "\nnpa_vlm_port=9001\n" in preamble
    assert "--trust-remote-code" in preamble

    without_trust = render_self_hosted_vlm_preamble({"vlm_trust_remote_code": "0"})
    assert "--trust-remote-code" not in without_trust


@pytest.mark.parametrize(
    ("tool_ref", "config", "expected"),
    [
        ("workbench.vlm_eval.run", {"vlm_backend": "self-hosted"}, True),
        ("workbench.vlm_eval.run", {"vlm_backend": "self_hosted"}, True),
        ("workbench.vlm_eval.benchmark", {"vlm_backend": "self-hosted"}, True),
        # The hosted backend needs no server.
        ("workbench.vlm_eval.run", {"vlm_backend": "api"}, False),
        ("workbench.vlm_eval.run", {}, False),
        # Unrelated tools never get a preamble.
        ("workbench.sonic.eval", {"vlm_backend": "self-hosted"}, False),
        ("", {"vlm_backend": "self-hosted"}, False),
    ],
)
def test_preamble_is_scoped_to_self_hosted_vlm_stages(
    tool_ref: str, config: dict, expected: bool
) -> None:
    rendered = render_run_preamble_for_tool(tool_ref, config=config)

    assert bool(rendered) is expected


def test_run_script_places_the_preamble_after_the_interpreter_shim() -> None:
    """The server must start with the same python3 the command will use."""

    script = render_task_run_script(["npa", "workbench", "vlm-eval", "run"], preamble="PREAMBLE\n")

    assert "PREAMBLE" in script
    assert script.index("/tmp/npa-python") < script.index("PREAMBLE")
    assert script.index("PREAMBLE") < script.index("npa workbench vlm-eval run")


def test_run_script_without_a_preamble_is_unchanged() -> None:
    """Additive: every stage that needs no preamble renders exactly as before."""

    command = ["npa", "workbench", "mjlab", "eval"]

    assert render_task_run_script(command) == render_task_run_script(command, preamble="")


def test_shipped_self_hosted_spec_renders_a_server_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the spec that failed live now renders the server bootstrap."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="render-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    assert spec.config["vlm_backend"] == "self-hosted"
    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    run_script = docs[1]["run"]
    assert "vllm.entrypoints.openai.api_server" in run_script
    assert "npa workbench vlm-eval run" in run_script
    assert run_script.index("api_server") < run_script.index("npa workbench vlm-eval run")
    assert_no_unresolved_placeholders(text)


def test_hosted_spec_renders_no_server_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "vlm-eval-token-factory.yaml")
    plan = build_plan(spec, run_id="render-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    assert "vllm" not in text
