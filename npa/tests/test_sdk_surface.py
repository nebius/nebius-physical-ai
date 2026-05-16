"""Tests for the public SDK surface."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType


def _public_modules() -> list[ModuleType]:
    from npa import convert, demo, network, rerun, workflow, workbench

    return [
        convert,
        demo,
        rerun,
        network,
        workflow,
        workbench.cosmos,
        workbench.fiftyone,
        workbench.genesis,
        workbench.groot,
        workbench.isaac_lab,
        workbench.lancedb,
        workbench.lerobot,
    ]


def test_top_level_imports_are_modules() -> None:
    """All top-level SDK namespaces are importable modules."""
    from npa import convert, demo, errors, network, rerun, workflow, workbench

    for module in [convert, demo, rerun, workbench, network, workflow, errors]:
        assert inspect.ismodule(module)


def test_convert_public_surface() -> None:
    """npa.convert exposes the supported conversion commands."""
    from npa import convert

    assert convert.__all__ == ["lerobot_to_mp4", "lerobot_to_rrd"]
    assert callable(convert.lerobot_to_mp4)
    assert callable(convert.lerobot_to_rrd)


def test_demo_public_surface() -> None:
    """npa.demo exposes stage and verify."""
    from npa import demo

    assert demo.__all__ == ["stage", "verify"]
    assert callable(demo.stage)
    assert callable(demo.verify)


def test_rerun_public_surface() -> None:
    """npa.rerun exposes hosted Rerun sharing commands."""
    from npa import rerun

    assert rerun.__all__ == ["host", "share", "list_shares", "revoke"]
    for name in rerun.__all__:
        assert callable(getattr(rerun, name))


def test_workbench_public_surface() -> None:
    """npa.workbench exposes tool submodules with their command wrappers."""
    from npa import workbench

    expected = {
        "cosmos": ["deploy", "serve", "infer", "status", "system_info"],
        "fiftyone": ["deploy", "launch", "load_dataset", "status", "system_info"],
        "genesis": ["train_teacher", "generate_demos", "simulate", "deploy"],
        "groot": ["deploy", "serve", "infer", "convert", "status"],
        "isaac_lab": ["deploy", "train", "eval", "export_lerobot", "status"],
        "lancedb": ["import_bdd100k"],
        "lerobot": ["deploy", "train", "eval", "serve", "infer"],
    }
    for tool, names in expected.items():
        tool_module = getattr(workbench, tool)
        assert inspect.ismodule(tool_module)
        for name in names:
            assert callable(getattr(tool_module, name)), f"{tool}.{name} missing"


def test_errors_public_surface() -> None:
    """npa.errors exposes public exception types."""
    from npa.errors import NpaError, ScopedCredentialError

    assert issubclass(ScopedCredentialError, NpaError)
    assert issubclass(NpaError, Exception)


def test_sdk_compatibility_namespace_exposes_lancedb_import() -> None:
    from npa.sdk.workbench.lancedb import import_bdd100k

    assert callable(import_bdd100k)


def test_public_functions_have_docstrings_and_no_typer_signature_leaks() -> None:
    """Public SDK functions do not expose Typer defaults or annotations."""
    forbidden = {"Option", "Argument", "OptionInfo", "ArgumentInfo", "typer"}
    for module in _public_modules():
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            assert obj.__doc__, f"{module.__name__}.{name} missing docstring"
            sig = inspect.signature(obj)
            sig_text = str(sig)
            for token in forbidden:
                assert token not in sig_text, (
                    f"{module.__name__}.{name} leaks Typer in {sig_text}"
                )


def test_convert_lerobot_to_mp4_delegates_to_adapter(mocker) -> None:
    """npa.convert.lerobot_to_mp4 calls the underlying render implementation."""
    from npa import convert
    from npa.adapter.lerobot.render import LeRobotMP4RenderResult

    rendered = LeRobotMP4RenderResult(
        local_path=Path("/tmp/out.mp4"),
        saved_to="/tmp/out.mp4",
        duration_s=1.0,
        resolution=(1280, 720),
        fps=30,
        frame_count=30,
    )
    mock_render = mocker.patch(
        "npa.convert.render_lerobot_to_mp4_result", return_value=rendered
    )

    result = convert.lerobot_to_mp4(
        input_path="dataset",
        output_path="/tmp/out.mp4",
        renderer="matplotlib",
        duration=1.0,
    )

    assert result is rendered
    mock_render.assert_called_once()
    assert mock_render.call_args.kwargs["input_path"] == "dataset"
    assert mock_render.call_args.kwargs["renderer"] == "matplotlib"


def test_convert_lerobot_to_rrd_delegates_to_adapter(mocker) -> None:
    """npa.convert.lerobot_to_rrd calls the Rerun adapter."""
    from npa import convert

    mock_adapter = mocker.patch("npa.convert.lerobot_to_rerun")

    result = convert.lerobot_to_rrd(
        input_path="dataset",
        output_path="/tmp/out.rrd",
        duration=2.0,
    )

    assert result == Path("/tmp/out.rrd")
    mock_adapter.assert_called_once_with("dataset", Path("/tmp/out.rrd"), duration_s=2.0)


def test_demo_stage_delegates_to_stage_artifacts(mocker, tmp_path: Path) -> None:
    """npa.demo.stage forwards SDK arguments to stage_artifacts."""
    from npa import demo

    manifest = tmp_path / "manifest.yaml"
    mock_stage = mocker.patch("npa.demo.stage_artifacts", return_value=[])

    assert demo.stage(target_bucket="bucket", manifest=manifest) == []
    mock_stage.assert_called_once()
    assert mock_stage.call_args.kwargs["target_bucket"] == "bucket"
    assert mock_stage.call_args.kwargs["manifest_path"] == manifest


def test_rerun_host_delegates_to_host_recording(mocker) -> None:
    """npa.rerun.host forwards SDK arguments to host_recording."""
    from npa import rerun

    mock_host = mocker.patch("npa.rerun.host_recording", return_value="result")

    assert rerun.host("recording.rrd", target_bucket="bucket") == "result"
    mock_host.assert_called_once()
    assert mock_host.call_args.args == ("recording.rrd",)
    assert mock_host.call_args.kwargs["target_bucket"] == "bucket"


def test_network_ensure_ingress_parses_cli_style_ports(mocker) -> None:
    """npa.network.ensure_ingress accepts CLI-style comma-separated ports."""
    from npa import network

    mock_ensure = mocker.patch("npa.network._ensure_ingress", return_value="ok")

    assert network.ensure_ingress(vm="vm-id", ports="5151,8080") == "ok"
    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.kwargs["ports"] == [5151, 8080]
