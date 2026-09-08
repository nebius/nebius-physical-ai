"""Keep explicit video settings through the upstream tracked-argument boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

import pytest

from npa.workbench.cosmos import nano_video_engine as engine
from npa.workbench.cosmos import nano_video_server as server


@pytest.fixture
def upstream(monkeypatch):
    events = []

    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            parsed = super().parse_args(args, namespace)
            parsed.explicit_keys = frozenset(vars(parsed))
            return parsed

    class Command:
        def subparser_init(self, subparsers):
            parser = subparsers.add_parser("serve")
            parser.add_argument("model_tag")
            for flag in ("omni", "no-guardrails", "enable-diffusion-pipeline-profiler"):
                parser.add_argument("--" + flag, action="store_true")
            for flag in ("model-class-name", "host", "port", "tensor-parallel-size", "dtype", "init-timeout"):
                parser.add_argument("--" + flag)
            return parser

        def validate(self, args):
            events.append(("validate", args))

        def cmd(self, args):
            events.append(("serve", args))

    modules = {
        "vllm.entrypoints.serve.utils.api_utils": {"cli_env_setup": lambda: events.append(("environment", None))},
        "vllm_omni.entrypoints.cli.serve": {"OmniServeCommand": Command},
        "vllm_omni.utils.tracking_parser": {"TrackingArgumentParser": Parser},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return events


def test_video_model_settings_survive_upstream_explicit_argument_filter(upstream):
    argv = server.server_argv(Path("/models/Cosmos3-Nano"), 18080)[3:]
    command, args = engine.create_command(argv)
    explicit = {key: value for key, value in vars(args).items() if key in args.explicit_keys}
    assert explicit["model_config"] == {"sound_gen": False, "guardrails": False}
    assert explicit["dtype"] == "bfloat16"
    assert explicit["tensor_parallel_size"] == "1"
    assert explicit["model_class_name"] == server.PIPELINE
    assert explicit["enable_diffusion_pipeline_profiler"] is True
    assert [event for event, _ in upstream] == ["environment", "validate"]
    assert upstream[-1][1] is args
    assert callable(command.cmd)


def test_launcher_invokes_validated_upstream_service(upstream, monkeypatch):
    argv = server.server_argv(Path("/models/Cosmos3-Nano"), 18080)[3:]
    monkeypatch.setattr(sys, "argv", ["nano_video_engine", *argv])
    engine.main()
    assert [event for event, _ in upstream] == ["environment", "validate", "serve"]
    assert upstream[-1][1] is upstream[-2][1]
    assert upstream[-1][1].model_config == {"sound_gen": False, "guardrails": False}


def test_obsolete_stage_flag_is_rejected_before_service_launch(upstream):
    with pytest.raises(SystemExit) as failure:
        engine.create_command(["serve", "/models/Cosmos3-Nano", "--stage-configs-path", "old.json"])
    assert failure.value.code == 2
    assert [event for event, _ in upstream] == ["environment"]


def test_upstream_validation_failure_prevents_service_launch(upstream, monkeypatch):
    command = sys.modules["vllm_omni.entrypoints.cli.serve"].OmniServeCommand

    def reject(self, args):
        raise ValueError("invalid upstream configuration")

    monkeypatch.setattr(command, "validate", reject)
    monkeypatch.setattr(sys, "argv", ["nano_video_engine", "serve", "/models/Cosmos3-Nano"])
    with pytest.raises(ValueError, match="invalid upstream configuration"):
        engine.main()
    assert [event for event, _ in upstream] == ["environment"]
