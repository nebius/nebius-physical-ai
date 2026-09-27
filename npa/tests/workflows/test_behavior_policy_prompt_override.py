"""Exercise the TRAIN-only literal prompt override across serving and recording."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.behavior_challenge import __main__ as behavior_main
from npa.workflows.behavior_challenge import campaign_runner
from npa.workflows.behavior_challenge import native_comet_server
from npa.workflows.behavior_challenge import policy_prompt
from npa.workflows.behavior_challenge import serving_identity
from npa.workflows.behavior_challenge import train_experience
from npa.workflows.behavior_challenge import train_prompt
from npa.workflows.behavior_challenge import trained_comet_policy


PROMPT = "picking_up_trash"
RELEASED_PROMPT = (
    "Put the three can of soda from the living room inside the tash can in the kitchen."
)
DIGEST = "1" * 64


def _prompt_binding(prompt: str = PROMPT, source_kind: str = "literal_override"):
    payload = {
        "schema": train_prompt.SCHEMA,
        "task_name": "picking_up_trash",
        "task_id": 1,
        "effective_prompt": prompt,
        "source_kind": source_kind,
        "source_files": {
            name: {"bytes": 1, "sha256": DIGEST}
            for name in (
                "task_mapping",
                "wrapper",
                "tokenizer",
                "transforms",
                "training_config",
            )
        },
        "tokenizer_contract": {
            "implementation": "PaligemmaTokenizer",
            "model_uri": "gs://big_vision/paligemma_tokenizer.model",
            "model_bytes_status": "not_observed_bind_at_training_projection",
        },
    }
    return {
        **payload,
        "binding_sha256": native_comet_server._canonical_digest(payload),
    }


def _args(**changes):
    values = {
        "policy_kind": "comet-trained",
        "train_experience": True,
        "policy_prompt_override": PROMPT,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _server_args(**changes):
    values = {
        "case_id": "case-1",
        "task_name": "picking_up_trash",
        "task_prompt_override": PROMPT,
        "instance_id": 200,
        "rollout_id": 0,
        "case_seed": 7,
        "checkpoint_sha256": DIGEST,
        "rng_contract_sha256": "2" * 64,
        "trace_configuration_sha256": "3" * 64,
        "action_trace": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_prompt_override_is_recorded_train_only():
    assert policy_prompt.prompt_override(_args(), train_panel=True) == PROMPT
    for changes, train_panel in (
        ({"policy_kind": "comet50"}, True),
        ({"train_experience": False}, True),
        ({}, False),
    ):
        with pytest.raises(ValueError, match="recorded comet-trained TRAIN"):
            policy_prompt.prompt_override(_args(**changes), train_panel=train_panel)


@pytest.mark.parametrize("value", ("", " prompt", "prompt ", "bad\nprompt"))
def test_prompt_override_rejects_noncanonical_literals(value):
    with pytest.raises(ValueError, match="canonical literal"):
        policy_prompt.prompt_override(
            _args(policy_prompt_override=value), train_panel=True
        )


def test_cli_parses_literal_override():
    parser = argparse.ArgumentParser()
    behavior_main._add_policy_arguments(parser)
    args = parser.parse_args(
        [
            "--policy-prompt-override",
            PROMPT,
        ]
    )
    assert args.policy_prompt_override == PROMPT


def test_scope_rejects_before_worker_startup(monkeypatch):
    monkeypatch.setattr(
        campaign_runner,
        "_validate_worker_startup_binding",
        lambda *_: (_ for _ in ()).throw(AssertionError("startup reached")),
    )
    panel = {"schema": "npa.behavior.campaign-panel.v1", "split": "development"}
    with pytest.raises(ValueError, match="TRAIN"):
        campaign_runner._execute_partition(
            _args(), panel, {}, object(), Path("/workspace")
        )


def test_literal_changes_serving_identity_without_changing_default(monkeypatch):
    monkeypatch.setattr(serving_identity, "_serving_source", lambda _: {"x": DIGEST})
    monkeypatch.setattr(serving_identity, "_trained_configuration", lambda _: {})
    base = _args(policy_prompt_override=None, policy_archive="archive")
    base.policy_execution_variant = "native"
    base.policy_task_name = "picking_up_trash"
    override = copy.copy(base)
    override.policy_prompt_override = PROMPT
    first = serving_identity.serving_artifact(base)
    assert serving_identity.serving_artifact(copy.copy(base)) == first
    assert serving_identity.serving_artifact(override) != first


def test_server_command_and_process_identity_bind_literal(tmp_path):
    args = _args(
        policy_python=Path("/policy/python"),
        policy_root=Path("/policy/root"),
        policy_checkpoint=Path("/checkpoint"),
        port=8080,
    )
    admission = {
        "manager_step": 20000,
        "normalization": {"asset_id": "asset"},
        "task_id": 1,
        "task": "picking_up_trash",
        "serving_tree_sha256": DIGEST,
        "rng_contract": {"sha256": "2" * 64},
        "trace": {"enabled": False},
    }
    case = {"case_id": "case-1", "instance_id": 200, "rollout_id": 0}
    plan = {"upstream_commit": "6" * 40}
    command = trained_comet_policy._server_identity_arguments(
        args, plan, tmp_path, admission, case, 7
    )
    assert command[-2:] == ["--task-prompt-override", PROMPT]
    server = _server_args()
    payload = native_comet_server._process_identity_payload(server, "4" * 64)
    assert payload["task_prompt_override"] == PROMPT


def test_loader_keeps_slug_lookup_then_overrides_effective_prompt(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    overlay = tmp_path / "overlay"
    (source / "scripts").mkdir(parents=True)
    (source / "packages/openpi-client/src").mkdir(parents=True)
    (checkpoint / "20000/assets/asset").mkdir(parents=True)
    overlay.mkdir()
    mapping = {
        "picking_up_trash": {
            "task_index": 1,
            "task": RELEASED_PROMPT,
        }
    }
    (source / "scripts/task_mapping.json").write_text(json.dumps(mapping) + "\n")
    for relative in (
        "src/openpi/shared/eval_b1k_wrapper.py",
        "src/openpi/models/tokenizer.py",
        "src/openpi/transforms.py",
        "src/openpi/training/config.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")
    (checkpoint / "20000/assets/asset/norm_stats.json").write_text("{}\n")
    policy = SimpleNamespace(_rng=None)
    observed = {}

    def wrapper(value, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            policy=value,
            task_name=kwargs["task_name"],
            task_prompt=RELEASED_PROMPT,
        )

    random = SimpleNamespace(
        key=lambda seed: np.asarray([0, seed], dtype=np.uint32),
        key_data=np.asarray,
    )
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(random=random))
    monkeypatch.setitem(
        sys.modules,
        "openpi.policies.policy_config",
        SimpleNamespace(create_trained_policy=lambda *_: policy),
    )
    monkeypatch.setitem(
        sys.modules,
        "openpi.shared.eval_b1k_wrapper",
        SimpleNamespace(B1KPolicyWrapper=wrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "openpi.training.config",
        SimpleNamespace(get_config=lambda *_: object()),
    )
    args = SimpleNamespace(
        source_root=source,
        task_name="picking_up_trash",
        task_prompt_override=PROMPT,
        task_id=1,
        checkpoint=checkpoint,
        manager_step=20000,
        asset_id="asset",
        case_seed=17,
        train_experience_root=Path("/experience"),
    )
    wrapped, loaded = native_comet_server._load_policy(args, overlay)
    assert loaded is policy
    assert wrapped.policy is policy
    assert wrapped.task_prompt == PROMPT
    assert observed == {
        "task_name": PROMPT,
        "control_mode": "receeding_horizon",
        "max_len": 32,
        "fine_grained_level": 0,
    }


def test_default_binding_verifies_constructor_prompt_without_overwriting() -> None:
    args = SimpleNamespace(task_name="picking_up_trash")
    binding = _prompt_binding(RELEASED_PROMPT, "task_mapping_default")
    wrapper = SimpleNamespace(task_name="picking_up_trash", task_prompt=RELEASED_PROMPT)
    native_comet_server._apply_prompt_binding(wrapper, args, binding)
    assert wrapper.task_prompt == RELEASED_PROMPT

    mismatched = SimpleNamespace(
        task_name="picking_up_trash", task_prompt="different constructor prompt"
    )
    with pytest.raises(ValueError, match="effective prompt differs"):
        native_comet_server._apply_prompt_binding(mismatched, args, binding)
    assert mismatched.task_prompt == "different constructor prompt"


def test_load_qualification_rejects_changed_literal():
    args = _server_args(
        manager_step=20000,
        asset_id="asset",
        initial_rng_sha256="4" * 64,
        process_identity_sha256="5" * 64,
        effective_prompt_binding=_prompt_binding(),
        train_experience_root=Path("/experience"),
    )
    value = native_comet_server._load_qualification(args)
    expected = dict(value)
    expected.pop("initial_rng_sha256")
    expected.pop("process_identity_sha256")
    payload = native_comet_server._process_identity_payload(
        args, args.initial_rng_sha256
    )
    value["process_identity_sha256"] = native_comet_server._canonical_digest(payload)
    assert native_comet_server.validate_load_qualification(value, expected) == value
    changed = dict(value, task_prompt_override="different")
    with pytest.raises(ValueError, match="qualification differs"):
        native_comet_server.validate_load_qualification(changed, expected)


def test_nonrecording_qualification_keeps_legacy_shape() -> None:
    args = _server_args(
        manager_step=20000,
        asset_id="asset",
        initial_rng_sha256="4" * 64,
        process_identity_sha256="5" * 64,
        effective_prompt_binding=_prompt_binding(),
        train_experience_root=None,
    )
    assert "prompt_binding" not in native_comet_server._load_qualification(args)


def test_experience_config_preserves_literal_and_default_shape():
    value = {
        "schema": train_experience.CONFIG_SCHEMA,
        "status": "train_only_recording_enabled",
        "split": "train",
        "cadence": train_experience.DECISION_CADENCE,
        "action_horizon": train_experience.ACTION_HORIZON,
        "include_depth": False,
        "case": {
            "case_id": "case-1",
            "task": "picking_up_trash",
            "instance_id": 200,
            "rollout_id": 0,
            "split": "train",
        },
        "panel_sha256": DIGEST,
        "policy_identity_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "rng_contract_sha256": "4" * 64,
        "source_commit": "5" * 40,
        "prompt_binding": _prompt_binding(),
    }
    assert train_experience.validate_experience_config(value) == value
    legacy = dict(value)
    legacy["schema"] = train_experience.LEGACY_CONFIG_SCHEMA
    legacy.pop("prompt_binding")
    assert train_experience.validate_experience_config(legacy) == legacy
    override = dict(legacy, policy_prompt_override=PROMPT)
    assert train_experience.validate_experience_config(override) == override
    with pytest.raises(ValueError, match="prompt override differs"):
        train_experience.validate_experience_config(
            dict(legacy, policy_prompt_override=None)
        )
