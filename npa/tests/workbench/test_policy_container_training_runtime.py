"""Bind native training arguments to the installed LeRobot runtime contract."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from npa.workbench.lerobot import policy_container as trainer
from npa.workbench.training_config import TrainingConfig, build_training_config


def _command(**kwargs):
    return trainer.build_lerobot_train_command(
        dataset_path="dataset",
        output_dir="output",
        steps=1,
        dataset_repo_id="synthetic/test",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("version", "flag"),
    [
        ("0.5.1", "--eval_freq=1000000"),
        ("0.6.0", "--env_eval_freq=1000000"),
        ("0.6.1+npa2", "--env_eval_freq=1000000"),
    ],
)
def test_exact_training_versions_select_native_argument(monkeypatch, version, flag):
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    command = _command(lerobot_version=version)
    assert flag in command
    assert (
        sum(arg.startswith(("--eval_freq=", "--env_eval_freq=")) for arg in command)
        == 1
    )


def test_unbound_command_builder_preserves_manifest_default(monkeypatch):
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    assert "--eval_freq=1000000" in _command()
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    assert "--env_eval_freq=1000000" in _command()


def test_derivative_executes_verified_module_in_same_interpreter():
    command = _command(lerobot_version="0.6.1+npa2")
    assert command[:3] == [
        trainer.sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
    ]
    assert _command(lerobot_version="0.5.1")[0] == "lerobot-train"
    assert _command(lerobot_version="0.6.0")[0] == "lerobot-train"


@pytest.mark.parametrize("version", ["0.6.1", "0.6.1+npa1", "0.6.1+npa3", "0.7.0"])
def test_unqualified_versions_remain_unsupported(version):
    with pytest.raises(ValueError, match="Unsupported"):
        _command(lerobot_version=version)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_type": "diffusion"},
        {"extra_args": ["--policy.type=diffusion"]},
        {"extra_args": ["--policy.type", "diffusion"]},
        {"extra_args": ["--policy.type"]},
        {"training_config": TrainingConfig(overrides=("policy.type=diffusion",))},
    ],
)
def test_derivative_rejects_non_act_including_overrides(kwargs):
    with pytest.raises(trainer.PolicyContainerError, match="ACT only"):
        _command(lerobot_version="0.6.1+npa2", **kwargs)


@pytest.mark.parametrize(
    "option",
    [
        "reward_model.type=reward_classifier",
        "reward_model.path=unqualified/reward-model",
        "reward_model=unqualified/reward-config.json",
        "config_path=unqualified/train-config.json",
        "policy=unqualified/policy-config.json",
        "policy.path=unqualified/pretrained-policy",
        "policy.discover_packages_path=unqualified.plugin",
    ],
)
@pytest.mark.parametrize("form", ["equals", "split", "training_config"])
def test_derivative_rejects_alternative_trainable_indirection(option, form):
    if form == "training_config":
        arguments = {"training_config": build_training_config(overrides=[option])}
    elif form == "split":
        key, value = option.split("=", 1)
        arguments = {"extra_args": [f"--{key}", value]}
    else:
        arguments = {"extra_args": [f"--{option}"]}
    with pytest.raises(trainer.PolicyContainerError, match="ACT-only.*indirection"):
        _command(lerobot_version="0.6.1+npa2", **arguments)


@pytest.mark.parametrize("version", ["0.5.1", "0.6.0", "0.6.1+npa2"])
def test_act_hyperparameter_overrides_are_not_configuration_indirection(version):
    config = build_training_config(
        overrides=["policy.type=act", "policy.chunk_size=4", "policy.n_action_steps=2"]
    )
    command = _command(
        lerobot_version=version,
        training_config=config,
        extra_args=["--policy.optimizer_lr", "0.0001", "--seed=595"],
    )
    assert "--policy.chunk_size=4" in command
    assert "--policy.n_action_steps=2" in command
    assert command[command.index("--policy.optimizer_lr") + 1] == "0.0001"
    assert "--seed=595" in command


@pytest.mark.parametrize("version", ["0.5.1", "0.6.0"])
def test_supported_versions_keep_existing_config_and_resume_arguments(version):
    command = _command(
        lerobot_version=version,
        resume=True,
        extra_args=["--config_path", "checkpoint/train_config.json"],
        training_config=build_training_config(
            overrides=["reward_model.type=reward_classifier"]
        ),
    )
    assert "--resume=true" in command
    assert command[command.index("--config_path") + 1] == "checkpoint/train_config.json"
    assert "--reward_model.type=reward_classifier" in command


def _native_source_fixture(tmp_path):
    sources = {
        "lerobot/configs/train.py": b"native parser",
        "lerobot/scripts/lerobot_train.py": b"native trainer",
    }
    modules = {}
    for relative, content in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        modules[relative.removesuffix(".py").replace("/", ".")] = SimpleNamespace(
            __file__=str(path)
        )
    return sources, modules


@pytest.fixture
def installed(monkeypatch, tmp_path):
    metadata = "Name: lerobot\nVersion: 0.6.1+npa2\n"
    sources, modules = _native_source_fixture(tmp_path)
    distribution = SimpleNamespace(
        version="0.6.1+npa2",
        read_text=lambda _: metadata,
        locate_file=lambda relative: tmp_path / relative,
    )
    monkeypatch.setattr(
        trainer.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(trainer.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(
        trainer,
        "_ROBOCASA_ACT_TRAINING_METADATA_SHA256",
        hashlib.sha256(metadata.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        trainer,
        "_ROBOCASA_ACT_TRAINING_SOURCES",
        {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
    )
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    return distribution, modules


def test_installed_exact_derivative_binds_metadata_and_both_sources(installed):
    assert trainer._installed_training_version("act") == "0.6.1+npa2"


@pytest.mark.parametrize("version", ["0.5.1", "0.6.0"])
def test_supported_native_runtime_without_global_default_override(installed, version):
    distribution, _ = installed
    distribution.version = version
    assert trainer._installed_training_version("diffusion") == version


@pytest.mark.parametrize("version", ["0.5.1", "0.6.0", "0.6.1+npa2"])
def test_explicit_selected_version_matches_real_metadata(
    installed, monkeypatch, version
):
    distribution, _ = installed
    distribution.version = version
    monkeypatch.setenv("NPA_LEROBOT_VERSION", version)
    assert trainer._installed_training_version("act") == version


@pytest.mark.parametrize("selected", ["0.5.1", "0.6.0", "0.6.1", "0.6.1+npa3"])
def test_installed_version_mismatch_rejected(installed, monkeypatch, selected):
    monkeypatch.setenv("NPA_LEROBOT_VERSION", selected)
    with pytest.raises(trainer.PolicyContainerError, match="differs from installed"):
        trainer._installed_training_version("act")


@pytest.mark.parametrize(
    "version", ["", None, " 0.6.0", "0.6.1", "0.7.0", "0.6.1+npa3"]
)
def test_unknown_or_invalid_installed_metadata_rejected(installed, version):
    distribution, _ = installed
    distribution.version = version
    with pytest.raises(trainer.PolicyContainerError):
        trainer._installed_training_version("act")


def test_missing_installed_metadata_rejected(monkeypatch):
    def missing(_):
        raise trainer.importlib.metadata.PackageNotFoundError("lerobot")

    monkeypatch.setattr(trainer.importlib.metadata, "distribution", missing)
    with pytest.raises(
        trainer.PolicyContainerError, match="installed package metadata"
    ):
        trainer._installed_training_version("act")


@pytest.mark.parametrize("metadata", [None, "Name: changed\n"])
def test_changed_derivative_metadata_rejected(installed, metadata):
    distribution, _ = installed
    distribution.read_text = lambda _: metadata
    with pytest.raises(trainer.PolicyContainerError, match="metadata does not match"):
        trainer._installed_training_version("act")


@pytest.mark.parametrize(
    "relative", ["lerobot/configs/train.py", "lerobot/scripts/lerobot_train.py"]
)
def test_changed_native_parser_or_trainer_rejected(installed, relative):
    distribution, _ = installed
    distribution.locate_file(relative).write_bytes(b"changed implementation")
    with pytest.raises(trainer.PolicyContainerError, match="source does not match"):
        trainer._installed_training_version("act")


def test_shadowed_native_module_rejected(installed, tmp_path):
    _, modules = installed
    modules["lerobot.configs.train"].__file__ = str(tmp_path / "shadow.py")
    with pytest.raises(trainer.PolicyContainerError, match="module is shadowed"):
        trainer._installed_training_version("act")


def test_runtime_mismatch_stops_before_directory_or_subprocess(
    installed, monkeypatch, tmp_path
):
    monkeypatch.setattr(trainer, "assert_lerobot_importable", lambda: None)
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    monkeypatch.setattr(
        trainer.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess reached"),
    )
    output = tmp_path / "not-created" / "training"
    with pytest.raises(trainer.PolicyContainerError, match="differs from installed"):
        trainer.run_lerobot_training(
            dataset_path=tmp_path,
            output_dir=output,
            steps=1,
            dataset_repo_id="synthetic/test",
        )
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "override",
    [
        "reward_model.type=reward_classifier",
        "reward_model.path=unqualified/reward-model",
        "config_path=unqualified/train-config.json",
    ],
)
def test_indirection_stops_before_training_subprocess_and_log(
    installed, monkeypatch, tmp_path, override
):
    monkeypatch.setattr(trainer, "assert_lerobot_importable", lambda: None)
    monkeypatch.setattr(
        trainer.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("training subprocess reached"),
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    output = tmp_path / "training"
    log = tmp_path / "training.log"
    with pytest.raises(trainer.PolicyContainerError, match="ACT-only.*indirection"):
        trainer.run_lerobot_training(
            dataset_path=dataset,
            output_dir=output,
            log_path=log,
            steps=1,
            dataset_repo_id="synthetic/test",
            training_config=build_training_config(overrides=[override]),
        )
    assert not output.exists()
    assert not log.exists()
