"""Exercise exact TRAIN prompt resolution and legacy evidence derivation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from copy import deepcopy
import hashlib

import pytest

from npa.workflows.behavior_challenge import train_prompt
from npa.workflows.behavior_challenge.autonomous_training_data import (
    _dataset_prompt_binding,
)


DIGEST = "a" * 64
TASK = "picking_up_trash"
PROMPT = "Put the three cans in the trash can."


def _source(root: Path) -> tuple[Path, str]:
    mapping = {TASK: {"task_index": 1, "task": PROMPT}}
    path = root / "scripts/task_mapping.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(mapping) + "\n")
    for relative in (
        "src/openpi/shared/eval_b1k_wrapper.py",
        "src/openpi/models/tokenizer.py",
        "src/openpi/transforms.py",
        "src/openpi/training/config.py",
    ):
        member = root / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(f"# {relative}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=NPA Test",
            "-c",
            "user.email=npa@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, revision


def _legacy(source_commit: str, override: str | None = None) -> tuple[dict, dict]:
    config = {
        "schema": "npa.behavior.train-experience-config.v1",
        "status": "train_only_recording_enabled",
        "split": "train",
        "case": {
            "case_id": "train-1",
            "task": TASK,
            "instance_id": 200,
            "rollout_id": 0,
        },
        "checkpoint_sha256": DIGEST,
        "rng_contract_sha256": "b" * 64,
        "source_commit": source_commit,
    }
    qualification = {
        "schema": "npa.behavior.comet-native-serving-load-qualification.v1",
        "status": "checkpoint_loaded_with_explicit_case_rng",
        "case_id": "train-1",
        "task": TASK,
        "instance_id": 200,
        "rollout_id": 0,
        "checkpoint_content_sha256": DIGEST,
        "rng_contract_sha256": "b" * 64,
        "source_commit": source_commit,
    }
    if override is not None:
        config["policy_prompt_override"] = override
        qualification["task_prompt_override"] = override
    return config, qualification


def test_default_and_override_bind_exact_source(tmp_path: Path) -> None:
    source, _revision = _source(tmp_path / "source")
    default = train_prompt.effective_prompt_binding(source, TASK, 1, None)
    override = train_prompt.effective_prompt_binding(source, TASK, 1, TASK)

    assert default["effective_prompt"] == PROMPT
    assert default["source_kind"] == "task_mapping_default"
    assert override["effective_prompt"] == TASK
    assert override["source_kind"] == "literal_override"
    assert train_prompt.validate_prompt_binding(default) == default
    assert default["binding_sha256"] != override["binding_sha256"]


def test_source_or_prompt_mutation_rejects(tmp_path: Path) -> None:
    binding = train_prompt.effective_prompt_binding(
        _source(tmp_path / "source")[0], TASK, 1, None
    )
    changed_prompt = dict(binding, effective_prompt="different")
    with pytest.raises(ValueError, match="digest"):
        train_prompt.validate_prompt_binding(changed_prompt)
    changed_source = json.loads(json.dumps(binding))
    changed_source["source_files"]["wrapper"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="digest"):
        train_prompt.validate_prompt_binding(changed_source)


def test_legacy_override_derives_without_rewriting_recording(tmp_path: Path) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision, TASK)
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    before = config_path.read_bytes()

    binding = train_prompt.derive_legacy_prompt_binding(
        config_path, qualification_path, source
    )

    assert binding["prompt_binding"]["effective_prompt"] == TASK
    assert config_path.read_bytes() == before
    assert train_prompt.validate_legacy_prompt_derivation(binding, config_path)
    config_path.write_text(json.dumps({**config, "checkpoint_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="derivation differs"):
        train_prompt.validate_legacy_prompt_derivation(binding, config_path)


def test_legacy_default_requires_matching_runtime_prompt(tmp_path: Path) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision)
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    log_path = tmp_path / "policy.log"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    with pytest.raises(ValueError, match="runtime log artifact"):
        train_prompt.derive_legacy_prompt_binding(
            config_path, qualification_path, source
        )
    log_path.write_text(f"INFO:policy:self.task_prompt={PROMPT!r}\n")
    binding = train_prompt.derive_legacy_prompt_binding(
        config_path, qualification_path, source, observed_prompt_artifact=log_path
    )
    assert binding["prompt_binding"]["effective_prompt"] == PROMPT
    assert binding["observed_prompt_artifact"]["sha256"]


def test_legacy_observed_prompt_rejects_incidental_or_conflicting_text(
    tmp_path: Path,
) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision)
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    log_path = tmp_path / "policy.log"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    for content in (
        f"incidental expected text: {PROMPT}\n",
        f"INFO:policy:self.task_prompt={'different'!r}\nexpected={PROMPT!r}\n",
        f"INFO:policy:self.task_prompt={PROMPT!r}\n"
        "INFO:policy:self.task_prompt='different'\n",
    ):
        log_path.write_text(content)
        with pytest.raises(ValueError, match="observed prompt"):
            train_prompt.derive_legacy_prompt_binding(
                config_path,
                qualification_path,
                source,
                observed_prompt_artifact=log_path,
            )


def test_legacy_override_does_not_accept_logged_constructor_default(
    tmp_path: Path,
) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision, TASK)
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    log_path = tmp_path / "policy.log"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    log_path.write_text(f"INFO:policy:self.task_prompt={PROMPT!r}\n")
    with pytest.raises(ValueError, match="observed prompt"):
        train_prompt.derive_legacy_prompt_binding(
            config_path,
            qualification_path,
            source,
            observed_prompt_artifact=log_path,
        )


def test_legacy_lineage_mutation_rejects(tmp_path: Path) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision, TASK)
    qualification["checkpoint_content_sha256"] = "e" * 64
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    with pytest.raises(ValueError, match="lineage"):
        train_prompt.derive_legacy_prompt_binding(
            config_path, qualification_path, source
        )


def test_dataset_requires_externally_pinned_legacy_derivation(tmp_path: Path) -> None:
    source, revision = _source(tmp_path / "source")
    config, qualification = _legacy(revision, TASK)
    config_path = tmp_path / "config.json"
    qualification_path = tmp_path / "qualification.json"
    config_path.write_text(json.dumps(config) + "\n")
    qualification_path.write_text(json.dumps(qualification) + "\n")
    receipt = train_prompt.derive_legacy_prompt_binding(
        config_path, qualification_path, source
    )
    admitted = train_prompt.legacy_prompt_derivation_sha256(receipt)
    assert (
        _dataset_prompt_binding(config, receipt, admitted, config_path)[
            "effective_prompt"
        ]
        == TASK
    )

    rewritten = deepcopy(receipt)
    rewritten["prompt_binding"]["effective_prompt"] = "forged prompt"
    binding_payload = {
        name: value
        for name, value in rewritten["prompt_binding"].items()
        if name != "binding_sha256"
    }
    rewritten["prompt_binding"]["binding_sha256"] = _canonical_digest(binding_payload)
    derivation_payload = {
        name: value for name, value in rewritten.items() if name != "derivation_sha256"
    }
    rewritten["derivation_sha256"] = _canonical_digest(derivation_payload)
    with pytest.raises(ValueError, match="identity differs"):
        _dataset_prompt_binding(config, rewritten, admitted, config_path)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
