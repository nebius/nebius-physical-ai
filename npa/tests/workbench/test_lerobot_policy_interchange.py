"""Contract tests for real LeRobot policy-container interchange wiring."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import npa.workbench.lerobot.policy_container as policy_container
from npa.workbench.lerobot.policy_container import (
    INTERCHANGE_MANIFEST_FILENAME,
    PolicyContainerError,
    _load_interchange_manifest,
    _materialize_policy_source,
    _processor_identity,
    _smolvla_batch,
    _validate_native_policy_features,
    build_parser,
    main,
    prepare_smolvla_base_checkpoint,
    write_smolvla_interchange_manifest,
)


def _checkpoint(tmp_path: Path) -> tuple[Path, str]:
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    weights = checkpoint / "pytorch_model.bin"
    weights.write_bytes(b"real-checkpoint-binding-fixture")
    (checkpoint / "policy_preprocessor.json").write_text(
        '{"normalize":{"mean":[0.1],"std":[0.2]}}\n', encoding="utf-8"
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        '{"unnormalize":{"mean":[0.1],"std":[0.2]}}\n', encoding="utf-8"
    )
    (
        checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    ).write_bytes(b"native-normalizer-state")
    (
        checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    ).write_bytes(b"native-unnormalizer-state")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    return checkpoint, digest


def test_exact_revision_policy_source_is_forwarded_to_huggingface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def model_info(self, *, repo_id: str, revision: str) -> SimpleNamespace:
            assert repo_id == "lerobot/smolvla_base"
            return SimpleNamespace(sha=revision)

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        return str(target)

    module.snapshot_download = snapshot_download
    module.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    revision = "a" * 40
    result = _materialize_policy_source(
        f"hf://lerobot/smolvla_base@{revision}", tmp_path / "cache"
    )
    assert result.is_dir()
    assert calls == [
        {
            "repo_id": "lerobot/smolvla_base",
            "revision": revision,
            "local_dir": tmp_path / "cache" / "lerobot__smolvla_base" / revision,
        }
    ]
    assert json.loads((result / ".npa-hf-source.json").read_text()) == {
        "repo_id": "lerobot/smolvla_base",
        "repo_type": "model",
        "revision": revision,
    }


def test_malformed_huggingface_revision_is_rejected_before_download(
    tmp_path: Path,
) -> None:
    with pytest.raises(PolicyContainerError, match="40 lowercase hex"):
        _materialize_policy_source("hf://lerobot/smolvla_base@main", tmp_path / "cache")


def _manifest(digest: str, normalization_identity: str = "c" * 64) -> dict:
    return {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "policy_identity": "smolvla-six-axis",
        "robot_identity": "six-axis-arm",
        "task_identity": "released-surface-v1",
        "normalization_identity": normalization_identity,
        "checkpoint_sha256": digest,
        "measured_state_names": [
            "j1",
            "j2",
            "j3",
            "j4",
            "j5",
            "j6",
            "gripper",
        ],
        "measured_state_units": ["rad"] * 6 + ["normalized_closed"],
        "state_feature_key": "observation.state",
        "action_feature_key": "action",
        "cameras": [
            {
                "key": "scene",
                "feature_key": "observation.images.scene",
                "layout": "HWC",
                "image_size_wh": [224, 224],
                "maximum_state_skew_s": 1.0 / 60.0,
            },
            {
                "key": "tool",
                "feature_key": "observation.images.tool",
                "layout": "HWC",
                "image_size_wh": [224, 224],
                "maximum_state_skew_s": 1.0 / 60.0,
            },
        ],
        "action": {
            "representation": "absolute_joint_position",
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
            "units": ["rad", "rad", "rad", "rad", "rad", "rad", "normalized_closed"],
            "gripper_semantics": "normalized_closed",
            "horizon": 50,
            "control_hz": 15.0,
            "execute_prefix": 1,
        },
    }


def _write_manifest(checkpoint: Path, payload: dict) -> None:
    (checkpoint / INTERCHANGE_MANIFEST_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _png() -> str:
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.new("RGB", (224, 224), (12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _observation(manifest: dict) -> dict:
    return {
        "schema": "npa.sim2real.policy_observation.v1",
        "policy_identity": manifest["policy_identity"],
        "robot_identity": manifest["robot_identity"],
        "task_identity": manifest["task_identity"],
        "prompt": "place the object",
        "privileged_object_state": False,
        "measured_state_names": manifest["measured_state_names"],
        "measured_state_units": manifest["measured_state_units"],
        "measured_state": [0.0] * len(manifest["measured_state_names"]),
        "views": [
            {
                "key": camera["key"],
                "layout": camera["layout"],
                "mask": True,
                "capture_timestamp_s": 2.0,
                "state_timestamp_s": 2.0,
                "image_base64": _png(),
            }
            for camera in manifest["cameras"]
        ],
    }


def test_manifest_is_cryptographically_bound_to_real_weight_file(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    _write_manifest(checkpoint, payload)
    assert _load_interchange_manifest(checkpoint)["checkpoint_sha256"] == digest
    payload["checkpoint_sha256"] = "0" * 64
    _write_manifest(checkpoint, payload)
    with pytest.raises(PolicyContainerError, match="not bound"):
        _load_interchange_manifest(checkpoint)


def test_writer_binds_complete_adapter_to_weight_bytes(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, _ = _processor_identity(checkpoint)
    manifest = _manifest(digest, normalization)
    path = write_smolvla_interchange_manifest(
        checkpoint,
        policy_identity=manifest["policy_identity"],
        robot_identity=manifest["robot_identity"],
        task_identity=manifest["task_identity"],
        normalization_identity=manifest["normalization_identity"],
        measured_state_names=manifest["measured_state_names"],
        measured_state_units=manifest["measured_state_units"],
        cameras=manifest["cameras"],
        action=manifest["action"],
    )
    payload = json.loads(path.read_text())
    assert payload["checkpoint_sha256"] == digest
    assert payload["normalization_identity"] == normalization
    assert payload["action"]["names"][-1] == "gripper"


def test_writer_derives_normalization_identity_from_native_processors(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    manifest = _manifest(digest)
    path = write_smolvla_interchange_manifest(
        checkpoint,
        policy_identity=manifest["policy_identity"],
        robot_identity=manifest["robot_identity"],
        task_identity=manifest["task_identity"],
        measured_state_names=manifest["measured_state_names"],
        measured_state_units=manifest["measured_state_units"],
        cameras=manifest["cameras"],
        action=manifest["action"],
    )
    payload = json.loads(path.read_text())
    assert payload["normalization_identity"] == _processor_identity(checkpoint)[0]


def test_prepared_base_keeps_weights_and_binds_dataset_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, digest = _checkpoint(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "prepared"
    manifest = _manifest(digest)

    class FeatureType(Enum):
        STATE = "STATE"
        VISUAL = "VISUAL"
        ACTION = "ACTION"

    features = {
        "observation.state": SimpleNamespace(type=FeatureType.STATE, shape=(7,)),
        "observation.images.scene": SimpleNamespace(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        "observation.images.tool": SimpleNamespace(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        "action": SimpleNamespace(type=FeatureType.ACTION, shape=(7,)),
    }

    class Metadata:
        def __init__(self, *, repo_id: str, root: Path) -> None:
            assert repo_id == "npa/six-axis-placement"
            assert root == dataset
            self.features = features
            self.stats = {"training_split_only": True}

    class Config:
        input_features = {}
        output_features = {}
        pretrained_path = "public-base"
        device = "cuda"
        saved_pretrained_path = "unset"
        saved_device = "unset"

        @classmethod
        def from_pretrained(cls, path: str) -> Config:
            assert Path(path) == base
            return cls()

        def save_pretrained(self, path: Path) -> None:
            (path / "config.json").write_text("{}\n", encoding="utf-8")
            Config.input_features = self.input_features
            Config.output_features = self.output_features
            Config.saved_pretrained_path = self.pretrained_path
            Config.saved_device = self.device

    class Processor:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def save_pretrained(self, path: Path, *, config_filename: str) -> None:
            (path / config_filename).write_text("{}\n", encoding="utf-8")
            state = (
                "policy_preprocessor_step_5_normalizer_processor.safetensors"
                if self.prefix == "pre"
                else "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            )
            (path / state).write_bytes(f"{self.prefix}-training-stats".encode())

    class Policy:
        config = Config

        @classmethod
        def from_pretrained(cls, path: str) -> Policy:
            assert Path(path) == output
            return cls()

        def to(self, device: str) -> Policy:
            assert device == "cpu"
            return self

        def eval(self) -> Policy:
            return self

    def api() -> tuple:
        return (
            FeatureType,
            Metadata,
            lambda **kwargs: (Processor("pre"), Processor("post")),
            Config,
            Policy,
            "policy_postprocessor",
            "policy_preprocessor",
            lambda value: value,
        )

    monkeypatch.setattr(policy_container, "_native_smolvla_preparation_api", api)
    receipt = prepare_smolvla_base_checkpoint(
        base,
        dataset_root=dataset,
        dataset_repo_id="npa/six-axis-placement",
        adapter=manifest,
        output_dir=output,
    )
    assert receipt["native_reload_passed"] is True
    assert receipt["weight_update_l2"] == 0.0
    assert receipt["source_weight_sha256"] == digest
    assert (output / "pytorch_model.bin").read_bytes() == (
        base / "pytorch_model.bin"
    ).read_bytes()
    assert Config.saved_pretrained_path is None
    assert Config.saved_device == "cpu"


def test_manifest_rejects_changed_normalization_processor(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    _write_manifest(checkpoint, payload)
    (checkpoint / "policy_preprocessor.json").write_text(
        '{"normalize":{"mean":[9.9],"std":[0.2]}}\n', encoding="utf-8"
    )
    with pytest.raises(PolicyContainerError, match="normalization"):
        _load_interchange_manifest(checkpoint)


def test_checkpoint_features_must_match_camera_state_and_action_dimensions() -> None:
    manifest = _manifest("a" * 64)
    feature = lambda shape: SimpleNamespace(shape=shape)  # noqa: E731
    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.state": feature((7,)),
                "observation.images.scene": feature((3, 224, 224)),
                "observation.images.tool": feature((3, 224, 224)),
            },
            output_features={"action": feature((7,))},
        )
    )
    _validate_native_policy_features(policy, manifest)
    policy.config.output_features["action"] = feature((8,))
    with pytest.raises(PolicyContainerError, match="action feature"):
        _validate_native_policy_features(policy, manifest)


def test_native_runtime_reset_clears_policy_and_processor_queues() -> None:
    from npa.workbench.lerobot.policy_container import LeRobotPolicyRuntime

    calls: list[str] = []

    class Component:
        def __init__(self, name: str) -> None:
            self.name = name

        def reset(self) -> None:
            calls.append(self.name)

    runtime = object.__new__(LeRobotPolicyRuntime)
    runtime.policy = Component("policy")
    runtime.preprocessor = Component("preprocessor")
    runtime.postprocessor = Component("postprocessor")
    runtime.reset()
    assert calls == ["policy", "preprocessor", "postprocessor"]


def test_native_runtime_consumes_every_fresh_observation_with_paired_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import npa.workbench.lerobot.policy_container as policy_container
    from npa.workbench.lerobot.policy_container import LeRobotPolicyRuntime

    calls: list[tuple[object, object]] = []

    class Policy:
        config = SimpleNamespace(chunk_size=50, max_action_dim=32)

        def predict_action_chunk(self, batch, *, noise):
            calls.append((batch, noise.clone()))
            result = torch.zeros((1, 50, 7), dtype=torch.float32)
            result[:, 0, :] = float(len(calls))
            return result

        def select_action(self, *_args, **_kwargs):
            raise AssertionError(
                "queued select_action must not serve fresh observations"
            )

    runtime = object.__new__(LeRobotPolicyRuntime)
    runtime.device = "cpu"
    runtime.manifest = _manifest("a" * 64)
    runtime.policy = Policy()
    runtime.preprocessor = lambda batch: batch
    runtime.postprocessor = lambda action: action
    monkeypatch.setattr(policy_container, "_smolvla_batch", lambda *_args: {"x": 1})

    first, first_sampling = runtime.infer_with_sampling({}, sampling_seed=19)
    second, second_sampling = runtime.infer_with_sampling({}, sampling_seed=19)
    chunk, chunk_sampling = runtime.infer_action_chunk_with_sampling(
        {}, sampling_seed=19
    )

    assert first == [1.0] * 7
    assert second == [2.0] * 7
    assert chunk[0] == [3.0] * 7
    assert chunk[1:] == [[0.0] * 7 for _ in range(49)]
    assert len(calls) == 3
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[1][1], calls[2][1])
    assert (
        first_sampling["sampler_noise_sha256"]
        == second_sampling["sampler_noise_sha256"]
        == chunk_sampling["sampler_noise_sha256"]
    )

    from npa.workflows.sim2real.interchange_contract import PolicyAction

    profile = SimpleNamespace(
        backend="lerobot_smolvla",
        identity=runtime.manifest["policy_identity"],
        checkpoint_sha256=runtime.manifest["checkpoint_sha256"],
        action=PolicyAction(
            representation="absolute_joint_position",
            names=tuple(runtime.manifest["action"]["names"]),
            units=tuple(runtime.manifest["action"]["units"]),
            frame="joint",
            gripper_semantics="normalized_closed",
            horizon=50,
            control_hz=15.0,
            execute_prefix=1,
        ),
    )
    observation = SimpleNamespace(
        input_identity="fresh-input",
        to_wire_dict=lambda: {},
    )
    output, typed_sampling = runtime.infer_typed_with_sampling(
        observation, profile, sampling_seed=19
    )
    assert len(output.action_chunk) == 50
    assert output.action_chunk[0] == (4.0,) * 7
    assert output.action_chunk[1:] == ((0.0,) * 7,) * 49
    assert (
        typed_sampling["sampler_noise_sha256"] == first_sampling["sampler_noise_sha256"]
    )


def test_real_image_decode_and_stale_camera_rejection() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    batch = _smolvla_batch(observation, manifest, "cpu")
    assert tuple(batch["observation.images.scene"].shape) == (1, 3, 224, 224)
    assert tuple(batch["observation.state"].shape) == (1, 7)
    assert batch["task"] == ["place the object"]
    observation["views"][1]["capture_timestamp_s"] = 1.0
    with pytest.raises(PolicyContainerError, match="stale"):
        _smolvla_batch(observation, manifest, "cpu")


def test_typed_raw_rgb_wire_is_consumed_and_hash_bound() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    rgb8 = bytes([23]) * (224 * 224 * 3)
    for view in observation["views"]:
        view.pop("image_base64")
        view["image_rgb8_base64"] = base64.b64encode(rgb8).decode("ascii")
        view["image_sha256"] = hashlib.sha256(rgb8).hexdigest()
    batch = _smolvla_batch(observation, manifest, "cpu")
    assert tuple(batch["observation.images.tool"].shape) == (1, 3, 224, 224)
    observation["views"][1]["image_sha256"] = "0" * 64
    with pytest.raises(PolicyContainerError, match="hash"):
        _smolvla_batch(observation, manifest, "cpu")


def test_six_axis_manifest_cannot_accept_reordered_or_padded_state() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    observation["measured_state_names"] = [
        observation["measured_state_names"][1],
        observation["measured_state_names"][0],
        *observation["measured_state_names"][2:],
    ]
    with pytest.raises(PolicyContainerError, match="names/order"):
        _smolvla_batch(observation, manifest, "cpu")
    observation = _observation(manifest)
    observation["measured_state"].insert(6, 0.0)
    with pytest.raises(PolicyContainerError, match="complete and finite"):
        _smolvla_batch(observation, manifest, "cpu")


def test_smolvla_rejects_structured_privileged_object_state() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    observation["privileged_object_state"] = {"object_xyz": [0.0, 0.0, 0.0]}
    with pytest.raises(PolicyContainerError, match="privileged"):
        _smolvla_batch(observation, manifest, "cpu")


def test_cartesian_manifest_requires_qualified_controller(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    payload["action"]["representation"] = "ee_pose"
    _write_manifest(checkpoint, payload)
    with pytest.raises(PolicyContainerError, match="IK/controller"):
        _load_interchange_manifest(checkpoint)


def test_interchange_cli_requires_explicit_checkpoint_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    checkpoint = tmp_path / "checkpoint"
    adapter = tmp_path / "adapter.json"
    bound_output = tmp_path / "bound.json"
    bound = parser.parse_args(
        [
            "bind-interchange",
            "--checkpoint-path",
            str(checkpoint),
            "--adapter-spec",
            str(adapter),
            "--output-json",
            str(bound_output),
        ]
    )
    assert bound.command == "bind-interchange"
    observation = tmp_path / "observation.json"
    inference_output = tmp_path / "output.json"
    inferred = parser.parse_args(
        [
            "infer-interchange",
            "--checkpoint-path",
            str(checkpoint),
            "--observation-json",
            str(observation),
            "--output-json",
            str(inference_output),
        ]
    )
    assert inferred.command == "infer-interchange"
    trained = parser.parse_args(
        [
            "train",
            "--dataset-repo-id",
            "operator/surface-placement",
            "--output-dir",
            str(tmp_path / "training"),
            "--steps",
            "100",
            "--interchange-adapter-relative-path",
            "meta/npa_policy_adapter.json",
        ]
    )
    assert trained.interchange_adapter_relative_path == "meta/npa_policy_adapter.json"


def test_health_exposes_loaded_interchange_identities_and_byte_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest_path = checkpoint / INTERCHANGE_MANIFEST_FILENAME
    manifest_path.write_text('{"sealed":true}\n', encoding="utf-8")
    manifest = _manifest("a" * 64, "b" * 64)
    runtime = SimpleNamespace(
        checkpoint_path=checkpoint,
        manifest=manifest,
        validation=SimpleNamespace(to_dict=lambda: {"status": "loadable"}),
    )
    monkeypatch.setenv("NPA_POLICY_CHECKPOINT", str(checkpoint))
    import npa.workbench.lerobot.policy_container as policy_container

    monkeypatch.setattr(policy_container, "LeRobotPolicyRuntime", lambda _: runtime)
    monkeypatch.setattr(
        policy_container,
        "assert_lerobot_importable",
        lambda: SimpleNamespace(to_dict=lambda: {"status": "ok"}),
    )
    response = fastapi_testclient.TestClient(policy_container.create_app()).get(
        "/health"
    )
    assert response.status_code == 200, response.text
    interchange = response.json()["interchange"]
    assert interchange["robot_identity"] == "six-axis-arm"
    assert interchange["checkpoint_sha256"] == "a" * 64
    assert interchange["normalization_identity"] == "b" * 64
    assert (
        interchange["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )


def test_typed_endpoint_binds_fresh_observation_and_exact_sampler_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    calls: list[int | None] = []
    manifest = _manifest("a" * 64, "b" * 64)
    response_payload = {
        "schema": "npa.sim2real.policy_output.v1",
        "observation_input_identity": "sealed-input",
        "policy_identity": manifest["policy_identity"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "action": manifest["action"],
        "action_chunk": [[0.0] * 7],
        "generated_timestamp_s": 2.0,
    }

    class Runtime:
        def __init__(self) -> None:
            self.manifest = manifest

        def infer_typed_with_sampling(
            self, _observation, _profile, *, sampling_seed=None
        ):
            calls.append(sampling_seed)
            return SimpleNamespace(to_dict=lambda: response_payload), {
                "sampler_noise_sha256": "c" * 64,
                "sampler_noise_shape": [1, 50, 32],
            }

    monkeypatch.setenv("NPA_POLICY_CHECKPOINT", "/sealed/checkpoint")
    monkeypatch.setenv("NPA_POLICY_AUTH_TOKEN", "private-token")
    monkeypatch.setattr(policy_container, "LeRobotPolicyRuntime", lambda _: Runtime())
    client = fastapi_testclient.TestClient(policy_container.create_app())
    payload = _observation(manifest)
    payload["input_identity"] = "sealed-input"
    rgb8 = bytes([31]) * (224 * 224 * 3)
    for index, view in enumerate(payload["views"]):
        view.update(
            {
                "role": "exterior" if index == 0 else "wrist",
                "calibration_sha256": str(index + 1) * 64,
                "image_size_wh": [224, 224],
                "image_rgb8_base64": base64.b64encode(rgb8).decode("ascii"),
                "image_sha256": hashlib.sha256(rgb8).hexdigest(),
            }
        )
        view.pop("image_base64")
    response = client.post(
        "/infer-typed",
        json=payload,
        headers={
            "Authorization": "Bearer private-token",
            "X-NPA-Sampling-Seed": "29",
            "X-NPA-Sampling-Sequence": "scene-3-decision-7",
        },
    )

    assert response.status_code == 200, response.text
    assert calls == [29]
    assert response.headers["x-npa-fresh-observation-consumed"] == "true"
    assert response.headers["x-npa-sampler-noise-sha256"] == "c" * 64
    assert response.headers["x-npa-sampler-noise-shape"] == "1,50,32"
    assert response.headers["x-npa-sampling-sequence"] == "scene-3-decision-7"


def test_interchange_training_forbids_early_checkpoint_only_upload(
    tmp_path: Path,
) -> None:
    with pytest.raises(PolicyContainerError, match="checkpoint-only upload"):
        main(
            [
                "train",
                "--dataset-repo-id",
                "operator/surface-placement",
                "--output-dir",
                str(tmp_path / "output"),
                "--steps",
                "100",
                "--interchange-adapter-relative-path",
                "meta/npa_policy_adapter.json",
                "--checkpoint-s3-uri",
                "s3://example/checkpoint/",
            ]
        )
