"""Qualify LoRAFleet's reconstructed OpenVLA-OFT LIBERO adapters.

This module is deliberately executable from the operator-built OpenVLA-OFT
image, not from the lightweight NPA control-plane environment.  It keeps model
weights in the Hugging Face cache, reconstructs only from the verified base
checkpoint, and writes small run-scoped evidence files for the BYOF worker to
publish.  It does not train, publish, or serve a shared model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


RELEASE_REPO = "LoRAFleet/openvla-oft-libero-reconstructed-r64"
RELEASE_REVISION = "0b75d5b19ba43ac3efa870bdf830192e98cb4d4e"
BASE_REPO = "openvla/openvla-7b"
BASE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
OFT_REPO = "https://github.com/moojink/openvla-oft"
OFT_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
DLIMP_REPO = "https://github.com/kvablack/dlimp"
DLIMP_REVISION = "92e3eca97af3b14d0b6aa15182c0dc240407698d"
DLIMP_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
DLIMP_ORIGINAL_DATASET_SHA256 = (
    "54f7cf110d4ca1300c3c31726ee89d1f02d4105beb0b3520ffa0c0abbda77c42"
)
DLIMP_DETERMINISTIC_DATASET_SHA256 = (
    "86bc13c005112961498170f5258639dd52cc46a7614995aff6b8ef6851803d48"
)
DLIMP_PROVENANCE_NAME = "npa_lorafleet_dlimp_provenance.json"
SCHEMA_VERSION = "npa.lorafleet-oft-adapters/v1"
SUITE_ORDER = ("spatial", "object", "goal", "10")
_OPENVLA_CLASSES_REGISTERED = False


@dataclass(frozen=True)
class SuiteSpec:
    """Pinned source and LIBERO execution details for one adapter suite."""

    name: str
    libero_suite: str
    source_repo: str
    source_revision: str
    head_step: int


SUITES = {
    "spatial": SuiteSpec(
        "spatial",
        "libero_spatial",
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "6d0231af0e48c5985f1ff86908f4674b84bc049b",
        150000,
    ),
    "object": SuiteSpec(
        "object",
        "libero_object",
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "4c89574e1c538b6c102f43f0526d60a9d3650148",
        150000,
    ),
    "goal": SuiteSpec(
        "goal",
        "libero_goal",
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "c2d0f9fbbd82674683b397ff923168a12f6a307b",
        50000,
    ),
    "10": SuiteSpec(
        "10",
        "libero_10",
        "moojink/openvla-7b-oft-finetuned-libero-10",
        "95220f9a3421a7ff12d4218e73d09ade830fa9a3",
        150000,
    ),
}


def _json_load(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object files."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, self-contained JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    """Return the complete-byte digest of ``path`` without retaining it in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_dlimp_runtime(
    *,
    source_root: Path = Path("/opt/dlimp"),
    provenance_path: Path = Path(f"/opt/byof/{DLIMP_PROVENANCE_NAME}"),
    module_path: Path | None = None,
) -> dict[str, Any]:
    """Reject a runtime that did not use the pinned licensed deterministic dlimp.

    ``moojink/dlimp_openvla`` is deliberately never fetched. The Apache-2.0
    upstream parent is byte-identical except for its LICENSE and the original
    ``options.deterministic`` default, so the image recipe makes that one
    explicit local modification and records both source file digests.
    """

    provenance = _json_load(provenance_path)
    expected = {
        "schema": "npa.lorafleet.dlimp-runtime/v1",
        "source": DLIMP_REPO,
        "revision": DLIMP_REVISION,
        "license": "Apache-2.0",
        "license_sha256": DLIMP_LICENSE_SHA256,
        "original_dataset_sha256": DLIMP_ORIGINAL_DATASET_SHA256,
        "modified_dataset_sha256": DLIMP_DETERMINISTIC_DATASET_SHA256,
        "modification": {
            "path": "dlimp/dataset.py",
            "from": "options.deterministic = False",
            "to": "options.deterministic = True",
        },
    }
    if provenance != expected:
        raise ValueError("dlimp provenance does not match the licensed pinned recipe")

    license_path = source_root / "LICENSE"
    dataset_path = source_root / "dlimp" / "dataset.py"
    if _sha256(license_path) != DLIMP_LICENSE_SHA256:
        raise ValueError("dlimp Apache-2.0 notice is missing or changed")
    if _sha256(dataset_path) != DLIMP_DETERMINISTIC_DATASET_SHA256:
        raise ValueError(
            "dlimp deterministic source modification is missing or changed"
        )
    source = dataset_path.read_text(encoding="utf-8")
    if source.count("options.deterministic = True") != 1 or (
        "options.deterministic = False" in source
    ):
        raise ValueError("dlimp deterministic override is not exact")

    if module_path is None:
        import dlimp  # type: ignore[import-not-found]

        module_path = Path(str(dlimp.__file__ or ""))
    try:
        module_path.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "imported dlimp is not the pinned licensed source tree"
        ) from exc
    return provenance


def _exercise_dlimp_determinism() -> dict[str, Any]:
    """Run the patched loader through parallel mapping and OFT's real RLDS reader.

    This deliberately creates a tiny local RLDS dataset instead of substituting
    fixture JSON. It is an image-runtime regression check, not a model-training
    claim, and avoids downloading the separately governed training data.
    """

    import dlimp  # type: ignore[import-not-found]
    from dlimp.dataset import _wrap  # type: ignore[import-not-found]
    import tensorflow as tf  # type: ignore[import-not-found]
    import tensorflow_datasets as tfds  # type: ignore[import-not-found]
    from prismatic.vla.datasets.rlds import dataset as oft_rlds  # type: ignore[import-not-found]

    def delayed(value: Any) -> Any:
        # A reverse delay makes the assertion meaningful when map ordering is
        # allowed to float; the dlimp option must keep source order instead.
        time.sleep((7 - int(value.numpy())) * 0.001)
        return value

    def mapped(value: Any) -> Any:
        return tf.py_function(delayed, [value], Tout=tf.int64)

    parallel = _wrap(tf.data.Dataset.from_tensor_slices, False)(
        tf.range(8, dtype=tf.int64)
    )._apply_options()
    parallel = parallel.map(mapped, num_parallel_calls=4)
    observed_order = [int(value.numpy()) for value in parallel]
    if parallel.options().deterministic is not True or observed_order != list(range(8)):
        raise RuntimeError("dlimp parallel map did not preserve deterministic ordering")

    class FixtureRlds(tfds.core.GeneratorBasedBuilder):
        """A two-step RLDS source consumed through the actual OFT reader."""

        VERSION = tfds.core.Version("1.0.0")

        def _info(self) -> Any:
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "state": tfds.features.Tensor(
                                            shape=(1,), dtype=tf.float32
                                        )
                                    }
                                ),
                                "action": tfds.features.Tensor(
                                    shape=(2,), dtype=tf.float32
                                ),
                            }
                        )
                    }
                ),
            )

        def _split_generators(self, _manager: Any) -> dict[str, Any]:
            return {"train": self._generate_examples()}

        def _generate_examples(self) -> Iterable[tuple[str, dict[str, Any]]]:
            yield (
                "fixture-episode",
                {
                    "steps": [
                        {"observation": {"state": [0.0]}, "action": [0.0, 1.0]},
                        {"observation": {"state": [1.0]}, "action": [1.0, 0.0]},
                    ]
                },
            )

    with tempfile.TemporaryDirectory(prefix="npa-lorafleet-rlds-") as directory:
        builder = FixtureRlds(data_dir=directory)
        builder.download_and_prepare()
        direct = dlimp.DLataset.from_rlds(
            builder, split="train", shuffle=False, num_parallel_reads=2
        )
        direct_item = next(iter(direct.take(1)))
        direct_frames = int(tf.shape(direct_item["action"])[0].numpy())

        original_builder = oft_rlds.tfds.builder
        oft_rlds.tfds.builder = lambda _name, data_dir: builder
        try:
            oft_dataset, _ = oft_rlds.make_dataset_from_rlds(
                "fixture_rlds",
                directory,
                train=True,
                shuffle=False,
                state_obs_keys=("state",),
                action_proprio_normalization_type="normal",
                dataset_statistics={
                    "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "proprio": {"mean": [0.0], "std": [1.0]},
                },
                num_parallel_reads=2,
                num_parallel_calls=2,
            )
            oft_item = next(iter(oft_dataset.take(1)))
        finally:
            oft_rlds.tfds.builder = original_builder
        oft_frames = int(tf.shape(oft_item["action"])[0].numpy())

    if direct_frames != 2 or oft_frames != 2:
        raise RuntimeError("dlimp/OFT RLDS regression fixture lost trajectory frames")
    return {
        "parallel_map_order": observed_order,
        "direct_rlds_trajectory_frames": direct_frames,
        "oft_rlds_trajectory_frames": oft_frames,
    }


def _cache_root() -> Path:
    """Use a run-writable cache while allowing the platform to supply one."""
    root = Path(os.environ.get("NPA_MODEL_CACHE_DIR", "/workspace/model-cache"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    return root


def _hub_modules() -> tuple[Any, Any, Any]:
    """Import runtime-only Hub and SafeTensors dependencies."""
    from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]
    from safetensors import safe_open  # type: ignore[import-not-found]

    return HfApi, snapshot_download, safe_open


def _snapshot(repo: str, revision: str, patterns: list[str] | None = None) -> Path:
    """Fetch a revision-pinned snapshot into the immutable Hub cache."""
    _, snapshot_download, _ = _hub_modules()
    location = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=patterns,
        cache_dir=str(_cache_root() / "huggingface"),
        token=os.environ.get("HF_TOKEN") or None,
    )
    return Path(location)


def _model_commit(repo: str, revision: str) -> str:
    """Resolve the exact Hub commit and reject a moving alias."""
    HfApi, _, _ = _hub_modules()
    sha = str(HfApi().model_info(repo_id=repo, revision=revision).sha)
    if sha != revision:
        raise ValueError(
            f"{repo} resolved {sha}, expected immutable revision {revision}"
        )
    return sha


def _release_patterns() -> list[str]:
    """Return every release artifact that the verifier must byte-check."""
    patterns = [
        "README.md",
        "manifest.json",
        "SHA256SUMS.json",
        "base_verification.json",
    ]
    for suite in SUITE_ORDER:
        patterns.extend(
            [
                f"{suite}/action_head--*_checkpoint.pt",
                f"{suite}/proprio_projector--*_checkpoint.pt",
                f"{suite}/dataset_statistics.json",
                f"{suite}/evaluation.json",
                f"{suite}/non_lora_audit.json",
                f"{suite}/reconstruction.json",
                f"{suite}/lora_adapter/adapter_config.json",
                f"{suite}/lora_adapter/adapter_model.safetensors",
            ]
        )
    return patterns


def _release_files(root: Path, suite: SuiteSpec) -> list[Path]:
    """Return the expected suite heads, statistics, and rank-64 factors."""
    directory = root / suite.name
    return [
        directory / f"action_head--{suite.head_step}_checkpoint.pt",
        directory / f"proprio_projector--{suite.head_step}_checkpoint.pt",
        directory / "dataset_statistics.json",
        directory / "evaluation.json",
        directory / "non_lora_audit.json",
        directory / "reconstruction.json",
        directory / "lora_adapter" / "adapter_config.json",
        directory / "lora_adapter" / "adapter_model.safetensors",
    ]


def _required_file_hashes(root: Path) -> dict[str, str]:
    """Digest every required artifact; a missing artifact is a hard failure."""
    files = [
        root / name
        for name in (
            "README.md",
            "manifest.json",
            "SHA256SUMS.json",
            "base_verification.json",
        )
    ]
    for suite in SUITES.values():
        files.extend(_release_files(root, suite))
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release is missing required artifacts: {missing}")
    return {str(path.relative_to(root)): _sha256(path) for path in files}


def _expected_release_hashes(root: Path) -> dict[str, str]:
    """Parse the publisher's path-to-SHA256 inventory."""
    payload = _json_load(root / "SHA256SUMS.json")
    hashes = payload.get("sha256") or payload.get("files") or payload
    if not isinstance(hashes, dict):
        raise ValueError("SHA256SUMS.json does not contain a path-to-hash mapping")
    return {str(key): str(value) for key, value in hashes.items()}


def _check_release_hashes(actual: dict[str, str], expected: dict[str, str]) -> None:
    """Require publisher hashes for every consumed payload and compare them."""
    # The checksum manifest authenticates the payload inventory but naturally
    # cannot contain a hash of itself. Its Hub revision is independently pinned.
    checked = {
        path: digest for path, digest in actual.items() if path != "SHA256SUMS.json"
    }
    absent = sorted(path for path in checked if path not in expected)
    mismatched = sorted(
        path for path, digest in checked.items() if expected.get(path) != digest
    )
    if absent or mismatched:
        raise ValueError(
            f"release checksum failure; absent={absent}, mismatched={mismatched}"
        )


def _validate_reconstruction_metadata(
    payload: dict[str, Any], suite: SuiteSpec
) -> None:
    """Reject factors whose arithmetic or upstream source is not the published one."""
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{suite.name}: missing source record")
    if (
        source.get("repo_id") != suite.source_repo
        or source.get("revision") != suite.source_revision
    ):
        raise ValueError(f"{suite.name}: source does not match the publisher manifest")
    if payload.get("rank") != 64 or payload.get("non_lora_exact") is not True:
        raise ValueError(f"{suite.name}: expected rank-64 and exact non-LoRA tensors")
    if payload.get("ab_parameters") != 221656576:
        raise ValueError(f"{suite.name}: unexpected published A/B parameter count")


def _factor_keys(adapter_path: Path) -> tuple[list[str], list[str]]:
    """Read factor names without loading full factor tensors into host memory."""
    _, _, safe_open = _hub_modules()
    with safe_open(str(adapter_path), framework="pt", device="cpu") as factors:
        keys = list(factors.keys())
    a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
    b_keys = sorted(key for key in keys if key.endswith(".lora_B.weight"))
    if len(a_keys) != 439 or len(b_keys) != 439:
        raise ValueError(
            f"expected 439 A and B tensors, got {len(a_keys)} and {len(b_keys)}"
        )
    return a_keys, b_keys


def _validate_adapter(root: Path, suite: SuiteSpec) -> dict[str, Any]:
    """Validate one factor payload before any weight is reconstructed."""
    metadata = _json_load(root / suite.name / "reconstruction.json")
    _validate_reconstruction_metadata(metadata, suite)
    config = _json_load(root / suite.name / "lora_adapter" / "adapter_config.json")
    if config.get("r") != 64 or config.get("lora_alpha") != 64:
        raise ValueError(f"{suite.name}: expected r=lora_alpha=64")
    a_keys, b_keys = _factor_keys(
        root / suite.name / "lora_adapter" / "adapter_model.safetensors"
    )
    expected_b = [key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys]
    if expected_b != b_keys:
        raise ValueError(f"{suite.name}: A/B factor keys do not pair exactly")
    audit = _json_load(root / suite.name / "non_lora_audit.json")
    return {"rank": 64, "target_modules": len(a_keys), "non_lora_audit": audit}


def _verify_stage(output: Path) -> None:
    """Fetch and verify all immutable public payloads needed by later stages."""
    dlimp = _verify_dlimp_runtime()
    dlimp_regression = _exercise_dlimp_determinism()
    release_root = _snapshot(RELEASE_REPO, RELEASE_REVISION, _release_patterns())
    release_hashes = _required_file_hashes(release_root)
    _check_release_hashes(release_hashes, _expected_release_hashes(release_root))
    manifest = _json_load(release_root / "manifest.json")
    suites = {
        name: _validate_adapter(release_root, spec) for name, spec in SUITES.items()
    }
    commits = {
        "reconstructed_release": _model_commit(RELEASE_REPO, RELEASE_REVISION),
        "base": _model_commit(BASE_REPO, BASE_REVISION),
    }
    commits.update(
        {
            f"source_{name}": _model_commit(spec.source_repo, spec.source_revision)
            for name, spec in SUITES.items()
        }
    )
    _json_dump(
        output,
        {
            "schema": SCHEMA_VERSION,
            "stage": "verify-inputs",
            "passed": True,
            "release": {
                "repo": RELEASE_REPO,
                "revision": RELEASE_REVISION,
                "file_sha256": release_hashes,
            },
            "base": {"repo": BASE_REPO, "revision": BASE_REVISION},
            "upstream": {"repo": OFT_REPO, "revision": OFT_REVISION},
            "runtime_dependency": {"dlimp": dlimp, "regression": dlimp_regression},
            "hub_commits": commits,
            "publisher_manifest": manifest,
            "suites": suites,
            "limitations": [
                "reconstructed factors are approximate and are not original training adapters",
                "the publisher's eight-initial-state, one-task smoke is not equivalence evidence",
                "this workflow does not qualify a shared-base serving system",
            ],
        },
    )


def _s3_parts(uri: str) -> tuple[str, str]:
    """Split only a concrete S3 URI; do not silently accept another transport."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"expected concrete s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _read_s3_json(uri: str) -> dict[str, Any]:
    """Read an exact prior-stage JSON artifact through the worker's scoped S3 access."""
    import boto3  # type: ignore[import-not-found]

    bucket, key = _s3_parts(uri)
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    payload = json.loads(response["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{uri} is not a JSON object")
    return payload


def _require_prior_stage(payload: dict[str, Any], stage: str) -> None:
    """Check the schema and named successful input before consuming it."""
    if (
        payload.get("schema") != SCHEMA_VERSION
        or payload.get("stage") != stage
        or payload.get("passed") is not True
    ):
        raise ValueError(f"required successful {stage} artifact was not supplied")


def _base_expected_hashes(payload: Any) -> set[str]:
    """Collect SHA256 values recursively from the publisher's base verification file."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            found.update(_base_expected_hashes(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_base_expected_hashes(value))
    elif (
        isinstance(payload, str)
        and len(payload) == 64
        and all(char in "0123456789abcdef" for char in payload)
    ):
        found.add(payload)
    return found


def _verify_base_shards(base_root: Path, release_root: Path) -> dict[str, str]:
    """Byte-check every downloaded base shard before it can be a reconstruction operand."""
    expected = _base_expected_hashes(
        _json_load(release_root / "base_verification.json")
    )
    index = _json_load(base_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("base safetensors index has no weight map")
    shards = sorted({str(value) for value in weight_map.values()})
    hashes = {name: _sha256(base_root / name) for name in shards}
    if set(hashes.values()) != expected:
        raise ValueError(
            "base shard digests do not match the release's base verification record"
        )
    return hashes


def _base_tensor_lookup(base_root: Path) -> tuple[dict[str, str], Any]:
    """Return a tensor-to-shard map and a SafeTensors opener for factor application."""
    _, _, safe_open = _hub_modules()
    index = _json_load(base_root / "model.safetensors.index.json")
    weights = index.get("weight_map")
    if not isinstance(weights, dict):
        raise ValueError("base model has no safetensors weight map")
    return {str(key): str(value) for key, value in weights.items()}, safe_open


def _register_openvla_classes() -> tuple[Any, Any]:
    """Register upstream custom classes once for the process-wide HF auto maps."""
    global _OPENVLA_CLASSES_REGISTERED
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )  # type: ignore[import-not-found]
    from experiments.robot.openvla_utils import (
        OpenVLAConfig,
        OpenVLAForActionPrediction,
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    if not _OPENVLA_CLASSES_REGISTERED:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        _OPENVLA_CLASSES_REGISTERED = True
    return AutoModelForVision2Seq, OpenVLAConfig


def _load_base_vla(base_root: Path, *, revision: str) -> Any:
    """Load a staged immutable VLA snapshot without allowing Hub fallback."""
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA-OFT qualification requires an assigned CUDA GPU")
    if not base_root.is_dir():
        raise FileNotFoundError(
            "staged immutable model snapshot is absent; refusing network fallback"
        )
    AutoModelForVision2Seq, _ = _register_openvla_classes()
    model = AutoModelForVision2Seq.from_pretrained(
        str(base_root),
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.vision_backbone.set_num_images_in_input(2)
    return model.eval().to(torch.device("cuda:0"))


def _apply_suite_factors(
    model: Any, base_root: Path, adapter_path: Path
) -> dict[str, str]:
    """Set each target to base + FP32(B @ A), never add factors to a target weight."""
    import torch  # type: ignore[import-not-found]

    weights, safe_open = _base_tensor_lookup(base_root)
    samples: dict[str, str] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as factors:
        for index, a_key in enumerate(
            sorted(key for key in factors.keys() if key.endswith(".lora_A.weight"))
        ):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            module_name = a_key.removeprefix("base_model.model.").removesuffix(
                ".lora_A.weight"
            )
            base_key = f"{module_name}.weight"
            if base_key not in weights:
                raise KeyError(
                    f"base tensor absent for reconstructed target {base_key}"
                )
            target = model.get_submodule(module_name).weight
            with safe_open(
                str(base_root / weights[base_key]), framework="pt", device="cpu"
            ) as shard:
                base = shard.get_tensor(base_key)
            a, b = factors.get_tensor(a_key), factors.get_tensor(b_key)
            if a.shape[0] != 64 or b.shape[1] != 64:
                raise ValueError(f"{a_key}: factor rank is not 64")
            delta = b.float().reshape(b.shape[0], -1) @ a.float().reshape(
                a.shape[0], -1
            )
            if tuple(delta.shape) != tuple(base.shape) or tuple(base.shape) != tuple(
                target.shape
            ):
                raise ValueError(f"{a_key}: factor/base/target shapes are incompatible")
            with torch.no_grad():
                target.copy_(
                    (
                        base.to(target.device, dtype=torch.float32)
                        + delta.to(target.device)
                    ).to(target.dtype)
                )
            if index in {0, 219, 438}:
                samples[module_name] = hashlib.sha256(
                    target.detach().float().cpu().numpy().tobytes()
                ).hexdigest()
            del base, a, b, delta
    return samples


def _release_root() -> Path:
    """Return the release snapshot needed by reconstruction and evaluation stages."""
    return _snapshot(RELEASE_REPO, RELEASE_REVISION, _release_patterns())


def _reconstruct_stage(verification_uri: str, output: Path) -> None:
    """Apply all four verified rank-64 factors to a verified unmerged base one suite at a time."""
    verification = _read_s3_json(verification_uri)
    _require_prior_stage(verification, "verify-inputs")
    release_root = _release_root()
    base_root = _snapshot(BASE_REPO, BASE_REVISION)
    base_hashes = _verify_base_shards(base_root, release_root)
    reports: dict[str, Any] = {}
    for suite in SUITES.values():
        model = _load_base_vla(base_root, revision=BASE_REVISION)
        samples = _apply_suite_factors(
            model,
            base_root,
            release_root / suite.name / "lora_adapter" / "adapter_model.safetensors",
        )
        reports[suite.name] = {
            "target_modules": 439,
            "formula": "W_base + FP32(B @ A)",
            "sample_weight_sha256": samples,
        }
        del model
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()
    _json_dump(
        output,
        {
            "schema": SCHEMA_VERSION,
            "stage": "reconstruct-weights",
            "passed": True,
            "inputs": {"verification_uri": verification_uri},
            "base_shard_sha256": base_hashes,
            "suites": reports,
            "guardrail": "factors were assigned from verified base tensors; no merged target is an operand",
        },
    )


def _source_snapshot(suite: SuiteSpec) -> Path:
    """Download a revision-pinned original merged policy for the baseline only."""
    return _snapshot(suite.source_repo, suite.source_revision)


def _stage_cfg(suite: SuiteSpec, checkpoint: Path) -> Any:
    """Build the upstream's documented LIBERO evaluation configuration."""
    from experiments.robot.libero.run_libero_eval import GenerateConfig  # type: ignore[import-not-found]

    return GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=suite.libero_suite,
        num_trials_per_task=50,
        center_crop=True,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        num_open_loop_steps=8,
        lora_rank=64,
        use_wandb=False,
    )


def _load_local_model(
    cfg: Any, suite: SuiteSpec, checkpoint: Path
) -> tuple[Any, Any, Any, Any, Any]:
    """Load a local immutable checkpoint without upstream cache-mutating sync helpers."""
    from experiments.robot.libero.run_libero_eval import check_unnorm_key
    from experiments.robot.openvla_utils import (
        _load_dataset_stats,
        get_action_head,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_image_resize_size

    model = _load_base_vla(checkpoint, revision=suite.source_revision)
    _load_dataset_stats(model, str(checkpoint))
    check_unnorm_key(cfg, model)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    return model, action_head, proprio, get_processor(cfg), get_image_resize_size(cfg)


def _load_reconstructed_model(
    cfg: Any, suite: SuiteSpec, base_root: Path, release_root: Path
) -> tuple[Any, Any, Any, Any, Any]:
    """Build a policy from the base and factors, then load its suite-specific heads/statistics."""
    from experiments.robot.libero.run_libero_eval import check_unnorm_key
    from experiments.robot.openvla_utils import (
        _load_dataset_stats,
        get_action_head,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_image_resize_size

    model = _load_base_vla(base_root, revision=BASE_REVISION)
    _apply_suite_factors(
        model,
        base_root,
        release_root / suite.name / "lora_adapter" / "adapter_model.safetensors",
    )
    suite_dir = release_root / suite.name
    _load_dataset_stats(model, str(suite_dir))
    check_unnorm_key(cfg, model)
    component_cfg = _stage_cfg(suite, suite_dir)
    action_head = get_action_head(component_cfg, model.llm_dim)
    proprio = get_proprio_projector(component_cfg, model.llm_dim, proprio_dim=8)
    processor_cfg = _stage_cfg(suite, base_root)
    return (
        model,
        action_head,
        proprio,
        get_processor(processor_cfg),
        get_image_resize_size(cfg),
    )


def _protocol(suites: Iterable[SuiteSpec]) -> list[dict[str, Any]]:
    """Return the upstream full-suite/default-trial protocol with explicit paired initial-state indices."""
    from libero.libero import benchmark  # type: ignore[import-not-found]

    protocol: list[dict[str, Any]] = []
    benchmarks = benchmark.get_benchmark_dict()
    for suite in suites:
        task_suite = benchmarks[suite.libero_suite]()
        for task_id in range(task_suite.n_tasks):
            for initial_state_index in range(50):
                protocol.append(
                    {
                        "suite": suite.name,
                        "task_id": task_id,
                        "initial_state_index": initial_state_index,
                    }
                )
    return protocol


def _configure_libero_runtime() -> None:
    """Create the non-interactive LIBERO path record before importing its benchmark."""
    # The private base provides the Ubuntu FFmpeg executable. The imageio plugin
    # is retained without its wheel-bundled static binary, so make this source
    # explicit rather than silently downloading or using a vendored executable.
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/bin/ffmpeg")
    source_root = Path("/opt/libero/libero")
    package_root = source_root / "libero"
    if not (package_root / "__init__.py").is_file():
        raise FileNotFoundError(f"pinned LIBERO source is missing from {source_root}")
    config_root = _cache_root() / "libero-config"
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(_cache_root() / "libero-datasets"),
        "assets": str(package_root / "assets"),
    }
    (config_root / "config.yaml").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def _run_episode_set(
    policy_kind: str,
    suite: SuiteSpec,
    cfg: Any,
    model_parts: tuple[Any, Any, Any, Any, Any],
    episodes: list[dict[str, Any]],
    output: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Execute upstream LIBERO episodes and retain one genuine decoded MP4 per suite."""
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from experiments.robot.libero import run_libero_eval as runner  # type: ignore[import-not-found]
    from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore[import-not-found]

    model, action_head, proprio, processor, resize_size = model_parts
    task_suite = benchmark.get_benchmark_dict()[suite.libero_suite]()
    results: list[dict[str, Any]] = []
    video_path: Path | None = None
    for episode in episodes:
        task = task_suite.get_task(int(episode["task_id"]))
        initial_states, _ = runner.load_initial_states(
            cfg, task_suite, int(episode["task_id"])
        )
        env, description = runner.get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res
        )
        set_seed_everywhere(cfg.seed)
        success, frames = runner.run_episode(
            cfg,
            env,
            description,
            model,
            resize_size,
            processor,
            action_head,
            proprio,
            None,
            initial_states[int(episode["initial_state_index"])],
            None,
        )
        results.append(
            {**episode, "success": bool(success), "frame_count": len(frames)}
        )
        if video_path is None:
            raw_video = Path(
                runner.save_rollout_video(frames, 1, bool(success), description)
            )
            video_path = output.parent / f"{policy_kind}-{suite.name}-rollout.mp4"
            shutil.move(str(raw_video), video_path)
        env.close()
    if video_path is None:
        raise RuntimeError(f"{suite.name}: no rollouts were run")
    return results, video_path.name


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate honest empirical success summaries from actual episode records."""
    successes = sum(1 for item in results if item["success"])
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
    }


def _rollout_stage(
    kind: str,
    verification_uri: str,
    reconstruction_uri: str,
    baseline_uri: str | None,
    output: Path,
) -> None:
    """Run either original merged baselines or factor-reconstructed policies over the same full protocol."""
    verification = _read_s3_json(verification_uri)
    reconstruction = _read_s3_json(reconstruction_uri)
    _require_prior_stage(verification, "verify-inputs")
    _require_prior_stage(reconstruction, "reconstruct-weights")
    _configure_libero_runtime()
    if baseline_uri is not None:
        baseline = _read_s3_json(baseline_uri)
        _require_prior_stage(baseline, "published-baseline-rollouts")
        protocol = list(baseline["protocol"])
    else:
        protocol = _protocol(SUITES.values())
    release_root = _release_root()
    base_root = _snapshot(BASE_REPO, BASE_REVISION) if kind == "reconstructed" else None
    all_results: list[dict[str, Any]] = []
    videos: dict[str, str] = {}
    for suite in SUITES.values():
        suite_episodes = [item for item in protocol if item["suite"] == suite.name]
        checkpoint = _source_snapshot(suite) if kind == "baseline" else base_root
        cfg = _stage_cfg(suite, checkpoint)
        model_parts = (
            _load_local_model(cfg, suite, checkpoint)
            if kind == "baseline"
            else _load_reconstructed_model(cfg, suite, base_root, release_root)
        )
        results, video = _run_episode_set(
            kind, suite, cfg, model_parts, suite_episodes, output
        )
        all_results.extend(results)
        videos[suite.name] = video
        del model_parts
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()
    stage = (
        "published-baseline-rollouts"
        if kind == "baseline"
        else "reconstructed-adapter-rollouts"
    )
    _json_dump(
        output,
        {
            "schema": SCHEMA_VERSION,
            "stage": stage,
            "passed": True,
            "inputs": {
                "verification_uri": verification_uri,
                "reconstruction_uri": reconstruction_uri,
                "baseline_uri": baseline_uri,
            },
            "protocol": protocol,
            "episodes": all_results,
            "summary": _summarize(all_results),
            "videos": videos,
            "claims": "full upstream suite/default 50 trials per task; this is a finite simulator comparison, not a shared-serving or physical-robot claim",
        },
    )


def _rrd(output: Path, baseline: dict[str, Any], reconstructed: dict[str, Any]) -> str:
    """Write and validate a Rerun recording from observed paired episode outcomes."""
    import rerun as rr  # type: ignore[import-not-found]

    path = output.parent / "comparison.rrd"
    rr.init(
        "npa.lorafleet-oft-adapters",
        recording_id=os.environ.get("NPA_WORKFLOW_RUN_ID", "local"),
    )
    rr.save(str(path))
    for index, (left, right) in enumerate(
        zip(baseline["episodes"], reconstructed["episodes"], strict=True)
    ):
        _rr_set_time(rr, index)
        rr.log("metrics/baseline_success", _rr_scalar(rr, float(left["success"])))
        rr.log("metrics/reconstructed_success", _rr_scalar(rr, float(right["success"])))
    rr.log(
        "provenance/release",
        rr.TextDocument(
            json.dumps({"release": RELEASE_REPO, "revision": RELEASE_REVISION})
        ),
    )
    rr.disconnect()
    rerun_cli = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(rerun_cli), "rrd", "verify", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode:
        subprocess.run(
            [str(rerun_cli), "rrd", "print", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return path.name


def _rr_set_time(rerun: Any, index: int) -> None:
    """Support the pinned Rerun API while retaining newer local test support."""
    if hasattr(rerun, "set_time"):
        rerun.set_time("episode", sequence=index)
    else:
        rerun.set_time_sequence("episode", index)


def _rr_scalar(rerun: Any, value: float) -> Any:
    """Create one scalar component across supported Rerun SDK revisions."""
    component = getattr(rerun, "Scalars", None) or rerun.Scalar
    return component(value)


def _compare_stage(baseline_uri: str, reconstructed_uri: str, output: Path) -> None:
    """Pair actual matched-state rollouts, emit metrics, and explicitly bound claims."""
    baseline = _read_s3_json(baseline_uri)
    reconstructed = _read_s3_json(reconstructed_uri)
    _require_prior_stage(baseline, "published-baseline-rollouts")
    _require_prior_stage(reconstructed, "reconstructed-adapter-rollouts")
    if baseline.get("protocol") != reconstructed.get("protocol"):
        raise ValueError(
            "rollout comparison requires exactly the baseline protocol and initial states"
        )
    pairs = list(zip(baseline["episodes"], reconstructed["episodes"], strict=True))
    if any(
        (a["suite"], a["task_id"], a["initial_state_index"])
        != (b["suite"], b["task_id"], b["initial_state_index"])
        for a, b in pairs
    ):
        raise ValueError("rollout episode identities are not paired")
    both = sum(a["success"] and b["success"] for a, b in pairs)
    changed = sum(a["success"] != b["success"] for a, b in pairs)
    rrd_name = _rrd(output, baseline, reconstructed)
    _json_dump(
        output,
        {
            "schema": SCHEMA_VERSION,
            "stage": "compare-and-visualize",
            "passed": True,
            "inputs": {
                "baseline_uri": baseline_uri,
                "reconstructed_uri": reconstructed_uri,
            },
            "baseline": _summarize(baseline["episodes"]),
            "reconstructed": _summarize(reconstructed["episodes"]),
            "paired": {
                "episodes": len(pairs),
                "both_success": both,
                "outcome_changed": changed,
            },
            "rrd": rrd_name,
            "mp4": {
                "baseline": baseline["videos"],
                "reconstructed": reconstructed["videos"],
            },
            "conclusion": "Observed simulator behavior is reported only for this immutable evaluation protocol; it does not establish adapter equivalence, benchmark superiority, shared-base serving, convergence, or physical-robot success.",
        },
    )


def _parser() -> argparse.ArgumentParser:
    """Create the isolated runtime CLI used by each substantive workflow stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("verify", "reconstruct", "baseline", "reconstructed", "compare"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verification-uri")
    parser.add_argument("--reconstruction-uri")
    parser.add_argument("--baseline-uri")
    parser.add_argument("--reconstructed-uri")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch a single real stage and require every declared input URI."""
    args = _parser().parse_args(argv)
    if args.stage == "verify":
        _verify_stage(args.output)
    elif args.stage == "reconstruct":
        if not args.verification_uri:
            raise ValueError("reconstruct requires --verification-uri")
        _reconstruct_stage(args.verification_uri, args.output)
    elif args.stage == "baseline":
        if not args.verification_uri or not args.reconstruction_uri:
            raise ValueError(
                "baseline requires verification and reconstruction artifacts"
            )
        _rollout_stage(
            "baseline",
            args.verification_uri,
            args.reconstruction_uri,
            None,
            args.output,
        )
    elif args.stage == "reconstructed":
        if (
            not args.verification_uri
            or not args.reconstruction_uri
            or not args.baseline_uri
        ):
            raise ValueError(
                "reconstructed requires verification, reconstruction, and baseline artifacts"
            )
        _rollout_stage(
            "reconstructed",
            args.verification_uri,
            args.reconstruction_uri,
            args.baseline_uri,
            args.output,
        )
    else:
        if not args.baseline_uri or not args.reconstructed_uri:
            raise ValueError("compare requires baseline and reconstructed artifacts")
        _compare_stage(args.baseline_uri, args.reconstructed_uri, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
