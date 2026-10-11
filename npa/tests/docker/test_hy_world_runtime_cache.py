"""Regression tests for the HY-World runtime-fetch cache layout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "docker" / "workbench" / "hy-world" / "hy_world_runtime.sh"


def _bash(
    script: str, *, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )


def _fake_diffusers(root: Path) -> Path:
    """Provide the sharded-resolver boundary without downloading a model."""

    module = root / "diffusers" / "utils"
    module.mkdir(parents=True)
    (root / "diffusers" / "__init__.py").write_text("", encoding="utf-8")
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "hub_utils.py").write_text(
        """
import json
import os
from pathlib import Path


def _get_checkpoint_shard_files(snapshot, index, *, local_files_only, subfolder):
    assert local_files_only is True
    assert subfolder == \"transformer\"
    index_path = Path(index)
    payload = json.loads(index_path.read_text(encoding=\"utf-8\"))
    paths = [index_path.parent / name for name in payload[\"weight_map\"].values()]
    if os.environ.get(\"NPA_TEST_RETURN_BLOB_PATHS\") == \"1\":
        paths = [path.resolve() for path in paths]
    return (
        [str(path) for path in paths],
        {\"weight_map\": payload[\"weight_map\"]},
    )
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_default_runtime_root_is_frozen_before_hf_cache_isolated(
    tmp_path: Path,
) -> None:
    """Normal image defaults cannot nest source/venv after HF_HOME is repointed."""

    shared_hf = tmp_path / "shared-hf"
    shared_ref = shared_hf / "hub" / "models--unrelated--model" / "refs"
    shared_ref.mkdir(parents=True)
    (shared_ref / "main").write_text("unrelated-main\n", encoding="utf-8")
    result_path = tmp_path / "result.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "configure_runtime_hf_cache",
                'model_cache="$(hub_cache_directory "$HY_WORLD_MODEL")"',
                'mkdir -p "$model_cache/snapshots/$HY_WORLD_MODEL_REF"',
                'register_hub_main_ref "$HY_WORLD_MODEL" "$HY_WORLD_MODEL_REF"',
                f'printf "%s\\n%s\\n%s\\n" "$(cache_root)" "$HF_HOME" "$(cat "$model_cache/refs/main")" > {result_path}',
                'test "$(source_tree)" = "$(cache_root)/source-$SOURCE_REF"',
            )
        ),
        environment={
            "HF_HOME": str(shared_hf),
            "NPA_HY_WORLD_RUNTIME_CACHE": "",
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    root, isolated_hf, main_ref = result_path.read_text(encoding="utf-8").splitlines()
    assert root == f"{shared_hf}/hy-world/runtime"
    assert isolated_hf.startswith(f"{root}/hf-")
    assert "/hf-" not in root
    assert main_ref == "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"
    assert (
        Path(isolated_hf) / "hub/models--tencent--HY-World-2.0/refs/main"
    ).read_bytes() == (b"d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0")
    assert (shared_ref / "main").read_text(encoding="utf-8") == "unrelated-main\n"


def test_legacy_matching_hub_ref_is_normalized_to_exact_snapshot_bytes(
    tmp_path: Path,
) -> None:
    """A prior newline-terminated ref is repaired without accepting a mismatch."""

    result_path = tmp_path / "ref.bin"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$HY_WORLD_MODEL")"',
                'mkdir -p "$cache/snapshots/$HY_WORLD_MODEL_REF" "$cache/refs"',
                'printf "%s\\n" "$HY_WORLD_MODEL_REF" > "$cache/refs/main"',
                'register_hub_main_ref "$HY_WORLD_MODEL" "$HY_WORLD_MODEL_REF"',
                f'cp "$cache/refs/main" {result_path}',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result_path.read_bytes() == b"d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"


def test_disagreeing_hub_ref_refuses_without_rewriting_the_cache(
    tmp_path: Path,
) -> None:
    """Only the exact legacy newline form is repairable in the isolated cache."""

    main_ref = tmp_path / "main"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$HY_WORLD_MODEL")"',
                'mkdir -p "$cache/snapshots/$HY_WORLD_MODEL_REF" "$cache/refs"',
                'printf "%s" "${HY_WORLD_MODEL_REF%?}1" > "$cache/refs/main"',
                'trap \'cp "$cache/refs/main" "$NPA_TEST_REF_AFTER"\' EXIT',
                'register_hub_main_ref "$HY_WORLD_MODEL" "$HY_WORLD_MODEL_REF"',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_REF_AFTER": str(main_ref),
        },
    )

    assert result.returncode != 0
    assert "isolated cache main ref disagrees" in result.stderr
    assert main_ref.read_bytes() != b"d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"


def test_grounding_dino_fetches_a_full_pinned_snapshot_for_its_consumer(
    tmp_path: Path,
) -> None:
    """The trajectory loader needs processor metadata as well as its weights."""

    arguments = tmp_path / "hf-arguments.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "hf() {",
                '  printf "%s\\t" "$@" >> "$NPA_TEST_HF_ARGUMENTS"',
                '  printf "\\n" >> "$NPA_TEST_HF_ARGUMENTS"',
                '  cache="$(hub_cache_directory "$2")"',
                '  mkdir -p "$cache/snapshots/$4"',
                '  if [[ "$2" == "$GROUNDING_DINO_MODEL" ]]; then',
                '    : > "$cache/snapshots/$4/config.json"',
                '    : > "$cache/snapshots/$4/preprocessor_config.json"',
                '    : > "$cache/snapshots/$4/model.safetensors"',
                "  fi",
                "}",
                "fetch_models",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_HF_ARGUMENTS": str(arguments),
            "NPA_TEST_PYTHON": sys.executable,
        },
    )

    assert result.returncode == 0, result.stderr
    downloads = [
        line.rstrip("\t").split("\t")
        for line in arguments.read_text(encoding="utf-8").splitlines()
    ]
    grounding_dino = next(
        download
        for download in downloads
        if download[1] == "IDEA-Research/grounding-dino-tiny"
    )
    assert grounding_dino == [
        "download",
        "IDEA-Research/grounding-dino-tiny",
        "--revision",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    ]


def test_grounding_dino_snapshot_refuses_missing_processor_metadata(
    tmp_path: Path,
) -> None:
    """A partial cache fails before the upstream trajectory loader executes."""

    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$GROUNDING_DINO_MODEL")"',
                'mkdir -p "$cache/snapshots/$GROUNDING_DINO_REF"',
                'touch "$cache/snapshots/$GROUNDING_DINO_REF/model.safetensors"',
                'verify_hub_snapshot_files "$GROUNDING_DINO_MODEL" "$GROUNDING_DINO_REF" config.json preprocessor_config.json model.safetensors',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode != 0
    assert "pinned Hugging Face snapshot is missing config.json" in result.stderr


def test_worldstereo_aliases_use_pinned_snapshots_and_reject_source_symlinks(
    tmp_path: Path,
) -> None:
    """Model-ID aliases cannot redirect the offline loader outside WorldGen."""

    worldgen = tmp_path / "worldgen"
    receipt = tmp_path / "result.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                'worldstereo="$(hub_cache_directory "$WORLD_STEREO_MODEL")/snapshots/$WORLD_STEREO_REF"',
                'wan="$(hub_cache_directory "$WAN_I2V_MODEL")/snapshots/$WAN_I2V_REF"',
                'mkdir -p "$worldstereo/worldstereo-memory-dmd" "$wan"',
                'printf \'%s\\n\' \'{"base_model": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"}\' > "$worldstereo/worldstereo-memory-dmd/config.json"',
                f"mkdir -p {worldgen}",
                f"prepare_offline_worldstereo_aliases {worldgen}",
                f'printf "%s\\n%s\\n" "$(readlink -f {worldgen}/hanshanxue/WorldStereo)" "$(readlink -f {worldgen}/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)" > {receipt}',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
        },
    )

    assert result.returncode == 0, result.stderr
    aliases = receipt.read_text(encoding="utf-8").splitlines()
    assert aliases[0].endswith(
        "models--hanshanxue--WorldStereo/snapshots/ac2ad97ecb043fe80c2f19cd1898006becb9d66e"
    )
    assert aliases[1].endswith(
        "models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/b184e23a8a16b20f108f727c902e769e873ffc73"
    )

    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (worldgen / "Wan-AI" / "Wan2.1-I2V-14B-480P-Diffusers").unlink()
    (worldgen / "Wan-AI").rmdir()
    (worldgen / "Wan-AI").symlink_to(escaped, target_is_directory=True)
    collision = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                f"prepare_offline_worldstereo_aliases {worldgen}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
        },
    )

    assert collision.returncode != 0
    assert "upstream model namespace is a symlink" in collision.stderr
    assert not (escaped / "Wan2.1-I2V-14B-480P-Diffusers").exists()


def test_wan_sharded_resolver_accepts_hub_blob_symlinks_but_not_escaped_paths(
    tmp_path: Path,
) -> None:
    """The actual verifier accepts normal Hub blobs and refuses traversal."""

    fake_modules = _fake_diffusers(tmp_path / "modules")
    receipt = tmp_path / "wan-resolution.json"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$WAN_I2V_MODEL")"',
                'snapshot="$cache/snapshots/$WAN_I2V_REF"',
                'mkdir -p "$cache/blobs" "$snapshot/transformer"',
                'printf \'%s\\n\' \'{"weight_map": {"transformer.weight": "model-00001-of-00001.safetensors"}}\' > "$cache/blobs/index"',
                'printf weight > "$cache/blobs/shard"',
                'ln -s ../../../blobs/index "$snapshot/transformer/model.safetensors.index.json"',
                'ln -s ../../../blobs/shard "$snapshot/transformer/model-00001-of-00001.safetensors"',
                'mkdir -p "$(source_tree)"',
                "export HF_HUB_OFFLINE=1",
                'verify_offline_wan_sharded_snapshot "$(source_tree)"',
                f'cp "$(source_tree)/npa_wan_offline_shard_resolution.json" {receipt}',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(fake_modules),
        },
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "index_sha256": "1ec575028a26d148cef741008b1fb9d07ed3967fa34c6009dc9d1a29571cbace",
        "repository": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "revision": "b184e23a8a16b20f108f727c902e769e873ffc73",
        "schema": "npa.hy_world.wan_offline_shards.v1",
        "shard_count": 1,
        "weight_map_entries": 1,
    }

    direct_blob_receipt = tmp_path / "wan-resolution-direct-blobs.json"
    direct_blob = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                'mkdir -p "$(source_tree)"',
                "export HF_HUB_OFFLINE=1 NPA_TEST_RETURN_BLOB_PATHS=1",
                'verify_offline_wan_sharded_snapshot "$(source_tree)"',
                f'cp "$(source_tree)/npa_wan_offline_shard_resolution.json" {direct_blob_receipt}',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(fake_modules),
        },
    )

    assert direct_blob.returncode == 0, direct_blob.stderr
    assert (
        json.loads(direct_blob_receipt.read_text(encoding="utf-8"))["shard_count"] == 1
    )

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    external = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$WAN_I2V_MODEL")"',
                'snapshot="$cache/snapshots/$WAN_I2V_REF"',
                'ln -sfn "$NPA_TEST_OUTSIDE" "$snapshot/transformer/model-00001-of-00001.safetensors"',
                "export HF_HUB_OFFLINE=1",
                'verify_offline_wan_sharded_snapshot "$(source_tree)"',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_OUTSIDE": str(outside),
            "NPA_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(fake_modules),
        },
    )

    assert external.returncode != 0
    assert "bytes escape the pinned repository cache" in external.stderr

    escaped = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                "configure_runtime_hf_cache",
                'cache="$(hub_cache_directory "$WAN_I2V_MODEL")"',
                'snapshot="$cache/snapshots/$WAN_I2V_REF"',
                'printf "%s\\n" \'{"weight_map": {"transformer.weight": "../escaped.safetensors"}}\' > "$cache/blobs/index"',
                "export HF_HUB_OFFLINE=1",
                'verify_offline_wan_sharded_snapshot "$(source_tree)"',
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(fake_modules),
        },
    )

    assert escaped.returncode != 0
    assert "traverses its snapshot" in escaped.stderr


def test_worldnav_handoff_requires_a_real_reconstruct_pairs_file(
    tmp_path: Path,
) -> None:
    """Regular cameras cannot substitute for WorldNav's explicit handoff."""

    scene = tmp_path / "scene"
    pairs = scene / "navmesh" / "reconstruct_pairs.json"
    pairs.parent.mkdir(parents=True)
    pairs.write_text('[{"target_pt": [1.0, 2.0, 3.0]}]', encoding="utf-8")
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                f"verify_worldnav_handoff {scene}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
        },
    )
    assert result.returncode == 0, result.stderr

    pairs.write_text("[]", encoding="utf-8")
    no_target = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                f"verify_worldnav_handoff {scene}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
        },
    )
    assert no_target.returncode == 0, no_target.stderr

    pairs.unlink()
    missing = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'runtime_python() { printf "%s\\n" "$NPA_TEST_PYTHON"; }',
                f"verify_worldnav_handoff {scene}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_PYTHON": sys.executable,
        },
    )
    assert missing.returncode != 0
    assert "WorldNav did not write" in missing.stderr


def test_completed_cache_layout_is_checked_for_relocated_console_scripts(
    tmp_path: Path,
) -> None:
    """A legacy .complete cache with an .incomplete script cannot be reused."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    venv = tree / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(tree / ".report-venv")], check=True
    )
    stale = venv / "bin" / "legacy-console"
    stale.write_text(f"#!{tree}.incomplete/.venv/bin/python\n", encoding="utf-8")
    stale.chmod(0o755)

    result = _bash(
        f"source {RUNTIME}; verify_runtime_layout {tree}",
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode != 0
    assert "still references staging tree" in result.stderr


def test_completed_cu126_cache_is_not_accepted_for_the_cu128_runtime(
    tmp_path: Path,
) -> None:
    """A source-matched legacy cache cannot bypass the CUDA wheel contract."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    tree.mkdir(parents=True)
    (tree / ".complete").touch()
    (tree / "npa_runtime_contract.json").write_text(
        """{
  "schema": "npa.hy_world.runtime_contract.v1",
  "revision": "legacy-cu126",
  "torch": "2.7.1+cu126",
  "torchvision": "0.22.1+cu126",
  "pytorch_wheel_index": "https://download.pytorch.org/whl/cu126"
}
""",
        encoding="utf-8",
    )

    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'git() { printf "%s\\n" "$SOURCE_REF"; }',
                "if source_is_complete; then exit 91; fi",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr


def test_runtime_contract_retains_the_exact_pytorch_wheel_index(tmp_path: Path) -> None:
    """The contract records the exact wheel URL and full VGG asset identity."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    subprocess.run([sys.executable, "-m", "venv", str(tree / ".venv")], check=True)
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                f"write_runtime_contract {tree}",
                f"verify_runtime_contract {tree}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    contract = json.loads(
        (tree / "npa_runtime_contract.json").read_text(encoding="utf-8")
    )
    assert contract["pytorch_wheel_index"] == "https://download.pytorch.org/whl/cu128"
    assert contract["rtree"] == "1.4.1"
    assert contract["vgg16_sha256"] == (
        "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
    )


def test_blackwell_wheel_install_uses_the_reviewed_cu128_selection(
    tmp_path: Path,
) -> None:
    """The bootstrap asks uv for the reviewed CUDA 12.8 torch pair."""

    arguments = tmp_path / "uv-arguments.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'uv() { printf "%s\\n" "$@" > "$NPA_TEST_UV_ARGUMENTS"; }',
                "install_blackwell_torch_wheels /runtime/python",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_UV_ARGUMENTS": str(arguments),
        },
    )

    assert result.returncode == 0, result.stderr
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "pip",
        "install",
        "--python",
        "/runtime/python",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
        "torch==2.7.1+cu128",
        "torchvision==0.22.1+cu128",
    ]


@pytest.mark.parametrize(
    ("torch_requirement", "torchvision_requirement"),
    (
        ("torch==2.7.2", "torchvision==0.22.1"),
        ("torch==2.7.1", "torchvision==0.22.2"),
    ),
)
def test_requirements_torch_or_vision_drift_refuses_before_uv_install(
    tmp_path: Path,
    torch_requirement: str,
    torchvision_requirement: str,
) -> None:
    """Unreviewed upstream torch pins cannot silently replace the cu128 pair."""

    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "requirements.txt").write_text(
        "\n".join(
            (
                torch_requirement,
                torchvision_requirement,
                "numpy==1.26.4",
                "scipy==1.14.1",
                "torchmetrics==1.9.0",
                "kept-package==3.0",
                "",
            )
        ),
        encoding="utf-8",
    )
    invoked = tmp_path / "uv-invoked.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'uv() { printf invoked > "$NPA_TEST_UV_INVOKED"; }',
                f"install_upstream_requirements {tree} /runtime/python",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_UV_INVOKED": str(invoked),
        },
    )

    assert result.returncode != 0
    assert "upstream native requirements drifted" in result.stderr
    assert not invoked.exists()


def test_requirements_install_keeps_non_torch_requirements(tmp_path: Path) -> None:
    """Removing reviewed torch pins leaves the rest of upstream requirements intact."""

    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "requirements.txt").write_text(
        "\n".join(
            (
                "torch==2.7.1",
                "torchvision==0.22.1",
                "numpy==1.26.4",
                "scipy==1.14.1",
                "torchmetrics",
                "kept-package==3.0",
                "",
            )
        ),
        encoding="utf-8",
    )
    arguments = tmp_path / "uv-arguments.txt"
    copied_requirements = tmp_path / "filtered-requirements.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "uv() {",
                '  printf "%s\\n" "$@" > "$NPA_TEST_UV_ARGUMENTS";',
                '  cp "$6" "$NPA_TEST_FILTERED_REQUIREMENTS";',
                "}",
                f"install_upstream_requirements {tree} /runtime/python",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_UV_ARGUMENTS": str(arguments),
            "NPA_TEST_FILTERED_REQUIREMENTS": str(copied_requirements),
        },
    )

    assert result.returncode == 0, result.stderr
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "pip",
        "install",
        "--python",
        "/runtime/python",
        "-r",
        str(tree / "npa_requirements_without_torch.txt"),
        "torchmetrics==1.9.0",
        "rtree==1.4.1",
    ]
    assert copied_requirements.read_text(encoding="utf-8") == (
        "numpy==1.26.4\nscipy==1.14.1\nkept-package==3.0\n"
    )


@pytest.mark.parametrize(
    "drifted_requirement",
    ("numpy==2.4.6", "scipy==1.14.2", "torchmetrics==1.10.0"),
)
def test_requirements_native_dependency_drift_refuses_before_uv_install(
    tmp_path: Path, drifted_requirement: str
) -> None:
    """The report SDK cannot silently replace the upstream scientific closure."""

    tree = tmp_path / "source"
    tree.mkdir()
    requirements = {
        "numpy": "numpy==1.26.4",
        "scipy": "scipy==1.14.1",
        "torchmetrics": "torchmetrics",
    }
    requirements[drifted_requirement.partition("==")[0]] = drifted_requirement
    (tree / "requirements.txt").write_text(
        "\n".join(
            (
                "torch==2.7.1",
                "torchvision==0.22.1",
                requirements["numpy"],
                requirements["scipy"],
                requirements["torchmetrics"],
                "",
            )
        ),
        encoding="utf-8",
    )
    invoked = tmp_path / "uv-invoked.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'uv() { printf invoked > "$NPA_TEST_UV_INVOKED"; }',
                f"install_upstream_requirements {tree} /runtime/python",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_UV_INVOKED": str(invoked),
        },
    )

    assert result.returncode != 0
    assert "upstream native requirements drifted" in result.stderr
    assert not invoked.exists()


def test_report_sdk_is_installed_in_a_separate_report_interpreter(
    tmp_path: Path,
) -> None:
    """Rerun uses its own closure and cannot replace native NumPy in-place."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    arguments = tmp_path / "uv-arguments.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                'uv() { printf "%s\\n" "$@" >> "$NPA_TEST_UV_ARGUMENTS"; }',
                f"install_report_runtime {tree} /native/.venv/bin/python",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
            "NPA_TEST_UV_ARGUMENTS": str(arguments),
        },
    )

    assert result.returncode == 0, result.stderr
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "venv",
        "--python",
        "/native/.venv/bin/python",
        str(tree / ".report-venv"),
        "pip",
        "install",
        "--python",
        str(tree / ".report-venv/bin/python"),
        "rerun-sdk==0.31.4",
    ]


def test_upstream_nested_torchrun_resolves_from_the_native_runtime(
    tmp_path: Path,
) -> None:
    """WorldMirror inherits the native venv executable rather than image PATH."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    native = tree / ".venv" / "bin"
    native.mkdir(parents=True)
    torchrun = native / "torchrun"
    torchrun.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    torchrun.chmod(0o755)
    result_path = tmp_path / "torchrun-path"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "prefer_native_runtime_bin",
                f"command -v torchrun > {result_path}",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result_path.read_text(encoding="utf-8").strip() == str(torchrun)


def test_layout_rejects_a_completed_cache_with_a_stale_runtime_contract(
    tmp_path: Path,
) -> None:
    """The reuse path verifies the runtime wheel contract as well as venv paths."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    venv = tree / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(tree / ".report-venv")], check=True
    )
    (tree / "npa_runtime_contract.json").write_text("{}\n", encoding="utf-8")

    result = _bash(
        f"source {RUNTIME}; verify_runtime_layout {tree}",
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode != 0
    assert "runtime contract does not match" in result.stderr


def test_ensure_revalidates_an_existing_complete_runtime_before_reuse(
    tmp_path: Path,
) -> None:
    """The complete-cache fast path invokes the layout guard before returning."""

    cache_root = tmp_path / "runtime"
    checked = tmp_path / "checked.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "health() { :; }",
                "validate_runtime_paths() { :; }",
                "source_is_complete() { return 0; }",
                f'verify_runtime_layout() {{ printf "%s\\n" "$1" > {checked}; }}',
                "verify_torch_blackwell_runtime() { :; }",
                "ensure",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(cache_root),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert checked.read_text(encoding="utf-8").strip() == (
        f"{cache_root}/source-df9988efb87bfc0f4947eb3889411cf957478b06"
    )
