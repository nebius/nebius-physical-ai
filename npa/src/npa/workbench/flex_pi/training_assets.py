"""Materialize and verify the pinned public Flex-Pi training inputs at runtime."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(root, files):
    for entry in files:
        path = root / entry["path"]
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"immutable training input mismatch: {entry['path']}")


def _huggingface(entry, root):
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=entry["repository"],
            repo_type=entry.get("repo_type", "model"),
            revision=entry["revision"],
            local_dir=str(root),
            allow_patterns=[row["path"] for row in entry["files"]],
        )
    )
    _verify(path, entry["files"])
    return path


def _modelscope(entry, root):
    from modelscope import snapshot_download

    remote = subprocess.run(
        [
            "git",
            "ls-remote",
            f"https://www.modelscope.cn/{entry['repository']}.git",
            f"refs/heads/{entry['sdk_branch']}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if remote.stdout.split()[:1] != [entry["revision"]]:
        raise RuntimeError("ModelScope branch moved from the pinned training revision")
    snapshot_download(
        entry["repository"],
        revision=entry["sdk_branch"],
        local_dir=str(root),
        allow_file_pattern=[row["path"] for row in entry["files"]],
    )
    _verify(root, entry["files"])
    return root


def _source(entry, root):
    for row in entry["files"]:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and _sha256(path) == row["sha256"]:
            continue
        url = f"https://raw.githubusercontent.com/{entry['repository']}/{entry['revision']}/{row['path']}"
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    _verify(root, entry["files"])


def _hydra_overrides(root):
    return [
        "task=yam_unified_flex_3cam_32d_rel_1e-4",
        f"data.train.dataset_dirs=[{root / 'dataset'}]",
        f"data.train.text_embedding_cache_dir={root / 'text'}",
        "data.train.val_set_proportion=0.1",
        "data.train.skip_padding_as_possible=false",
    ]


def _prepare_derived(root, input_identity):
    action = root / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    receipt = root / "derived-initialization.json"
    env = os.environ.copy()
    for name in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
    ):
        env.pop(name, None)
    if receipt.is_file():
        expected = json.loads(receipt.read_text())
        if (
            expected.get("source_manifest_sha256") != input_identity
            or _sha256(action) != expected["sha256"]
        ):
            raise RuntimeError("cached derived ActionDiT initialization changed")
    else:
        if action.exists():
            raise RuntimeError(
                "unreceipted derived ActionDiT cache requires a fresh asset directory"
            )
        _derive_action_backbone(root / "upstream", action, env)
        receipt.write_text(
            json.dumps(
                {"sha256": _sha256(action), "source_manifest_sha256": input_identity}
            )
        )
    _prepare_text(root, input_identity, env)


def _derive_action_backbone(source, action, env):
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/preprocess_action_dit_backbone.py"),
            "--model-config",
            str(source / "configs/model/flexpi.yaml"),
            "--output",
            str(action),
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
        ],
        check=True,
        env=env,
    )


def _prepare_text(root, input_identity, env):
    text_receipt = root / "text-cache-manifest.json"
    if text_receipt.is_file():
        expected = json.loads(text_receipt.read_text())
        if expected.get("source_manifest_sha256") != input_identity or not expected.get(
            "files"
        ):
            raise RuntimeError("cached text embeddings have a different input identity")
        _verify(root / "text", expected["files"])
        return
    if any((root / "text").rglob("*.pt")):
        raise RuntimeError(
            "unreceipted text embeddings require a fresh asset directory"
        )
    subprocess.run(
        [
            sys.executable,
            str(root / "upstream/scripts/precompute_text_embeds.py"),
            *_hydra_overrides(root),
        ],
        check=True,
        env=env,
    )
    files = [
        {
            "path": path.relative_to(root / "text").as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted((root / "text").rglob("*.pt"))
    ]
    if not files:
        raise RuntimeError("official text preprocessing produced no cached embeddings")
    text_receipt.write_text(
        json.dumps(
            {"source_manifest_sha256": input_identity, "files": files}, sort_keys=True
        )
    )


def prepare_assets(root: Path) -> dict:
    """Fetch the public dataset, initialization assets and official preparation code.

    Args:
        root: Private run-owned runtime cache outside any image layers.
    Returns:
        Content identity receipt covering every declared input byte.
    Raises:
        RuntimeError: A source, revision, or content digest does not match.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).with_name("training_sources.json")
    manifest = json.loads(manifest_path.read_text())
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root / "models")
    _source(manifest["source"], root / "upstream")
    _huggingface(manifest["dataset"], root / "dataset")
    for entry in manifest["models"]:
        destination = root / "models" / entry["repository"]
        if entry["provider"] == "modelscope":
            _modelscope(entry, destination)
        else:
            _huggingface(entry, destination)
        if entry["repository"].startswith("timm/"):
            os.environ["FLEX_PI_DINO_CHECKPOINT"] = str(
                destination / "model.safetensors"
            )
    _prepare_derived(root, _sha256(manifest_path))
    return {
        "source_manifest_sha256": _sha256(manifest_path),
        "immutable_identity_verified": True,
        "dataset_files_verified": len(manifest["dataset"]["files"]),
        "derived_initialization_sha256": _sha256(
            root / manifest["derived_initialization"]["output"]
        ),
        "derived_text_manifest_sha256": _sha256(root / "text-cache-manifest.json"),
    }
