"""Select an explicit FA2/FA4 benchmark backend without silently substituting kernels."""

from importlib import metadata
import hashlib
import os
from pathlib import Path
import re
import subprocess

FA_COMMIT = "eed1971f5132630dc296fe37601e834d4b57a248"
FA4_VERSION = "4.0.0b33.dev10+geed1971"


def _validate_selection(backend, tile, tuning):
    if backend not in ("fa2", "fa4") or ((tile or tuning) and backend != "fa4"):
        raise ValueError("Tiles require FA4; backend must be fa2 or fa4")
    if tuning not in (None, "rtx6000-inference") or (tile and tuning):
        raise ValueError("Select one qualified tuning profile or one tile")
    if os.getenv("NPA_ATTENTION_BACKEND") != backend:
        raise RuntimeError("Run each backend in its matching base image")


def attention_backend(backend, tile=None, tuning=None):
    """Load the installed backend, optionally testing an FA4 inference tile.

    Args:
        backend: ``fa2`` or ``fa4``; each requires its own base image.
        tile: Experimental FA4 forward tile (M, N), or None for upstream defaults.
        tuning: Opt-in ``rtx6000-inference`` profile, mutually exclusive with tile.
    Returns:
        A Q/K/V callable accepting the causal keyword and returning a tensor.
    Raises:
        ValueError: The backend or tile is invalid.
        RuntimeError: The installed image does not match the requested backend.
    """
    _validate_selection(backend, tile, tuning)
    if backend == "fa2":
        if not metadata.version("flash-attn").startswith("2."):
            raise RuntimeError("Standalone FA2 2.x is required")
        # The package loads PyTorch's shared libraries before its CUDA extension.
        from flash_attn import flash_attn_func
        import flash_attn_2_cuda  # noqa: F401

        return flash_attn_func
    if tuning:
        from flash_attn.rtx import make_inference_attention

        return make_inference_attention()
    from flash_attn.cute import flash_attn_func

    if tile:
        return _tiled_inference(tile)

    def fa4(query, key, value, *, causal=False):
        result = flash_attn_func(
            query, key, value, causal=causal, pack_gqa=False, num_splits=1
        )
        return result[0] if isinstance(result, tuple) else result

    return fa4


def _pinned_version():
    version, separator, source = metadata.version("flash-attn-4").partition("+g")
    # Match the pinned source even when Git chooses a longer unique abbreviation.
    return (
        version == FA4_VERSION.partition("+g")[0]
        and bool(separator)
        and len(source) >= 7
        and FA_COMMIT.startswith(source)
    )


def _tiled_inference(tile):
    import torch

    if os.getenv("NPA_FLASH_ATTN_COMMIT") != FA_COMMIT or not _pinned_version():
        raise RuntimeError("Experimental tiles require the exact pinned FA4 source")
    if tuple(tile) not in ((64, 64), (64, 128), (128, 64), (128, 128)):
        raise ValueError("Unqualified tile candidate")
    from flash_attn.cute.interface import _flash_attn_fwd

    def forward(query, key, value, *, causal=False):
        if torch.is_grad_enabled() or any(t.requires_grad for t in (query, key, value)):
            raise ValueError("Experimental tiles are inference-only")
        if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (
            12,
            0,
        ):
            raise ValueError("Experimental tiles require SM120")
        return _flash_attn_fwd(
            query,
            key,
            value,
            causal=causal,
            tile_mn=tuple(tile),
            pack_gqa=False,
            num_splits=1,
        )[0]

    return forward


def _benchmark_sources():
    names = (
        "attention_benchmark_backend.py",
        "attention_kernel_benchmark.py",
        "attention_sdxl_benchmark.py",
        "fa4_sdxl_attention.py",
        "fa4_sdxl_validation.py",
        "fa4_sdxl_model.json",
    )
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }


def _package_versions(backend):
    # Normalize distribution names, and resolve duplicates in sys.path order.
    names = {
        re.sub(r"[-_.]+", "-", entry.metadata["Name"]).lower()
        for entry in metadata.distributions()
    }
    required = "flash-attn" if backend == "fa2" else "flash-attn-4"
    if required not in names:
        raise RuntimeError(f"Missing distribution for requested backend: {required}")
    return {name: metadata.version(name) for name in sorted(names)}


def _attention_sources(backend):
    if backend != "fa4":
        return {}
    from flash_attn import rtx

    return {
        "rtx_inference": hashlib.sha256(Path(rtx.__file__).read_bytes()).hexdigest()
    }


def benchmark_environment(backend):
    """Capture portable software and GPU identity for comparing like environments.

    Args:
        backend: The requested FA2 or FA4 backend.
    Returns:
        GPU model, compute capability and installed package versions.
    Raises:
        RuntimeError: CUDA is unavailable or the GPU is not SM120.
    """
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This comparison requires an SM120 GPU")
    image_id = os.getenv("NPA_BENCHMARK_IMAGE_ID", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeError(
            "Pass the inspected Docker image ID as NPA_BENCHMARK_IMAGE_ID"
        )
    driver = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "image_id": image_id,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "flash_attention_commit": os.getenv("NPA_FLASH_ATTN_COMMIT"),
        "benchmark_sources_sha256": _benchmark_sources(),
        "attention_sources_sha256": _attention_sources(backend),
        "packages": _package_versions(backend),
    }
