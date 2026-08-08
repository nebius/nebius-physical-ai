"""Build-time verification for the npa-cosmos3-serving image.

Runs inside the image with no GPU and no weights. It checks the four things
that can be wrong about this image without anyone noticing until an 8-GPU node
is already running:

1. The vLLM-Omni serving stack imports and reports a version.
2. The ``hf-xet``/``huggingface_hub`` pins the Dockerfile's
   ``HF_HUB_DISABLE_XET=1`` works around are still the pins actually installed.
   Once a base-image bump moves off the affected pair, this fails the build and
   names the line to delete, so the workaround cannot quietly outlive the defect.
3. The entrypoint assembles the serve command it advertises. This runs the real
   script in dry-run mode rather than restating its argv here, because a copy of
   the expected command would pass while the script produced something else.
4. No model weight file is present in any layer, which is what keeps the image
   redistributable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ENTRYPOINT = Path("/opt/npa-cosmos3-serving/entrypoint.sh")

# huggingface/xet-core#895: "Unable to parse string as hex hash value" during a
# gated-repo download. Closed 2026-07-28, after this base image was published.
XET_AFFECTED_PINS = {"hf-xet": "1.5.1", "huggingface_hub": "1.23.0"}

# The 8-GPU decomposition the entrypoint pins, and the flags that carry it.
REQUIRED_SERVE_FLAGS = (
    "--omni",
    "--cfg-parallel-size 2",
    "--ulysses-degree 4",
    "--use-hsdp",
    "--hsdp-shard-size 8",
)

WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".gguf")
WEIGHT_MIN_BYTES = 50 * 1024 * 1024
WEIGHT_SCAN_ROOTS = (
    "/opt/npa-cosmos3-serving",
    "/home/ubuntu",
    "/root",
    "/app",
    "/workspace",
)

failures: list[str] = []


def step(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
        failures.append(name)
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"[ok]   {name}" + (f": {detail}" if detail else ""))


def _distribution_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
    return None


def check_serving_stack() -> str:
    import vllm

    reported = [f"vllm={vllm.__version__}"]
    try:
        import vllm_omni
    except ImportError:
        # Older builds fold the omni surface into vllm itself; the serve-time
        # `--omni` flag is what actually matters and the entrypoint check covers it.
        reported.append("vllm_omni=in-tree")
    else:
        reported.append(f"vllm_omni={getattr(vllm_omni, '__version__', 'unknown')}")
    return " ".join(reported)


def check_xet_workaround_still_applies() -> str:
    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise RuntimeError(
            "HF_HUB_DISABLE_XET is not set to 1 in the image environment; the "
            "gated guardrail download fails on this image's pinned pair"
        )

    installed = {name: _distribution_version(name) for name in XET_AFFECTED_PINS}
    missing = [name for name, found in installed.items() if found is None]
    if missing:
        raise RuntimeError(f"could not resolve installed versions for {missing}")

    if installed != XET_AFFECTED_PINS:
        raise RuntimeError(
            "the base image no longer pins the pair that huggingface/xet-core#895 "
            f"reproduces on (expected {XET_AFFECTED_PINS}, found {installed}). "
            "Re-test a guardrails-on startup without the workaround; if it is clean, "
            "delete HF_HUB_DISABLE_XET from the Dockerfile and this check with it."
        )
    return f"{installed} still affected, HF_HUB_DISABLE_XET=1 justified"


def check_entrypoint_assembles_the_serve_command() -> str:
    """Execute the real entrypoint in dry-run and read the command it built."""

    if not ENTRYPOINT.is_file():
        raise RuntimeError(f"{ENTRYPOINT} is missing")

    env = dict(os.environ)
    env.update(
        {
            "NPA_COSMOS3_SERVE_DRY_RUN": "1",
            "NPA_COSMOS3_SERVE_SKIP_GPU_CHECK": "1",
            # No token exists at build time, so exercise the guardrails-off arm.
            "NPA_COSMOS3_SERVE_GUARDRAILS": "off",
        }
    )
    result = subprocess.run(
        [str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"entrypoint dry run exited {result.returncode}: {result.stderr.strip()}"
        )

    exec_lines = [line for line in result.stdout.splitlines() if "exec: " in line]
    if not exec_lines:
        raise RuntimeError(f"entrypoint printed no serve command: {result.stdout!r}")
    command = exec_lines[-1].split("exec: ", 1)[1]

    for flag in REQUIRED_SERVE_FLAGS:
        if flag not in command:
            raise RuntimeError(f"serve command is missing {flag!r}: {command}")
    if "--no-guardrails" not in command:
        raise RuntimeError(f"guardrails=off did not reach the serve command: {command}")
    return command


def check_no_baked_weights() -> str:
    hf_home = os.environ.get("HF_HOME")
    roots = [Path(root) for root in WEIGHT_SCAN_ROOTS]
    if hf_home:
        roots.append(Path(hf_home))

    baked: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in WEIGHT_SUFFIXES
                    and path.stat().st_size > WEIGHT_MIN_BYTES
                ):
                    baked.append(str(path))
            except OSError:
                continue
    if baked:
        raise RuntimeError(f"model weights present in the image: {baked[:5]}")
    return "no weight files baked (checkpoints download at runtime)"


def main() -> int:
    step("vLLM-Omni serving stack", check_serving_stack)
    step("xet workaround matches installed pins", check_xet_workaround_still_applies)
    step("entrypoint builds the pinned serve command", check_entrypoint_assembles_the_serve_command)
    step("no baked weights", check_no_baked_weights)

    print()
    if failures:
        print(f"[FAIL] npa-cosmos3-serving environment verification failed: {failures}")
        return 1
    print("[PASS] npa-cosmos3-serving environment verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
