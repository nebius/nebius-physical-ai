"""Unit tests for `npa.workbench.cosmos.text_to_image`.

The retired template could not be tested at all: its inference procedure lived in a multi-line
environment variable. As a module, the parts that do not need an H100 are checkable — the job
document, the argv, and above all the verification, which is what stands between "exit 0" and
"a real image".
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import pytest

from npa.workbench.cosmos import text_to_image as t2i


def _png(width: int, height: int, *, pad: int = 4096) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        # Random padding: zlib squashes a run of zeros to ~30 bytes, which the size check
        # (correctly) rejects as a truncated image.
        + chunk(b"IDAT", zlib.compress(os.urandom(pad)))
        + chunk(b"IEND", b"")
    )


def _jpeg(width: int, height: int, *, pad: int = 4096) -> bytes:
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x00" * 9
    return b"\xff\xd8" + sof + b"\xff\xdb" + struct.pack(">H", 2 + pad) + b"\x00" * pad + b"\xff\xd9"


def test_reads_png_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(_png(1024, 576))
    assert t2i.image_dimensions(path) == (1024, 576)


def test_reads_jpeg_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(_jpeg(1280, 720))
    assert t2i.image_dimensions(path) == (1280, 720)


def test_verify_accepts_a_real_image(tmp_path: Path) -> None:
    path = tmp_path / "ok.png"
    path.write_bytes(_png(512, 512))
    size, width, height = t2i.verify_image(path)
    assert (width, height) == (512, 512)
    assert size > t2i.MIN_IMAGE_BYTES


def test_verify_rejects_a_missing_image(tmp_path: Path) -> None:
    with pytest.raises(t2i.Cosmos3TextToImageError, match="produced no image"):
        t2i.verify_image(tmp_path / "nope.png")


def test_verify_rejects_a_truncated_image(tmp_path: Path) -> None:
    """The failure worth catching: the framework exits 0 having written nothing useful."""

    path = tmp_path / "tiny.png"
    path.write_bytes(_png(8, 8, pad=1))
    with pytest.raises(t2i.Cosmos3TextToImageError, match="expected at least"):
        t2i.verify_image(path)


def test_verify_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_bytes(b"traceback follows\n" * 500)
    with pytest.raises(t2i.Cosmos3TextToImageError, match="unrecognised image format"):
        t2i.verify_image(path)


def test_job_document_matches_the_framework_contract() -> None:
    assert t2i.build_job_document("a robot") == {
        "model_mode": "text2image",
        "name": "npa-t2i",
        "prompt": "a robot",
    }


def test_inference_argv_is_an_argv_not_a_string(tmp_path: Path) -> None:
    argv = t2i.inference_argv(
        input_json=tmp_path / "in.json",
        output_dir=tmp_path / "out",
        checkpoint_name="Cosmos3-Nano",
        seed=7,
        guardrails=False,
    )

    assert argv[:3] == [".venv/bin/python", "-m", "cosmos_framework.scripts.inference"]
    assert "--no-guardrails" in argv
    assert "--seed=7" in argv
    assert argv[argv.index("--checkpoint-path") + 1] == "Cosmos3-Nano"
    # No shell interpolation anywhere: every element is a discrete token.
    assert all(" " not in part or part.startswith("/") or "tmp" in part for part in argv)


def test_guardrails_flag_is_opt_in(tmp_path: Path) -> None:
    argv = t2i.inference_argv(
        input_json=tmp_path / "in.json",
        output_dir=tmp_path / "out",
        checkpoint_name="Cosmos3-Nano",
        seed=0,
        guardrails=True,
    )
    assert "--no-guardrails" not in argv


def test_publish_writes_the_manifest_and_uploads_both(tmp_path: Path, monkeypatch) -> None:
    uploads: list[tuple[str, str]] = []

    class _Client:
        @staticmethod
        def from_environment() -> "_Client":
            return _Client()

        def upload_file(self, local: str, uri: str) -> str:
            uploads.append((Path(local).name, uri))
            return uri

    monkeypatch.setitem(
        __import__("sys").modules,
        "npa.clients.storage",
        type("m", (), {"StorageClient": _Client}),
    )

    image = tmp_path / t2i.IMAGE_FILENAME
    image.write_bytes(_png(64, 64))
    result = t2i.TextToImageResult(
        status="ok",
        prompt="a robot",
        model_id="nvidia/Cosmos3-Nano",
        output_image=str(image),
        bytes=image.stat().st_size,
        width=64,
        height=64,
        seed=0,
        source_dir="/tmp/src",
        checkpoint_dir="/tmp/ckpt",
    )

    published = t2i.publish(result, tmp_path, "s3://bucket/run")

    assert published["image_uri"] == "s3://bucket/run/" + t2i.IMAGE_FILENAME
    assert published["manifest_uri"] == "s3://bucket/run/" + t2i.MANIFEST_FILENAME
    manifest = json.loads((tmp_path / t2i.MANIFEST_FILENAME).read_text())
    assert manifest["schema"] == "npa.cosmos3.text_to_image.v1"
    assert manifest["prompt"] == "a robot"
    assert sorted(name for name, _ in uploads) == sorted(
        [t2i.IMAGE_FILENAME, t2i.MANIFEST_FILENAME]
    )


def test_generate_requires_a_prompt(tmp_path: Path) -> None:
    from npa.workbench.cosmos.cosmos3 import Cosmos3AccessConfig

    with pytest.raises(t2i.Cosmos3TextToImageError, match="prompt is required"):
        t2i.generate(
            Cosmos3AccessConfig.from_env(environ={}),
            prompt="   ",
            output_dir=tmp_path,
        )


def test_the_spec_declares_the_manifest_the_tool_writes() -> None:
    import yaml

    spec_path = (
        Path(__file__).resolve().parents[3]
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "cosmos3-text-to-image.yaml"
    )
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    declared = spec["states"]["text-to-image"]["outputs"][0]["uri"]

    assert declared.endswith(t2i.MANIFEST_FILENAME)


def test_generate_reports_a_failed_fetch_instead_of_a_confusing_attribute_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Live job 289 died with `'Cosmos3FetchResult' object has no attribute 'source_dir'`.

    Two bugs in one line: the wrong field name, and no check of `ok` — so a fetch that failed
    for a real reason (no token, no network, gated repo) would surface as an AttributeError
    with the actual cause discarded.
    """

    from npa.workbench.cosmos import cosmos3 as cosmos3_module
    from npa.workbench.cosmos.cosmos3 import Cosmos3AccessConfig, Cosmos3FetchResult

    monkeypatch.setattr(
        t2i,
        "fetch_cosmos3_artifacts",
        lambda *a, **k: Cosmos3FetchResult(
            ok=False,
            cache_dir="/tmp/c",
            source_checkout="",
            checkpoint_dir="",
            checkpoint="",
            reasoning_parser="qwen3",
            tool_call_parser="hermes",
            errors=("HF token missing",),
        ),
    )
    assert cosmos3_module is not None

    with pytest.raises(t2i.Cosmos3TextToImageError, match="HF token missing"):
        t2i.generate(
            Cosmos3AccessConfig.from_env(environ={}),
            prompt="a robot",
            output_dir=tmp_path / "out",
        )


def test_generate_uses_the_field_the_fetch_result_actually_has(tmp_path: Path, monkeypatch) -> None:
    from npa.workbench.cosmos.cosmos3 import Cosmos3AccessConfig, Cosmos3FetchResult

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(
        t2i,
        "fetch_cosmos3_artifacts",
        lambda *a, **k: Cosmos3FetchResult(
            ok=True,
            cache_dir=str(tmp_path),
            source_checkout=str(source),
            checkpoint_dir=str(tmp_path / "ckpt"),
            checkpoint="Cosmos3-Nano",
            reasoning_parser="qwen3",
            tool_call_parser="hermes",
        ),
    )
    ran: list[list[str]] = []

    def fake_run(argv, *, cwd, env, what):
        ran.append(list(argv))
        if "cosmos_framework.scripts.inference" in argv:
            produced = Path(env["NPA_TEST_OUTPUT_DIR"]) / t2i.FRAMEWORK_OUTPUT_RELPATH
            produced.parent.mkdir(parents=True, exist_ok=True)
            produced.write_bytes(_png(320, 240))

    monkeypatch.setattr(t2i, "_run", fake_run)
    # uv resolution is covered separately; this test is about the fetch-result plumbing.
    monkeypatch.setattr(t2i, "uv_argv", lambda: ["uv"])
    monkeypatch.setenv("NPA_TEST_OUTPUT_DIR", str(tmp_path / "out"))

    result = t2i.generate(
        Cosmos3AccessConfig.from_env(environ={}),
        prompt="a robot",
        output_dir=tmp_path / "out",
    )

    assert result.status == "ok"
    assert (result.width, result.height) == (320, 240)
    assert result.source_dir == str(source)
    assert ran[0][:2] == ["uv", "sync"]


def test_hf_cli_falls_back_to_the_module_when_the_script_is_not_on_path(monkeypatch) -> None:
    """Live job 290: `No such file or directory: 'huggingface-cli'`, moments after installing it.

    Console scripts land in whichever scripts directory pip chose; under a PEP 668 `--user`
    fallback that is not the one on PATH. The module is importable from the interpreter that
    installed it by construction, so it is the more reliable entry point.
    """

    import sys

    from npa.workbench.cosmos import cosmos3 as cosmos3_module

    monkeypatch.setattr(cosmos3_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        cosmos3_module.importlib.util, "find_spec", lambda name: object() if name else None
    )
    argv = cosmos3_module._huggingface_cli()

    assert argv == [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli"]


def test_hf_cli_keeps_the_plain_name_when_neither_is_available(monkeypatch) -> None:
    """So the error names the missing tool rather than a confusing module path."""

    from npa.workbench.cosmos import cosmos3 as cosmos3_module

    monkeypatch.setattr(cosmos3_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cosmos3_module.importlib.util, "find_spec", lambda _name: None)

    assert cosmos3_module._huggingface_cli() == ["huggingface-cli"]


def test_hf_cli_prefers_a_real_executable(monkeypatch) -> None:
    from npa.workbench.cosmos import cosmos3 as cosmos3_module

    monkeypatch.setattr(cosmos3_module.shutil, "which", lambda _name: "/usr/bin/huggingface-cli")

    assert cosmos3_module._huggingface_cli() == ["/usr/bin/huggingface-cli"]


def test_uv_argv_prefers_the_module(monkeypatch) -> None:
    """Live job 291: a uv on setup's PATH is not a uv the stage command can run."""

    import importlib.util
    import shutil
    import sys

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "uv" else None)
    monkeypatch.setattr(shutil, "which", lambda _name: "/somewhere/else/uv")

    assert t2i.uv_argv() == [sys.executable, "-m", "uv"]


def test_uv_argv_says_what_is_missing(monkeypatch) -> None:
    import importlib.util
    import shutil

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(t2i.Cosmos3TextToImageError, match="uv is required"):
        t2i.uv_argv()


def test_runtime_library_dir_picks_a_libstdcxx_that_exports_the_symbol(tmp_path: Path) -> None:
    """Live job 296: `version GLIBCXX_3.4.29 not found` for transformer_engine.

    The check reads the library rather than trusting a path or a distro version, because that is
    literally the question the loader will ask.
    """

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "libstdc++.so.6").write_bytes(b"GLIBCXX_3.4.25\x00GLIBCXX_3.4.28\x00")
    (new / "libstdc++.so.6").write_bytes(b"GLIBCXX_3.4.28\x00GLIBCXX_3.4.29\x00")

    assert t2i.runtime_library_dir([old, new]) == str(new)
    assert t2i.runtime_library_dir([new, old]) == str(new)


def test_runtime_library_dir_is_empty_when_nothing_qualifies(tmp_path: Path) -> None:
    (tmp_path / "libstdc++.so.6").write_bytes(b"GLIBCXX_3.4.25\x00")

    assert t2i.runtime_library_dir([tmp_path, tmp_path / "missing"]) == ""


def test_only_libstdcxx_is_put_on_the_loader_path(tmp_path: Path, monkeypatch) -> None:
    """Live job 319: `cuDNN version incompatibility ... conflicting cuDNN in LD_LIBRARY_PATH`.

    A directory on LD_LIBRARY_PATH brings everything in it. The conda prefix that supplies a
    modern libstdc++ also supplies an older cuDNN, and PyTorch refuses to run against a cuDNN
    other than the one it bundles.
    """

    source_dir = tmp_path / "conda-lib"
    source_dir.mkdir()
    (source_dir / "libstdc++.so.6").write_bytes(b"GLIBCXX_3.4.29\x00")
    (source_dir / "libcudnn.so.9").write_bytes(b"an older cudnn")

    monkeypatch.setattr(t2i, "runtime_library_dir", lambda: str(source_dir))
    monkeypatch.setattr(t2i, "_has_required_glibcxx", lambda path: "conda-lib" in str(path))

    shim = t2i.link_runtime_library(tmp_path / "shim")

    assert shim == str(tmp_path / "shim")
    assert sorted(p.name for p in (tmp_path / "shim").iterdir()) == ["libstdc++.so.6"]
    assert (tmp_path / "shim" / "libstdc++.so.6").resolve() == source_dir / "libstdc++.so.6"


def test_nothing_is_added_when_the_host_libstdcxx_is_already_new_enough(tmp_path: Path, monkeypatch) -> None:
    """Adding to the loader path when the host is fine is pure risk."""

    monkeypatch.setattr(t2i, "runtime_library_dir", lambda: "/opt/conda/lib")
    monkeypatch.setattr(t2i.Path, "is_file", lambda self: True)
    monkeypatch.setattr(t2i, "_has_required_glibcxx", lambda path: True)

    assert t2i.link_runtime_library(tmp_path / "shim") == ""
