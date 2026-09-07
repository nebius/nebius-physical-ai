"""Exercise shipped tree downloaders with adversarial object listings."""

from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
from pathlib import Path
from functools import partial
from unittest.mock import Mock
from urllib.parse import urlparse

import pytest

from npa import fiftyone_lerobot
from npa.cli import fiftyone
from npa.cli import groot
from npa.cli.workbench import lerobot
from npa.clients.storage import StorageError
from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup
from npa.workbench.robocasa.capabilities import _download_s3_tree
from npa.workflows.token_factory_triage import download_textual_artifacts
from npa.workflows import distill, distill_two_vm


def _generated_fiftyone_downloader(builder, root):
    script = shlex.split(builder("dataset", "s3://bucket/models/"))[2]
    if "sudo docker exec -i" in script:
        tokens = shlex.split(script)
        script = next(tokens[index + 2] for index, token in enumerate(tokens) if token == "bash" and tokens[index + 1] == "-lc")
    code = script.split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    tree = ast.parse(code)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "download_s3")
    namespace = {"Path": Path, "urlparse": urlparse, "os": os, "shutil": shutil, "DATASETS_DIR": root, "NAME": "dataset"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "generated_fiftyone", "exec"), namespace)
    return namespace["download_s3"]


@pytest.mark.parametrize("loader", ["fiftyone", "fiftyone-vm", "fiftyone-container", "robocasa", "triage"])
@pytest.mark.parametrize("relative", ["../../../../escape.txt", "nested/../../escape.txt", "nested\\escape.txt", "nested/valid.txt"])
def test_tree_downloaders_contain_remote_names(tmp_path, monkeypatch, loader, relative):
    s3 = Mock()
    s3.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "models/" + relative}]}]
    s3.download_file.side_effect = lambda _b, _k, p: Path(p).write_bytes(b"source-data")
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: s3)
    root = tmp_path / "root"
    if loader == "fiftyone":
        run = partial(fiftyone_lerobot._download_s3_source, "s3://bucket/models/", "dataset", root)
    elif loader.startswith("fiftyone-"):
        builder = fiftyone._build_load_dataset_command if loader == "fiftyone-vm" else fiftyone._build_container_load_dataset_command
        download = _generated_fiftyone_downloader(builder, root)
        run = partial(download, "s3://bucket/models/")
    elif loader == "robocasa":
        run = partial(_download_s3_tree, "s3://bucket/models/", root)
    else:
        run = partial(download_textual_artifacts, "s3://bucket/models/", root, storage_client=Mock(s3=s3))
    if relative == "nested/valid.txt":
        run()
        assert list(root.rglob("valid.txt"))[0].read_bytes() == b"source-data"
    else:
        with pytest.raises((ValueError, StorageError)):
            run()
        s3.download_file.assert_not_called()
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("relative", ["../../escape", "nested/../../escape", "nested/valid.py"])
def test_rendered_source_staging_contains_downloads(tmp_path, monkeypatch, relative):
    blocks = re.findall(r"python3 - <<'PY'\n(.*?)\nPY", default_npa_setup(), re.S)
    blocks = [block for block in blocks if "s3.download_file" in block]
    assert len(blocks) == 2
    for index, block in enumerate(blocks):
        root = tmp_path / str(index)
        # Redirect only the fixed staging location. Execute the actual rendered
        # listing and write control flow with an in-memory provider response.
        block = re.sub(r"dest = pathlib.Path\('[^']+'\)", f"dest = pathlib.Path({str(root)!r})", block)
        s3 = Mock()
        s3.list_objects_v2.return_value = {"Contents": [{"Key": "source/" + relative}]}
        s3.download_file.side_effect = lambda _b, _k, p: Path(p).write_bytes(b"source-data")
        monkeypatch.setattr("boto3.client", lambda *args, **kwargs: s3)
        monkeypatch.setenv("NPA_SRC_S3_URI", "s3://bucket/source/")
        if relative == "nested/valid.py":
            exec(compile(block, "rendered_source_staging", "exec"), {})
            assert (root / "nested/valid.py").read_bytes() == b"source-data"
        else:
            with pytest.raises(ValueError):
                exec(compile(block, "rendered_source_staging", "exec"), {})
            s3.download_file.assert_not_called()


@pytest.mark.parametrize("loader", ["lerobot", "groot", "distill", "distill-two-vm"])
@pytest.mark.parametrize("relative", ["../escape", "/escape", "nested\\escape", "nested/valid.bin"])
def test_remote_checkpoint_download_scripts_reject_traversal(tmp_path, monkeypatch, loader, relative):
    s3 = Mock()
    s3.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "models/" + relative}]}]
    s3.download_file.side_effect = lambda _b, _k, p: Path(p).write_bytes(b"weights")
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: s3)
    root = tmp_path / "cache with 'quotes' and $literal"
    if loader in {"lerobot", "groot"}:
        module = lerobot if loader == "lerobot" else groot
        command = module._remote_download_dir_cmd("s3://bucket/models/", str(root))
    else:
        ssh = Mock()
        ssh.run.return_value = (0, "s3_download_count=1\ns3_download_done", "")
        if loader == "distill":
            distill._s3_sync_dir(ssh, "", direction="download", s3_uri="s3://bucket/models/", local_path=str(root))
        else:
            monkeypatch.setattr(distill_two_vm, "_conda_activate", lambda _: "")
            distill_two_vm._s3_download(ssh, "", "bucket", "models/", str(root))
        command = ssh.run.call_args.args[0]
    tokens = shlex.split(command)
    code = tokens[tokens.index("-c") + 1]
    if relative == "nested/valid.bin":
        exec(compile(code, "remote_checkpoint_download", "exec"), {})
        assert (root / "nested/valid.bin").read_bytes() == b"weights"
    else:
        with pytest.raises(ValueError):
            exec(compile(code, "remote_checkpoint_download", "exec"), {})
        s3.download_file.assert_not_called()
    assert not (tmp_path / "escape").exists()
