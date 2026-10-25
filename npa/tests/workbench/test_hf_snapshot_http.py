"""Exercise the real Hub SDK against a branch that moves during transfer."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlsplit

import pytest


FIRST = "a" * 40
SECOND = "b" * 40
REPOSITORY = "owner/fixture"


class _HubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _content(self, path):
        if "/tree/" in path:
            revision = path.split("/tree/", 1)[1].split("/", 1)[0]
            body = json.dumps(
                [{"type": "file", "oid": "c" * 40, "size": 41, "path": "selected.txt"}]
            ).encode()
        elif path.startswith("/api/"):
            selected = path.rsplit("/", 1)[-1]
            revision = self.server.revision if selected == "main" else selected
            self.server.revision = SECOND
            body = json.dumps(
                {
                    "id": REPOSITORY,
                    "sha": revision,
                    "private": False,
                    "gated": False,
                    "downloads": 0,
                    "likes": 0,
                    "tags": [],
                    "siblings": [{"rfilename": "selected.txt"}],
                }
            ).encode()
        elif "/resolve/" in path:
            revision = path.split("/resolve/", 1)[1].split("/", 1)[0]
            body = (revision + "\n").encode()
        else:
            return None
        return revision, body

    def _response(self):
        if self.headers.get("Authorization"):
            self.send_error(400, "fixture must not receive operator credentials")
            return
        path = urlsplit(self.path).path
        self.server.paths.append(path)
        content = self._content(path)
        if content is None:
            self.send_error(404)
            return
        revision, body = content
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json" if path.startswith("/api/") else "text/plain",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"' + hashlib.sha256(body).hexdigest() + '"')
        self.send_header("X-Repo-Commit", revision)
        self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)

    do_GET = _response
    do_HEAD = _response


def _child_code(kind, directory):
    call = (
        "from npa.fiftyone_lerobot import _download_huggingface_source as download\n"
        f"target=download('{REPOSITORY}@{FIRST}', 'review', root)\n"
        if kind == "dataset"
        else "from npa.workbench.lerobot.policy_container import _download_eval_checkpoint as download\n"
        f"target=download('{REPOSITORY}@{FIRST}', root)\n"
    )
    return (
        "from pathlib import Path\nimport json\n"
        f"root=Path({str(directory)!r})\n"
        + call
        + f"assert (target/'selected.txt').read_text() == {FIRST!r}+'\\n'\n"
        f"assert json.loads((target/'.npa-hf-source.json').read_text())['revision'] == {FIRST!r}\n"
    )


@pytest.mark.parametrize("kind", ["dataset", "model"])
def test_real_sdk_download_uses_original_commit_after_main_moves(tmp_path, kind):
    pytest.importorskip("huggingface_hub")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
    server.paths = []
    server.revision = FIRST
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HF_", "HUGGING_FACE_", "HUGGINGFACE_"))
    }
    environment.update(
        HF_ENDPOINT=f"http://127.0.0.1:{server.server_port}",
        HF_HOME=str(tmp_path / "hub-cache"),
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        NO_PROXY="127.0.0.1,localhost",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", _child_code(kind, tmp_path / "data")],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stderr
    assert server.revision == SECOND
    assert not any(path.endswith("/main") for path in server.paths)
    downloads = [path for path in server.paths if "/resolve/" in path]
    assert downloads
    assert all(f"/resolve/{FIRST}/" in path for path in downloads)
