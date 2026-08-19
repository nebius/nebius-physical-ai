"""Batch-inference tool tests.

The fake transport mirrors the request/response sequence observed against the
live API: a dataset upload that returns ``current_version``, an operation that
reports ``queued`` before ``succeeded``, results served as an OpenAI-standard
batch output file, and a destination dataset export as the fallback.

The failure case reproduces the real shape exactly: the operations endpoint
returns a single empty error string while the batch record's error file holds the
per-row reason (``not a known batch endpoint routing key``) and
``request_counts`` reports the rows as invalid.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from npa.clients.token_factory import (
    DEFAULT_BATCH_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.token_factory import (
    TokenFactoryToolError,
    _parse_batch_export,
    batch_collect,
    batch_generate,
    batch_operation_uri_for,
)

DATASET_ID = "ds-source-0001"
DATASET_VERSION = "ver-0001"
RESULT_DATASET_ID = "ds-result-0001"
OPERATION_ID = "batch__test-0001"
OUTPUT_FILE_ID = "file-output-0001"
ERROR_FILE_ID = "file-error-0001"
# Verbatim from a live failed batch's error file.
ROUTING_KEY_ERROR = (
    "Invalid request rows 3 of 3 exceed the 10% limit.\n"
    'Line:1 custom_id:p1 model "meta-llama/Llama-3.3-70B-Instruct" is not a known '
    "batch endpoint routing key\n"
)


class FakeBatchApi:
    """Minimal in-memory Token Factory batch API."""

    def __init__(
        self,
        *,
        statuses: list[str] | None = None,
        completions: dict[str, str] | None = None,
        errors: list[str] | None = None,
        in_progress: bool = True,
        result_row_key: str = "response",
        error_file_text: str = "",
        serve_batch_view: bool = True,
        serve_output_file: bool = True,
    ) -> None:
        self.statuses = statuses or ["queued", "succeeded"]
        self.completions = completions if completions is not None else {}
        self.errors = errors if errors is not None else []
        self.in_progress = in_progress
        self.result_row_key = result_row_key
        self.error_file_text = error_file_text
        self.serve_batch_view = serve_batch_view
        self.serve_output_file = serve_output_file
        self.uploaded_rows: list[dict] = []
        self.operation_payload: dict = {}
        self.deleted: list[str] = []
        self.poll_count = 0
        self.exported_datasets: list[str] = []
        self.downloaded_files: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "POST" and path.endswith("/datasets"):
            body = json.loads(request.content.decode("utf-8"))
            self.uploaded_rows = body["rows"]
            return httpx.Response(
                200,
                json={
                    "id": DATASET_ID,
                    "name": body["name"],
                    "status": "READY",
                    "current_version": DATASET_VERSION,
                    "schema": body["schema"],
                },
            )
        if method == "POST" and path.endswith("/operations"):
            self.operation_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json=self._operation("queued"))
        if method == "GET" and path.endswith(f"/operations/{OPERATION_ID}"):
            status = self.statuses[min(self.poll_count, len(self.statuses) - 1)]
            self.poll_count += 1
            return httpx.Response(200, json=self._operation(status))
        if method == "GET" and path.endswith("/errors"):
            return httpx.Response(200, json={"object": "list", "data": self.errors})
        if method == "GET" and f"/batches/{OPERATION_ID}" in path:
            if not self.serve_batch_view:
                return httpx.Response(404, json={"detail": "no batch view"})
            return httpx.Response(200, json=self._batch())
        if method == "GET" and "/files/" in path and path.endswith("/content"):
            file_id = path.split("/files/")[1].split("/")[0]
            self.downloaded_files.append(file_id)
            if file_id == ERROR_FILE_ID:
                return httpx.Response(200, text=self.error_file_text)
            return httpx.Response(200, text=self._results_jsonl())
        if method == "GET" and path.endswith("/export"):
            dataset_id = path.split("/datasets/")[1].split("/")[0]
            self.exported_datasets.append(dataset_id)
            return httpx.Response(200, text=self._export(dataset_id))
        if method == "DELETE" and "/datasets/" in path:
            self.deleted.append(path.rsplit("/", 1)[1])
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"unexpected {method} {path}"})

    def _batch(self) -> dict:
        status = self.statuses[min(max(self.poll_count - 1, 0), len(self.statuses) - 1)]
        failed = status == "failed"
        total = max(len(self.uploaded_rows), len(self.completions))
        return {
            "id": OPERATION_ID,
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "status": "failed" if failed else status,
            "completion_window": "24h",
            "request_counts": {
                "total": total,
                "completed": 0 if failed else len(self.completions),
                "failed": 0,
                "invalid": total if failed else 0,
            },
            "output_file_id": (
                OUTPUT_FILE_ID if status == "succeeded" and self.serve_output_file else None
            ),
            "error_file_id": ERROR_FILE_ID if failed and self.error_file_text else None,
        }

    def _operation(self, status: str) -> dict:
        return {
            "id": OPERATION_ID,
            "type": "batch_inference",
            "status": status,
            "params": {"model": DEFAULT_BATCH_MODEL, "completion_window": "24h"},
            "src": [{"id": DATASET_ID, "version": DATASET_VERSION}],
            "dst": [{"id": RESULT_DATASET_ID, "version": "ver-out"}],
            "in_progress_at": 1 if self.in_progress else None,
        }

    def _export(self, dataset_id: str) -> str:
        if dataset_id == DATASET_ID:
            return "".join(json.dumps(row) + "\n" for row in self.uploaded_rows)
        return self._results_jsonl()

    def _results_jsonl(self) -> str:
        lines = []
        for custom_id, completion in self.completions.items():
            body = {
                "choices": [{"message": {"role": "assistant", "content": completion}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
            lines.append(json.dumps({"custom_id": custom_id, self.result_row_key: body}))
        return "\n".join(lines) + "\n"


def _client(api: FakeBatchApi) -> TokenFactoryClient:
    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(
        config,
        http_client=httpx.Client(transport=httpx.MockTransport(api.handler)),
        sleeper=lambda _seconds: None,
    )


def _prompts(tmp_path: Path, count: int = 2) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"p{index}", "prompt": f"prompt {index}"}) + "\n"
            for index in range(1, count + 1)
        ),
        encoding="utf-8",
    )
    return path


def test_batch_generate_completes_and_maps_prompts(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "answer one", "p2": "answer two"})

    result = batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert result.status == "completed"
    assert result.operation_status == "succeeded"
    assert result.prompt_count == 2
    assert result.generation_count == 2
    assert [(item.id, item.completion) for item in result.generations] == [
        ("p1", "answer one"),
        ("p2", "answer two"),
    ]
    # Prompts come back attached to the completion, recovered by custom_id.
    assert {item.prompt for item in result.generations} == {"prompt 1", "prompt 2"}
    assert result.result_uri.endswith("/generations.jsonl")
    assert result.usage.total_tokens == 30


def test_batch_generate_uploads_messages_and_custom_ids(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "a", "p2": "b"})

    batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        system_prompt="be terse",
        max_tokens=64,
        completion_window="12h",
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert [row["custom_id"] for row in api.uploaded_rows] == ["p1", "p2"]
    messages = json.loads(api.uploaded_rows[0]["messages"])
    assert messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "prompt 1"},
    ]
    source = api.operation_payload["src"][0]
    assert source == {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "mapping": {
            "type": "text_messages",
            "messages": {"type": "column", "name": "messages"},
            "custom_id": {"type": "column", "name": "custom_id"},
            "max_tokens": {"type": "number", "value": 64},
        },
    }
    assert api.operation_payload["params"] == {
        "model": DEFAULT_BATCH_MODEL,
        "completion_window": "12h",
    }


def test_batch_generate_cleans_up_scratch_datasets(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "a", "p2": "b"})

    batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert set(api.deleted) == {DATASET_ID, RESULT_DATASET_ID}


def test_batch_generate_keep_datasets_leaves_them(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "a", "p2": "b"})

    batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        keep_datasets=True,
        client=_client(api),
    )

    assert api.deleted == []


def test_batch_generate_no_wait_returns_operation_handle(tmp_path: Path) -> None:
    api = FakeBatchApi()

    result = batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        wait=False,
        client=_client(api),
    )

    assert result.status == "pending"
    assert result.operation_id == OPERATION_ID
    assert result.generation_count == 0
    # A pending run must not delete the source dataset: the operation reads it
    # server-side after this process exits.
    assert api.deleted == []
    assert api.poll_count == 0


def test_batch_generate_reports_the_error_file_reason_and_a_model_hint(tmp_path: Path) -> None:
    # The live failure shape: the operations endpoint says nothing useful, and the
    # batch record's error file carries the real per-row reason.
    api = FakeBatchApi(
        statuses=["failed"],
        errors=[""],
        in_progress=False,
        error_file_text=ROUTING_KEY_ERROR,
    )

    with pytest.raises(TokenFactoryToolError) as excinfo:
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            client=_client(api),
        )

    message = str(excinfo.value)
    assert "not a known batch endpoint routing key" in message
    assert "2/2 rows rejected" in message
    assert "not available for batch inference" in message
    assert DEFAULT_BATCH_MODEL in message
    assert ERROR_FILE_ID in api.downloaded_files


def test_batch_generate_without_a_batch_view_still_reports_the_failure(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["failed"], errors=[""], serve_batch_view=False)

    with pytest.raises(TokenFactoryToolError) as excinfo:
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            client=_client(api),
        )

    assert "reported no error detail" in str(excinfo.value)


def test_batch_generate_prefers_the_batch_output_file_over_a_dataset_export(
    tmp_path: Path,
) -> None:
    api = FakeBatchApi(completions={"p1": "from the output file", "p2": "b"})

    result = batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert result.generations[0].completion == "from the output file"
    assert OUTPUT_FILE_ID in api.downloaded_files
    # The result dataset is never exported when the standard output file is there.
    assert RESULT_DATASET_ID not in api.exported_datasets


def test_batch_generate_falls_back_to_the_dataset_export(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "from the export", "p2": "b"}, serve_output_file=False)

    result = batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert result.generations[0].completion == "from the export"
    assert RESULT_DATASET_ID in api.exported_datasets


def test_batch_generate_reports_request_counts(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "a", "p2": "b"})

    result = batch_generate(
        input_path=str(_prompts(tmp_path)),
        output_path=str(tmp_path / "out"),
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert result.request_counts["total"] == 2
    assert result.request_counts["completed"] == 2
    assert result.request_counts["invalid"] == 0


def test_batch_generate_cleans_up_after_a_failed_operation(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["failed"], errors=["quota exceeded"], in_progress=False)

    with pytest.raises(TokenFactoryToolError):
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            client=_client(api),
        )

    assert set(api.deleted) == {DATASET_ID, RESULT_DATASET_ID}


def test_batch_generate_timeout_keeps_datasets_for_the_running_operation(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["running"])

    with pytest.raises(TokenFactoryToolError):
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            timeout_s=0.0,
            client=_client(api),
        )

    # The operation is still running server-side and still reads its source rows.
    assert api.deleted == []


def test_batch_generate_surfaces_reported_error_detail(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["failed"], errors=["quota exceeded"], in_progress=False)

    with pytest.raises(TokenFactoryToolError, match="quota exceeded"):
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            client=_client(api),
        )


def test_batch_generate_times_out_without_losing_operation_id(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["running"])

    with pytest.raises(TokenFactoryToolError) as excinfo:
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            poll_interval_s=0.01,
            timeout_s=0.0,
            client=_client(api),
        )

    assert OPERATION_ID in str(excinfo.value)
    assert "keeps running" in str(excinfo.value)


def test_batch_generate_hints_when_the_model_is_not_text_to_text(tmp_path: Path) -> None:
    api = FakeBatchApi()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/operations"):
            # Verbatim from the live API when handed a vision model.
            return httpx.Response(
                400, json={"detail": "Batch inference is only supported for text2text models"}
            )
        return api.handler(request)

    config = resolve_config(api_key="test-key", environ={})
    client = TokenFactoryClient(
        config,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(TokenFactoryToolError) as excinfo:
        batch_generate(
            input_path=str(_prompts(tmp_path)),
            output_path=str(tmp_path / "out"),
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            client=client,
        )

    message = str(excinfo.value)
    assert "is not a text model" in message
    assert "token-factory caption" in message
    # The scratch dataset must not survive a rejected submit.
    assert api.deleted == [DATASET_ID]


def test_batch_generate_rejects_empty_prompt_file(tmp_path: Path) -> None:
    empty = tmp_path / "prompts.jsonl"
    empty.write_text("\n", encoding="utf-8")

    with pytest.raises(TokenFactoryToolError, match="No prompts found"):
        batch_generate(
            input_path=str(empty),
            output_path=str(tmp_path / "out"),
            client=_client(FakeBatchApi()),
        )


def test_batch_generate_respects_max_prompts(tmp_path: Path) -> None:
    api = FakeBatchApi(completions={"p1": "a"})

    result = batch_generate(
        input_path=str(_prompts(tmp_path, count=5)),
        output_path=str(tmp_path / "out"),
        max_prompts=1,
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert len(api.uploaded_rows) == 1
    assert result.prompt_count == 1


@pytest.mark.parametrize("row_key", ["response", "body", "result"])
def test_batch_collect_parses_each_result_wrapper(tmp_path: Path, row_key: str) -> None:
    api = FakeBatchApi(completions={"p1": "collected"}, result_row_key=row_key)

    result = batch_collect(
        operation_id=OPERATION_ID,
        output_path=str(tmp_path / "out"),
        wait=True,
        poll_interval_s=0.01,
        client=_client(api),
    )

    assert result.status == "completed"
    assert [item.completion for item in result.generations] == ["collected"]


def test_batch_collect_reports_pending_without_blocking(tmp_path: Path) -> None:
    api = FakeBatchApi(statuses=["running"])

    result = batch_collect(
        operation_id=OPERATION_ID,
        output_path=str(tmp_path / "out"),
        wait=False,
        client=_client(api),
    )

    assert result.status == "pending"
    assert result.operation_status == "running"
    assert api.deleted == []


def test_parse_batch_export_reads_the_standard_batch_row() -> None:
    # The documented OpenAI-compatible batch output row, wrapper fields included.
    row = {
        "id": "batch_req_0001",
        "custom_id": "p1",
        "response": {
            "status_code": 200,
            "request_id": "req-0001",
            "body": {
                "id": "chatcmpl-0001",
                "object": "chat.completion",
                "model": DEFAULT_BATCH_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "a standard answer"},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            },
        },
        "error": None,
    }

    generations, failures, usage = _parse_batch_export(
        json.dumps(row) + "\n", {"p1": "the prompt"}
    )

    assert failures == []
    assert [(g.id, g.prompt, g.completion) for g in generations] == [
        ("p1", "the prompt", "a standard answer")
    ]
    assert usage.total_tokens == 12


def test_parse_batch_export_records_a_per_row_error_with_its_message() -> None:
    row = {
        "custom_id": "p1",
        "response": {
            "status_code": 400,
            "body": {"error": {"message": "context length exceeded", "type": "invalid_request"}},
        },
        "error": None,
    }

    generations, failures, _usage = _parse_batch_export(json.dumps(row) + "\n", {})

    assert generations == []
    assert failures[0]["id"] == "p1"
    assert "context length exceeded" in failures[0]["error"]


def test_batch_operation_uri_sits_beside_generations() -> None:
    assert batch_operation_uri_for("s3://bucket/run/out/") == (
        "s3://bucket/run/out/batch_operation.json"
    )
    assert batch_operation_uri_for("s3://bucket/run/out/generations.jsonl") == (
        "s3://bucket/run/out/batch_operation.json"
    )
