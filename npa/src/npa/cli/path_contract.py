"""Shared CLI path contract for workbench tool handoffs."""

from __future__ import annotations

import re
from urllib.parse import urlparse


S3_CONTRACT_MESSAGE = (
    "Workbench tool paths use the S3 handoff contract: tools write to s3:// URIs "
    "and read from s3:// URIs, with Hugging Face Hub datasets accepted for read-only "
    "dataset inputs. VM-local paths, local filesystem paths, file:// URIs, and "
    "plain http:// URLs are not supported in the public CLI."
)
FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR = (
    "FiftyOne load-dataset expects an S3 URI or a Hugging Face Hub dataset. "
    "VM-local paths are not supported. If you generated this with cosmos infer, "
    "pass the same s3:// URI you used for --output-path."
)

_HF_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{3,95}/"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
    r"(?::[A-Za-z0-9][A-Za-z0-9._/-]{0,255})?$"
)
_LOCALISH_FIRST_SEGMENTS = {
    "abs",
    "data",
    "dataset",
    "datasets",
    "file",
    "home",
    "input",
    "inputs",
    "local",
    "model",
    "models",
    "mnt",
    "opt",
    "output",
    "outputs",
    "path",
    "rel",
    "relative",
    "run",
    "runs",
    "tmp",
    "var",
}


class PathContractError(ValueError):
    """Raised when a public CLI path violates the workbench handoff contract."""


def is_s3_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "s3" and bool(parsed.netloc) and bool(parsed.path.lstrip("/"))


def is_huggingface_dataset_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return (
        len(parts) >= 3
        and parts[0] == "datasets"
        and _is_hub_repo_parts(parts[1], parts[2])
    )


def is_huggingface_identifier(value: str) -> bool:
    if not _HF_REPO_ID_RE.match(value):
        return False
    namespace, rest = value.split("/", 1)
    repo = rest.split(":", 1)[0]
    return _is_hub_repo_parts(namespace, repo)


def validate_read_path(
    value: str,
    *,
    tool: str,
    option: str = "--input-path",
    allow_hf: bool = True,
    required: bool = True,
    vm_local_message: str | None = None,
) -> str:
    path = value.strip()
    if not path:
        if required:
            raise PathContractError(f"{tool} {option} must not be empty.")
        return path
    if is_s3_uri(path):
        return path
    if allow_hf and (
        is_huggingface_identifier(path) or is_huggingface_dataset_url(path)
    ):
        return path
    raise PathContractError(
        _error_message(
            tool,
            option,
            path,
            read=True,
            allow_hf=allow_hf,
            vm_local_message=vm_local_message,
        )
    )


def validate_write_path(
    value: str,
    *,
    tool: str,
    option: str = "--output-path",
    required: bool = False,
) -> str:
    path = value.strip()
    if not path:
        if required:
            raise PathContractError(f"{tool} {option} must not be empty.")
        return path
    if is_s3_uri(path):
        return path
    raise PathContractError(
        _error_message(tool, option, path, read=False, allow_hf=False)
    )


def _is_hub_repo_parts(namespace: str, repo: str) -> bool:
    if namespace.lower() in _LOCALISH_FIRST_SEGMENTS:
        return False
    if namespace.endswith(".") or repo.endswith("."):
        return False
    if "--" in namespace or "--" in repo:
        return False
    if ".." in namespace or ".." in repo:
        return False
    return True


def _error_message(
    tool: str,
    option: str,
    path: str,
    *,
    read: bool,
    allow_hf: bool,
    vm_local_message: str | None = None,
) -> str:
    if _is_vm_local_path(path) and vm_local_message:
        return vm_local_message
    expected = "an S3 URI"
    if read and allow_hf:
        expected += " or a Hugging Face Hub dataset"
    return f"{tool} {option} expects {expected}. {S3_CONTRACT_MESSAGE}"


def _is_vm_local_path(path: str) -> bool:
    return path.startswith("/") or path.startswith("~/")
