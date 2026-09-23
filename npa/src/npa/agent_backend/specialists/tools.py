"""Expose scoped source editing and operator-defined Workbench commands to models."""

from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from npa.agent_backend.trajectory import redact

from .call_policy import _classification


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Read(_Arguments):
    path: str


class _ReadFile(_Read):
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class _Edit(_Arguments):
    path: str
    expected_sha256: str
    old: str
    new: str


class _Operation(_Arguments):
    name: str


_ARGUMENTS = {
    "list_files": _Read,
    "read_file": _ReadFile,
    "edit_file": _Edit,
    "run_operation": _Operation,
}
_DESCRIPTIONS = {
    "list_files": "List source files under an authorized directory, using a workspace-relative path.",
    "read_file": "Read an authorized source file or inclusive 1-based line range, with the full-file SHA-256 for editing. Paths are workspace-relative.",
    "edit_file": "Replace exactly one occurrence after verifying SHA-256. For a new file use empty old and the SHA-256 of empty bytes.",
    "run_operation": "Execute one named operator-defined test, workflow or status command. Never supply shell commands or arguments.",
}


class WorkbenchTools:
    """Apply one specialist's source and operation grants with durable receipts.

    Args: profile: Configured specialist. store: Task journal. task_id: Durable task identity.
    Returns: Scoped tool executor.
    Raises: ValueError: An operation or path violates configured policy.
    """

    def __init__(self, profile, store, task_id):
        self.profile, self.store, self.task_id = profile, store, task_id
        self.root = profile.workspace.resolve(strict=True)

    def schemas(self):
        """Return model tool schemas without commands, secrets or host paths.

        Args: None.
        Returns: OpenAI-compatible function schemas.
        Raises: None.
        """
        names = ["list_files", "read_file"]
        if self.profile.write_paths:
            names.append("edit_file")
        if self.profile.operations:
            names.append("run_operation")
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _DESCRIPTIONS[name],
                    "parameters": _ARGUMENTS[name].model_json_schema(),
                },
            }
            for name in names
        ]

    def execute(self, call):
        """Execute or reuse a tool result while refusing uncertain replay.

        Args: call: Persisted provider tool call with id, function name and JSON arguments.
        Returns: Sanitized tool receipt, including real command exit status.
        Raises: UncertainOperation, OSError: The operation needs reconciliation.
        """
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        invocation = {"name": name, "arguments": arguments}
        previous = self.store._begin_call(
            self.task_id,
            call["id"],
            invocation,
            classification=_classification(self.profile, invocation),
        )
        if previous is not None:
            return previous
        try:
            if name not in _ARGUMENTS:
                raise ValueError("unknown tool")
            parsed = _ARGUMENTS[name].model_validate_json(arguments)
            result = getattr(self, "_" + name)(parsed)
        except ValueError as error:
            result = {"ok": False, "error": redact(str(error))}
        result = redact(result)
        self.store._finish_call(self.task_id, call["id"], result)
        self.store._event(
            self.task_id,
            {"type": "tool", "call_id": call["id"], "name": name, "result": result},
        )
        return result

    def _path(self, name, *, write=False, directory=False):
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or ".git" in relative.parts
        ):
            raise ValueError("path must be relative and outside .git")
        scopes = (
            self.profile.write_paths
            if write
            else [*self.profile.read_paths, *self.profile.write_paths]
        )
        if not any(
            relative == PurePosixPath(scope) or PurePosixPath(scope) in relative.parents
            for scope in scopes
        ):
            raise ValueError("path is outside the specialist's configured scopes")
        path = self.root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("symlink paths are not allowed")
        if directory and path.is_dir():
            return path
        if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
            raise ValueError("path must be a singly linked regular file")
        return path

    def _read_file(self, arguments):
        path = self._path(arguments.path)
        if not path.exists():
            return {"ok": False, "error": "file does not exist", "sha256": _digest("")}
        content = path.read_text()
        lines = content.splitlines(keepends=True)
        start, end = arguments.start_line, arguments.end_line
        if end is not None and end < start:
            raise ValueError("end_line must be at least start_line")
        if start > max(len(lines), 1):
            raise ValueError("start_line exceeds the file's line count")
        end = min(end if end is not None else len(lines), len(lines))
        return {
            "ok": True,
            "path": arguments.path,
            "sha256": _digest(content),
            "content": "".join(lines[start - 1 : end]),
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
        }

    def _list_files(self, arguments):
        path = self._path(arguments.path, directory=True)
        if not path.is_dir():
            raise ValueError("list_files requires an existing authorized directory")
        paths = []
        for current, directories, files in os.walk(path, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name != ".git" and not (Path(current) / name).is_symlink()
            ]
            for name in files:
                item = Path(current) / name
                if not item.is_symlink():
                    paths.append(item.relative_to(self.root).as_posix())
        return {"ok": True, "paths": sorted(paths)}

    def _edit_file(self, arguments):
        path = self._path(arguments.path, write=True)
        original = path.read_text() if path.exists() else None
        current = original or ""
        if _digest(current) != arguments.expected_sha256:
            raise ValueError("file changed; read it again before editing")
        if original is not None and (
            not arguments.old or current.count(arguments.old) != 1
        ):
            raise ValueError("old text must match exactly once in an existing file")
        if original is None and arguments.old:
            raise ValueError("new files require empty old text")
        updated = (
            current.replace(arguments.old, arguments.new, 1)
            if original is not None
            else arguments.new
        )
        self.store._original(self.task_id, arguments.path, original)
        _atomic_text(path, updated)
        self._save_patch()
        return {
            "ok": True,
            "path": arguments.path,
            "sha256": _digest(updated),
            "patch": "changes.diff",
        }

    def _run_operation(self, arguments):
        operation = self.profile.operations.get(arguments.name)
        if operation is None:
            raise ValueError("operation is not configured for this specialist")
        substitutions = {
            "python": sys.executable,
            "workspace": str(self.root),
            "task_id": self.task_id,
            "run_id": "specialist-" + self.task_id,
        }
        argv = [value.format_map(substitutions) for value in operation.argv]
        names = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", *operation.pass_env}
        environment = {name: os.environ[name] for name in names if name in os.environ}
        result = subprocess.run(
            argv,
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "operation": arguments.name,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "run_id": substitutions["run_id"],
        }

    def _save_patch(self):
        patch = []
        for original in self.store._originals(self.task_id):
            path = self._path(original["path"], write=True)
            before = original["content"]
            after = path.read_text()
            patch.extend(
                difflib.unified_diff(
                    (before or "").splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile="a/" + original["path"]
                    if before is not None
                    else "/dev/null",
                    tofile="b/" + original["path"],
                )
            )
        directory = self.store.directory / "artifacts" / self.task_id
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        text = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in patch
        )
        _atomic_text(directory / "changes.diff", text)


def _digest(content):
    return hashlib.sha256(content.encode()).hexdigest()


def _atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".specialist-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
