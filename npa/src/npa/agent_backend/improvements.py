"""Durable, scoped improvement work packages and evidence-backed lesson reuse.

The service detects and coordinates work; it never runs a shell, changes source,
or publishes code. HTTP input is not validation evidence. A local coordinator
writes protected receipts from completed checks and independently supplied review
reports. Reviewer identity is coordinator-attested, not authenticated by the
agent's shared HTTP login.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
import stat
from typing import Any, Iterator, Mapping, Sequence

try:
    from agent_backend.trajectory import _redact_identifiers, redact
except ImportError:  # installed SDK / repository tests
    from npa.agent_backend.trajectory import _redact_identifiers, redact


class ImprovementError(ValueError):
    """Invalid scope, stale ownership, or unverified evidence."""


DETECTOR_VERSION = "1"
LESSONS = {
    "trajectory_observation_conservation": {
        "targets": ("agent-run-data-collection", "trajectory"),
        "instruction": (
            "Preserve action phase, status and args in trajectory events. Verify "
            "serialized bytes against the synthetic confidentiality and secret "
            "cases before publication; a delivery receipt only proves delivery."
        ),
    },
    "inspect_failed_tool_evidence": {
        "targets": (),  # the item's configured component is the only target
        "instruction": (
            "Inspect the recorded failure and current authoritative status before "
            "retrying this tool. Reuse the verified reproducer and validation "
            "evidence attached to the improvement work package."
        ),
    },
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", value):
        raise ImprovementError("expected a simple nonempty handle")
    if redact(value) != value:
        raise ImprovementError("handle contains private identifiers")
    return value


def _no_symlink(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ImprovementError("symlink paths are not supported")


def _private_dir(path: Path) -> Path:
    path = path.absolute()
    _no_symlink(path)
    if path.exists() and not path.is_dir():
        raise ImprovementError("storage path must be a directory")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = path.stat()
    if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
        raise ImprovementError("storage directory must be owner-only")
    return path


def _read_private(path: Path) -> bytes:
    _no_symlink(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ImprovementError("evidence must be a regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        mode = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(mode.st_mode)
            or mode.st_uid != os.getuid()
            or stat.S_IMODE(mode.st_mode) & 0o077
            or mode.st_nlink != 1
        ):
            raise ImprovementError("evidence must be an owner-only regular file")
        content = stream.read()
    if not content:
        raise ImprovementError("evidence must be nonempty")
    return content


def _write_private(path: Path, content: bytes) -> None:
    _no_symlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str):
        raise ImprovementError("scope files must be relative paths")
    path = PurePosixPath(value)
    if (
        not value or path.is_absolute() or ".." in path.parts
        or str(path) != value or any(char in value for char in "*?[]\\\n\r")
    ):
        raise ImprovementError("scope requires exact normalized relative file paths")
    target = root / value
    _no_symlink(target)
    if not target.resolve().is_relative_to(root):
        raise ImprovementError("scope escapes repository")
    if target.exists() and not target.is_file():
        raise ImprovementError("scope entries must be files")
    return target


@dataclass(frozen=True)
class ImprovementScope:
    """Coordinator configuration, never inferred from tool output."""

    scope_id: str
    component: str
    files: tuple[str, ...]
    base_revision: str
    required_checks: tuple[str, ...]
    lesson_keys: tuple[str, ...] = ("inspect_failed_tool_evidence",)
    version: str = "1"


def find_improvements(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Detect explicit action/drive failures; successful recovery stays successful.

    Empty terminal discovery and confirmation/refusal are not defects. Findings
    are triage candidates, not assertions that source code is broken.
    """
    findings = []
    steps = result.get("steps") or result.get("iterations") or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        component = step.get("tool") or ("sim2real-drive" if "iterations" in result else "action-loop")
        if "iterations" in result and (step.get("error") or step.get("adjust_error")):
            kind = "drive_adjust_error" if step.get("adjust_error") else "drive_error"
        elif ("iterations" in result and isinstance(step.get("diagnosis"), dict)
              and step["diagnosis"].get("error")):
            kind = "drive_diagnosis_error"
        elif step.get("status") == "error":
            recovered = any(
                isinstance(later, dict)
                and later.get("tool") == step.get("tool")
                and (later.get("status") == "ok" or later.get("status") == "success")
                for later in steps[index + 1:]
            )
            if recovered and result.get("ok"):
                continue
            kind = "tool_error"
        elif step.get("status") == "empty" and not step.get("terminal_observation"):
            kind = "empty_tool_result"
        elif isinstance(step.get("observation"), dict) and step["observation"].get("truncated"):
            kind = "truncated_observation"
        else:
            continue
        findings.append({"kind": kind, "component": component, "event_index": index, "evidence": step})
    if result.get("stopped_reason") == "max_steps":
        findings.append({
            "kind": "max_steps_exhausted", "component": "action-loop",
            "event_index": len(steps), "evidence": {"stopped_reason": "max_steps"},
        })
    return findings


class ImprovementStore:
    """One coordinator's local queue; SQLite serializes independent processes."""

    def __init__(
        self, directory: Path, *, repository: Path, evidence_directory: Path,
        scopes: Sequence[ImprovementScope], reviewers: Sequence[str],
        private_literals: Sequence[str] = (),
    ) -> None:
        self.repository = repository.absolute()
        _no_symlink(self.repository)
        self.repository = self.repository.resolve()
        if not self.repository.is_dir():
            raise ImprovementError("repository root is missing")
        _no_symlink(directory.absolute())
        _no_symlink(evidence_directory.absolute())
        directory, evidence_directory = directory.resolve(), evidence_directory.resolve()
        if directory.is_relative_to(self.repository) or evidence_directory.is_relative_to(self.repository):
            raise ImprovementError("queue and evidence must live outside the source repository")
        self.directory = _private_dir(directory)
        self.evidence_directory = _private_dir(evidence_directory)
        self.private_literals = tuple(sorted(set(private_literals), key=len, reverse=True))
        if any(not isinstance(value, str) or not value for value in self.private_literals):
            raise ImprovementError("private literals must be nonempty strings")
        self.reviewers = frozenset(_handle(value) for value in reviewers)
        self.scopes: dict[str, ImprovementScope] = {}
        for scope in scopes:
            for handle in (scope.scope_id, scope.component, scope.version):
                _handle(handle)
            if not re.fullmatch(r"[0-9a-f]{40,64}", scope.base_revision):
                raise ImprovementError("scope must bind an exact base revision")
            if not scope.files or len(set(scope.files)) != len(scope.files):
                raise ImprovementError("scope must contain distinct exact files")
            for filename in scope.files:
                _relative_file(self.repository, filename)
            if not scope.required_checks or len(set(scope.required_checks)) != len(scope.required_checks):
                raise ImprovementError("scope must define distinct required checks")
            for check in scope.required_checks:
                _handle(check)
            if not set(scope.lesson_keys) <= LESSONS.keys():
                raise ImprovementError("unknown structured lesson")
            if scope.component in self.scopes:
                raise ImprovementError("component scope is ambiguous")
            if self._safe(asdict(scope)) != json.loads(_json(asdict(scope))):
                raise ImprovementError("scope configuration contains private text")
            self.scopes[scope.component] = scope
        self.path = self.directory / "improvements.sqlite3"
        _no_symlink(self.path)
        try:
            _write_private(self.path, b"")
        except FileExistsError:
            mode = self.path.stat()
            if (not stat.S_ISREG(mode.st_mode) or mode.st_uid != os.getuid()
                    or stat.S_IMODE(mode.st_mode) & 0o077 or mode.st_nlink != 1):
                raise ImprovementError("database must be owner-only") from None
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS occurrences (
                    id TEXT PRIMARY KEY, item_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, data TEXT NOT NULL);
            """)

    def _safe(self, value: Any) -> Any:
        # Use the trajectory's collision-safe, idempotent literal pass for the
        # queue too. Reports and receipts cross this boundary more than once;
        # replacing text inside generated markers corrupts their retained proof.
        return _redact_identifiers(
            redact(value), {literal: "<private-ref>" for literal in self.private_literals}
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        _no_symlink(self.path)
        parent = self.directory.stat()
        before = self.path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o077 or parent.st_uid != os.getuid()
                or stat.S_IMODE(parent.st_mode) & 0o077):
            raise ImprovementError("database and parent must remain private regular storage")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True)
        try:
            after = self.path.lstat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ImprovementError("database was replaced during open")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _load(self, db: sqlite3.Connection, item_id: str) -> dict[str, Any]:
        row = db.execute("SELECT data FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise ImprovementError("unknown improvement item")
        item = json.loads(row[0])
        scope = self.scopes.get(item["component"])
        if scope is None or item["scope"] != json.loads(_json(asdict(scope))):
            raise ImprovementError("scope configuration changed; explicit reconciliation required")
        return item

    def _save(self, db: sqlite3.Connection, item: dict, event: str, details: Any = None) -> None:
        item["version"] += 1
        # All variable data crosses the sanitizer before SQLite sees it.
        db.execute("INSERT OR REPLACE INTO items VALUES (?, ?)", (item["id"], _json(self._safe(item))))
        db.execute("INSERT INTO events(item_id, data) VALUES (?, ?)", (
            item["id"], _json(self._safe({"event": event, "at": _now(), "version": item["version"], "details": details})),
        ))

    @staticmethod
    def _public(item: dict) -> dict:
        return {key: value for key, value in item.items() if key != "claim_digest"}

    def observe(self, *, component: str, kind: str, episode_id: str, event_index: int, evidence: Any, session_id: str = "") -> dict:
        scope = self.scopes.get(component)
        if scope is None:
            raise ImprovementError("component has no coordinator-approved scope")
        kind = _handle(kind)
        if not episode_id or not isinstance(event_index, int) or event_index < 0:
            raise ImprovementError("observation needs an episode and event index")
        safe_evidence = self._safe(evidence)
        if not safe_evidence:
            raise ImprovementError("observation evidence is empty")
        scope_data = json.loads(_json(asdict(scope)))
        item_id = _digest({"scope": scope_data, "kind": kind, "detector": DETECTOR_VERSION})
        occurrence = {
            "episode_ref": _digest(episode_id), "event_index": event_index,
            "session_ref": _digest(session_id) if session_id else "",
            "evidence_sha256": _digest(safe_evidence), "evidence": safe_evidence,
        }
        occurrence_id = _digest({"item_id": item_id, **occurrence})
        with self._transaction() as db:
            row = db.execute("SELECT data FROM items WHERE id = ?", (item_id,)).fetchone()
            item = json.loads(row[0]) if row else {
                "id": item_id, "component": component, "kind": kind, "scope": scope_data,
                "state": "observed", "version": 0, "generation": 0, "owner": "",
                "claim_digest": "", "occurrences": 0, "first_observed": _now(),
                "validations": {}, "lesson_key": "", "review": None,
            }
            inserted = db.execute("INSERT OR IGNORE INTO occurrences VALUES (?, ?, ?)", (
                occurrence_id, item_id, _json(self._safe(occurrence)),
            )).rowcount
            if inserted:
                recurrence = item["state"] == "verified"
                if recurrence:
                    item.update(state="observed", lesson_key="", review=None, validations={})
                item["occurrences"] += 1
                item["last_observed"] = _now()
                self._save(db, item, "recurrence" if recurrence else "observed", {"occurrence_id": occurrence_id})
            return self._public(item)

    def observe_action(self, result: Mapping[str, Any], *, episode_id: str, session_id: str = "") -> list[dict]:
        return [self.observe(episode_id=episode_id, session_id=session_id, **finding) for finding in find_improvements(result)
                if finding["component"] in self.scopes]

    def list_items(self) -> list[dict]:
        with self._transaction() as db:
            return [self._public(json.loads(row[0])) for row in db.execute("SELECT data FROM items ORDER BY id")]

    def history(self, item_id: str) -> dict:
        with self._transaction() as db:
            item = self._load(db, item_id)
            return {
                "item": self._public(item),
                "occurrences": [json.loads(row[0]) for row in db.execute(
                    "SELECT data FROM occurrences WHERE item_id = ? ORDER BY id", (item_id,))],
                "events": [json.loads(row[0]) for row in db.execute(
                    "SELECT data FROM events WHERE item_id = ? ORDER BY sequence", (item_id,))],
            }

    def claim(self, item_id: str, *, owner: str, version: int) -> dict:
        owner = _handle(owner)
        if self._safe(owner) != owner:
            raise ImprovementError("owner handle contains private text")
        with self._transaction() as db:
            item = self._load(db, item_id)
            if item["version"] != version or item["state"] != "observed":
                raise ImprovementError("item is stale or not claimable")
            for row in db.execute("SELECT data FROM items"):
                other = json.loads(row[0])
                overlap = any(left == right or left.startswith(right + "/") or right.startswith(left + "/")
                              for left in other["scope"]["files"] for right in item["scope"]["files"])
                if other["owner"] and overlap:
                    raise ImprovementError("another active claim owns overlapping files")
            claim_token = secrets.token_urlsafe(32)
            item.update(owner=owner, implementation_owner=owner, state="claimed", generation=item["generation"] + 1,
                        claim_digest=_digest(claim_token), validations={}, review=None)
            self._save(db, item, "claimed", {"owner": owner, "generation": item["generation"]})
            observations = [json.loads(row[0]) for row in db.execute(
                "SELECT data FROM occurrences WHERE item_id = ? ORDER BY id", (item_id,))]
            return {**self._public(item), "claim_token": claim_token, "observations": observations}

    def _owned(self, db: sqlite3.Connection, item_id: str, owner: str, generation: int, claim_token: str) -> dict:
        item = self._load(db, item_id)
        if (item["owner"] != owner or item["generation"] != generation
                or not claim_token or not secrets.compare_digest(item["claim_digest"], _digest(claim_token))):
            raise ImprovementError("stale or invalid claim fence")
        return item

    def release(self, item_id: str, *, owner: str, generation: int, claim_token: str) -> dict:
        """Coordinator calls after joining the worker; no implicit timed expiry."""
        with self._transaction() as db:
            item = self._owned(db, item_id, owner, generation, claim_token)
            item.update(owner="", claim_digest="", state="observed", validations={}, review=None, lesson_key="")
            self._save(db, item, "released")
            return self._public(item)

    def _snapshot(self, scope: Mapping[str, Any]) -> dict:
        files = []
        for name in scope["files"]:
            target = _relative_file(self.repository, name)
            digest = None
            if target.exists():
                fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    mode = os.fstat(stream.fileno())
                    if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
                        raise ImprovementError("candidate source must be a regular unlinked file")
                    hasher = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                    digest = hasher.hexdigest()
            files.append({"path": name, "sha256": digest})
        return {"base_revision": scope["base_revision"], "files": files}

    def begin_candidate(self, item_id: str, *, changed_files: Sequence[str], **ownership: Any) -> dict:
        """Trusted coordinator captures this BEFORE starting its prescribed checks.

        changed_files is the coordinator's complete candidate diff, isolated from
        unrelated parallel work. Source bytes are verified again on completion.
        """
        with self._transaction() as db:
            item = self._owned(db, item_id, **ownership)
            changed = sorted(set(changed_files))
            if not changed or not set(changed) <= set(item["scope"]["files"]):
                raise ImprovementError("candidate changes escape the owned scope")
            snapshot = self._snapshot(item["scope"])
            return {"item_id": item_id, "generation": item["generation"], "owner": item["owner"],
                    "scope_sha256": _digest(item["scope"]), "changed_files": changed,
                    "snapshot": snapshot, "candidate_sha256": _digest(snapshot)}

    def write_validation_receipt(self, candidate: dict, *, check: str, completed: Any, report: bytes) -> str:
        """Local adapter for an actual CompletedProcess, never exposed over HTTP.

        The caller owns execution provenance. No arbitrary command is executed by
        this module, and only sanitized report bytes are persisted.
        """
        from subprocess import CompletedProcess

        if not isinstance(completed, CompletedProcess) or not isinstance(completed.returncode, int):
            raise ImprovementError("validation requires a completed process result")
        with self._transaction() as db:
            item = self._load(db, candidate["item_id"])
            if check not in item["scope"]["required_checks"]:
                raise ImprovementError("check is not coordinator-prescribed")
            self._verify_candidate(item, candidate)
            return self._write_receipt({
                "kind": "validation", "candidate": candidate, "check": check,
                "exit_code": completed.returncode,
            }, report)

    def _write_receipt(self, receipt: dict, report: bytes) -> str:
        if not isinstance(report, bytes) or not report.strip():
            raise ImprovementError("completed check needs a nonempty report")
        safe_report = self._safe(report.decode("utf-8")).encode()
        report_id = secrets.token_hex(32)
        _write_private(self.evidence_directory / (report_id + ".txt"), safe_report)
        receipt = self._safe({**receipt, "report_ref": report_id, "report_sha256": hashlib.sha256(safe_report).hexdigest()})
        receipt_id = secrets.token_hex(32)
        _write_private(self.evidence_directory / (receipt_id + ".json"), _json(receipt).encode())
        return receipt_id

    def _read_receipt(self, reference: str, kind: str) -> dict:
        if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-f]{64}", reference):
            raise ImprovementError("invalid evidence reference")
        receipt = json.loads(_read_private(self.evidence_directory / (reference + ".json")))
        if receipt.get("kind") != kind or self._safe(receipt) != receipt:
            raise ImprovementError("evidence receipt kind or privacy check failed")
        report_ref = receipt.get("report_ref", "")
        if not re.fullmatch(r"[0-9a-f]{64}", report_ref):
            raise ImprovementError("invalid report reference")
        report = _read_private(self.evidence_directory / (report_ref + ".txt"))
        if hashlib.sha256(report).hexdigest() != receipt.get("report_sha256"):
            raise ImprovementError("report content changed")
        if self._safe(report.decode()) != report.decode():
            raise ImprovementError("report privacy check failed")
        return receipt

    def _verify_candidate(self, item: dict, candidate: dict) -> None:
        if (candidate.get("item_id") != item["id"] or candidate.get("generation") != item["generation"]
                or candidate.get("owner") != item.get("implementation_owner", item["owner"]) or candidate.get("scope_sha256") != _digest(item["scope"])
                or not candidate.get("changed_files")
                or not set(candidate["changed_files"]) <= set(item["scope"]["files"])
                or candidate.get("snapshot") != self._snapshot(item["scope"])
                or candidate.get("candidate_sha256") != _digest(candidate["snapshot"])):
            raise ImprovementError("candidate is stale or outside its claimed scope")

    def record_validation(self, item_id: str, *, evidence_ref: str, **ownership: Any) -> dict:
        with self._transaction() as db:
            item = self._owned(db, item_id, **ownership)
            if item["state"] not in {"claimed", "validation_failed", "ready_for_review"}:
                raise ImprovementError("item cannot accept validation")
            receipt = self._read_receipt(evidence_ref, "validation")
            self._verify_candidate(item, receipt["candidate"])
            check = receipt["check"]
            if check not in item["scope"]["required_checks"]:
                raise ImprovementError("unexpected validation check")
            if item.get("candidate_sha256") != receipt["candidate"]["candidate_sha256"]:
                item["validations"] = {}
            item["candidate_sha256"] = receipt["candidate"]["candidate_sha256"]
            item["validations"][_digest(check)] = {"evidence_ref": evidence_ref, "sha256": _digest(receipt), "exit_code": receipt["exit_code"]}
            failures = any(value["exit_code"] != 0 for value in item["validations"].values())
            complete = set(item["validations"]) == {_digest(check) for check in item["scope"]["required_checks"]}
            item["state"] = "validation_failed" if failures else "ready_for_review" if complete else "claimed"
            self._save(db, item, "validation", {"check": check, "evidence_sha256": _digest(receipt), "exit_code": receipt["exit_code"]})
            return self._public(item)

    def write_review_receipt(self, item_id: str, *, reviewer: str, lesson_key: str, report: bytes, accepted: bool) -> str:
        """Trusted local adapter for independently obtained review evidence.

        The coordinator must establish the reviewer outside the shared HTTP
        identity. Merely receiving a different name in a POST is insufficient.
        """
        with self._transaction() as db:
            item = self._load(db, item_id)
            if reviewer not in self.reviewers or reviewer == item["owner"]:
                raise ImprovementError("independent configured reviewer required")
            if item["state"] != "ready_for_review" or lesson_key not in item["scope"]["lesson_keys"]:
                raise ImprovementError("review requires complete validation and a configured lesson")
            self._verify_validations(item)
            return self._write_receipt({
                "kind": "review", "item_id": item_id, "generation": item["generation"],
                "candidate_sha256": item["candidate_sha256"], "validations_sha256": _digest(item["validations"]),
                "reviewer": reviewer, "accepted": accepted is True, "lesson_key": lesson_key,
                "identity_provenance": "coordinator-attested-external-review",
            }, report)

    def _verify_validations(self, item: dict) -> None:
        if set(item["validations"]) != {_digest(check) for check in item["scope"]["required_checks"]}:
            raise ImprovementError("required validation is incomplete")
        for check, recorded in item["validations"].items():
            receipt = self._read_receipt(recorded["evidence_ref"], "validation")
            if (_digest(receipt) != recorded["sha256"] or receipt["exit_code"] != 0
                    or _digest(receipt["check"]) != check or receipt["candidate"]["candidate_sha256"] != item["candidate_sha256"]):
                raise ImprovementError("required validation failed or changed")
            self._verify_candidate(item, receipt["candidate"])

    def review(self, item_id: str, *, evidence_ref: str) -> dict:
        with self._transaction() as db:
            item = self._load(db, item_id)
            if item["state"] != "ready_for_review":
                raise ImprovementError("item is not ready for independent review")
            self._verify_validations(item)
            review = self._read_receipt(evidence_ref, "review")
            if (review.get("item_id") != item_id or review.get("generation") != item["generation"]
                    or review.get("reviewer") not in self.reviewers or review["reviewer"] == item["owner"]
                    or review.get("candidate_sha256") != item["candidate_sha256"]
                    or review.get("validations_sha256") != _digest(item["validations"])
                    or review.get("lesson_key") not in item["scope"]["lesson_keys"]
                    or review.get("identity_provenance") != "coordinator-attested-external-review"):
                raise ImprovementError("independent review does not bind this validated candidate")
            item["review"] = {"evidence_ref": evidence_ref, "sha256": _digest(review), "reviewer": review["reviewer"],
                              "identity_provenance": review["identity_provenance"]}
            if review.get("accepted") is True:
                item.update(state="verified", lesson_key=review["lesson_key"], owner="", claim_digest="")
            else:
                item.update(state="validation_failed", lesson_key="")
            self._save(db, item, "review", {"accepted": review["accepted"], "review_sha256": _digest(review)})
            return self._public(item)

    def _matching_lessons(self, db: sqlite3.Connection, targets: Sequence[str]) -> list[dict]:
        lessons = []
        for row in db.execute("SELECT id FROM items ORDER BY id").fetchall():
            item = self._load(db, row[0])
            key = item["lesson_key"]
            if item["state"] != "verified" or key not in LESSONS:
                continue
            lesson = LESSONS[key]
            applicable = set(lesson["targets"]) if lesson["targets"] else {item["component"]}
            if not applicable.intersection(targets):
                continue
            # Revoked reviewer, missing evidence or changed source deactivates
            # reuse even after restart; a cached approval is not sufficient.
            try:
                self._verify_validations(item)
                review = self._read_receipt(item["review"]["evidence_ref"], "review")
                if (_digest(review) != item["review"]["sha256"] or review["reviewer"] not in self.reviewers
                        or review["reviewer"] == item["implementation_owner"]):
                    continue
            except (ImprovementError, OSError):
                continue
            lessons.append({"item_id": item["id"], "lesson_key": key, "version": item["scope"]["version"],
                            "instruction": lesson["instruction"], "review_sha256": item["review"]["sha256"]})
        return lessons

    def matching_verified_lessons(self, targets: Sequence[str]) -> list[dict]:
        with self._transaction() as db:
            return self._matching_lessons(db, targets)

    def consume_lessons(self, targets: Sequence[str], *, request_id: str) -> list[dict]:
        with self._transaction() as db:
            lessons = self._matching_lessons(db, targets)
            for lesson in lessons:
                item = self._load(db, lesson["item_id"])
                if item["state"] != "verified":
                    continue
                self._save(db, item, "lesson_used", {"request_ref": _digest(request_id), "lesson_key": lesson["lesson_key"]})
        return lessons

    def record_lesson_outcome(self, lessons: Sequence[dict], *, request_id: str, outcome: str) -> None:
        if outcome not in {"succeeded", "failed", "confirmation", "unknown"}:
            raise ImprovementError("unknown lesson outcome")
        with self._transaction() as db:
            for lesson in lessons:
                item = self._load(db, lesson["item_id"])
                self._save(db, item, "lesson_outcome", {"request_ref": _digest(request_id), "outcome": outcome,
                                                       "lesson_key": lesson["lesson_key"]})


def lesson_context(lessons: Sequence[dict]) -> str:
    """Render only fixed known instructions; no evidence prose enters a prompt."""
    lines = []
    for lesson in lessons:
        key = lesson.get("lesson_key")
        if key in LESSONS and re.fullmatch(r"[0-9a-f]{64}", lesson.get("item_id", "")):
            lines.append(f"[verified-lesson:{key}; item:{lesson['item_id']}] {LESSONS[key]['instruction']}")
    return "\n".join(lines)


def store_from_config(path: Path) -> ImprovementStore:
    """Load explicit owner-only runtime configuration; no ambient repo discovery."""
    config = json.loads(_read_private(path))
    scopes = [ImprovementScope(**{**entry, "files": tuple(entry["files"]),
                                  "required_checks": tuple(entry["required_checks"]),
                                  "lesson_keys": tuple(entry.get("lesson_keys", ("inspect_failed_tool_evidence",)))})
              for entry in config["scopes"]]
    literals = list(config.get("private_literals", []))
    # Dataset literals apply here too, before any queue persistence. Include the
    # isolated prefix so path fragments cannot escape URI-pattern redaction.
    dataset_uri = os.environ.get("NPA_AGENT_DATASET_URI", "")
    if dataset_uri:
        from urllib.parse import urlsplit
        parsed = urlsplit(dataset_uri)
        literals.extend(value for value in (dataset_uri, parsed.netloc, parsed.path.strip("/")) if value)
    denylist = os.environ.get("NPA_AGENT_DATASET_REDACTION_FILE", "")
    if denylist:
        literals.extend(json.loads(_read_private(Path(denylist))).get("literals", []))
    return ImprovementStore(Path(config["directory"]), repository=Path(config["repository"]),
                            evidence_directory=Path(config["evidence_directory"]), scopes=scopes,
                            reviewers=config.get("reviewers", []), private_literals=literals)
