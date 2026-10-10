"""Persist local people, hashed access keys, and explicitly linked external identities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager

from .errors import AuthenticationError, ConflictError, TeamError
from .models import Actor

KEY_PREFIX = "npa_wb_"


class Accounts:
    """Keep stable user identities separate from changeable login credentials.

    Args:
        config: Installation with an immutable account namespace and private state.
    Returns:
        Local account store; no external identity service is required.
    Raises:
        TeamError, OSError, sqlite3.Error: Account storage cannot be opened safely.
    """

    def __init__(self, config):
        """Open or initialize the installation's private account database.

        Args:
            config: Validated local-account installation.
        Returns:
            None.
        Raises:
            TeamError, OSError, sqlite3.Error: Storage or ownership domain is invalid.
        """
        if config.account_namespace is None:
            raise TeamError("local accounts are not configured")
        self.issuer = config.principal_issuer
        config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        config.state_dir.chmod(0o700)
        self.path = config.state_dir / "accounts.sqlite3"
        descriptor = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        with self._transaction() as db:
            db.executescript(_SCHEMA)
            db.execute(
                "INSERT OR IGNORE INTO installation VALUES (1, ?)", (self.issuer,)
            )
            if (
                db.execute("SELECT issuer FROM installation").fetchone()[0]
                != self.issuer
            ):
                raise ConflictError(
                    "account namespace changed; restore the original value"
                )

    def create(self, name: str, groups=()):
        """Create a permanent user without accepting a caller-selected identity.

        Args:
            name: Unique lowercase account name used only for display and administration.
            groups: Administrator-managed group names.
        Returns:
            Public user metadata containing its generated stable ID.
        Raises:
            TeamError, ConflictError: Name or groups are invalid or already registered.
        """
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
            raise TeamError(
                "account name must be a lowercase name of at most 48 characters"
            )
        membership = _groups(groups)
        user_id = str(uuid.uuid4())
        try:
            with self._transaction() as db:
                db.execute(
                    "INSERT INTO users VALUES (?, ?, ?, 0)", (user_id, name, membership)
                )
                self._audit(db, "user-create", user_id)
        except sqlite3.IntegrityError as exc:
            raise ConflictError("account name is already registered") from exc
        return {
            "id": user_id,
            "name": name,
            "groups": json.loads(membership),
            "disabled": False,
        }

    def list(self):
        """List account metadata for the local server operator, without credentials.

        Args:
            None.
        Returns:
            Account IDs, names, groups, disabled state and credential metadata.
        Raises:
            sqlite3.Error: Durable account state cannot be read.
        """
        with self._transaction() as db:
            users = [
                dict(row) for row in db.execute("SELECT * FROM users ORDER BY name")
            ]
            for user in users:
                user["groups"] = json.loads(user["groups"])
                user["disabled"] = bool(user["disabled"])
                user["keys"] = [
                    dict(row)
                    for row in db.execute(
                        "SELECT id, revoked, created_at FROM access_keys WHERE user_id=?",
                        (user["id"],),
                    )
                ]
        return users

    def update(self, user_id, *, disabled=None, groups=None):
        """Change local membership or disable all authentication for an exact user.

        Args:
            user_id: Immutable account ID.
            disabled, groups: Optional account state and replacement membership.
        Returns:
            None.
        Raises:
            TeamError: User is absent or group names are invalid.
        """
        with self._transaction() as db:
            self._user(db, user_id, active=False)
            if disabled is not None:
                db.execute(
                    "UPDATE users SET disabled=? WHERE id=?", (int(disabled), user_id)
                )
            if groups is not None:
                db.execute(
                    "UPDATE users SET groups=? WHERE id=?", (_groups(groups), user_id)
                )
            self._audit(db, "user-update", user_id)

    def issue_key(self, user_id):
        """Create a random 256-bit key and retain only its SHA-256 digest.

        Args:
            user_id: Enabled account receiving a new independently revocable key.
        Returns:
            Key ID and secret; the caller must deliver the secret privately once.
        Raises:
            AuthenticationError: Account is missing or disabled.
        """
        secret = KEY_PREFIX + secrets.token_urlsafe(32)
        key_id = str(uuid.uuid4())
        with self._transaction() as db:
            self._user(db, user_id)
            db.execute(
                "INSERT INTO access_keys (id,user_id,digest) VALUES (?,?,?)",
                (key_id, user_id, _digest(secret)),
            )
            self._audit(db, "key-create", user_id, key_id)
        return key_id, secret

    def revoke(self, key_id):
        """Revoke an exact credential before it can authorize another request.

        Args:
            key_id: Credential ID from issuance or the operator account list.
        Returns:
            None; repeated revocation is safe.
        Raises:
            TeamError: Credential ID is unknown.
        """
        with self._transaction() as db:
            row = db.execute(
                "SELECT user_id FROM access_keys WHERE id=?", (key_id,)
            ).fetchone()
            if row is None:
                raise TeamError("access key does not exist")
            db.execute("UPDATE access_keys SET revoked=1 WHERE id=?", (key_id,))
            self._audit(db, "key-revoke", row[0], key_id)

    def authenticate(self, secret):
        """Authenticate one key without storing its plaintext or trusting client identity.

        Args:
            secret: Untrusted bearer credential.
        Returns:
            Current local actor and revocation handle.
        Raises:
            AuthenticationError: Key or user is invalid or revoked.
        """
        if not isinstance(secret, str) or not re.fullmatch(
            r"npa_wb_[A-Za-z0-9_-]{43}", secret
        ):
            raise AuthenticationError("access key could not be verified")
        with self._transaction() as db:
            row = db.execute(
                "SELECT id,user_id FROM access_keys WHERE digest=? AND revoked=0",
                (_digest(secret),),
            ).fetchone()
            if row is None:
                raise AuthenticationError("access key could not be verified")
            return self._actor(self._user(db, row["user_id"])), row["id"]

    def refresh(self, actor):
        """Recheck account disablement and local groups during admitted workflows.

        Args:
            actor: Previously authenticated internal identity.
        Returns:
            Same immutable identity with current group membership.
        Raises:
            AuthenticationError: Account is disabled, absent, or from another installation.
        """
        if actor.issuer != self.issuer:
            raise AuthenticationError("account could not be verified")
        with self._transaction() as db:
            return self._actor(self._user(db, actor.subject))

    def link(self, user_id, issuer, subject):
        """Bind an external issuer and immutable subject to an existing local account.

        Args:
            user_id: Exact account authorized by the local operator.
            issuer, subject: Provider-verified identity; never an email-based match.
        Returns:
            None.
        Raises:
            ConflictError, AuthenticationError: Identity is already linked or user is invalid.
        """
        if not issuer or not subject:
            raise TeamError("issuer and immutable external subject are required")
        try:
            with self._transaction() as db:
                db.execute("BEGIN IMMEDIATE")
                self._user(db, user_id)
                if db.execute(
                    "SELECT 1 FROM external_identities WHERE user_id=?", (user_id,)
                ).fetchone():
                    raise ConflictError(
                        "local account already has an external identity link"
                    )
                db.execute(
                    "INSERT INTO external_identities VALUES (?,?,?)",
                    (issuer, subject, user_id),
                )
                self._audit(db, "identity-link", user_id)
        except sqlite3.IntegrityError as exc:
            raise ConflictError("external identity is already linked") from exc

    def unlink(self, user_id, issuer, subject):
        """Remove only the selected external link without changing the local account.

        Args:
            user_id: Exact account selected by the local operator, including disabled users.
            issuer, subject: Exact existing external identity, including a retired provider.
        Returns:
            Whether a matching link was removed; repeated removal is a no-op.
        Raises:
            TeamError, AuthenticationError: Identity is incomplete or the account is absent.
        """
        if not issuer or not subject:
            raise TeamError("issuer and immutable external subject are required")
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            self._user(db, user_id, active=False)
            changed = db.execute(
                "DELETE FROM external_identities WHERE user_id=? AND issuer=? AND subject=?",
                (user_id, issuer, subject),
            ).rowcount
            if changed:
                self._audit(db, "identity-unlink", user_id)
        return bool(changed)

    def resolve(self, external):
        """Map a verified external identity to an explicitly linked local person.

        Args:
            external: Identity accepted by the configured JWT verifier.
        Returns:
            Stable local actor with administrator-managed local groups.
        Raises:
            AuthenticationError: There is no explicit link or the account is disabled.
        """
        with self._transaction() as db:
            row = db.execute(
                "SELECT user_id FROM external_identities WHERE issuer=? AND subject=?",
                (external.issuer, external.subject),
            ).fetchone()
            if row is None:
                raise AuthenticationError(
                    "external identity has no Workbench account link"
                )
            return self._actor(self._user(db, row[0]))

    def _actor(self, user):
        return Actor(
            issuer=self.issuer,
            subject=user["id"],
            groups=json.loads(user["groups"]),
            display_name=user["name"],
        )

    def _user(self, db, user_id, *, active=True):
        user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if user is None or (active and user["disabled"]):
            raise AuthenticationError("account could not be verified")
        return user

    def _audit(self, db, action, user_id, key_id=None):
        db.execute(
            "INSERT INTO account_audit (action,user_id,key_id,operator_uid) VALUES (?,?,?,?)",
            (action, user_id, key_id, os.getuid()),
        )

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()


def _groups(groups):
    if any(
        not isinstance(group, str)
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", group)
        for group in groups
    ):
        raise TeamError(
            "group names must be nonempty identifiers of at most 128 characters"
        )
    return json.dumps(sorted(set(groups)))


def _digest(secret):
    return hashlib.sha256(secret.encode()).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS installation (id INTEGER PRIMARY KEY CHECK(id=1), issuer TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, groups TEXT NOT NULL, disabled INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS access_keys (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), digest TEXT NOT NULL UNIQUE, revoked INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS external_identities (issuer TEXT NOT NULL, subject TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id), PRIMARY KEY(issuer,subject));
CREATE TABLE IF NOT EXISTS account_audit (id INTEGER PRIMARY KEY, action TEXT NOT NULL, user_id TEXT NOT NULL, key_id TEXT, operator_uid INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""
