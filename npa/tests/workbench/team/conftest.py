"""Provide synthetic identities and isolated configuration for team acceptance tests."""

import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from npa.workbench.team.authentication import TokenVerifier
from npa.workbench.team.authorization import bind_execution
from npa.workbench.team.models import Actor, TeamConfig


@pytest.fixture
def config(tmp_path):
    allocations = []
    for subject in ("alice", "bob"):
        credential = tmp_path / f"{subject}.json"
        credential.write_text(
            json.dumps(
                {
                    "aws_access_key_id": f"test-{subject}",
                    "aws_secret_access_key": f"secret-{subject}",
                }
            )
        )
        credential.chmod(0o600)
        allocations.append(
            {
                "subject": subject,
                "gpu_limit": 2,
                "clusters": {"east": 1, "west": 1},
                "storage": {
                    "endpoint": "https://objects.example.test",
                    "bucket": f"team-test-{subject}",
                    "prefix": "personal",
                    "principal": f"principal-{subject}",
                    "credentials_file": credential,
                },
            }
        )
    return TeamConfig.model_validate(
        {
            "identity": {
                "issuer": "https://identity.example.test",
                "audience": "workbench",
                "jwks_url": "https://identity.example.test/jwks",
            },
            "state_dir": tmp_path / "state",
            "sky_endpoint": "http://127.0.0.1:46580",
            "sky_python": tmp_path / "sky-python",
            "clusters": {
                name: {
                    "context": name,
                    "kubeconfig": tmp_path / f"{name}.yaml",
                }
                for name in ("east", "west")
            },
            "workspaces": {
                "robotics": {
                    "grants": [{"kind": "group", "value": "researchers"}],
                    "gpu_limit": 4,
                    "gpu_limits": {"east": 2, "west": 2},
                    "allocations": allocations,
                }
            },
        }
    )


@pytest.fixture
def actor(config):
    return Actor(issuer=config.identity.issuer, subject="alice", groups={"researchers"})


@pytest.fixture
def binding(config, actor):
    return bind_execution(config, actor, "robotics", "east")


@pytest.fixture
def workflow():
    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "team-smoke"},
        "config": {},
        "resources": {
            "cpu": {"cloud": "kubernetes", "cpus": 1, "image": "ubuntu:24.04"}
        },
        "initial": "hello",
        "states": {
            "hello": {
                "run": {"shell": "echo hello"},
                "resources": "cpu",
                "terminal": True,
            }
        },
    }


@pytest.fixture
def tokens(config):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keys = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=private.public_key())
    )
    verifier = TokenVerifier(config.identity, jwks_client=keys)

    def sign(subject="alice", **overrides):
        claims = {
            "iss": config.identity.issuer,
            "aud": config.identity.audience,
            "sub": subject,
            "groups": ["researchers"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
            **overrides,
        }
        return jwt.encode(
            claims, private, algorithm="RS256", headers={"kid": "test-key"}
        )

    return SimpleNamespace(sign=sign, verifier=verifier)
