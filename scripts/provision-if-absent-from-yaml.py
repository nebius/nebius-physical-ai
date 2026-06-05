#!/usr/bin/env python3
"""Run provision-if-absent from a standalone YAML settings file."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="infra/bootstrap/provision-if-absent.yaml")
    parser.add_argument("target", nargs="?", default="all", choices=("all", "s3", "k8s"))
    args = parser.parse_args()

    config_path = Path(args.config)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{config_path} must contain a YAML mapping")

    storage = _mapping(data.get("storage"))
    provision = _mapping(data.get("provision"))
    env = os.environ.copy()
    env.update(
        {
            "NPA_PROJECT_ID": _string(data.get("project_id")),
            "NPA_TENANT_ID": _string(data.get("tenant_id")),
            "NPA_REGION": _string(data.get("region") or "eu-north1"),
            "NPA_REGISTRY_ID": _string(data.get("registry_id")),
            "NPA_REGISTRY": _string(data.get("registry")),
            "NPA_S3_BUCKET": _string(storage.get("s3_bucket")),
            "NPA_STORAGE_ENDPOINT": _string(storage.get("s3_endpoint")),
            "AWS_ENDPOINT_URL": _string(storage.get("s3_endpoint")),
            "NEBIUS_S3_ENDPOINT": _string(storage.get("s3_endpoint")),
            "AWS_ACCESS_KEY_ID": _string(storage.get("aws_access_key_id")),
            "AWS_SECRET_ACCESS_KEY": _string(storage.get("aws_secret_access_key")),
            "TERRAFORM_DIR": _string(provision.get("terraform_dir") or "deploy/cluster"),
            "CLUSTER_CONTEXT": _string(provision.get("cluster_context") or "npa-cluster"),
            "KUBECONFIG_PATH": _string(provision.get("kubeconfig_path")),
        }
    )
    script = Path(_string(provision.get("script") or "scripts/provision-if-absent.sh"))
    return subprocess.run([str(script), args.target], env=env, check=False).returncode


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
