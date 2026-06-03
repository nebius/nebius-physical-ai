"""Compatibility imports for Cosmos workbench SDK functions."""

from __future__ import annotations

from npa.workbench.cosmos import (
    Cosmos3AccessConfig,
    Cosmos3AccessError,
    Cosmos3CheckResult,
    Cosmos3FetchResult,
    Cosmos3ServeConfig,
    Cosmos3SkillEnv,
    Cosmos3SkillSpec,
    build_cosmos3_inference_args,
    build_cosmos3_skill_env,
    check,
    check_cosmos3_access,
    fetch,
    fetch_cosmos3_artifacts,
    get_cosmos3_skill,
    list_cosmos3_skills,
    skill,
    skills,
)

__all__ = [
    "Cosmos3AccessConfig",
    "Cosmos3AccessError",
    "Cosmos3CheckResult",
    "Cosmos3FetchResult",
    "Cosmos3ServeConfig",
    "Cosmos3SkillEnv",
    "Cosmos3SkillSpec",
    "build_cosmos3_inference_args",
    "build_cosmos3_skill_env",
    "check",
    "check_cosmos3_access",
    "fetch",
    "fetch_cosmos3_artifacts",
    "get_cosmos3_skill",
    "list_cosmos3_skills",
    "skill",
    "skills",
]
