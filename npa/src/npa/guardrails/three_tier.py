"""Three-tier CLI, SDK, and SkyPilot YAML coherence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
from pathlib import Path

from npa.guardrails.skypilot import env_names_for_yaml, env_refs_for_yaml


@dataclass(frozen=True)
class ParameterContract:
    """A single parameter expected to exist across CLI, SDK, and YAML."""

    cli_param: str
    sdk_param: str
    yaml_env: str
    cli_flag: str


@dataclass(frozen=True)
class CapabilityContract:
    """A workbench capability contract across all three access tiers."""

    name: str
    cli_module: str
    cli_callback: str
    sdk_module: str
    sdk_attr: str
    yaml_path: Path
    params: tuple[ParameterContract, ...]


def callback_parameters(module_name: str, callback_name: str) -> dict[str, inspect.Parameter]:
    module = import_module(module_name)
    callback = getattr(module, callback_name)
    return dict(inspect.signature(callback).parameters)


def option_flags(param: inspect.Parameter) -> set[str]:
    default = param.default
    flags: set[str] = set()
    for decl in getattr(default, "param_decls", ()):
        for part in str(decl).split("/"):
            if part.startswith("--"):
                flags.add(part)
    return flags


def sdk_parameters(module_name: str, attr_name: str) -> set[str]:
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    wrapped_module = getattr(attr, "__npa_cli_module__", "")
    wrapped_callback = getattr(attr, "__npa_cli_callback__", "")
    if wrapped_module and wrapped_callback:
        return set(callback_parameters(wrapped_module, wrapped_callback))
    return set(inspect.signature(attr).parameters)


def validate_contract(contract: CapabilityContract, *, repo_root: Path) -> list[str]:
    """Return validation failures for a capability contract."""

    failures: list[str] = []
    cli_params = callback_parameters(contract.cli_module, contract.cli_callback)
    sdk_params = sdk_parameters(contract.sdk_module, contract.sdk_attr)
    yaml_path = repo_root / contract.yaml_path
    yaml_envs = env_names_for_yaml(yaml_path)
    yaml_refs = env_refs_for_yaml(yaml_path)

    for param in contract.params:
        cli = cli_params.get(param.cli_param)
        if cli is None:
            failures.append(f"{contract.name}: CLI param missing: {param.cli_param}")
        elif param.cli_flag not in option_flags(cli):
            failures.append(
                f"{contract.name}: CLI flag {param.cli_flag} missing for {param.cli_param}"
            )
        if param.sdk_param not in sdk_params:
            failures.append(f"{contract.name}: SDK param missing: {param.sdk_param}")
        if param.yaml_env not in yaml_envs:
            failures.append(f"{contract.name}: YAML env missing: {param.yaml_env}")
        elif param.yaml_env not in yaml_refs:
            failures.append(f"{contract.name}: YAML env not referenced: {param.yaml_env}")
    return failures


def registered_workbench_tools() -> set[str]:
    """Return registered `npa workbench` tool names."""

    from npa.cli.workbench import app

    return {str(group.name) for group in app.registered_groups if group.name}
