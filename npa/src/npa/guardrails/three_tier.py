"""Three-tier CLI, SDK, and workflow-surface coherence helpers.

A workbench capability is reachable three ways, and the three must not drift:

1. **CLI** — ``npa workbench <tool> <verb> --flag`` (Typer callback signature).
2. **SDK** — ``npa.sdk.workbench.<tool>.<attr>`` (Python signature).
3. **Workflow** — an ``npa.workflow/v0.0.1`` spec that invokes the capability by
   ``toolRef``, whose argv template comes from
   :data:`npa.orchestration.npa_workflow.catalog.TOOL_CATALOG`.

Tier 3 used to be a raw SkyPilot task YAML, checked by "the YAML declares an
``envs`` key named after each CLI flag, and references it in ``setup``/``run``".
That was a proxy for "the shipped way to run this tool at scale exposes its
parameters". As the raw SkyPilot catalog is retired, tier 3 moves onto the surface
that survives: the spec plus its toolRef argv.

The migrated check is sharper in one way and narrower in another, and both are
deliberate:

* **Sharper** — it verifies the parameter is passed *by the flag name the CLI
  actually accepts*. The old env check could not: a YAML could declare
  ``TRAIN_LEARNING_RATE`` and the tool could still be invoked with a nonexistent
  ``--learning-rate`` (which is precisely the drift ``DESIGN.md`` §7 recorded).
* **Narrower** — a catalog argv template exposes fewer knobs than a SkyPilot
  YAML's full ``envs`` block did. Rather than hide that, every contract pins its
  :attr:`CapabilityContract.spec_gap`: the CLI parameters that are *not* reachable
  from a spec today. The test asserts the computed gap equals the declared gap
  exactly, so the gap is visible in review, cannot grow silently, and shrinks only
  by a deliberate edit.

A capability whose npa.workflow twin does not exist yet may still declare
``yaml_path`` and per-parameter ``yaml_env`` values to keep the legacy check until
its spec lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
from pathlib import Path

from npa.guardrails.skypilot import env_names_for_yaml, env_refs_for_yaml
from npa.guardrails.tool_catalog_argv import argv_template_flags


@dataclass(frozen=True)
class ParameterContract:
    """A single parameter expected to exist across every access tier."""

    cli_param: str
    sdk_param: str
    cli_flag: str
    #: Only used by contracts still on the legacy SkyPilot-YAML third tier.
    yaml_env: str = ""


@dataclass(frozen=True)
class CapabilityContract:
    """A workbench capability contract across all three access tiers."""

    name: str
    cli_module: str
    cli_callback: str
    sdk_module: str
    sdk_attr: str
    params: tuple[ParameterContract, ...]
    #: Third tier (preferred): the npa.workflow spec that invokes the capability.
    spec_path: Path | None = None
    #: The ``toolRef`` the spec must declare, and whose argv carries the flags.
    tool_ref: str = ""
    #: ``cli_param`` names that a spec cannot set today, because the toolRef argv
    #: template does not pass their flag. Pinned; may only shrink.
    spec_gap: tuple[str, ...] = ()
    #: Third tier (legacy): a raw SkyPilot task YAML, for capabilities whose
    #: npa.workflow twin does not exist yet.
    yaml_path: Path | None = None


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

    failures.extend(_validate_workflow_tier(contract, repo_root=repo_root))
    return failures


def _validate_workflow_tier(contract: CapabilityContract, *, repo_root: Path) -> list[str]:
    if contract.spec_path is not None:
        return _validate_spec_tier(contract, repo_root=repo_root)
    if contract.yaml_path is not None:
        return _validate_legacy_yaml_tier(contract, repo_root=repo_root)
    return [
        f"{contract.name}: contract declares no third tier; set spec_path + tool_ref "
        "(preferred) or yaml_path"
    ]


def _validate_spec_tier(contract: CapabilityContract, *, repo_root: Path) -> list[str]:
    """Check the npa.workflow tier: spec loads, uses the toolRef, argv carries flags."""

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.spec import load_spec

    failures: list[str] = []
    if not contract.tool_ref:
        return [f"{contract.name}: spec_path set without tool_ref"]

    entry = TOOL_CATALOG.get(contract.tool_ref)
    if entry is None:
        return [f"{contract.name}: unknown toolRef {contract.tool_ref!r}"]

    spec_path = repo_root / contract.spec_path if contract.spec_path else None
    assert spec_path is not None  # narrowed by the caller
    try:
        # load_spec also resolves every {{config.*}} token in the toolRef argv, so a
        # spec that declares the toolRef but forgets a config key fails right here.
        spec = load_spec(spec_path)
    except NpaWorkflowError as exc:
        return [f"{contract.name}: spec {contract.spec_path} does not load: {exc}"]

    declared = {state.tool_ref for state in spec.states.values() if state.tool_ref}
    if contract.tool_ref not in declared:
        failures.append(
            f"{contract.name}: spec {contract.spec_path} does not invoke "
            f"{contract.tool_ref!r} (declares {sorted(declared)})"
        )

    argv_flags = set(argv_template_flags(entry.argv_template))
    gap = tuple(
        param.cli_param for param in contract.params if param.cli_flag not in argv_flags
    )
    if gap != contract.spec_gap:
        failures.append(
            f"{contract.name}: spec_gap drifted. The toolRef {contract.tool_ref!r} "
            f"argv reaches every parameter except {list(gap)}, but the contract "
            f"pins {list(contract.spec_gap)}. Either wire the missing flag into the "
            "argv template (preferred) or update spec_gap deliberately."
        )
    return failures


def _validate_legacy_yaml_tier(contract: CapabilityContract, *, repo_root: Path) -> list[str]:
    failures: list[str] = []
    assert contract.yaml_path is not None  # narrowed by the caller
    yaml_path = repo_root / contract.yaml_path
    yaml_envs = env_names_for_yaml(yaml_path)
    yaml_refs = env_refs_for_yaml(yaml_path)
    for param in contract.params:
        if not param.yaml_env:
            failures.append(
                f"{contract.name}: legacy YAML tier needs yaml_env for {param.cli_param}"
            )
            continue
        if param.yaml_env not in yaml_envs:
            failures.append(f"{contract.name}: YAML env missing: {param.yaml_env}")
        elif param.yaml_env not in yaml_refs:
            failures.append(f"{contract.name}: YAML env not referenced: {param.yaml_env}")
    return failures


def registered_workbench_tools() -> set[str]:
    """Return registered `npa workbench` tool names."""

    from npa.cli.workbench import app

    return {str(group.name) for group in app.registered_groups if group.name}
