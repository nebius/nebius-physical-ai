"""Every opt-in E2E gate must have a declared runner or a reviewed manual reason."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
E2E = ROOT / "npa" / "tests" / "e2e"
RUNNER_FILES = (
    ROOT / "scripts" / "dev-vm-daily-tests.sh",
    ROOT / ".github" / "workflows" / "dev-vm-daily-tests.yml",
)

# These specialized suites intentionally remain operator-invoked. The reason is
# machine-reviewed here instead of letting an environment gate silently rot.
MANUAL_GATES = {
    "NPA_BURST_E2E_IMAGE": "operator supplies the exact immutable burst validation image",
    "NPA_E2E_BURST": "full burst GPU coverage is an explicitly selected live suite",
    "NPA_BYOF_LIVE_CONTAINER": "BYOF executes third-party source only after operator review",
    "NPA_BYOF_LIVE_GPU": "BYOF GPU mutation requires a reviewed onboarding target",
    "NPA_BYOF_OD_VERIFY_RUN": "Open Dreamer verification requires an explicitly selected run",
    "NPA_BYOF_OPEN_DREAMER_LIVE_GPU": "Open Dreamer GPU mutation remains an operator acceptance test",
    "NPA_BYOF_OPENPI_LIVE_B200": "OpenPI B200 validation requires live GPU and registry access",
    "NPA_OPENPI_ANTIOCH_LIVE": (
        "Antioch validation requires an operator project and reachable policy endpoint"
    ),
    "NPA_BYOF_WAN22_LIVE_GPU": "Wan single-GPU BYOF mutation requires an explicitly selected validation run",
    "NPA_BYOF_WAN22_MULTIGPU_LIVE_GPU": "Wan multi-GPU BYOF mutation requires an explicitly selected validation run",
    "NPA_BYOF_LIVE_UBUNTU": "BYOF Ubuntu mutation is a dedicated onboarding acceptance",
    # Not merely operator-selected: an automated runner *must not* reach this
    # suite. It needs a token entitled to the gated Lightricks/LTX-2.5
    # repository, which Lightricks grants only after a human accepts its terms
    # there — so a runner that supplied one would be running under an acceptance
    # nobody in this repository made.
    "NPA_LTX2_LIVE_GPU": "LTX-2.5 requires the operator's own gated-repository token",
    "NPA_E2E_BYOVM_SELF_HEAL": "targets an explicitly selected existing BYOVM service",
    "NPA_CONFIGURE_E2E": "creates storage configuration and needs exact project selectors",
    "NPA_E2E_S3_ACCESS_KEY_ID": "runtime prerequisite, not an authorization gate",
    "NPA_E2E_S3_SECRET_ACCESS_KEY": "runtime prerequisite, not an authorization gate",
    "NPA_E2E_SERVERLESS_PROJECT": "serverless suite is reachable through e2e-serverless",
    "NPA_E2E_FORCE_NER": "optional fallback knob inside the reachable serverless tier",
    "NPA_E2E_MK8S_GPU_HEALTH": "fresh reserved mk8s GPU validation requires explicit operator authorization",
    "NPA_E2E_MK8S_FRESH_CLUSTER": "prevents accidentally targeting an existing or production-adjacent cluster",
    "NPA_E2E_MK8S_RESERVED_CAPACITY": "prevents silent fallback from reviewed reserved GPU capacity",
    "NPA_TEST_GROOT_NGC_E2E": "gated NGC model access remains a product-specific manual test",
    "NPA_E2E_CLEAR_WORKBENCH_IMAGES": "optional negative-path knob, not a suite gate",
    "NPA_SRC_S3_URI": "runtime source-staging prerequisite, not an authorization gate",
    "NPA_PREEMPTIBLE_E2E": "destructive preemptible VM suite remains operator-selected",
    "NPA_DRY_RUN": "test-mode selector used by specialized workflow durability suites",
    "NPA_FLEET_MIG_RUN_WORKLOAD_MATRIX": (
        "destructive all-slice MIG qualification requires an operator-selected cluster"
    ),
}


def _skip_gates(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    gates: set[str] = set()
    for node in ast.walk(tree):
        relevant = False
        expression: ast.AST | None = None
        if isinstance(node, ast.If):
            relevant = any(
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "skip"
                for statement in node.body
                for item in ast.walk(statement)
            )
            expression = node.test
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skipif"
        ):
            relevant = True
            expression = node
        if not relevant or expression is None:
            continue
        gates.update(
            str(item.value)
            for item in ast.walk(expression)
            if isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value.startswith("NPA_")
        )
    return gates


def test_every_e2e_environment_gate_is_reachable_or_explicitly_manual() -> None:
    declared = set().union(*(_skip_gates(path) for path in E2E.glob("test_*.py")))
    runner_text = "\n".join(path.read_text(encoding="utf-8") for path in RUNNER_FILES)
    unreachable = sorted(
        gate
        for gate in declared
        if gate not in runner_text and gate not in MANUAL_GATES
    )
    stale_allowlist = sorted(set(MANUAL_GATES) - declared)
    assert not unreachable, (
        f"E2E gates have no runner mapping or manual reason: {unreachable}"
    )
    assert not stale_allowlist, f"manual E2E gate reasons are stale: {stale_allowlist}"
    assert all(len(reason.split()) >= 5 for reason in MANUAL_GATES.values())


def test_pr218_mutation_gates_are_runner_reachable_not_manual() -> None:
    runner_text = "\n".join(path.read_text(encoding="utf-8") for path in RUNNER_FILES)
    for gate in (
        "NPA_PR218_LIVE_LIFECYCLE",
        "NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION",
    ):
        assert gate in runner_text
        assert gate not in MANUAL_GATES
