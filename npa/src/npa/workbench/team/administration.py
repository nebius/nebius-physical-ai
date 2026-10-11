"""Provide an explicit local operator cancellation path after a user is offboarded."""

import json
import os

from .deployment import allocations
from .errors import ConflictError
from .ledger import TeamLedger
from .models import Actor
from .service import binding_snapshot, public_run
from .sky_backend import SkyBackend


def stop_run(config, run_id, *, backend_factory=SkyBackend):
    """Cancel recorded jobs under the authority of the local configuration owner.

    Args:
        config: Private server configuration accessible only to its operator.
        run_id: Exact durable team run ID; no user token or name-based deletion.
        backend_factory: Injectable scheduler adapter for acceptance tests.
    Returns:
        Run status; cancelling remains nonterminal until provider termination is proven.
    Raises:
        ConflictError, BackendError: Allocation changed or provider state is uncertain.
    """
    ledger = TeamLedger(config.state_dir)
    record = ledger.get(run_id)
    if record["status"] in ("succeeded", "failed", "cancelled"):
        return public_run(record)
    matches = [
        binding
        for binding in allocations(config)
        if (binding.workspace, binding.cluster, binding.allocation.subject)
        == (record["workspace"], record["cluster"], record["subject"])
    ]
    if len(matches) != 1 or binding_snapshot(matches[0]) != json.loads(
        record["binding"]
    ):
        raise ConflictError(
            "restore the recorded allocation before operator cancellation"
        )
    ledger.transition(
        run_id, ("accepted", "running", "recovery_required"), "cancelling"
    )
    operator = Actor(issuer="local-operator", subject=f"uid:{os.getuid()}")
    ledger.audit(operator, "operator-cancel", run_id)
    if backend_factory(config, matches[0], ledger, run_id).cancel_run():
        ledger.transition(run_id, ("cancelling",), "cancelled")
    return public_run(ledger.get(run_id))
