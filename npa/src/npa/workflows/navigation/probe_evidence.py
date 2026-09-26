"""Retain measured isolation traces and identify exact repeatability differences."""

import numpy as np


def save_trace(output, name, trace):
    """Preserve all robot states and focal observations before probe gates.

    Args:
        output: Native stage artifact directory.
        name: Internal probe control name.
        trace: Validated simulator snapshots and observation arrays.
    Returns:
        None.
    Raises:
        OSError: The measured trace cannot be written.
    """
    arrays = {
        f"{step}/{group}/{key}": value if group == "state" else value[:1]
        for step, row in enumerate(trace)
        for group, fields in row.items()
        for key, value in fields.items()
    }
    np.savez_compressed(output / f"probe-{name}.npz", **arrays)


def load_trace(path, population, steps):
    """Decode complete finite control traces without loading pickled objects.

    Args:
        path: Hash-verified native NPZ artifact.
        population: Expected full shared-scene robot count.
        steps: Number of prescribed control actions.
    Returns:
        Measured trace with full state and focal observation arrays.
    Raises:
        ValueError: Steps, groups, population or finite values are invalid.
    """
    trace = [{"state": {}, "observations": {}, "native": {}} for _ in range(steps + 1)]
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate probe array")
        for name in archive.files:
            step, group, key = name.split("/", 2)
            if not step.isdecimal() or str(int(step)) != step or not key:
                raise ValueError("invalid probe array name")
            if int(step) > steps or group not in trace[0]:
                raise ValueError("unexpected probe step or group")
            value = archive[name]
            count = population if group == "state" else 1
            if (
                not value.ndim
                or value.shape[0] != count
                or not np.isfinite(value).all()
            ):
                raise ValueError("invalid probe population or values")
            trace[int(step)][group][key] = value.copy()
    _validate_loaded_trace(trace, population)
    return trace


def _validate_loaded_trace(trace, population):
    from types import SimpleNamespace
    from npa.workflows.navigation.measure import snapshot

    schema = None
    for row in trace:
        if not row["state"] or not row["observations"]:
            raise ValueError("incomplete probe trace")
        if not row["native"]:
            del row["native"]
        raw = row["state"]
        checked = snapshot(SimpleNamespace(measure=lambda _: raw), None, population)
        if set(raw) != set(checked):
            raise ValueError("unexpected measured-state fields")
        current = {
            group: {key: value.shape for key, value in fields.items()}
            for group, fields in row.items()
        }
        if schema is not None and current != schema:
            raise ValueError("probe streams or shapes changed")
        schema = current


def trace_difference(baseline, changed):
    """Describe every focal-stream delta and unexpected contact across all robots.

    Args:
        baseline: Solo-control measured trace.
        changed: Repeated-reset or overlapping-peer measured trace.
    Returns:
        Per-step differences, maximum difference and maximum peer force.
    Raises:
        ValueError: Trace lengths, stream names or array shapes differ.
    """
    differences = []
    native = []
    peer_force = 0.0
    for step, (left, right) in enumerate(zip(baseline, changed, strict=True)):
        for group in ("state", "observations"):
            differences.extend(_group_differences(step, group, left, right))
        if "native" in left or "native" in right:
            native.extend(_group_differences(step, "native", left, right))
        peer_force = max(peer_force, float(np.max(right["state"]["peer_contact"])))
    return {
        "maximum": max(differences, key=lambda item: item["absolute_delta"]),
        "maximum_peer_contact": peer_force,
        "differences": differences,
        "native_differences": native,
    }


def _group_differences(step, group, left, right):
    if (
        group not in left
        or group not in right
        or left[group].keys() != right[group].keys()
    ):
        raise ValueError("observation/measurement streams changed during probe")
    return [
        _difference(step, group, name, array, right[group][name])
        for name, array in left[group].items()
    ]


def _difference(step, group, name, baseline, changed):
    if baseline.shape != changed.shape:
        raise ValueError("probe measurement shapes changed")
    left, right = np.asarray(baseline[0]), np.asarray(changed[0])
    delta = np.abs(left - right)
    flat_index = int(np.argmax(delta))
    return {
        "step": step,
        "group": group,
        "stream": name,
        "component": [
            int(index) for index in np.unravel_index(flat_index, delta.shape)
        ],
        "absolute_delta": float(delta.flat[flat_index]),
        "baseline": float(left.flat[flat_index]),
        "changed": float(right.flat[flat_index]),
    }
