"""Compare every gradient, optimizer moment and applied update in batch parity."""

import math

import torch


def tensor_difference(reference, candidate):
    """Measure finite tensor differences without hiding a zero reference.

    Args:
        reference: Original CPU tensor.
        candidate: Candidate tensor with identical shape and dtype.
    Returns:
        Exact equality, finite checks and additive norm statistics.
    Raises:
        RuntimeError: Tensor identity differs before numerical comparison.
    """
    candidate = candidate.detach().cpu()
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError("parity tensor shape or dtype differs")
    result = {
        "reference_squared": 0.0,
        "candidate_squared": 0.0,
        "difference_squared": 0.0,
        "dot": 0.0,
        "maximum_absolute": 0.0,
        "elements": reference.numel(),
        "exact": torch.equal(reference, candidate),
        "finite": True,
    }
    left, right = reference.reshape(-1), candidate.reshape(-1)
    _accumulate_differences(result, left, right)
    return result


def _accumulate_differences(result, left, right):
    for start in range(0, left.numel(), 1024 * 1024):
        first = left[start : start + 1024 * 1024].double()
        second = right[start : start + 1024 * 1024].double()
        difference = second - first
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise RuntimeError("nonfinite fixed-batch parity tensor")
        result["reference_squared"] += float(first.square().sum())
        result["candidate_squared"] += float(second.square().sum())
        result["difference_squared"] += float(difference.square().sum())
        result["dot"] += float((first * second).sum())
        result["maximum_absolute"] = max(
            result["maximum_absolute"], float(difference.abs().max())
        )


def _derived(statistics):
    reference = math.sqrt(statistics["reference_squared"])
    candidate = math.sqrt(statistics["candidate_squared"])
    error = math.sqrt(statistics["difference_squared"])
    return {
        "relative_l2": error / reference if reference else None,
        "norm_relative_error": abs(candidate - reference) / reference
        if reference
        else None,
        "cosine": statistics["dot"] / (reference * candidate)
        if reference and candidate
        else None,
        "rms_absolute": error / math.sqrt(statistics["elements"])
        if statistics["elements"]
        else 0.0,
    }


def _accepted(kind, statistics, *, exact):
    if not statistics["finite"]:
        return False
    if exact or statistics["reference_squared"] == 0:
        return statistics["exact"]
    derived = _derived(statistics)
    if derived["relative_l2"] > 0.05:
        return False
    if kind == "gradient":
        return (
            derived["cosine"] is not None
            and derived["cosine"] >= 0.999
            and derived["norm_relative_error"] <= 0.02
        )
    return True


class ParityComparison:
    """Retain reference tensors privately or compare one candidate at a time.

    Args:
        reference: Reference tensor mapping, shared between passes.
        capture: Populate reference tensors instead of comparing them.
        exact: Require exact B1 capture/replay transparency.
    Returns:
        A collector of per-parameter and aggregate comparison evidence.
    Raises:
        RuntimeError: A tensor's identity or expected population differs.
    """

    def __init__(self, reference, *, capture=False, exact=False):
        self.reference = reference
        self.capture = capture
        self.exact = exact
        self.records = []
        self.observed = set()

    def observe(self, kind, name, value):
        """Check one complete parameter tensor without exporting its values.

        Args:
            kind: Gradient, applied update, or optimizer moment.
            name: Stable model parameter name.
            value: Complete candidate tensor.
        Returns:
            None; records contain only numerical summaries.
        Raises:
            RuntimeError: A tensor is duplicated or absent from the reference.
        """
        key = (kind, name)
        if key in self.observed:
            raise RuntimeError("duplicate parity tensor")
        self.observed.add(key)
        if self.capture:
            if not torch.isfinite(value).all():
                raise RuntimeError("nonfinite fixed-batch reference tensor")
            self.reference[key] = value.detach().cpu().clone()
            return
        statistics = tensor_difference(self.reference[key], value)
        self.records.append(
            {
                "kind": kind,
                "parameter": name,
                **statistics,
                **_derived(statistics),
                "passed": _accepted(kind, statistics, exact=self.exact),
            }
        )

    def finish(self):
        """Require full tensor coverage and summarize the actual comparisons.

        Args:
            None.
        Returns:
            Per-parameter numerical evidence and overall acceptance.
        Raises:
            RuntimeError: A parameter or optimizer state was omitted.
        """
        if self.observed != set(self.reference):
            raise RuntimeError("parity tensor population changed")
        aggregates = _aggregate(self.records, exact=self.exact)
        return {
            "passed": all(row["passed"] for row in self.records + aggregates),
            "exact_required": self.exact,
            "tensor_count": len(self.observed),
            "parameters": self.records,
            "aggregates": aggregates,
        }


def _aggregate(records, *, exact):
    result = []
    for kind in sorted({row["kind"] for row in records}):
        rows = [row for row in records if row["kind"] == kind]
        statistics = {
            key: sum(row[key] for row in rows)
            for key in (
                "reference_squared",
                "candidate_squared",
                "difference_squared",
                "dot",
                "elements",
            )
        }
        statistics.update(
            maximum_absolute=max(row["maximum_absolute"] for row in rows),
            finite=all(row["finite"] for row in rows),
            exact=all(row["exact"] for row in rows),
        )
        result.append(
            {
                "kind": kind,
                **statistics,
                **_derived(statistics),
                "passed": _accepted(kind, statistics, exact=exact),
            }
        )
    return result
