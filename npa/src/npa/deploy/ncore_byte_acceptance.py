"""Validate explicit raw-clean versus adjudicated NCore byte disposition fields."""

import re


_ATTRIBUTION_FIELDS = (
    "attribution_receipt_sha256",
    "attribution_replay_report_sha256",
    "attribution_replay_ledger_sha256",
)


def _require(ok, field):
    if not ok:
        raise RuntimeError(f"NCore acceptance requires valid {field}")


def _exact(parent, key, expected):
    value = parent.get(key)
    _require(type(value) is type(expected) and value == expected, key)


def validate_disposition(scan, prepublication):
    """Check structural disposition without claiming to authenticate evidence.

    Args:
        scan: Byte-scan manifest record with raw and resolved counts.
        prepublication: Manifest record with conditional attribution hashes.
    Returns:
        None.
    Raises:
        RuntimeError: Raw findings, policy or attribution fields contradict the branch.
    """
    resolution = scan.get("resolution")
    _require(resolution in {"raw-clean", "public-attribution"}, "byte resolution")
    _require(
        scan.get("policy_kind") in {"regex-v1", "exact-literals-v1"}, "byte policy kind"
    )
    clean = resolution == "raw-clean"
    _exact(scan, "raw_valid", clean)
    _exact(scan, "raw_findings", 0 if clean else 2)
    _exact(scan, "dispositioned_findings", 0 if clean else 2)
    _exact(scan, "unresolved_findings", 0)
    if not clean:
        _exact(scan, "policy_kind", "regex-v1")
    for field in _ATTRIBUTION_FIELDS:
        value = prepublication.get(field)
        if clean:
            _require(value is None, field)
        else:
            _require(
                isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                field,
            )
