"""Verify actionable capacity guidance for quota-blocked deployments."""

from npa.cluster_backends.quotas import QuotaShortfall, shortfall_message


def test_disk_shortfall_preserves_bytes_and_explains_reserved_capacity():
    shortage = QuotaShortfall(
        name="compute.disk.size.network-ssd",
        region="us-central1",
        required=2302 * 1024**3,
        limit=25600 * 1024**3,
        available=137 * 1024**3,
        unit="byte",
    )
    message = shortfall_message([shortage], "<tenant>")
    assert f"needs {2302 * 1024**3} byte" in message
    assert "requires 2,302.00 GiB" in message
    assert "available 137.00 GiB" in message
    assert "shortfall 2,165.00 GiB" in message
    assert "Reserved GPUs still require worker boot-disk quota" in message
    assert "rerun the same deploy with preflight enabled" in message
    assert "Deploy with --no-preflight" not in message


def test_count_shortfall_stays_in_counts_and_uses_available_capacity():
    shortage = QuotaShortfall("compute.instance.count", "us-central1", 4, 65, 2, "count")
    message = shortfall_message([shortage], "<tenant>")
    assert "needs 4 count, 2 available from tenant limit 65; shortfall 2" in message
    assert "GiB" not in message
    assert "worker boot-disk" not in message


def test_shortfall_without_usage_uses_the_limit():
    shortage = QuotaShortfall("compute.disk.count", "us-central1", 4, 3)
    assert shortage.describe().endswith("tenant limit 3; shortfall 1")
