"""Summarize repeated fixed-workload measurement windows without cold steps."""

import statistics

PROFILE_UPDATES = 30
EXCLUDED_INITIAL_UPDATES = 6
WINDOW_UPDATES = 8


def summarize_measurements(rows):
    """Calculate throughput over complete steady-state windows of global batch 96.

    Args:
        rows: Actual optimizer-update timing and sample-count records.
    Returns:
        Cold/trace overhead and repeated steady-state rates separately.
    Raises:
        ValueError: Fewer than three complete steady-state windows were measured.
    """
    steady = [row for row in rows[EXCLUDED_INITIAL_UPDATES:] if row["samples"] == 96]
    windows = []
    for offset in range(0, len(steady) - WINDOW_UPDATES + 1, WINDOW_UPDATES):
        window = steady[offset:offset + WINDOW_UPDATES]
        samples = sum(row["samples"] for row in window)
        seconds = sum(row["seconds"] for row in window)
        windows.append({"samples": samples, "seconds": seconds,
                        "samples_per_second": samples / seconds})
    if len(windows) < 3:
        raise ValueError("at least three complete steady-state windows are required")
    rates = [row["samples_per_second"] for row in windows]
    return {
        "excluded_initial_updates": EXCLUDED_INITIAL_UPDATES,
        "initial_seconds": sum(row["seconds"] for row in rows[:EXCLUDED_INITIAL_UPDATES]),
        "windows": windows,
        "median_samples_per_second": statistics.median(rates),
        "minimum_samples_per_second": min(rates),
        "maximum_samples_per_second": max(rates),
        "all_update_samples": sum(row["samples"] for row in rows),
    }
