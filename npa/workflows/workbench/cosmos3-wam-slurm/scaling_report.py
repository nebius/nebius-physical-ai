"""Compare three separate timing repetitions per measured B200 topology."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("saved measurement disagrees with numeric iteration series")


def _series(directory, measurement):
    raw = (directory / "iteration-series.csv").read_bytes()
    if _digest(raw) != measurement["iteration_series_sha256"]:
        raise ValueError("iteration series hash mismatch")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    if [int(row["step"]) for row in rows] != list(range(52, 201)):
        raise ValueError("timing repetition must contain every update from 52 to 200")
    seconds = [float(row["iteration_seconds"]) for row in rows]
    tokens = [int(row["tokens"]) for row in rows]
    values = [float(value) for row in rows for value in row.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite numeric iteration series")
    if min(seconds) <= 0 or min(tokens) <= 0:
        raise ValueError("step durations and token counts must be positive")
    return seconds, tokens


def _verify_summary(measurement, seconds, tokens):
    if measurement["measured_steps"] != len(seconds):
        raise ValueError("incorrect measured step count")
    expected = {
        "step_mean_seconds": statistics.mean(seconds),
        "step_p50_seconds": statistics.median(seconds),
        "step_p95_seconds": statistics.quantiles(seconds, n=100, method="inclusive")[
            94
        ],
    }
    for key, value in expected.items():
        _close(measurement[key], value)
    work = measurement["work"]
    _close(work["measured_tokens"], sum(tokens))
    _close(work["mean_tokens_per_optimizer_step"], statistics.mean(tokens))
    _close(work["tokens_per_second"], sum(tokens) / sum(seconds))


def _read(directory):
    raw = (directory / "measurement.json").read_bytes()
    measurement = json.loads(raw)
    contract = measurement["comparison_contract"]
    if (
        measurement["schema"] != "npa.cosmos3.wam-measurement.v1"
        or measurement["status"] != "measured"
        or measurement["gpus"] not in (8, 16)
        or measurement["nodes"] * 8 != measurement["gpus"]
        or measurement["warmup_steps_excluded"] != 50
        or contract["steps"] != 200
        or contract["profile"] is not False
    ):
        raise ValueError("requires completed, unprofiled 200-update timing reports")
    seconds, tokens = _series(directory, measurement)
    _verify_summary(measurement, seconds, tokens)
    return {
        "measurement_sha256": _digest(raw),
        "measurement": measurement,
        "seconds": seconds,
        "tokens": tokens,
    }


def _spread(values):
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _repeat(record):
    report = record["measurement"]
    return {
        "measurement_sha256": record["measurement_sha256"],
        "iteration_series_sha256": report["iteration_series_sha256"],
        "measured_steps": report["measured_steps"],
        "step_mean_seconds": report["step_mean_seconds"],
        "step_p50_seconds": report["step_p50_seconds"],
        "step_p95_seconds": report["step_p95_seconds"],
        "measured_tokens": report["work"]["measured_tokens"],
        "tokens_per_second": report["work"]["tokens_per_second"],
    }


def _group(records):
    if len(records) != 3:
        raise ValueError("the campaign requires three timing repetitions per topology")
    repetitions = [_repeat(record) for record in records]
    means = [repeat["step_mean_seconds"] for repeat in repetitions]
    throughputs = [repeat["tokens_per_second"] for repeat in repetitions]
    total_seconds = sum(sum(record["seconds"]) for record in records)
    total_tokens = sum(sum(record["tokens"]) for record in records)
    return {
        "repetitions": repetitions,
        "replicate_step_means_seconds": _spread(means),
        "replicate_token_throughputs": _spread(throughputs),
        "measured_steps": sum(repeat["measured_steps"] for repeat in repetitions),
        "measured_tokens": total_tokens,
        "pooled_tokens_per_second": total_tokens / total_seconds,
    }


def _matching_records(directories):
    if len({directory.resolve() for directory in directories}) != len(directories):
        raise ValueError("duplicate timing-run directory")
    records = [_read(directory) for directory in directories]
    if len({record["measurement_sha256"] for record in records}) != len(records):
        raise ValueError("duplicate timing measurement; copied reports are not repeats")
    baseline = records[0]["measurement"]
    for record in records:
        for key in ("comparison_contract", "hardware"):
            if record["measurement"][key] != baseline[key]:
                raise ValueError(
                    "timing repetitions have different protocol or hardware"
                )
    return records, baseline


def _comparison(groups):
    if "16" not in groups:
        return None
    baseline, candidate = groups["8"], groups["16"]
    speedup = (
        baseline["replicate_step_means_seconds"]["mean"]
        / candidate["replicate_step_means_seconds"]["mean"]
    )
    return {
        "speedup_vs_8_gpus": speedup,
        "scaling_efficiency": speedup / 2,
        "token_throughput_speedup": candidate["pooled_tokens_per_second"]
        / baseline["pooled_tokens_per_second"],
        "token_work_ratio": candidate["measured_tokens"] / baseline["measured_tokens"],
    }


def _summarize(directories):
    records, baseline = _matching_records(directories)
    grouped = {}
    for record in records:
        grouped.setdefault(str(record["measurement"]["gpus"]), []).append(record)
    if "8" not in grouped:
        raise ValueError("requires the eight-GPU baseline")
    groups = {
        key: _group(value)
        for key, value in sorted(grouped.items(), key=lambda item: int(item[0]))
    }
    return {
        "schema": "npa.cosmos3.wam-repeated-scaling.v1",
        "status": "scaling_measured" if "16" in groups else "baseline_measured",
        "comparison_contract": baseline["comparison_contract"],
        "hardware": baseline["hardware"],
        "groups": groups,
        "comparison": _comparison(groups),
        "interpretation": (
            "Variability is the sample standard deviation of three run means, "
            "not a confidence interval or a distribution over independent optimizer steps. "
            "Repetitions use the same training seed; they do not estimate policy-quality "
            "variation across seeds. Final checkpoint writing is outside these timed "
            "iterations. Full-schedule duration, allocation time and quality are separate."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    arguments = parser.parse_args()
    result = _summarize(arguments.run_dirs)
    arguments.output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
