"""Capture real B300 GPU activities through the matching native CUPTI interface."""

import ctypes
import json
import os
from pathlib import Path


class CuptiProfile:
    """Trace update five, retaining the existing six-update measurement exclusion.

    Args:
        destination: Private Chrome trace output path.
    Returns:
        A profiler context with the same per-update step protocol as PyTorch.
    Raises:
        RuntimeError: Native tracing fails, drops records or captures no kernels.
    """

    def __init__(self, destination):
        from cupti import cupti
        import torch

        self.api = cupti
        self.synchronize = torch.cuda.synchronize
        self.destination = Path(destination)
        self.library = ctypes.CDLL(os.environ["NPA_FLEX_PI_CUPTI_LIBRARY"])
        self.kinds = [
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        ]
        self.updates = 0
        self.active = False
        self.records = []
        self.callback_errors = []
        self.receipt = None

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if self.active:
            self._stop()
        if error_type is None and self.receipt is None:
            raise RuntimeError("B300 profile did not reach its instrumented update")
        return False

    @staticmethod
    def _requested():
        return 8 * 1024 * 1024, 0

    def _completed(self, activities):
        # Exceptions cannot be propagated through the native callback boundary.
        try:
            for activity in activities:
                start, end = int(activity.start), int(activity.end)
                if start <= 0 or end <= start:
                    raise RuntimeError("CUPTI returned an invalid activity interval")
                kernel = activity.kind == self.api.ActivityKind.CONCURRENT_KERNEL
                self.records.append(
                    {
                        "name": activity.name if kernel else str(activity.kind),
                        "cat": "kernel" if kernel else "gpu_memory",
                        "ph": "X",
                        "pid": int(activity.device_id),
                        "tid": int(activity.stream_id),
                        "start_ns": start,
                        "end_ns": end,
                    }
                )
        except Exception as error:
            self.callback_errors.append(type(error).__name__)

    def step(self):
        """Advance after one completed optimizer update, outside its timing."""
        self.updates += 1
        if self.updates == 4:
            self.synchronize()
            self.api.activity_register_callbacks(self._requested, self._completed)
            enabled = []
            try:
                for kind in self.kinds:
                    self.api.activity_enable(kind)
                    enabled.append(kind)
            except Exception:
                for kind in enabled:
                    self.api.activity_disable(kind)
                raise
            self.active = True
        elif self.updates == 5:
            self._stop()

    def _stop(self):
        try:
            self.synchronize()
            self.api.activity_flush_all(1)
        finally:
            for kind in self.kinds:
                self.api.activity_disable(kind)
            self.active = False
        dropped = ctypes.c_size_t()
        query = self.library.cuptiActivityGetNumDroppedRecords
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        query.restype = ctypes.c_int
        status = query(None, 0, ctypes.byref(dropped))
        if status or dropped.value or self.callback_errors:
            raise RuntimeError(
                "B300 GPU profile has missing or invalid activity records"
            )
        self.receipt = write_gpu_trace(self.destination, self.records)


def write_gpu_trace(destination, records):
    """Write a complete GPU trace, rejecting CPU-only or malformed evidence.

    Args:
        destination: Private JSON trace path.
        records: Native kernel and memory intervals with integer nanoseconds.
    Returns:
        A small receipt describing the actual instrumented update.
    Raises:
        RuntimeError: No GPU kernels or invalid timestamps were recorded.
    """
    kernels = sum(row["cat"] == "kernel" for row in records)
    if not kernels:
        raise RuntimeError("B300 profiling captured no GPU kernels")
    if any(row["start_ns"] <= 0 or row["end_ns"] <= row["start_ns"] for row in records):
        raise RuntimeError("B300 profiling contains invalid GPU timestamps")
    origin = min(row["start_ns"] for row in records)
    events = [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {"start_ns", "end_ns"}
            },
            "ts": (row["start_ns"] - origin) / 1000,
            "dur": (row["end_ns"] - row["start_ns"]) / 1000,
        }
        for row in records
    ]
    receipt = {
        "backend": "cupti13",
        "instrumented_update": 5,
        "rank": 0,
        "gpu_kernel_events": kernels,
        "gpu_memory_events": len(records) - kernels,
        "dropped_records": 0,
        "excluded_initial_updates": 6,
        "scope": "single instrumented update; use steady windows for throughput",
    }
    destination = Path(destination)
    destination.write_text(json.dumps({"traceEvents": events, "metadata": receipt}))
    destination.chmod(0o600)
    return receipt
