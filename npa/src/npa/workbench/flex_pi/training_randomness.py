"""Capture and replay per-anchor stochastic inputs for untimed batch parity."""

from dataclasses import fields

import torch
from torch.overrides import TorchFunctionMode


_EVENT_ORDER = ("randn_like", "rand") * 4
_OTHER_RANDOM = {
    "randn",
    "randint",
    "randperm",
    "bernoulli",
    "normal",
    "multinomial",
    "dropout",
    "uniform_",
    "normal_",
    "random_",
    "bernoulli_",
}


class AnchorRandomness(TorchFunctionMode):
    """Keep a private stochastic tape for one immutable group of real anchors.

    Args:
        entries: Mutable per-anchor records; initially empty for capture.
        capture: Record a microbatch of one, or replay recorded groups.
    Returns:
        A Torch function mode and flag-sampler wrapper.
    Raises:
        RuntimeError: Stochastic operations differ from the pinned architecture.
    """

    def __init__(self, entries, *, capture):
        super().__init__()
        self.entries = entries
        self.capture = capture
        self.group = []
        self.event = 0
        self.flags_seen = False
        self.inside_flags = False

    def begin(self, anchors):
        """Select the next group without exporting tensor contents.

        Args:
            anchors: Fixture-relative indices for the current microbatch.
        Returns:
            None.
        Raises:
            RuntimeError: Capture is requested for a batch larger than one.
        """
        if self.capture and len(anchors) != 1:
            raise RuntimeError("stochastic capture requires microbatch one")
        self.group = list(anchors)
        self.event = 0
        self.flags_seen = False
        if self.capture:
            self.entries[self.group[0]] = {"events": []}

    def finish(self):
        """Require exactly the pinned eight random events and one flag draw.

        Args:
            None.
        Returns:
            None.
        Raises:
            RuntimeError: An expected stochastic input is missing.
        """
        if self.event != len(_EVENT_ORDER) or not self.flags_seen:
            raise RuntimeError("stochastic tape was not completely consumed")

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = function.__name__
        if self.inside_flags:
            return function(*args, **kwargs)
        if name == "dropout" and _deterministic_dropout(args, kwargs):
            return function(*args, **kwargs)
        if name in _OTHER_RANDOM:
            raise RuntimeError(f"unexpected parity stochastic operation: {name}")
        if name not in {"randn_like", "rand"}:
            return function(*args, **kwargs)
        if self.event >= len(_EVENT_ORDER) or name != _EVENT_ORDER[self.event]:
            raise RuntimeError("stochastic event order changed")
        if self.capture:
            value = function(*args, **kwargs)
            if value.shape[0] != 1:
                raise RuntimeError("capture random tensor has an unexpected batch")
            self.entries[self.group[0]]["events"].append(value.detach().cpu().clone())
        else:
            value = self._replay_event(name, args, kwargs)
        self.event += 1
        return value

    def _replay_event(self, name, args, kwargs):
        saved = [self.entries[index]["events"][self.event] for index in self.group]
        if name == "randn_like":
            shape, dtype, device = args[0].shape, args[0].dtype, args[0].device
        else:
            shape = (
                args[0]
                if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size))
                else args
            )
            dtype, device = kwargs["dtype"], kwargs["device"]
        value = torch.cat(saved, dim=0)
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise RuntimeError("stochastic replay shape or dtype changed")
        return value.to(device=device)

    def flags(self, original, *args, **kwargs):
        """Wrap the pinned flag sampler while retaining its concrete return type.

        Args:
            original: Unmodified upstream flag sampler.
            *args: Upstream sampler positional arguments.
            **kwargs: Upstream sampler keyword arguments.
        Returns:
            Original or exactly replayed FlexBatchFlags.
        Raises:
            RuntimeError: Flags are duplicated or cross-modal losses are disabled.
        """
        if self.flags_seen:
            raise RuntimeError("duplicate parity flag draw")
        self.flags_seen = True
        if self.capture:
            return self._capture_flags(original, args, kwargs)
        return self._replay_flags(args, kwargs)

    def _capture_flags(self, original, args, kwargs):
        self.inside_flags = True
        try:
            result = original(*args, **kwargs)
        finally:
            self.inside_flags = False
        values = {}
        for field in fields(result):
            value = getattr(result, field.name)
            values[field.name] = (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
        if any(values[key] is not True for key in ("cm_v", "cm_d", "cm_p")):
            raise RuntimeError("batch parity requires all cross-modal losses")
        self.entries[self.group[0]]["flags"] = values
        self.entries[self.group[0]]["flag_type"] = type(result)
        return result

    def _replay_flags(self, args, kwargs):
        device = kwargs.get("device", args[2] if len(args) > 2 else None)
        batch = kwargs.get("batch_size", args[1] if len(args) > 1 else None)
        if batch != len(self.group) or device is None:
            raise RuntimeError("flag replay batch or device differs")
        first = self.entries[self.group[0]]
        values = {}
        for key, value in first["flags"].items():
            rows = [self.entries[index]["flags"][key] for index in self.group]
            if isinstance(value, torch.Tensor):
                values[key] = torch.cat(rows).to(device=device)
            else:
                if any(row != value for row in rows):
                    raise RuntimeError("cross-modal flag changed across anchors")
                values[key] = value
        return first["flag_type"](**values)


def _deterministic_dropout(args, kwargs):
    probability = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    training = kwargs.get("training", args[2] if len(args) > 2 else True)
    return not training or probability == 0
