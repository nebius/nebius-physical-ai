"""Expose the planar teaching fixture only through the explicit local demo runner."""

from functools import partial

from .batch import _batch
from .cli import _run
from .data import curate, split
from .gate import gate


if __name__ == "__main__":
    _run(
        {
            "curate": curate,
            "split": split,
            "batch": partial(_batch, _reference=True),
            "gate": partial(gate, _reference=True),
        }
    )
