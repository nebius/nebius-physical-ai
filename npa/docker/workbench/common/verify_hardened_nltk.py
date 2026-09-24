"""Exercise NLTK's fixes for GHSA-8mgp-746c-j5xp at image-build time."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import nltk.data
from nltk.classify.maxent import save_maxent_params
from nltk.parse.transitionparser import TransitionParser
from nltk.tag.perceptron import AveragedPerceptron, PerceptronTagger

HARDENED_VERSION = "3.10.3.post1+npa.cbc98458"


class _Weights:
    def tolist(self) -> list[float]:
        return []


def _expect_denied(label: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except (PermissionError, ValueError):
        return
    raise RuntimeError(f"{label} accepted a path outside the NLTK data sandbox")


def _verify_transition_train_guard() -> None:
    source = inspect.getsource(TransitionParser.train)
    expected = 'pathsec_open(modelfile, "wb", context="TransitionParser.train")'
    if expected not in source:
        raise RuntimeError(
            "TransitionParser.train does not use the pathsec write guard"
        )


def main() -> None:
    version = importlib.metadata.version("nltk")
    if version != HARDENED_VERSION:
        raise RuntimeError(f"nltk={version}; expected {HARDENED_VERSION}")

    root = Path(tempfile.mkdtemp(prefix="npa-nltk-security-"))
    allowed = root / "allowed"
    outside = root / "outside"
    allowed.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    nltk.data.path[:] = [str(allowed)]

    try:
        model = AveragedPerceptron(weights={"feature": {"TAG": 1.0}})
        safe_model = allowed / "perceptron.json"
        model.save(safe_model)
        AveragedPerceptron().load(safe_model)

        denied_model = outside / "perceptron.json"
        _expect_denied("AveragedPerceptron.save", lambda: model.save(denied_model))
        denied_model.write_text("{}", encoding="utf-8")
        before = hashlib.sha256(denied_model.read_bytes()).digest()
        _expect_denied(
            "AveragedPerceptron.load", lambda: AveragedPerceptron().load(denied_model)
        )

        denied_tagger = outside / "tagger"
        _expect_denied(
            "PerceptronTagger.save_to_json",
            lambda: PerceptronTagger(load=False).save_to_json(loc=denied_tagger),
        )
        denied_maxent = outside / "maxent"
        _expect_denied(
            "save_maxent_params",
            lambda: save_maxent_params(_Weights(), {}, [], {}, str(denied_maxent)),
        )
        _expect_denied(
            "TransitionParser.parse",
            lambda: TransitionParser(TransitionParser.ARC_STANDARD).parse(
                [], str(denied_model)
            ),
        )
        _verify_transition_train_guard()

        if hashlib.sha256(denied_model.read_bytes()).digest() != before:
            raise RuntimeError("a denied NLTK operation modified an outside file")
        if denied_tagger.exists() or denied_maxent.exists():
            raise RuntimeError("a denied NLTK operation created an outside directory")
    finally:
        shutil.rmtree(root)

    print(f"[PASS] NLTK path sandbox verified ({HARDENED_VERSION})")


if __name__ == "__main__":
    main()
