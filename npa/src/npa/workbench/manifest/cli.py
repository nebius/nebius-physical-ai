"""CLI surface: a thin waiter over Runtime.invoke.

Usage: python -m npa.workbench.manifest.cli --backend nebius run <tool> <version> <command> [--input k=v ...]
"""

from __future__ import annotations

import argparse
import json

from . import Catalog, LocalDockerBackend, Runtime


def build_runtime(descriptors: str, records: str, backend: str) -> Runtime:
    backend_objs: dict = {"local": LocalDockerBackend()}
    try:
        from .nebius_backend import NebiusBackend

        backend_objs["nebius"] = NebiusBackend()
    except Exception:
        pass
    if backend not in backend_objs:
        raise SystemExit(f"backend {backend!r} unavailable")
    return Runtime(Catalog(descriptors), backend_objs, records_dir=records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="manifest")
    ap.add_argument("--descriptors", default="descriptors")
    ap.add_argument("--records", default="records")
    ap.add_argument("--backend", default="local")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("tool")
    p_run.add_argument("version")
    p_run.add_argument("command")
    p_run.add_argument(
        "--input", action="append", default=[], help="key=value (repeatable)"
    )

    sub.add_parser("list")

    args = ap.parse_args(argv)
    rt = build_runtime(args.descriptors, args.records, args.backend)

    if args.cmd == "list":
        for tid in rt.catalog.list():
            print(tid)
        return 0

    inputs: dict = {}
    for item in args.input:
        k, _, v = item.partition("=")
        inputs[k] = _coerce(v)
    res = rt.invoke(
        args.tool,
        args.version,
        args.command,
        inputs=inputs,
        surface="cli",
        backend=args.backend,
    )
    print(json.dumps(res.verification_record(), indent=2))
    return 0 if res.success else 1


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


if __name__ == "__main__":
    raise SystemExit(main())
