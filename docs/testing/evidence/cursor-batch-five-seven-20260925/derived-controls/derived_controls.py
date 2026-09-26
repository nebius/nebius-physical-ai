"""Exercise production replay using private copied evidence; never contact providers."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml
from npa.orchestration.npa_workflow import runtime
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_steps_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec

CLASSIFICATION = "derived fault-injected integration; not native GPU events"
IDENTITIES = ("workflow_sha256", "source_sha256", "image_digest")


def digest(body):
    return hashlib.sha256(body).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def bounded_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("unsafe copied evidence path")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("copied evidence path escapes its private root")
    return target


class EvidenceFailure(RuntimeError):
    """The copied artifact no longer satisfies its verified readback contract."""


class ProviderTrap:
    def __init__(self):
        self.calls = {}

    def seam(self, name):
        self.calls.setdefault(name, 0)

        def forbidden(*_args, **_kwargs):
            self.calls[name] += 1
            raise RuntimeError("forbidden provider boundary: " + name)

        return forbidden

    def assert_unused(self):
        assert all(count == 0 for count in self.calls.values()), self.calls


@contextmanager
def offline_boundaries(trap, source_uri):
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"NPA_SRC_S3_URI": source_uri}))
        for name in (
            "socket.create_connection",
            "socket.socket.connect",
            "subprocess.Popen",
            "boto3.client",
            "boto3.session.Session.client",
        ):
            stack.enter_context(patch(name, trap.seam(name)))
        yield


def verified_objects(root, inventory):
    verified = {}
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise EvidenceFailure("readback object tree contains a symlink")
    for entry in inventory:
        relative = entry["relative"]
        if relative in verified:
            raise ValueError("duplicate artifact inventory entry")
        path = bounded_path(root, relative)
        body = path.read_bytes()
        if digest(body) != entry["sha256"] or len(body) != entry["bytes"]:
            raise EvidenceFailure("readback inventory hash or length differs")
        verified[relative] = dict(entry)
    if not verified:
        raise EvidenceFailure("readback inventory is empty")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual != set(verified):
        raise EvidenceFailure("readback inventory omits copied objects")
    return verified


def verify_source(candidate, expected):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=candidate, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=candidate, text=True
    ).strip()
    module = Path(runtime.__file__).resolve()
    assert actual == expected and not dirty
    assert module.is_relative_to(candidate.resolve() / "npa/src")
    return {"git_sha": actual, "runtime_file_sha256": digest(module.read_bytes())}


def read_bundle(path):
    data = json.loads(path.read_text())
    root = path.parent
    if data["evidence_class"] != "completed-native-gpu-readback":
        raise ValueError("bundle must explicitly bind completed native GPU evidence")
    required_options = {
        "registry",
        "image_overrides",
        "image_digest_pins",
        "gpu_target",
        "image_variant",
    }
    if not required_options.issubset(data["render_options"]):
        raise EvidenceFailure("effective image-selection projection is incomplete")
    for name, expected in data["metadata_sha256"].items():
        if digest(bounded_path(root, name).read_bytes()) != expected:
            raise EvidenceFailure("bundle metadata hash differs")
    inventory = json.loads(bounded_path(root, data["inventory"]).read_text())
    objects = bounded_path(root, data["objects"])
    entries = verified_objects(objects, inventory)
    spec = load_spec(bounded_path(objects, data["workflow_artifact"]))
    options = SkypilotRenderOptions(**data["render_options"])
    state = json.loads((objects / "npa-workflow/runtime.json").read_bytes())
    assert state["status"] == "succeeded" and state["waves"]
    assert state["run_id"] == data["run_id"] and state["workflow"] == spec.name
    assert all(wave["status"] == "succeeded" for wave in state["waves"])
    data.update(
        root=root, object_root=objects, entries=entries, spec=spec, options=options
    )
    return data


class CopiedStore:
    def __init__(self, bundle, destination):
        shutil.copytree(bundle["object_root"], destination)
        self.root = destination
        self.bucket = bundle["bucket"]
        self.prefix = bundle["prefix"].strip("/")
        self.entries = bundle["entries"]
        self.output_checks = 0
        self.writes = 0
        self.store = RunStateStore(
            bucket=self.bucket,
            prefix=self.prefix,
            reader=self.read,
            writer=self.write,
            artifact_lister=self.list_objects,
        )

    def relative(self, bucket, key):
        if bucket != self.bucket or not key.startswith(self.prefix + "/"):
            raise ValueError("copied store access outside exact baseline scope")
        return key.removeprefix(self.prefix + "/")

    def read(self, bucket, key):
        return bounded_path(self.root, self.relative(bucket, key)).read_bytes()

    def write(self, bucket, key, body):
        path = bounded_path(self.root, self.relative(bucket, key))
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(body)
        path.chmod(0o600)
        self.writes += 1

    def list_objects(self, bucket, prefix):
        relative = self.relative(bucket, prefix)
        return [
            self.prefix + "/" + path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
            and path.relative_to(self.root).as_posix().startswith(relative)
        ]

    def output_names(self, uri):
        base = f"s3://{self.bucket}/{self.prefix}/"
        if not uri.startswith(base):
            raise EvidenceFailure("output is outside exact copied run scope")
        name = uri.removeprefix(base)
        matches = (
            [key for key in self.entries if key.startswith(name)]
            if uri.endswith("/")
            else [name]
        )
        selected = [
            key for key in matches if self.entries.get(key, {}).get("bytes", 0) > 0
        ]
        if not selected:
            raise EvidenceFailure("no verified nonempty output in baseline inventory")
        return selected

    def check_output(self, uri):
        self.output_checks += 1
        for name in self.output_names(uri):
            try:
                body = bounded_path(self.root, name).read_bytes()
            except FileNotFoundError as exc:
                raise EvidenceFailure("copied output is absent") from exc
            if digest(body) != self.entries[name]["sha256"]:
                raise EvidenceFailure("copied output hash is corrupt")
        return True


def executor_for(bundle, copied, trap, *, options=None):
    ledger = runtime.RuntimeLedger(
        copied.store,
        workflow=bundle["spec"].name,
        run_id=bundle["run_id"],
        api_version=bundle["spec"].api_version,
        resume=True,
    )
    seams = {
        name: trap.seam(name)
        for name in (
            "submitter",
            "status_fn",
            "timeline_fn",
            "canceller",
            "name_lookup_fn",
            "reconcile_fn",
        )
    }
    return runtime.SkyPilotWaveExecutor(
        bundle["spec"],
        run_id=bundle["run_id"],
        ledger=ledger,
        render_options=options or bundle["options"],
        options=runtime.RuntimeOptions(resume=True),
        output_checker=copied.check_output,
        **seams,
    )


def mutate_identity(copied, field, change):
    state = copied.store.read_runtime_state()
    identity = state.waves[0]["immutable_identity"]
    if change == "missing":
        identity.pop(field)
    else:
        identity[field] = "0" * 64 if identity[field] != "0" * 64 else "1" * 64
    copied.store.write_runtime_state(state)


def mutate_artifact(copied, change):
    state = copied.store.read_runtime_state()
    output = state.waves[0]["outputs"][0]
    uri = output if isinstance(output, str) else output["uri"]
    names = copied.output_names(uri)
    for name in names:
        path = bounded_path(copied.root, name)
        if change == "absent":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"\nFAULT-INJECTED-CORRUPTION\n")


def verify_replay_bindings(bundle, copied):
    state = copied.store.read_runtime_state()
    expected = dict(
        zip(
            IDENTITIES,
            (
                runtime._workflow_identity(bundle["spec"]),
                runtime._source_identity(),
                runtime._image_identity(bundle["options"]),
            ),
        )
    )
    for wave in state.waves:
        assert wave["immutable_identity"] == expected, (
            "real baseline identity reconstruction differs"
        )
        for output in wave["outputs"]:
            copied.check_output(output if isinstance(output, str) else output["uri"])
    checked = copied.output_checks
    copied.output_checks = 0
    return checked


def run_case(bundle, destination, name, *, identity=None, artifact=None):
    copied = CopiedStore(bundle, destination / name / "objects")
    trap = ProviderTrap()
    with offline_boundaries(trap, bundle["source_uri"]):
        integrity_checks = verify_replay_bindings(bundle, copied)
        if identity:
            mutate_identity(copied, *identity)
        if artifact:
            mutate_artifact(copied, artifact)
        executor = executor_for(bundle, copied, trap)
        report = runtime.run_workflow_runtime(
            bundle["spec"],
            run_id=bundle["run_id"],
            executor=executor,
            options=executor.options,
        )
    trap.assert_unused()
    assert report.status == ("failed" if identity or artifact else "succeeded")
    if identity:
        assert "IMMUTABLE_IDENTITY_MISMATCH" in report.error
        assert identity[0] in report.error and copied.output_checks == 0
    if artifact:
        assert "copied output" in report.error and copied.output_checks > 0
    if not identity and not artifact:
        assert report.waves and all(wave["replayed"] for wave in report.waves)
    result = {
        "case": name,
        "status": "passed",
        "runtime_status": report.status,
        "provider_calls": trap.calls,
        "output_checks": copied.output_checks,
        "baseline_integrity_output_checks": integrity_checks,
        "local_writes": copied.writes,
        "classification": CLASSIFICATION,
    }
    write_json(destination / name / "result.private.json", result)
    return result


def selector_options():
    refs = ["cr.example/caption:fixture", "cr.example/generate:fixture"]
    images = [
        "cr.example/caption@sha256:" + "a" * 64,
        "cr.example/generate@sha256:" + "b" * 64,
    ]
    selectors = ["workbench.token_factory.caption", "workbench.token_factory.generate"]
    return SkypilotRenderOptions(
        image_overrides=dict(zip(selectors, refs)),
        image_digest_pins=dict(zip(refs, images)),
        materialize_registry_secrets=False,
        default_setup=False,
        include_aws_endpoint=False,
    )


def rendered_tool_images(spec, options):
    steps = build_plan(spec, run_id="synthetic-selector-control").steps
    assert len(steps) == 2 and len({step.tool_ref for step in steps}) == 2
    images = {}
    for step in steps:
        rendered = render_skypilot_steps_yaml(
            spec,
            [step],
            run_id="synthetic-selector-control",
            options=options,
        )
        documents = [doc for doc in yaml.safe_load_all(rendered) if doc]
        resources = [doc["resources"] for doc in documents if "resources" in doc]
        assert len(resources) == 1
        images[step.tool_ref] = resources[0]["image_id"]
    return images


def check_selector_rendering(spec_path):
    spec, initial = load_spec(spec_path), selector_options()
    reordered = replace(
        initial,
        image_overrides=dict(reversed(list(initial.image_overrides.items()))),
        image_digest_pins=dict(reversed(list(initial.image_digest_pins.items()))),
    )
    keys, refs = list(initial.image_overrides), list(initial.image_overrides.values())
    swapped = replace(initial, image_overrides=dict(zip(keys, reversed(refs))))
    trap = ProviderTrap()
    with offline_boundaries(trap, ""):
        original = rendered_tool_images(spec, initial)
        assert original == rendered_tool_images(spec, reordered)
        assert runtime._image_identity(initial) == runtime._image_identity(reordered)
        changed = rendered_tool_images(spec, swapped)
        assert changed[keys[0]] == original[keys[1]]
        assert changed[keys[1]] == original[keys[0]]
        assert len(set(original.values())) == 2
        assert runtime._image_identity(initial) != runtime._image_identity(swapped)
    trap.assert_unused()
    return {
        "classification": "synthetic renderer control; not GPU evidence",
        "status": "passed",
        "tool_refs": keys,
        "distinct_images": 2,
        "reordered_mapping_unchanged": True,
        "swapped_mapping_changes_identity": True,
        "provider_calls": trap.calls,
        "spec_sha256": digest(spec_path.read_bytes()),
    }


def run_bundle(bundle, output):
    before = verified_objects(bundle["object_root"], list(bundle["entries"].values()))
    results = [run_case(bundle, output, "same-identity-replay")]
    for field in IDENTITIES:
        for change in ("changed", "missing"):
            results.append(
                run_case(bundle, output, f"{field}-{change}", identity=(field, change))
            )
    for change in ("absent", "corrupt"):
        results.append(
            run_case(bundle, output, f"copied-output-{change}", artifact=change)
        )
    after = verified_objects(bundle["object_root"], list(bundle["entries"].values()))
    assert before == after
    return results


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    source = verify_source(args.candidate, args.source_sha)
    selector = check_selector_rendering(
        Path(__file__).with_name("selector-fixture.yaml")
    )
    result = {
        "classification": CLASSIFICATION,
        "source": source,
        "synthetic_selector": selector,
    }
    if args.bundle:
        bundle = read_bundle(args.bundle)
        assert bundle["source_sha"] == args.source_sha
        result["derived_cases"] = run_bundle(bundle, args.output)
        result["baseline_objects_unchanged"] = True
        result["bundle_sha256"] = digest(args.bundle.read_bytes())
        result["verified_object_count"] = len(bundle["entries"])
    else:
        result["baseline_validation"] = "pending completed GPU readback handoff"
    write_json(args.output / "summary.private.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "derived_cases": len(result.get("derived_cases", [])),
                "synthetic_selector": "passed",
                "classification": CLASSIFICATION,
            }
        )
    )


if __name__ == "__main__":
    main()
