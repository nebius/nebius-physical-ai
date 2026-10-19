"""Keep an old image producer distinct from its reviewed current evidence consumer."""

import fnmatch
import hashlib
import io
import re
import shlex
import subprocess

from image_byte_scan import core as W, prepare as P

from . import process, retained_receipts as receipts

FORMAT = "npa_ncore_retained_source_compatibility_v1"
RECIPE = "npa/docker/workbench/ncore"
BRIDGE_PATH = "retained/source-compatibility.json"
HOST_PATHS = (*process.SOURCE_PATHS, *process.CONTEXT)


def _git(*args, input_bytes=None):
    return subprocess.check_output(["git", *args], cwd=process.ROOT, input=input_bytes)


def _files(commit, paths):
    entries = _git("ls-tree", "-rz", commit, "--", *paths).split(b"\0")
    rows = []
    for entry in filter(None, entries):
        header, path = entry.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        W.require(
            kind == "blob" and mode in {"100644", "100755"}, "retained_source_file_mode"
        )
        rows.append({"path": path.decode(), "mode": mode, "blob": blob})
    stream = io.BytesIO(
        _git(
            "cat-file",
            "--batch",
            input_bytes="".join(row["blob"] + "\n" for row in rows).encode(),
        )
    )
    for row in rows:
        blob, kind, size = stream.readline().decode().split()
        raw = stream.read(int(size))
        W.require(
            blob == row["blob"] and kind == "blob" and stream.read(1) == b"\n",
            "retained_git_object",
        )
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    W.require(not stream.read(), "retained_git_population")
    return sorted(rows, key=lambda row: row["path"])


def _copy_paths(commit, context):
    dockerfile = _git("show", f"{commit}:{RECIPE}/Dockerfile").decode()
    selected = set()
    for line in dockerfile.replace("\\\n", " ").splitlines():
        words = shlex.split(line, comments=True)
        if not words:
            continue
        W.require(words[0].upper() != "ADD", "retained_unreviewed_add_input")
        if words[0].upper() != "COPY":
            continue
        sources = _copy_sources(words[1:])
        for source in sources:
            pattern = "npa/" + source.rstrip("/")
            matches = {
                row["path"]
                for row in context
                if fnmatch.fnmatchcase(row["path"], pattern)
                or row["path"].startswith(pattern + "/")
            }
            W.require(matches, "retained_copy_input_missing")
            selected.update(matches)
    W.require(selected, "retained_copy_population_empty")
    return sorted(selected)


def _copy_sources(words):
    flags = [word for word in words if word.startswith("--")]
    W.require(
        all(flag.startswith(("--from=", "--chmod=")) for flag in flags),
        "retained_copy_flag_unsupported",
    )
    if any(flag.startswith("--from=") for flag in flags):
        return []
    values = [word for word in words if not word.startswith("--")]
    W.require(
        len(values) >= 2
        and all(
            not value.startswith(("[", "/")) and ".." not in value.split("/")
            for value in values[:-1]
        ),
        "retained_copy_syntax_unsupported",
    )
    return values[:-1]


def identity(commit):
    """Derive complete immutable context, recipe/COPY and host-source closures.

    Args:
        commit: Full immutable Git commit, not an ancestry shortcut.
    Returns:
        Deterministic path/mode/blob/content identities and separate context hash.
    Raises:
        ValueError, OSError: Commit, modes or recipe syntax is unsupported.
    """
    W.require(re.fullmatch(r"[0-9a-f]{40}", str(commit)), "retained_full_commit")
    W.require(
        _git("rev-parse", commit + "^{commit}").decode().strip() == commit,
        "retained_commit_required",
    )
    context = _files(commit, process.CONTEXT)
    copies = set(_copy_paths(commit, context))
    return {
        "commit": commit,
        "tree": _git("rev-parse", commit + "^{tree}").decode().strip(),
        "context_sha256": hashlib.sha256(
            _git("archive", commit, *process.CONTEXT)
        ).hexdigest(),
        "context_files": context,
        "copy_inputs": [row for row in context if row["path"] in copies],
        "recipe_inputs": [
            row for row in context if row["path"].startswith(RECIPE + "/")
        ],
        "host_files": _files(commit, HOST_PATHS),
    }


def _deltas(producer, consumer):
    before = {row["path"]: row for row in producer["host_files"]}
    after = {row["path"]: row for row in consumer["host_files"]}
    return [
        {"path": path, "producer": before.get(path), "consumer": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]


def _build_bindings(bridge, root, manifest):
    records = bridge.get("build_receipts", {})
    W.require(
        set(records)
        == {
            "build",
            "metadata",
            "prepublication",
            "graph",
            "inspection",
            "driver",
            "loaded",
            "local_binding",
        },
        "retained_build_receipt_population",
    )
    values = {
        name: receipts.bound_json(root, record)
        for name, record in records.items()
        if name != "loaded"
    }
    build = values["build"]
    W.require(
        records["build"]["sha256"] == process.file_sha(root / "build/build.json")
        and build["source_sha"] == manifest["development_sha"]
        and build["context_sha256"] == bridge["producer"]["context_sha256"]
        and build["metadata"]["sha256"] == records["metadata"]["sha256"]
        and bridge.get("build_argv") == build.get("argv"),
        "retained_build_identity",
    )
    _build_graph(values, bridge, manifest)
    _build_provenance(values, bridge)
    loaded = W.bound_json(P.binding(receipts.bound_file(root, records["loaded"])))
    _loaded_labels(loaded, values, bridge)


def _build_graph(values, bridge, manifest):
    build, pre, graph = values["build"], values["prepublication"], values["graph"]
    W.require(
        graph.get("valid") is True
        and pre["source_sha"] == build["source_sha"]
        and pre["image_digest"] == build["image_digest"] == manifest["oci_digest"]
        and graph["image_index_digest"] == manifest["oci_digest"]
        and graph["image_manifest_digest"] == manifest["amd64_manifest"]
        and graph["image_config_digest"] == manifest["config_digest"]
        and graph["archive_sha256"]
        == build["archive_sha256"]
        == pre["archive_sha256"]
        == manifest["prepublication"]["archive_sha256"],
        "retained_build_graph",
    )
    inspection = values["inspection"]
    W.require(
        inspection["config_digest"] == graph["image_config_digest"]
        and [row["diff_id"] for row in inspection["layers"]]
        == graph["verified_layer_diff_ids"]
        and bridge.get("ordered_layers") == inspection["layers"],
        "retained_ordered_layers",
    )


def _loaded_labels(loaded, values, bridge):
    W.require(
        isinstance(loaded, list) and len(loaded) == 1 and isinstance(loaded[0], dict),
        "retained_loaded_config",
    )
    local, build, inspection = (
        values["local_binding"],
        values["build"],
        values["inspection"],
    )
    W.require(
        loaded[0].get("Id") == local.get("local_image_id")
        and local.get("archive_sha256") == build["archive_sha256"]
        and local.get("inspection_sha256") == inspection["inspection_sha256"]
        and local.get("image_digest") == build["image_digest"]
        and local.get("config_digest") == inspection["config_digest"],
        "retained_local_config_binding",
    )
    labels = loaded[0].get("Config", {}).get("Labels")
    W.require(
        isinstance(labels, dict)
        and labels == bridge.get("original_labels")
        and labels.get("org.opencontainers.image.revision") == build["source_sha"]
        and labels.get("npa.source_revision") == build["source_sha"],
        "retained_original_labels",
    )


def _build_provenance(values, bridge):
    build = values["build"]
    metadata = values["metadata"]
    W.require(
        metadata.get("containerimage.digest") == build["image_digest"]
        and metadata.get("buildx.build.provenance") == bridge.get("build_provenance"),
        "retained_build_provenance",
    )
    provenance = metadata["buildx.build.provenance"]
    arguments = provenance.get("invocation", {}).get("parameters", {}).get("args", {})
    W.require(
        arguments.get("build-arg:SOURCE_SHA") == build["source_sha"],
        "retained_build_arguments",
    )
    driver = values["driver"]
    W.require(
        type(driver.get("returncode")) is int
        and driver["returncode"] == 0
        and driver.get("source_sha") == build["source_sha"]
        and driver.get("image_digest") == build["image_digest"]
        and driver.get("archive_sha256") == build["archive_sha256"],
        "retained_original_build_execution",
    )


def _execution_bindings(bridge, root, manifest):
    expected = _execution_hashes(manifest)
    _verify_executions(bridge, root, manifest, expected)


def _execution_hashes(manifest):
    expected = {
        "conversion": {
            "candidate": manifest["qualification_controls"][
                "candidate_image_receipt_sha256"
            ],
            "execution": manifest["qualification_controls"][
                "conversion_execution_receipt_sha256"
            ],
            "output": manifest["conversion"]["report_sha256"],
        },
        "runtime": {
            "runtime": manifest["rtx_proof"]["runtime_image_attestation_sha256"],
            "readback": manifest["rtx_proof"]["complete_readback_receipt_sha256"],
            "output": manifest["rtx_proof"]["final_report_sha256"],
        },
        "hosted": {
            "freeze": manifest["rtx_proof"]["visual_review"]["control_manifest_sha256"],
            "requests_responses": manifest["rtx_proof"]["visual_review"][
                "raw_transport_manifest_sha256"
            ],
            "output": manifest["rtx_proof"]["visual_review"]["final_result_sha256"],
        },
    }
    return expected


def _verify_executions(bridge, root, manifest, expected):
    executions = bridge.get("executions", {})
    W.require(set(executions) == set(expected), "retained_execution_population")
    for role, hashes in expected.items():
        record = executions[role]
        W.require(
            re.fullmatch(r"[0-9a-f]{40}", str(record.get("source_commit", "")))
            and record.get("artifact_hashes") == hashes,
            "retained_execution_identity",
        )
        original = receipts.bound_json(root, record.get("execution_receipt"))
        field = record.get("source_field")
        _execution_source(original, field, record["source_commit"])
        receipts.bound_file(root, record.get("independent_review"))
    W.require(
        executions["conversion"]["source_commit"] == manifest["development_sha"],
        "retained_conversion_producer",
    )


def _execution_source(original, field, commit):
    if field == "source_before.head":
        before, after = (
            original.get("source_before", {}),
            original.get("source_after", {}),
        )
        W.require(
            before == after
            and before.get("head") == commit
            and before.get("dirty") == ""
            and before.get("tree")
            == _git("rev-parse", commit + "^{tree}").decode().strip(),
            "retained_execution_source",
        )
    else:
        W.require(
            field in {"head", "source_sha", "execution_sha", "source_commit"}
            and original.get(field) == commit,
            "retained_execution_source",
        )


def verify(manifest, root, evidence_root):
    """Verify a versioned retained producer/current consumer join, without scans.

    Args:
        manifest: Proposed aggregate with explicit compatibility binding.
        root: Protected analysis root.
        evidence_root: Protected qualification directory containing the bridge.
    Returns:
        Exact producer/consumer/bridge summary for statement and final review.
    Raises:
        ValueError, OSError: Any closure, HEAD, import, dirty or receipt check fails.
    """
    contract = manifest.get("retained_compatibility")
    W.require(
        isinstance(contract, dict) and contract.get("format") == FORMAT,
        "retained_compatibility_format",
    )
    path = evidence_root / BRIDGE_PATH
    W.require(
        process.file_sha(path) == contract.get("bridge_sha256"), "retained_bridge_hash"
    )
    bridge = W.bound_json(P.binding(path))
    W.require(bridge.get("format") == FORMAT, "retained_bridge_format")
    _source_closure(bridge, contract, manifest)
    _build_bindings(bridge, root, manifest)
    _execution_bindings(bridge, root, manifest)
    return dict(contract)


def _source_closure(bridge, contract, manifest):
    consumer = identity(contract.get("consumer_commit"))
    W.require(
        process.committed_source(consumer["commit"]) == consumer["context_sha256"],
        "retained_consumer_checkout",
    )
    producer = identity(manifest["development_sha"])
    W.require(
        bridge.get("producer") == producer
        and bridge.get("consumer") == consumer
        and contract.get("producer_commit") == producer["commit"]
        and contract.get("consumer_tree") == consumer["tree"],
        "retained_source_closure",
    )
    W.require(
        producer["copy_inputs"] == consumer["copy_inputs"]
        and producer["recipe_inputs"] == consumer["recipe_inputs"],
        "retained_image_inputs_changed",
    )
    W.require(
        bridge.get("reviewed_host_deltas") == _deltas(producer, consumer),
        "retained_host_deltas",
    )
