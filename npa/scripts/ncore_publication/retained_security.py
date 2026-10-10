"""Authenticate retained scratch-image Trivy omissions and supplemental coverage."""

from datetime import datetime
import ast
import hashlib
from pathlib import Path

from image_byte_scan import core as W, prepare as P
from npa.deploy.publish_public import _TRIVY_CONTAINER_IMAGE
from npa.deploy.ncore_acceptance import validate_selected_base_scan
from npa.deploy.ncore_component_scan import coverage_failures
from npa.deploy.ncore_component_inventory import _canonical
from npa.deploy.ncore_selected_sbom import _verified_counts

from . import process, retained_receipts as receipts

FORMAT = "npa_ncore_retained_trivy_provenance_v1"
SOURCES = (
    "npa/scripts/ncore_publication/gates.py",
    "npa/scripts/ncore_publication/process.py",
    "npa/src/npa/deploy/publish_public.py",
    "npa/scripts/publish_ncore_oci.py",
    "npa/scripts/ncore_publication/cli.py",
    "npa/scripts/ncore_publication/__init__.py",
)


def counts(payload, *, omission_authenticated=False):
    """Count every finding, accepting omission only after provenance verification.

    Args:
        payload: Original Trivy JSON object.
        omission_authenticated: Internal result of the retained provenance adapter.
    Returns:
        Exact critical fixed/unfixed and secret counts, not supplemental totals.
    Raises:
        ValueError: Rows/findings are malformed or omission is unauthenticated.
    """
    if "Results" not in payload:
        W.require(omission_authenticated is True, "acceptance_trivy_results")
        results = []
    else:
        results = payload["Results"]
    W.require(isinstance(results, list), "acceptance_trivy_results")
    total = fixed = secrets = 0
    for result in results:
        W.require(isinstance(result, dict), "acceptance_trivy_result")
        secrets += len(_findings(result, "Secrets"))
        for finding in _findings(result, "Vulnerabilities"):
            severity = finding.get("Severity")
            W.require(
                severity in {"UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"},
                "acceptance_trivy_severity",
            )
            version = finding.get("FixedVersion", "")
            W.require(isinstance(version, str), "acceptance_trivy_fixed_version")
            total += severity == "CRITICAL"
            fixed += severity == "CRITICAL" and bool(version.strip())
    return {
        "critical_total": total,
        "critical_with_fix": fixed,
        "critical_unfixed": total - fixed,
        "secrets": secrets,
    }


def _findings(result, field):
    values = result.get(field, [])
    W.require(
        isinstance(values, list) and all(isinstance(row, dict) for row in values),
        "acceptance_trivy_findings",
    )
    return values


def _driver(provenance, root, manifest):
    driver = receipts.bound_json(root, provenance.get("driver"))
    W.require(
        type(driver.get("returncode")) is int
        and driver["returncode"] == 0
        and driver.get("source_sha") == manifest["development_sha"]
        and driver.get("image_digest") == manifest["oci_digest"]
        and driver.get("archive_sha256")
        == manifest["prepublication"]["archive_sha256"],
        "retained_trivy_successful_driver",
    )
    start = datetime.fromisoformat(driver["started_at"])
    end = datetime.fromisoformat(driver["completed_at"])
    W.require(
        start.tzinfo is not None and end.tzinfo is not None and end > start,
        "retained_trivy_completed_process",
    )
    return _driver_options(driver, provenance, manifest)


def _driver_options(driver, provenance, manifest):
    argv = driver.get("argv")
    W.require(
        isinstance(argv, list)
        and len(argv) > 3
        and argv[1:3] == ["npa/scripts/publish_ncore_oci.py", "check"]
        and all(isinstance(arg, str) for arg in argv),
        "retained_trivy_driver_argv",
    )
    options = argv[3:]
    W.require(
        len(options) % 2 == 0 and len(set(options[::2])) == len(options[::2]),
        "retained_trivy_driver_options",
    )
    options = dict(zip(options[::2], options[1::2]))
    required = {
        "--source-sha",
        "--analysis-root",
        "--output-dir",
        "--annex",
        "--native-source",
        "--metadata",
        "--bootstrap-source",
    }
    allowed = required | {"--keyring", "--policy-mode", "--literal-inventory"}
    W.require(required <= set(options) <= allowed, "retained_trivy_original_options")
    W.require(
        options.get("--source-sha") == manifest["development_sha"]
        and isinstance(options.get("--output-dir"), str),
        "retained_trivy_driver_source",
    )
    W.require(provenance.get("driver_argv") == argv, "retained_trivy_original_argv")
    W.require(
        argv[0].endswith("/npa/.venv/bin/python")
        and str(Path(argv[0]).parents[3]) == provenance.get("original_checkout"),
        "retained_trivy_original_checkout",
    )
    return options["--output-dir"]


def _commands(directory, checkout):
    common = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{directory}:/evidence:ro",
        "--volume",
        f"{checkout}/.trivyignore:/policy:ro",
        _TRIVY_CONTAINER_IMAGE,
        "image",
        "--input",
        "/evidence/inspection.tar",
        "--scanners",
        "vuln,secret,license",
    ]
    return {key: common + flags for key, flags in _scanner_flags().items()}


def _scanner_flags():
    return {
        "policy": [
            "--ignorefile",
            "/policy",
            "--ignore-unfixed",
            "--severity",
            "CRITICAL",
            "--exit-code",
            "1",
            "--format",
            "json",
        ],
        "all": [
            "--ignorefile",
            "/dev/null",
            "--ignore-unfixed=false",
            "--severity",
            "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL",
            "--exit-code",
            "0",
            "--format",
            "json",
        ],
    }


def _identity(payload, graph):
    metadata = payload.get("Metadata")
    W.require(
        payload.get("SchemaVersion") == 2
        and payload.get("ArtifactType") == "container_image"
        and payload.get("Trivy") == {"Version": "0.72.0"}
        and payload.get("ArtifactID") == graph["image_config_digest"]
        and isinstance(metadata, dict)
        and metadata.get("ImageID") == graph["image_config_digest"]
        and metadata.get("DiffIDs") == graph["verified_layer_diff_ids"],
        "retained_trivy_external_image_identity",
    )


def _supplements(root, provenance, manifest, graph):
    records = provenance.get("supplements", {})
    required = {
        "selected",
        "selected_sbom",
        "selected_report",
        "components",
        "component_inventory",
        "licenses",
        "source_delivery",
        "source_inventory",
        "sbom",
        "base_lock",
        "source_lock",
    }
    W.require(set(records) == required, "retained_trivy_supplement_population")
    paths = {name: receipts.bound_file(root, row) for name, row in records.items()}

    def read(name):
        return W.bound_json(P.binding(paths[name]))

    selected, components = read("selected"), read("components")
    _supplement_identities(selected, components, manifest, graph)
    _supplement_hashes(records, selected, components, manifest, read)


def _supplement_identities(selected, components, manifest, graph):
    W.require(selected == manifest["selected_base_scan"], "retained_selected_receipt")
    validate_selected_base_scan(manifest)
    for field in ("image_digest", "platform_digest", "config_digest", "rootfs_sha256"):
        W.require(selected[field] == components.get(field), "retained_component_image")
    W.require(
        selected["rootfs_sha256"] == graph["verified_layer_diff_ids"][0][7:]
        and components.get("status") == "pass"
        and components.get("failures") == []
        and components.get("licenses", {}).get("missing_license_scope") == []
        and components["licenses"].get("delivery_failures") == [],
        "retained_component_outcome",
    )


def _supplement_hashes(records, selected, components, manifest, read):
    for field, name in (
        ("sbom_sha256", "selected_sbom"),
        ("report_sha256", "selected_report"),
        ("lock_sha256", "base_lock"),
    ):
        W.require(
            selected[field] == records[name]["sha256"], "retained_selected_binding"
        )
    population = read("component_inventory")
    W.require(
        all(
            selected.get(key) == value
            for key, value in _verified_counts(read("selected_sbom")).items()
        ),
        "retained_selected_sbom_population",
    )
    W.require(
        components.get("inventory_sha256") == records["component_inventory"]["sha256"]
        and components.get("population_sha256")
        == population.get("population_sha256")
        == hashlib.sha256(_canonical(population["components"])).hexdigest()
        and components["licenses"].get("report_sha256")
        == records["licenses"]["sha256"],
        "retained_component_binding",
    )
    _supplement_population(manifest, records, selected, components, population, read)


def _supplement_population(manifest, records, selected, components, population, read):
    W.require(
        population.get("source_sha") == manifest["development_sha"]
        and population.get("base_lock_sha256") == records["base_lock"]["sha256"]
        and population.get("source_lock_sha256") == records["source_lock"]["sha256"]
        and not coverage_failures(
            population, components["evaluations"], components["licenses"]
        ),
        "retained_component_coverage",
    )
    pre = manifest["prepublication"]
    for name, field in (
        ("selected", "selected_base_receipt_sha256"),
        ("components", "component_receipt_sha256"),
        ("sbom", "provenance_sbom_sha256"),
        ("source_delivery", "source_delivery_receipt_sha256"),
    ):
        W.require(
            records[name]["sha256"] == pre[field], "retained_supplement_prepublication"
        )
    W.require(
        records["source_inventory"]["sha256"]
        == manifest["source"]["post_patch_inventory_sha256"]
        and records["base_lock"]["sha256"] == manifest["source"]["lock_sha256"],
        "retained_source_delivery_binding",
    )
    W.require(
        counts(read("selected_report"))
        == {
            key: selected[key]
            for key in (
                "critical_total",
                "critical_with_fix",
                "critical_unfixed",
                "secrets",
            )
        },
        "retained_selected_findings",
    )


def authenticated_counts(manifest, root, gate_dir):
    """Require original successful execution and complete same-image supplements.

    Args:
        manifest: Aggregate preserving original image and scan identities.
        root: Protected analysis root containing original receipts.
        gate_dir: Hash-bound original gate directory.
    Returns:
        Original unfiltered scratch counts, separate from selected-base counts.
    Raises:
        ValueError, OSError: Provenance, process, policy or coverage is absent.
    """
    path = gate_dir / "retained-trivy-provenance.json"
    W.require(
        process.file_sha(path)
        == manifest["prepublication"].get("retained_trivy_provenance_sha256"),
        "retained_trivy_provenance_binding",
    )
    provenance = W.bound_json(P.binding(path))
    W.require(provenance.get("format") == FORMAT, "retained_trivy_format")
    directory = _driver(provenance, root, manifest)
    _source_contract(provenance, manifest, directory)
    graph = receipts.bound_json(root, provenance.get("graph"))
    W.require(
        provenance["graph"]["sha256"]
        == manifest["prepublication"]["graph_receipt_sha256"],
        "retained_trivy_graph_binding",
    )
    _graph_identity(graph, manifest)
    _supplements(root, provenance, manifest, graph)
    return _reports(provenance, root, gate_dir, graph)


def _reports(provenance, root, gate_dir, graph):
    reports = {}
    for name in ("all", "policy"):
        record = provenance.get("reports", {}).get(name)
        report = receipts.bound_json(root, record)
        W.require(
            record["sha256"] == process.file_sha(gate_dir / f"trivy-{name}.json"),
            "retained_trivy_report_binding",
        )
        _identity(report, graph)
        reports[name] = counts(report, omission_authenticated=True)
        W.require(
            reports[name]["critical_with_fix"] == 0 and reports[name]["secrets"] == 0,
            "retained_trivy_security_findings",
        )
    return reports["all"]


def _source_contract(provenance, manifest, directory):
    sources = provenance.get("sources", {})
    W.require(set(sources) == set(SOURCES), "retained_trivy_source_population")
    for path, record in sources.items():
        raw = receipts.source_file(record, path)
        W.require(
            record["commit"] == manifest["development_sha"],
            "retained_trivy_source_commit",
        )
        _unchanged_command_source(path, raw)
    checkout = provenance.get("original_checkout")
    W.require(
        isinstance(checkout, str)
        and checkout.startswith("/")
        and provenance.get("commands") == _commands(directory, checkout)
        and provenance.get("scanner_image") == _TRIVY_CONTAINER_IMAGE,
        "retained_trivy_commands",
    )
    policy = receipts.source_file(provenance.get("policy"), ".trivyignore")
    W.require(
        provenance["policy"]["commit"] == manifest["development_sha"]
        and hashlib.sha256(policy).hexdigest() == provenance.get("policy_sha256"),
        "retained_trivy_policy",
    )


def _unchanged_command_source(path, raw):
    # The private writer can evolve independently of subprocess success semantics.
    current = (process.ROOT / path).read_bytes()
    if path != SOURCES[1]:
        W.require(raw == current, "retained_trivy_command_contract")
        return
    trees = [ast.parse(value) for value in (raw, current)]
    functions = [
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        for tree in trees
    ]
    W.require(
        ast.dump(functions[0]) == ast.dump(functions[1]),
        "retained_trivy_process_contract",
    )


def _graph_identity(graph, manifest):
    W.require(
        graph.get("valid") is True
        and graph.get("archive_sha256") == manifest["prepublication"]["archive_sha256"]
        and graph.get("image_index_digest") == manifest["oci_digest"]
        and graph.get("image_manifest_digest") == manifest["amd64_manifest"]
        and graph.get("image_config_digest") == manifest["config_digest"]
        and isinstance(graph.get("verified_layer_diff_ids"), list)
        and graph["verified_layer_diff_ids"]
        and all(
            isinstance(value, str)
            and value.startswith("sha256:")
            and receipts.HASH.fullmatch(value[7:])
            for value in graph["verified_layer_diff_ids"]
        ),
        "retained_trivy_graph_identity",
    )
