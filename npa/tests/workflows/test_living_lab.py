"""Unit coverage for the living-lab parameterized neural-reconstruction twin.

The topology is derived from capture x sector inputs (default 16 = 8 x 2, also
24 = 8 x 3), and the fan-out proof is size-neutral. No S3, GPU, NGC, or HF
needed: the join and zone model are pure logic, and the fan-out spec is
asserted for real-component / GPU-routing invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workflows import living_lab

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "living-lab-nurec-fanout.yaml"
)


@pytest.fixture()
def local_storage(monkeypatch, tmp_path: Path):
    root = tmp_path / "s3"

    def _local(uri: str) -> Path:
        return root / uri[len("s3://") :] if uri.startswith("s3://") else Path(uri)

    def fake_download_json(uri: str):
        path = _local(uri)
        if not path.is_file():
            raise FileNotFoundError(uri)
        return json.loads(path.read_text(encoding="utf-8"))

    def fake_upload_file(local: Path, uri: str):
        dest = _local(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local).read_bytes())
        return uri

    def fake_upload_bytes(payload: bytes, uri: str):
        dest = _local(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return uri

    def fake_download_path(uri: str, tmp: str):
        path = _local(uri)
        if path.is_dir():
            imgs = sorted(
                p for p in path.rglob("*") if p.suffix in (".png", ".jpg", ".jpeg")
            )
            if not imgs:
                raise FileNotFoundError(uri)
            return imgs[0]
        if not path.is_file():
            raise FileNotFoundError(uri)
        return path

    monkeypatch.setattr(living_lab, "_download_json", fake_download_json)
    monkeypatch.setattr(living_lab, "_upload_file", fake_upload_file)
    monkeypatch.setattr(living_lab, "_upload_bytes", fake_upload_bytes)
    monkeypatch.setattr(living_lab, "_download_path", fake_download_path)
    return root


def _zone_identity(zone_name: str) -> tuple[str, str]:
    """Derive (scene, variant) from a zone name like ``huerstholz-auto-a``."""
    parts = zone_name.split("-")
    return parts[0], parts[1]


def _write_zone(
    root: Path,
    zone_name: str,
    *,
    gpu: str = "RTX PRO 6000",
    gpu_uuid: str | None = None,
    node_name: str | None = None,
    metrics: dict[str, float] | None = None,
    provenance: dict[str, str] | None = None,
    started_epoch: int = 1000,
    ended_epoch: int = 2000,
    metrics_path: str = "s3://bucket/living-lab/zones/{zone}/reconstruction/val/metrics.yaml",
    include_all_metrics: bool = True,
) -> None:
    path = (
        root / "bucket/living-lab/zones" / zone_name / living_lab.ZONE_MANIFEST_FILENAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scene, variant = _zone_identity(zone_name)
    default_metrics = {"test/psnr": 31.19, "test/ssim": 0.833, "test/lpips": 0.267}
    payload_metrics = metrics if metrics is not None else default_metrics
    if not include_all_metrics:
        payload_metrics = {
            k: v for k, v in payload_metrics.items() if k != "test/lpips"
        }
    mt = metrics_path.format(zone=zone_name)
    payload = {
        "schema": "npa.living_lab.zone_manifest.v1",
        "zone_name": zone_name,
        "status": "ok",
        "provenance": provenance
        if provenance is not None
        else {"scene": scene, "variant": variant},
        "gpu_uuid": gpu_uuid if gpu_uuid is not None else f"GPU-{zone_name}",
        "gpu_name": gpu,
        "node_name": node_name if node_name is not None else f"node-{zone_name}",
        "pod_name": f"pod-{zone_name}",
        "usdz_path": f"s3://bucket/living-lab/zones/{zone_name}/reconstruction/last.usdz",
        "reconstruction_uri": f"s3://bucket/living-lab/zones/{zone_name}/reconstruction/",
        "novel_views_uri": f"s3://bucket/living-lab/zones/{zone_name}/novel_views/",
        "started_epoch": started_epoch,
        "ended_epoch": ended_epoch,
        "metrics_path": mt,
        "metrics": payload_metrics,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    # a real, openable preview frame for the panorama
    png = path.parent / "novel_views/frame.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (32, 32), (int(120), int(80), int(40))).save(png)


def test_living_lab_zones_are_exactly_16_and_deterministic() -> None:
    zones = living_lab.living_lab_zones()
    assert len(zones) == 16
    names = [z["zone_name"] for z in zones]
    assert len(set(names)) == 16
    # every real scene x variant appears twice (view sectors a and b)
    from collections import Counter

    seq_counts = Counter(f"{z['scene']}-{z['variant']}" for z in zones)
    assert all(count == 2 for count in seq_counts.values())
    assert len(seq_counts) == 8
    # deterministic ordering
    assert living_lab.living_lab_zones() == zones


def test_all_zones_have_novel_non_zero_render_offsets() -> None:
    for zone in living_lab.living_lab_zones():
        trans = tuple(float(v) for v in str(zone["rig_translation_offset"]).split(","))
        rot = tuple(float(v) for v in str(zone["rig_rotation_offset"]).split(","))
        assert any(trans), zone["zone_name"]
        assert any(rot) or any(v != 0.0 for v in trans), zone["zone_name"]


def test_zone_uris_are_run_scoped() -> None:
    uris = living_lab.zone_uris(
        run_uri="s3://bucket/living-lab/run-1", zone_name="toro-auto-a"
    )
    assert uris["zone_uri"] == "s3://bucket/living-lab/run-1/zones/toro-auto-a/"
    assert uris["manifest_uri"].endswith(living_lab.ZONE_MANIFEST_FILENAME)
    assert uris["rrd_uri"].endswith("reports/sim2real.rrd")


def test_zone_names_defaults_to_16(local_storage) -> None:
    assert len(living_lab.zone_names()) == 16
    assert living_lab.zone_names("a,b") == ["a", "b"]


def test_join_merges_all_16_zones(local_storage) -> None:
    for zone in living_lab.living_lab_zones():
        _write_zone(local_storage, zone["zone_name"])

    report = living_lab.join_living_lab_zones(
        zones_uri="s3://bucket/living-lab/zones/",
        report_uri="s3://bucket/living-lab/reports/",
        panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        run_id="run-1",
    )

    assert report["schema"] == living_lab.DIGITAL_TWIN_SCHEMA
    assert report["zone_count"] == 16
    assert report["joined_zones"] == 16
    assert report["missing_zones"] == []
    # Sixteen distinct GPU UUIDs / nodes, all materially overlapping.
    assert report["distinct_gpu_uuid_count"] == 16
    assert report["distinct_node_count"] == 16
    assert report["concurrency"]["required_device_count"] == 16
    assert report["concurrency"]["all_required_overlap"] is True
    assert report["aggregate_metrics"]["test/ssim_mean"] == 0.833
    assert report["aggregate_metrics"]["test/lpips_mean"] == 0.267
    assert report["aggregate_metrics"]["test/psnr_mean"] == 31.19
    assert report["panorama"]["cells"] == 16
    assert report["panorama"]["panorama_uri"].endswith("panorama.png")
    written = json.loads(
        (local_storage / "bucket/living-lab/reports/digital_twin.json").read_text()
    )
    assert len(written["zones"]) == 16
    for zone in written["zones"]:
        assert zone["gpu_uuid"]
        assert zone["node_name"]
        assert zone["metrics_path"]
        assert zone["provenance"]["scene"]
        assert zone["metrics"]["test/psnr"] > 0.0


def test_join_fails_when_a_zone_is_missing(local_storage) -> None:
    for zone in living_lab.living_lab_zones()[:15]:
        _write_zone(local_storage, zone["zone_name"])

    with pytest.raises(RuntimeError, match="1 of 16 zones missing"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_requires_real_gpu_uuid(local_storage) -> None:
    for zone in living_lab.living_lab_zones():
        _write_zone(local_storage, zone["zone_name"], gpu_uuid="", node_name="")
    with pytest.raises(RuntimeError, match="16 of 16 zones missing"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_fails_when_validation_metrics_missing(local_storage) -> None:
    for i, zone in enumerate(living_lab.living_lab_zones()):
        _write_zone(
            local_storage,
            zone["zone_name"],
            metrics={"test/psnr": None, "test/ssim": None, "test/lpips": None},
        )
    with pytest.raises(RuntimeError, match="16 of 16 zones missing"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_fails_on_placeholder_zero_metrics(local_storage) -> None:
    # The prior defect substituted missing NRE metrics with numeric zero; the
    # join must fail rather than accept 0.0 placeholders.
    for zone in living_lab.living_lab_zones():
        _write_zone(
            local_storage,
            zone["zone_name"],
            metrics={"test/psnr": 0.0, "test/ssim": 0.0, "test/lpips": 0.0},
        )
    with pytest.raises(RuntimeError, match="16 of 16 zones missing"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_fails_on_provenance_mismatch(local_storage) -> None:
    # A zone that fetched a capture other than the one it was defined to
    # reconstruct (e.g. every shard reverting to struktur28/auto) must fail.
    for zone in living_lab.living_lab_zones():
        _write_zone(
            local_storage,
            zone["zone_name"],
            provenance={"scene": "struktur28", "variant": "auto"},
        )
    # Only the two struktur28/auto zones can legitimately claim this provenance;
    # every other zone must be rejected for fetching a capture it did not ask for.
    with pytest.raises(RuntimeError, match="14 of 16 zones missing"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_concurrency_requires_overlap(local_storage) -> None:
    zones = living_lab.living_lab_zones()
    for i, zone in enumerate(zones):
        # Non-overlapping windows: zone i runs in [i*100, i*100+10], so there
        # is no common instant across all sixteen.
        _write_zone(
            local_storage,
            zone["zone_name"],
            started_epoch=100 * i,
            ended_epoch=100 * i + 10,
        )
    # Insufficient overlap is a hard, fail-closed condition: the report alone is
    # never success.
    with pytest.raises(RuntimeError, match="all-required-overlap proof failed"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_fails_closed_on_duplicate_gpu_uuids(local_storage) -> None:
    # All 16 zones claim the SAME non-empty GPU UUID: every per-zone manifest is
    # otherwise valid, but the device-count proof must detect that only one
    # distinct device participated and fail closed. Device count is never
    # inferred from model names.
    for zone in living_lab.living_lab_zones():
        _write_zone(local_storage, zone["zone_name"], gpu_uuid="GPU-DUPLICATE")
    with pytest.raises(RuntimeError, match="distinct GPU UUID proof failed"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_join_fails_closed_when_timestamps_missing(local_storage) -> None:
    # Zone manifests with no started/ended timestamps cannot participate in the
    # all-zone temporal-overlap proof; the join must fail closed rather than
    # report a fabricated overlap.
    for i, zone in enumerate(living_lab.living_lab_zones()):
        _write_zone(
            local_storage,
            zone["zone_name"],
            started_epoch=None,
            ended_epoch=None,
        )
    with pytest.raises(RuntimeError, match="all-required-overlap proof failed"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        )


def test_spec_has_16_gpu_shards_and_a_join() -> None:
    spec = living_lab.build_living_lab_workflow_spec()
    states = spec["states"]
    shard_names = [n for n in states if n.startswith("zone-")]
    assert len(shard_names) == 16
    group = states["living-lab-zones"]
    assert list(group["parallel"]) == shard_names
    # The expected device count is derived from the topology and cross-checked
    # against the parallel member list via parallelCount (fail-closed).
    assert spec["config"]["expected_device_count"] == "16"
    assert group["parallelCount"] == "{{config.expected_device_count}}"
    assert len(group["parallel"]) == 16
    assert group["maxConcurrency"] == "{{config.max_concurrency}}"
    assert group["next"] == "join"
    join = states["join"]
    assert join["needs"] == ["living-lab-zones"]
    assert join["terminal"] is True


def test_spec_scales_to_24_zone_topology() -> None:
    """The same generator emits a 24-zone (8 x 3) topology with a third sector."""
    spec = living_lab.build_living_lab_workflow_spec(sectors=("a", "b", "c"))
    states = spec["states"]
    shard_names = [n for n in states if n.startswith("zone-")]
    assert len(shard_names) == 24
    group = states["living-lab-zones"]
    assert list(group["parallel"]) == shard_names
    assert spec["config"]["expected_device_count"] == "24"
    assert spec["config"]["sector_count"] == "3"
    assert spec["config"]["capture_count"] == "8"
    assert group["parallelCount"] == "{{config.expected_device_count}}"
    assert len(group["parallel"]) == 24
    assert states["join"]["needs"] == ["living-lab-zones"]
    assert states["join"]["terminal"] is True


def test_living_lab_zones_parameterized_sizes() -> None:
    """8, 16, and 24 expected devices derive from capture x sector inputs."""
    # 8 captures x 1 sector = 8 zones / devices
    z8 = living_lab.living_lab_zones(sectors=("a",))
    assert len(z8) == 8
    # default 8 captures x 2 sectors = 16 zones / devices
    z16 = living_lab.living_lab_zones()
    assert len(z16) == 16
    # 8 captures x 3 sectors = 24 zones / devices
    z24 = living_lab.living_lab_zones(sectors=("a", "b", "c"))
    assert len(z24) == 24
    assert len({z["zone_name"] for z in z24}) == 24
    # every capture appears exactly sector_count times
    from collections import Counter

    per_seq = Counter(f"{z['scene']}-{z['variant']}" for z in z24)
    assert len(per_seq) == 8
    assert all(c == 3 for c in per_seq.values())


def test_24_zone_has_three_distinct_sector_offsets_per_capture() -> None:
    """For the 24-zone test, each capture has three genuinely distinct offsets.

    8 capture pairs x 3 distinct view sectors = 24 unique zone names; the three
    sectors are view sectors (a/b/c) of the same eight public captures, NOT 24
    independent physical captures.
    """
    z24 = living_lab.living_lab_zones(sectors=("a", "b", "c"))
    for scene, variant in living_lab.SCENES:
        zones = [z for z in z24 if z["scene"] == scene and z["variant"] == variant]
        assert len(zones) == 3
        offsets = {
            (z["view_sector"], z["rig_rotation_offset"], z["rig_translation_offset"])
            for z in zones
        }
        assert len(offsets) == 3, (scene, variant, offsets)
        sectors = sorted(z["view_sector"] for z in zones)
        assert sectors == ["a", "b", "c"]


def test_living_lab_zones_rejects_empty_topology() -> None:
    with pytest.raises(ValueError, match="at least one capture pair"):
        living_lab.living_lab_zones(captures=(), sectors=("a",))
    with pytest.raises(ValueError, match="at least one capture pair"):
        living_lab.living_lab_zones(captures=living_lab.SCENES, sectors=())
    with pytest.raises(ValueError, match="unknown living-lab view sector"):
        living_lab.living_lab_zones(sectors=("x",))


def test_join_merges_24_zone_topology(local_storage) -> None:
    """A 24-zone join derives expected devices from the zone list, not 16."""
    z24 = living_lab.living_lab_zones(sectors=("a", "b", "c"))
    for zone in z24:
        _write_zone(local_storage, zone["zone_name"])

    report = living_lab.join_living_lab_zones(
        zones_uri="s3://bucket/living-lab/zones/",
        report_uri="s3://bucket/living-lab/reports/",
        panorama_uri="s3://bucket/living-lab/reports/panorama.png",
        shards=[z["zone_name"] for z in z24],
        run_id="run-24",
    )
    assert report["zone_count"] == 24
    assert report["joined_zones"] == 24
    assert report["missing_zones"] == []
    assert report["distinct_gpu_uuid_count"] == 24
    assert report["concurrency"]["required_device_count"] == 24
    assert report["concurrency"]["all_required_overlap"] is True


def test_join_rejects_non_positive_expected_count(local_storage, monkeypatch) -> None:
    """A topology that resolves to zero expected zones must fail closed."""
    monkeypatch.setattr(living_lab, "zone_names", lambda *a, **k: [])
    with pytest.raises(ValueError, match="non-empty expected zone set"):
        living_lab.join_living_lab_zones(
            zones_uri="s3://bucket/living-lab/zones/",
            report_uri="s3://bucket/living-lab/reports/",
            panorama_uri="s3://bucket/living-lab/reports/panorama.png",
            shards="",
        )


def test_spec_rejects_non_positive_device_count() -> None:
    """The generator refuses a topology with fewer than one expected device."""
    with pytest.raises(ValueError, match="at least one capture pair"):
        living_lab.build_living_lab_workflow_spec(sectors=())


def test_parallelcount_contract_rejects_understated_devices(tmp_path) -> None:
    """Operators cannot weaken the proof by understating required devices.

    The generator derives ``expected_device_count`` from the zone list and wires
    it to ``parallelCount``; overriding it below the actual member count must
    fail ``load_spec`` before plan/render/submit.
    """
    from npa.orchestration.npa_workflow.spec import load_spec

    spec_yaml = tmp_path / "spec.yaml"
    spec_yaml.write_text(living_lab.living_lab_workflow_yaml(), encoding="utf-8")
    # round-trips as valid with the correct count first
    load_spec(str(spec_yaml))

    data = yaml.safe_load(spec_yaml.read_text(encoding="utf-8"))
    data["config"]["expected_device_count"] = "8"
    understated = tmp_path / "understated.yaml"
    understated.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(Exception, match="parallelCount resolves to 8"):
        load_spec(str(understated))

    # Overstating is equally a mismatch and must fail too.
    data["config"]["expected_device_count"] = "32"
    overstated = tmp_path / "overstated.yaml"
    overstated.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(Exception, match="parallelCount resolves to 32"):
        load_spec(str(overstated))


def test_every_shard_is_real_nurec_work_on_rtx_gpu() -> None:
    spec = living_lab.build_living_lab_workflow_spec()
    gpu = spec["resources"]["gpu"]
    for name, state in spec["states"].items():
        if not name.startswith("zone-"):
            continue
        assert state["resources"] == "gpu"
        shell = state["run"]["shell"]
        # Every shard runs the full real single-pod NRE pipeline inside the
        # NRE container: check -> fetch -> reconstruct -> render -> visualize
        # -> finalize, exactly like the validated reference
        # (npa/src/npa/workbench/nurec/examples/nurec-reconstruct.yaml).
        assert "npa workbench nurec check" in shell
        assert "npa workbench nurec fetch" in shell
        assert "npa workbench nurec reconstruct" in shell
        assert "npa workbench nurec render" in shell
        assert "npa workbench nurec visualize" in shell
        assert "npa workbench nurec finalize" in shell
        # One GPU per shard: never a disguised single-GPU-unaware program.
        assert "--world-size 1" in shell
        # Novel-view rendering requires a real non-zero rig offset.
        assert "--rig-translation-offset" in shell and "--rig-rotation-offset" in shell

        # --- flag-level correctness (catches CLI-flag drift that validates
        # but crashes on real submit) -------------------------------------
        # fetch -> reconstruct handoff: same pod, so --ncore-json points at the
        # local meta-file the fetch stage unpacked (never --ncore-uri, which
        # expects an S3 *published sequence* that fetch --publish-sequence writes).
        assert "--ncore-json" in shell and "--ncore-uri" not in shell
        # reconstruct needs the derived rig pose group + reference camera.
        assert "--poses-component-group" in shell
        assert "--camera-id" in shell
        # render must target the trained .usdz artifact, not an S3 dir, and
        # must pass --camera-id (required by `nre render`).
        assert "--artifact-path" in shell
        assert shell.count("--camera-id") == 2  # reconstruct + render require it

        # The zone run root must expose input/, reconstruction/, novel_views/
        # for visualize, so it uses the zone prefix ""+ ZU, not a sub-dir.
        assert 'visualize --input-uri "${ZU}"' in shell

        # The NRE container ships no npa/ffmpeg/runtime deps: the shard must
        # install them (ffmpeg + nvidia-ncore + rerun-sdk + npa guard).
        assert "ffmpeg" in shell
        assert "nvidia-ncore" in shell
        assert "rerun-sdk" in shell
        assert "command -v npa" in shell

        # --- defect regression guard: the fetch must pass the exact dataset,
        # scene and variant the zone was defined to reconstruct, and must fail
        # closed on any provenance mismatch (previously every shard silently
        # reverted to the default struktur28/auto capture). ---
        assert "nurec fetch --dataset" in shell
        assert "--scene" in shell
        assert "--variant" in shell
        assert "PROVENANCE MISMATCH" in shell
        # The gate validates against independently observed unpacked content
        # (observed_scene/observed_variant), not the echoed request args.
        assert "observed_scene" in shell
        assert "observed_variant" in shell
        assert "observed content" in shell

        # Manifest writer is a child python process: the shell vars it reads
        # must be exported.
        assert "export ZONE" in shell
        assert "export GPU_UUID" in shell
        assert "GPU_UUID GPU_NAME POD_NAME NODE_NAME START" in shell
        assert "export USDZ" in shell
    assert gpu["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1", (
        "shard must route to RTX PRO 6000"
    )


def test_no_stub_toolrefs_and_all_shells_are_real() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    for name, state in spec["states"].items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref in TOOL_CATALOG
            assert TOOL_CATALOG[tool_ref].stub is False, name
        run = state.get("run")
        if not run:
            continue
        command = str(run.get("shell", "")) or " ".join(
            str(item) for item in run.get("argv", [])
        )
        assert "npa workbench" in command or "living_lab" in command, (
            f"state '{name}' run is not a real command/module call"
        )


def test_committed_yaml_matches_generator() -> None:
    assert SPEC_PATH.read_text() == living_lab.living_lab_workflow_yaml()


def test_shard_shell_runs_full_pipeline_and_writes_manifest(tmp_path) -> None:
    """Execute a resolved zone shard shell end-to-end with stubbed tools.

    Guards the *operational* correctness the flag checks cannot: the shard must
    actually install + guard runtime deps, run every real nurec verb in order,
    export the shell vars a child python process reads, and publish a
    load-bearing zone_manifest.json — not crash on os.environ KeyError.
    """
    import os
    import subprocess
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    boost = tmp_path / "py"

    def write(name: str, content: str) -> None:
        p = bin_dir / name
        p.write_text(content, encoding="utf-8")
        p.chmod(0o755)

    write(
        "nvidia-smi",
        "#!/bin/sh\ncase \"$*\" in\n  *uuid*) echo 'GPU-00000000-1111-2222-3333-444444444444' ;;\n  *) echo 'NVIDIA RTX PRO 6000 Blackwell' ;;\nesac\n",
    )
    write("ffmpeg", "#!/bin/sh\nexit 0\n")
    write(
        "npa",
        """#!/bin/sh
fetch_prov() {
  dataset_id=""; scene=""; variant=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dataset) dataset_id="$2"; shift 2 ;;
      --scene) scene="$2"; shift 2 ;;
      --variant) variant="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  # Simulate a *correct* fetch: the observed unpacked scene_dir content matches
  # the requested scene/variant (observed_variant derives the real dir layout).
  observed_variant="standard"
  case "$variant" in auto) observed_variant="auto" ;; esac
  echo '{"status":"ok","dataset_id":"'"$dataset_id"'","scene":"'"$scene"'","variant":"'"$variant"'","observed_scene":"'"$scene"'","observed_variant":"'"$observed_variant"'","scene_dir":"'"$scene"'","ncore_json":"/tmp/n.json","poses_component_group":"npa_rig","reference_camera":"cam1"}'
}
case "$*" in
  *"nurec check"*) echo '{"status":"ok","has_rt_cores":true}' ;;
  *"nurec fetch"*) fetch_prov "$@" ;;
  *"nurec reconstruct"*) echo '{"status":"ok","usdz_path":"/tmp/last.usdz","metrics_path":"/tmp/npa-nurec-out/nre/val/metrics.yaml","metrics":{"test/psnr":31.19,"test/ssim":0.833,"test/lpips":0.267}}' ;;
  *"nurec render"*) echo '{"status":"ok"}' ;;
  *"nurec visualize"*) echo '{"status":"completed"}' ;;
  *"nurec finalize"*) echo '{"status":"ok","has_usdz":true,"has_rrd":true,"artifact_count":42}' ;;
  *) echo '{}' ;;
esac
""",
    )
    # Intercept only `python3 -m pip ...` (a no-op) so the setup's dep install
    # passes through; everything else runs the real interpreter.
    real_py = sys.executable
    write(
        "python3",
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 0; fi\n'
        f'exec "{real_py}" "$@"\n',
    )

    (boost / "npa").mkdir(parents=True)
    (boost / "npa" / "clients").mkdir(parents=True)
    (boost / "npa" / "__init__.py").write_text("", encoding="utf-8")
    (boost / "npa" / "clients" / "__init__.py").write_text("", encoding="utf-8")
    (boost / "npa" / "clients" / "storage.py").write_text(
        "class StorageClient:\n"
        "    @staticmethod\n"
        "    def from_environment():\n"
        "        return StorageClient()\n"
        "    def upload_file(self, local, uri):\n"
        "        print(f'UPLOAD {local} -> {uri}')\n"
        "        return uri\n",
        encoding="utf-8",
    )

    shell = living_lab.build_living_lab_workflow_spec()["states"][
        "zone-toro-standard-b"
    ]["run"]["shell"]
    zone_cfg = {
        "config.zone_name": "toro-standard-b",
        "config.dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "config.scene": "toro",
        "config.variant": "standard",
        "config.run_prefix_uri": "s3://bucket/prefix/",
        "config.nurec_image": "nvcr.io/nvidia/nre/nre-ga:26.04",
        "config.rig_translation_offset": "0,0,0",
        "config.rig_rotation_offset": "120,0,0",
    }
    for tok, val in zone_cfg.items():
        shell = shell.replace("{{" + tok + "}}", val)
    script = tmp_path / "shard.sh"
    script.write_text(shell, encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYTHONPATH"] = str(boost)
    env["AWS_ACCESS_KEY_ID"] = "TESTKEY"
    env["AWS_SECRET_ACCESS_KEY"] = "TESTSECRET"
    proc = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "UPLOAD" in proc.stdout
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["zone_name"] == "toro-standard-b"
    assert payload["status"] == "ok"
    assert "RTX PRO 6000" in payload["gpu_name"]
    assert payload["usdz_path"] == "/tmp/last.usdz"
    assert payload["finalize"]["has_usdz"] is True
    assert payload["finalize"]["artifact_count"] == 42
    assert payload["metrics"]["test/ssim"] == 0.833
    assert payload["metrics"]["test/psnr"] == 31.19
    assert payload["metrics"]["test/lpips"] == 0.267
    # Provenance must reflect what was requested and actually fetched.
    assert payload["provenance"]["dataset_id"] == "nvidia/PhysicalAI-NuRec-PPISP"
    assert payload["provenance"]["scene"] == "toro"
    assert payload["provenance"]["variant"] == "standard"
    # The shard records what it was asked to fetch and proves it equals the
    # actual fetched capture.
    assert payload["provenance"]["requested"] == {
        "DATASET_ID": "nvidia/PhysicalAI-NuRec-PPISP",
        "SCENE": "toro",
        "VARIANT": "standard",
    }
    assert payload["metrics_path"].endswith("metrics.yaml")
    assert payload["gpu"]["gpu_uuid"].startswith("GPU-")
    assert payload["gpu"]["node_name"] == "unknown"  # NODE_NAME unset outside k8s


def test_provenance_gate_catches_wrong_content_echoing_requested_labels(
    tmp_path: Path,
) -> None:
    """The real embedded shard provenance gate fails closed on wrong content.

    Simulates a fetch that *echoes* the requested dataset/scene/variant in its
    top-level fields (a mere copy of the request arguments) while the
    independently observed unpacked content disagrees. The gate must reject it.
    """
    import os
    import subprocess
    import sys

    shell = living_lab.build_living_lab_workflow_spec()["states"][
        "zone-toro-standard-b"
    ]["run"]["shell"]
    zone_cfg = {
        "config.zone_name": "toro-standard-b",
        "config.dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "config.scene": "toro",
        "config.variant": "standard",
        "config.run_prefix_uri": "s3://bucket/prefix/",
        "config.nurec_image": "nvcr.io/nvidia/nre/nre-ga:26.04",
        "config.rig_translation_offset": "0,0,0",
        "config.rig_rotation_offset": "120,0,0",
    }
    for tok, val in zone_cfg.items():
        shell = shell.replace("{{" + tok + "}}", val)

    # Extract the provenance gate heredoc body (between the <<'PY' and PY lines).
    marker = "<<'PY'"
    start = shell.index(marker) + len(marker)
    end = shell.index("\nPY", start)
    gate_src = shell[start:end]
    assert "observed_scene" in gate_src

    gate = tmp_path / "gate.py"
    gate.write_text(gate_src, encoding="utf-8")

    fetch_json = tmp_path / "fetch.json"
    fetch_json.write_text(
        json.dumps(
            {
                "status": "ok",
                # Echoed request args (the echo a faulty fetch would produce) ...
                "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
                "scene": "toro",
                "variant": "standard",
                # ... but independently observed content is wrong.
                "observed_scene": "struktur28",
                "observed_variant": "standard",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["ZONE"] = "toro-standard-b"
    env["NUREC_FETCH_JSON"] = str(fetch_json)
    env["DATASET_ID"] = "nvidia/PhysicalAI-NuRec-PPISP"
    proc = subprocess.run(
        [sys.executable, str(gate), "toro", "standard"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, "gate must fail closed on wrong content"
    assert "PROVENANCE MISMATCH" in proc.stderr


def test_provenance_gate_passes_when_observed_matches(tmp_path: Path) -> None:
    """The same embedded gate accepts a fetch whose observed content matches."""
    import os
    import subprocess
    import sys

    shell = living_lab.build_living_lab_workflow_spec()["states"][
        "zone-toro-standard-b"
    ]["run"]["shell"]
    zone_cfg = {
        "config.zone_name": "toro-standard-b",
        "config.dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "config.scene": "toro",
        "config.variant": "standard",
        "config.run_prefix_uri": "s3://bucket/prefix/",
        "config.nurec_image": "nvcr.io/nvidia/nre/nre-ga:26.04",
        "config.rig_translation_offset": "0,0,0",
        "config.rig_rotation_offset": "120,0,0",
    }
    for tok, val in zone_cfg.items():
        shell = shell.replace("{{" + tok + "}}", val)

    marker = "<<'PY'"
    start = shell.index(marker) + len(marker)
    end = shell.index("\nPY", start)
    gate = tmp_path / "gate.py"
    gate.write_text(shell[start:end], encoding="utf-8")

    fetch_json = tmp_path / "fetch.json"
    fetch_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
                "scene": "toro",
                "variant": "standard",
                "observed_scene": "toro",
                "observed_variant": "standard",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["ZONE"] = "toro-standard-b"
    env["NUREC_FETCH_JSON"] = str(fetch_json)
    env["DATASET_ID"] = "nvidia/PhysicalAI-NuRec-PPISP"
    proc = subprocess.run(
        [sys.executable, str(gate), "toro", "standard"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "provenance-ok" in proc.stdout
