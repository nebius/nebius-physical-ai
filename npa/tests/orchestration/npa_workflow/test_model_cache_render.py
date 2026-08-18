"""The renderer must send runtime-downloaded weights to durable storage."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workbench.model_cache import (
    MODEL_CACHE_HOST_PATH_ENV,
    MODEL_CACHE_PVC_ENV,
    MODEL_CACHE_VOLUME_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
NPA_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"

OPTIONS = SkypilotRenderOptions(
    registry="registry.example", materialize_registry_secrets=False
)

# Every shipped spec targets Kubernetes, so the other branch needs its own spec.
VM_CLOUD_SPEC = textwrap.dedent(
    """
    apiVersion: npa.workflow/v0.0.1
    kind: Workflow

    metadata:
      name: vm-cloud-demo

    config:
      bucket: example-bucket
      prefix: "vm/{{run.id}}"
      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/"
      captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/"
      caption_model: model-a
      max_images: "4"
      max_tokens: "64"

    resources:
      vm:
        cloud: aws
        cpus: 4
        memory: 16Gi

    initial: caption

    states:
      caption:
        description: One captioning stage on a virtual machine.
        toolRef: workbench.token_factory.caption
        resources: vm
        params:
          input-path: "{{config.images_uri}}"
          output-path: "{{config.captions_uri}}"
          model: "{{config.caption_model}}"
          max-images: "{{config.max_images}}"
          max-tokens: "{{config.max_tokens}}"
        terminal: true
    """
)


def _last_task(name: str) -> dict:
    spec = load_spec(NPA_SPECS / name)
    rendered = render_skypilot_yaml(
        spec, build_plan(spec, run_id="cache"), run_id="cache", options=OPTIONS
    )
    return [doc for doc in yaml.safe_load_all(rendered) if doc][-1]


def _last_task_from_text(text: str, tmp_path: Path) -> dict:
    path = tmp_path / "spec.yaml"
    path.write_text(text, encoding="utf-8")
    spec = load_spec(path)
    rendered = render_skypilot_yaml(
        spec, build_plan(spec, run_id="cache"), run_id="cache", options=OPTIONS
    )
    return [doc for doc in yaml.safe_load_all(rendered) if doc][-1]


@pytest.fixture(autouse=True)
def _no_inherited_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (MODEL_CACHE_PVC_ENV, MODEL_CACHE_HOST_PATH_ENV, "NPA_MODEL_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_without_configured_storage_the_rendered_task_is_unchanged() -> None:
    task = _last_task("cosmos2-transfer.yaml")

    assert "NPA_MODEL_CACHE_DIR" not in task["envs"]
    assert "HF_HOME" not in task["envs"]
    assert "npa model cache" not in task["run"]
    volumes = task["config"]["kubernetes"]["pod_config"]["spec"]["volumes"]
    assert [item["name"] for item in volumes] == ["cosmos2-model-cache"]


def test_a_configured_claim_redirects_every_weight_cache_and_mounts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_CACHE_PVC_ENV, "npa-model-cache")

    task = _last_task("cosmos2-transfer.yaml")

    envs = task["envs"]
    assert envs["NPA_MODEL_CACHE_DIR"] == "/opt/npa-model-cache"
    assert envs["HF_HOME"] == "/opt/npa-model-cache/huggingface"
    assert envs["HF_HUB_CACHE"] == "/opt/npa-model-cache/huggingface/hub"
    # Cosmos-Transfer's image reads its own variable for the same cache.
    assert envs["COSMOS_HF_CACHE"] == "/opt/npa-model-cache/huggingface"

    pod_spec = task["config"]["kubernetes"]["pod_config"]["spec"]
    assert {
        "name": MODEL_CACHE_VOLUME_NAME,
        "persistentVolumeClaim": {"claimName": "npa-model-cache"},
    } in pod_spec["volumes"]
    mounts = pod_spec["containers"][0]["volumeMounts"]
    assert {
        "name": MODEL_CACHE_VOLUME_NAME,
        "mountPath": "/opt/npa-model-cache",
    } in mounts


def test_both_setup_and_run_materialize_the_cache_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SkyPilot runs setup and run in separate shells, and setup is where a stage
    # pre-fetches weights, so the directories have to exist in both.
    monkeypatch.setenv(MODEL_CACHE_PVC_ENV, "npa-model-cache")

    task = _last_task("cosmos2-transfer.yaml")

    for field in ("setup", "run"):
        assert "mkdir -p /opt/npa-model-cache/" in task[field]
        assert "/opt/npa-model-cache/huggingface/hub" in task[field]
        assert "npa model cache: /opt/npa-model-cache" in task[field]


def test_a_cosmos3_stage_gets_the_framework_download_cache_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_CACHE_PVC_ENV, "npa-model-cache")

    task = _last_task("cosmos3-text-to-image.yaml")

    assert task["envs"]["NPA_COSMOS3_CACHE"] == "/opt/npa-model-cache/cosmos3"
    assert (
        task["envs"]["COSMOS_DOWNLOAD_CACHE_DIR"]
        == "/opt/npa-model-cache/cosmos3/downloads"
    )
    # The spec no longer pins an ephemeral --cache-dir, so the env decides where
    # the framework checkout and the Cosmos3-Nano checkpoint land. The flag is
    # dropped rather than passed empty: Typer turns an empty Path into Path("."),
    # which an already-published image would cache a multi-gigabyte checkpoint
    # into. Absent, every image version falls back to NPA_COSMOS3_CACHE.
    assert "/tmp/npa-cosmos3-cache" not in task["run"]
    assert "--cache-dir" not in task["run"]


def test_an_explicit_root_without_backing_storage_sets_env_but_mounts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An operator whose cluster config already mounts a shared filesystem only
    # needs to name it; the renderer must not invent a claim for it.
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", "/mnt/shared/weights")

    task = _last_task("cosmos2-transfer.yaml")

    assert task["envs"]["HF_HOME"] == "/mnt/shared/weights/huggingface"
    volumes = task["config"]["kubernetes"]["pod_config"]["spec"]["volumes"]
    assert [item["name"] for item in volumes] == ["cosmos2-model-cache"]
    # ...and it must not claim the weights persist, because it did not mount the
    # thing that would make that true.
    assert "weights persist across runs" not in task["run"]
    assert "not mounted by npa" in task["run"]


def test_a_claim_does_not_follow_a_stage_onto_a_cloud_with_no_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A PVC is mountable only where there is a cluster to mount it in.

    On any other cloud SkyPilot hands the stage a fresh VM. Exporting the cache
    family anyway would point HF_HOME at `/opt/npa-model-cache` with nothing there,
    and `/opt` is root-owned in the workbench images while they run unprivileged --
    so the stage would fail where it used to work.
    """

    monkeypatch.setenv(MODEL_CACHE_PVC_ENV, "npa-model-cache")
    # This stage pins no workbench image, so the renderer wants a source location
    # for the npa package: irrelevant to the cache, required to render.
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")

    task = _last_task_from_text(VM_CLOUD_SPEC, tmp_path)

    assert task["resources"]["cloud"] == "aws"
    assert "NPA_MODEL_CACHE_DIR" not in task["envs"]
    assert "HF_HOME" not in task["envs"]
    assert "npa model cache" not in task["run"]


def test_a_vm_stage_still_honors_a_root_the_operator_says_is_mounted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The escape hatch has to keep working off Kubernetes: a SkyPilot VM stage can
    # have a data disk mounted by the cluster's own setup.
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", "/mnt/data/weights")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")

    task = _last_task_from_text(VM_CLOUD_SPEC, tmp_path)

    assert task["envs"]["HF_HOME"] == "/mnt/data/weights/huggingface"


def test_a_pinned_cache_dir_is_still_passed_through(tmp_path: Path) -> None:
    """Dropping the empty spelling must not drop a value the operator set."""

    from npa.orchestration.npa_workflow.catalog import drop_empty_optional_flags

    argv = ["npa", "workbench", "cosmos3", "text-to-image", "--cache-dir", "/mnt/pinned"]

    assert drop_empty_optional_flags("workbench.cosmos3.text_to_image", argv) == argv


def test_only_the_declared_flags_are_dropped_when_empty() -> None:
    from npa.orchestration.npa_workflow.catalog import drop_empty_optional_flags

    argv = ["npa", "x", "--cache-dir", "", "--prompt", "", "--seed", "0"]

    # --prompt is not declared droppable, so an empty prompt still reaches the CLI
    # and fails there rather than silently becoming the tool's default.
    assert drop_empty_optional_flags("workbench.cosmos3.text_to_image", argv) == [
        "npa",
        "x",
        "--prompt",
        "",
        "--seed",
        "0",
    ]
