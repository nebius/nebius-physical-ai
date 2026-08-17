"""Unit tests for the durable model-weight cache contract."""

from __future__ import annotations

import pytest

from npa.workbench.model_cache import (
    DEFAULT_MODEL_CACHE_MOUNT,
    MODEL_CACHE_DIR_ENV,
    MODEL_CACHE_ENV_NAMES,
    MODEL_CACHE_HOST_PATH_ENV,
    MODEL_CACHE_LAYOUT,
    MODEL_CACHE_PVC_ENV,
    MODEL_CACHE_VOLUME_NAME,
    RUNTIME_DOCKER,
    RUNTIME_KUBERNETES,
    RUNTIME_PREMOUNTED,
    ModelCacheError,
    docker_model_cache_volumes,
    model_cache_dirs,
    model_cache_env,
    model_cache_host_path,
    model_cache_pvc,
    pod_config_with_model_cache,
    render_model_cache_shell,
    resolve_model_cache_root,
)


ALL_RUNTIMES = [RUNTIME_KUBERNETES, RUNTIME_DOCKER, RUNTIME_PREMOUNTED]


@pytest.mark.parametrize("runtime", ALL_RUNTIMES)
def test_no_configured_storage_resolves_to_no_cache_root(runtime: str) -> None:
    # Guessing a path would put gated multi-gigabyte downloads somewhere the
    # operator never agreed to, and still lose them at the end of the run.
    assert resolve_model_cache_root({}, runtime=runtime) == ""
    assert model_cache_env("") == {}
    assert model_cache_dirs("") == ()
    assert render_model_cache_shell("", mounted=True) == ""


def test_a_kubernetes_claim_or_host_path_selects_the_default_mount() -> None:
    assert (
        resolve_model_cache_root(
            {MODEL_CACHE_PVC_ENV: "npa-model-cache"}, runtime=RUNTIME_KUBERNETES
        )
        == DEFAULT_MODEL_CACHE_MOUNT
    )
    assert (
        resolve_model_cache_root(
            {MODEL_CACHE_HOST_PATH_ENV: "/mnt/weights"}, runtime=RUNTIME_KUBERNETES
        )
        == DEFAULT_MODEL_CACHE_MOUNT
    )
    assert (
        resolve_model_cache_root(
            {MODEL_CACHE_HOST_PATH_ENV: "/mnt/weights"}, runtime=RUNTIME_DOCKER
        )
        == DEFAULT_MODEL_CACHE_MOUNT
    )


@pytest.mark.parametrize(
    ("runtime", "configured"),
    [
        # A claim exists only inside a cluster: a docker deploy cannot bind it, and
        # exporting HF_HOME at a path with nothing mounted turns a working run into
        # a failing one -- /opt is root-owned and the images run unprivileged.
        (RUNTIME_DOCKER, {MODEL_CACHE_PVC_ENV: "npa-model-cache"}),
        # A Serverless Job has no volume concept at all, so neither signal reaches
        # it. This is the case that would have broken working jobs the moment an
        # operator configured a claim for their Kubernetes workflows.
        (RUNTIME_PREMOUNTED, {MODEL_CACHE_PVC_ENV: "npa-model-cache"}),
        (RUNTIME_PREMOUNTED, {MODEL_CACHE_HOST_PATH_ENV: "/mnt/weights"}),
    ],
)
def test_a_runtime_ignores_storage_it_cannot_reach(
    runtime: str, configured: dict[str, str]
) -> None:
    assert resolve_model_cache_root(configured, runtime=runtime) == ""
    assert model_cache_env(resolve_model_cache_root(configured, runtime=runtime)) == {}


@pytest.mark.parametrize("runtime", ALL_RUNTIMES)
def test_an_explicit_root_is_honored_by_every_runtime(runtime: str) -> None:
    # NPA_MODEL_CACHE_DIR is the operator asserting the path is already mounted,
    # which is a claim only they can make -- and the only one a runtime that mounts
    # nothing can act on.
    root = resolve_model_cache_root(
        {MODEL_CACHE_DIR_ENV: "/mnt/shared/npa-weights/"}, runtime=runtime
    )

    assert root == "/mnt/shared/npa-weights"


def test_an_explicit_cache_dir_wins_over_the_storage_it_lives_on() -> None:
    root = resolve_model_cache_root(
        {
            MODEL_CACHE_DIR_ENV: "/mnt/shared/npa-weights/",
            MODEL_CACHE_PVC_ENV: "npa-model-cache",
        },
        runtime=RUNTIME_KUBERNETES,
    )

    assert root == "/mnt/shared/npa-weights"


def test_an_unknown_runtime_is_refused_rather_than_guessed() -> None:
    # A new call site that names a runtime this module does not model must fail
    # loudly; silently treating it as "mounts everything" is the bug class the
    # runtime argument exists to remove.
    with pytest.raises(ModelCacheError):
        resolve_model_cache_root({}, runtime="serverless-jobs")


def test_relative_cache_paths_are_rejected() -> None:
    with pytest.raises(ModelCacheError):
        resolve_model_cache_root(
            {MODEL_CACHE_DIR_ENV: "weights"}, runtime=RUNTIME_KUBERNETES
        )
    with pytest.raises(ModelCacheError):
        model_cache_host_path({MODEL_CACHE_HOST_PATH_ENV: "./weights"})


def test_a_claim_name_kubernetes_cannot_accept_is_rejected() -> None:
    assert (
        model_cache_pvc({MODEL_CACHE_PVC_ENV: "npa-model-cache"}) == "npa-model-cache"
    )
    with pytest.raises(ModelCacheError):
        model_cache_pvc({MODEL_CACHE_PVC_ENV: "Npa_Model_Cache"})


def test_the_whole_hugging_face_cache_family_lands_under_one_root() -> None:
    env = model_cache_env("/opt/npa-model-cache")

    assert env[MODEL_CACHE_DIR_ENV] == "/opt/npa-model-cache"
    assert env["HF_HOME"] == "/opt/npa-model-cache/huggingface"
    assert env["HF_HUB_CACHE"] == "/opt/npa-model-cache/huggingface/hub"
    assert env["HUGGINGFACE_HUB_CACHE"] == "/opt/npa-model-cache/huggingface/hub"
    assert env["HF_XET_CACHE"] == "/opt/npa-model-cache/huggingface/xet"
    assert env["TORCH_HOME"] == "/opt/npa-model-cache/torch"


@pytest.mark.parametrize(
    "name",
    [
        # Each of these is read by exactly one tool, and one of them left unset is
        # a silent multi-gigabyte re-download on the next run.
        "COSMOS_HF_CACHE",
        "NPA_COSMOS3_CACHE",
        "COSMOS_DOWNLOAD_CACHE_DIR",
        "NPA_COSMOS_REASON2_CACHE",
        "NPA_COSMOS_REASON3_CACHE",
        "NPA_COSMOS_CURATE_WEIGHTS_DIR",
        "HF_LEROBOT_HOME",
        "LEROBOT_HF_HOME",
        "WAN22_CACHE_DIR",
        "NPA_LTX_MODEL_CACHE",
    ],
)
def test_every_tool_specific_weight_directory_is_redirected(name: str) -> None:
    assert model_cache_env("/cache")[name].startswith("/cache/")


def test_the_env_allow_list_covers_every_variable_the_module_can_set() -> None:
    # k8s_components filters a sibling Job's env down to an allow-list; a variable
    # missing from it would leak that tool's download to a container-local path.
    assert set(model_cache_env("/cache")) == set(MODEL_CACHE_ENV_NAMES)
    assert (
        len(MODEL_CACHE_ENV_NAMES) == len({name for name, _ in MODEL_CACHE_LAYOUT}) + 1
    )


def test_the_cache_shell_creates_every_directory_and_says_it_is_durable() -> None:
    shell = render_model_cache_shell("/cache", mounted=True)

    assert shell.startswith("mkdir -p ")
    for path in model_cache_dirs("/cache"):
        assert path in shell
    assert "weights persist across runs" in shell


def test_the_cache_shell_does_not_promise_persistence_it_cannot_vouch_for() -> None:
    # With an explicit NPA_MODEL_CACHE_DIR and no volume of ours, the operator may
    # have pointed at a real mount or at an empty container directory. The log line
    # is how an operator checks a run is caching, so it must not assert the second
    # case is the first.
    shell = render_model_cache_shell("/cache", mounted=False)

    assert "mkdir -p " in shell
    assert "weights persist across runs" not in shell
    assert "not mounted by npa" in shell


def test_pod_config_gains_a_claim_backed_volume_and_mount() -> None:
    patched = pod_config_with_model_cache(
        None, root="/opt/npa-model-cache", pvc="npa-model-cache"
    )

    assert patched["spec"]["volumes"] == [
        {
            "name": MODEL_CACHE_VOLUME_NAME,
            "persistentVolumeClaim": {"claimName": "npa-model-cache"},
        }
    ]
    assert patched["spec"]["containers"] == [
        {
            "name": "ray-node",
            "volumeMounts": [
                {"name": MODEL_CACHE_VOLUME_NAME, "mountPath": "/opt/npa-model-cache"}
            ],
        }
    ]


def test_pod_config_keeps_the_volumes_the_spec_already_declared() -> None:
    original = {
        "spec": {
            "containers": [
                {
                    "name": "ray-node",
                    "volumeMounts": [
                        {
                            "name": "cosmos2-model-cache",
                            "mountPath": "/opt/cosmos/model-cache",
                        }
                    ],
                }
            ],
            "volumes": [{"name": "cosmos2-model-cache", "emptyDir": {}}],
        }
    }

    patched = pod_config_with_model_cache(
        original, root="/opt/npa-model-cache", pvc="npa-model-cache"
    )

    mounts = patched["spec"]["containers"][0]["volumeMounts"]
    assert [item["mountPath"] for item in mounts] == [
        "/opt/cosmos/model-cache",
        "/opt/npa-model-cache",
    ]
    assert [item["name"] for item in patched["spec"]["volumes"]] == [
        "cosmos2-model-cache",
        MODEL_CACHE_VOLUME_NAME,
    ]
    # The caller's mapping is not mutated in place.
    assert len(original["spec"]["volumes"]) == 1


def test_a_spec_that_owns_the_cache_path_is_left_alone() -> None:
    original = {
        "spec": {
            "containers": [
                {
                    "name": "ray-node",
                    "volumeMounts": [
                        {
                            "name": "operator-weights",
                            "mountPath": "/opt/npa-model-cache",
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "operator-weights",
                    "persistentVolumeClaim": {"claimName": "operator-weights"},
                }
            ],
        }
    }

    patched = pod_config_with_model_cache(
        original, root="/opt/npa-model-cache/", pvc="npa-model-cache"
    )

    assert patched == original


def test_a_host_path_cache_is_mounted_when_there_is_no_claim() -> None:
    patched = pod_config_with_model_cache(
        None, root="/opt/npa-model-cache", host_path="/mnt/weights"
    )

    assert patched["spec"]["volumes"][0]["hostPath"] == {
        "path": "/mnt/weights",
        "type": "DirectoryOrCreate",
    }


def test_mounting_without_any_backing_storage_is_a_configuration_error() -> None:
    with pytest.raises(ModelCacheError):
        pod_config_with_model_cache(None, root="/opt/npa-model-cache")


def test_docker_volume_args_bind_the_host_cache_at_the_shared_root() -> None:
    environ = {MODEL_CACHE_HOST_PATH_ENV: "/mnt/weights"}

    assert docker_model_cache_volumes(environ=environ) == (
        f"/mnt/weights:{DEFAULT_MODEL_CACHE_MOUNT}",
    )
    assert docker_model_cache_volumes(environ={}) == ()
