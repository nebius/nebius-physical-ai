"""Keep exact workflow-image declarations consistent through native consumers."""

from typing import Annotated

from pydantic import TypeAdapter
import pytest

from npa.workbench.nurec.navigation_probe import _declared_image
from npa.workflows.field_failure.contracts import _Adapter
from npa.workflows.isaac_rgbd.contract import ISAAC_SIM_VERSION, _runtime_materials
from npa.workflows.navigation.contract import Recipe


def _navigation_recipe_image(reference):
    image = Recipe.model_fields["image"]
    annotation = Annotated.__class_getitem__((str, *image.metadata))
    TypeAdapter(annotation).validate_python(reference, strict=True)


def _field_failure_image(reference):
    _Adapter(
        entrypoint="example.worker:run",
        source_sha256="b" * 64,
        runtime_image=reference,
    )


def _warehouse_material_image(reference):
    _runtime_materials(
        {
            "isaac_sim_version": ISAAC_SIM_VERSION,
            "image": reference,
            "library_sha256": "b" * 64,
            "modules": {
                "Default.mdl": {"path": "core/Default.mdl", "sha256": "c" * 64}
            },
        }
    )


CONSUMERS = (
    _declared_image,
    _navigation_recipe_image,
    _field_failure_image,
    _warehouse_material_image,
)


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda consumer: consumer.__name__)
@pytest.mark.parametrize(
    "repository",
    [
        "registry.example.invalid/runtime",
        "ghcr.io/team__simulation/runtime---1",
        "registry--mirror.example.invalid/team/runtime",
        "Registry--Mirror.example.invalid/team/runtime",
        "[2001:db8::1]/team/runtime",
        "[2001:db8::1]:5000/team/runtime",
    ],
)
def test_valid_operator_image_names_reach_each_exact_consumer(consumer, repository):
    consumer(repository + "@sha256:" + "a" * 64)


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda consumer: consumer.__name__)
@pytest.mark.parametrize(
    "reference",
    [
        "",
        "tool://isaac-lab",
        "registry.example.invalid/runtime:latest",
        "namespace/runtime@sha256:" + "a" * 64,
        "registry.example.invalid/runtime:tag@sha256:" + "a" * 64,
        "registry.example.invalid/runtime@sha256:" + "a" * 63,
        "registry.example.invalid/runtime@sha256:" + "A" * 64,
        "registry.example.invalid/runtime@sha256:" + "a" * 64 + "\n",
        "registry.example.invalid/team___simulation/runtime@sha256:" + "a" * 64,
        "registry.example.invalid/team/runtime..1@sha256:" + "a" * 64,
        "[2001:db8::1%zone]/runtime@sha256:" + "a" * 64,
        "registry.example.invalid:5000:9000/runtime@sha256:" + "a" * 64,
    ],
)
def test_each_consumer_rejects_mutable_unqualified_and_malformed_inputs(
    consumer, reference
):
    with pytest.raises(ValueError):
        consumer(reference)
