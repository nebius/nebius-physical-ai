"""Create an isolated Encord project and persist content-bound prelabels."""

from __future__ import annotations

import hashlib
from typing import Any

from npa.clients.storage import StorageClient
from npa.workbench.encord.client import _default_user_client
from npa.workbench.encord.integrity import compare_checksums
from npa.workbench.encord.label_plan import LabelPlan
from npa.workbench.encord.schemas import EncordToolError, PushReceipt
from npa.workbench.encord.storage import ConditionalArtifactStore, json_bytes


def import_labels(
    *,
    input_path: str,
    receipt_uri: str,
    project_title: str,
    output_path: str,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
) -> dict:
    """Import prelabels into a new project without changing existing projects.

    Args:
        input_path: S3 label-plan JSON; receipt_uri is its completed push receipt.
        project_title: Unique title for a new project and ontology.
        output_path: New S3 receipt JSON, checkpointed after each mutation.
        workflow_run: Provenance identifier.
        user_client, storage_client: Optional injected provider clients.
    Returns:
        Completed receipt with exact project, row, object, and scene identities.
    Raises:
        EncordToolError: Inputs conflict or an SDK operation fails.
        ArtifactConflict: The output exists; retries never create duplicates.
    """
    storage = storage_client or StorageClient.from_environment()
    store = ConditionalArtifactStore(storage)
    payload, plan, push = _import_inputs(store, input_path, receipt_uri)
    client = user_client or _default_user_client()
    if not project_title.strip() or list(client.get_projects(title_eq=project_title)):
        raise EncordToolError("Choose a unique, nonempty new project title")
    receipt = _initial_receipt(plan, payload, input_path, push, workflow_run)
    version = store.create_json(output_path, receipt)

    def checkpoint():
        nonlocal version
        version = store.replace_json(output_path, receipt, version)

    return _import_project(client, plan, push, project_title, receipt, checkpoint)


def _import_inputs(store, input_path, receipt_uri):
    payload = store.read_json(input_path)
    plan = LabelPlan.model_validate(payload)
    push = PushReceipt.model_validate(store.read_json(receipt_uri))
    _validate_sources(plan, push)
    return payload, plan, push


def _validate_sources(plan, push):
    if push.status != "completed" or not push.dataset_hash:
        raise EncordToolError("Label import requires a completed push with a dataset")
    sources = {item.source_uri: item for item in push.items}
    if set(sources) != {video.source_uri for video in plan.videos}:
        raise EncordToolError("Label plan must cover exactly the pushed media")
    for video in plan.videos:
        item = sources[video.source_uri]
        if (
            item.outcome != "successful"
            or compare_checksums(
                item.source_checksum,
                item.source_checksum_kind,
                video.source_sha256,
                "sha256",
            )
            is not True
        ):
            raise EncordToolError("Label plan does not match the pushed video SHA-256")


def _initial_receipt(plan, payload, input_path, push, workflow_run):
    return {
        "schema_version": "npa.encord.label_receipt.v1",
        "status": "running",
        "workflow_run": workflow_run,
        "plan_uri": input_path,
        "plan_sha256": hashlib.sha256(json_bytes(payload)).hexdigest(),
        "provenance": plan.provenance,
        "review_status": "unreviewed_prelabels",
        "dataset_hash": push.dataset_hash,
        "ontology_hash": "",
        "project_hash": "",
        "items": [],
    }


def _import_project(client, plan, push, title, receipt, checkpoint):
    try:
        ontology = client.create_ontology(
            title + " ontology",
            description=plan.provenance,
            structure=_ontology_structure(plan),
        )
        receipt["ontology_hash"] = str(ontology.ontology_hash)
        checkpoint()
        receipt["project_hash"] = str(
            client.create_project(
                project_title=title,
                dataset_hashes=[push.dataset_hash],
                ontology_hash=receipt["ontology_hash"],
                project_description="NPA imported prelabels; human review is pending. "
                + plan.provenance,
            )
        )
        checkpoint()
        project = client.get_project(receipt["project_hash"])
        _save_project_rows(project, plan, push, receipt, checkpoint)
        receipt["status"] = "completed"
        checkpoint()
        return receipt
    except Exception as exc:  # noqa: BLE001 - SDK failures vary by operation
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        checkpoint()
        raise EncordToolError(
            "Label import failed; inspect its durable receipt"
        ) from exc


def _ontology_structure(plan):
    from encord.objects import OntologyStructure
    from encord.objects.ontology_labels_impl import Shape
    from encord.objects.attributes import RadioAttribute

    structure = OntologyStructure()
    classes = {
        track.class_name: track.shape for video in plan.videos for track in video.tracks
    }
    for name, shape in sorted(classes.items()):
        structure.add_object(name, Shape(shape))
    options = {}
    for video in plan.videos:
        for classification in video.classifications:
            options.setdefault(classification.name, set()).update(
                segment.value for segment in classification.segments
            )
    for name, values in sorted(options.items()):
        classification = structure.add_classification()
        attribute = classification.add_attribute(RadioAttribute, name, required=True)
        for value in sorted(values):
            attribute.add_option(value)
    return structure


def _save_project_rows(project, plan, push, receipt, checkpoint):
    rows = list(project.list_label_rows_v2())
    by_item = {str(row.backing_item_uuid): row for row in rows}
    expected = {item.item_uuid for item in push.items}
    if len(by_item) != len(rows) or set(by_item) != expected:
        raise EncordToolError("Project rows do not match the exact pushed item UUIDs")
    sources = {item.source_uri: item for item in push.items}
    for video in plan.videos:
        row = by_item[sources[video.source_uri].item_uuid]
        row.initialise_labels()
        _validate_geometry(row, video)
        saved = _populate_row(row, video)
        receipt["items"].append(saved)
        checkpoint()
        row.save(validate_before_saving=True)
        saved["status"] = "saved"
        checkpoint()


def _validate_geometry(row, video):
    if (row.width, row.height, row.number_of_frames) != (
        video.width,
        video.height,
        video.frame_count,
    ):
        raise EncordToolError(
            "Encord video dimensions or frame count differ from the plan"
        )
    if list(row.get_object_instances()) or list(row.get_classification_instances()):
        raise EncordToolError("New project unexpectedly contains annotations")


def _populate_row(row, video):
    return {
        "source_uri": video.source_uri,
        "item_uuid": str(row.backing_item_uuid),
        "data_hash": row.data_hash,
        "label_hash": row.label_hash,
        "tracks": [_save_track(row, track) for track in video.tracks],
        "classifications": _save_classifications(row, video.classifications),
        "status": "saving",
    }


def _save_track(row, track):
    from encord.objects import Object

    ontology_object = row.ontology_structure.get_child_by_title(
        title=track.class_name, type_=Object
    )
    instance = ontology_object.create_instance()
    for annotation in track.annotations:
        instance.set_for_frames(
            coordinates=_coordinates(annotation),
            frames=annotation.frame,
            manual_annotation=False,
        )
    row.add_object_instance(instance)
    return {"track_id": track.track_id, "object_hash": instance.object_hash}


def _coordinates(annotation):
    from encord.objects.coordinates import (
        BoundingBoxCoordinates,
        PointCoordinate,
        PolygonCoordinates,
    )
    from npa.workbench.encord.label_plan import BoxAnnotation

    if isinstance(annotation, BoxAnnotation):
        return BoundingBoxCoordinates(
            height=annotation.height,
            width=annotation.width,
            top_left_x=annotation.x,
            top_left_y=annotation.y,
        )
    return PolygonCoordinates(
        [PointCoordinate(point.x, point.y) for point in annotation.points]
    )


def _save_classifications(row, classifications):
    from encord.objects import Classification, Option
    from encord.objects.frames import Range

    saved = []
    for classification in classifications:
        ontology_class = row.ontology_structure.get_child_by_title(
            title=classification.name, type_=Classification
        )
        for segment in classification.segments:
            option = ontology_class.get_child_by_title(
                title=segment.value, type_=Option
            )
            instance = ontology_class.create_instance()
            instance.set_answer(option)
            for answer in instance.get_all_static_answers():
                answer.is_manual_annotation = False
            instance.set_for_frames(
                frames=Range(start=segment.start_frame, end=segment.end_frame),
                manual_annotation=False,
            )
            row.add_classification_instance(instance)
            saved.append(
                {
                    "name": classification.name,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "classification_hash": instance.classification_hash,
                    "feature_hash": instance.feature_hash,
                    "answers": [
                        answer.to_encord_dict()
                        for answer in instance.get_all_static_answers()
                    ],
                }
            )
    return saved
