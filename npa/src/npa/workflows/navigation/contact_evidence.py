"""Retain probe-only native contact samples without changing contact classification."""

from contextlib import contextmanager
import re

import numpy as np

from npa.workflows.navigation.artifacts import write_json


def _tensor(value):
    from npa.workflows.navigation.reference_contacts import _tensor as convert

    return convert(value)


def _robot_index(path):
    match = re.match(r"^/World/envs/env_(\d+)/Robot(?:/|$)", path)
    if match is None:
        raise ValueError("Contact evidence cannot identify a native robot path")
    return int(match.group(1))


def _mapping(contact, bodies, body_view, root_view):
    sensors = list(contact.sensor_paths)
    body_paths = list(body_view.prim_paths)
    root_paths = list(root_view.prim_paths)
    body_lookup = {path: index for index, path in enumerate(body_paths)}
    root_lookup = {_robot_index(path): index for index, path in enumerate(root_paths)}
    if len(body_lookup) != len(body_paths) or len(root_lookup) != len(root_paths):
        raise ValueError("Native contact pose views have ambiguous paths")
    rows = []
    for sensor, path in enumerate(sensors):
        robot = _robot_index(path)
        rows.append(
            {
                "sensor_index": sensor,
                "actual_sensor_path": path,
                "actual_body_name": path.rsplit("/", 1)[1],
                "actual_robot_index": robot,
                "body_pose_index": body_lookup[path],
                "root_pose_index": root_lookup[robot],
                "classified_robot_index": sensor // len(bodies),
                "classified_body_index": sensor % len(bodies),
                "classified_body_name": bodies[sensor % len(bodies)],
            }
        )
    if len(rows) != contact.sensor_count:
        raise ValueError("Native contact paths do not cover the contact sensors")
    return rows


def _filter_rows(paths, sensor_count, filter_count):
    if len(paths) == sensor_count and all(isinstance(row, list) for row in paths):
        if any(len(row) != filter_count for row in paths):
            raise ValueError("Native per-sensor filter paths have unexpected shape")
        return paths
    if len(paths) == filter_count and all(isinstance(row, str) for row in paths):
        return [paths] * sensor_count
    raise ValueError("Unsupported native contact filter-path layout")


def _strongest(pairs, indices, magnitudes, classified, count, filters, bodies):
    import torch

    robots = pairs // (filters * bodies)
    eligible = torch.where(classified, magnitudes, 0.0)
    maxima = torch.zeros(count, device=eligible.device, dtype=eligible.dtype)
    maxima.scatter_reduce_(0, robots, eligible, reduce="amax", include_self=True)
    candidates = torch.arange(len(indices), device=indices.device)
    selected = torch.where(
        (eligible == maxima[robots]) & (eligible > 0), candidates, len(indices)
    )
    chosen = torch.full((count,), len(indices), device=indices.device, dtype=torch.long)
    chosen.scatter_reduce_(0, robots, selected, reduce="amin", include_self=True)
    return chosen, maxima


def _native_handles(env, measurements):
    from isaaclab_physx.physics import PhysxManager
    from isaacsim.core.simulation_manager.impl.extension import (
        acquire_simulation_manager_interface,
    )

    names = "(" + "|".join(measurements.body_names) + ")"
    body_view = PhysxManager.get_physics_sim_view().create_rigid_body_view(
        f"/World/envs/env_*/Robot/{names}"
    )
    return (
        body_view,
        env.scene["robot"].root_view,
        acquire_simulation_manager_interface(),
    )


class ContactEvidence:
    """Keep each classified robot's strongest contact in each probe control interval.

    Args:
        env: Native reference environment.
        measurements: Existing contact measurements, with unchanged classification.
        output: Private native artifact directory.
        name: Current physical probe name.
    Returns:
        A recorder with copied samples and complete native sensor-path metadata.
    Raises:
        ValueError: Native identity or contact data is inconsistent.
    """

    def __init__(self, env, measurements, output, name):
        self.env, self.measurements = env, measurements
        self.view = measurements.view
        self.body_view, self.root_view, self.clock = _native_handles(env, measurements)
        self.mapping = _mapping(
            self.view, measurements.body_names, self.body_view, self.root_view
        )
        self.filters = [
            list(row) if not isinstance(row, str) else row
            for row in self.view.filter_paths
        ]
        self.filter_rows = _filter_rows(
            self.filters, self.view.sensor_count, self.view.filter_count
        )
        self.output = output / f"probe-{name}-contacts"
        self.output.mkdir()
        self.step = 0
        self.active = False
        self.rows = {}
        self._metadata()

    def _metadata(self):
        write_json(
            self.output / "index.json",
            {
                "schema": "npa.navigation.probe-contact-samples.v1",
                "selection": "strongest classified individual contact per classified robot per control interval; ties retain the first sampled contact",
                "scope": "Probe steps only; no initial-reset sampling; not a complete contact history",
                "pose_format": "world xyz and quaternion xyzw; native PhysxManager subspace root is /",
                "sensor_mapping": self.mapping,
                "native_filter_paths": self.filters,
                "configured_filter_paths": self.measurements.filter_paths,
                "classification": "abs(normal.z) < 0.7 or classified_body_index == base_index",
                "base_index": self.measurements.base_index,
                "sensor_count": self.view.sensor_count,
                "filter_count": self.view.filter_count,
                "population": self.env.num_envs,
                "physics_dt": self.env.physics_dt,
                "native_clock_source": "isaacsim.core.simulation_manager.native_step_events",
                "aggregate_scope": "sample_tick_sum is distinct from control_interval_peak_sum; current 0.02N threshold and classification remain unchanged",
            },
        )

    @contextmanager
    def interval(self):
        """Copy interval samples to disk even when a native step raises.

        Args:
            None.
        Returns:
            Context spanning one original native high-level step.
        Raises:
            OSError: Raw probe evidence cannot be retained.
        """
        self.step += 1
        self.rows = {}
        self.substep = 0
        self.active = True
        complete = False
        try:
            yield
            complete = True
        finally:
            self.active = False
            self._save(complete)

    def sample(self, data, pairs, indices, classified, aggregate):
        """Copy selected native contact values before the next physics update.

        Args:
            data: Original six-array native contact tuple, never modified.
            pairs: Existing sensor/filter pair indices.
            indices: Existing starts/counts-derived contact buffer indices.
            classified: Existing obstacle-classification mask.
            aggregate: Existing per-robot sum before the original force threshold.
        Returns:
            None.
        Raises:
            ValueError: Selected contact evidence is nonfinite or malformed.
        """
        if not self.active:
            return
        self.substep += 1
        if not len(indices):
            return
        magnitudes = _tensor(data[0]).flatten()[indices].abs()
        chosen, maxima = _strongest(
            pairs,
            indices,
            magnitudes,
            classified,
            self.env.num_envs,
            self.view.filter_count,
            self.measurements.body_count,
        )
        selected = self._improved(chosen, maxima, len(indices))
        if not selected:
            return
        self._copy_samples(data, pairs, indices, classified, aggregate, selected)

    def _improved(self, chosen, maxima, sentinel):
        chosen = chosen.detach().cpu().numpy()
        maxima = maxima.detach().cpu().numpy()
        return [
            (robot, int(index))
            for robot, index in enumerate(chosen)
            if index != sentinel
            and maxima[robot]
            > self.rows.get(robot, {"force_magnitude_n": 0.0})["force_magnitude_n"]
        ]

    def _copy_samples(self, data, pairs, indices, classified, aggregate, selected):
        import torch

        tick = int(self.clock.get_num_physics_steps())
        seconds = float(self.clock.get_simulation_time())
        contributions = self._contribution_counts(data, pairs, indices, classified)
        positions = torch.tensor([item[1] for item in selected], device=indices.device)
        contacts = indices[positions]
        pair_ids = pairs[positions].detach().cpu().numpy()
        arrays = [
            _tensor(value)[contacts].detach().cpu().numpy().copy() for value in data[:4]
        ]
        sums = aggregate.detach().cpu().numpy().copy()
        counts = _tensor(data[4]).detach().cpu().numpy().copy()
        starts = _tensor(data[5]).detach().cpu().numpy().copy()
        poses = self._poses(pair_ids)
        if tick != int(self.clock.get_num_physics_steps()):
            raise ValueError("Native physics advanced while contact poses were copied")
        for offset, (robot, _) in enumerate(selected):
            sensor, filter_index = divmod(int(pair_ids[offset]), self.view.filter_count)
            row = self._row(robot, sensor, filter_index, arrays, offset)
            row.update(
                physics_event_count=tick,
                physics_substep=self.substep,
                sample_tick_contributing_contacts=int(contributions[robot]),
                physics_seconds=seconds,
                contact_buffer_index=int(contacts[offset]),
                pair_contact_count=int(counts[sensor, filter_index]),
                pair_contact_start=int(starts[sensor, filter_index]),
                sample_tick_classified_sum_n=float(sums[robot]),
                body_pose_world_xyzw=poses[0][offset],
                root_pose_world_xyzw=poses[1][offset],
            )
            self.rows[robot] = row

    def _contribution_counts(self, data, pairs, indices, classified):
        import torch

        robots = pairs // (self.view.filter_count * self.measurements.body_count)
        positive = classified & (_tensor(data[0]).flatten()[indices].abs() > 0)
        counts = torch.zeros(self.env.num_envs, device=pairs.device, dtype=torch.long)
        counts.scatter_add_(0, robots, positive.long())
        return counts.detach().cpu().numpy().copy()

    def _poses(self, pair_ids):
        sensors = pair_ids // self.view.filter_count
        body_indices = [
            self.mapping[int(sensor)]["body_pose_index"] for sensor in sensors
        ]
        root_indices = [
            self.mapping[int(sensor)]["root_pose_index"] for sensor in sensors
        ]
        bodies = _tensor(self.body_view.get_transforms())[body_indices]
        roots = _tensor(self.root_view.get_root_transforms())[root_indices]
        return (
            bodies.detach().cpu().numpy().copy(),
            roots.detach().cpu().numpy().copy(),
        )

    def _row(self, robot, sensor, filter_index, arrays, offset):
        mapping = self.mapping[sensor]
        force = float(arrays[0][offset, 0])
        row = dict(mapping)
        row.update(
            filter_index=filter_index,
            native_filter_path=self.filter_rows[sensor][filter_index],
            force_n=force,
            force_magnitude_n=abs(force),
            point_world_m=arrays[1][offset],
            normal_world=arrays[2][offset],
            separation_m=float(arrays[3][offset, 0]),
            classified_by_steep_normal=bool(abs(arrays[2][offset, 2]) < 0.7),
            classified_by_base=mapping["classified_body_index"]
            == self.measurements.base_index,
            sensor_order_matches_classification=(
                mapping["actual_robot_index"] == robot
                and mapping["actual_body_name"] == mapping["classified_body_name"]
            ),
        )
        return row

    def _save(self, complete):
        arrays = {
            key: np.asarray([row[key] for row in self.rows.values()])
            for key in next(iter(self.rows.values()), {})
        }
        arrays["control_interval_peak_classified_sum_n"] = (
            _tensor(self.measurements.obstacle).detach().cpu().numpy().copy()
        )
        arrays["control_step"] = np.asarray(self.step)
        arrays["physics_substeps_observed"] = np.asarray(self.substep)
        arrays["step_completed"] = np.asarray(complete)
        for array in arrays.values():
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError("Native contact evidence contains nonfinite values")
        np.savez_compressed(self.output / f"{self.step:06d}.npz", **arrays)
