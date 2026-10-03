"""Drive Isaac's external renderer clock for static, physics-free sensor capture."""

from __future__ import annotations

from fractions import Fraction

from .contract import _integer, _keys

RENDERER_CLOCK = "external_static_trajectory"


def _set_render_time(timestamp_ns):
    from isaacsim.core.experimental.utils import prim as prim_utils
    from isaacsim.core.experimental.utils import stage as stage_utils
    from pxr import Sdf

    # Isaac 6's multi-tick renderer reads Fabric, independently of the USD timeline.
    # This sensor-only workflow owns that clock while physics stays paused.
    stage = stage_utils.get_current_stage(backend="fabric")
    prim = stage.GetPrimAtPath("/ExternalSimulationTime")
    if not prim:
        prim = stage.DefinePrim("/ExternalSimulationTime", "")
    clock = prim.GetAttribute("omni:time")
    if not clock:
        clock = prim_utils.create_prim_attribute(
            prim, name="omni:time", type_name=Sdf.ValueTypeNames.Double
        )
    clock.Set(timestamp_ns / 1e9)


def _reference_time(value):
    _keys(value, "numerator denominator", "render_reference_time")
    numerator = _integer(value["numerator"], "reference time numerator")
    denominator = _integer(value["denominator"], "reference time denominator", 1)
    return Fraction(numerator, denominator)


def _check_render_times(views, timestamp_ns):
    times = [_reference_time(view["render_reference_time"]) for view in views]
    if not times or len(set(times)) != 1:
        raise ValueError("cross-camera render reference times are not synchronized")
    expected = Fraction(timestamp_ns, 1_000_000_000)
    if abs(times[0] - expected) > Fraction(1, 1_000_000_000):
        raise ValueError(
            f"stale capture: rendered reference time {times[0]} "
            f"does not match requested time {expected}"
        )
    return times[0]
