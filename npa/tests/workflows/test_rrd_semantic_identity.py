"""Real encoded diagnostic recordings, not policy or model acceptance."""

from pathlib import Path

import pytest
import rerun as rr
import rerun.blueprint as rrb
from rerun.chunk import RrdReader

from npa.workflows import data_factory_viz as viz


def _record(
    path: Path,
    *,
    values=("PASS", "FAIL"),
    origins=("source", "generated"),
    temporal=False,
    split=False,
):
    stream = rr.RecordingStream("semantic-control", recording_id="frozen-control")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            *[rrb.TextDocumentView(origin=origin, name=origin) for origin in origins]
        )
    )
    stream.save(path, default_blueprint=blueprint)
    for value in values:
        if temporal:
            stream.set_time("frame", sequence=5)
        stream.log("diagnostic/status", rr.TextDocument(value), static=not temporal)
        if split:
            stream.flush()
    stream.flush()
    stream.disconnect()


@pytest.mark.parametrize("temporal", [False, True])
def test_static_and_same_time_precedence_is_not_an_unordered_multiset(
    tmp_path, temporal
):
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _record(left, temporal=temporal)
    _record(right, values=("FAIL", "PASS"), temporal=temporal)
    assert viz._rrd_semantic_sha256(left) != viz._rrd_semantic_sha256(right)
    with pytest.raises(viz.DataFactoryVizError, match="current run inputs"):
        viz._verify_existing_rrd_semantics(left, right)


def test_blueprint_references_preserve_ordered_view_topology(tmp_path):
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _record(left)
    _record(right, origins=("generated", "source"))
    assert viz._rrd_semantic_sha256(left) != viz._rrd_semantic_sha256(right)


@pytest.mark.parametrize("temporal", [False, True])
def test_equivalent_reencoding_accepts_new_uuid_time_and_chunk_boundaries(
    tmp_path, temporal
):
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _record(left, temporal=temporal)
    _record(right, temporal=temporal, split=True)
    assert left.read_bytes() != right.read_bytes()
    assert viz._rrd_semantic_sha256(left) == viz._rrd_semantic_sha256(right)
    viz._verify_existing_rrd_semantics(left, right)


def _indexed_record(path, *, reverse=False, split=False, clear=False):
    stream = rr.RecordingStream("selected-index-control", recording_id="frozen-index")
    stream.save(path)
    stream.set_time("frame", sequence=5)
    if clear:
        if not reverse:
            stream.log("diagnostic", rr.Clear(recursive=True))
        stream.log("diagnostic/status", rr.TextDocument("PASS"))
        if reverse:
            stream.log("diagnostic", rr.Clear(recursive=True))
    else:
        updates = [(10, "FAIL"), (20, "PASS")]
        for auxiliary, status in reversed(updates) if reverse else updates:
            stream.set_time("auxiliary", sequence=auxiliary)
            stream.log("diagnostic/status", rr.TextDocument(status))
            if split:
                stream.flush()
    stream.flush()
    stream.disconnect()


def _selected_frame_rows(path):
    reader = RrdReader(path)
    store = reader.store(store=reader.recordings()[0]).stream().collect()
    rows = store.reader(index="frame", contents="/diagnostic/status").to_arrow_table()
    return {
        name: values for name, values in rows.to_pydict().items() if name != "log_time"
    }


@pytest.mark.parametrize("clear", [False, True])
def test_selected_timeline_winner_and_recursive_clear_order_are_bound(tmp_path, clear):
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _indexed_record(left, clear=clear)
    _indexed_record(right, clear=clear, reverse=True)
    assert _selected_frame_rows(left) != _selected_frame_rows(right)
    assert viz._rrd_semantic_sha256(left) != viz._rrd_semantic_sha256(right)
    with pytest.raises(viz.DataFactoryVizError, match="current run inputs"):
        viz._verify_existing_rrd_semantics(left, right)


def test_equivalent_multitimeline_reencoding_retains_selected_winner(tmp_path):
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _indexed_record(left)
    _indexed_record(right, split=True)
    assert left.read_bytes() != right.read_bytes()
    assert _selected_frame_rows(left) == _selected_frame_rows(right)
    viz._verify_existing_rrd_semantics(left, right)
