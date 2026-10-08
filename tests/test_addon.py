"""Tests for the add-on layer: non-blocking façade, plots, cell status, drag collapsing."""
import pytest

from flowio_capture import Matrix, PlotSummary, SafeBridge, SessionContext, Store, histogram2d
from flowio_capture.bridge import FlowIoBridge
from flowio_capture.dataset import derive, replay
from flowio_capture.privacy import Pseudonymizer
from flowio_capture.recorder import CaptureSession
from flowio_capture.schema import SchemaError

ROWS = ("FITC", "PE", "APC")
COLS = ("FITC-A", "PE-A", "APC-A")


def mat(fp=0.10, pf=0.02, pa=0.01):
    return Matrix.from_lists(ROWS, COLS, [[1.0, fp, 0.0], [pf, 1.0, pa], [0.0, 0.0, 1.0]])


class Clock:
    def __init__(self, step=1.5):
        self.t, self.step = 1_760_000_000.0, step

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture
def pseudo():
    return Pseudonymizer(b"0123456789abcdef-test-salt")


def sess(tmp_path, sid="s1", clock=None, **ctx):
    return CaptureSession.start(Store(tmp_path), SessionContext(experiment_id="exp", **ctx),
                                session_id=sid, clock=clock or Clock())


def labels(tmp_path, sid="s1"):
    return derive(replay(Store(tmp_path).read(sid)))


def first_session_labels(root):
    store = Store(root)
    sid = store.session_ids()[0]
    return derive(replay(store.read(sid)))


# ---- SafeBridge ---------------------------------------------------------------------

def test_safe_bridge_never_raises_and_preserves_order(tmp_path, pseudo):
    sb = SafeBridge(tmp_path, pseudo=pseudo)
    sb.on_cell_changed("nope", "FITC", "PE-A", 0.3)        # no session: swallowed
    sb.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS))
    sb.on_compensation_loaded("w", mat().to_dict(), source="auto")
    sb.on_cell_changed("w", "FITC", "PE-A", 0.2)
    sb.handle({"event": "cell_changed", "workspace_ref": "w", "row": "BAD", "col": "x", "new_value": 1})
    sb.on_export("w")
    assert sb.flush()
    st = sb.stats
    assert st["errors"] == 2 and st["processed"] == 4 and st["dropped"] == 0
    sb.close()
    recs, skip = first_session_labels(tmp_path)
    assert skip is None
    assert Matrix.from_dict(recs["accepted"][0]["accepted"]).get("FITC", "PE-A") == 0.2


def test_safe_bridge_stamps_enqueue_time(tmp_path, pseudo):
    now = [1000.0]
    sb = SafeBridge(tmp_path, pseudo=pseudo, clock=lambda: now[0])
    sb.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS))
    sb.on_compensation_loaded("w", mat().to_dict(), source="auto")
    sb.on_cell_changed("w", "FITC", "PE-A", 0.2)
    now[0] += 30
    sb.on_cell_changed("w", "FITC", "PE-A", 0.25)
    sb.on_export("w")
    sb.close()
    store = Store(tmp_path)
    edits = [e for e in store.read(store.session_ids()[0]) if e["type"] == "cell_edit"]
    assert edits[1]["data"]["dt_since_last_edit"] == 30
    assert edits[0]["t"].startswith("1970-01-01T00:16:40")


def test_safe_bridge_unknown_attribute(tmp_path, pseudo):
    sb = SafeBridge(tmp_path, pseudo=pseudo)
    with pytest.raises(AttributeError):
        sb.on_something_else
    sb.close()


# ---- plots ---------------------------------------------------------------------------

def test_histogram2d_bins_and_clipping():
    counts, xr, yr, n = histogram2d([0.5, 1.5, 3.9, 10.0], [0.5] * 4, bins=4, x_range=(0, 4), y_range=(0, 4),
                                    x_scale="linear", y_scale="linear")
    assert n == 4 and xr == (0.0, 4.0)
    assert counts[0][0] == 1 and counts[1][0] == 1 and counts[3][0] == 2


def test_plot_summary_validation():
    with pytest.raises(SchemaError):
        PlotSummary("a", "b", [[1, 2], [3]])
    with pytest.raises(SchemaError):
        PlotSummary("a", "b", [[1, -1]])
    with pytest.raises(SchemaError):
        PlotSummary("a", "b", [[0] * 129])
    with pytest.raises(SchemaError):
        PlotSummary.from_dict({"x": "a", "y": "b", "counts": [[1]], "patient": "x"})
    p = PlotSummary.from_dict(PlotSummary("a", "b", [[1, 2], [3, 4]]).to_dict())
    assert p.n_events == 10 and p.shape == (2, 2)


def test_plot_attached_to_edits_and_deduped(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s.set_view("FITC-A", "PE-A")
    p = PlotSummary.from_events("FITC-A", "PE-A", [10, 200, 3000], [5, 50, 500], bins=8)
    seq = s.observe_plot(p)
    assert seq is not None and s.observe_plot(p) == seq
    ctrl = PlotSummary.from_events("FITC-A", "PE-A", [1, 2], [3, 4], bins=4, kind="control",
                                   control_fluorochrome="FITC")
    cseq = s.observe_plot(ctrl)
    s.edit_cell("FITC", "PE-A", 0.2)
    s.set_view("PE-A", "APC-A")
    s.edit_cell("PE", "APC-A", 0.03)   # no plot for this pair
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    d = {x["row"]: x for x in recs["cell_directions"]}
    assert d["FITC"]["plot_refs"] == {"sample": seq, "control": cseq}
    assert d["PE"]["plot_refs"] is None
    assert [pl["seq"] for pl in recs["plots"]] == [seq, cseq]
    assert recs["plots"][0]["matrix_hash"] is not None and "counts" in recs["plots"][0]


def test_plot_budget(tmp_path):
    s = CaptureSession.start(Store(tmp_path), SessionContext(experiment_id="e"), session_id="s1",
                             clock=Clock(), plot_budget_bytes=400)
    s.snapshot(mat(), source="auto")
    small = PlotSummary("FITC-A", "PE-A", [[1]])
    big = PlotSummary("FITC-A", "APC-A", [[1] * 32 for _ in range(32)])
    assert s.observe_plot(small) is not None
    assert s.observe_plot(big) is None
    assert s.observe_plot(big) is None
    s.end()
    ev = Store(tmp_path).read("s1")
    assert sum(1 for e in ev if e["type"] == "plot") == 1
    assert sum(1 for e in ev if e["type"] == "note" and e["data"]["kind"] == "plot_budget_exhausted") == 1


def test_bridge_plot_capture_off_by_default(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    br.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS))
    br.on_compensation_loaded("w", mat(), source="auto")
    assert br.on_plot_observed("w", x="FITC-A", y="PE-A", xs=[1, 2], ys=[3, 4]) == {"ignored": "capture_plots is off"}
    br.on_view_changed("w", "FITC-A", "PE-A", plot=PlotSummary("FITC-A", "PE-A", [[1]]).to_dict())
    br.close_all()
    ev = Store(tmp_path).read(Store(tmp_path).session_ids()[0])
    assert not any(e["type"] == "plot" for e in ev)


def test_bridge_plot_capture_on(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo, capture_plots=True)
    br.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS))
    br.on_compensation_loaded("w", mat(), source="auto")
    seq = br.on_plot_observed("w", x="FITC-A", y="PE-A", xs=[1, 200, 4000], ys=[3, 4, 5], bins=16,
                              population="Lymph", extra={"file": "/Users/jdoe/x.fcs"})
    assert isinstance(seq, int)
    br.close_all()
    raw = Store(tmp_path).path_for(Store(tmp_path).session_ids()[0]).read_text()
    assert "jdoe" not in raw and "Lymph" in raw


# ---- cell status ---------------------------------------------------------------------

def test_cell_status_edited_examined_unexamined(tmp_path):
    s = sess(tmp_path, panel=list(ROWS), detectors=list(COLS))
    s.snapshot(mat(), source="auto")
    s.set_view("FITC-A", "PE-A")
    s.edit_cell("FITC", "PE-A", 0.2)
    s.set_view("PE-A", "APC-A")
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    acc = recs["accepted"][0]
    cells = {(c["row"], c["col"]): c for c in acc["cells"]}
    assert ("FITC", "FITC-A") not in cells  # diagonal excluded
    assert cells[("FITC", "PE-A")]["status"] == "edited"
    assert cells[("PE", "FITC-A")]["status"] == "examined_accepted"
    assert cells[("PE", "APC-A")]["status"] == "examined_accepted"
    assert cells[("APC", "PE-A")]["status"] == "examined_accepted"
    assert cells[("FITC", "APC-A")]["status"] == "unexamined"
    assert cells[("APC", "FITC-A")]["status"] == "unexamined" and not cells[("APC", "FITC-A")]["viewed"]
    assert acc["cell_status_counts"] == {"edited": 1, "examined_accepted": 3, "unexamined": 2}
    assert acc["n_views"] == 2


def test_viewed_at_wrong_value_is_not_examined_accepted(tmp_path):
    s = sess(tmp_path, panel=list(ROWS), detectors=list(COLS))
    s.snapshot(mat(), source="auto")
    s.set_view("PE-A", "APC-A")        # PE/APC-A = 0.01 on screen
    s.set_view("FITC-A", "PE-A")
    s.snapshot(mat(pa=0.05), source="auto")   # app changed PE/APC-A while not on screen
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    cells = {(c["row"], c["col"]): c for c in recs["accepted"][0]["cells"]}
    assert cells[("PE", "APC-A")]["status"] == "unexamined" and cells[("PE", "APC-A")]["viewed"]
    assert cells[("APC", "PE-A")]["status"] == "examined_accepted"


# ---- drag collapsing -------------------------------------------------------------------

def test_drag_collapsed_into_one_judgment(tmp_path):
    c = Clock(step=0.05)
    s = sess(tmp_path, clock=c)
    s.snapshot(mat(0.10), source="auto")
    s.set_view("FITC-A", "PE-A")
    for v in (0.12, 0.15, 0.19, 0.17, 0.16):   # one drag; 0.19 is a mid-drag overshoot
        s.edit_cell("FITC", "PE-A", v)
    c.step = 2.0
    s.edit_cell("PE", "APC-A", 0.03)
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    acc = recs["accepted"][0]
    assert acc["n_raw_edits"] == 6 and acc["n_judgment_edits"] == 2
    d = [x for x in recs["cell_directions"] if x["row"] == "FITC"]
    assert len(d) == 1
    assert (d[0]["value_before"], d[0]["value_after"], d[0]["n_merged"]) == (0.10, 0.16, 5)
    assert recs["cell_hard_negatives"] == []
    kinds = sorted(p["kind"] for p in recs["preference_pairs"])
    assert kinds == ["baseline", "intermediate"]   # mid-drag states excluded
    tol = {(t["row"], t["col"]): t for t in recs["tolerances"]}
    assert tol[("FITC", "PE-A")]["lo"] == 0.10 and tol[("FITC", "PE-A")]["hi"] is None


def test_slow_edits_not_collapsed(tmp_path):
    s = sess(tmp_path, clock=Clock(step=1.0))
    s.snapshot(mat(0.10), source="auto")
    for v in (0.12, 0.19, 0.16):
        s.edit_cell("FITC", "PE-A", v)
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    assert len(recs["cell_directions"]) == 3
    assert {n["reason"] for n in recs["cell_hard_negatives"]} == {"overshoot"}


def test_drag_dt_zero_disables_collapsing(tmp_path):
    s = sess(tmp_path, clock=Clock(step=0.01))
    s.snapshot(mat(0.10), source="auto")
    for v in (0.12, 0.14):
        s.edit_cell("FITC", "PE-A", v)
    s.export()
    s.end()
    recs, _ = derive(replay(Store(tmp_path).read("s1")), drag_dt=0)
    assert len(recs["cell_directions"]) == 2
