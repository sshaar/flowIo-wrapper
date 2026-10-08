"""Regression tests for issues found in adversarial review."""
import json
import threading
import urllib.request

import pytest

from flowio_capture import FlowIoBridge, Matrix, SessionContext, Store
from flowio_capture.dataset import build_dataset, derive, replay
from flowio_capture.normalize import MatrixConvention, invert
from flowio_capture.privacy import Pseudonymizer
from flowio_capture.recorder import CaptureSession
from flowio_capture.schema import SchemaError
from flowio_capture.server import make_server

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


def sess(tmp_path, sid="s1", clock=None):
    return CaptureSession.start(Store(tmp_path), SessionContext(experiment_id="exp"), session_id=sid,
                                clock=clock or Clock())


def labels(tmp_path, sid="s1", mid="default"):
    recs, skip = derive(replay(Store(tmp_path).read(sid), mid))
    return recs, skip


# ---- recorder state ---------------------------------------------------------------

def test_undo_after_resync_does_not_replay_stale_stack(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.0), source="auto")
    s.edit_cell("FITC", "PE-A", 0.1)
    s.edit_cell("FITC", "PE-A", 0.3, old_value=0.2)  # resync 0.1->0.2, then user 0.2->0.3
    s.undo()   # 0.3 -> 0.2
    s.undo()   # stale entry (0.0->0.1) must not apply
    s.export()
    s.end()
    rp = replay(Store(tmp_path).read("s1"))
    assert rp.errors == []
    _, skip = derive(rp)
    assert skip and "desynced" in skip  # final export cannot be trusted


def test_empty_stack_undo_marks_desync_until_host_reports_matrix(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s.undo()                       # host undid something we never saw
    s.edit_cell("FITC", "PE-A", 0.2)
    s.observe_matrix(mat(0.2, 0.05))  # authoritative: clears desync
    s.edit_cell("FITC", "PE-A", 0.25)
    s.export()
    s.end()
    recs, skip = labels(tmp_path)
    assert skip is None
    dirs = recs["cell_directions"]
    assert [d["value_before"] for d in dirs] == [0.2]  # desynced edit and resync excluded


def test_resync_does_not_steal_dt(tmp_path):
    c = Clock(step=10)
    s = sess(tmp_path, clock=c)
    s.snapshot(mat(), source="auto")
    s.edit_cell("FITC", "PE-A", 0.2)
    c.step = 30
    s.edit_cell("FITC", "PE-A", 0.3, old_value=0.25)
    s.export()
    s.end()
    edits = replay(Store(tmp_path).read("s1")).edits
    resync, user = edits[-2], edits[-1]
    assert resync.origin == "resync" and resync.dt is None
    assert user.dt is not None and user.dt > 20


def test_export_with_matrix_is_atomic_and_exact(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s.export(matrix=mat(0.13))
    s.end()
    recs, _ = labels(tmp_path)
    assert Matrix.from_dict(recs["accepted"][0]["accepted"]).get("FITC", "PE-A") == 0.13


def test_reopen_after_torn_line_keeps_session_readable(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s._w.close()
    p = Store(tmp_path).path_for("s1")
    with open(p, "a") as fh:
        fh.write('{"v":1,"seq":2,"ty')
    w = Store(tmp_path).open_writer("s1")
    w.write("note", {"kind": "x"}, "2026-01-01T00:00:00Z")
    w.close()
    ev = Store(tmp_path).read("s1")
    assert [e["seq"] for e in ev] == [0, 1, 2]


def test_failed_start_leaves_no_file(tmp_path):
    with pytest.raises(SchemaError):
        CaptureSession.start(Store(tmp_path), SessionContext(experiment_id="e", extra={"x": float("nan")}),
                             session_id="bad")
    assert not Store(tmp_path).path_for("bad").exists()


# ---- labels ---------------------------------------------------------------------------

def test_app_recompute_after_edit_starts_new_segment(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.05), source="acquisition")
    s.edit_cell("FITC", "PE-A", 0.06)
    s.snapshot(mat(0.20, 0.04), source="auto")  # app recomputed
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    acc = recs["accepted"][0]
    assert acc["delta"] == [] and acc["segment_source"] == "auto"
    assert acc["snapshot_after_first_edit"] is True
    assert recs["cell_directions"] == [] and recs["preference_pairs"] == []


def test_chains_break_at_snapshot_no_fake_overshoot(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.10), source="auto")
    s.edit_cell("FITC", "PE-A", 0.31)
    s.snapshot(mat(0.50), source="import")
    s.edit_cell("FITC", "PE-A", 0.30)
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    assert recs["cell_hard_negatives"] == []
    assert all(t["hi"] != 0.31 for t in recs["tolerances"])


def test_overshoot_requires_crossing_and_user_moves(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.10), source="auto")
    s.edit_cell("FITC", "PE-A", 0.14)  # up, final is 0.16 -> no crossing
    s.edit_cell("FITC", "PE-A", 0.12)  # down (wrong direction, not overshoot)
    s.edit_cell("FITC", "PE-A", 0.16)
    s.edit_cell("PE", "FITC-A", 0.05)
    s.undo()
    s.redo()                           # redo restores 0.05 -> not "undone"
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    reasons = {(n["row"], n["rejected_value"]): n["reason"] for n in recs["cell_hard_negatives"]}
    assert ("FITC", 0.14) not in reasons
    assert ("PE", 0.05) not in reasons


def test_multi_cell_undo_not_cell_negative(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s.observe_matrix(mat(0.2, 0.05))
    s.undo()
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    assert recs["cell_hard_negatives"] == []


def test_eps_respected_in_pairs(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.10), source="auto")
    s.edit_cell("FITC", "PE-A", 0.10000001)
    s.edit_cell("FITC", "PE-A", 0.2)
    s.export()
    s.end()
    recs, _ = labels(tmp_path)
    assert all(p["n_cells_diff"] > 0 for p in recs["preference_pairs"])
    assert len(recs["preference_pairs"]) == 1


def test_snapshots_after_accept_not_leaked(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(), source="auto")
    s.export()
    s.snapshot(mat(0.3), source="auto")
    s.end()
    recs, _ = derive(replay(Store(tmp_path).read("s1")))
    # last export is the only one; snapshot after it must not appear
    assert len(recs["accepted"][0]["snapshots"]) == 1


def test_matrix_ids_are_labeled_separately(tmp_path):
    s = sess(tmp_path)
    s.snapshot(mat(0.10), source="auto", matrix_id="tube_A")
    s.snapshot(mat(0.30), source="auto", matrix_id="tube_B")
    s.edit_cell("FITC", "PE-A", 0.12, matrix_id="tube_A")
    s.export(matrix_id="tube_A")
    s.edit_cell("FITC", "PE-A", 0.35, matrix_id="tube_B")
    s.export(matrix_id="tube_B")
    s.end()
    summary = build_dataset(Store(tmp_path), tmp_path / "ds")
    assert summary["labeled"] == 2
    pairs = [json.loads(l) for l in open(tmp_path / "ds" / "preference_pairs.jsonl")]
    assert not any(p["kind"] == "revised_export" for p in pairs)


# ---- conventions ------------------------------------------------------------------

def test_invert_roundtrip():
    m = [[1, 0.1, 0], [0.02, 1, 0.01], [0, 0.003, 1]]
    back = invert(invert(m))
    assert all(abs(a - b) < 1e-12 for r1, r2 in zip(m, back) for a, b in zip(r1, r2))


def test_percent_and_compensation_normalized():
    pct = Matrix.from_lists(ROWS, COLS, [[100, 10, 0], [2, 100, 1], [0, 0, 100]])
    out = MatrixConvention(unit="percent").to_canonical(pct)
    assert out.get("FITC", "PE-A") == pytest.approx(0.10)
    comp = Matrix.from_lists(ROWS, COLS, invert([list(r) for r in mat().values]))
    out = MatrixConvention(kind="compensation").to_canonical(comp)
    assert out.get("FITC", "PE-A") == pytest.approx(0.10)
    with pytest.raises(SchemaError, match="percent"):
        MatrixConvention().to_canonical(pct)
    with pytest.raises(SchemaError):
        MatrixConvention().to_canonical(comp)  # strongly negative off-diagonals


def test_fcs_detector_rows_relabelled_and_cell_edit_by_detector(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    br.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS))
    fcs_style = Matrix.from_lists(COLS, COLS, [list(r) for r in mat().values])
    br.on_compensation_loaded("w", fcs_style, source="acquisition")
    br.on_cell_changed("w", "FITC-A", "PE-A", 0.2)  # host addresses row by detector
    br.on_export("w")
    br.close_all()
    sid = Store(tmp_path).session_ids()[0]
    recs, _ = derive(replay(Store(tmp_path).read(sid)))
    assert Matrix.from_dict(recs["accepted"][0]["accepted"]).get("FITC", "PE-A") == 0.2


def test_compensation_host_cell_edit(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    br.on_workspace_opened("w", panel=list(ROWS), detectors=list(COLS), matrix_kind="compensation")
    comp = Matrix.from_lists(ROWS, COLS, invert([list(r) for r in mat().values]))
    br.on_compensation_loaded("w", comp, source="auto")
    br.on_cell_changed("w", "FITC", "PE-A", comp.get("FITC", "PE-A") - 0.05)
    br.on_export("w")
    br.close_all()
    sid = Store(tmp_path).session_ids()[0]
    rp = replay(Store(tmp_path).read(sid))
    assert rp.errors == [] and len(rp.edits) >= 1


# ---- privacy / server -----------------------------------------------------------

def test_free_form_fields_do_not_leak(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    leak = "/Users/jdoe/patient_MRN123"
    br.on_workspace_opened("w", extra={"note": leak}, user="jdoe",
                           instrument={"model": "Aurora", "owner": leak, "SN": "X1"},
                           controls=[{"fluorochrome": "FITC", "file_path": leak + ".fcs"}])
    br.on_compensation_loaded("w", mat(), source="auto", label=leak)
    br.on_view_changed("w", "FITC-A", "PE-A", population="Lymph", gate_path=leak)
    br.on_export("w", kind=leak, target_path=leak)
    br.on_workspace_closed("w", reason=leak)
    raw = "".join(p.read_text() for p in (tmp_path / "sessions").glob("*.jsonl"))
    assert "jdoe" not in raw and "MRN123" not in raw and "X1" not in raw
    assert "Aurora" in raw and "Lymph" in raw


def test_salt_refuses_store_location(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWIO_CAPTURE_SALT", raising=False)
    monkeypatch.setenv("FLOWIO_CAPTURE_SALT_FILE", str(tmp_path / "store" / ".salt"))
    with pytest.raises(ValueError):
        Pseudonymizer.load(tmp_path / "store")
    monkeypatch.setenv("FLOWIO_CAPTURE_SALT_FILE", str(tmp_path / "keys" / "salt"))
    assert Pseudonymizer.load(tmp_path / "store").token("a") == Pseudonymizer.load(tmp_path / "store").token("a")


def test_event_id_dedup_and_rejected_note(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    br.handle({"event": "workspace_opened", "workspace_ref": "w"})
    br.handle({"event": "compensation_loaded", "workspace_ref": "w", "matrix": mat().to_dict(), "source": "auto"})
    e = {"event": "cell_changed", "workspace_ref": "w", "row": "FITC", "col": "PE-A",
         "new_value": 0.2, "event_id": "e1"}
    br.handle(e)
    assert br.handle(e) == {"duplicate": True}
    with pytest.raises(SchemaError):
        br.handle({"event": "cell_changed", "workspace_ref": "w", "row": "NOPE", "col": "PE-A", "new_value": 1})
    with pytest.raises(SchemaError):
        br.handle({"event": "cell_changed", "workspace_ref": "w"})  # missing args -> SchemaError, not TypeError
    br.close_all()
    ev = Store(tmp_path).read(Store(tmp_path).session_ids()[0])
    assert sum(1 for x in ev if x["type"] == "cell_edit") == 1
    assert any(x["type"] == "note" and x["data"]["kind"] == "rejected_event" for x in ev)


def _post(port, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/event", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_hardening(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    srv = make_server(br, token="t0k", port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ok = {"Content-Type": "application/json", "X-Capture-Token": "t0k"}
    try:
        body = json.dumps({"event": "workspace_opened", "workspace_ref": "w"}).encode()
        assert _post(port, body, {**ok, "Content-Type": "text/plain"})[0] == 415
        assert _post(port, body, {**ok, "Origin": "https://evil.example"})[0] == 403
        assert _post(port, b"[1,", ok)[0] == 400
        assert _post(port, json.dumps(["x"]).encode(), ok)[0] == 400
        code, r = _post(port, json.dumps([
            {"event": "workspace_opened", "workspace_ref": "w"},
            {"event": "cell_changed", "workspace_ref": "w", "row": "a", "col": "b", "new_value": 1},
        ]).encode(), ok)
        assert code == 400 and r["applied"] == 1 and r["failed_index"] == 1
    finally:
        srv.shutdown()
        br.close_all()
