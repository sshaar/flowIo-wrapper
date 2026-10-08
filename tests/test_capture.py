import json
import threading
import urllib.request
from pathlib import Path

import pytest

from flowio_capture import FlowIoBridge, Matrix, SessionContext, Store
from flowio_capture.dataset import build_dataset, derive, replay
from flowio_capture.fcs import parse_spillover, parse_text, read_text_segment
from flowio_capture.privacy import Pseudonymizer, scrub_keywords
from flowio_capture.recorder import CaptureSession
from flowio_capture.schema import SchemaError
from flowio_capture.server import make_server

ROWS = ("FITC", "PE", "APC")
COLS = ("FITC-A", "PE-A", "APC-A")


def mat(fp=0.10, pf=0.02, pa=0.01):
    return Matrix.from_lists(ROWS, COLS, [[1.0, fp, 0.0], [pf, 1.0, pa], [0.0, 0.0, 1.0]])


class Clock:
    def __init__(self):
        self.t = 1_760_000_000.0

    def __call__(self):
        self.t += 1.5
        return self.t


@pytest.fixture
def pseudo():
    return Pseudonymizer(b"0123456789abcdef-test-salt")


def new_session(tmp_path, sid="s1"):
    return CaptureSession.start(Store(tmp_path), SessionContext(experiment_id="exp"), session_id=sid,
                                clock=Clock())


# ---- schema -------------------------------------------------------------------

def test_matrix_validation():
    with pytest.raises(SchemaError):
        Matrix.from_lists(["a"], ["x", "y"], [[1.0]])
    with pytest.raises(SchemaError):
        Matrix.from_lists(["a", "a"], ["x", "y"], [[1, 0], [0, 1]])
    with pytest.raises(SchemaError):
        Matrix.from_lists(["a"], ["x"], [[float("nan")]])
    m = mat()
    assert m.with_value("FITC", "PE-A", 0.2).diff(m) == [("FITC", "PE-A", 0.2, 0.10)]


# ---- recorder + replay --------------------------------------------------------

def test_full_session_labels(tmp_path):
    s = new_session(tmp_path)
    s.snapshot(mat(0.05), source="acquisition")
    s.snapshot(mat(0.10), source="auto")
    s.set_view("FITC-A", "PE-A")
    s.edit_cell("FITC", "PE-A", 0.14)   # up
    s.edit_cell("FITC", "PE-A", 0.20)   # up (overshoots final 0.16)
    s.edit_cell("FITC", "PE-A", 0.16)   # down -> 0.20 is an overshoot
    s.set_view("PE-A", "APC-A")
    s.edit_cell("PE", "APC-A", 0.05)    # A->B
    s.edit_cell("PE", "APC-A", 0.01)    # B->A : 0.05 reverted
    s.edit_cell("PE", "FITC-A", 0.04)
    s.undo()                            # 0.04 undone -> back to 0.02
    s.export("report")                  # first acceptance, later revised
    s.edit_cell("PE", "FITC-A", 0.03)
    s.export("report")                  # final acceptance
    s.edit_cell("PE", "FITC-A", 0.09)   # post-export tweak: must be ignored
    s.end()

    rp = replay(Store(tmp_path).read("s1"))
    assert rp.errors == []
    assert len(rp.exports) == 2
    recs, skip = derive(rp)
    assert skip is None

    acc = recs["accepted"][0]
    assert acc["baseline_origin"] == "snapshot:auto" and acc["clean_baseline"]
    assert Matrix.from_dict(acc["baseline"]) == mat(0.10)
    final = Matrix.from_dict(acc["accepted"])
    assert final.get("FITC", "PE-A") == pytest.approx(0.16)
    assert final.get("PE", "FITC-A") == pytest.approx(0.03)  # not the post-export 0.09
    assert {(d["row"], d["col"]) for d in acc["delta"]} == {("FITC", "PE-A"), ("PE", "FITC-A")}

    dirs = recs["cell_directions"]
    first = dirs[0]
    assert (first["value_before"], first["label_before"], first["moved_toward_accepted"]) == (0.10, 1, True)
    assert all(d["value_after"] != 0.09 for d in dirs)
    assert dirs[0]["view"]["x"] == "FITC-A"

    negs = {(n["row"], n["col"], n["rejected_value"]): n["reason"] for n in recs["cell_hard_negatives"]}
    assert negs[("FITC", "PE-A", 0.20)] == "overshoot"
    assert negs[("PE", "APC-A", 0.05)] == "reverted"
    assert negs[("PE", "FITC-A", 0.04)] == "undone"

    kinds = {p["kind"] for p in recs["preference_pairs"]}
    assert {"baseline", "intermediate", "revised_export"} <= kinds
    rev = [p for p in recs["preference_pairs"] if p["kind"] == "revised_export"][0]
    assert rev["n_cells_diff"] == 1 and rev["hard"]
    # The accepted state itself is never a rejected example.
    assert all(Matrix.from_dict(p["rejected"]) != final for p in recs["preference_pairs"])

    tol = {(t["row"], t["col"]): t for t in recs["tolerances"]}
    assert tol[("FITC", "PE-A")]["lo"] == pytest.approx(0.14)
    assert tol[("FITC", "PE-A")]["hi"] == pytest.approx(0.20)


def test_matrix_only_host_produces_cell_edits(tmp_path):
    s = new_session(tmp_path)
    s.observe_matrix(mat())                       # first observation -> snapshot
    assert s.observe_matrix(mat(0.12, 0.03)) == 2  # batch of 2 cell edits
    assert s.observe_matrix(mat(0.12, 0.03)) == 0
    s.undo(matrix_after=mat())
    s.export()
    s.end()
    rp = replay(Store(tmp_path).read("s1"))
    assert rp.errors == []
    edits = [e for e in rp.edits if e.origin == "user"]
    assert len(edits) == 2 and edits[0].step == edits[1].step  # one step, same batch
    assert len(rp.steps) == 3  # snapshot, batch, undo


def test_unlabeled_session_skipped(tmp_path):
    s = new_session(tmp_path)
    s.snapshot(mat(), source="auto")
    s.edit_cell("FITC", "PE-A", 0.3)
    s.end()
    summary = build_dataset(Store(tmp_path), tmp_path / "ds")
    assert summary["labeled"] == 0
    assert summary["skipped"]["s1"].startswith("no export")


def test_accepted_without_edits_is_zero_delta_positive(tmp_path):
    s = new_session(tmp_path)
    s.snapshot(mat(), source="auto")
    s.export()
    s.end()
    recs, _ = derive(replay(Store(tmp_path).read("s1")))
    assert recs["accepted"][0]["delta"] == []
    assert recs["accepted"][0]["baseline_origin"] == "accepted_unedited"
    assert recs["preference_pairs"] == []


def test_desync_is_logged_and_replay_consistent(tmp_path):
    s = new_session(tmp_path)
    s.snapshot(mat(), source="auto")
    s.edit_cell("FITC", "PE-A", 0.2, old_value=0.15)  # host says old was 0.15, we had 0.10
    s.export()
    s.end()
    rp = replay(Store(tmp_path).read("s1"))
    assert rp.errors == []
    assert [e.origin for e in rp.edits] == ["resync", "user"]
    d = derive(rp)[0]["cell_directions"]
    assert len(d) == 1 and d[0]["value_before"] == 0.15


def test_torn_last_line_tolerated(tmp_path):
    s = new_session(tmp_path)
    s.snapshot(mat(), source="auto")
    s.end()
    p = Store(tmp_path).path_for("s1")
    with open(p, "a") as fh:
        fh.write('{"v":1,"seq":')
    assert len(Store(tmp_path).read("s1")) == 3


def test_session_ids_validated(tmp_path):
    with pytest.raises(SchemaError):
        Store(tmp_path).path_for("../escape")


# ---- privacy / FCS ------------------------------------------------------------

def _write_fcs(path: Path, keywords: dict) -> None:
    text = "|" + "".join(f"{k}|{v.replace('|', '||')}|" for k, v in keywords.items())
    start = 58
    end = start + len(text.encode()) - 1
    header = b"FCS3.1    " + f"{start:>8}{end:>8}{0:>8}{0:>8}{0:>8}{0:>8}".encode()
    path.write_bytes(header + text.encode())


def test_fcs_spillover_and_scrub(tmp_path, pseudo):
    kw = {"$PAR": "2", "$P1N": "FITC-A", "$P1S": "CD3", "$P2N": "PE-A", "$P2S": "CD4",
          "$SPILLOVER": "2,FITC-A,PE-A,1,0.12,0.03,1", "$FIL": "patient_123.fcs",
          "$DATE": "01-JAN-2026", "PATIENT ID": "MRN 999", "$COM": "a|b"}
    f = tmp_path / "s.fcs"
    _write_fcs(f, kw)
    parsed = read_text_segment(f)
    assert parsed["$COM"] == "a|b"
    m = parse_spillover(parsed["$SPILLOVER"])
    assert m.get("FITC-A", "PE-A") == 0.12
    clean = scrub_keywords(parsed, pseudo)
    assert "PATIENT ID" not in clean and "$DATE" not in clean
    assert clean["$FIL"].startswith("fil_") and "patient" not in json.dumps(clean)
    assert clean["$P1S"] == "CD3"


def test_parse_text_escaped_delimiter():
    assert parse_text("/A/x//y/B/z/") == {"A": "x/y", "B": "z"}


def test_bridge_end_to_end(tmp_path, pseudo):
    f = tmp_path / "sample_JohnDoe.fcs"
    _write_fcs(f, {"$PAR": "3", "$SPILLOVER": "3,FITC-A,PE-A,APC-A,1,0.1,0,0.02,1,0.01,0,0,1",
                   "$FIL": "sample_JohnDoe.fcs"})
    br = FlowIoBridge(tmp_path / "store", pseudo=pseudo)
    sid = br.on_workspace_opened("/Users/jdoe/ws.wsp", sample_paths=[str(f)], user="jdoe",
                                 panel=list(ROWS), detectors=list(COLS),
                                 instrument={"model": "Aurora", "serial": "SN-1"},
                                 controls=[{"fluorochrome": "FITC", "pos_median": {"FITC-A": 5e4}}])
    br.on_compensation_loaded("/Users/jdoe/ws.wsp", mat(0.11).to_dict(), source="auto")
    br.on_view_changed("/Users/jdoe/ws.wsp", "FITC-A", "PE-A")
    br.on_cell_changed("/Users/jdoe/ws.wsp", "FITC", "PE-A", 0.15)
    br.on_export("/Users/jdoe/ws.wsp", "report", target_path="/Users/jdoe/out.pdf")
    br.on_workspace_closed("/Users/jdoe/ws.wsp")
    raw = (tmp_path / "store" / "sessions" / f"{sid}.jsonl").read_text()
    for leak in ("jdoe", "JohnDoe", "SN-1", "out.pdf"):
        assert leak not in raw
    rp = replay(Store(tmp_path / "store").read(sid))
    assert [src for _, src, _, _ in rp.snapshots] == ["acquisition", "auto"]
    assert rp.errors == []


def test_http_server(tmp_path, pseudo):
    br = FlowIoBridge(tmp_path, pseudo=pseudo)
    srv = make_server(br, port=0, token="t0k")
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    def post(body, token="t0k"):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/event", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Capture-Token": token})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert post({"event": "workspace_opened", "workspace_ref": "w"}, token="bad")[0] == 401
        code, body = post([{"event": "workspace_opened", "workspace_ref": "w"},
                           {"event": "compensation_loaded", "workspace_ref": "w", "matrix": mat().to_dict(),
                            "source": "auto"},
                           {"event": "cell_changed", "workspace_ref": "w", "row": "FITC",
                            "col": "PE-A", "new_value": 0.2},
                           {"event": "export", "workspace_ref": "w"}])
        assert code == 200 and body["ok"]
        code, body = post({"event": "cell_changed", "workspace_ref": "nope", "row": "x",
                           "col": "y", "new_value": 1})
        assert code == 400 and "no open session" in body["error"]
        code, body = post({"event": "rm_rf", "workspace_ref": "w"})
        assert code == 400
    finally:
        srv.shutdown()
        br.close_all()
    assert build_dataset(Store(tmp_path), tmp_path / "ds")["labeled"] == 1
