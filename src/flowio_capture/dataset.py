"""Turn raw session logs into training signals.

Labels are derived here, not at capture time, so the scheme can change
without re-collecting data. Each (session, matrix_id) is replayed and
labeled independently. Produced record types (one JSONL file each):

accepted.jsonl
    One per labeled matrix: baseline (state before the scientist's first
    edit in the final segment), accepted matrix (state at the last export),
    per-cell delta, snapshot provenance, the controls in effect, and the
    session context (model inputs). Primary regression target:
    ``accepted - baseline``. Check ``clean_baseline`` before training on the
    delta: it is False when the starting matrix was not an acquisition/auto
    matrix (e.g. a reopened, already hand-edited workspace).

cell_directions.jsonl
    One per judgment edit (user/undo/redo) in the final segment: was the
    pre-edit value too low / too high relative to the accepted value.

preference_pairs.jsonl
    (accepted > earlier state) pairs. ``hard`` marks near-misses.
    ``kind="revised_export"``: exported, then changed -- the strongest
    negatives.

cell_hard_negatives.jsonl
    Values the scientist explicitly rejected: ``undone`` (single-cell undo,
    not later redone), ``reverted`` (A->B->A), ``overshoot`` (moved across the
    accepted value, then came back).

tolerances.jsonl
    Per edited cell, the tightest bracket [lo, hi] around the accepted value
    that the scientist visited and moved away from.

Segments: a snapshot (app recompute, import, layout change, ...) replaces
the matrix wholesale, so edits before it are not judged against the accepted
matrix. Only the *final segment* (after the last snapshot before acceptance)
yields per-cell labels; revised exports from earlier segments are kept as
preference pairs.

Sessions/matrices without a (trustworthy) export are unlabeled and skipped;
the reason is recorded in ``summary.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import schema as S
from .recorder import DEFAULT_MATRIX, JUDGMENT_ORIGINS, matrix_hash
from .schema import Matrix, SchemaError
from .store import Store

RECORD_TYPES = ("accepted", "cell_directions", "preference_pairs", "cell_hard_negatives", "tolerances")
MATRIX_EVENTS = (S.SNAPSHOT, S.CELL_EDIT, S.EXPORT, S.UNDO, S.REDO)


@dataclass
class Edit:
    seq: int
    step: int
    row: str
    col: str
    old: float
    new: float
    origin: str
    view: Optional[Dict[str, Any]]
    dt: Optional[float]
    batch: Optional[str] = None
    desynced: bool = False


@dataclass
class Step:
    step: int
    seq: int
    matrix: Matrix
    origin: str  # "snapshot:<source>" | "user" | "undo" | "redo" | "resync"


@dataclass
class Export:
    seq: int
    step: int
    matrix: Matrix
    kind: str
    desynced: bool
    t: str


@dataclass
class Replay:
    session_id: str
    matrix_id: str
    context: Dict[str, Any]
    snapshots: List[Tuple[int, str, Optional[str], Matrix]] = field(default_factory=list)  # seq, src, label, m
    steps: List[Step] = field(default_factory=list)
    edits: List[Edit] = field(default_factory=list)
    exports: List[Export] = field(default_factory=list)
    controls: List[Tuple[int, List[Dict[str, Any]]]] = field(default_factory=list)
    ended: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    t_start: Optional[str] = None
    t_end: Optional[str] = None


def matrix_ids(events: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for e in events:
        if e.get("type") in MATRIX_EVENTS:
            mid = e.get("data", {}).get("matrix_id", DEFAULT_MATRIX)
            if mid not in ids:
                ids.append(mid)
    return ids


def replay(events: List[Dict[str, Any]], matrix_id: str = DEFAULT_MATRIX) -> Replay:
    if not events or events[0].get("type") != S.SESSION_STARTED:
        raise SchemaError("log does not begin with session_started")
    rp = Replay(session_id=events[0]["session_id"], matrix_id=matrix_id,
                context=events[0]["data"]["context"], t_start=events[0]["t"])
    cur: Optional[Matrix] = None
    step = -1
    open_batch: Optional[str] = None
    gap_reported = False

    for i, e in enumerate(events):
        if e["seq"] != i and not gap_reported:
            rp.errors.append(f"seq gap near {i} (got {e['seq']}): log has missing or corrupt lines")
            gap_reported = True
        t, d = e["type"], e.get("data", {})
        rp.t_end = e["t"]
        if t in MATRIX_EVENTS and d.get("matrix_id", DEFAULT_MATRIX) != matrix_id:
            continue
        if t != S.CELL_EDIT:
            open_batch = None
        if t == S.SNAPSHOT:
            cur = Matrix.from_dict(d["matrix"])
            rp.snapshots.append((e["seq"], d["source"], d.get("label"), cur))
            step += 1
            rp.steps.append(Step(step, e["seq"], cur, "snapshot:" + d["source"]))
        elif t == S.CELL_EDIT:
            if cur is None:
                rp.errors.append(f"edit before any matrix at seq {e['seq']}")
                continue
            try:
                if abs(cur.get(d["row"], d["col"]) - d["old"]) > 1e-9:
                    rp.errors.append(f"old value mismatch at seq {e['seq']}")
                cur = cur.with_value(d["row"], d["col"], d["new"])
            except SchemaError as ex:
                rp.errors.append(f"seq {e['seq']}: {ex}")
                continue
            batch = d.get("batch")
            if batch is not None and batch == open_batch:
                rp.steps[-1] = Step(step, rp.steps[-1].seq, cur, d["origin"])
            else:
                step += 1
                rp.steps.append(Step(step, e["seq"], cur, d["origin"]))
            open_batch = batch
            rp.edits.append(Edit(e["seq"], step, d["row"], d["col"], d["old"], d["new"], d["origin"],
                                 d.get("view"), d.get("dt_since_last_edit"), batch,
                                 bool(d.get("desynced"))))
            if d.get("matrix_hash") and _batch_complete(events, i) and matrix_hash(cur) != d["matrix_hash"]:
                rp.errors.append(f"state hash mismatch at seq {e['seq']}")
        elif t == S.EXPORT:
            if d.get("matrix") is None:
                continue
            m = Matrix.from_dict(d["matrix"])
            if cur is not None and (not cur.same_layout(m) or cur.diff(m)):
                rp.errors.append(f"export matrix differs from replayed state at seq {e['seq']}")
            rp.exports.append(Export(e["seq"], step, m, d.get("kind", "export"),
                                     bool(d.get("desynced")), e["t"]))
        elif t == S.CONTROLS_UPDATED:
            rp.controls.append((e["seq"], d.get("controls", [])))
        elif t == S.SESSION_ENDED:
            rp.ended = d.get("reason")
    return rp


def _batch_complete(events: List[Dict[str, Any]], i: int) -> bool:
    b = events[i]["data"].get("batch")
    if b is None:
        return True
    nxt = events[i + 1] if i + 1 < len(events) else None
    return not (nxt and nxt["type"] == S.CELL_EDIT and nxt["data"].get("batch") == b)


def _sign(x: float, eps: float) -> int:
    return 0 if abs(x) <= eps else (1 if x > 0 else -1)


def _delta(a: Matrix, b: Matrix, eps: float) -> List[Dict[str, Any]]:
    return [{"row": r, "col": c, "from": x, "to": y, "delta": y - x} for r, c, x, y in a.diff(b, tol=eps)]


def _distance(a: Matrix, b: Matrix, eps: float) -> Tuple[int, float, float]:
    d = a.diff(b, tol=eps)
    return len(d), sum(abs(x - y) for _, _, x, y in d), max((abs(x - y) for _, _, x, y in d), default=0.0)


def _qkey(m: Matrix, eps: float) -> Tuple:
    """Dedup key: matrices equal within eps (up to rounding) share a key."""
    return (m.rows, m.cols, tuple(round(v / eps) for _, _, v in m.cells()))


def derive(rp: Replay, eps: float = 1e-4, hard_max_cells: int = 2,
           hard_max_abs: float = 0.05) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
    """Training records for one replayed matrix, plus a skip reason (or None).

    ``eps`` is the smallest spillover difference treated as meaningful.
    """
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in RECORD_TYPES}
    if not rp.exports:
        return out, "no export (unlabeled)"
    acc = rp.exports[-1]
    if acc.desynced:
        return out, "final export made while recorder was desynced from host"
    accepted, acc_step, acc_seq = acc.matrix, acc.step, acc.seq
    base = {"session_id": rp.session_id, "matrix_id": rp.matrix_id}

    # ---- final segment ---------------------------------------------------------
    snap_steps = [s.step for s in rp.steps if s.origin.startswith("snapshot:") and s.step <= acc_step]
    seg_start = snap_steps[-1] if snap_steps else 0
    seg_source = rp.steps[seg_start].origin.split(":", 1)[-1] if rp.steps else "unknown"
    seg_edits = [e for e in rp.edits if seg_start < e.step <= acc_step and e.seq < acc_seq]
    judged = [e for e in seg_edits if e.origin in JUDGMENT_ORIGINS and not e.desynced]
    first_step = min((e.step for e in judged), default=None)
    edited_before_segment = any(e.origin in JUDGMENT_ORIGINS and e.step < seg_start for e in rp.edits)

    if first_step is None:
        baseline, baseline_origin = accepted, "accepted_unedited"
    else:
        prev = rp.steps[first_step - 1]
        baseline, baseline_origin = prev.matrix, prev.origin
    same_layout = baseline.same_layout(accepted)

    controls = rp.context.get("controls", [])
    for seq, c in rp.controls:
        if seq < acc_seq:
            controls = c

    out["accepted"].append({
        **base, "context": rp.context, "controls_at_accept": controls, "export_kind": acc.kind,
        "segment_source": seg_source, "baseline_origin": baseline_origin,
        "clean_baseline": seg_source in S.CLEAN_BASELINE_SOURCES,
        "snapshot_after_first_edit": edited_before_segment,
        "baseline": baseline.to_dict(), "accepted": accepted.to_dict(),
        "delta": _delta(baseline, accepted, eps) if same_layout else None,
        "snapshots": [{"seq": s, "source": src, "label": lbl, "matrix": m.to_dict()}
                      for s, src, lbl, m in rp.snapshots if s < acc_seq],
        "n_judgment_edits": len(judged), "n_exports": len(rp.exports),
        "t_start": rp.t_start, "t_accepted": acc.t, "replay_errors": rp.errors,
    })

    # ---- per-edit direction labels -------------------------------------------
    for e in judged:
        try:
            final = accepted.get(e.row, e.col)
        except SchemaError:
            continue
        before = _sign(final - e.old, eps)   # +1: value was too low
        move = _sign(e.new - e.old, eps)
        out["cell_directions"].append({
            **base, "row": e.row, "col": e.col, "origin": e.origin,
            "value_before": e.old, "value_after": e.new, "accepted_value": final,
            "label_before": before, "label_after": _sign(final - e.new, eps),
            "moved_toward_accepted": move != 0 and move == before,
            "view": e.view, "dt_since_last_edit": e.dt, "step": e.step,
        })

    # ---- preference pairs ------------------------------------------------------
    good_exports = [x for x in rp.exports[:-1] if not x.desynced and x.matrix.same_layout(accepted)]
    revised = {_qkey(x.matrix, eps) for x in good_exports}
    seen = {_qkey(accepted, eps)}
    desynced_steps = {e.step for e in seg_edits if e.desynced}

    def pair(m: Matrix, kind: str, st: int) -> None:
        k = _qkey(m, eps)
        if k in seen:
            return
        seen.add(k)
        n, l1, linf = _distance(accepted, m, eps)
        if n == 0:
            return
        out["preference_pairs"].append({
            **base, "chosen": accepted.to_dict(), "rejected": m.to_dict(),
            "kind": "revised_export" if k in revised else kind, "step": st,
            "n_cells_diff": n, "l1": l1, "linf": linf,
            "hard": n <= hard_max_cells and linf <= hard_max_abs,
            "diff": _delta(m, accepted, eps),
        })

    for x in good_exports:
        pair(x.matrix, "revised_export", x.step)
    if first_step is not None:
        for s in rp.steps[first_step - 1:acc_step + 1]:
            if s.origin == "resync" or s.step in desynced_steps or not s.matrix.same_layout(accepted):
                continue
            pair(s.matrix, "baseline" if s.step == first_step - 1 else "intermediate", s.step)

    # ---- per-cell chains: hard negatives and tolerance -------------------------
    batch_sizes: Dict[str, int] = {}
    for e in seg_edits:
        if e.batch:
            batch_sizes[e.batch] = batch_sizes.get(e.batch, 0) + 1
    chains: Dict[Tuple[str, str], List[List[Edit]]] = {}
    for e in seg_edits:
        cell = chains.setdefault((e.row, e.col), [[]])
        if e.origin not in JUDGMENT_ORIGINS or e.desynced:
            cell.append([])  # a correction breaks the chain; it is not a judgment
            continue
        if cell[-1] and abs(e.old - cell[-1][-1].new) > 1e-9:
            cell.append([])
        cell[-1].append(e)

    for (r, c), cell_chains in chains.items():
        try:
            final = accepted.get(r, c)
        except SchemaError:
            continue
        rejected: Dict[float, str] = {}
        for chain in cell_chains:
            for k, e in enumerate(chain):
                if (e.origin == "undo" and batch_sizes.get(e.batch or "", 1) == 1
                        and abs(e.old - final) > eps
                        and not any(x.origin == "redo" and abs(x.new - e.old) <= eps for x in chain[k + 1:])):
                    rejected.setdefault(e.old, "undone")
                if k == 0:
                    continue
                p = chain[k - 1]
                if p.origin != "user" or e.origin != "user":
                    continue
                if abs(e.new - p.old) <= eps and abs(p.new - final) > eps:
                    rejected.setdefault(p.new, "reverted")
                d1, d2 = _sign(p.new - p.old, eps), _sign(e.new - e.old, eps)
                if (d1 != 0 and d2 == -d1 and _sign(p.old - final, eps) == -d1
                        and _sign(p.new - final, eps) == d1):
                    rejected.setdefault(p.new, "overshoot")
        for v, why in rejected.items():
            out["cell_hard_negatives"].append({
                **base, "row": r, "col": c, "rejected_value": v,
                "accepted_value": final, "error": v - final, "reason": why,
            })
        last = next((ch for ch in reversed(cell_chains) if ch), None)
        if not last or abs(last[-1].new - final) > eps:
            continue  # this cell's final value was not set by its last chain
        visited = [e.old for e in last] + [e.new for e in last[:-1]]
        lo = [v for v in visited if v < final - eps]
        hi = [v for v in visited if v > final + eps]
        if lo or hi:
            out["tolerances"].append({
                **base, "row": r, "col": c, "accepted_value": final,
                "lo": max(lo) if lo else None, "hi": min(hi) if hi else None,
                "n_edits": len(last),
            })
    return out, None


def build_dataset(store: Store, out_dir: Path, **kw: Any) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {n: open(out_dir / f"{n}.jsonl", "w", encoding="utf-8") for n in RECORD_TYPES}
    summary: Dict[str, Any] = {"sessions": 0, "labeled": 0, "skipped": {},
                               "counts": {n: 0 for n in RECORD_TYPES},
                               "with_replay_errors": [], "unclean_baselines": 0}
    try:
        for sid in store.session_ids():
            summary["sessions"] += 1
            try:
                events = store.read(sid)
                replay(events)  # validates the header
                ids = matrix_ids(events)
            except (SchemaError, KeyError, ValueError, TypeError) as ex:
                summary["skipped"][sid] = f"unreadable: {ex}"
                continue
            if not ids:
                summary["skipped"][sid] = "no matrix recorded"
                continue
            for mid in ids:
                key = sid if mid == DEFAULT_MATRIX else f"{sid}/{mid}"
                try:
                    rp = replay(events, mid)
                    recs, skip = derive(rp, **kw)
                except (SchemaError, KeyError, ValueError, TypeError) as ex:
                    summary["skipped"][key] = f"unreadable: {ex}"
                    continue
                if skip:
                    summary["skipped"][key] = skip
                    continue
                if rp.errors:
                    summary["with_replay_errors"].append({"key": key, "errors": rp.errors[:5]})
                summary["labeled"] += 1
                if not recs["accepted"][0]["clean_baseline"]:
                    summary["unclean_baselines"] += 1
                for n in RECORD_TYPES:
                    for rec in recs[n]:
                        files[n].write(json.dumps(rec, separators=(",", ":")) + "\n")
                    summary["counts"][n] += len(recs[n])
    finally:
        for fh in files.values():
            fh.close()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
