"""Session recorder: the API a Flow.Io plugin (or any host) calls.

Typical flow::

    rec = CaptureSession.start(store, context)
    rec.snapshot(acq_matrix, source="acquisition")
    rec.snapshot(auto_matrix, source="auto")        # if the app auto-computes
    rec.set_view("FITC-A", "PE-A")                  # user looking at a plot
    rec.edit_cell("FITC", "PE-A", 0.18)             # fine-grained edit, or
    rec.observe_matrix(current_matrix)              # host only reports whole matrices
    rec.export("report")                            # implicit acceptance
    rec.end()

The recorder keeps its own copy of the current matrix so that hosts which
only emit "matrix changed" notifications still yield per-cell edits.

A workspace may hold several compensation matrices (e.g. one per sample
group). Every matrix event takes a ``matrix_id``; state, undo history and
labels are kept separately per matrix_id.

When the recorder can no longer be sure its copy matches the host (an undo
it cannot reproduce, a stale undo stack), it marks that matrix *desynced*:
subsequent edits and exports carry ``"desynced": true`` and are excluded
from labels until the host reports the true matrix again.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import schema as S
from .plots import PlotSummary
from .schema import ControlStats, Matrix, SchemaError, SessionContext
from .store import SessionWriter, Store

DEFAULT_MATRIX = "default"
JUDGMENT_ORIGINS = {"user", "undo", "redo"}
DEFAULT_PLOT_BUDGET = 64 << 20  # bytes of plot JSON per session


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def matrix_hash(m: Matrix) -> str:
    payload = json.dumps(m.to_dict(), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


Change = Tuple[str, str, float, float]  # (row, col, old, new)


@dataclass
class _MatrixState:
    current: Optional[Matrix] = None
    undo: List[List[Change]] = field(default_factory=list)
    redo: List[List[Change]] = field(default_factory=list)
    desynced: bool = False


class CaptureSession:
    def __init__(self, writer: SessionWriter, clock: Callable[[], float] = time.time,
                 plot_budget_bytes: int = DEFAULT_PLOT_BUDGET):
        self._w = writer
        self._clock = clock
        self._lock = threading.RLock()
        self.session_id = writer.session_id
        self._m: Dict[str, _MatrixState] = {}
        self.view: Optional[Dict[str, Any]] = None
        self._ended = False
        self._last_edit_ts: Optional[float] = None
        # Plots: latest plot seq per (x, y, kind), attached to subsequent edits.
        self.plot_budget_bytes = plot_budget_bytes
        self._plot_bytes = 0
        self._plot_hashes: Dict[str, int] = {}
        self._plots_for_view: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._plot_budget_noted = False

    # ---- lifecycle ---------------------------------------------------------
    @classmethod
    def start(cls, store: Store, context: SessionContext, session_id: Optional[str] = None,
              clock: Callable[[], float] = time.time,
              plot_budget_bytes: int = DEFAULT_PLOT_BUDGET) -> "CaptureSession":
        sid = session_id or uuid.uuid4().hex
        path = store.path_for(sid)
        if path.exists():
            raise SchemaError(f"session {sid} already exists")
        ctx = context.to_dict()
        _check_finite(ctx, "context")
        writer = store.open_writer(sid)
        sess = cls(writer, clock=clock, plot_budget_bytes=plot_budget_bytes)
        try:
            sess._emit(S.SESSION_STARTED, {"context": ctx})
        except Exception:
            writer.close()
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
            raise
        return sess

    def end(self, reason: str = "closed") -> None:
        with self._lock:
            if self._ended:
                return
            self._emit(S.SESSION_ENDED, {"reason": reason, "matrices": {
                mid: self._state(mid) for mid in self._m}})
            self._ended = True
            self._w.close()

    def __enter__(self) -> "CaptureSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end("error" if exc_type else "closed")

    @property
    def current(self) -> Optional[Matrix]:
        st = self._m.get(DEFAULT_MATRIX)
        return st.current if st else None

    def matrix(self, matrix_id: str = DEFAULT_MATRIX) -> Optional[Matrix]:
        st = self._m.get(matrix_id)
        return st.current if st else None

    # ---- matrix state ------------------------------------------------------
    def snapshot(self, matrix: Matrix, source: str, label: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        """Record a whole matrix that did not come from cell-by-cell editing.

        Use for the acquisition ($SPILLOVER) matrix, the app's auto-computed
        matrix, an imported template, a reloaded workspace, or a layout
        change. Resets undo history and clears desync, since the host has
        told us the true state.
        """
        if source not in S.SNAPSHOT_SOURCES:
            raise SchemaError(f"unknown snapshot source {source!r}")
        with self._lock:
            self._check_open()
            st = self._st(matrix_id)
            st.current = matrix
            st.undo.clear()
            st.redo.clear()
            st.desynced = False
            self._emit(S.SNAPSHOT, {"matrix_id": matrix_id, "source": source, "label": label,
                                    "meta": meta or {}, "matrix": matrix.to_dict(),
                                    "matrix_hash": matrix_hash(matrix)})

    def edit_cell(self, row: str, col: str, new_value: float, old_value: Optional[float] = None,
                  meta: Optional[Dict[str, Any]] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        """A single user edit. ``old_value`` is checked against tracked state."""
        new_value = _finite(new_value, "new_value")
        with self._lock:
            self._check_open()
            st = self._st(matrix_id)
            cur = self._require_current(st)
            tracked = cur.get(row, col)
            if old_value is not None:
                old_value = _finite(old_value, "old_value")
                if abs(old_value - tracked) > 1e-9:
                    # Host and recorder disagree: we missed an event. Logged as
                    # a "resync" edit so replay stays consistent; dataset
                    # building never treats resync as the scientist's judgment.
                    self._apply(st, matrix_id, [(row, col, tracked, old_value)], origin="resync")
                    tracked = old_value
            if abs(new_value - tracked) <= 1e-12:
                return
            ch = [(row, col, tracked, new_value)]
            self._apply(st, matrix_id, ch, origin="user", meta=meta)
            st.undo.append(ch)
            st.redo.clear()

    def observe_matrix(self, matrix: Matrix, origin: str = "user",
                       meta: Optional[Dict[str, Any]] = None, matrix_id: str = DEFAULT_MATRIX) -> int:
        """Host reported the full current matrix; record the per-cell delta.

        Returns the number of changed cells. A layout change (or the first
        matrix seen) is recorded as a snapshot. A full matrix from the host
        is authoritative, so it clears desync.
        """
        with self._lock:
            self._check_open()
            st = self._st(matrix_id)
            if st.current is None or not st.current.same_layout(matrix):
                self.snapshot(matrix, source="external" if st.current is not None else "unknown",
                              label="layout_change" if st.current is not None else "first_observed",
                              matrix_id=matrix_id)
                return 0
            was_desynced = st.desynced
            st.desynced = False
            changes = st.current.diff(matrix)
            if was_desynced and changes:
                origin = "resync"
            if not changes:
                return 0
            self._apply(st, matrix_id, changes, origin=origin, meta=meta)
            if origin == "user":
                st.undo.append(changes)
                st.redo.clear()
            else:
                st.undo.clear()
                st.redo.clear()
            return len(changes)

    def undo(self, matrix_after: Optional[Matrix] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        """Host performed undo. Pass the resulting matrix whenever the host knows
        it; without it the recorder replays its own undo stack, which can only
        be trusted if every edit went through this recorder."""
        self._undo_redo(S.UNDO, matrix_after, matrix_id)

    def redo(self, matrix_after: Optional[Matrix] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        self._undo_redo(S.REDO, matrix_after, matrix_id)

    # ---- context signals ---------------------------------------------------
    def set_view(self, x_channel: Optional[str], y_channel: Optional[str],
                 population: Optional[str] = None, **extra: Any) -> None:
        """What the scientist is looking at. Attached to subsequent edits."""
        with self._lock:
            self._check_open()
            view = {"x": x_channel, "y": y_channel, "population": population, **extra}
            if view == self.view:
                return
            self.view = view
            self._emit(S.VIEW_CHANGED, dict(view))

    def observe_plot(self, plot: PlotSummary, matrix_id: str = DEFAULT_MATRIX) -> Optional[int]:
        """The host refreshed a plot. Returns the event seq, or None if dropped.

        The plot is tagged with the hash of the matrix it was compensated
        with, and its seq is attached (as ``plot_refs``) to every later edit
        made while the same channel pair is on screen. Identical plots are
        logged once; a per-session byte budget bounds the log size.
        """
        with self._lock:
            self._check_open()
            data = plot.to_dict()
            data["matrix_id"] = matrix_id
            data["matrix_hash"] = self._state(matrix_id)["matrix_hash"]
            key = (plot.x, plot.y)
            h = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
            if h in self._plot_hashes:
                seq = self._plot_hashes[h]
                self._plots_for_view.setdefault(key, {})[plot.kind] = seq
                return seq
            size = len(json.dumps(data, separators=(",", ":")))
            if self._plot_bytes + size > self.plot_budget_bytes:
                if not self._plot_budget_noted:
                    self._plot_budget_noted = True
                    self._emit(S.NOTE, {"kind": "plot_budget_exhausted", "bytes": self._plot_bytes})
                return None
            self._plot_bytes += size
            rec = self._emit(S.PLOT, data)
            self._plot_hashes[h] = rec["seq"]
            self._plots_for_view.setdefault(key, {})[plot.kind] = rec["seq"]
            return rec["seq"]

    def _plot_refs(self) -> Optional[Dict[str, int]]:
        if not self.view or not self.view.get("x") or not self.view.get("y"):
            return None
        refs = self._plots_for_view.get((self.view["x"], self.view["y"]))
        return dict(refs) if refs else None

    def update_controls(self, controls: List[ControlStats]) -> None:
        """Control statistics changed (e.g. the scientist re-gated a control)."""
        with self._lock:
            self._check_open()
            data = {"controls": [c.__dict__ if isinstance(c, ControlStats) else dict(c) for c in controls]}
            _check_finite(data, "controls")
            self._emit(S.CONTROLS_UPDATED, data)

    def export(self, kind: str = "export", target: Optional[str] = None,
               meta: Optional[Dict[str, Any]] = None, matrix: Optional[Matrix] = None,
               matrix_id: str = DEFAULT_MATRIX) -> None:
        """Implicit acceptance: the matrix in use when results left the app.

        If the host passes the exported ``matrix``, it is reconciled first,
        atomically with the export, so the accepted matrix is exactly what
        was used. ``target`` must already be pseudonymized.
        """
        with self._lock:
            self._check_open()
            st = self._st(matrix_id)
            if matrix is not None:
                self.observe_matrix(matrix, origin="resync", matrix_id=matrix_id)
            self._require_current(st)
            self._emit(S.EXPORT, {"matrix_id": matrix_id, "kind": kind, "target": target,
                                  "meta": meta or {}, "desynced": st.desynced, **self._state(matrix_id)})

    def note(self, kind: str, **data: Any) -> None:
        with self._lock:
            self._check_open()
            self._emit(S.NOTE, {"kind": kind, **data})

    # ---- internals ---------------------------------------------------------
    def _undo_redo(self, type_: str, matrix_after: Optional[Matrix], matrix_id: str) -> None:
        with self._lock:
            self._check_open()
            st = self._st(matrix_id)
            cur = self._require_current(st)
            src, dst = (st.undo, st.redo) if type_ == S.UNDO else (st.redo, st.undo)
            if matrix_after is not None:
                if not cur.same_layout(matrix_after):
                    self._emit(type_, {"matrix_id": matrix_id, "n_cells": None, "layout_change": True})
                    self.snapshot(matrix_after, source="external", label=f"{type_}_layout_change",
                                  matrix_id=matrix_id)
                    return
                changes = cur.diff(matrix_after)
                origin = type_
                if st.desynced:
                    origin = "resync"  # host's matrix is truth, but this is not a judgment
                    st.desynced = False
                    src.clear()
                    dst.clear()
                elif src and _inverse_matches(src[-1], changes, type_):
                    dst.append(src.pop())
                else:
                    src.clear()
                    dst.clear()
            else:
                if not src or not _batch_applies(src[-1], cur, type_):
                    # We cannot reproduce what the host did: stop trusting our copy.
                    st.desynced = True
                    src.clear()
                    dst.clear()
                    self._emit(S.NOTE, {"kind": f"{type_}_unreproducible", "matrix_id": matrix_id,
                                        "desync": True})
                    return
                batch = src.pop()
                dst.append(batch)
                changes = ([(r, c, b, a) for r, c, a, b in batch] if type_ == S.UNDO
                           else list(batch))
                origin = type_
            self._emit(type_, {"matrix_id": matrix_id, "n_cells": len(changes)})
            if changes:
                self._apply(st, matrix_id, changes, origin=origin)

    def _apply(self, st: _MatrixState, matrix_id: str, changes: List[Change], origin: str,
               meta: Optional[Dict[str, Any]] = None) -> None:
        cur = self._require_current(st)
        now = self._clock()
        dt = None
        if origin in JUDGMENT_ORIGINS:
            dt = None if self._last_edit_ts is None else round(now - self._last_edit_ts, 3)
            self._last_edit_ts = now
        else:
            # A correction invalidates undo history that touched these cells.
            touched = {(r, c) for r, c, _, _ in changes}
            st.undo[:] = [b for b in st.undo if not touched & {(r, c) for r, c, _, _ in b}]
            st.redo[:] = [b for b in st.redo if not touched & {(r, c) for r, c, _, _ in b}]
        batch = uuid.uuid4().hex[:12] if len(changes) > 1 else None
        for r, c, a, b in changes:
            cur = cur.with_value(r, c, b)
        st.current = cur
        h = matrix_hash(cur)
        for r, c, a, b in changes:
            self._emit(S.CELL_EDIT, {"matrix_id": matrix_id, "row": r, "col": c, "old": a, "new": b,
                                     "origin": origin, "batch": batch,
                                     "view": dict(self.view) if self.view else None,
                                     "plot_refs": self._plot_refs(),
                                     "dt_since_last_edit": dt, "meta": meta or {},
                                     "desynced": st.desynced, "matrix_hash": h}, ts=now)

    def _st(self, matrix_id: str) -> _MatrixState:
        if not isinstance(matrix_id, str) or not matrix_id:
            raise SchemaError("matrix_id must be a non-empty string")
        return self._m.setdefault(matrix_id, _MatrixState())

    def _state(self, matrix_id: str) -> Dict[str, Any]:
        st = self._m.get(matrix_id)
        if st is None or st.current is None:
            return {"matrix": None, "matrix_hash": None}
        return {"matrix": st.current.to_dict(), "matrix_hash": matrix_hash(st.current)}

    @staticmethod
    def _require_current(st: _MatrixState) -> Matrix:
        if st.current is None:
            raise SchemaError("no matrix yet: call snapshot() or observe_matrix() first")
        return st.current

    def _check_open(self) -> None:
        if self._ended:
            raise RuntimeError(f"session {self.session_id} has ended")

    def _emit(self, type_: str, data: Dict[str, Any], ts: Optional[float] = None) -> Dict[str, Any]:
        return self._w.write(type_, data, _iso(self._clock() if ts is None else ts))


def _batch_applies(batch: List[Change], cur: Matrix, type_: str) -> bool:
    """The stack entry is consistent with the current matrix."""
    for r, c, a, b in batch:
        expect = b if type_ == S.UNDO else a
        try:
            if abs(cur.get(r, c) - expect) > 1e-9:
                return False
        except SchemaError:
            return False
    return True


def _inverse_matches(batch: List[Change], changes: List[Change], type_: str) -> bool:
    expected = {(r, c): (b, a) if type_ == S.UNDO else (a, b) for r, c, a, b in batch}
    got = {(r, c): (a, b) for r, c, a, b in changes}
    if expected.keys() != got.keys():
        return False
    return all(abs(expected[k][0] - got[k][0]) < 1e-9 and abs(expected[k][1] - got[k][1]) < 1e-9
               for k in got)


def _finite(v: Any, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SchemaError(f"{name} must be a number")
    if not math.isfinite(f):
        raise SchemaError(f"{name} must be finite")
    return f


def _check_finite(obj: Any, where: str) -> None:
    try:
        json.dumps(obj, allow_nan=False)
    except ValueError:
        raise SchemaError(f"{where} contains NaN/inf")
    except TypeError as ex:
        raise SchemaError(f"{where} is not JSON-serializable: {ex}")
