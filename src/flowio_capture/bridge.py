"""Bridge between Flow.Io plugin events and capture sessions.

Flow.Io's plugin SDK is not wired in yet. This class exposes one handler per
host event you need to hook; the Flow.Io plugin should call these (directly
from Python, or by POSTing JSON to ``server.py`` from any language).

    Flow.Io event (to map)              -> bridge handler
    ----------------------------------------------------------------------
    workspace / experiment opened       -> on_workspace_opened
    compensation computed or loaded     -> on_compensation_loaded (source required)
    single cell edited in comp editor   -> on_cell_changed
    matrix changed (no cell detail)     -> on_matrix_changed
    bivariate plot / NxN cell focused   -> on_view_changed
    control re-gated / stats changed    -> on_controls_updated
    undo / redo in comp editor          -> on_undo / on_redo  (send matrix_after!)
    export, report, or batch analysis   -> on_export   (acceptance signal)
    workspace closed / app quit         -> on_workspace_closed

Every matrix handler takes ``matrix_id`` (default "default") for workspaces
with several compensation matrices. Matrices are given in the host's own
convention (``MatrixConvention``: spillover or compensation, fraction or
percent) and stored canonically as spillover fractions.

Privacy: workspace paths, sample/control files, user names, export paths and
any free-form strings (``extra``, unknown instrument keys, view extras) are
pseudonymized before they reach disk.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import fcs
from .normalize import MatrixConvention
from .privacy import Pseudonymizer, keyed_file_token, regular_file, sanitize, scrub_keywords
from .recorder import DEFAULT_MATRIX, CaptureSession
from .schema import ControlStats, Matrix, SchemaError, SessionContext
from .store import Store

# Instrument fields kept verbatim (configuration, not identity).
SAFE_INSTRUMENT_KEYS = {"model", "manufacturer", "cytometer_type", "lasers", "filters", "detectors",
                        "voltages", "gains", "configuration", "spectral", "n_detectors"}
SERIAL_KEYS = {"serial", "serial_number", "sn", "cytsn"}
# Fixed vocabularies; anything else is tokenized.
EXPORT_KINDS = {"export", "report", "batch", "table", "fcs_export", "layout", "statistics", "save"}
CLOSE_REASONS = {"closed", "quit", "reopened", "crashed", "shutdown", "server_stopped", "error"}
SNAPSHOT_LABELS = {"fcs_spillover", "autospill", "wizard", "recompute", "template", "reload"}
SAFE_VIEW_KEYS = {"x", "y", "population", "plot_type", "transform"}


class FlowIoBridge:
    def __init__(self, root: os.PathLike, pseudo: Optional[Pseudonymizer] = None,
                 app_version: Optional[str] = None, convention: Optional[MatrixConvention] = None):
        self.store = Store(root)
        self.pseudo = pseudo or Pseudonymizer.load(root)
        self.app_version = app_version
        self.default_convention = convention
        self._sessions: Dict[str, CaptureSession] = {}
        self._conventions: Dict[str, MatrixConvention] = {}
        self._seen_ids: "OrderedDict[Tuple[str, str], None]" = OrderedDict()
        self._lock = threading.RLock()

    # ---- lifecycle -----------------------------------------------------------
    def on_workspace_opened(self, workspace_ref: str, sample_paths: Iterable[str] = (),
                            panel: Iterable[str] = (), detectors: Iterable[str] = (),
                            instrument: Optional[Dict[str, Any]] = None,
                            controls: Iterable[Dict[str, Any]] = (),
                            user: Optional[str] = None,
                            extra: Optional[Dict[str, Any]] = None,
                            matrix_kind: Optional[str] = None, matrix_unit: Optional[str] = None,
                            read_fcs_spillover: bool = True) -> str:
        """Start a session; returns its id. Keyed by ``workspace_ref``.

        ``panel[i]`` should be the fluorochrome measured primarily on
        ``detectors[i]``; that pairing relabels the FCS $SPILLOVER matrix
        (detector names on both axes) to fluorochrome rows, and validates
        diagonals. If sample files are readable, the acquisition spillover
        matrix is recorded as the first snapshot.
        """
        key = self._key(workspace_ref)
        panel, detectors = list(panel), list(detectors)
        base = self.default_convention
        conv = MatrixConvention.from_panel(
            panel, detectors,
            kind=matrix_kind or (base.kind if base else "spillover"),
            unit=matrix_unit or (base.unit if base else "fraction"))
        # File I/O happens before taking the lock (FCS files can be large).
        samples, acq = [], None
        for p in sample_paths:
            tok, kw, m = self._inspect_fcs(p, read_fcs_spillover)
            samples.append({"token": tok, "keywords": kw})
            if acq is None and m is not None:
                acq = m
        ctx = SessionContext(
            experiment_id=self.pseudo.token(workspace_ref, "workspace"),
            sample_hashes=[s["token"] for s in samples],
            panel=panel, detectors=detectors,
            instrument=self._instrument(instrument or {}),
            controls=[self._control(c) for c in controls],
            user_id=self.pseudo.token(user, "user"), app_version=self.app_version,
            extra={"host": sanitize(extra or {}, self.pseudo, "extra"),
                   "sample_keywords": {s["token"]: s["keywords"] for s in samples if s["keywords"]},
                   "primary_detector": conv.primary,
                   "host_matrix_convention": {"kind": conv.kind, "unit": conv.unit}},
        )
        acq_canon = None
        if acq is not None:
            try:  # FCS $SPILLOVER is always spillover fractions, detector-labeled
                acq_canon = MatrixConvention(primary=conv.primary).to_canonical(acq)
            except SchemaError:
                acq_canon = None
        sess = CaptureSession.start(self.store, ctx)  # validates ctx before touching old session
        with self._lock:
            old = self._sessions.pop(key, None)
            self._sessions[key] = sess
            self._conventions[key] = conv
        if old:
            old.end("reopened")
        if acq_canon is not None:
            sess.snapshot(acq_canon, source="acquisition", label="fcs_spillover")
        elif acq is not None:
            sess.note("acquisition_spillover_rejected")
        return sess.session_id

    def on_workspace_closed(self, workspace_ref: str, reason: str = "closed") -> None:
        with self._lock:
            sess = self._sessions.pop(self._key(workspace_ref), None)
            self._conventions.pop(workspace_ref, None)
        if sess:
            sess.end(self._vocab(reason, CLOSE_REASONS, "reason"))

    def close_all(self, reason: str = "shutdown") -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
            self._conventions.clear()
        for s in sessions:
            s.end(self._vocab(reason, CLOSE_REASONS, "reason"))

    # ---- matrix events -----------------------------------------------------
    def on_compensation_loaded(self, workspace_ref: str, matrix: Any, source: str,
                               label: Optional[str] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        """``source``: auto | acquisition | import | saved_workspace | manual | unknown."""
        sess, conv = self._get(workspace_ref)
        sess.snapshot(self._canon(conv, matrix), source=source,
                      label=None if label is None else self._vocab(label, SNAPSHOT_LABELS, "label"),
                      matrix_id=matrix_id)

    def on_cell_changed(self, workspace_ref: str, row: str, col: str, new_value: float,
                        old_value: Optional[float] = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        sess, conv = self._get(workspace_ref)
        row = conv.detector_to_fluor.get(row, row) if row not in conv.primary else row
        cur = sess.matrix(matrix_id)
        if conv.kind == "compensation":
            # One compensation cell maps to many spillover cells: apply the
            # edit in host space and record the full canonical result.
            if cur is None:
                raise SchemaError("no matrix yet: call on_compensation_loaded first")
            host = conv.from_canonical(cur)
            if old_value is not None and abs(host.get(row, col) - float(old_value)) > 1e-6:
                sess.note("compensation_old_value_mismatch", matrix_id=matrix_id)
            sess.observe_matrix(conv.to_canonical(host.with_value(row, col, float(new_value))),
                                matrix_id=matrix_id)
            return
        scale = 100.0 if conv.unit == "percent" else 1.0
        sess.edit_cell(row, col, float(new_value) / scale,
                       None if old_value is None else float(old_value) / scale, matrix_id=matrix_id)

    def on_matrix_changed(self, workspace_ref: str, matrix: Any, matrix_id: str = DEFAULT_MATRIX) -> int:
        sess, conv = self._get(workspace_ref)
        return sess.observe_matrix(self._canon(conv, matrix), matrix_id=matrix_id)

    def on_view_changed(self, workspace_ref: str, x: Optional[str], y: Optional[str],
                        population: Optional[str] = None, **extra: Any) -> None:
        sess, _ = self._get(workspace_ref)
        safe = {k: v for k, v in extra.items() if k in SAFE_VIEW_KEYS and not isinstance(v, (dict, list))}
        other = {k: v for k, v in extra.items() if k not in safe}
        if other:
            safe["extra"] = sanitize(other, self.pseudo, "view")
        sess.set_view(x, y, population, **safe)

    def on_controls_updated(self, workspace_ref: str, controls: Iterable[Dict[str, Any]]) -> None:
        sess, _ = self._get(workspace_ref)
        sess.update_controls([self._control(c) for c in controls])

    def on_undo(self, workspace_ref: str, matrix_after: Any = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        sess, conv = self._get(workspace_ref)
        sess.undo(None if matrix_after is None else self._canon(conv, matrix_after), matrix_id=matrix_id)

    def on_redo(self, workspace_ref: str, matrix_after: Any = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        sess, conv = self._get(workspace_ref)
        sess.redo(None if matrix_after is None else self._canon(conv, matrix_after), matrix_id=matrix_id)

    def on_export(self, workspace_ref: str, kind: str = "export", target_path: Optional[str] = None,
                  matrix: Any = None, matrix_id: str = DEFAULT_MATRIX) -> None:
        """Acceptance signal. Pass the matrix actually used for the export when
        available; it is reconciled atomically with the export record."""
        sess, conv = self._get(workspace_ref)
        sess.export(kind=self._vocab(kind, EXPORT_KINDS, "kind"),
                    target=self.pseudo.token(target_path, "export"),
                    matrix=None if matrix is None else self._canon(conv, matrix), matrix_id=matrix_id)

    # ---- generic dispatch (used by the HTTP server) -----------------------
    HANDLERS = ("on_workspace_opened", "on_workspace_closed", "on_compensation_loaded",
                "on_cell_changed", "on_matrix_changed", "on_view_changed", "on_controls_updated",
                "on_undo", "on_redo", "on_export")
    MAX_SEEN_IDS = 100_000

    def handle(self, event: Any) -> Any:
        """``{"event": "cell_changed", "workspace_ref": ..., "event_id": optional, ...}``

        ``event_id`` makes retries idempotent: a repeated id is acknowledged
        and ignored. Rejected events for an open session are logged as notes
        so lost edits can be counted.
        """
        if not isinstance(event, dict):
            raise SchemaError("event must be a JSON object")
        name = event.get("event")
        if not isinstance(name, str):
            raise SchemaError("missing 'event' name")
        handler = name if name.startswith("on_") else f"on_{name}"
        if handler not in self.HANDLERS:
            raise SchemaError(f"unknown event {name!r}")
        args = {k: v for k, v in event.items() if k not in ("event", "event_id")}
        eid = event.get("event_id")
        dedup_key = (str(args.get("workspace_ref")), str(eid)) if eid is not None else None
        with self._lock:
            if dedup_key and dedup_key in self._seen_ids:
                return {"duplicate": True}
        try:
            result = getattr(self, handler)(**args)
        except TypeError as ex:
            raise SchemaError(f"bad arguments for {name}: {ex}")
        except SchemaError as ex:
            self._note_rejected(args.get("workspace_ref"), name, str(ex))
            raise
        if dedup_key:
            with self._lock:
                self._seen_ids[dedup_key] = None
                while len(self._seen_ids) > self.MAX_SEEN_IDS:
                    self._seen_ids.popitem(last=False)
        return result

    # ---- helpers -------------------------------------------------------------
    def _note_rejected(self, workspace_ref: Any, name: str, error: str) -> None:
        with self._lock:
            sess = self._sessions.get(workspace_ref) if isinstance(workspace_ref, str) else None
        if sess:
            try:
                sess.note("rejected_event", event=name, error=error[:300])
            except RuntimeError:
                pass

    def _key(self, workspace_ref: Any) -> str:
        if not isinstance(workspace_ref, str) or not workspace_ref:
            raise SchemaError("workspace_ref is required")
        return workspace_ref

    def _get(self, workspace_ref: Any) -> Tuple[CaptureSession, MatrixConvention]:
        key = self._key(workspace_ref)
        with self._lock:
            sess = self._sessions.get(key)
            conv = self._conventions.get(key)
        if sess is None or conv is None:
            raise SchemaError(f"no open session for workspace {workspace_ref!r}; "
                              "call on_workspace_opened first")
        return sess, conv

    def _canon(self, conv: MatrixConvention, m: Any) -> Matrix:
        return conv.to_canonical(_as_matrix(m))

    def _vocab(self, value: Any, allowed: set, kind: str) -> Optional[str]:
        if value is None:
            return None
        v = str(value)
        return v if v in allowed else self.pseudo.token(v, kind)

    def _instrument(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in inst.items():
            kl = str(k).lower()
            if kl in SERIAL_KEYS:
                out[k] = self.pseudo.token(str(v), "serial")
            elif kl in SAFE_INSTRUMENT_KEYS:
                out[k] = v
            else:
                out[k] = sanitize(v, self.pseudo, "instrument")
        return out

    def _inspect_fcs(self, path: str, read_spill: bool):
        if not regular_file(path):
            return self.pseudo.token(str(path), "sample"), None, None
        try:
            tok = keyed_file_token(path, self.pseudo, "sample")
        except OSError:
            return self.pseudo.token(str(path), "sample"), None, None
        if not read_spill:
            return tok, None, None
        try:
            kw = fcs.read_text_segment(path)
        except (SchemaError, OSError, ValueError):
            return tok, None, None
        try:
            m = fcs.spillover_from_keywords(kw)
        except (SchemaError, ValueError):
            m = None
        return tok, scrub_keywords(kw, self.pseudo), m

    def _control(self, c: Any) -> ControlStats:
        if isinstance(c, ControlStats):
            c = dict(c.__dict__)
        if not isinstance(c, dict):
            raise SchemaError("control must be an object")
        c = dict(c)
        path = c.pop("file_path", None)
        if path and not c.get("file_hash"):
            if regular_file(path):
                try:
                    c["file_hash"] = keyed_file_token(path, self.pseudo, "control")
                    kw = fcs.read_text_segment(path)
                    c.setdefault("acq_settings", {
                        k: v for k, v in scrub_keywords(kw, self.pseudo).items()
                        if k.startswith("$P") and k[-1] in "NVG" or k == "$CYT"})
                except (OSError, SchemaError, ValueError):
                    c.setdefault("file_hash", self.pseudo.token(str(path), "control"))
            else:
                c["file_hash"] = self.pseudo.token(str(path), "control")
        allowed = set(ControlStats.__dataclass_fields__)
        unknown = set(c) - allowed
        if unknown:
            raise SchemaError(f"unknown control fields {sorted(unknown)}")
        if c.get("control_type") is not None:
            c["control_type"] = self._vocab(c["control_type"], {"beads", "cells"}, "control_type")
        return ControlStats(**c)


def _as_matrix(m: Any) -> Matrix:
    if isinstance(m, Matrix):
        return m
    if isinstance(m, dict):
        return Matrix.from_dict(m)
    raise SchemaError(f"cannot interpret {type(m).__name__} as a matrix; "
                      "pass Matrix or {'rows', 'cols', 'values'}")
