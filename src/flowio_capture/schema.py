"""Event schema for compensation-editing sessions.

A session is an append-only, ordered log of events. Everything needed for
training is derived from this log later (see ``dataset.py``), so labeling
decisions are not baked in at capture time.

Matrix convention: ``values[i][j]`` is the spillover of fluorochrome
``rows[i]`` into detector ``cols[j]`` (the FCS $SPILLOVER convention, with a
diagonal of 1.0). Cells are addressed by label, not index, so matrices from
panels of different sizes can be pooled.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Matrix:
    rows: Tuple[str, ...]  # fluorochromes (sources)
    cols: Tuple[str, ...]  # detectors (targets)
    values: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        rows, cols = tuple(self.rows), tuple(self.cols)
        values = tuple(tuple(float(v) for v in r) for r in self.values)
        if len(values) != len(rows) or any(len(r) != len(cols) for r in values):
            raise SchemaError(
                f"matrix shape {len(values)}x{len(values[0]) if values else 0} "
                f"does not match labels {len(rows)}x{len(cols)}"
            )
        if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
            raise SchemaError("duplicate row/column labels")
        if any(not math.isfinite(v) for r in values for v in r):
            raise SchemaError("matrix contains NaN/inf")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_lists(cls, rows, cols, values) -> "Matrix":
        return cls(tuple(rows), tuple(cols), tuple(tuple(r) for r in values))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Matrix":
        if not isinstance(d, dict) or not {"rows", "cols", "values"} <= d.keys():
            raise SchemaError("matrix must be {'rows', 'cols', 'values'}")
        try:
            return cls.from_lists(d["rows"], d["cols"], d["values"])
        except (TypeError, ValueError) as ex:
            if isinstance(ex, SchemaError):
                raise
            raise SchemaError(f"bad matrix: {ex}")

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": list(self.rows), "cols": list(self.cols),
                "values": [list(r) for r in self.values]}

    def _idx(self, row: str, col: str) -> Tuple[int, int]:
        try:
            return self.rows.index(row), self.cols.index(col)
        except ValueError:
            raise SchemaError(f"unknown cell ({row!r}, {col!r}); rows={list(self.rows)} "
                              f"cols={list(self.cols)}")

    def get(self, row: str, col: str) -> float:
        i, j = self._idx(row, col)
        return self.values[i][j]

    def with_value(self, row: str, col: str, value: float) -> "Matrix":
        i, j = self._idx(row, col)
        vals = [list(r) for r in self.values]
        vals[i][j] = float(value)
        return Matrix.from_lists(self.rows, self.cols, vals)

    def cells(self):
        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                yield r, c, self.values[i][j]

    def same_layout(self, other: "Matrix") -> bool:
        return self.rows == other.rows and self.cols == other.cols

    def relabel_rows(self, mapping: Dict[str, str]) -> "Matrix":
        return Matrix.from_lists([mapping.get(r, r) for r in self.rows], self.cols, self.values)

    def diff(self, other: "Matrix", tol: float = 1e-9) -> List[Tuple[str, str, float, float]]:
        """Cells that differ, as (row, col, self_value, other_value).

        Requires identical labels; a layout change (channel added/removed)
        is not an edit and must be recorded as a new snapshot instead.
        """
        if not self.same_layout(other):
            raise SchemaError("cannot diff matrices with different layouts")
        out = []
        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                a, b = self.values[i][j], other.values[i][j]
                if abs(a - b) > tol:
                    out.append((r, c, a, b))
        return out


@dataclass
class ControlStats:
    """Summary of one single-stain control.

    Medians/robust SDs are per detector, for the positive and negative
    populations the scientist (or auto-gating) selected. These make the
    learning problem well-posed: the matrix is largely a function of them.
    """
    fluorochrome: str
    file_hash: Optional[str] = None
    pos_median: Dict[str, float] = field(default_factory=dict)
    neg_median: Dict[str, float] = field(default_factory=dict)
    pos_rsd: Dict[str, float] = field(default_factory=dict)
    neg_rsd: Dict[str, float] = field(default_factory=dict)
    n_pos: Optional[int] = None
    n_neg: Optional[int] = None
    control_type: Optional[str] = None  # "beads" | "cells" | ...
    acq_settings: Dict[str, Any] = field(default_factory=dict)  # scrubbed $PnN/$PnV/$PnG of the control file


@dataclass
class SessionContext:
    """The model's input side: everything the matrix depends on."""
    experiment_id: str  # pseudonymized
    sample_hashes: List[str] = field(default_factory=list)
    panel: List[str] = field(default_factory=list)  # fluorochromes
    detectors: List[str] = field(default_factory=list)
    instrument: Dict[str, Any] = field(default_factory=dict)  # model, serial hash, lasers, filters, voltages
    controls: List[ControlStats] = field(default_factory=list)
    user_id: Optional[str] = None  # pseudonymized
    app_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionContext":
        d = dict(d)
        d["controls"] = [c if isinstance(c, ControlStats) else ControlStats(**c)
                         for c in d.get("controls", [])]
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Event types. Kept as strings in the log for language-agnostic producers.
SESSION_STARTED = "session_started"
SNAPSHOT = "snapshot"            # full matrix; source says where it came from
CELL_EDIT = "cell_edit"          # one cell changed by the user
VIEW_CHANGED = "view_changed"    # which channel pair / population is on screen
UNDO = "undo"
REDO = "redo"
EXPORT = "export"                # implicit acceptance signal
CONTROLS_UPDATED = "controls_updated"  # control stats changed (re-gating)
PLOT = "plot"                    # downsampled histogram of the plot on screen
SESSION_ENDED = "session_ended"
NOTE = "note"                    # free-form annotation

EVENT_TYPES = {SESSION_STARTED, SNAPSHOT, CELL_EDIT, VIEW_CHANGED, UNDO, REDO,
               EXPORT, CONTROLS_UPDATED, PLOT, SESSION_ENDED, NOTE}

# Snapshot sources. "acquisition" = FCS $SPILLOVER; "auto" = computed from
# controls by the app; "manual" = state after user edits; "import" = loaded
# from another workspace/template (the scientist did not derive it here);
# "saved_workspace" = reopened, possibly already hand-edited in an earlier
# session; "external" = changed outside our view (layout change, desync);
# "unknown" = first matrix seen with no provenance from the host.
SNAPSHOT_SOURCES = {"acquisition", "auto", "manual", "import", "saved_workspace",
                    "external", "unknown"}
# Baseline sources that represent an un-edited starting point.
CLEAN_BASELINE_SOURCES = {"acquisition", "auto"}
