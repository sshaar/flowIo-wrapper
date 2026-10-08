"""Bring host matrices into the canonical form the log stores.

Canonical form: a **spillover** matrix of **fractions**, rows labeled by
fluorochrome, columns by detector, with spill(f, primary_detector(f)) == 1.

Hosts differ: some report percentages (diagonal 100), some report the
compensation matrix (the inverse of spillover), and FCS $SPILLOVER labels
both axes with detector names. Mixing these silently corrupts targets, so
everything goes through ``MatrixConvention.to_canonical`` and a matrix that
fails validation is rejected rather than logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import Matrix, SchemaError

DIAG_TOL = 1e-6
MIN_OFFDIAG = -0.05  # small negatives happen with over-compensated controls


def invert(values: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inverse with partial pivoting (small matrices only)."""
    n = len(values)
    if any(len(r) != n for r in values):
        raise SchemaError("only square matrices can be inverted")
    a = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(values)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise SchemaError("matrix is singular")
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [v / p for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0.0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


@dataclass
class MatrixConvention:
    """How a particular host reports matrices.

    kind: "spillover" or "compensation" (inverse of spillover).
    unit: "fraction" or "percent".
    primary: fluorochrome -> its primary detector. Used to relabel
        detector-labeled rows (FCS $SPILLOVER) and to validate the diagonal.
    """
    kind: str = "spillover"
    unit: str = "fraction"
    primary: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("spillover", "compensation"):
            raise SchemaError(f"kind must be spillover|compensation, got {self.kind!r}")
        if self.unit not in ("fraction", "percent"):
            raise SchemaError(f"unit must be fraction|percent, got {self.unit!r}")

    @classmethod
    def from_panel(cls, panel: List[str], detectors: List[str], **kw) -> "MatrixConvention":
        """panel[i] is measured primarily on detectors[i] (when lengths match)."""
        primary = dict(zip(panel, detectors)) if panel and len(panel) == len(detectors) else {}
        return cls(primary=primary, **kw)

    @property
    def detector_to_fluor(self) -> Dict[str, str]:
        return {d: f for f, d in self.primary.items()}

    def relabel_detector_rows(self, m: Matrix) -> Matrix:
        """FCS-style matrix (detector names on both axes) -> fluorochrome rows."""
        inv = self.detector_to_fluor
        if not inv or not all(r in inv for r in m.rows):
            return m
        return m.relabel_rows(inv)

    def to_canonical(self, m: Matrix) -> Matrix:
        vals = [list(r) for r in m.values]
        if self.unit == "percent":
            vals = [[v / 100.0 for v in r] for r in vals]
        if self.kind == "compensation":
            vals = invert(vals)
            # An edited compensation matrix inverts to a spillover whose
            # diagonal drifts from 1; rescale each row to its primary detector.
            tmp = self.relabel_detector_rows(Matrix.from_lists(m.rows, m.cols, vals))
            for r, c in diagonal_cells(tmp, self.primary):
                d = vals[tmp.rows.index(r)][tmp.cols.index(c)]
                if abs(d) > 1e-12:
                    vals[tmp.rows.index(r)] = [v / d for v in vals[tmp.rows.index(r)]]
        out = self.relabel_detector_rows(Matrix.from_lists(m.rows, m.cols, vals))
        validate_spillover(out, self.primary)
        return out

    def from_canonical(self, m: Matrix) -> Matrix:
        vals = [list(r) for r in m.values]
        if self.kind == "compensation":
            vals = invert(vals)
        if self.unit == "percent":
            vals = [[v * 100.0 for v in r] for r in vals]
        return Matrix.from_lists(m.rows, m.cols, vals)


def diagonal_cells(m: Matrix, primary: Optional[Dict[str, str]] = None):
    """(row, col) pairs that should equal 1.0."""
    primary = primary or {}
    out = []
    for i, r in enumerate(m.rows):
        if r in primary and primary[r] in m.cols:
            out.append((r, primary[r]))
        elif r in m.cols:  # detector-labeled rows
            out.append((r, r))
        elif len(m.rows) == len(m.cols):  # same order, square
            out.append((r, m.cols[i]))
    return out


def validate_spillover(m: Matrix, primary: Optional[Dict[str, str]] = None) -> None:
    diag = diagonal_cells(m, primary)
    for r, c in diag:
        v = m.get(r, c)
        if abs(v - 1.0) > DIAG_TOL:
            hint = (" (looks like percent units)" if abs(v - 100.0) < 1e-3 else
                    " (compensation matrix or unnormalized?)")
            raise SchemaError(f"spillover diagonal ({r}, {c}) = {v}, expected 1.0{hint}")
    dset = set(diag)
    for r, c, v in m.cells():
        if (r, c) not in dset and v < MIN_OFFDIAG:
            raise SchemaError(f"spillover ({r}, {c}) = {v} is strongly negative; "
                              "is this a compensation (inverse) matrix?")
