"""Downsampled 2D histograms of what the scientist was looking at.

A compensation edit is a judgment about a plot: the fully stained sample (or
a control) shown on the two channels whose spillover is being adjusted,
compensated with the *current* matrix. The matrix alone does not tell a model
why the value moved; the plot does. ``PlotSummary`` is a small, aggregate
representation of that plot (default 64x64 bins, a few KB), logged by the
host whenever the view refreshes and attached by seq to the edits made while
it was on screen.

Histograms are aggregates, but they are still derived from sample data, so
plot capture is off by default in the bridge and subject to a per-session
byte budget.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import SchemaError

MAX_BINS = 128
PLOT_KINDS = {"sample", "control"}
SCALES = {"linear", "log", "arcsinh", "biexp", "unknown"}
ARCSINH_COFACTOR = 150.0


def transform(v: float, scale: str, cofactor: float = ARCSINH_COFACTOR) -> float:
    if scale == "log":
        return math.log10(v) if v > 1.0 else 0.0
    if scale in ("arcsinh", "biexp"):
        return math.asinh(v / cofactor)
    return v


def histogram2d(xs: Sequence[float], ys: Sequence[float], bins: int = 64,
                x_range: Optional[Tuple[float, float]] = None,
                y_range: Optional[Tuple[float, float]] = None,
                x_scale: str = "arcsinh", y_scale: str = "arcsinh",
                max_events: Optional[int] = 200_000) -> Tuple[List[List[int]], Tuple[float, float], Tuple[float, float], int]:
    """Bin raw event coordinates. Returns (counts[bx][by], x_range, y_range, n).

    Ranges are in *transformed* units. If a range is None it is taken from
    the data. Values outside the range land in the edge bins. ``max_events``
    keeps the pure-Python loop bounded; events beyond it are stride-sampled.
    """
    if not 1 <= bins <= MAX_BINS:
        raise SchemaError(f"bins must be in [1, {MAX_BINS}]")
    n_total = min(len(xs), len(ys))
    if n_total == 0:
        raise SchemaError("empty event list")
    stride = 1 if not max_events or n_total <= max_events else math.ceil(n_total / max_events)
    tx = [transform(float(xs[i]), x_scale) for i in range(0, n_total, stride)]
    ty = [transform(float(ys[i]), y_scale) for i in range(0, n_total, stride)]
    if x_range is None:
        x_range = (min(tx), max(tx))
    if y_range is None:
        y_range = (min(ty), max(ty))
    counts = [[0] * bins for _ in range(bins)]
    xlo, xhi = x_range
    ylo, yhi = y_range
    xw = (xhi - xlo) or 1.0
    yw = (yhi - ylo) or 1.0
    last = bins - 1
    for a, b in zip(tx, ty):
        i = int((a - xlo) / xw * bins)
        j = int((b - ylo) / yw * bins)
        counts[min(max(i, 0), last)][min(max(j, 0), last)] += 1
    return counts, (float(xlo), float(xhi)), (float(ylo), float(yhi)), len(tx)


@dataclass
class PlotSummary:
    x: str                      # detector on the x axis
    y: str                      # detector on the y axis
    counts: List[List[int]]     # counts[x_bin][y_bin]
    kind: str = "sample"        # "sample" | "control"
    x_range: Tuple[float, float] = (0.0, 1.0)   # in transformed units
    y_range: Tuple[float, float] = (0.0, 1.0)
    x_scale: str = "unknown"
    y_scale: str = "unknown"
    population: Optional[str] = None
    control_fluorochrome: Optional[str] = None  # for kind="control"
    n_events: Optional[int] = None
    compensated: bool = True    # plotted with the matrix current at the time
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in PLOT_KINDS:
            raise SchemaError(f"plot kind must be one of {sorted(PLOT_KINDS)}")
        for s in (self.x_scale, self.y_scale):
            if s not in SCALES:
                raise SchemaError(f"unknown scale {s!r}")
        if not self.counts or not all(isinstance(r, (list, tuple)) for r in self.counts):
            raise SchemaError("counts must be a non-empty list of rows")
        bx, by = len(self.counts), len(self.counts[0])
        if bx > MAX_BINS or by > MAX_BINS:
            raise SchemaError(f"plot exceeds {MAX_BINS} bins per axis")
        if any(len(r) != by for r in self.counts):
            raise SchemaError("counts must be rectangular")
        for r in self.counts:
            for v in r:
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise SchemaError("counts must be non-negative integers")
        self.counts = [list(r) for r in self.counts]
        self.x_range = (float(self.x_range[0]), float(self.x_range[1]))
        self.y_range = (float(self.y_range[0]), float(self.y_range[1]))
        if self.n_events is None:
            self.n_events = sum(sum(r) for r in self.counts)

    @classmethod
    def from_events(cls, x: str, y: str, xs: Sequence[float], ys: Sequence[float],
                    bins: int = 64, x_scale: str = "arcsinh", y_scale: str = "arcsinh",
                    **kw: Any) -> "PlotSummary":
        counts, xr, yr, n = histogram2d(xs, ys, bins, kw.pop("x_range", None), kw.pop("y_range", None),
                                        x_scale, y_scale)
        return cls(x=x, y=y, counts=counts, x_range=xr, y_range=yr, x_scale=x_scale, y_scale=y_scale,
                   n_events=n, **kw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlotSummary":
        if not isinstance(d, dict):
            raise SchemaError("plot must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise SchemaError(f"unknown plot fields {sorted(unknown)}")
        try:
            return cls(**d)
        except TypeError as ex:
            raise SchemaError(f"bad plot: {ex}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["x_range"] = list(self.x_range)
        d["y_range"] = list(self.y_range)
        return d

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.counts), len(self.counts[0])
