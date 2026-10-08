"""Capture compensation-matrix editing sessions from Flow.Io for model training."""
from .bridge import FlowIoBridge
from .normalize import MatrixConvention
from .plots import PlotSummary, histogram2d
from .recorder import CaptureSession
from .safe import SafeBridge
from .schema import ControlStats, Matrix, SessionContext
from .store import Store

__all__ = ["FlowIoBridge", "SafeBridge", "CaptureSession", "ControlStats", "Matrix",
           "MatrixConvention", "PlotSummary", "SessionContext", "Store", "histogram2d"]
__version__ = "0.2.0"
