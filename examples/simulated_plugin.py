"""Simulates a Flow.Io plugin driving the bridge, then builds a dataset.

    python examples/simulated_plugin.py /tmp/capture_demo

Replace the simulated calls with your Flow.Io SDK event handlers. Inside the
app use SafeBridge (shown here): calls return immediately, never raise, and
are applied on a background thread so the UI is never affected.
"""
import random
import sys

from flowio_capture import Matrix, SafeBridge, Store
from flowio_capture.dataset import build_dataset

root = sys.argv[1] if len(sys.argv) > 1 else "./capture_demo"
bridge = SafeBridge(root, app_version="flowio-sim-0.1", capture_plots=True)
rng = random.Random(0)


def fake_plot_events(n=5000):
    """Stand-in for the compensated sample events the host would hand over."""
    return [rng.lognormvariate(5, 1.2) for _ in range(n)], [rng.lognormvariate(4, 1.0) for _ in range(n)]
ws = "/data/experiments/2026-10-07_tcell_panel.workspace"
rows, cols = ["FITC", "PE", "APC"], ["FITC-A", "PE-A", "APC-A"]

bridge.on_workspace_opened(
    ws, panel=rows, detectors=cols, user="scientist_a",
    instrument={"model": "LSRFortessa", "serial": "H0012", "voltages": {"FITC-A": 480, "PE-A": 520, "APC-A": 600}},
    controls=[{"fluorochrome": "FITC", "control_type": "beads",
               "pos_median": {"FITC-A": 52000, "PE-A": 6100, "APC-A": 40},
               "neg_median": {"FITC-A": 120, "PE-A": 95, "APC-A": 30}}],
)
auto = Matrix.from_lists(rows, cols, [[1, 0.115, 0.0], [0.021, 1, 0.012], [0, 0.003, 1]])
bridge.on_compensation_loaded(ws, auto, source="auto")

xs, ys = fake_plot_events()
bridge.on_view_changed(ws, "FITC-A", "PE-A", population="Lymphocytes")
bridge.on_plot_observed(ws, x="FITC-A", y="PE-A", xs=xs, ys=ys, bins=64, population="Lymphocytes")
bridge.on_cell_changed(ws, "FITC", "PE-A", 0.13)
bridge.on_cell_changed(ws, "FITC", "PE-A", 0.16)   # overshoot
bridge.on_cell_changed(ws, "FITC", "PE-A", 0.14)
bridge.on_view_changed(ws, "PE-A", "APC-A", population="Lymphocytes")
bridge.on_cell_changed(ws, "PE", "APC-A", 0.02)
bridge.on_undo(ws)
bridge.on_export(ws, kind="report", target_path="/data/reports/tcell.pdf")
bridge.on_workspace_closed(ws)
bridge.close()                       # flushes the queue; also runs at interpreter exit
print(bridge.stats)

print(build_dataset(Store(root), f"{root}/dataset"))
