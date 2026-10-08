# flowio-capture

Records how scientists edit compensation matrices in Flow.Io, and turns those
sessions into training data for a model that predicts the matrix (or the
correction to the auto-computed matrix).

## What gets captured

Each workspace session is an append-only JSONL log (`capture_data/sessions/<id>.jsonl`):

| event | meaning |
|---|---|
| `session_started` | model inputs: panel, detectors, instrument config, single-stain control stats, pseudonymized sample/user IDs |
| `snapshot` | whole matrix from `acquisition` (FCS `$SPILLOVER`), `auto` (computed from controls), `import`, or a layout change |
| `cell_edit` | one cell changed: old → new, the plot the scientist was viewing, time since the last edit |
| `view_changed` | which channel pair / population is on screen |
| `undo` / `redo` | followed by the cell edits they caused |
| `plot` | downsampled 2D histogram of the plot on screen (off by default; see below) |
| `export` | **acceptance signal**: the matrix in use when results left the app |
| `session_ended` | final state |

Labels are **not** decided at capture time. `flowio-capture build-dataset` derives them:

| file | use |
|---|---|
| `accepted.jsonl` | inputs + baseline matrix + accepted matrix + per-cell records (primary target: `accepted − baseline`). Each cell has a status: `edited`, `examined_accepted` (the pair was on screen at its accepted value), or `unexamined` (never looked at, so delta 0 is not a judgment) |
| `cell_directions.jsonl` | per edit: was the value before it too low / too high relative to the accepted value |
| `preference_pairs.jsonl` | accepted ≻ intermediate state; `hard` = near-miss; `revised_export` = exported, then changed (strongest negatives) |
| `cell_hard_negatives.jsonl` | values the scientist explicitly rejected: `undone`, `reverted` (A→B→A), `overshoot` |
| `tolerances.jsonl` | per cell, the tightest bracket around the accepted value that the scientist moved away from |
| `plots.jsonl` | plot histograms logged before acceptance, referenced from `cell_directions.plot_refs` |

Sessions with no export are recorded but left unlabeled. Rapid runs of edits to one cell (slider drags, ≤250 ms
apart) are collapsed into a single judgment before labels are derived; mid-drag states are not negatives.

## Wiring it into Flow.Io

The collector is a pure add-on: it observes, never changes the UI or the matrix, and must never slow or break the
host. Inside the app use `SafeBridge`: every call enqueues and returns immediately, nothing raises, and a single
background thread applies events in order (timestamps are taken at call time, so edit timing is preserved).
`FlowIoBridge` is the same API, synchronous and raising, for tests and the HTTP server.

Call the handlers from your plugin's event hooks (see `src/flowio_capture/bridge.py` for the mapping table):

```python
from flowio_capture import SafeBridge
bridge = SafeBridge("./capture_data", app_version=flowio.version)   # add capture_plots=True after privacy review

# in your plugin's event handlers:
bridge.on_workspace_opened(ws_path, sample_paths=fcs_paths, panel=fluors, detectors=dets,  # panel[i] -> detectors[i]
                           instrument=..., controls=..., user=...,
                           matrix_kind="spillover", matrix_unit="fraction")  # or "compensation" / "percent"
bridge.on_compensation_loaded(ws_path, matrix_dict, source="auto")
bridge.on_view_changed(ws_path, "FITC-A", "PE-A", population="Lymphocytes")
bridge.on_cell_changed(ws_path, "FITC", "PE-A", 0.16, old_value=0.12)   # or on_matrix_changed(ws_path, matrix_dict)
bridge.on_plot_observed(ws_path, x="FITC-A", y="PE-A", xs=events_x, ys=events_y, bins=64)  # what the scientist sees
bridge.on_controls_updated(ws_path, controls)                            # after re-gating a control
bridge.on_undo(ws_path, matrix_after=matrix_dict)                        # send matrix_after whenever possible
bridge.on_export(ws_path, kind="report", target_path=out_path, matrix=matrix_dict)
bridge.on_workspace_closed(ws_path)
bridge.close()            # at app exit (also registered with atexit); bridge.stats shows errors/drops
```

### Plots

A compensation edit is a judgment about a plot, and the matrix alone does not say why a value moved. With
`capture_plots=True`, `on_plot_observed` logs a small 2D histogram (default 64×64 bins, arcsinh-scaled, a few KB)
of the sample or control on the two channels being viewed, tagged with the matrix it was compensated with.
Pass raw event coordinates and the wrapper bins them, or pass a ready `PlotSummary` dict. Each later edit on that
channel pair carries `plot_refs` to the sample and control plots on screen. Histograms are aggregate, but they are
derived from sample data, so capture is off by default and bounded by a per-session byte budget (64 MB).

`on_compensation_loaded` requires a `source` (`auto`, `acquisition`, `import`, `saved_workspace`, `manual`,
`unknown`), which decides whether the baseline is "clean". Every matrix handler takes `matrix_id` for workspaces
with several compensation matrices; each is labeled separately.

If the plugin isn't written in Python, run `flowio-capture serve` and POST the same events as JSON to
`http://127.0.0.1:8765/event` with `Content-Type: application/json` and `X-Capture-Token` (from
`$FLOWIO_CAPTURE_TOKEN`, or generated into `<root>/.server_token`). Add an `event_id` to make retries idempotent;
batch responses report `applied` so a client resends only what failed.

```json
{"event": "cell_changed", "workspace_ref": "ws-1", "row": "FITC", "col": "PE-A", "new_value": 0.16, "event_id": "42"}
```

Matrices are `{"rows": [...], "cols": [...detectors], "values": [[...]]}` in the host's convention (declared at
`on_workspace_opened`). They are stored canonically as **spillover fractions** with fluorochrome rows; FCS-style
detector-labeled rows are relabeled through the panel→detector pairing, and matrices whose diagonal isn't 1
(percent units or an un-declared compensation matrix) are rejected rather than silently logged.

If the recorder can't reproduce what the host did (e.g. an undo without `matrix_after` that doesn't match its own
history), it marks that matrix *desynced*. Edits and exports are excluded from labels until the host sends a full matrix.

## Privacy

Workspace paths, sample files, user names, instrument serials and export paths are replaced with
HMAC tokens keyed by a per-lab salt (`$FLOWIO_CAPTURE_SALT`, or `~/.config/flowio_capture/salt`; it is refused
inside the capture store so sharing the store never shares the key). File-content hashes are keyed too. FCS
keywords pass through an allowlist, and other free-form strings (`extra`, unknown instrument keys, view extras,
unrecognized export kinds/labels) are tokenized. Get IRB / lab sign-off before collecting.

## Dev

```bash
uv venv .venv --python 3.9 && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/simulated_plugin.py /tmp/capture_demo
```
