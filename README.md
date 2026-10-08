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
| `export` | **acceptance signal**: the matrix in use when results left the app |
| `session_ended` | final state |

Labels are **not** decided at capture time. `flowio-capture build-dataset` derives them:

| file | use |
|---|---|
| `accepted.jsonl` | inputs + baseline matrix + accepted matrix + per-cell delta (primary target: `accepted − baseline`) |
| `cell_directions.jsonl` | per edit: was the value before it too low / too high relative to the accepted value |
| `preference_pairs.jsonl` | accepted ≻ intermediate state; `hard` = near-miss; `revised_export` = exported, then changed (strongest negatives) |
| `cell_hard_negatives.jsonl` | values the scientist explicitly rejected: `undone`, `reverted` (A→B→A), `overshoot` |
| `tolerances.jsonl` | per cell, the tightest bracket around the accepted value that the scientist moved away from |

Sessions with no export are recorded but left unlabeled.

## Wiring it into Flow.Io

Call `FlowIoBridge` handlers from your plugin's event hooks (see `src/flowio_capture/bridge.py` for the mapping table):

```python
from flowio_capture import FlowIoBridge
bridge = FlowIoBridge("./capture_data", app_version=flowio.version)

# in your plugin's event handlers:
bridge.on_workspace_opened(ws_path, sample_paths=fcs_paths, panel=fluors, detectors=dets,  # panel[i] -> detectors[i]
                           instrument=..., controls=..., user=...,
                           matrix_kind="spillover", matrix_unit="fraction")  # or "compensation" / "percent"
bridge.on_compensation_loaded(ws_path, matrix_dict, source="auto")
bridge.on_view_changed(ws_path, "FITC-A", "PE-A", population="Lymphocytes")
bridge.on_cell_changed(ws_path, "FITC", "PE-A", 0.16, old_value=0.12)   # or on_matrix_changed(ws_path, matrix_dict)
bridge.on_controls_updated(ws_path, controls)                            # after re-gating a control
bridge.on_undo(ws_path, matrix_after=matrix_dict)                        # send matrix_after whenever possible
bridge.on_export(ws_path, kind="report", target_path=out_path, matrix=matrix_dict)
bridge.on_workspace_closed(ws_path)
```

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
