# Installing the collector

The collector has two halves:

1. **The wrapper** (this repo): a Python package that records sessions, stores them, and builds datasets.
   Installs on any Mac with Python 3.9+. Fully built and tested.
2. **The Flow.Io plugin**: the small piece of code inside Flow.Io that calls the wrapper when the scientist
   opens a workspace, edits a cell, exports, etc. **Not built yet**, because it depends on Flow.Io's plugin
   SDK, which is not publicly documented. See "What's needed from Flow.Io" below.

## 1. Install the wrapper

```bash
# Python 3.9+ required (macOS: `python3 --version`)
python3 -m pip install "git+https://github.com/sshaar/flowIo-wrapper.git"

# verify
flowio-capture --help
python3 -c "import flowio_capture; print(flowio_capture.__version__)"
```

For development instead:

```bash
git clone https://github.com/sshaar/flowIo-wrapper.git && cd flowIo-wrapper
uv venv .venv --python 3.9 && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
```

## 2. One-time setup per machine

**Pseudonymization key.** All identifiers (workspace paths, sample files, user names, serials, export paths)
are replaced with keyed hashes. The key is created automatically on first use at
`~/.config/flowio_capture/salt` (mode 0600). If several machines in the lab should produce *matching* tokens
(same scientist/sample → same token), copy that file to each machine, or set the same value in
`FLOWIO_CAPTURE_SALT` for the collector process. Never copy the key into the data directory or ship it with
a dataset.

**Data directory.** Pick one, e.g. `~/flowio_capture_data`. Sessions land in `sessions/*.jsonl`.

## 3. Run the collector

How you run it depends on the plugin shape (section 4). Both shapes work on the Mac desktop app.

### Shape A: Python plugin (in-process)

If Flow.Io plugins can run Python, there is no separate process. The plugin imports the package:

```python
from flowio_capture import SafeBridge
bridge = SafeBridge("~/flowio_capture_data", app_version=flowio.version)
# ... call bridge.on_* from the plugin's event hooks (see README) ...
bridge.close()   # at app exit; also runs automatically via atexit
```

`SafeBridge` never raises and never blocks: every call returns immediately and is applied on a background
thread.

### Shape B: plugin in another language (Java, JavaScript, Swift, …)

Run the collector as a background service and have the plugin POST JSON to it.

```bash
flowio-capture --root ~/flowio_capture_data serve --port 8765
# prints the token location: ~/flowio_capture_data/.server_token
```

The plugin sends events as `POST http://127.0.0.1:8765/event` with headers
`Content-Type: application/json` and `X-Capture-Token: <contents of .server_token>`. Bodies are a single event
or a list; see README for the event format. Include an `event_id` per event so retries are idempotent.

To start it at login on macOS, save this as `~/Library/LaunchAgents/edu.cornell.flowio-capture.plist`
(adjust the paths) and run `launchctl load ~/Library/LaunchAgents/edu.cornell.flowio-capture.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>edu.cornell.flowio-capture</string>
  <key>ProgramArguments</key><array>
    <string>/usr/local/bin/flowio-capture</string>
    <string>--root</string><string>/Users/YOU/flowio_capture_data</string>
    <string>serve</string><string>--port</string><string>8765</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/Users/YOU/flowio_capture_data/collector.log</string>
</dict></plist>
```

(`which flowio-capture` gives the right path for `ProgramArguments`.)

## 4. The plugin itself: what's needed from Flow.Io

To write the plugin I need to know, for the Flow.Io desktop app:

| Need | Why |
|---|---|
| Plugin language and how plugins are loaded | decides Shape A vs B |
| An event or callback when a **compensation matrix is computed/loaded**, with its values and provenance (auto vs imported) | the baseline every label is measured from |
| An event when a **matrix cell changes** (or at minimum a way to read the current matrix on a timer) | the edit trajectory, which is most of the signal |
| An event when a **bivariate plot / NxN cell is shown**, with the channel names and, ideally, access to the compensated event data for that plot | tells us which cells were examined, and feeds the plot histograms |
| An event on **export / report / batch run** | the acceptance signal |
| Access to the **sample FCS paths, panel, instrument settings, and single-stain control statistics** | the model's inputs |
| Whether the app reports **spillover or compensation** matrices, **fractions or percent**, and labels rows by **fluorochrome or detector** | set once in `on_workspace_opened` so matrices are stored canonically |

If the SDK cannot provide cell-level edit events, a fallback is to poll the current matrix every ~200 ms while
the compensation editor is open and call `on_matrix_changed`; the wrapper still produces per-cell edits by
diffing. If there is no SDK at all, the remaining option is diffing saved workspace files, which loses the
trajectory and most of the value.

Point me at the SDK documentation (or a machine with the app installed) and I will write and test the
plugin against the real UI.

## 5. Building a dataset

```bash
flowio-capture --root ~/flowio_capture_data sessions                 # list what was recorded
flowio-capture --root ~/flowio_capture_data inspect <session_id>     # one session's summary
flowio-capture --root ~/flowio_capture_data build-dataset --out ./dataset
cat ./dataset/summary.json
```

`summary.json` reports how many sessions were labeled, why others were skipped (no export, desynced), how many
had an unclean baseline, and the counts of edited / examined / unexamined cells.

## 6. Before collecting real data

- Lab / IRB sign-off for collecting workflow data; the dataset contains no raw events or identifiers, but the
  plot histograms (if enabled) are derived from sample data.
- Keep `capture_plots=False` until that review clears it.
- Confirm the salt file lives outside the data directory (the wrapper refuses otherwise).
- Run the example once to see what a session file looks like:
  `python3 -m flowio_capture.cli --root /tmp/demo sessions` after `python3 examples/simulated_plugin.py /tmp/demo`.
