"""Non-blocking, never-raising façade for use inside the host application.

``FlowIoBridge`` validates input and raises, and its handlers write to disk
synchronously (fsync per event) and may hash FCS files. Neither is
acceptable on a UI thread. ``SafeBridge`` has the same handler methods, but
each call only enqueues the event and returns immediately; a single worker
thread applies events to the real bridge in order, so nothing the collector
does can slow, block or crash Flow.Io.

    bridge = SafeBridge("./capture_data")
    bridge.on_workspace_opened(ws, panel=..., detectors=...)   # returns None, never raises
    ...
    bridge.close()        # at app exit; flushes the queue (also registered with atexit)

Timestamps are taken when the event is *enqueued*, so edit timing is not
distorted by queue latency. Errors are counted and logged through the
``logging`` module (logger ``flowio_capture``), never raised. If the queue
fills (default 100k events), further events are dropped and counted.
"""
from __future__ import annotations

import atexit
import functools
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from .bridge import FlowIoBridge

log = logging.getLogger("flowio_capture")
_STOP = object()


class SafeBridge:
    def __init__(self, root: Any, queue_size: int = 100_000, clock: Callable[[], float] = time.time,
                 **bridge_kwargs: Any):
        self._wall = clock
        self._now: Optional[float] = None
        self._bridge = FlowIoBridge(root, clock=self._clock, **bridge_kwargs)
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._dropped = 0
        self._errors = 0
        self._processed = 0
        self._last_error: Optional[str] = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="flowio-capture", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ---- handler façade ---------------------------------------------------------
    def __getattr__(self, name: str) -> Callable[..., None]:
        if name in FlowIoBridge.HANDLERS or name == "handle":
            return functools.partial(self._enqueue, name)
        raise AttributeError(name)

    def _enqueue(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait((self._wall(), name, args, kwargs))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                if self._dropped == 1:
                    log.warning("flowio-capture queue full; dropping events")

    # ---- worker -------------------------------------------------------------------------
    def _clock(self) -> float:
        return self._now if self._now is not None else self._wall()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                ts, name, args, kwargs = item
                self._now = ts
                try:
                    getattr(self._bridge, name)(*args, **kwargs)
                    with self._lock:
                        self._processed += 1
                except Exception as ex:  # never propagate into the host
                    with self._lock:
                        self._errors += 1
                        self._last_error = f"{name}: {ex}"
                    log.warning("flowio-capture: %s failed: %s", name, ex)
                finally:
                    self._now = None
            finally:
                self._q.task_done()

    # ---- control ------------------------------------------------------------------------
    def flush(self, timeout: Optional[float] = 10.0) -> bool:
        """Wait until everything enqueued so far has been applied."""
        if self._closed or not self._thread.is_alive():
            return True
        marker = threading.Event()
        try:
            self._q.put(marker, timeout=timeout)
        except queue.Full:
            return False
        return marker.wait(timeout)

    def close(self, reason: str = "shutdown", timeout: float = 10.0) -> None:
        """Flush, end all sessions, stop the worker. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put((self._wall(), "close_all", (reason,), {}), timeout=timeout)
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            log.warning("flowio-capture: queue full at close; some events lost")
        self._thread.join(timeout)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"queued": self._q.qsize(), "processed": self._processed, "errors": self._errors,
                    "dropped": self._dropped, "last_error": self._last_error}

    @property
    def bridge(self) -> FlowIoBridge:
        """The underlying synchronous bridge (for tests and diagnostics)."""
        return self._bridge
