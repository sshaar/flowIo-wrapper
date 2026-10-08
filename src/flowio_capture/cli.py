"""flowio-capture CLI: serve | sessions | inspect | build-dataset"""
from __future__ import annotations

import argparse
import json
import sys

from .dataset import build_dataset, derive, matrix_ids, replay
from .store import Store


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flowio-capture")
    p.add_argument("--root", default="./capture_data", help="capture store directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the localhost event endpoint")
    s.add_argument("--port", type=int, default=8765)

    sub.add_parser("sessions", help="list recorded sessions")

    i = sub.add_parser("inspect", help="summarize one session")
    i.add_argument("session_id")

    b = sub.add_parser("build-dataset", help="derive training records from all sessions")
    b.add_argument("--out", default="./dataset")
    b.add_argument("--eps", type=float, default=1e-4)

    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        serve(a.root, port=a.port)
        return 0
    store = Store(a.root)
    if a.cmd == "sessions":
        for sid in store.session_ids():
            ev = store.read(sid)
            types = [e["type"] for e in ev]
            print(f"{sid}  events={len(ev)}  edits={types.count('cell_edit')}  "
                  f"exports={types.count('export')}  ended={'session_ended' in types}")
        return 0
    if a.cmd == "inspect":
        events = store.read(a.session_id)
        report = []
        for mid in matrix_ids(events):
            rp = replay(events, mid)
            recs, skip = derive(rp)
            report.append({
                "matrix_id": mid, "snapshots": [src for _, src, _, _ in rp.snapshots],
                "steps": len(rp.steps), "edits": len(rp.edits), "exports": len(rp.exports),
                "ended": rp.ended, "errors": rp.errors, "skipped": skip,
                "derived": {k: len(v) for k, v in recs.items()},
            })
        print(json.dumps({"session_id": a.session_id, "matrices": report}, indent=2))
        return 0
    if a.cmd == "build-dataset":
        print(json.dumps(build_dataset(store, a.out, eps=a.eps), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
