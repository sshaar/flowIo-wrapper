"""Minimal FCS TEXT-segment reader (no event data, no dependencies).

Used to capture the acquisition-time spillover matrix ($SPILLOVER / $SPILL /
SPILL) and channel names, which form the baseline the scientist edits from.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from .schema import Matrix, SchemaError

SPILL_KEYS = ("$SPILLOVER", "$SPILL", "SPILL")


def read_text_segment(path: os.PathLike) -> Dict[str, str]:
    with open(path, "rb") as fh:
        header = fh.read(58)
        if len(header) < 58 or not header.startswith(b"FCS"):
            raise SchemaError(f"{path}: not an FCS file")
        try:
            start = int(header[10:18].strip())
            end = int(header[18:26].strip())
        except ValueError:
            raise SchemaError(f"{path}: bad FCS header offsets")
        if start <= 0 or end < start:
            raise SchemaError(f"{path}: bad TEXT offsets {start}-{end}")
        fh.seek(start)
        raw = fh.read(end - start + 1)
    text = raw.decode("utf-8", errors="replace")
    return parse_text(text)


def parse_text(text: str) -> Dict[str, str]:
    """Parse a TEXT segment. A doubled delimiter is an escaped literal."""
    if not text:
        return {}
    delim = text[0]
    tokens = []
    buf = []
    i = 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == delim:
            if i + 1 < n and text[i + 1] == delim:
                buf.append(delim)
                i += 2
                continue
            tokens.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf and "".join(buf).strip():
        tokens.append("".join(buf))
    out: Dict[str, str] = {}
    for k, v in zip(tokens[0::2], tokens[1::2]):
        out[k.strip().upper()] = v.strip()
    return out


def parse_spillover(value: str) -> Matrix:
    """``"n,P1,...,Pn,v11,v12,...,vnn"`` -> Matrix (rows=cols=channel names)."""
    parts = [p.strip() for p in value.split(",")]
    try:
        n = int(parts[0])
    except (ValueError, IndexError):
        raise SchemaError("bad spillover: missing dimension")
    if len(parts) != 1 + n + n * n:
        raise SchemaError(f"bad spillover: expected {1 + n + n * n} fields, got {len(parts)}")
    names = parts[1:1 + n]
    nums = [float(x) for x in parts[1 + n:]]
    vals = [nums[i * n:(i + 1) * n] for i in range(n)]
    return Matrix.from_lists(names, names, vals)


def spillover_from_keywords(keywords: Dict[str, str]) -> Optional[Matrix]:
    for k in SPILL_KEYS:
        if keywords.get(k):
            return parse_spillover(keywords[k])
    return None


def channel_labels(keywords: Dict[str, str]) -> Dict[str, str]:
    """Map $PnN (detector) -> $PnS (stain/marker label) where present."""
    out = {}
    try:
        npar = int(keywords.get("$PAR", "0"))
    except ValueError:
        npar = 0
    for i in range(1, npar + 1):
        name = keywords.get(f"$P{i}N")
        if name:
            out[name] = keywords.get(f"$P{i}S", "")
    return out
