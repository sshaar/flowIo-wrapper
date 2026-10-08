"""Pseudonymization and FCS metadata scrubbing.

FCS TEXT segments routinely contain sample names, patient/study IDs,
operator names and file paths. Nothing identifying is written to the log
in the clear: identifiers are replaced by keyed hashes (HMAC-SHA256) so the
same sample/user maps to the same token across sessions, but the token
cannot be reversed without the lab's secret salt.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

SALT_ENV = "FLOWIO_CAPTURE_SALT"
SALT_FILE_ENV = "FLOWIO_CAPTURE_SALT_FILE"
DEFAULT_SALT_FILE = Path.home() / ".config" / "flowio_capture" / "salt"

# Keywords that are safe to keep verbatim (instrument/channel description).
# Anything else is dropped; keywords in HASHED are kept as tokens.
SAFE_KEYWORDS = {"$CYT", "$MODE", "$DATATYPE", "$PAR", "$TOT", "$TIMESTEP",
                 "$SPILLOVER", "$SPILL", "SPILL"}
SAFE_PARAM_SUFFIXES = ("N", "S", "R", "E", "G", "V", "B")  # $PnN, $PnS, $PnR, ...
HASHED_KEYWORDS = {"$CYTSN", "$FIL", "$SRC", "$SMNO", "$OP", "$PROJ", "$EXP",
                   "$INST", "$CELLS", "$COM"}


class Pseudonymizer:
    def __init__(self, salt: bytes):
        if len(salt) < 16:
            raise ValueError("salt must be at least 16 bytes")
        self._salt = salt

    @classmethod
    def load(cls, store_root: Optional[os.PathLike] = None) -> "Pseudonymizer":
        """Salt from $FLOWIO_CAPTURE_SALT, else the file at
        $FLOWIO_CAPTURE_SALT_FILE or ~/.config/flowio_capture/salt (created
        once, mode 0600).

        The salt is deliberately kept outside the capture store so that
        copying the store (to share the dataset) does not copy the key.
        Share it across machines in the same lab if tokens should match.
        """
        env = os.environ.get(SALT_ENV)
        if env:
            return cls(env.encode("utf-8"))
        path = Path(os.environ.get(SALT_FILE_ENV) or DEFAULT_SALT_FILE).expanduser().resolve()
        if store_root is not None:
            root = Path(store_root).resolve()
            if root == path.parent or root in path.parents:
                raise ValueError(f"salt file {path} must not live inside the capture store {root}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_hex(32))
        return cls(path.read_text().strip().encode("utf-8"))

    def token(self, value: Optional[str], kind: str = "id") -> Optional[str]:
        if value is None or value == "":
            return None
        mac = hmac.new(self._salt, f"{kind}:{value}".encode("utf-8"), hashlib.sha256)
        return f"{kind}_{mac.hexdigest()[:20]}"


def regular_file(path: Any) -> bool:
    try:
        return Path(path).is_file()
    except (OSError, TypeError, ValueError):
        return False


def keyed_file_token(path: os.PathLike, pseudo: "Pseudonymizer", kind: str) -> str:
    """Keyed token of a file's *content*. A plain SHA-256 would let anyone
    holding a candidate file confirm it is in the dataset; the HMAC does not."""
    return pseudo.token(file_sha256(path), kind)


def sanitize(obj: Any, pseudo: "Pseudonymizer", kind: str = "text") -> Any:
    """Keep numbers/bools/None and structure; replace every string with a token.

    For free-form host fields whose contents we cannot vouch for."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, str):
        return pseudo.token(obj, kind)
    if isinstance(obj, dict):
        return {str(k): sanitize(v, pseudo, kind) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, pseudo, kind) for v in obj]
    return pseudo.token(repr(obj), kind)


def file_sha256(path: os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_safe_param_keyword(key: str) -> bool:
    # $P<n><suffix>, e.g. $P3N, $P12S
    if not key.startswith("$P") or len(key) < 4:
        return False
    body, suffix = key[2:-1], key[-1]
    return body.isdigit() and suffix in SAFE_PARAM_SUFFIXES


def scrub_keywords(keywords: Dict[str, str], pseudo: Pseudonymizer) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in keywords.items():
        ku = k.upper()
        if ku in SAFE_KEYWORDS or _is_safe_param_keyword(ku):
            out[ku] = v
        elif ku in HASHED_KEYWORDS:
            tok = pseudo.token(v, kind=ku.strip("$").lower())
            if tok:
                out[ku] = tok
        # everything else (dates, free-text, vendor keywords) is dropped
    return out
