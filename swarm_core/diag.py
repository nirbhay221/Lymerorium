"""Diagnostic logging for swarm runs - persistent, timestamped, shareable.

Writes a structured timeline to a file so a post-mortem of "what failed and WHEN"
is possible even after a node (e.g. the Jetson) dies mid-run and its own logs are
unreachable. Captures, per LLM call: the lane, the model actually contacted, latency,
and success / failure / why-it-fell-back; and per swarm: START / END summaries with a
per-lane "who actually served" tally.

Log file: $SWARM_DIAG_LOG (default ./swarm_diag.log, append mode).
This is separate from the existing human-readable print() output on stdout.
"""
import logging
import os
import threading
from collections import Counter

_LOG_PATH = os.getenv("SWARM_DIAG_LOG", "swarm_diag.log")

_lock = threading.Lock()
_CALL_LEDGER: Counter = Counter()  # e.g. "vision/served:qwen3:8b" -> 4


def _build_logger() -> logging.Logger:
    lg = logging.getLogger("swarm.diag")
    if lg.handlers:  # already configured (module imported once, but be safe)
        return lg
    lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except Exception:
        pass  # logging must never break the call path
    lg.propagate = False
    return lg


log = _build_logger()


def event(key: str, n: int = 1) -> None:
    """Record a per-lane call event (thread-safe; called from concurrent groups)."""
    with _lock:
        _CALL_LEDGER[key] += n


def reset_ledger() -> None:
    with _lock:
        _CALL_LEDGER.clear()


def read_ledger() -> dict:
    with _lock:
        return dict(_CALL_LEDGER)
