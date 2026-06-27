"""
Proactive host resource guard - stop work *before* the local LLM OOM-crashes.

The circuit breaker in agents.py is REACTIVE: it trips only after an endpoint has
already failed. This guard is PREVENTIVE: it samples local GPU VRAM + system RAM
and refuses to dispatch a heavy local LLM call (or aborts the debate) when the
machine is about to run out - trading a graceful skip/abort for an OS-level OOM kill
or an Ollama/llama.cpp "CUDA out of memory" crash.

Scope: only the LOCAL host is measurable from here. Remote endpoints (Jetson, cloud)
are covered by the circuit breaker + timeouts, so the guard only gates calls whose
base_url is localhost/127.0.0.1.

Two levels:
  - SOFT: divert local calls to the fallback endpoint (offload the strained GPU)
  - CRITICAL: raise ResourceExhausted; nodes assert at entry, api.run_simulation
              catches it and returns a clean "aborted" verdict.

All thresholds are env-tunable (MB). Defaults suit an 8GB-VRAM / 32GB-RAM laptop;
lower them on the Jetson via env. Degrades to a no-op if neither GPU nor RAM can be
read (no nvidia-smi, no psutil, unsupported OS).

Background on the failure mode this prevents:
  - Ollama/llama.cpp slow to a crawl or crash when a model spills past VRAM into RAM.
  - OLLAMA_GPU_OVERHEAD reserves headroom; this guard is the application-side equivalent.
"""

import os
import time
import shutil
import threading
import subprocess

# Thresholds in MB. Worst-of(GPU, RAM) decides the level.
# NOTE: a resident model makes free VRAM naturally low (~600 MB with an 8B in 8 GB) -
# that's the healthy steady state, not pressure. So GPU thresholds are deliberately low:
# only a genuine near-OOM should trip them. RAM is the real OOM risk (CPU model + Flask
# + embeddings sharing system memory), so the RAM thresholds stay meaningful.
GPU_SOFT_FREE_MB = int(os.getenv("GUARD_GPU_SOFT_MB", "200"))
GPU_CRIT_FREE_MB = int(os.getenv("GUARD_GPU_CRIT_MB", "100"))
RAM_SOFT_FREE_MB = int(os.getenv("GUARD_RAM_SOFT_MB", "1500"))
RAM_CRIT_FREE_MB = int(os.getenv("GUARD_RAM_CRIT_MB", "600"))

# Samples are cached briefly so we don't spawn nvidia-smi on every LLM call.
_SAMPLE_TTL = float(os.getenv("GUARD_SAMPLE_TTL", "2.0"))

_lock = threading.Lock()
_cache = {"t": 0.0, "gpu_free": None, "ram_free": None}


class ResourceExhausted(RuntimeError):
    """Raised when host memory is critically low - abort rather than OOM-crash."""


# Samplers (best-effort, never raise)

def _gpu_free_mb():
    """Minimum free VRAM across local NVIDIA GPUs, via nvidia-smi. None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode != 0:
            return None
        vals = [int(x) for x in out.stdout.split("\n") if x.strip().isdigit()]
        return min(vals) if vals else None
    except Exception:
        return None


def _ram_free_mb():
    """Available system RAM in MB. Tries psutil, then /proc/meminfo (Linux), then ctypes (Windows)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        pass
    try:  # Linux / Jetson
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    try:  # Windows (no psutil)
        import ctypes

        class _MEMSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys / (1024 * 1024))
    except Exception:
        return None


def snapshot():
    """(gpu_free_mb, ram_free_mb), cached for _SAMPLE_TTL seconds. Either may be None."""
    now = time.time()
    with _lock:
        if now - _cache["t"] < _SAMPLE_TTL:
            return _cache["gpu_free"], _cache["ram_free"]
    gpu, ram = _gpu_free_mb(), _ram_free_mb()
    with _lock:
        _cache.update(t=now, gpu_free=gpu, ram_free=ram)
    return gpu, ram


# Decisions

def level():
    """Return 'ok' | 'soft' | 'critical' from worst of GPU/RAM headroom."""
    gpu, ram = snapshot()
    if (gpu is not None and gpu < GPU_CRIT_FREE_MB) or (ram is not None and ram < RAM_CRIT_FREE_MB):
        return "critical"
    if (gpu is not None and gpu < GPU_SOFT_FREE_MB) or (ram is not None and ram < RAM_SOFT_FREE_MB):
        return "soft"
    return "ok"


def is_local(base_url: str) -> bool:
    base_url = base_url or ""
    return ("localhost" in base_url) or ("127.0.0.1" in base_url) or ("0.0.0.0" in base_url)


def should_divert(base_url: str) -> bool:
    """True if base_url is a local endpoint AND the host is under memory pressure."""
    return is_local(base_url) and level() != "ok"


def assert_safe(label: str = "") -> None:
    """Raise ResourceExhausted if the host is at the critical line. No-op otherwise."""
    if level() == "critical":
        raise ResourceExhausted(f"{snapshot_str()} below critical threshold ({label})")


# Reporting

def snapshot_str() -> str:
    gpu, ram = snapshot()
    g = f"{gpu}MB" if gpu is not None else "n/a"
    r = f"{ram}MB" if ram is not None else "n/a"
    return f"GPU_free={g} RAM_free={r}"


def status() -> dict:
    gpu, ram = snapshot()
    return {
        "level": level(),
        "gpu_free_mb": gpu,
        "ram_free_mb": ram,
        "thresholds": {
            "gpu_soft": GPU_SOFT_FREE_MB, "gpu_crit": GPU_CRIT_FREE_MB,
            "ram_soft": RAM_SOFT_FREE_MB, "ram_crit": RAM_CRIT_FREE_MB,
        },
    }
