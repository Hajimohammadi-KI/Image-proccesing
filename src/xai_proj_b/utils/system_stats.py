from __future__ import annotations

import csv
import multiprocessing as mp
import os
import subprocess
import threading
import time
from typing import Dict, Optional

# cspell:ignore typeperf

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - torch is required by the project but keep import guarded
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


_CPU_KEYS = ["cpu_percent", "ram_percent", "gpu_percent", "gpu_memory_used", "gpu_memory_total"]


class _SystemStatsMonitor:
    """Background monitor that samples system usage at a fixed cadence."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = max(0.25, interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: Dict[str, Optional[float]] = {key: None for key in _CPU_KEYS}
        self._timestamp = 0.0

        # Prime psutil so first reading is meaningful rather than 0.0.
        if psutil is not None:  # pragma: no branch
            try:
                psutil.cpu_percent(interval=0.0)
            except Exception:
                pass

        # Take an initial quick sample before starting the main loop.
        initial = self._collect_stats(initial=True)
        with self._lock:
            self._latest.update(initial)
            self._timestamp = time.time()

        self._thread = threading.Thread(target=self._run, name="system-stats-monitor", daemon=True)
        self._thread.start()

    def snapshot(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return dict(self._latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            stats = self._collect_stats(initial=False)
            with self._lock:
                self._latest.update(stats)
                self._timestamp = time.time()
            # No extra sleep needed when psutil is sampling with an interval;
            # baton back to the loop if the stop event is triggered.
            if psutil is None:
                self._stop.wait(self._interval)

    def _collect_stats(self, *, initial: bool) -> Dict[str, Optional[float]]:
        cpu_percent = self._sample_cpu(initial=initial)
        ram_percent = self._sample_ram()
        gpu_stats = self._sample_gpu()
        stats: Dict[str, Optional[float]] = {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "gpu_percent": gpu_stats.get("gpu_percent"),
            "gpu_memory_used": gpu_stats.get("gpu_memory_used"),
            "gpu_memory_total": gpu_stats.get("gpu_memory_total"),
        }
        return stats

    def _sample_cpu(self, *, initial: bool) -> Optional[float]:
        if psutil is not None:
            try:
                sample_interval = 0.25 if initial else self._interval
                value = psutil.cpu_percent(interval=sample_interval)
                # psutil can occasionally return 0.0 immediately after the warm-up; fall back on Windows counters.
                if value <= 0 and os.name == "nt":
                    fallback = _read_windows_cpu_counter()
                    return fallback if fallback is not None else value
                return value
            except Exception:
                pass
        return _read_windows_cpu_counter()

    def _sample_ram(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return None

    def _sample_gpu(self) -> Dict[str, Optional[float]]:
        if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():  # pragma: no cover
            return {}
        stats: Dict[str, Optional[float]] = {}

        # Windows exposes the same counters that Task Manager graphs via perf counters.
        # Query them first so we can mirror its behavior before falling back to nvidia-smi.
        if os.name == "nt":
            percent = _read_windows_gpu_counter()
            if percent is not None:
                stats["gpu_percent"] = percent

        # Memory accounting is cheap to obtain directly from CUDA.
        mem_stats = _read_cuda_memory()
        stats.update(mem_stats)

        # If we are still missing utilization, fall back to nvidia-smi output as before.
        if stats.get("gpu_percent") is None:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=True,
                )
                first_line = result.stdout.strip().splitlines()[0]
                util_str, mem_used_str, mem_total_str = [part.strip() for part in first_line.split(",")]
                stats.setdefault("gpu_percent", float(util_str))
                # Only fill memory keys if CUDA did not already provide them.
                stats.setdefault("gpu_memory_used", float(mem_used_str))
                stats.setdefault("gpu_memory_total", float(mem_total_str))
            except Exception:
                pass

        return stats


def _read_windows_cpu_counter() -> Optional[float]:
    if os.name != "nt":  # pragma: no cover - Windows only fallback
        return None
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "LoadPercentage", "/value"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "LoadPercentage" in line:
                _, value = line.split("=", 1)
                value_str = value.strip().replace(",", ".")
                if value_str:
                    return float(value_str)
    except Exception:
        return None
    return None


def _read_windows_gpu_counter() -> Optional[float]:
    if os.name != "nt":  # pragma: no cover - Windows only fallback
        return None
    try:
        # cspell: disable-next-line
        result = subprocess.run(
            [
                "typeperf",
                "\\GPU Engine(*)\\Utilization Percentage",
                "-sc",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
    except Exception:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        success_index = lines.index("The command completed successfully.")
        lines = lines[:success_index]
    except ValueError:
        pass

    csv_lines = [line for line in lines if line.startswith("(") or line.startswith("\"")]
    if len(csv_lines) < 2:
        return None

    try:
        header = next(csv.reader([csv_lines[0]]))
        values = next(csv.reader([csv_lines[-1]]))
    except Exception:
        return None

    if not header or not values or len(header) != len(values):
        return None

    engine_values: list[float] = []
    for name, value in zip(header[1:], values[1:]):  # skip timestamp column
        if not value:
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue

        engine_type = _extract_engine_type(name)
        if engine_type in {"3D", "Compute"}:
            engine_values.append(numeric)

    if not engine_values:
        return None

    # Sum the active engines and clamp to 100% to avoid overshooting when multiple engines fire.
    total = min(100.0, sum(engine_values))
    return total


def _extract_engine_type(counter_path: str) -> Optional[str]:
    # cspell: disable-next-line
    marker = "engtype_"
    if marker not in counter_path:
        return None
    suffix = counter_path.split(marker, 1)[1]
    end = suffix.find(")")
    if end == -1:
        return suffix
    return suffix[:end]


def _read_cuda_memory() -> Dict[str, Optional[float]]:
    if torch is None or not hasattr(torch.cuda, "mem_get_info"):
        return {}
    try:
        device_index = torch.cuda.current_device()
    except Exception:
        device_index = 0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    except TypeError:  # pragma: no cover - older signatures allow no args
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return {}
    except Exception:
        return {}

    total_mb = total_bytes / (1024**2)
    used_mb = max(0.0, total_mb - (free_bytes / (1024**2)))
    return {
        "gpu_memory_used": used_mb,
        "gpu_memory_total": total_mb,
    }


_MONITOR: Optional[_SystemStatsMonitor] = None


def _is_main_process() -> bool:
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True


if _is_main_process() and os.environ.get("XAI_DISABLE_SYSTEM_STATS") != "1":
    try:
        _MONITOR = _SystemStatsMonitor(interval=1.0)
    except Exception:
        _MONITOR = None


def get_system_stats() -> Dict[str, Optional[float]]:
    """Return the latest system stats snapshot."""
    if _MONITOR is None:
        return {key: None for key in _CPU_KEYS}
    return _MONITOR.snapshot()