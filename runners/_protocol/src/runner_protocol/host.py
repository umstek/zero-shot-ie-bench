"""Host fingerprint served on GET /info.

Stdlib only (no psutil): the bench matches this against a machine_id in
catalog/machines/, so the values must be the real hardware, not defaults.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _cpu_name() -> str:
    """CPU model string: registry on Windows, /proc/cpuinfo on Linux, else platform."""
    if sys.platform == "win32":
        import winreg  # noqa: PLC0415 — Windows-only import kept local

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(name).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().replace(" ", "").startswith("modelname"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    """Total physical RAM in GiB."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
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

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / 1024**3, 1)
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            return round(page_size * page_count / 1024**3, 1)
        except (ValueError, OSError, AttributeError):
            pass
    return 0.0


def _gpus() -> list[dict[str, Any]]:
    """Discrete GPUs via nvidia-smi when present; empty list is a valid answer."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        name, _, mib = line.partition(",")
        try:
            gpus.append({"model": name.strip(), "vram_gb": round(float(mib) / 1024.0, 1)})
        except ValueError:
            continue
    return gpus


def host_fingerprint() -> dict[str, Any]:
    """Return {hostname, cpu, ram_gb, gpus, os} for the machine this process runs on."""
    return {
        "hostname": platform.node() or "unknown",
        "cpu": _cpu_name(),
        "ram_gb": _ram_gb(),
        "gpus": _gpus(),
        "os": platform.platform(terse=True),
    }
