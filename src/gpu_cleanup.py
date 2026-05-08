"""
Lightweight GPU/process cleanup helpers.

Provides:
 - safe_run_cmd(cmd_list, cwd): runs a subprocess in its own process group and
   attempts to terminate the group on interrupts.
 - cleanup_torch(): best-effort garbage collection and device cache clearing.
 - register_signal_handlers(): register simple handlers to run cleanup_torch on SIGINT/SIGTERM.
"""
from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


def safe_run_cmd(cmd: Sequence[str], cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """Run a command as a subprocess in a new process group.

    Args:
        cmd: list of program + args. Prefer full interpreter path (sys.executable).
        cwd: working directory.

    On KeyboardInterrupt the child process group will be signalled with SIGTERM.
    """
    if isinstance(cmd, str):
        # defensive: split into tokens
        import shlex

        cmd = shlex.split(cmd)

    proc = subprocess.Popen(list(cmd), cwd=cwd, preexec_fn=os.setsid)
    try:
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
        return subprocess.CompletedProcess(proc.args, ret)
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        raise
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        raise


def cleanup_torch() -> None:
    """Best-effort cleanup of GPU-related resources.

    This function is intentionally defensive: it will not raise on failures.
    """
    try:
        gc.collect()
    except Exception:
        pass

    if torch is None:
        return

    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if hasattr(torch, "backends") and getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available() and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    except Exception:
        pass


def register_signal_handlers() -> None:
    """Register SIGINT/SIGTERM handlers that run cleanup and then exit.

    Call this from scripts that hold GPU resources to improve cleanup on abrupt exits.
    """

    def _handler(signum, frame):
        try:
            cleanup_torch()
        finally:
            # exit with 128 + signal for conventional code
            try:
                os._exit(128 + signum)
            except Exception:
                raise SystemExit(128 + signum)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except Exception:
            # non-POSIX or other environments might fail here
            pass
