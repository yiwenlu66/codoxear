from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _set_pdeathsig(sig: int) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception:
        return


def _require_proc(*, proc_root: Path, platform: str = sys.platform, stderr: Any = sys.stderr) -> None:
    if platform.startswith("linux"):
        if not (proc_root / "self" / "fd").is_dir():
            stderr.write("error: codoxear-broker requires /proc (missing /proc/self/fd).\n")
            raise SystemExit(2)
    elif platform == "darwin":
        pass  # macOS is supported via lsof/pgrep
    else:
        stderr.write(f"error: codoxear-broker requires Linux or macOS (unsupported: {platform}).\n")
        raise SystemExit(2)
