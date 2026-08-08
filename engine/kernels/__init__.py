"""Triton kernels (Phase 5): fused softmax, Flash-Attention forward, INT8 matmul.

Environment note (Blackwell sm_120 / WSL2): Triton JIT-compiles a small CUDA shim with the
system ``gcc`` at first use, which needs the CPython development headers (``Python.h`` +
the multiarch ``pyconfig.h``). Those aren't installed here and installing them needs sudo,
so the headers were extracted from the ``python3.12-dev`` / ``libpython3.12-dev`` .debs into
``~/.local/pydev-include`` (no root). We prepend that location to ``CPATH`` here — before
Triton is imported — so the ``gcc`` subprocess Triton spawns inherits it. This keeps the
workaround inside the project (no shell-profile edits) and reproducible. If the headers are
already on the system, this is a harmless no-op.
"""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_python_headers_on_cpath() -> None:
    base = Path.home() / ".local" / "pydev-include"
    py = base / "python3.12"            # contains Python.h
    if not (py / "Python.h").exists():
        return                          # headers not staged here; assume system provides them
    # pyconfig.h is included as <x86_64-linux-gnu/python3.12/pyconfig.h>, so `base` (the
    # parent of the multiarch dir) must also be on the include path.
    extra = f"{py}:{base}"
    cur = os.environ.get("CPATH", "")
    if str(py) not in cur:
        os.environ["CPATH"] = f"{extra}:{cur}" if cur else extra


_ensure_python_headers_on_cpath()
