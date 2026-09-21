from __future__ import annotations

import os


def apply_runtime_env() -> None:
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "4"))


apply_runtime_env()
