from __future__ import annotations

from pathlib import Path

EASY_VARS = ("temp", "ph", "turbidity", "ec", "do")
SOFT_VARS = ("tp", "tn", "chla")
FORECAST_VARS = ("do", "chla")
ALL_VARS = EASY_VARS + tuple(v for v in SOFT_VARS if v not in EASY_VARS)

LAKEBED_ALIASES: dict[str, tuple[str, ...]] = {
    "temp": ("temp", "temperature", "water_temp"),
    "ph": ("ph", "pH"),
    "turbidity": ("turbidity", "turb", "turbidity_ntu", "fdom", "phyco"),
    "ec": (
        "ec",
        "conductivity",
        "specific_conductance",
        "specific_conductivity",
        "sp_cond",
        "sp_conductivity",
        "cond",
    ),
    "do": ("do", "dissolved_oxygen", "dissolved_oxygen_concentration"),
    "tp": ("tp", "total_phosphorus", "total_p"),
    "tn": ("tn", "total_nitrogen", "total_n"),
    "chla": ("chla_ugl", "chla"),
    "chla_rfu": ("chla_rfu",),
}

SOFT_LAB_VARS = ("tp", "tn", "chla")

ACCEPTABLE_FLAGS = {0, 5, 10, 19, 23, 25, 32, 43, 47, 51, 52}

LAKEBED_CORE_LAKES = ("ME", "BVR", "FCR", "TR", "SP")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
