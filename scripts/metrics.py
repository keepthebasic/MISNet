from __future__ import annotations

import numpy as np


def _valid_pairs(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if mask is None:
        valid = np.isfinite(pred) & np.isfinite(obs)
    else:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        valid = (m > 0) & np.isfinite(pred) & np.isfinite(obs)
    return pred[valid], obs[valid]


def nse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, o = _valid_pairs(pred, obs, mask)
    if len(o) < 2:
        return float("nan")
    denom = np.sum((o - o.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((p - o) ** 2) / denom)


def rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, o = _valid_pairs(pred, obs, mask)
    if len(o) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def mae(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, o = _valid_pairs(pred, obs, mask)
    if len(o) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - o)))


def r2(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, o = _valid_pairs(pred, obs, mask)
    if len(o) < 2:
        return float("nan")
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


METRIC_FNS = {
    "nse": nse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}


def compute_masked_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray | None,
    metrics: tuple[str, ...] = ("nse", "rmse", "mae", "r2"),
) -> dict[str, float]:
    return {m: METRIC_FNS[m](pred, obs, mask) for m in metrics if m in METRIC_FNS}
