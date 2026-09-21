from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDataset(Dataset):

    def __init__(self, npz_path: Path, indices: np.ndarray | None = None):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self._keys = list(data.files)
        self._store = {k: data[k] for k in self._keys}
        data.close()

        n = int(self._store["x"].shape[0])
        if indices is None:
            self.indices = np.arange(n, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    @property
    def has_exo(self) -> bool:
        return "exo" in self._store

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])

        def _label(arr: np.ndarray, m: np.ndarray) -> np.ndarray:
            out = arr.copy()
            invalid = (m <= 0) | ~np.isfinite(out)
            out[invalid] = 0.0
            return out

        soft_m = self._store["soft_y_mask"][i]
        fc_m = self._store["forecast_y_mask"][i]
        batch: dict[str, torch.Tensor] = {
            "x": torch.from_numpy(self._store["x"][i]),
            "mask": torch.from_numpy(self._store["mask"][i]),
            "soft_y": torch.from_numpy(_label(self._store["soft_y"][i], soft_m)),
            "soft_y_mask": torch.from_numpy(soft_m),
            "forecast_y": torch.from_numpy(_label(self._store["forecast_y"][i], fc_m)),
            "forecast_y_mask": torch.from_numpy(fc_m),
        }
        if "soft_lab_mask" in self._store:
            batch["soft_lab_mask"] = torch.from_numpy(self._store["soft_lab_mask"][i])
        if self.has_exo:
            batch["exo"] = torch.from_numpy(self._store["exo"][i])
            if "exo_mask" in self._store:
                batch["exo_mask"] = torch.from_numpy(self._store["exo_mask"][i])
        return batch


def load_meta(proc_dir: Path) -> dict[str, Any]:
    meta_path = Path(proc_dir) / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def split_npz_paths(proc_dir: Path) -> dict[str, Path]:
    proc_dir = Path(proc_dir)
    return {
        "train": proc_dir / "train.npz",
        "val": proc_dir / "val.npz",
        "test": proc_dir / "test.npz",
    }


def subsample_indices(n: int, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(n, dtype=np.int64)
    k = max(1, int(n * frac))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def merge_npz_files(npz_paths: list[Path], out_path: Path) -> int:
    paths = [Path(p) for p in npz_paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(f"No NPZ files to merge: {npz_paths}")
    chunks = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = set(chunks[0].keys())
    for c in chunks[1:]:
        keys &= set(c.keys())
    keys.discard("soft_lab_mask")
    required = {"x", "mask", "soft_y", "soft_y_mask", "forecast_y", "forecast_y_mask"}
    missing = required - keys
    if missing:
        raise KeyError(f"NPZ merge missing required keys: {sorted(missing)}")
    keys_list = sorted(keys)
    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys_list}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    return int(merged["x"].shape[0])


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def _denorm_npz(
    store: dict[str, np.ndarray],
    meta: dict[str, Any],
    scalers: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in store.items()}
    easy = list(meta["easy_vars"])
    soft = list(meta["soft_vars"])
    fc = list(meta["forecast_vars"])

    for i, v in enumerate(easy):
        if v not in scalers:
            continue
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        out["x"][:, i, :] = out["x"][:, i, :].astype(np.float64) * s + m

    for i, v in enumerate(soft):
        if v not in scalers:
            continue
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        out["soft_y"][:, i] = out["soft_y"][:, i].astype(np.float64) * s + m

    for i, v in enumerate(fc):
        if v not in scalers:
            continue
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        out["forecast_y"][:, i, :] = out["forecast_y"][:, i, :].astype(np.float64) * s + m

    if "exo" in out:
        exo_names = ["exo_precip", "exo_temp"]
        for i, v in enumerate(exo_names):
            if i >= out["exo"].shape[1] or v not in scalers:
                continue
            m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
            out["exo"][:, i, :] = out["exo"][:, i, :].astype(np.float64) * s + m
    return out


def _fit_scaler_1d(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    x = values.astype(np.float64).ravel()
    if mask is not None:
        m = mask.astype(np.float64).ravel() > 0
        x = x[m]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": 0.0, "std": 1.0}
    std = float(x.std())
    return {"mean": float(x.mean()), "std": std if std > 1e-8 else 1.0}


def _fit_scalers_from_physical_stores(
    stores: list[dict[str, np.ndarray]],
    meta: dict[str, Any],
) -> dict[str, dict[str, float]]:
    easy = list(meta["easy_vars"])
    soft = list(meta["soft_vars"])
    fc = list(meta["forecast_vars"])
    scalers: dict[str, dict[str, float]] = {}

    for i, v in enumerate(easy):
        vals = np.concatenate([s["x"][:, i, :].ravel() for s in stores])
        masks = np.concatenate([s["mask"][:, i, :].ravel() for s in stores])
        scalers[v] = _fit_scaler_1d(vals, masks)

    for i, v in enumerate(soft):
        vals = np.concatenate([s["soft_y"][:, i].ravel() for s in stores])
        masks = np.concatenate([s["soft_y_mask"][:, i].ravel() for s in stores])
        scalers[v] = _fit_scaler_1d(vals, masks)

    for i, v in enumerate(fc):
        if v in scalers and v in soft:
            continue
        vals = np.concatenate([s["forecast_y"][:, i, :].ravel() for s in stores])
        masks = np.concatenate([s["forecast_y_mask"][:, i, :].ravel() for s in stores])
        scalers[v] = _fit_scaler_1d(vals, masks)

    if any("exo" in s for s in stores):
        for i, v in enumerate(["exo_precip", "exo_temp"]):
            chunks_v, chunks_m = [], []
            for s in stores:
                if "exo" not in s or i >= s["exo"].shape[1]:
                    continue
                chunks_v.append(s["exo"][:, i, :].ravel())
                if "exo_mask" in s:
                    chunks_m.append(s["exo_mask"][:, i, :].ravel())
            if not chunks_v:
                continue
            vals = np.concatenate(chunks_v)
            masks = np.concatenate(chunks_m) if chunks_m else None
            scalers[v] = _fit_scaler_1d(vals, masks)
    return scalers


def _apply_scalers_npz(
    store: dict[str, np.ndarray],
    meta: dict[str, Any],
    scalers: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in store.items()}
    easy = list(meta["easy_vars"])
    soft = list(meta["soft_vars"])
    fc = list(meta["forecast_vars"])

    for i, v in enumerate(easy):
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        z = (out["x"][:, i, :].astype(np.float64) - m) / s
        bad = (out["mask"][:, i, :] <= 0) | ~np.isfinite(z)
        z[bad] = 0.0
        out["x"][:, i, :] = z.astype(np.float32)

    for i, v in enumerate(soft):
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        z = (out["soft_y"][:, i].astype(np.float64) - m) / s
        bad = (out["soft_y_mask"][:, i] <= 0) | ~np.isfinite(z)
        z[bad] = 0.0
        out["soft_y"][:, i] = z.astype(np.float32)

    for i, v in enumerate(fc):
        m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
        z = (out["forecast_y"][:, i, :].astype(np.float64) - m) / s
        bad = (out["forecast_y_mask"][:, i, :] <= 0) | ~np.isfinite(z)
        z[bad] = 0.0
        out["forecast_y"][:, i, :] = z.astype(np.float32)

    if "exo" in out:
        for i, v in enumerate(["exo_precip", "exo_temp"]):
            if i >= out["exo"].shape[1] or v not in scalers:
                continue
            m, s = float(scalers[v]["mean"]), float(scalers[v]["std"])
            z = (out["exo"][:, i, :].astype(np.float64) - m) / s
            if "exo_mask" in out:
                bad = (out["exo_mask"][:, i, :] <= 0) | ~np.isfinite(z)
            else:
                bad = ~np.isfinite(z)
            z[bad] = 0.0
            out["exo"][:, i, :] = z.astype(np.float32)

    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            out[k] = v.astype(np.float32)
    return out


def build_lolo_proc_dir(
    proc_root: Path,
    train_lakes: list[str],
    test_lake: str,
    out_dir: Path,
    scaler_mode: str = "target_train_only",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_root = Path(proc_root)
    test_proc = proc_root / f"lakebed_{test_lake}_4h"
    meta = _load_json(test_proc / "meta.json")

    train_phys: list[dict[str, np.ndarray]] = []
    for lk in train_lakes:
        lp = proc_root / f"lakebed_{lk}_4h"
        sc = _load_json(lp / "scalers.json")
        store = _load_npz_dict(lp / "train.npz")
        train_phys.append(_denorm_npz(store, _load_json(lp / "meta.json"), sc))

    test_sc = _load_json(test_proc / "scalers.json")
    test_meta = meta
    target_train_phys = _denorm_npz(
        _load_npz_dict(test_proc / "train.npz"), test_meta, test_sc
    )
    target_val_phys = _denorm_npz(
        _load_npz_dict(test_proc / "val.npz"), test_meta, test_sc
    )
    target_test_phys = _denorm_npz(
        _load_npz_dict(test_proc / "test.npz"), test_meta, test_sc
    )

    if scaler_mode == "train_lakes":
        fit_stores = train_phys
        scalers_source = "merged_train_lakes_physical"
    else:
        scaler_mode = "target_train_only"
        fit_stores = [target_train_phys]
        scalers_source = f"lakebed_{test_lake}_4h/train.npz (physical, re-fit)"

    scalers = _fit_scalers_from_physical_stores(fit_stores, test_meta)

    train_scaled = [_apply_scalers_npz(s, test_meta, scalers) for s in train_phys]
    key_sets = [set(s.keys()) for s in train_scaled]
    keys = set.intersection(*key_sets) if key_sets else set()
    keys.discard("soft_lab_mask")
    required = {"x", "mask", "soft_y", "soft_y_mask", "forecast_y", "forecast_y_mask", "timestamp"}
    missing = required - keys
    if missing:
        raise KeyError(
            f"LOLO merge missing required keys {sorted(missing)}; "
            f"per-lake key counts={[len(s) for s in key_sets]}"
        )
    keys_list = sorted(keys)
    merged_train = {k: np.concatenate([s[k] for s in train_scaled], axis=0) for k in keys_list}
    np.savez_compressed(out_dir / "train.npz", **merged_train)

    def _strip_optional(store: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: store[k] for k in keys_list if k in store}

    np.savez_compressed(
        out_dir / "val.npz",
        **_strip_optional(_apply_scalers_npz(target_val_phys, test_meta, scalers)),
    )
    np.savez_compressed(
        out_dir / "test.npz",
        **_strip_optional(_apply_scalers_npz(target_test_phys, test_meta, scalers)),
    )

    meta_out = dict(test_meta)
    meta_out["lolo"] = {
        "train_lakes": train_lakes,
        "test_lake": test_lake,
        "scaler_mode": scaler_mode,
        "n_train": int(merged_train["x"].shape[0]),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, default=str)
    with open(out_dir / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)

    note = {
        "protocol": "E4b_lolo_v2_unified_scaler",
        "train_lakes": train_lakes,
        "test_lake": test_lake,
        "scaler_mode": scaler_mode,
        "scalers_source": scalers_source,
        "fix": (
            "Denormalize per-lake NPZ → fit one scaler → re-standardize. "
            "Do not concatenate heterogeneous z-score spaces."
        ),
    }
    (out_dir / "lolo_protocol.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(
        f"[lolo] test={test_lake} train={train_lakes} mode={scaler_mode} "
        f"n_train={merged_train['x'].shape[0]} -> {out_dir}"
    )
    return out_dir
