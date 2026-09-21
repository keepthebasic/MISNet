from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import EASY_VARS, FORECAST_VARS, PROJECT_ROOT, SOFT_VARS


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _load_era5(exo_dir: Path, freq: str) -> pd.DataFrame | None:
    if not exo_dir.is_dir():
        return None
    files = sorted(exo_dir.glob("*.csv"))
    if not files:
        return None

    parts = [pd.read_csv(f) for f in files]
    era5 = pd.concat(parts, ignore_index=True)
    time_col = "time" if "time" in era5.columns else "datetime"
    era5["datetime"] = pd.to_datetime(era5[time_col], utc=True, errors="coerce")
    era5 = era5.dropna(subset=["datetime"]).sort_values("datetime")

    era5 = era5.set_index("datetime")
    agg = {}
    if "precipitation" in era5.columns:
        agg["precipitation"] = "sum"
    if "temperature_2m" in era5.columns:
        agg["temperature_2m"] = "mean"
    if not agg:
        return None

    era5 = era5[list(agg.keys())].resample(freq).agg(agg)
    era5 = era5.reset_index()
    era5["datetime"] = era5["datetime"].dt.tz_convert(None)
    era5 = era5.rename(
        columns={"precipitation": "exo_precip", "temperature_2m": "exo_temp"}
    )
    return era5


def _merge_exo(df: pd.DataFrame, era5: pd.DataFrame | None) -> pd.DataFrame:
    if era5 is None:
        df["exo_precip"] = np.nan
        df["exo_temp"] = np.nan
        df["exo_precip_mask"] = 0.0
        df["exo_temp_mask"] = 0.0
        return df

    merged = pd.merge_asof(
        df.sort_values("datetime"),
        era5.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )
    for col in ("exo_precip", "exo_temp"):
        if col not in merged.columns:
            merged[col] = np.nan
        merged[f"{col}_mask"] = (~merged[col].isna()).astype(np.float32)
    return merged


def _blocked_split(n: int, ratios: tuple[float, float, float]) -> tuple[slice, slice, slice]:
    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    i1 = int(n * r_train)
    i2 = int(n * (r_train + r_val))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def _invalidate_constant_runs(
    df: pd.DataFrame,
    cols: list[str],
    min_run: int = 36,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        mcol = f"{col}_mask"
        if col not in out.columns or mcol not in out.columns:
            continue
        vals = out[col].to_numpy(dtype=np.float64, copy=True)
        mask = (out[mcol].fillna(0.0).to_numpy(dtype=np.float64) > 0) & np.isfinite(vals)
        n = len(vals)
        i = 0
        scrubbed = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j] and abs(vals[j] - vals[i]) <= 1e-6:
                j += 1
            if (j - i) >= min_run:
                mask[i:j] = False
                scrubbed += j - i
            i = j
        if scrubbed:
            out[mcol] = mask.astype(np.float32)
            out.loc[~mask, col] = np.nan
            print(f"[data_process] scrubbed {scrubbed} constant-run points for {col} (min_run={min_run})")
    return out


def _fit_scalers(
    df: pd.DataFrame, cols: list[str], train_slice: slice
) -> dict[str, dict[str, float]]:
    scalers: dict[str, dict[str, float]] = {}
    train = df.iloc[train_slice]
    for col in cols:
        mask_col = f"{col}_mask"
        if mask_col in train.columns:
            valid = train[col].where(train[mask_col] > 0)
        else:
            valid = train[col]
        valid = valid.dropna()
        if len(valid) == 0:
            scalers[col] = {"mean": 0.0, "std": 1.0}
        else:
            std = float(valid.std())
            scalers[col] = {"mean": float(valid.mean()), "std": std if std > 1e-8 else 1.0}
    return scalers


def _apply_scalers(df: pd.DataFrame, cols: list[str], scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        m, s = scalers[col]["mean"], scalers[col]["std"]
        mask_col = f"{col}_mask"
        if mask_col in out.columns:
            observed = out[mask_col] > 0
            out.loc[observed, col] = (out.loc[observed, col] - m) / s
        else:
            out[col] = (out[col] - m) / s
    return out


def _build_windows(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    stride: int,
    row_slice: slice,
) -> dict[str, np.ndarray]:
    sub = df.iloc[row_slice].reset_index(drop=True)
    n_rows = len(sub)
    need = seq_len + pred_len
    if n_rows < need:
        return {}

    easy = np.stack([sub[v].to_numpy(dtype=np.float32) for v in EASY_VARS], axis=0)
    easy_mask = np.stack(
        [sub[f"{v}_mask"].to_numpy(dtype=np.float32) for v in EASY_VARS], axis=0
    )

    soft = np.stack([sub[v].to_numpy(dtype=np.float32) for v in SOFT_VARS], axis=0)
    soft_mask = np.stack(
        [sub[f"{v}_mask"].to_numpy(dtype=np.float32) for v in SOFT_VARS], axis=0
    )

    fc_vars = FORECAST_VARS
    fc = np.stack([sub[v].to_numpy(dtype=np.float32) for v in fc_vars], axis=0)
    fc_mask = np.stack(
        [sub[f"{v}_mask"].to_numpy(dtype=np.float32) for v in fc_vars], axis=0
    )

    has_exo = "exo_precip" in sub.columns
    if has_exo:
        exo = np.stack(
            [sub["exo_precip"].to_numpy(dtype=np.float32), sub["exo_temp"].to_numpy(dtype=np.float32)],
            axis=0,
        )
        exo_mask = np.stack(
            [
                sub["exo_precip_mask"].to_numpy(dtype=np.float32),
                sub["exo_temp_mask"].to_numpy(dtype=np.float32),
            ],
            axis=0,
        )

    xs, masks, soft_ys, soft_ms = [], [], [], []
    fc_ys, fc_ms = [], []
    exos, exo_ms, times = [], [], []

    for start in range(0, n_rows - need + 1, stride):
        end = start + seq_len
        fut_end = end + pred_len

        xs.append(np.where(easy_mask[:, start:end] > 0, easy[:, start:end], 0.0))
        masks.append(easy_mask[:, start:end])
        soft_ys.append(soft[:, end - 1])
        soft_ms.append(soft_mask[:, end - 1])

        fc_ys.append(fc[:, end:fut_end])
        fc_ms.append(fc_mask[:, end:fut_end])

        if has_exo:
            exos.append(exo[:, start:end])
            exo_ms.append(exo_mask[:, start:end])

        times.append(sub.loc[end - 1, "datetime"])

    if not xs:
        return {}

    pack: dict[str, np.ndarray] = {
        "x": np.stack(xs, axis=0),
        "mask": np.stack(masks, axis=0),
        "soft_y": np.stack(soft_ys, axis=0),
        "soft_y_mask": np.stack(soft_ms, axis=0),
        "forecast_y": np.stack(fc_ys, axis=0),
        "forecast_y_mask": np.stack(fc_ms, axis=0),
        "timestamp": np.array(times, dtype="datetime64[ns]"),
    }
    if has_exo:
        pack["exo"] = np.stack(exos, axis=0)
        pack["exo_mask"] = np.stack(exo_ms, axis=0)
    return pack


def _event_stats(df: pd.DataFrame, test_slice: slice) -> dict:
    test = df.iloc[test_slice]
    stats = {}
    if "do" in test.columns and f"do_mask" in test.columns:
        do = test["do"].where(test["do_mask"] > 0)
        stats["low_do_events"] = int((do < 5.0).sum())
    if "tp" in test.columns and f"tp_mask" in test.columns:
        tp = test["tp"].where(test["tp_mask"] > 0)
        p90 = tp.quantile(0.9) if tp.notna().any() else np.nan
        stats["tp_p90"] = float(p90) if pd.notna(p90) else None
        stats["high_tp_events"] = int((tp > p90).sum()) if pd.notna(p90) else 0
    if "chla" in test.columns and f"chla_mask" in test.columns:
        chla = test["chla"].where(test["chla_mask"] > 0)
        stats["bloom_events"] = int((chla > 10.0).sum())
    return stats


def process_dataset(
    input_csv: Path,
    out_dir: Path,
    seq_len: int = 168,
    pred_len: int = 6,
    stride: int = 1,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    freq: str = "4h",
    exo_era5: Path | None = None,
) -> dict:
    df = _load_csv(input_csv)
    era5 = _load_era5(exo_era5, freq) if exo_era5 else None
    df = _merge_exo(df, era5)
    qc_cols = [c for c in list(EASY_VARS) + list(SOFT_VARS) if c in df.columns]
    df = _invalidate_constant_runs(df, qc_cols, min_run=36)

    n = len(df)
    train_sl, val_sl, test_sl = _blocked_split(n, split)

    all_cols = list(EASY_VARS) + list(SOFT_VARS) + ["exo_precip", "exo_temp"]
    scalers = _fit_scalers(df, [c for c in all_cols if c in df.columns], train_sl)
    df_scaled = _apply_scalers(df, [c for c in all_cols if c in df.columns], scalers)

    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_sl, "val": val_sl, "test": test_sl}
    counts = {}
    for name, sl in splits.items():
        pack = _build_windows(df_scaled, seq_len, pred_len, stride, sl)
        if not pack:
            print(f"[data_process] WARNING: no windows for split={name}")
            counts[name] = 0
            continue
        np.savez_compressed(out_dir / f"{name}.npz", **pack)
        counts[name] = int(pack["x"].shape[0])
        print(f"[data_process] {name}: {counts[name]} samples -> {out_dir / (name + '.npz')}")

    missing = {
        v: float(1.0 - (df[f"{v}_mask"].fillna(0.0) > 0).mean())
        if f"{v}_mask" in df.columns
        else 1.0
        for v in list(EASY_VARS) + list(SOFT_VARS)
    }

    meta = {
        "input_csv": str(input_csv),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "stride": stride,
        "split": list(split),
        "n_rows": n,
        "n_samples": counts,
        "easy_vars": list(EASY_VARS),
        "soft_vars": list(SOFT_VARS),
        "forecast_vars": list(FORECAST_VARS),
        "missing_rate": missing,
        "split_dates": {
            "train_end": str(df.iloc[train_sl.stop - 1]["datetime"]) if train_sl.stop > 0 else None,
            "val_end": str(df.iloc[val_sl.stop - 1]["datetime"]) if val_sl.stop > val_sl.start else None,
            "test_start": str(df.iloc[test_sl.start]["datetime"]) if test_sl.start < n else None,
        },
        "events_test": _event_stats(df, test_sl),
    }

    with open(out_dir / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[data_process] scalers -> {out_dir / 'scalers.json'}")
    print(f"[data_process] meta    -> {out_dir / 'meta.json'}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test NPZ windows")
    parser.add_argument("--input", type=Path, required=True, help="Aligned 4h CSV from lakebed_to_csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split", default="0.70,0.15,0.15", help="train,val,test ratios")
    parser.add_argument("--split-mode", default="blocked", choices=["blocked"])
    parser.add_argument("--standardize", default="per_site")
    parser.add_argument("--stats-from", default="train")
    parser.add_argument("--exo-era5", type=Path, default=None, help="ERA5 hourly CSV directory")
    parser.add_argument("--make-splits", default="train,val,test")
    args = parser.parse_args()

    ratios = tuple(float(x) for x in args.split.split(","))
    if len(ratios) != 3:
        raise SystemExit("--split must have 3 comma-separated values")

    process_dataset(
        input_csv=args.input,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        split=ratios,
        exo_era5=args.exo_era5,
    )


if __name__ == "__main__":
    main()
