from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (
    ACCEPTABLE_FLAGS,
    ALL_VARS,
    LAKEBED_ALIASES,
    PROJECT_ROOT,
    SOFT_LAB_VARS,
)


def _parse_freq(freq: str) -> str:
    f = freq.strip().lower()
    if f.endswith("h") and f[:-1].isdigit():
        return f"{int(f[:-1])}h"
    return f


def _find_column(df: pd.DataFrame, standard: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in LAKEBED_ALIASES.get(standard, (standard,)):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _parquet_columns_needed() -> set[str]:
    cols = {"datetime", "depth", "flag"}
    for aliases in LAKEBED_ALIASES.values():
        cols.update(a.lower() for a in aliases)
    return cols


def _read_parquet(fp: Path, depth_surface: float | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema_names = {name.lower(): name for name in pq.read_schema(fp).names}
    wanted = _parquet_columns_needed()
    use = [schema_names[k] for k in wanted if k in schema_names]
    if "datetime" not in {c.lower() for c in use}:
        return pd.DataFrame()

    depth_col = schema_names.get("depth")
    size_mb = fp.stat().st_size / (1024 * 1024)
    num_rows = pq.read_metadata(fp).num_rows
    large = size_mb >= 50 or num_rows > 2_000_000

    if not large:
        df = pd.read_parquet(fp, columns=use)
        if depth_surface is not None and depth_col is not None and depth_col in df.columns:
            df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
            lo = max(0.0, depth_surface - 2.0)
            hi = depth_surface + 2.0
            df = df[df[depth_col].between(lo, hi)]
        return df

    if depth_surface is None or depth_col is None:
        return pd.read_parquet(fp, columns=use)

    def _batched_between(lo: float, hi: float) -> pd.DataFrame:
        pf = pq.ParquetFile(fp)
        chunks: list[pd.DataFrame] = []
        for batch in pf.iter_batches(batch_size=200_000, columns=use):
            part = batch.to_pandas()
            part[depth_col] = pd.to_numeric(part[depth_col], errors="coerce")
            part = part[part[depth_col].between(lo, hi)]
            if not part.empty:
                chunks.append(part)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    exact = _batched_between(depth_surface, depth_surface)
    if not exact.empty:
        return exact
    lo = max(0.0, depth_surface - 2.0)
    hi = depth_surface + 2.0
    return _batched_between(lo, hi)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "datetime" not in {c.lower() for c in df.columns}:
        return pd.DataFrame()
    dt_col = next(c for c in df.columns if c.lower() == "datetime")
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    if "flag" in {c.lower() for c in df.columns}:
        flag_col = next(c for c in df.columns if c.lower() == "flag")
        df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce")
        df = df[df[flag_col].isin(ACCEPTABLE_FLAGS) | df[flag_col].isna()]
    return df


def _load_parquet_dir(raw_dir: Path, depth_surface: float = 1.0) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {raw_dir}. Download LakeBeD first:\n"
            "  pip install huggingface_hub\n"
            "  python scripts/download_lakebed.py --hf-only"
        )

    parts: list[pd.DataFrame] = []
    for fp in files:
        df = _read_parquet(fp, depth_surface=depth_surface)
        if df.empty:
            continue
        df = _normalize_df(df)
        if df.empty:
            continue
        parts.append(df)

    if not parts:
        raise ValueError(f"Could not parse any parquet in {raw_dir}")

    return pd.concat(parts, ignore_index=True).sort_values("datetime")


def _select_surface(df: pd.DataFrame, depth_surface: float) -> pd.DataFrame:
    if "depth" not in {c.lower() for c in df.columns}:
        return df.drop_duplicates(subset=["datetime"], keep="last")

    depth_col = next(c for c in df.columns if c.lower() == "depth")
    df = df.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[depth_col])

    if df[depth_col].nunique() == 1:
        return df.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")

    df["_depth_delta"] = (df[depth_col] - depth_surface).abs()
    idx = df.groupby("datetime", sort=False)["_depth_delta"].idxmin()
    return df.loc[idx].drop(columns="_depth_delta").sort_values("datetime")


def _map_standard_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    out = pd.DataFrame({"datetime": df["datetime"]})
    sources: dict[str, str | None] = {}
    for std in ALL_VARS:
        src = _find_column(df, std)
        sources[std] = src
        if src is None:
            out[std] = np.nan
            out[f"{std}_mask"] = 0.0
        else:
            out[std] = pd.to_numeric(df[src], errors="coerce")
            out[f"{std}_mask"] = (~out[std].isna()).astype(np.float32)
    return out, sources


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.set_index("datetime").sort_index()
    value_cols = list(ALL_VARS)
    mask_cols = [f"{v}_mask" for v in ALL_VARS]

    resampled_values = df[value_cols].resample(freq).mean()
    resampled_mask = df[mask_cols].resample(freq).max().fillna(0.0)

    out = pd.concat([resampled_values, resampled_mask], axis=1)
    out = out.reset_index()
    if out["datetime"].dt.tz is not None:
        out["datetime"] = out["datetime"].dt.tz_convert(None)
    return out


def _select_lf_epilimnion(df: pd.DataFrame) -> pd.DataFrame:
    if "depth" not in {c.lower() for c in df.columns}:
        return df.drop_duplicates(subset=["datetime"], keep="last")

    depth_col = next(c for c in df.columns if c.lower() == "depth")
    df = df.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[depth_col])
    df = df[df[depth_col] >= 0.0]

    idx = df.groupby("datetime", sort=False)[depth_col].idxmin()
    return df.loc[idx].sort_values("datetime")


def _load_low_frequency(low_freq_dir: Path, depth_surface: float) -> pd.DataFrame:
    del depth_surface
    files = sorted(low_freq_dir.glob("*2D*.parquet"))
    if not files:
        files = sorted(low_freq_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = _normalize_df(df)
        if df.empty:
            continue
        parts.append(_select_lf_epilimnion(df))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("datetime")


def _merge_lf_onto_hf(hf: pd.DataFrame, lf_raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    if lf_raw.empty:
        return hf

    lf_mapped, lf_sources = _map_standard_columns(lf_raw)
    lf = _resample(lf_mapped, freq)
    print("  LF column mapping:")
    for v in ALL_VARS:
        print(f"    {v:10s} <- {lf_sources.get(v) or '(not in LF)'}")

    out = hf.sort_values("datetime").reset_index(drop=True)
    fill_vars = [v for v in ALL_VARS if v not in ("temp", "do")]
    lf_cols = ["datetime"] + [c for v in fill_vars for c in (v, f"{v}_mask") if c in lf.columns]

    merged = out.merge(lf[lf_cols], on="datetime", how="left", suffixes=("", "_lf"))

    for var in fill_vars:
        lf_val_col = f"{var}_lf"
        lf_mask_col = f"{var}_mask_lf"
        if lf_val_col not in merged.columns:
            continue
        lf_present = (
            merged[lf_mask_col] > 0
            if lf_mask_col in merged.columns
            else merged[lf_val_col].notna()
        )
        if var in SOFT_LAB_VARS:
            use_lf = lf_present
        else:
            hf_missing = merged[f"{var}_mask"] <= 0
            use_lf = hf_missing & lf_present
        merged.loc[use_lf, var] = merged.loc[use_lf, lf_val_col]
        merged.loc[use_lf, f"{var}_mask"] = 1.0

        if var in SOFT_LAB_VARS:
            hf_only = (merged[f"{var}_mask"] > 0) & ~lf_present
            if var == "chla" and hf_only.any():
                vals = merged.loc[hf_only, var]
                bad = vals > 500.0
                if bad.any():
                    idx = vals.index[bad]
                    merged.loc[idx, var] = np.nan
                    merged.loc[idx, f"{var}_mask"] = 0.0
                    print(f"  [qc] dropped {int(bad.sum())} HF-only chla values > 500 (likely RFU)")

    drop_cols = [c for c in merged.columns if c.endswith("_lf") or c.endswith("_mask_lf")]
    return merged.drop(columns=drop_cols, errors="ignore")


def _make_demo_csv(out: Path, freq: str, n_days: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n_days * (24 // 4), freq=freq)
    rng = np.random.default_rng(42)
    t = np.arange(len(idx), dtype=np.float32)

    data = {
        "datetime": idx,
        "temp": 15 + 8 * np.sin(2 * np.pi * t / (24 // 4 * 30)) + rng.normal(0, 0.5, len(idx)),
        "ph": 7.5 + 0.3 * np.sin(2 * np.pi * t / (24 // 4 * 7)) + rng.normal(0, 0.05, len(idx)),
        "turbidity": 5 + rng.normal(0, 1, len(idx)).clip(0),
        "ec": 200 + rng.normal(0, 10, len(idx)),
        "do": 8 + 2 * np.sin(2 * np.pi * t / (24 // 4 * 14)) + rng.normal(0, 0.3, len(idx)),
        "tp": 20 + rng.normal(0, 5, len(idx)).clip(0),
        "tn": 500 + rng.normal(0, 50, len(idx)).clip(0),
        "chla": 5 + 3 * np.sin(2 * np.pi * t / (24 // 4 * 60)).clip(0) + rng.normal(0, 0.5, len(idx)).clip(0),
    }
    df = pd.DataFrame(data)
    for v in ALL_VARS:
        df[f"{v}_mask"] = 1.0
        if v in ("tp", "tn"):
            miss = rng.random(len(df)) < 0.05
            df.loc[miss, v] = np.nan
            df.loc[miss, f"{v}_mask"] = 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[lakebed_to_csv] DEMO synthetic rows={len(df)} -> {out}")
    return df


def lakebed_to_csv(
    lake: str,
    raw_dir: Path,
    out: Path,
    freq: str = "4h",
    depth_surface: float = 1.0,
    low_freq_dir: Path | None = None,
) -> pd.DataFrame:
    raw = _load_parquet_dir(raw_dir, depth_surface=depth_surface)
    raw = _select_surface(raw, depth_surface)

    mapped, sources = _map_standard_columns(raw)
    resampled = _resample(mapped, _parse_freq(freq))

    if low_freq_dir and low_freq_dir.is_dir():
        lf_raw = _load_low_frequency(low_freq_dir, depth_surface)
        if not lf_raw.empty:
            resampled = _merge_lf_onto_hf(resampled, lf_raw, _parse_freq(freq))
            print(f"  merged LowFrequency from {low_freq_dir}")

    out.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(out, index=False)

    n = len(resampled)
    miss = {v: float(1.0 - resampled[f"{v}_mask"].mean()) for v in ALL_VARS}
    print(f"[lakebed_to_csv] lake={lake} rows={n} freq={freq} -> {out}")
    print("  HF column mapping:")
    for v in ALL_VARS:
        src = sources.get(v)
        print(f"    {v:10s} <- {src or '(not in HF)'}")
    for v in ALL_VARS:
        print(f"  {v:10s} missing={miss[v]:.1%}")
    return resampled


def main() -> None:
    parser = argparse.ArgumentParser(description="LakeBeD parquet → 4h CSV")
    parser.add_argument("--lake", required=True, help="Lake ID e.g. ME, BVR")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--low-freq-dir", type=Path, default=None)
    parser.add_argument("--freq", default="4h")
    parser.add_argument("--depth-surface", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-days", type=int, default=400)
    args = parser.parse_args()

    if args.demo:
        _make_demo_csv(args.out, _parse_freq(args.freq), n_days=args.demo_days)
        return

    raw_dir = args.raw_dir or (
        PROJECT_ROOT / "data" / "LakeBeD-US-CSE" / "Data" / "HighFrequency" / args.lake
    )
    low_freq = args.low_freq_dir or (
        PROJECT_ROOT / "data" / "LakeBeD-US-CSE" / "Data" / "LowFrequency" / args.lake
    )

    try:
        lakebed_to_csv(
            lake=args.lake,
            raw_dir=raw_dir,
            out=args.out,
            freq=args.freq,
            depth_surface=args.depth_surface,
            low_freq_dir=low_freq if low_freq.is_dir() else None,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
