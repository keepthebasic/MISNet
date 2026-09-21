from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _horizon_hours(pred_len: int, step_h: float = 4.0) -> list[float]:
    return [step_h * (i + 1) for i in range(pred_len)]


def count_split(npz_path: Path, chla_idx: int = 1, step_h: float = 4.0) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    mask = z["forecast_y_mask"]
    ts = z["timestamp"].astype("datetime64[ns]")
    n_win, n_var, n_h = mask.shape
    chla_idx = min(max(chla_idx, 0), n_var - 1)
    hours = _horizon_hours(n_h, step_h)
    by_h = {}
    for hi, hrs in enumerate(hours):
        valid = mask[:, chla_idx, hi] > 0
        n_windows = int(valid.sum())
        delta = np.timedelta64(int(hrs * 3600), "s")
        tgt = ts + delta
        unique = np.unique(tgt[valid]) if n_windows else np.array([], dtype="datetime64[ns]")
        by_h[f"h{hi + 1}"] = {
            "horizon_hours": hrs,
            "n_overlapping_windows": n_windows,
            "n_unique_target_times": int(unique.size),
            "inflation": (
                None
                if unique.size == 0
                else round(n_windows / float(unique.size), 3)
            ),
        }
    any_valid = (mask[:, chla_idx, :] > 0).any(axis=1)
    return {
        "n_windows_total": int(n_win),
        "chla_channel_index": chla_idx,
        "by_horizon": by_h,
        "n_windows_any_chla_horizon": int(any_valid.sum()),
        "primary_claim": "Use n_unique_target_times at h6 (24 h) as the independent N, not overlapping windows.",
    }


def count_csv_events(csv_path: Path, split_dates: dict | None) -> dict:
    if not csv_path.exists():
        return {"error": f"missing {csv_path}"}
    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    if "chla_mask" not in df.columns:
        return {"error": "no chla_mask"}
    obs = df.loc[df["chla_mask"] > 0, "datetime"]
    out: dict = {
        "n_chla_obs_all": int(len(obs)),
        "n_unique_chla_times_all": int(obs.nunique()),
    }
    if split_dates and "test_start" in split_dates:
        t0 = pd.Timestamp(split_dates["test_start"])
        test_obs = obs[obs >= t0]
        out["n_chla_obs_test_period"] = int(len(test_obs))
        out["n_unique_chla_times_test_period"] = int(test_obs.nunique())
        if "chla" in df.columns:
            ch = df.loc[(df["chla_mask"] > 0) & (df["datetime"] >= t0), "chla"]
            out["n_bloom_gt10_test"] = int((ch > 10).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=ROOT / "data" / "processed" / "lakebed_BVR_4h")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "BVR_4h" / "independent_n.json",
    )
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.proc_dir / "meta.json").read_text(encoding="utf-8"))
    fc = list(meta.get("forecast_vars") or ["do", "chla"])
    chla_idx = fc.index("chla") if "chla" in fc else 1
    payload = {
        "proc_dir": str(args.proc_dir),
        "splits": {},
        "csv_events": {},
    }
    csvs = list(args.proc_dir.glob("*_4h.csv"))
    if csvs:
        payload["csv_events"] = count_csv_events(csvs[0], meta.get("split_dates"))

    for split in ("train", "val", "test"):
        p = args.proc_dir / f"{split}.npz"
        if p.exists():
            payload["splits"][split] = count_split(p, chla_idx=chla_idx)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = args.out_md or args.out_json.with_suffix(".md")
    lines = [
        "# Independent Chl-a N",
        "",
        f"CSV unique Chl times (all): {payload['csv_events'].get('n_unique_chla_times_all')}",
        f"CSV unique Chl times (test period): {payload['csv_events'].get('n_unique_chla_times_test_period')}",
        "",
        "| split | windows (h6) | unique target times (h6) | inflation |",
        "|-------|-------------:|-------------------------:|----------:|",
    ]
    for split, block in payload["splits"].items():
        h6 = (block.get("by_horizon") or {}).get("h6") or {}
        lines.append(
            f"| {split} | {h6.get('n_overlapping_windows')} | "
            f"{h6.get('n_unique_target_times')} | {h6.get('inflation')} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[independent_n] wrote {args.out_json}")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
