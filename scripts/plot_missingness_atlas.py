from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAKES = ("BVR", "ME", "FCR", "TR", "SP")


def _gap_stats(mask_1d: np.ndarray) -> dict:
    missing = mask_1d < 0.5
    rate = float(missing.mean()) if len(missing) else 1.0
    runs = []
    i, n = 0, len(missing)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        runs.append(j - i)
        i = j
    runs_a = np.asarray(runs, dtype=np.float64) if runs else np.asarray([0.0])
    return {
        "missing_rate": rate,
        "n_gap_runs": int(len(runs)),
        "gap_len_mean": float(runs_a.mean()),
        "gap_len_p95": float(np.quantile(runs_a, 0.95)) if runs else 0.0,
        "gap_len_max": float(runs_a.max()) if runs else 0.0,
        "dead_channel": rate >= 0.999,
    }


def atlas_one(proc_dir: Path) -> dict:
    meta_path = proc_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    from_meta = meta.get("missing_rates") or meta.get("var_missing_rates") or {}
    easy = list(meta.get("easy_vars") or ["temp", "ph", "turbidity", "ec", "do"])
    soft = list(meta.get("soft_vars") or ["tp", "tn", "chla"])
    out = {
        "proc_dir": str(proc_dir),
        "n_rows": meta.get("n_rows"),
        "channels": {},
        "from_meta_rates": from_meta,
    }
    npz_path = proc_dir / "train.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        if "mask" in z.files:
            mask = z["mask"]
            if mask.ndim == 3:
                c = min(mask.shape[1], len(easy))
                for ci in range(c):
                    m = mask[:, ci, :].reshape(-1)
                    name = easy[ci] if ci < len(easy) else f"ch{ci}"
                    out["channels"][name] = _gap_stats(m)
        if "soft_y_mask" in z.files:
            sm = z["soft_y_mask"]
            for si, name in enumerate(soft[: sm.shape[-1]]):
                out["channels"][name] = _gap_stats(sm[:, si])
    for k, v in from_meta.items():
        if k not in out["channels"]:
            rate = float(v) if not isinstance(v, dict) else float(v.get("missing_rate", v))
            out["channels"][k] = {
                "missing_rate": rate,
                "n_gap_runs": None,
                "gap_len_mean": None,
                "gap_len_p95": None,
                "gap_len_max": None,
                "dead_channel": rate >= 0.999,
            }
    dead = [k for k, v in out["channels"].items() if v.get("dead_channel")]
    out["dead_channels"] = dead
    out["summary_missing_range"] = None
    rates = [v["missing_rate"] for v in out["channels"].values() if v.get("missing_rate") is not None]
    if rates:
        out["summary_missing_range"] = [float(min(rates)), float(max(rates))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--lakes", default=",".join(DEFAULT_LAKES))
    parser.add_argument("--out-json", type=Path, default=ROOT / "results" / "BVR_4h" / "missingness_atlas.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "results" / "BVR_4h" / "missingness_atlas.md")
    args = parser.parse_args()

    lakes = [x.strip() for x in args.lakes.split(",") if x.strip()]
    atlas = {}
    for lake in lakes:
        proc = args.processed_root / f"lakebed_{lake}_4h"
        if not proc.exists():
            atlas[lake] = {"error": f"missing {proc}"}
            continue
        atlas[lake] = atlas_one(proc)

    payload = {"lakes": atlas}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Missingness atlas (five lakes)", ""]
    for lake, block in atlas.items():
        lines.append(f"## {lake}")
        if "error" in block:
            lines.append(f"- ERROR: {block['error']}")
            continue
        lines.append(f"- n_rows: {block.get('n_rows')}")
        lines.append(f"- dead_channels: {block.get('dead_channels')}")
        lines.append(f"- missing_rate range: {block.get('summary_missing_range')}")
        for ch, st in sorted((block.get("channels") or {}).items()):
            lines.append(
                f"- {ch}: rate={st.get('missing_rate'):.3f} "
                f"gap_mean={st.get('gap_len_mean')} gap_p95={st.get('gap_len_p95')} "
                f"dead={st.get('dead_channel')}"
            )
        lines.append("")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[atlas] wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
