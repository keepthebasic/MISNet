from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POLICY = {
    "forecast_do": "High-frequency buoy DO (mg/L), 4 h bins; native missing retained.",
    "forecast_chla": (
        "Aligned onto 4 h bins from LakeBeD. LowFrequency laboratory Chl-a (µg/L) "
        "overrides HighFrequency whenever LF is present so buoy RFU cannot masquerade "
        "as lab µg/L (see scripts/lakebed_to_csv.py:_merge_lf_onto_hf). "
        "No forward-fill of lab chemistry."
    ),
    "soft_tp_tn_chla": (
        "Same 4 h series as forecast Chl-a / nutrients: lab LF overrides HF; "
        "soft head is nowcast at last input step, forecast head is t+4h…t+24h."
    ),
    "qc": "HF-only Chl-a values >500 dropped as likely RFU scale (lakebed_to_csv QC).",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=ROOT / "data" / "processed" / "lakebed_BVR_4h")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "BVR_4h" / "label_provenance.json",
    )
    args = parser.parse_args()
    meta = {}
    mp = args.proc_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    payload = {
        "proc_dir": str(args.proc_dir),
        "policy": POLICY,
        "easy_vars": meta.get("easy_vars"),
        "soft_vars": meta.get("soft_vars"),
        "forecast_vars": meta.get("forecast_vars"),
        "missing_rate": meta.get("missing_rate"),
        "split_dates": meta.get("split_dates"),
        "preprocessing": {
            "depth": "surface / epilimnion selection in lakebed_to_csv (depth_surface=1.0 m)",
            "resample": "4 h bins",
            "causal_windows": (
                "Input is [t-seq_len+1, t]; forecast labels are (t+1 … t+pred_len) only; "
                "no future easy-sensor values in x."
            ),
            "stride": meta.get("stride", 1),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = args.out_json.with_suffix(".md")
    md.write_text(
        "# Label provenance\n\n"
        f"- DO forecast: {POLICY['forecast_do']}\n"
        f"- Chl-a forecast: {POLICY['forecast_chla']}\n"
        f"- Soft TP/TN/Chl-a: {POLICY['soft_tp_tn_chla']}\n"
        f"- QC: {POLICY['qc']}\n"
        f"- Causal: {payload['preprocessing']['causal_windows']}\n",
        encoding="utf-8",
    )
    print(f"[labels] wrote {args.out_json} and {md}")


if __name__ == "__main__":
    main()
