from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import discover_run_dirs, load_scalers, parse_seeds, run_predictions, denormalize


def _event_masks(
    forecast: np.ndarray,
    forecast_mask: np.ndarray,
    soft: np.ndarray,
    soft_mask: np.ndarray,
    meta: dict,
    thresholds: dict[str, float],
    tp_p90: float,
) -> dict[str, np.ndarray]:
    fc_vars = meta["forecast_vars"]
    soft_vars = meta["soft_vars"]
    events: dict[str, np.ndarray] = {}
    if "do_low" in thresholds and "do" in fc_vars:
        i = fc_vars.index("do")
        pred_do = forecast[:, i, :].min(axis=1)
        events["do_low"] = pred_do < thresholds["do_low"]
    if "chla_bloom" in thresholds and "chla" in fc_vars:
        i = fc_vars.index("chla")
        pred_chla = forecast[:, i, :].max(axis=1)
        events["chla_bloom"] = pred_chla > thresholds["chla_bloom"]
    if "tp_high" in thresholds or "tp_high_p90" in thresholds:
        if "tp" in soft_vars:
            i = soft_vars.index("tp")
            events["tp_high"] = soft[:, i] > tp_p90
    return events


def _truth_events(obs, meta, thresholds, tp_p90, pred_len):
    fc_vars = meta["forecast_vars"]
    soft_vars = meta["soft_vars"]
    truth: dict[str, np.ndarray] = {}
    if "do" in fc_vars:
        i = fc_vars.index("do")
        tdo = obs["forecast"][:, i, :]
        tm = obs["forecast_mask"][:, i, :] > 0
        truth["do_low"] = np.where(tm.any(axis=1), (tdo < thresholds.get("do_low", 5.0)).any(axis=1), False)
    if "chla" in fc_vars:
        i = fc_vars.index("chla")
        tch = obs["forecast"][:, i, :]
        tm = obs["forecast_mask"][:, i, :] > 0
        truth["chla_bloom"] = np.where(tm.any(axis=1), (tch > thresholds.get("chla_bloom", 10.0)).any(axis=1), False)
    if "tp" in soft_vars:
        i = soft_vars.index("tp")
        truth["tp_high"] = (obs["soft"][:, i] > tp_p90) & (obs["soft_mask"][:, i] > 0)
    return truth


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    far = fp / (fp + tp) if (fp + tp) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "far": far, "tp": tp, "fp": fp, "fn": fn, "n_events": int(y_true.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--proc-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="do_low=5.0,chla_bloom=10.0")
    parser.add_argument("--main-events", default="do_low,chla_bloom")
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--metrics", default="f1,lead_time,precision,recall,pr_auc")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--models", default=None)
    parser.add_argument("--denormalize", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    thr = {}
    for part in args.thresholds.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            thr[k.strip()] = float(v.strip())

    scalers = load_scalers(args.proc_dir / "scalers.json") if args.denormalize else None
    train = __import__("scripts.npz_dataset", fromlist=["NPZDataset"]).NPZDataset(
        args.proc_dir / "train.npz"
    )
    tp_train = train._store["soft_y"][:, 0]
    tp_m = train._store["soft_y_mask"][:, 0] > 0
    tp_p90 = float(np.quantile(tp_train[tp_m], thr.get("tp_high_p90", 0.9))) if tp_m.any() else float("inf")

    seeds = parse_seeds(args.seeds)
    models = [m.strip() for m in (args.models or "limon,xgboost,persistence").split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_out: dict[str, Any] = {"scenario": args.scenario, "models": {}}

    for model in models:
        runs = discover_run_dirs(args.results_dir, model, seeds)
        model_rows = []
        for seed in seeds:
            run_dir = None if model == "persistence" else runs.get(seed)
            pred, obs, meta = run_predictions(model, run_dir, args.proc_dir, args.split)
            if scalers:
                for v in meta["forecast_vars"]:
                    i = meta["forecast_vars"].index(v)
                    pred["forecast"][:, i, :] = denormalize(pred["forecast"][:, i, :], v, scalers)
                    obs["forecast"][:, i, :] = denormalize(obs["forecast"][:, i, :], v, scalers)
                for v in meta["soft_vars"]:
                    i = meta["soft_vars"].index(v)
                    pred["soft"][:, i] = denormalize(pred["soft"][:, i], v, scalers)
                    obs["soft"][:, i] = denormalize(obs["soft"][:, i], v, scalers)

            pred_ev = _event_masks(pred["forecast"], obs["forecast_mask"], pred["soft"], obs["soft_mask"], meta, thr, tp_p90)
            true_ev = _truth_events(obs, meta, thr, tp_p90, int(meta["pred_len"]))
            row = {"seed": seed, "events": {}}
            for ev in [e.strip() for e in args.main_events.split(",") if e.strip()]:
                if ev not in pred_ev or ev not in true_ev:
                    continue
                sc = _scores(true_ev[ev], pred_ev[ev])
                if sc["n_events"] < args.min_events:
                    sc["skipped"] = True
                row["events"][ev] = sc
            model_rows.append(row)
        all_out["models"][model] = model_rows

    out_path = args.out_dir / f"warning_{args.scenario}.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"[evaluate_warning] saved -> {out_path}")


if __name__ == "__main__":
    main()
