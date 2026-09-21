from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import (
    discover_run_dirs,
    load_scalers,
    metrics_with_denorm,
    parse_seeds,
    run_predictions,
)
from scripts.npz_dataset import load_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["E1", "E2"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--proc-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--horizons", default="1,2,3,6")
    parser.add_argument("--metrics", default="nse,rmse,mae,r2")
    parser.add_argument("--denormalize", type=int, default=0)
    parser.add_argument("--scalers", type=Path, default=None)
    parser.add_argument("--min-valid-n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    meta = load_meta(args.proc_dir)
    scalers_path = args.scalers or (args.proc_dir / "scalers.json")
    scalers = load_scalers(scalers_path) if args.denormalize else None
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    if args.seeds and args.results_dir:
        seeds = parse_seeds(args.seeds)
        runs = discover_run_dirs(args.results_dir, args.model or "", seeds)
        if not runs and len(seeds) == 1:
            rd = args.results_dir
            if rd.exists() and (
                (rd / "best.pt").exists()
                or (rd / "sklearn_model.pkl").exists()
                or (rd / "run_config.json").exists()
            ):
                runs = {seeds[0]: rd}
        all_metrics = []
        for seed, run_dir in runs.items():
            pred, obs, _ = run_predictions(args.model or "limon", run_dir, args.proc_dir, args.split, args.batch_size, args.device)
            row = _eval_one(args, pred, obs, meta, scalers, metrics)
            row["seed"] = seed
            all_metrics.append(row)
        result = {"exp": args.exp, "model": args.model, "multi_seed": all_metrics}
    else:
        run_dir = args.results_dir
        if run_dir and not args.model:
            cfg = run_dir / "run_config.json"
            if cfg.exists():
                args.model = json.loads(cfg.read_text(encoding="utf-8")).get("model")
        if not args.model:
            raise SystemExit("Provide --model")
        pred, obs, _ = run_predictions(args.model, run_dir, args.proc_dir, args.split, args.batch_size, args.device)
        result = {
            "exp": args.exp, "model": args.model, "split": args.split,
            "proc_dir": str(args.proc_dir), "denormalize": bool(args.denormalize),
            "metrics": _eval_one(args, pred, obs, meta, scalers, metrics),
        }

    out = args.out
    if out is None and args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{args.model}_{args.exp}_metrics.json"
    if out is None:
        out = Path(f"{args.model}_{args.exp}_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize_for_json(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"[evaluate] saved -> {out}")


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.integer)):
            return _sanitize_for_json(float(obj))
        if isinstance(obj, np.ndarray):
            return _sanitize_for_json(obj.tolist())
    except Exception:
        pass
    return obj


def _eval_one(args, pred, obs, meta, scalers, metrics) -> dict[str, Any]:
    if args.exp == "E1":
        vars_ = [t.strip() for t in (args.targets or "tp,tn,chla").split(",") if t.strip()]
        idxs = [meta["soft_vars"].index(v) for v in vars_ if v in meta["soft_vars"]]
        return metrics_with_denorm(
            pred["soft"][:, idxs], obs["soft"][:, idxs], obs["soft_mask"][:, idxs],
            [vars_[i] for i in range(len(idxs))], scalers, bool(args.denormalize), metrics, args.min_valid_n,
        )
    vars_ = [t.strip() for t in (args.targets or "do,chla").split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    out: dict[str, Any] = {"by_target": {}, "by_horizon": {}}
    for v in vars_:
        if v not in meta["forecast_vars"]:
            continue
        i = meta["forecast_vars"].index(v)
        out["by_target"][v] = {}
        for h in horizons:
            hi = h - 1
            key = f"h{h}"
            out["by_target"][v][key] = metrics_with_denorm(
                pred["forecast"][:, i, hi], obs["forecast"][:, i, hi], obs["forecast_mask"][:, i, hi],
                [v], scalers, bool(args.denormalize), metrics, args.min_valid_n,
            )[v]
            out.setdefault("by_horizon", {}).setdefault(key, {})[v] = out["by_target"][v][key]
    return out


if __name__ == "__main__":
    try:
        main()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        import os as _os

        _os._exit(0)
    except SystemExit as exc:
        if exc.code in (0, None):
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            import os as _os

            _os._exit(0)
        raise
