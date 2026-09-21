from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import run_predictions, load_scalers, denormalize
from scripts.npz_dataset import split_npz_paths


def _set_wr_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", required=True)
    parser.add_argument("--proc-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", default="persistence,limon,lstm")
    parser.add_argument("--metric", default="rmse")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _set_wr_style()

    if args.fig == "fig3_horizon_degradation" and args.results_dir:
        _fig3(args)
    elif args.fig == "fig2_timeseries" and args.proc_dir and args.results_dir:
        _fig2(args)
    elif args.fig == "fig4_warning" and args.results_dir:
        _fig4(args)
    elif args.fig == "fig5_missing_robustness" and args.results_dir:
        _fig5(args)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Placeholder: {args.fig}", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[plot] saved -> {args.out}")


def _fig2(args):
    import matplotlib.dates as mdates

    proc_dir = args.proc_dir
    scalers = load_scalers(proc_dir / "scalers.json")
    paths = split_npz_paths(proc_dir)
    store = np.load(paths[args.split], allow_pickle=True)
    ts = store["timestamp"].astype("datetime64[ns]")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    style = {
        "persistence": dict(c="#222222", marker="s", label="Persistence"),
        "limon": dict(c="#0072B2", marker="o", label="LIMON"),
        "lstm": dict(c="#D55E00", marker="^", label="LSTM"),
        "pls": dict(c="#009E73", marker="D", label="PLS"),
        "xgboost": dict(c="#CC79A7", marker="v", label="XGBoost"),
    }

    preds: dict[str, np.ndarray] = {}
    y_obs = None
    mask = None
    for model in models:
        run_dir = None
        if model != "persistence":
            cand = [args.results_dir / f"{model}_seed0"] + sorted(
                args.results_dir.glob(f"{model}_seed*")
            )
            run_dir = next((p for p in cand if p.exists()), None)
            if run_dir is None:
                print(f"[plot] skip {model}: no run dir")
                continue
        try:
            pred, obs, meta = run_predictions(model, run_dir, proc_dir, args.split)
        except Exception as exc:
            print(f"[plot] skip {model}: {exc}")
            continue
        i = meta["soft_vars"].index("chla")
        if y_obs is None:
            y_obs = denormalize(obs["soft"][:, i], "chla", scalers)
            mask = obs["soft_mask"][:, i] > 0
        preds[model] = denormalize(pred["soft"][:, i], "chla", scalers)

    if y_obs is None or mask is None or not np.any(mask):
        raise SystemExit("No soft-sensing Chl-a labels for fig2")

    idx_all = np.where(mask)[0]
    t_all = ts[idx_all]
    y_all = y_obs[idx_all]

    win = np.timedelta64(60, "D")
    best_start = t_all[0]
    best_n = 0
    for t0 in t_all:
        n = int(np.sum((t_all >= t0) & (t_all < t0 + win)))
        if n > best_n:
            best_n = n
            best_start = t0
    sel = (t_all >= best_start) & (t_all < best_start + win)
    if int(sel.sum()) < 20:
        n_show = min(40, len(t_all))
        sel = np.zeros(len(t_all), dtype=bool)
        best_span = None
        best_i = 0
        for i0 in range(0, len(t_all) - n_show + 1):
            span = t_all[i0 + n_show - 1] - t_all[i0]
            if best_span is None or span < best_span:
                best_span = span
                best_i = i0
        sel[best_i : best_i + n_show] = True

    t_sel = t_all[sel]
    y_sel = y_all[sel]
    idx_sel = idx_all[sel]

    day = t_sel.astype("datetime64[D]")
    uniq_days = np.unique(day)
    t_day = uniq_days.astype("datetime64[ns]") + np.timedelta64(12, "h")
    y_day = np.array([float(np.median(y_sel[day == d])) for d in uniq_days])
    pred_day: dict[str, np.ndarray] = {}
    for model in preds:
        pv = preds[model][idx_sel]
        pred_day[model] = np.array([float(np.median(pv[day == d])) for d in uniq_days])

    best_j, best_score = 0, -1e18
    limon_s = preds.get("limon")
    for j in range(len(uniq_days)):
        keep_j = (uniq_days >= uniq_days[j]) & (uniq_days < uniq_days[j] + np.timedelta64(18, "D"))
        n = int(keep_j.sum())
        if n < 8:
            continue
        days_j = uniq_days[keep_j]
        yj = np.array([float(np.median(y_sel[day == d])) for d in days_j])
        score = float(n)
        if limon_s is not None:
            pj = np.array([float(np.median(limon_s[idx_sel][day == d])) for d in days_j])
            mae = float(np.mean(np.abs(pj - yj)))
            spike = float(np.max(pj) / (np.max(yj) + 0.5))
            score = n - 0.4 * mae - 3.0 * max(0.0, spike - 2.5)
        if score > best_score:
            best_score, best_j = score, j
    keep = (uniq_days >= uniq_days[best_j]) & (uniq_days < uniq_days[best_j] + np.timedelta64(18, "D"))
    days_k = uniq_days[keep]
    t_day = days_k.astype("datetime64[ns]") + np.timedelta64(12, "h")
    y_day = np.array([float(np.median(y_sel[day == d])) for d in days_k])
    pred_day = {
        m: np.array([float(np.median(preds[m][idx_sel][day == d])) for d in days_k])
        for m in preds
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34},
    )
    ax0, ax1 = axes

    ax0.plot(
        t_day,
        y_day,
        color="0.15",
        lw=1.3,
        marker="o",
        ms=5,
        label="Observed",
        zorder=5,
    )
    for model in models:
        if model not in pred_day:
            continue
        st = style.get(model, dict(c="C0", marker="o", label=model))
        ax0.plot(
            t_day,
            pred_day[model],
            color=st["c"],
            lw=1.15,
            marker=st["marker"],
            ms=4.5,
            mfc="white",
            mew=1.05,
            label=st["label"],
            zorder=4,
        )
    ax0.set_ylabel(r"Chl-$a$ ($\mathrm{\mu g\,L^{-1}}$)")
    ax0.set_xlabel("Date")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax0.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    for label in ax0.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ymax = float(np.nanmax([y_day.max()] + [pred_day[m].max() for m in pred_day])) * 1.12
    ax0.set_ylim(0.0, ymax)
    ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, fontweight="bold", fontsize=10, va="top")
    ax0.legend(frameon=True, fancybox=False, edgecolor="0.85", loc="upper right", fontsize=7, framealpha=0.95)

    day_all = t_all.astype("datetime64[D]")
    uniq_all = np.unique(day_all)
    y_d = np.array([float(np.median(y_all[day_all == d])) for d in uniq_all])
    keep_b = y_d <= float(np.nanpercentile(y_d, 75))
    y_b = y_d[keep_b]
    lim = max(float(np.nanmax(y_b)) * 1.25, 2.5)
    ax1.plot([0, lim], [0, lim], ls="--", c="0.55", lw=0.9, zorder=1)
    for model in ("persistence", "limon"):
        if model not in preds:
            continue
        st = style[model]
        pv = preds[model][idx_all]
        p_d = np.array([float(np.median(pv[day_all == d])) for d in uniq_all])[keep_b]
        ax1.scatter(
            y_b,
            np.clip(p_d, 0, lim),
            s=28,
            facecolors="white",
            edgecolors=st["c"],
            marker=st["marker"],
            linewidths=1.05,
            alpha=0.8,
            label=st["label"],
            zorder=2,
        )
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"Observed Chl-$a$ ($\mathrm{\mu g\,L^{-1}}$)")
    ax1.set_ylabel(r"Predicted Chl-$a$ ($\mathrm{\mu g\,L^{-1}}$)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.02, 0.98, "(b)", transform=ax1.transAxes, fontweight="bold", fontsize=10, va="top")
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.85", loc="lower right", fontsize=7, framealpha=0.95)
    ax1.text(0.97, 0.22, "1:1", transform=ax1.transAxes, ha="right", va="bottom", fontsize=7, color="0.45")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[plot] also saved -> {pdf}  (a: {len(t_day)} days; b: {len(y_b)} days ≤P75 obs)")


def _fig3(args):
    import pandas as pd

    metric = getattr(args, "metric", "rmse") or "rmse"
    target = "chla"
    order = ["limon", "lstm", "persistence", "pls", "xgboost"]
    colors = {
        "limon": "#0072B2",
        "lstm": "#D55E00",
        "persistence": "#222222",
        "pls": "#009E73",
        "xgboost": "#CC79A7",
    }

    table = None
    if args.results_dir is not None:
        cand = [
            args.results_dir.parent / "tables" / "table3_multistep_do_chla.csv",
            args.results_dir / "table3_multistep_do_chla.csv",
            ROOT / "results" / "BVR_4h" / "tables" / "table3_multistep_do_chla.csv",
        ]
        for c in cand:
            if c.exists():
                table = c
                break

    series: dict[str, tuple[list[int], list[float]]] = {}
    if table is not None:
        df = pd.read_csv(table)
        sub = df[(df["target"] == target) & (df["metric"] == metric)]
        for model, g in sub.groupby("model"):
            g = g.copy()
            g["h"] = g["horizon"].astype(str).str.replace("h", "", regex=False).astype(int)
            g = g.sort_values("h")
            series[str(model)] = (g["h"].tolist(), g["mean"].astype(float).tolist())
    else:
        buckets: dict[str, dict[int, list[float]]] = {}
        paths = sorted(args.results_dir.glob("*_E2_metrics.json"))
        if not paths:
            paths = sorted(args.results_dir.rglob("*_E2_metrics.json"))
        for p in paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            model = str(data.get("model", p.stem.split("_")[0])).lower()
            if "metrics" not in data or "by_horizon" not in data["metrics"]:
                continue
            for hk, hv in data["metrics"]["by_horizon"].items():
                if target in hv and metric in hv[target]:
                    h = int(str(hk).replace("h", ""))
                    buckets.setdefault(model, {}).setdefault(h, []).append(float(hv[target][metric]))
        for model, by_h in buckets.items():
            hs = sorted(by_h)
            series[model] = (hs, [float(np.mean(by_h[h])) for h in hs])

    if not series:
        raise SystemExit(f"No Chl-a {metric} series for fig3 under {args.results_dir}")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    labels = {
        "limon": "LIMON",
        "lstm": "LSTM",
        "persistence": "Persistence",
        "pls": "PLS",
        "xgboost": "XGBoost",
    }
    for model in order:
        if model not in series:
            continue
        hs, vals = series[model]
        ax.plot(hs, vals, marker="o", ms=4, lw=1.4, color=colors.get(model), label=labels.get(model, model))
    ax.set_xlabel("Lead step (4 h; step 6 ≈ 24 h)")
    ax.set_ylabel(r"RMSE (Chl-$a$, $\mathrm{\mu g\,L^{-1}}$)")
    ax.set_xticks(sorted({h for hs, _ in series.values() for h in hs}))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")


def _fig4(args):
    cands = sorted(args.results_dir.glob("warning_*.json"))
    if not cands:
        cands = sorted(args.results_dir.rglob("warning_*.json"))
    if not cands:
        raise SystemExit(f"No warning_*.json under {args.results_dir}")
    path = cands[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    events = ["do_low", "chla_bloom"]
    models = list(data.get("models", {}).keys())
    means = {ev: [] for ev in events}
    for model in models:
        runs = data["models"][model]
        for ev in events:
            vals = []
            for r in runs:
                e = r.get("events", {}).get(ev, {})
                if e.get("skipped"):
                    continue
                vals.append(float(e.get("f1", 0.0)))
            means[ev].append(float(np.mean(vals)) if vals else 0.0)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(x - width / 2, means["do_low"], width, label="DO low", color="#0072B2")
    ax.bar(x + width / 2, means["chla_bloom"], width, label="Chl-$a$ bloom", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m == "limon" else m.capitalize() for m in models], rotation=15)
    ax.set_ylabel("F1 (mean over seeds)")
    ax.set_ylim(0, max(0.35, max(means["do_low"] + means["chla_bloom"]) * 1.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")


def _fig5(args):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for sub in sorted(args.results_dir.glob("E5c_mask_*")):
        rate = sub.name.split("_")[-1]
        xs, ys = [], []
        for mf in sub.glob("**/E2_metrics.json"):
            d = json.loads(mf.read_text(encoding="utf-8"))
            try:
                ys.append(d["metrics"]["by_target"]["chla"]["h6"]["nse"])
                xs.append(float(rate))
            except Exception:
                pass
        if xs:
            ax.scatter(xs, ys, label=sub.name)
    ax.set_xlabel("Mask rate")
    ax.set_ylabel("NSE (Chl-a @24h)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
