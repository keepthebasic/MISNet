from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import load_artifact, load_meta
from scripts.npz_dataset import NPZDataset, split_npz_paths


def _sla_weights(model: torch.nn.Module, enc: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    sla = model.lab_readout
    avail = mask.mean(dim=-1)
    if getattr(sla, "mute_dead_channels", False):
        thr = float(getattr(sla, "dead_channel_thresh", 1e-4))
        avail = avail * (avail > thr).to(dtype=avail.dtype)
    bias = sla.lab_proxy(avail).unsqueeze(-1)
    logits = sla.score(enc) + bias
    w = torch.softmax(logits, dim=-1)
    return w.squeeze(0).squeeze(0).detach().cpu().numpy()


def _chla_forecast_mse(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    chla_idx: int,
    horizon_idx: int,
) -> float:
    out = model(x, mask)
    pred = out["forecast"][:, chla_idx, horizon_idx]
    tgt = y[:, chla_idx, horizon_idx]
    m = y_mask[:, chla_idx, horizon_idx]
    if float(m.sum()) < 1.0:
        return float("nan")
    err = (pred - tgt) ** 2
    return float((err * m).sum() / m.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E6")
    parser.add_argument("--model", default="limon")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--proc-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target", default="chla")
    parser.add_argument(
        "--method",
        default="sla_attn",
        choices=["shap", "sla_attn", "permute"],
        help="shap=|x| proxy; sla_attn=SLA weights; permute=channel shuffle ΔMSE",
    )
    parser.add_argument("--layer", default="lab_readout")
    parser.add_argument("--n-samples", type=int, default=32, help="average over N windows")
    parser.add_argument("--horizon-idx", type=int, default=5, help="0-based forecast step (5→24 h if H=6)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = load_meta(args.proc_dir)
    device = torch.device("cpu")
    run_dir = args.ckpt.parent
    model = load_artifact(args.model, run_dir, meta, device)
    model.eval()
    paths = split_npz_paths(args.proc_dir)
    ds = NPZDataset(paths[args.split])
    n = min(args.n_samples, len(ds))
    easy = list(meta.get("easy_vars", [f"c{i}" for i in range(ds[0]["x"].shape[0])]))
    forecast_vars = list(meta.get("forecast_vars", ["do", "chla"]))
    try:
        chla_f = forecast_vars.index("chla")
    except ValueError:
        chla_f = min(1, len(forecast_vars) - 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.method == "permute":
        store = ds._store
        fym_all = store["forecast_y_mask"]
        valid_idx = np.where(fym_all[:, chla_f, args.horizon_idx] > 0)[0]
        if len(valid_idx) == 0:
            raise SystemExit(
                f"[interpret] no valid {args.target} labels at horizon_idx={args.horizon_idx} "
                f"on split={args.split}; check meta forecast_vars / mask."
            )
        take = valid_idx[: min(n, len(valid_idx))]
        print(f"[interpret] permute using {len(take)}/{len(valid_idx)} labeled windows "
              f"(chla_idx={chla_f}, horizon_idx={args.horizon_idx})")

        batches = []
        with torch.no_grad():
            for i in take:
                x = torch.from_numpy(store["x"][int(i)]).unsqueeze(0)
                mask = torch.from_numpy(store["mask"][int(i)]).unsqueeze(0)
                y = torch.from_numpy(store["forecast_y"][int(i)]).unsqueeze(0)
                ym = torch.from_numpy(store["forecast_y_mask"][int(i)]).unsqueeze(0)
                y = y.clone()
                y[ym <= 0] = 0.0
                batches.append((x, mask, y, ym))

        base_errs = [
            _chla_forecast_mse(model, x, mask, y, ym, chla_f, args.horizon_idx)
            for x, mask, y, ym in batches
        ]
        base = float(np.nanmean(base_errs))
        if not np.isfinite(base):
            raise SystemExit(f"[interpret] baseline MSE is {base}; aborting.")

        deltas = []
        with torch.no_grad():
            for c in range(len(easy)):
                errs = []
                for x, mask, y, ym in batches:
                    xp = x.clone()
                    T = xp.shape[-1]
                    perm = torch.from_numpy(rng.permutation(T))
                    xp[0, c, :] = xp[0, c, perm]
                    errs.append(
                        _chla_forecast_mse(model, xp, mask, y, ym, chla_f, args.horizon_idx)
                    )
                d = float(np.nanmean(errs)) - base
                deltas.append(d if np.isfinite(d) else 0.0)
        order = np.argsort(deltas)[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(
            [easy[i] for i in order][::-1],
            [deltas[i] for i in order][::-1],
            color="#1f6f8b",
        )
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_xlabel(r"$\Delta$MSE (permuted $-$ baseline), 24 h Chl-$a$")
        ax.set_title(f"Channel permutation importance (n={len(take)} labeled windows)")
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[interpret] permute base_mse={base:.6f} deltas={dict(zip(easy, deltas))}")
        print(f"[interpret] saved -> {args.out}")
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if args.method == "sla_attn" and getattr(model, "lab_readout", None) is not None:
        weights = []
        with torch.no_grad():
            for i in range(n):
                batch = ds[i]
                x = batch["x"].unsqueeze(0)
                mask = batch["mask"].unsqueeze(0)
                enc = model.encode(x, mask)
                weights.append(_sla_weights(model, enc, mask))
        w = np.mean(np.stack(weights, axis=0), axis=0)
        ax.plot(np.arange(len(w)), w, color="#1f6f8b", lw=1.5)
        ax.fill_between(np.arange(len(w)), w, alpha=0.25, color="#1f6f8b")
        ax.set_xlabel("Time step in look-back window")
        ax.set_ylabel("SLA weight")
        ax.set_title(f"SLA temporal attention (mean of {n} test windows)")
    else:
        mats = []
        for i in range(n):
            mats.append(np.abs(ds[i]["x"].numpy()))
        imp = np.mean(np.stack(mats, axis=0), axis=0)
        if imp.ndim == 1:
            imp = imp.reshape(1, -1)
        im = ax.imshow(imp, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(np.arange(len(easy)))
        ax.set_yticklabels(easy)
        ax.set_xlabel("Time step")
        ax.set_title(f"Input |x| magnitude proxy ({args.target}, n={n})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|z|")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[interpret] saved -> {args.out}")


if __name__ == "__main__":
    main()
