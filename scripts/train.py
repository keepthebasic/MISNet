from __future__ import annotations

import os
import sys

_ROOT_BOOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_BOOT not in sys.path:
    sys.path.insert(0, _ROOT_BOOT)
import scripts.runtime_env

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.misnet import (
    MISNetLossWeights,
    build_misnet,
    build_misnet_ablation,
    build_misnet_forecast_only,
    build_misnet_full,
    build_misnet_lite,
    build_misnet_rev_l2,
    build_misnet_rev_l3,
    misnet_loss,
)
from models.misnet_hfd import (
    build_misnet_hfd,
    build_misnet_hfd_chl,
    build_misnet_hfd_chl_ams,
    build_misnet_hfd_chl_decay,
    build_misnet_hfd_chl_hs,
    build_misnet_hfd_chl_tune,
    misnet_hfd_loss,
)

LIMONLossWeights = MISNetLossWeights
limon_loss = misnet_loss
limon_hfd_loss = misnet_hfd_loss
build_limon = build_misnet
build_limon_ablation = build_misnet_ablation
build_limon_forecast_only = build_misnet_forecast_only
build_limon_full = build_misnet_full
build_limon_lite = build_misnet_lite
build_limon_rev_l2 = build_misnet_rev_l2
build_limon_rev_l3 = build_misnet_rev_l3
build_limon_hfd = build_misnet_hfd
build_limon_hfd_chl = build_misnet_hfd_chl
build_limon_hfd_chl_ams = build_misnet_hfd_chl_ams
build_limon_hfd_chl_decay = build_misnet_hfd_chl_decay
build_limon_hfd_chl_hs = build_misnet_hfd_chl_hs
build_limon_hfd_chl_tune = build_misnet_hfd_chl_tune
from models.pars import (
    apply_structured_outage_batch,
    build_pars,
    build_pars_ablation,
    pars_loss,
)
from scripts.baselines import (
    BaselineGRUD,
    BaselineInformer,
    BaselineLSTM,
    BaselineMaskLSTM,
    BaselineMTLSTM,
    BaselinePatchTST,
    BaselineTCN,
    TORCH_BASELINE_TYPES,
    baseline_lstm_loss,
    train_sklearn_baseline,
)
from scripts.common import FORECAST_VARS, SOFT_VARS
from scripts.experiment_utils import (
    NO_TRAIN_MODELS,
    SKLEARN_MODELS,
    TORCH_MODELS,
    apply_hyperparams_to_train_args,
    load_hyperparams,
    set_seed,
)
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths, subsample_indices

STUB_MODELS = {"ufreqxnet", "lf_unimon_lite"}


def canonicalize_model_name(name: str) -> str:
    if name.startswith("limon"):
        return "misnet" + name[len("limon") :]
    return name


def parse_kv_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v.strip())
    return out


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name in ("cuda", "auto", "gpu") and not torch.cuda.is_available():
        import sys
        print(
            "[train] WARNING: --device cuda requested but torch.cuda.is_available() is False; "
            f"using CPU (torch {torch.__version__}). "
            "Install GPU build: pip install torch --index-url https://download.pytorch.org/whl/cu128 "
            "(RTX 50 系 Blackwell 需 cu128，cu124 不支持 sm_120)",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_tasks(variant: str | None, tasks: str) -> str:
    if variant == "wo_soft_head":
        return "forecast"
    if variant == "wo_forecast_head":
        return "soft"
    return tasks


def build_torch_model(args: argparse.Namespace, meta: dict[str, Any]) -> nn.Module:
    seq_len = int(meta["seq_len"])
    pred_len = int(meta["pred_len"])
    n_soft = len(meta["soft_vars"])
    n_forecast = len(meta["forecast_vars"])
    cin = len(meta["easy_vars"])
    hidden = getattr(args, "hidden", 64)
    layers = getattr(args, "num_layers", 2)
    dropout = getattr(args, "dropout", 0.1)
    d_model = getattr(args, "d_model", 64)

    if args.model == "lstm":
        return BaselineLSTM(cin, hidden=hidden, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, num_layers=layers, dropout=dropout)
    if args.model == "mask_lstm":
        return BaselineMaskLSTM(cin, hidden=hidden, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, num_layers=layers, dropout=dropout)
    if args.model == "mtlstm":
        return BaselineMTLSTM(cin, hidden=hidden, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, num_layers=layers, dropout=dropout)
    if args.model == "grud":
        return BaselineGRUD(cin, hidden=hidden, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, dropout=dropout)
    if args.model == "tcn":
        return BaselineTCN(cin, hidden=hidden, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len)
    if args.model == "informer":
        return BaselineInformer(
            cin,
            d_model=d_model,
            n_soft=n_soft,
            n_forecast=n_forecast,
            pred_len=pred_len,
            n_heads=int(getattr(args, "n_heads", 4)),
            num_layers=layers,
            dropout=dropout,
            factor=int(getattr(args, "factor", 5)),
        )
    if args.model == "patchtst":
        return BaselinePatchTST(
            cin,
            seq_len=seq_len,
            d_model=d_model,
            n_soft=n_soft,
            n_forecast=n_forecast,
            pred_len=pred_len,
            n_heads=int(getattr(args, "n_heads", 4)),
            num_layers=layers,
            dropout=dropout,
            patch_len=int(getattr(args, "patch_len", 16)),
            stride=int(getattr(args, "stride", 8)),
        )

    forge_kw = dict(
        in_channels=cin,
        seq_len=seq_len,
        pred_len=pred_len,
        n_soft=n_soft,
        n_forecast=n_forecast,
        use_mask=bool(args.use_mask),
        d_model=d_model,
        dropout=dropout,
    )
    if args.model in ("pars", "pars_forecast_only", "lake_forge", "lake_forge_forecast_only") or str(args.model).startswith(("pars", "lake_forge")):
        variant = args.ablation or args.variant
        if variant and variant not in ("wo_soft_head", "wo_forecast_head", "wo_joint_train", None):
            return build_pars_ablation(str(variant), **forge_kw)
        return build_pars(**forge_kw)

    misnet_kw = dict(
        in_channels=cin, seq_len=seq_len, pred_len=pred_len, n_soft=n_soft, n_forecast=n_forecast,
        use_mask=bool(args.use_mask), use_sla=bool(args.use_sla), d_model=d_model, dropout=dropout,
    )
    variant = args.ablation or args.variant
    model_id = canonicalize_model_name(args.model)
    if model_id == "misnet_hfd_chl_tune":
        return build_misnet_hfd_chl_tune(**misnet_kw)
    if model_id == "misnet_hfd_chl_hs":
        return build_misnet_hfd_chl_hs(**misnet_kw)
    if model_id == "misnet_hfd_chl_decay":
        return build_misnet_hfd_chl_decay(**misnet_kw)
    if model_id == "misnet_hfd_chl_ams":
        return build_misnet_hfd_chl_ams(**misnet_kw)
    if model_id == "misnet_hfd_chl":
        return build_misnet_hfd_chl(**misnet_kw)
    if model_id == "misnet_hfd":
        return build_misnet_hfd(**misnet_kw)
    if model_id == "misnet_rev_l3":
        return build_misnet_rev_l3(**misnet_kw)
    if model_id == "misnet_rev_l2":
        return build_misnet_rev_l2(**misnet_kw)
    if model_id == "misnet_forecast_only":
        if variant in ("revision_l2", "revision_l3"):
            return build_misnet_ablation(variant, **misnet_kw)
        return build_misnet_forecast_only(**misnet_kw)
    if variant and variant not in ("wo_soft_head", "wo_forecast_head", "wo_joint_train") and model_id.startswith("misnet"):
        return build_misnet_ablation(variant, **misnet_kw)
    if model_id == "misnet_lite":
        return build_misnet_lite(**misnet_kw)
    if model_id == "misnet_full" or (model_id == "misnet" and args.use_exo):
        misnet_kw["use_exo"] = True
        misnet_kw["n_exo"] = 2
        return build_misnet_full(**misnet_kw)
    if model_id.startswith("misnet"):
        return build_misnet(**misnet_kw)
    return build_misnet(**misnet_kw)


def run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None, loss_weights: LIMONLossWeights,
    tasks: tuple[str, ...], train: bool, masked_loss: bool,
    struct_outage_prob: float = 0.0,
) -> dict[str, float]:
    model.train(train)
    totals = {"total": 0.0, "soft": 0.0, "forecast": 0.0, "moe_loss": 0.0}
    w = LIMONLossWeights(
        soft=loss_weights.soft if "soft" in tasks else 0.0,
        forecast=loss_weights.forecast if "forecast" in tasks else 0.0,
        moe_balance=loss_weights.moe_balance, risk=0.0,
    )
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            if train and struct_outage_prob > 0 and type(model).__name__ in ("PARS", "LakeForge"):
                x, mask = apply_structured_outage_batch(x, mask, prob=struct_outage_prob)
            exo = batch.get("exo")
            if exo is not None:
                exo = exo.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)

            if isinstance(model, TORCH_BASELINE_TYPES):
                out = model(x, mask)
                losses = baseline_lstm_loss(
                    out, batch["soft_y"].to(device), batch["forecast_y"].to(device),
                    batch["soft_y_mask"].to(device) if masked_loss else None,
                    batch["forecast_y_mask"].to(device) if masked_loss else None,
                    w_soft=w.soft, w_forecast=w.forecast,
                )
            else:
                soft_m = batch["soft_y_mask"].to(device) if masked_loss else None
                if type(model).__name__ in ("PARS", "LakeForge"):
                    out = model(x, mask, exo, soft_y_mask=soft_m)
                    losses = pars_loss(
                        out, batch["soft_y"].to(device), batch["forecast_y"].to(device),
                        soft_y_mask=soft_m,
                        forecast_y_mask=batch["forecast_y_mask"].to(device) if masked_loss else None,
                        weights=w,
                        cfg=getattr(model, "cfg", None),
                    )
                else:
                    soft_y = batch["soft_y"].to(device)
                    out = model(x, mask, exo, soft_y=soft_y, soft_y_mask=soft_m)
                    loss_fn = misnet_hfd_loss if type(model).__name__ in ("MISNetHFD", "LIMONHFD") else misnet_loss
                    losses = loss_fn(
                        out, soft_y, batch["forecast_y"].to(device),
                        soft_y_mask=soft_m,
                        forecast_y_mask=batch["forecast_y_mask"].to(device) if masked_loss else None,
                        weights=w,
                        cfg=getattr(model, "cfg", None),
                    )
            if train:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for k in totals:
                if k in losses:
                    totals[k] += float(losses[k].detach().cpu())
            n_batches += 1
    return {k: (v / max(n_batches, 1)) for k, v in totals.items()}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def train_torch(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    try:
        _train_torch_impl(args, meta)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            raise SystemExit(
                f"[train] {args.model} GPU error: {exc}\n"
                "Retry with: --batch-size 16 --device cuda  (or --device cpu)"
            ) from exc
        raise
    except Exception as exc:
        raise SystemExit(f"[train] {args.model} torch training failed: {exc}") from exc


def _train_torch_impl(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    device = pick_device(args.device)
    print(f"[train] device={device}")
    paths = split_npz_paths(args.proc_dir)
    train_idx = subsample_indices(len(NPZDataset(paths["train"])), args.train_frac, args.seed)
    train_loader = DataLoader(NPZDataset(paths["train"], train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NPZDataset(paths["val"]), batch_size=args.batch_size, shuffle=False)

    model = build_torch_model(args, meta).to(device)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[train] init from {args.init_checkpoint} (missing={len(missing)} unexpected={len(unexpected)})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    kv = parse_kv_weights(args.loss_weights)
    loss_weights = LIMONLossWeights(soft=kv.get("soft", 1.0), forecast=kv.get("forecast", 1.0), moe_balance=kv.get("moe_balance", 0.01))
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    best_val, best_epoch, stale = float("inf"), -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            model, train_loader, device, optimizer, loss_weights, tasks, True, bool(args.masked_loss),
            struct_outage_prob=float(getattr(args, "struct_outage_prob", 0.0)),
        )
        va = run_epoch(
            model, val_loader, device, None, loss_weights, tasks, False, bool(args.masked_loss),
            struct_outage_prob=0.0,
        )
        history.append({"epoch": epoch, "train": tr, "val": va, "sec": time.time() - t0})
        print(f"[train] epoch {epoch:03d} | train={tr['total']:.4f} val={va['total']:.4f}")
        if va["total"] < best_val:
            best_val, best_epoch, stale = va["total"], epoch, 0
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "meta": meta,
                "args": _json_safe(vars(args)),
                "limon_cfg": _json_safe(vars(model.cfg)) if hasattr(model, "cfg") else None,
                "pars_cfg": _json_safe(vars(model.cfg)) if hasattr(model, "cfg") else None,
                "best_val": best_val,
                "epoch": epoch,
            }, args.out_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[train] early stop @ epoch {epoch} (best={best_val:.4f} ep {best_epoch})")
                break
    with open(args.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(history), f, indent=2)
    if best_epoch < 0 or not (args.out_dir / "best.pt").exists():
        raise SystemExit(
            f"[train] {args.model}: no checkpoint saved (best_val={best_val}). "
            "Check for NaN loss or empty val loader."
        )
    print(f"[train] saved -> {args.out_dir / 'best.pt'} (best_val={best_val:.4f} ep {best_epoch})")


def train_sklearn(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    try:
        import sklearn
    except ImportError as exc:
        raise SystemExit("scikit-learn required for pls/xgboost: pip install scikit-learn") from exc
    if args.model == "xgboost":
        try:
            import xgboost
        except ImportError as exc:
            raise SystemExit("xgboost required: pip install xgboost") from exc

    paths = split_npz_paths(args.proc_dir)
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"Missing {name}.npz at {path}. Run data_process.py first.")

    idx = subsample_indices(len(NPZDataset(paths["train"])), args.train_frac, args.seed)
    train = NPZDataset(paths["train"], idx)
    data, idx = train._store, train.indices
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    try:
        bundle = train_sklearn_baseline(
            args.model, data["x"][idx], data["mask"][idx], data["soft_y"][idx], data["soft_y_mask"][idx],
            data["forecast_y"][idx], data["forecast_y_mask"][idx], tasks=tasks, seed=args.seed,
            n_components=getattr(args, "n_components", 10), max_depth=getattr(args, "max_depth", 6),
            xgb_lr=getattr(args, "xgb_lr", 0.05), n_estimators=getattr(args, "n_estimators", 300),
        )
    except Exception as exc:
        raise SystemExit(f"[train] {args.model} sklearn training failed: {exc}") from exc
    tmp = args.out_dir / "sklearn_model.pkl.tmp"
    final = args.out_dir / "sklearn_model.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    tmp.replace(final)
    print(f"[train] saved -> {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MISNet / baselines")
    parser.add_argument("--exp", default="joint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--proc-dir", type=Path, required=True)
    parser.add_argument("--hyperparams", type=Path, default=None)
    parser.add_argument("--tasks", default="soft,forecast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-weights", default="soft=1.0,forecast=1.0,moe_balance=0.01")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--use-mask", type=int, default=1)
    parser.add_argument("--use-sla", type=int, default=1)
    parser.add_argument("--use-exo", type=int, default=0)
    parser.add_argument("--masked-loss", type=int, default=1)
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--decoder", default="direct")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--factor", type=int, default=5, help="Informer ProbSparse sample factor")
    parser.add_argument(
        "--struct-outage-prob",
        type=float,
        default=0.0,
        help="PARS train-time contiguous channel outage probability (0=off; try 0.05–0.1)",
    )
    parser.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    args = parser.parse_args()

    if args.model in STUB_MODELS:
        raise SystemExit(f"Model '{args.model}' not implemented; see Table S4.")
    if args.decoder == "scroll":
        raise SystemExit("Scroll decoder not implemented.")

    hp = load_hyperparams(args.hyperparams, args.model)
    if not hp and args.model in (
        "limon_forecast_only",
        "misnet_forecast_only",
        "limon_rev_l2",
        "misnet_rev_l2",
        "limon_rev_l3",
        "misnet_rev_l3",
        "limon_hfd",
        "misnet_hfd",
        "limon_hfd_chl",
        "misnet_hfd_chl",
        "limon_hfd_chl_ams",
        "misnet_hfd_chl_ams",
        "limon_hfd_chl_decay",
        "misnet_hfd_chl_decay",
        "limon_hfd_chl_hs",
        "misnet_hfd_chl_hs",
        "limon_hfd_chl_tune",
        "misnet_hfd_chl_tune",
        "limon_lite",
        "misnet_lite",
    ):
        hp = load_hyperparams(args.hyperparams, "misnet") or load_hyperparams(args.hyperparams, "limon")
    if not hp and args.model in ("pars", "pars_forecast_only", "lake_forge", "lake_forge_forecast_only"):
        hp = load_hyperparams(args.hyperparams, "pars") or load_hyperparams(args.hyperparams, "lake_forge")
    apply_hyperparams_to_train_args(args, hp)
    variant = args.variant or args.ablation
    args.tasks = resolve_tasks(variant, args.tasks)
    if args.model in (
        "pars_forecast_only",
        "lake_forge_forecast_only",
        "limon_forecast_only",
        "misnet_forecast_only",
        "limon_rev_l2",
        "misnet_rev_l2",
        "limon_rev_l3",
        "misnet_rev_l3",
        "limon_hfd",
        "misnet_hfd",
        "limon_hfd_chl",
        "misnet_hfd_chl",
        "limon_hfd_chl_ams",
        "misnet_hfd_chl_ams",
        "limon_hfd_chl_decay",
        "misnet_hfd_chl_decay",
        "limon_hfd_chl_hs",
        "misnet_hfd_chl_hs",
        "limon_hfd_chl_tune",
        "misnet_hfd_chl_tune",
    ):
        args.tasks = "forecast"
    if not (str(args.model).startswith("pars") or str(args.model).startswith("lake_forge")):
        args.struct_outage_prob = 0.0

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.proc_dir)
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "exp": args.exp, "model": args.model, "proc_dir": str(args.proc_dir),
            "tasks": args.tasks, "seed": args.seed, "hyperparams": hp,
            "soft_vars": list(SOFT_VARS), "forecast_vars": list(FORECAST_VARS),
            "ablation": variant,
        }, f, indent=2, default=str)

    if args.model in NO_TRAIN_MODELS:
        print(f"[train] {args.model}: evaluate-only")
        return

    artifact: Path | None = None
    if args.model in SKLEARN_MODELS:
        artifact = args.out_dir / "sklearn_model.pkl"
    elif args.model in TORCH_MODELS or args.model.startswith(("limon", "misnet", "pars", "lake_forge")):
        artifact = args.out_dir / "best.pt"
    if artifact and artifact.exists() and artifact.stat().st_size > 0 and not args.force:
        print(f"[train] skip existing artifact -> {artifact}")
        return

    if args.model in SKLEARN_MODELS:
        train_sklearn(args, meta)
        return
    if args.model in TORCH_MODELS or args.model.startswith(("limon", "misnet", "pars", "lake_forge")):
        train_torch(args, meta)
        return
    raise SystemExit(f"Unknown model: {args.model}")


if __name__ == "__main__":
    try:
        main()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)
    except SystemExit as exc:
        if exc.code in (0, None):
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(0)
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from None
