from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import parse_seeds, run_subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E6")
    parser.add_argument("--model", default="pars")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--proc-dir", type=Path, required=True)
    parser.add_argument("--hyperparams", type=Path, default=None)
    parser.add_argument("--tasks", default="soft,forecast")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
    variant = args.variant
    ablation = None
    tasks = args.tasks
    if variant == "wo_soft_head":
        tasks = "forecast"
    elif variant == "wo_forecast_head":
        tasks = "soft"
    elif variant == "wo_joint_train":
        tasks = "soft"
        ablation = None
    elif variant not in ("full", "default"):
        ablation = variant

    for seed in seeds:
        out = args.out_dir if len(seeds) == 1 else args.out_dir / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--exp", "joint", "--model", args.model,
            "--proc-dir", str(args.proc_dir),
            "--tasks", tasks, "--seed", str(seed),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--out-dir", str(out), "--device", args.device,
        ]
        if args.hyperparams:
            cmd += ["--hyperparams", str(args.hyperparams)]
        if ablation:
            cmd += ["--ablation", ablation, "--variant", ablation]
        run_subprocess(cmd)
        for exp in ("E1", "E2"):
            ev = [
                sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                "--exp", exp, "--model", args.model,
                "--results-dir", str(out), "--proc-dir", str(args.proc_dir),
                "--denormalize", "1",
                "--device", args.device,
                "--out", str(out / f"{exp}_metrics.json"),
            ]
            if exp == "E1":
                ev += ["--targets", "chla,tp,tn"]
            else:
                ev += ["--targets", "do,chla", "--horizons", "1,2,3,6"]
            run_subprocess(ev)
        if variant == "wo_joint_train":
            cmd2 = cmd.copy()
            cmd2[cmd2.index("--tasks") + 1] = "forecast"
            cmd2[cmd2.index("--out-dir") + 1] = str(out / "stage2_forecast")
            run_subprocess(cmd2)


if __name__ == "__main__":
    main()
