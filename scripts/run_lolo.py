from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_utils import parse_seeds, run_subprocess
from scripts.npz_dataset import build_lolo_proc_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E4b")
    parser.add_argument("--lakes", default="ME,BVR,FCR,TR,SP")
    parser.add_argument("--proc-root", type=Path, required=True)
    parser.add_argument("--model", default="limon")
    parser.add_argument("--baselines", default="persistence,pls,xgboost,lstm")
    parser.add_argument("--tasks", default="soft,forecast")
    parser.add_argument("--scaler-mode", default="target_train_only")
    parser.add_argument("--hyperparams", type=Path, default=None)
    parser.add_argument("--denormalize", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even if best.pt / sklearn_model.pkl already exists",
    )
    args = parser.parse_args()

    lakes = [x.strip() for x in args.lakes.split(",") if x.strip()]
    seeds = parse_seeds(args.seeds)
    models = [args.model] + [m.strip() for m in args.baselines.split(",") if m.strip()]
    all_results: dict = {}

    for test_lake in lakes:
        train_lakes = [lk for lk in lakes if lk != test_lake]
        fold_out = args.out_dir / f"test_{test_lake}"
        fold_out.mkdir(parents=True, exist_ok=True)
        proc_dir = build_lolo_proc_dir(
            args.proc_root, train_lakes, test_lake,
            fold_out / "_proc", args.scaler_mode,
        )

        for model in models:
            for seed in seeds:
                run_dir = fold_out / f"{model}_seed{seed}"
                if model != "persistence":
                    cmd = [
                        sys.executable, str(ROOT / "scripts" / "train.py"),
                        "--exp", "E4b", "--model", model,
                        "--proc-dir", str(proc_dir),
                        "--tasks", args.tasks, "--seed", str(seed),
                        "--epochs", str(args.epochs), "--patience", str(args.patience),
                        "--out-dir", str(run_dir), "--device", args.device,
                    ]
                    if args.hyperparams:
                        cmd += ["--hyperparams", str(args.hyperparams)]
                    if args.force:
                        cmd.append("--force")
                    run_subprocess(cmd)
                ev = [
                    sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                    "--exp", "E2", "--model", model,
                    "--results-dir", str(run_dir), "--proc-dir", str(proc_dir),
                    "--denormalize", str(args.denormalize),
                    "--targets", "do,chla", "--horizons", "6",
                    "--device", args.device,
                    "--out", str(run_dir / "E2_test.json"),
                ]
                run_subprocess(ev)

        protocol = json.loads((proc_dir / "lolo_protocol.json").read_text(encoding="utf-8"))
        all_results[test_lake] = {
            "train_lakes": train_lakes,
            "test_lake": test_lake,
            "proc_dir": str(proc_dir),
            "protocol": protocol,
        }

    (args.out_dir / "lolo_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[run_lolo] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
