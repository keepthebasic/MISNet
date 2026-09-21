# MISNet

Code for missingness-aware multi-horizon forecasting of chlorophyll-a and dissolved oxygen from incomplete lake sensor streams.

## Contents

- `models/` — MISNet network definition
- `scripts/` — data preparation, training, evaluation, ablations, leave-one-lake-out, and figure helpers
- `configs/` — locked hyperparameters used in the reported experiments

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
export MKL_THREADING_LAYER=GNU
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Data

Raw LakeBeD tables are not redistributed in this repository. Download the public release, then build 4 h windows with the scripts below.

| Item | Detail |
|------|--------|
| Dataset | LakeBeD-US: Computer Science Edition (LakeBeD-US-CSE) |
| DOI | https://doi.org/10.57967/hf/3771 |
| Download | https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE |
| Related paper | https://doi.org/10.5194/essd-17-3141-2025 |

This work uses the CSE HighFrequency and LowFrequency products, resampled to 4 h windows (`seq_len=168`, 6-step / 24 h outlook). The deep case is Beaverdam Reservoir (BVR). Leave-one-lake-out uses ME, BVR, FCR, TR, and SP.

```bash
python scripts/lakebed_to_csv.py --help
python scripts/data_process.py --help
```

Processed windows should contain `train.npz`, `val.npz`, `test.npz`, `meta.json`, and `scalers.json`. Follow the LakeBeD-US license and citation terms when redistributing derived products.

## Reproduce

Train and evaluate MISNet on BVR with the locked configuration (seeds 0–4 in the reported panel):

```bash
python scripts/train.py --model misnet --proc-dir processed/BVR_4h --out-dir results/misnet_seed0 --seed 0 --hyperparams configs/hyperparams_locked.json
python scripts/evaluate.py --proc-dir processed/BVR_4h --results-dir results/misnet_seed0
```

The same entry points accept `lstm`, `mask_lstm`, `grud`, and `persistence`.

Random masks, structured outages, block ablations, and leave-one-lake-out:

```bash
python scripts/run_robustness.py --help
python scripts/run_ablation.py --help
python scripts/run_lolo.py --help
python scripts/interpret.py --help
```

## Citation

If you use this code, please cite the accompanying Journal of Hydrology manuscript on MISNet. Please also cite LakeBeD-US-CSE (DOI above) when using the data.
