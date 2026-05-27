# README Completion Notes

Last updated: 2026-05-25

This file is a handoff note for finishing `README.md` later. It records what has
actually been run, what changed in code, and what should still be verified before
the README is finalized.

## Current Project State

### Completed

- `project/src/debug_utils.py` exists.
- Raw datasets are present:
  - `project/data/raw/cic-ids2017/` with 8 CSV files.
  - `project/data/raw/cic-ids2018/Friday-02-03-2018.csv`.
  - `project/data/raw/ctu-13/scenario9_raw.csv`.
- Preprocessing has completed for all 3 datasets.
- `project/data/processed/cicids2017/selected_features.json` was generated from
  CIC-IDS2017 and reused by CIC-IDS2018 and CTU-13.

### Not Yet Completed

- No baseline model training has been completed in the current run.
- No SMOTE, GAN, or WCGAN-GP training/evaluation has been completed in the
  current run.
- No final `artifacts/models_*`, `artifacts/results_*`, or visualization outputs
  have been generated yet.
- README still needs final result tables after training/evaluation.

## Preprocessing Results

All commands were run from:

```powershell
cd C:\Users\amg85\Desktop\2026-1_AI\project\src
```

### CIC-IDS2017

Command:

```powershell
python preprocess_cicids2017.py
```

Result:

- Raw merged shape before cleaning: `(3119345, 83)`.
- Cleaned shape: `(2830743, 83)`.
- Label distribution:
  - Normal/non-bot: `2828777`
  - Botnet: `1966`
- Trainval shape after feature selection: `(2264594, 32)`.
- Test shape after feature selection: `(566149, 32)`.
- Sequence trainval shape: `(2264594, 32, 1)`.
- Generated:
  - `project/data/processed/cicids2017/flat/*.npy`
  - `project/data/processed/cicids2017/seq/*.npy`
  - `project/data/processed/cicids2017/seq/scaler_flow.pkl`
  - `project/data/processed/cicids2017/selected_features.json`

### CIC-IDS2018

Command:

```powershell
python preprocess_cicids2018.py
```

Result:

- Raw loaded shape: `(6311371, 78)`.
- Label distribution:
  - `BENIGN`: `6168188`
  - `Botnet Ares`: `142921`
  - `Botnet Ares - Attempted`: `262`
- Filtered bot count: `143183`.
- Trainval shape after feature selection: `(5049096, 32)`.
- Test shape after feature selection: `(1262275, 32)`.
- Sequence trainval shape: `(5049096, 32, 1)`.
- Generated:
  - `project/data/processed/cicids2018/flat/*.npy`
  - `project/data/processed/cicids2018/seq/*.npy`
  - `project/data/processed/cicids2018/aligner.pkl`
  - `project/data/processed/cicids2018/meta.json`

### CTU-13

Command:

```powershell
python preprocess_ctu13.py
```

Result:

- Raw loaded shape: `(2818416, 82)`.
- Column mappings applied: `79`.
- Label distribution:
  - Normal: `2752138`
  - Botnet: `66278`
- Botnet ratio printed by script: `0.0235`.
- Trainval shape after feature selection: `(2254732, 32)`.
- Test shape after feature selection: `(563684, 32)`.
- Sequence trainval shape: `(2254732, 32, 1)`.
- Generated:
  - `project/data/processed/ctu13/flat/*.npy`
  - `project/data/processed/ctu13/seq/*.npy`
  - `project/data/processed/ctu13/aligner.pkl`
  - `project/data/processed/ctu13/meta.json`

## Code Changes Made During Preprocessing

These changes were required because the full datasets exceeded available memory
when loaded or transformed with unnecessary copies.

### `project/src/preprocess_cicids2017.py`

- `load_all_csv()` now keeps only required base columns plus `ML_FEATURES`.
- Numeric ML features are converted to `float32` immediately after each CSV is
  loaded.
- `basic_cleaning()` avoids a full DataFrame copy when the blank-label/source-IP
  mask does not remove any rows.

### `project/src/preprocess_cicids2018.py`

- `load_raw()` now applies column cleanup/mapping per chunk.
- `load_raw()` keeps only `Label` plus `ML_FEATURES` before concatenating chunks.
- Numeric ML features are converted to `float32` per chunk.
- `apply_scaler()` uses reduced-copy scaling:
  - `cic_scaler.copy = False`
  - `MinMaxScaler(copy=False)`
- DataFrames are deleted after extracting NumPy arrays to reduce peak memory.

## Environment Notes

The initial Python environment was missing preprocessing dependencies. The
following packages were installed into the current user's Python environment:

```powershell
python -m pip install --user --force-reinstall numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 joblib==1.4.2 tqdm==4.66.4 pyarrow==16.1.0
python -m pip install --user matplotlib==3.8.4 seaborn==0.13.2
```

Notes:

- The first global pip install attempt failed with a Windows permission error.
- User-site packages are now the working path for preprocessing.
- Training may still require checking/installing:
  - `xgboost`
  - `imbalanced-learn`
  - `torch`, `torchvision`, `torchaudio`

## Known Design Issues To Mention Or Fix

- GAN/WCGAN-GP target ratio mismatch:
  - `augment_gan.py` and `augment_wcgan_gp.py` use `TARGET_RATIO = 0.005`.
  - `augment_utils.py` uses `TARGET_RATIO = 0.1`.
  - README should either document this or the code should be aligned.
- CNN/GRU inputs are feature sequences, not real temporal sessions:
  - Input shape is `(n, 32, 1)`.
  - The 32 selected features are treated as 32 sequence steps.
  - README/report should avoid describing these as true flow-time sequences.
- Some preprocessing docstrings still mention 20 features, but the actual code
  uses `N_FEATURES = 32`.
- CTU-13 botnet ratio in README should be updated from the old estimate if the
  final README reports actual preprocessed data:
  - Current script output: `66278 / 2818416`, about `2.35%`.
- CIC-IDS2017 chi-square printed some selected features with `nan` scores. This
  should be reviewed before final claims about feature ranking are made.

## Next Commands

Baseline training:

```powershell
python train_rf.py       --dataset cicids2017
python train_xgb.py      --dataset cicids2017
python train_cnn_lstm.py --dataset cicids2017
python train_gru.py      --dataset cicids2017
python train_cnn_gru.py  --dataset cicids2017

python train_rf.py       --dataset cicids2018
python train_xgb.py      --dataset cicids2018
python train_cnn_lstm.py --dataset cicids2018
python train_gru.py      --dataset cicids2018
python train_cnn_gru.py  --dataset cicids2018

python train_rf.py       --dataset ctu13
python train_xgb.py      --dataset ctu13
python train_cnn_lstm.py --dataset ctu13
python train_gru.py      --dataset ctu13
python train_cnn_gru.py  --dataset ctu13
```

Baseline evaluation:

```powershell
python evaluate.py --dataset cicids2017
python evaluate.py --dataset cicids2018
python evaluate.py --dataset ctu13
```

Augmented runs:

```powershell
python train_rf.py --dataset cicids2017 --augment smote
python evaluate.py --dataset cicids2017 --augment smote

python augment_gan.py --dataset cicids2017
python train_rf.py --dataset cicids2017 --augment gan
python evaluate.py --dataset cicids2017 --augment gan

python augment_wcgan_gp.py --dataset cicids2017
python train_rf.py --dataset cicids2017 --augment wcgan_gp
python evaluate.py --dataset cicids2017 --augment wcgan_gp
```

Repeat the augmented command pattern for `cicids2018` and `ctu13`, and for all
five model scripts.

## README Finalization Checklist

- Replace garbled Korean text if needed and ensure the file is saved as UTF-8.
- Add a short "Current Reproducibility Status" section:
  - Raw data present.
  - Preprocessing completed.
  - Training/evaluation pending.
- Update dataset table with actual preprocessed counts and ratios.
- Add a preprocessing output table:
  - Dataset
  - Trainval shape
  - Test shape
  - Botnet count
  - Botnet ratio
  - Shared feature selection source
- Add a section explaining the memory-safe preprocessing changes.
- Add final baseline and augmentation result tables after model runs.
- Add notes about limitations:
  - extreme imbalance in CIC-IDS2017
  - feature-sequence input for recurrent models
  - GAN/WCGAN-GP target-ratio mismatch if not fixed
- Confirm dependency installation instructions, especially PyTorch CUDA version.

## Active Automation

- `project/src/run_paper_experiments.ps1` was added to continue the full
  paper experiment pipeline sequentially.
- Current queue process: PowerShell PID `9032`.
- Queue behavior:
  - waits for the active CIC-IDS2017 CNN-LSTM target-ratio-0 run, PID `25952`
  - skips completed artifacts when they already exist
  - reruns CIC-IDS2017 deep baselines when the existing k-fold mean F1 is below
    `0.2`, which avoids keeping the previous overcorrected results
  - runs remaining baselines
  - runs SMOTE experiments
  - trains GAN and WCGAN-GP generators, then runs augmented experiments
  - runs `evaluate.py`, `visualize.py`, and final result summarization
- Queue logs:
  - `artifacts/logs/paper_experiment_queue.log`
  - `artifacts/logs/paper_experiment_queue.csv`
- Final generated summary files:
  - `artifacts/paper_summary.md`
  - `artifacts/paper_summary.csv`
- The corrected CIC-IDS2017 CNN-LSTM run is already much healthier than the
  overcorrected run:
  - old mean F1 was about `0.0500`
  - current folds are around `0.47` to `0.51` F1 so far
- Code review update:
  - `train_xgb.py` duplicate `subsample_benign` block removed.
  - `train_xgb.py` now uses `device="cpu"` for the current CPU-only environment.
  - SMOTE/GAN/WCGAN-GP target-ratio constants are now aligned to `0.1`.
  - `run_paper_experiments.ps1` now forces `evaluate.py` jobs to regenerate
    holdout results, avoiding stale `eval_results.json` files after retraining.
