# 2D Architecture-Sensitivity — Modeling Scripts

This folder contains the **modeling chain** for the 2D arm of the post hoc
architecture-sensitivity analysis (3D vs 2D): how the 2D DenseNet-121 sensitivity
models and their inputs were built, trained, persisted (weights + SHA-256), and
applied.

> **The 2D models here are SENSITIVITY / ABLATION models only. They were never deployed,
> never used to set any locked operating threshold, and are named distinctly from the
> deployed models. Only the 3D models are part of the deployed KITE-CRM pipeline.**

## Location in the final release

The final release contains this directory at
`analysis/2D_sensitivity_modeling/`. The public GitHub tree must be checked
against this exact directory at the `v1.0-eclinm` release tag. This directory is
part of the published public tree; local inclusion is not a substitute for the
tagged release contents.

## Final analysis scope

The final analysis scope documented by this release is:

- Task 1 final validation analysis: `n=469`.
- Task 2 final validation analysis: `n=235`.

The public repository does not include patient-level result workbooks or
case-level outputs. The final Task 2 analysis scope is the full validation cohort
(`n=235`).

The present folder documents the upstream *modeling* steps and provides an optional
end-to-end re-inference path (below). It does not by itself reproduce patient-level
results because the public release excludes the patient images, labels, training-pool
features, locked checkpoints, and result workbooks. A fresh run from separately supplied
inputs or checkpoints may produce different probabilities or cohort coverage; any such
difference must be attributed to the input data, weights, or analysis set.
## Pipeline (run order)

| Step | Script | Purpose |
|---|---|---|
| 0 | `sensitivity_paths.py` | Central path config — all data/checkpoint locations via environment variables (no hard-coded paths). |
| 0 | `build_items_task2.py` | Thin adapter exposing the **official Task 2 split + labels** (wire to the released classifier loader or a label table — see its docstring). |
| 1 | `build_2D_maxsection_task2.py` | Task 2 data prep: extract the max-lesion axial section (soft-tissue window, bbox-center crop, 224) → `cache_2d/task2/<split>/<label>/*.npy`. |
| 2 | `train_2D_densenet_task2.py` | Train the non-deployed 2D Task 2 sensitivity classifier (seeds 0/1/2), calculate its implemented ROC/Youden validation cut-off, and export per-seed predictions. This sensitivity-only cut-off is not the deployed Task 2 threshold and does not explain how `0.586442888` was selected. |
| 2′ | `train_2d_sensitivity_persist.py` | Compliance rerun of step 2 that **persists weights + SHA-256 + per-epoch logs** and runs a paired-subset consistency check. |
| 3 | `train_2D_densenet_task1.py` | Train the 2D **Task 1** (normal vs abnormal) arm using a **label-agnostic central-slice** rule (normals have no lesion mask). |
| 4 | `infer_2d_task2_anycohort.py` / `infer_2d_task1_anycohort.py` | Score the frozen 2D ensembles on any cohort. |
| 5 | `export_3D_task2_predictions.py` | (Optional) re-infer the **locked 3D** Task 2 model to case-level predictions. Requires the released classifier package (`train_fpn`) and the locked checkpoint. |
| 6 | `paired_heldout_neibu.py` / `paired_task1_3d_vs_2d.py` | Optional 3D-vs-2D comparison utilities from re-inferred probabilities; these scripts do not redefine the final Task 2 cohort. |

## Configuration

All site paths are read from environment variables — nothing is hard-coded. Set them
before running (placeholders intentionally do not resolve):

```bash
export KITE_DATA_ROOT=/path/to/data_root
export KITE_WORK_DIR=/path/to/work_dir          # caches, checkpoints, outputs
export KITE_LESION_CT_DIR=...                     # Task 2 CT volumes
export KITE_LESION_MASK_DIR=...                   # Task 2 predicted lesion masks (lesion == label 2)
export KITE_HELDOUT_CT_DIR=...                    # paired-subset CT (legacy variable name)
export KITE_HELDOUT_MASK_DIR=...                  # paired-subset masks (legacy variable name)
export KITE_TRAINPOOL_FEAT_DIR=...               # 3D decoder-feature pool (leakage flag)
export KITE_LOCKED_TASK1_CKPT=...                # locked 3D Task 1 checkpoint (not released)
export KITE_LOCKED_TASK2_CKPT=...                # locked 3D Task 2 checkpoint (not released)
export KITE_TASK1_LABELS_XLSX=...   KITE_TASK2_LABELS_XLSX=...
export KITE_TASK1_SPLIT_CSV=...                  # Task 1 clip split (path,label,split)
export KITE_PRED3D_TASK2_CSV=...                 # locked 3D Task 2 case-level predictions
```

## Data & reproducibility notes

- **Patient CT volumes, lesion masks, checkpoints, and label tables are NOT distributed**
  with this repository — they are restricted by the participating centres' data-governance
  agreements (see the manuscript Data and Code Availability statement). The scripts document
  the exact procedure; they do not ship the inputs.
- **Cohort scope:** use the full Task 2 validation analysis cohort (`n=235`) when the
  restricted inputs are supplied. The optional comparison utilities do not establish an
  independent or complete external validation cohort.
- **Adapter boundaries:** `build_items_task2.py` intentionally raises `NotImplementedError`
  until wired to the official Task 2 split/label loader. The optional 3D re-inference
  adapter requires a compatible external `train_fpn` module. Patient data, labels,
  feature pools, and locked checkpoints are not distributed.
- **Environment:** Python 3.9.19, PyTorch 2.8.0+cu128, torchvision 0.23.0, CUDA 12.8.
