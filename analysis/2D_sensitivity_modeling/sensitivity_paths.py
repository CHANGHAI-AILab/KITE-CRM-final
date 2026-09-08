"""Path configuration for the 2D architecture-sensitivity modeling scripts.

All dataset and checkpoint locations are read from environment variables so that no
absolute path or site-specific directory name is hard-coded in the released code.
Set these to your local paths before running; the placeholder defaults intentionally
do not resolve.

    export KITE_DATA_ROOT=/path/to/data_root       # dataset root (optional convenience)
    export KITE_WORK_DIR=/path/to/work_dir         # outputs, caches, checkpoints
    export KITE_LESION_CT_DIR=...                   # Task2 CT volumes
    export KITE_LESION_MASK_DIR=...                 # Task2 predicted lesion masks (lesion == label 2)
    export KITE_HELDOUT_CT_DIR=...                  # paired-subset CT (legacy variable name)
    export KITE_HELDOUT_MASK_DIR=...                # paired-subset masks (legacy variable name)
    export KITE_TRAINPOOL_FEAT_DIR=...             # 3D decoder-feature pool (leakage flag)
    export KITE_LOCKED_TASK1_CKPT=...              # locked 3D Task1 checkpoint (not released)
    export KITE_LOCKED_TASK2_CKPT=...              # locked 3D Task2 checkpoint (not released)
    export KITE_TASK1_LABELS_XLSX=...              # Task1 case label table (id / dataset_name / gold)
    export KITE_TASK2_LABELS_XLSX=...              # Task2 case label table (id / dataset_name / gold)
    export KITE_TASK1_SPLIT_CSV=...                # Task1 clip split table (path,label,split)
    export KITE_PRED3D_TASK2_CSV=...              # locked 3D Task2 case-level predictions (predictions_long)

Patient CT volumes and masks are restricted by the participating centres' data-governance
agreements and are not distributed with this repository (see the manuscript Data and Code
Availability statement).
"""
import os


def _env(name, default):
    return os.environ.get(name, default)


DATA_ROOT = _env("KITE_DATA_ROOT", "<DATA_ROOT>")
WORK_DIR = _env("KITE_WORK_DIR", ".")

# Task 2 (lesion) inputs — CT volumes and predicted lesion masks (lesion voxels == label 2)
LESION_CT_DIR = _env("KITE_LESION_CT_DIR", "<lesion_ct_dir>")
LESION_MASK_DIR = _env("KITE_LESION_MASK_DIR", "<lesion_mask_dir>")

# Complete-pair subset inputs; environment-variable names are retained for compatibility.
HELDOUT_CT_DIR = _env("KITE_HELDOUT_CT_DIR", "<heldout_val_ct_dir>")
HELDOUT_MASK_DIR = _env("KITE_HELDOUT_MASK_DIR", "<heldout_val_mask_dir>")

# 3D decoder-feature pool used only to flag original-training-pool leakage
TRAINPOOL_FEAT_DIR = _env("KITE_TRAINPOOL_FEAT_DIR", "<orig_train_feature_pool>")

# Locked 3D classifier checkpoints (NOT released here; see paper_configs/.../weights_manifest.csv).
# These are the deployed, threshold-locked models — never overwrite or relabel them.
LOCKED_TASK1_CKPT = _env("KITE_LOCKED_TASK1_CKPT", "<locked_task1_3d_checkpoint>")
LOCKED_TASK2_CKPT = _env("KITE_LOCKED_TASK2_CKPT", "<locked_task2_3d_checkpoint>")

# Case label tables (columns: id/编号, dataset_name, per-task gold-standard label)
TASK1_LABELS_XLSX = _env("KITE_TASK1_LABELS_XLSX", "<task1_labels_xlsx>")
TASK2_LABELS_XLSX = _env("KITE_TASK2_LABELS_XLSX", "<task2_labels_xlsx>")
# Backwards-compatible alias (Task 2 by default)
LABELS_XLSX = _env("KITE_LABELS_XLSX", TASK2_LABELS_XLSX)

# Task 1 clip split table (columns: path, label, split in {Training, Validation})
TASK1_SPLIT_CSV = _env("KITE_TASK1_SPLIT_CSV", "<task1_split_csv>")

# Locked 3D Task 2 case-level predictions (predictions_long: patient_id, cohort,
# label_malignant, probability, leakage_flag) — used as the 3D arm and pair-definition source
PRED3D_TASK2_CSV = _env("KITE_PRED3D_TASK2_CSV", "<predictions_long_3d_task2_csv>")

# Derived output locations under the work dir
CACHE_2D_TASK2 = os.path.join(WORK_DIR, "cache_2d", "task2")
CKDIR_TASK2 = os.path.join(WORK_DIR, "ckpt_2d_sensitivity")
CKDIR_TASK1 = os.path.join(WORK_DIR, "ckpt_2d_task1_sensitivity")
