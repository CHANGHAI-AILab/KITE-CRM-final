#!/usr/bin/env python3
"""3D arm — export CASE-LEVEL predictions for the LOCKED 3D FPN (LE) Task 2 classifier
(benign/malignant), using the original locked checkpoint (sensitivity_paths.LOCKED_TASK2_CKPT).

This is the faithful Task 2 pipeline. Emits predictions_long rows for development
(Training + Validation) with per-case patient_id, probability, predicted_class at the
LOCKED operating threshold, and a leakage flag marking cases that were in the original
3D-training feature pool. The output feeds paired_heldout_neibu.py as the 3D arm.

NOTE ON DEPENDENCIES: the locked 3D FPN model definition and its Task-2 item builder
live in the released classifier package (module `train_fpn` below: LE() model, build_items(),
norm()). Point PYTHONPATH at that package, or adapt the two imports, before running. The
decoder-feature .npz inputs and the locked checkpoint are NOT distributed with this repo
(data governance; see weights_manifest.csv). This script neither trains nor modifies the
locked model — it only scores it. Do not overwrite or relabel the locked checkpoint.
"""
import os, glob
import numpy as np, torch, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

import sensitivity_paths as P
# Locked 3D FPN (LE) model + Task-2 split loader from the released classifier package.
import train_fpn as T  # provides T.LE(), T.build_items(task_id), T.norm()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Locked operating threshold for the 3D Task 2 classifier (see paper_configs weights_manifest.csv).
THR = float(os.environ.get("KITE_TASK2_LOCKED_THR", "0.586442888"))
CKPT = P.LOCKED_TASK2_CKPT
TRAINPOOL = P.TRAINPOOL_FEAT_DIR   # original 3D-training feature pool (leaked cases live here)


def pid_from_path(p):
    return T.norm(os.path.basename(p).split("_")[0])


def trainpool_pids():
    s = set()
    for f in glob.glob(TRAINPOOL + "/*.npz"):
        s.add(T.norm(os.path.basename(f).split("_")[0]))
    return s


def infer(net, items):
    net.eval()
    out = []
    with torch.no_grad():
        for p, y in items:
            d = np.load(p)
            f0 = torch.from_numpy(d["feat1"].astype(np.float32)).to(device)
            f1 = torch.from_numpy(d["feat2"].astype(np.float32)).to(device)
            f2 = torch.from_numpy(d["feat3"].astype(np.float32)).to(device)
            prob = torch.softmax(net(f0, f1, f2), 1)[0, 1].item()
            out.append((pid_from_path(p), int(y), prob, p))
    return out


def main():
    tr, va = T.build_items(2)
    leaked = trainpool_pids()
    net = T.LE().to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))

    rows = []
    for split_name, items, cohort in [("development_train", tr, "Training"),
                                       ("development_val", va, "Validation")]:
        for pid, y, prob, path in infer(net, items):
            src = "orig_train_pool" if pid in leaked else "held_out"
            rows.append({
                "patient_id": pid,
                "lesion_id": "",
                "cohort": cohort,
                "split": "development",
                "reference_standard": "pathology",
                "label_malignant": y,
                "bosniak_category": "",
                "model_name": "Locked_3D_FPN_LE_Task2",
                "seed": "locked_ckpt",
                "probability": prob,
                "threshold": THR,
                "predicted_class": 1 if prob >= THR else 0,
                "mask_source": "nnUNet_predicted_lesion_mask",
                "maximal_slice_index": "",
                "exclusion_reason": "",
                "leakage_flag": src,
            })

    out = pd.DataFrame(rows)
    outp = os.path.join(P.WORK_DIR, "predictions_long_3D_task2.csv")
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    print("saved predictions_long_3D_task2.csv rows=", len(out))

    # sanity metrics
    for cohort in ["Validation"]:
        s = out[out["cohort"] == cohort]
        auc = roc_auc_score(s["label_malignant"], s["probability"])
        acc = accuracy_score(s["label_malignant"], s["predicted_class"])
        br = brier_score_loss(s["label_malignant"], s["probability"])
        print(f"{cohort}: n={len(s)} pos={s['label_malignant'].sum()} "
              f"AUROC={auc:.4f} ACC@{THR}={acc:.4f} Brier={br:.4f}")
        for lay in ["orig_train_pool", "held_out"]:
            ss = s[s["leakage_flag"] == lay]
            if len(ss) and ss["label_malignant"].nunique() > 1:
                print(f"    {lay}: n={len(ss)} AUROC={roc_auc_score(ss['label_malignant'], ss['probability']):.4f}")


if __name__ == "__main__":
    main()
