#!/usr/bin/env python3
"""Task 1 paired 3D-vs-2D on the SAME Validation clips.

3D = locked MONAI 3D DenseNet-121 (sensitivity_paths.LOCKED_TASK1_CKPT), scored here on
the validation clips. 2D = Task1_2D_DenseNet121_centralslice_sensitivity,
ensemble mean of the 3 persisted seeds (CKDIR_TASK1). Join by clip path. Reports
DeLong p + 2000-bootstrap ΔAUROC CI.

Report the paired ΔAUROC with its DeLong p and bootstrap CI exactly. A CI that crosses 0
means "no significant difference detected" — NOT equivalence or non-inferiority. The 2D
arm is a SENSITIVITY analysis; only the 3D arm is deployed.
"""
import os, csv, json
import numpy as np, SimpleITK as sitk, torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from monai.transforms import Resize
from torchvision.models import densenet121
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score

import sensitivity_paths as P

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE = Resize(spatial_size=(32, 128, 128))
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def resolve(p):
    for c in [p, os.path.join(P.DATA_ROOT, p), os.path.join(P.WORK_DIR, p)]:
        if os.path.exists(c):
            return c
    return None


def val_rows():
    rows = []
    for r in csv.DictReader(open(P.TASK1_SPLIT_CSV)):
        if r["split"] == "Validation":
            c = resolve(r["path"])
            if c:
                rows.append((c, int(r["label"])))
    return rows


def read3d(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32) / 255.0
    a = RESIZE(a[None])
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


def read2d(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32) / 255.0
    z = a.shape[0]; c = z // 2
    sl = a[max(0, c - 1):min(z, c + 2)].mean(0)
    sl = zoom(sl, (224 / sl.shape[0], 224 / sl.shape[1]), order=1)
    img = np.stack([sl, sl, sl], 0)
    img = (img - MEAN[:, None, None]) / STD[:, None, None]
    return torch.as_tensor(img.astype(np.float32))


def build2d():
    n = densenet121(weights=None)
    n.classifier = nn.Linear(n.classifier.in_features, 2)
    return n


def delong_p(y, p1, p2):
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N); out[J] = T
        return out

    def structural(preds, m):
        n = preds.shape[1] - m
        pos = preds[:, :m]; neg = preds[:, m:]
        k = preds.shape[0]
        tx = np.array([midrank(pos[r]) for r in range(k)])
        ty = np.array([midrank(neg[r]) for r in range(k)])
        tz = np.array([midrank(preds[r]) for r in range(k)])
        aucs = (tz[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:, :m] - tx) / n
        v10 = 1 - (tz[:, m:] - ty) / m
        sx = np.cov(v01); sy = np.cov(v10)
        s = sx / m + sy / n
        return aucs, s
    yb = np.asarray(y); idx = np.argsort(-yb)
    p1 = np.asarray(p1)[idx]; p2 = np.asarray(p2)[idx]; ys = yb[idx]
    m = int(ys.sum())
    preds = np.vstack([p1, p2])
    aucs, s = structural(preds, m)
    from scipy import stats
    l = np.array([[1, -1]])
    z2 = (l @ aucs) ** 2 / (l @ s @ l.T)
    p = float(stats.chi2.sf(z2[0, 0], 1))
    return aucs, p


def main():
    rows = val_rows()
    print(f"val cases: {len(rows)}", flush=True)

    # 3D inference (locked model)
    net3 = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEV)
    net3.load_state_dict(torch.load(P.LOCKED_TASK1_CKPT, map_location=DEV))
    net3.eval()
    p3 = []
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            x = torch.stack([read3d(p) for p, _ in b]).to(DEV)
            p3 += torch.softmax(net3(x), 1)[:, 1].cpu().tolist()
    print("3D done", flush=True)

    # 2D ensemble of 3 seeds
    p2_seeds = []
    for s in range(3):
        net2 = build2d().to(DEV)
        net2.load_state_dict(torch.load(f"{P.CKDIR_TASK1}/seed{s}.pt", map_location=DEV))
        net2.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(rows), 16):
                b = rows[i:i + 16]
                x = torch.stack([read2d(p) for p, _ in b]).to(DEV)
                ps += torch.softmax(net2(x), 1)[:, 1].cpu().tolist()
        p2_seeds.append(ps)
        print(f"2D seed{s} done", flush=True)
    p2 = np.mean(p2_seeds, 0)

    y = [l for _, l in rows]
    auc3 = roc_auc_score(y, p3); auc2 = roc_auc_score(y, p2)
    _, dp = delong_p(y, np.array(p3), np.array(p2))

    # bootstrap ΔAUROC
    rng = np.random.default_rng(0)
    y_ = np.array(y); p3_ = np.array(p3); p2_ = np.array(p2)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(y_), len(y_))
        if len(set(y_[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y_[idx], p3_[idx]) - roc_auc_score(y_[idx], p2_[idx]))
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

    with open(os.path.join(P.WORK_DIR, "preds_task1_paired.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "prob_3d", "prob_2d"])
        for (pth, lab), aa, bb in zip(rows, p3, p2):
            w.writerow([os.path.basename(pth), lab, f"{aa:.6f}", f"{bb:.6f}"])

    res = {"n": len(rows), "n_pos": int(sum(y)), "n_neg": int(len(y) - sum(y)),
           "auc_3d": round(auc3, 4), "auc_2d_ensemble": round(float(auc2), 4),
           "delta": round(auc3 - float(auc2), 4), "delong_p": round(dp, 4),
           "boot_delta_ci": [round(ci[0], 4), round(ci[1], 4)],
           "note": "3D=locked MONAI DenseNet121; 2D=centralslice sensitivity, 3-seed ensemble. "
                   "Same Validation clips. A CI crossing 0 = no detected difference, NOT equivalence."}
    json.dump(res, open(os.path.join(P.WORK_DIR, "paired_3d_vs_2d_TASK1.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
