#!/usr/bin/env python3
"""2D arm, stage 2: 2D DenseNet-121 benign/malignant classifier (Task 2), multi-seed.

- Input : CACHE_2D_TASK2/<split>/<label>/<pid>.npy  (224x224 uint8 max-lesion sections)
- Train on Training, calculate the implemented ROC/Youden validation cut-off, and export
  validation predictions for sensitivity analysis.
- Multiple seeds (default 0,1,2); export per-seed Validation case-level predictions.
- Outputs: preds_2d_task2_seed{S}.csv, summary_2d_task2.csv (under WORK_DIR)

Paired 3D-vs-2D comparison is done downstream (paired_heldout_neibu.py) on the shared
complete-pair subset. This script is a SENSITIVITY / ABLATION model only: it is never
deployed and never used to set any locked operating threshold. Its local cut-off is not
the source or selection rule for the deployed Task 2 threshold of 0.586442888.
"""
import os, glob, csv, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, brier_score_loss

import sensitivity_paths as P

ROOT = P.CACHE_2D_TASK2
SEEDS = [0, 1, 2]
EPOCHS = 40
BS = 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DS(Dataset):
    def __init__(self, split, aug=False):
        self.items = []
        for lab in (0, 1):
            for f in glob.glob(f"{ROOT}/{split}/{lab}/*.npy"):
                self.items.append((f, lab))
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        x = np.load(f).astype(np.float32) / 255.0
        if self.aug:
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
            if random.random() < 0.5:
                x = x[::-1, :].copy()
        x = (x - 0.485) / 0.229
        x = np.stack([x, x, x], 0)               # 3ch for imagenet densenet
        return torch.from_numpy(x), y, os.path.basename(f)[:-4]


def make_model():
    m = M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m.to(DEV)


def run_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    tr = DS("Training", aug=True); va = DS("Validation", aug=False)
    ytr = [y for _, y in tr.items]; n0 = ytr.count(0); n1 = ytr.count(1)
    w = torch.tensor([len(ytr) / (2 * n0), len(ytr) / (2 * n1)], dtype=torch.float32, device=DEV)
    dtr = DataLoader(tr, BS, shuffle=True, num_workers=4, drop_last=False)
    dva = DataLoader(va, BS, shuffle=False, num_workers=4)
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=w)
    best = None; best_auc = -1
    for ep in range(EPOCHS):
        m.train()
        for x, y, _ in dtr:
            x = x.to(DEV); y = y.to(DEV)
            opt.zero_grad(); out = m(x); l = lossf(out, y); l.backward(); opt.step()
        sch.step()
        # val
        m.eval(); Pr = []; Y = []; ID = []
        with torch.no_grad():
            for x, y, pid in dva:
                p = torch.softmax(m(x.to(DEV)), 1)[:, 1].cpu().numpy()
                Pr += p.tolist(); Y += y.tolist(); ID += list(pid)
        auc = roc_auc_score(Y, Pr)
        if auc > best_auc:
            best_auc = auc; best = (np.array(Pr), np.array(Y), ID)
        if (ep + 1) % 10 == 0:
            print(f"  seed{s} ep{ep+1} valAUC={auc:.4f}", flush=True)
    Pr, Y, ID = best
    fpr, tpr, thr = roc_curve(Y, Pr); j = (tpr - fpr).argmax(); cut = float(thr[j])
    pred = (Pr >= cut).astype(int)
    acc = accuracy_score(Y, pred); auc = roc_auc_score(Y, Pr); br = brier_score_loss(Y, Pr)
    with open(os.path.join(P.WORK_DIR, f"preds_2d_task2_seed{s}.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["pid", "label", "prob", "thr", "pred"])
        for pid, yy, pp in zip(ID, Y, Pr):
            wtr.writerow([pid, int(yy), f"{pp:.4f}", f"{cut:.4f}", int(pp >= cut)])
    print(f"[seed{s}] valAUC={auc:.4f} ACC={acc:.4f} thr={cut:.4f} Brier={br:.4f} n={len(Y)}", flush=True)
    return dict(seed=s, auc=auc, acc=acc, thr=cut, brier=br, n=len(Y))


def main():
    print(f"[start] dev={DEV} tr={len(DS('Training'))} va={len(DS('Validation'))}", flush=True)
    res = [run_seed(s) for s in SEEDS]
    aucs = [r['auc'] for r in res]
    with open(os.path.join(P.WORK_DIR, "summary_2d_task2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "auc", "acc", "thr", "brier", "n"])
        for r in res:
            w.writerow([r['seed'], f"{r['auc']:.4f}", f"{r['acc']:.4f}", f"{r['thr']:.4f}", f"{r['brier']:.4f}", r['n']])
        w.writerow(["mean±sd", f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}", "", "", "", ""])
    print(f"[DONE] 2D valAUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}", flush=True)


if __name__ == "__main__":
    main()
