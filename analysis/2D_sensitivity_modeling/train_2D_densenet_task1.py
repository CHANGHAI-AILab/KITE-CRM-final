#!/usr/bin/env python3
"""Task 1 2D sensitivity arm: normal(0) vs abnormal(1), 2D DenseNet-121.

Parallels the locked 3D Task 1 model (MONAI 3D DenseNet-121) for the 3D-vs-2D
architecture ablation. Because Task 1 includes NORMAL kidneys that have NO lesion
mask, we CANNOT use max-lesion-area slice selection (that would leak the label).
Instead we use a label-agnostic rule: the CENTRAL axial slice of each kidney clip
(the same clips the 3D model consumes), 3-slice mean for context, resize to 224,
ImageNet-normalized, replicated to 3 channels.

Split follows the Task 1 clip table exactly (same Validation split as the 3D arm;
see sensitivity_paths.TASK1_SPLIT_CSV). Persists weights + SHA256 + per-epoch logs.

This is a SENSITIVITY / ABLATION model only: never deployed, never used to set any
locked operating threshold.
"""
import os, csv, json, hashlib, argparse
import numpy as np, SimpleITK as sitk, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.ndimage import zoom

import sensitivity_paths as P

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def resolve(p):
    """Resolve a clip path from the split table against DATA_ROOT / WORK_DIR."""
    for c in [p, os.path.join(P.DATA_ROOT, p), os.path.join(P.WORK_DIR, p)]:
        if os.path.exists(c):
            return c
    return None


def load_rows(split):
    rows = []
    for r in csv.DictReader(open(P.TASK1_SPLIT_CSV)):
        if r["split"] != split:
            continue
        c = resolve(r["path"])
        if c:
            rows.append((c, int(r["label"])))
    return rows


def central_slice(path):
    """Label-agnostic 2D section: 3-slice mean around the clip's central axial slice."""
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32) / 255.0
    z = a.shape[0]
    c = z // 2
    lo, hi = max(0, c - 1), min(z, c + 2)
    sl = a[lo:hi].mean(0)                        # (H,W) in [0,1]
    sl = zoom(sl, (224 / sl.shape[0], 224 / sl.shape[1]), order=1)
    img = np.stack([sl, sl, sl], 0)             # (3,224,224)
    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return img.astype(np.float32)


class DS(Dataset):
    def __init__(self, rows, aug=False):
        self.rows = rows; self.aug = aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, lab = self.rows[i]
        img = central_slice(path)
        if self.aug:
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
        return torch.as_tensor(img), lab


def build_net():
    net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    net.classifier = nn.Linear(net.classifier.in_features, 2)
    return net


def evaluate(net, loader, device):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(net(x.to(device)), 1)[:, 1]
            ps += out.cpu().tolist(); ys += y.tolist()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    acc = accuracy_score(ys, [1 if p >= 0.5 else 0 for p in ps])
    return auc, acc, ys, ps


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_rows("Training"); va = load_rows("Validation")
    print(f"train={len(tr)} valid={len(va)} pos_tr={sum(l for _,l in tr)} pos_va={sum(l for _,l in va)}", flush=True)

    ckdir = P.CKDIR_TASK1
    os.makedirs(ckdir, exist_ok=True)
    seed_aucs = {}; last_preds = None
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed)
        g = torch.Generator(); g.manual_seed(seed)
        trl = DataLoader(DS(tr, aug=True), batch_size=a.bs, shuffle=True,
                         num_workers=8, drop_last=True, generator=g)
        val = DataLoader(DS(va), batch_size=a.bs, shuffle=False, num_workers=8)

        net = build_net().to(device)
        n1 = sum(l for _, l in tr); n0 = len(tr) - n1
        w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        scaler = torch.cuda.amp.GradScaler()

        log = []
        best = 0.0; best_state = None; best_preds = None
        for ep in range(a.epochs):
            net.train(); tot = 0.0
            for x, y in trl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = crit(net(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item()
            sch.step()
            auc, acc, ys, ps = evaluate(net, val, device)
            log.append({"epoch": ep, "train_loss": tot / max(len(trl), 1),
                        "val_auc": auc, "val_acc": acc})
            if auc == auc and auc > best:
                best = auc; best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                best_preds = (ys, ps)
            print(f"seed{seed} ep{ep:02d} loss={tot/max(len(trl),1):.4f} valAUC={auc:.4f} valACC={acc:.4f}", flush=True)

        ckpt = f"{ckdir}/seed{seed}.pt"
        torch.save(best_state, ckpt)
        h = sha256(ckpt)
        json.dump(log, open(f"{ckdir}/seed{seed}_log.json", "w"), indent=2)
        seed_aucs[seed] = {"best_auc": best, "sha256": h}
        last_preds = best_preds
        print(f"seed{seed} DONE best={best:.4f} sha={h[:8]}", flush=True)

    aucs = [v["best_auc"] for v in seed_aucs.values()]
    # per-case predictions from last seed's best
    ys, ps = last_preds
    with open(os.path.join(P.WORK_DIR, "preds_2d_task1_val.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["idx", "label", "prob_2d"])
        for i, (yv, pv) in enumerate(zip(ys, ps)):
            wr.writerow([i, yv, f"{pv:.6f}"])
    summary = {
        "arm": "Task1_2D_DenseNet121_centralslice_sensitivity",
        "n_train": len(tr), "n_valid": len(va),
        "val_auc_mean": float(np.mean(aucs)), "val_auc_std": float(np.std(aucs)),
        "per_seed": seed_aucs,
        "slice_rule": "label-agnostic central axial slice (3-slice mean), 224, ImageNet-norm",
        "note": "Same Validation split as the locked 3D Task 1 model. Sensitivity model only.",
    }
    json.dump(summary, open(os.path.join(P.WORK_DIR, "task1_2d_sensitivity_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
