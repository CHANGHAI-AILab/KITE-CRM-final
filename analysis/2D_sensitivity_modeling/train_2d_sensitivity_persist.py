#!/usr/bin/env python3
"""2D Task 2 sensitivity model — persistence run: save per-seed weights (+ SHA256) and
per-epoch training logs, so the 3D-vs-2D ablation is fully reproducible/auditable.

Reproduces the 3-seed recipe of paired_heldout_neibu.py (same seeds/hyperparameters),
training on the identical set (CACHE_2D_TASK2/Training). After training:
  1) save each seed to CKDIR_TASK2/seed{S}.pt (+ print sha256)
  2) log train_loss / full-Validation AUROC per epoch
  3) run inference on the paired complete-case subset and report 3D-vs-2D AUROC for a
     consistency check against paired_3d_vs_2d_HELDOUT.json / preds_2d_heldout.csv

Weights are saved at the FINAL epoch (no paired-subset model selection), matching the paired
comparison. This is a SENSITIVITY / ABLATION model only — never deployed, never used to
set a locked operating threshold. Do not relabel these weights as paper-deployment weights.

Outputs (under CKDIR_TASK2): seed{0,1,2}.pt, weight_hashes.txt, train_summary.json
"""
import os, glob, csv, random, hashlib, json
import numpy as np, nibabel as nib
from scipy import ndimage
import torch, torch.nn as nn
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import sensitivity_paths as P

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = P.CACHE_2D_TASK2
CKDIR = P.CKDIR_TASK2
os.makedirs(CKDIR, exist_ok=True)
SEEDS = [0, 1, 2]; EPOCHS = 40; BS = 32
IMG = P.HELDOUT_CT_DIR
MSK = P.HELDOUT_MASK_DIR
LES = 2; WL, WW = 40, 400; LO, HI = WL - WW / 2, WL + WW / 2; SZ = 224; MARGIN = 0.25


def norm(b):
    b = str(b).split('_')[0].lstrip('0'); return b or '0'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


# ---- paired-subset labels + locked 3D probabilities ----
r3 = list(csv.DictReader(open(P.PRED3D_TASK2_CSV)))
pk = [k for k in r3[0] if k.endswith('patient_id')][0]
held = {}; p3d = {}
for r in r3:
    if r['cohort'] == 'Validation' and 'bingli' not in r.get('leakage_flag', ''):
        k = norm(r[pk]); held[k] = int(r['label_malignant']); p3d[k] = float(r['probability'])
img = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(IMG + '/*.nii.gz')}
msk = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(MSK + '/*.nii.gz')}
hkeys = sorted(set(held) & set(img) & set(msk))


def win(x):
    x = np.clip(x, LO, HI); return ((x - LO) / (HI - LO) * 255).astype(np.uint8)


def extract(k):
    im = nib.load(img[k]).get_fdata(); mk = nib.load(msk[k]).get_fdata()
    if im.shape != mk.shape:
        return None
    les = (mk == LES); areas = les.reshape(-1, les.shape[2]).sum(0)
    if areas.max() == 0:
        return None
    z = int(areas.argmax()); sl = im[:, :, z]; m2 = les[:, :, z]
    ys, xs = np.where(m2); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    r = max(int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1 + 2 * MARGIN) / 2), 16)
    H, W = sl.shape; Y0, Y1, X0, X1 = cy - r, cy + r, cx - r, cx + r
    pad = [[max(0, -Y0), max(0, Y1 - H)], [max(0, -X0), max(0, X1 - W)]]
    patch = win(np.pad(sl[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)], pad, mode='edge'))
    patch = np.clip(ndimage.zoom(patch.astype(np.float32),
                    (SZ / patch.shape[0], SZ / patch.shape[1]), order=1), 0, 255).astype(np.uint8)
    return patch


Xheld = {}
for k in hkeys:
    p = extract(k)
    if p is not None:
        Xheld[k] = p
hkeys = [k for k in hkeys if k in Xheld]
print(f"[complete-pair] extracted={len(hkeys)}", flush=True)


class DS(Dataset):
    def __init__(s, split, aug):
        s.items = []; s.aug = aug
        for lab in (0, 1):
            for f in glob.glob(f"{ROOT}/{split}/{lab}/*.npy"):
                s.items.append((f, lab))

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        f, y = s.items[i]; x = np.load(f).astype(np.float32) / 255.
        if s.aug:
            if random.random() < .5:
                x = x[:, ::-1].copy()
            if random.random() < .5:
                x = x[::-1, :].copy()
        x = (x - .485) / .229
        return torch.from_numpy(np.stack([x] * 3, 0)), y, os.path.basename(f)[:-4]


def infer(model, cache):
    model.eval(); Pr = {}
    with torch.no_grad():
        for k in hkeys:
            x = cache[k].astype(np.float32) / 255.; x = (x - .485) / .229
            t = torch.from_numpy(np.stack([x] * 3, 0))[None].to(DEV)
            Pr[k] = float(torch.softmax(model(t), 1)[0, 1].cpu())
    return Pr


probs = {k: [] for k in hkeys}; hashes = {}
for s in SEEDS:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    tr = DS("Training", True); va = DS("Validation", False)
    yt = [y for _, y in tr.items]
    w = torch.tensor([len(yt) / (2 * yt.count(0)), len(yt) / (2 * yt.count(1))], device=DEV)
    dtr = DataLoader(tr, BS, shuffle=True, num_workers=4)
    dva = DataLoader(va, BS, shuffle=False, num_workers=4)
    m = M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, 2); m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.CrossEntropyLoss(weight=w)
    for ep in range(EPOCHS):
        m.train(); tot = 0; n = 0
        for x, y, _ in dtr:
            x = x.to(DEV); y = y.to(DEV); opt.zero_grad(); l = lf(m(x), y); l.backward(); opt.step()
            tot += l.item() * len(y); n += len(y)
        sch.step()
        # full-val AUROC (monitoring only, NOT used for model selection)
        m.eval(); Pr = []; Y = []
        with torch.no_grad():
            for x, y, _ in dva:
                Pr += torch.softmax(m(x.to(DEV)), 1)[:, 1].cpu().tolist(); Y += y.tolist()
        vauc = roc_auc_score(Y, Pr)
        print(f"[seed{s}][ep{ep+1:02d}] train_loss={tot/n:.4f} fullval_AUC={vauc:.4f}", flush=True)
    # save final-epoch weights (consistent with the paired comparison: no pair-subset selection)
    pt = f"{CKDIR}/seed{s}.pt"; torch.save(m.state_dict(), pt)
    hh = sha256(pt); hashes[f"seed{s}"] = hh
    print(f"[seed{s}] saved {pt} sha256={hh[:16]}...", flush=True)
    Ph = infer(m, Xheld)
    for k in hkeys:
        probs[k].append(Ph[k])

# ---- paired-subset consistency check ----
p2 = {k: float(np.mean(v)) for k, v in probs.items()}
y = np.array([held[k] for k in hkeys]); pa2 = np.array([p2[k] for k in hkeys]); pa3 = np.array([p3d[k] for k in hkeys])
auc2 = roc_auc_score(y, pa2); auc3 = roc_auc_score(y, pa3)
print(f"\n[complete-pair reproduction] 3D={auc3:.4f} 2D={auc2:.4f} Δ={auc3-auc2:+.4f}", flush=True)

with open(f"{CKDIR}/weight_hashes.txt", "w") as f:
    for k, v in hashes.items():
        f.write(f"{k}  sha256={v}\n")
json.dump(dict(seeds=SEEDS, epochs=EPOCHS, hashes=hashes,
    auc_2d_heldout=round(auc2, 4), auc_3d_heldout=round(auc3, 4),
    reproduced_delta=round(auc3 - auc2, 4),
    note="Sensitivity/ablation weights. Final-epoch; no paired-subset model selection."),
    open(f"{CKDIR}/train_summary.json", "w"), indent=2, ensure_ascii=False)
print(f"[DONE] weights + hashes + logs persisted -> {CKDIR}", flush=True)
