#!/usr/bin/env python3
"""Post hoc complete-pair 3D-vs-2D architecture-sensitivity analysis (Task 2).

The 95-case paired subset is formed by the existing provenance filter and availability of
aligned CT, lesion mask, label, and 3D probability records. It is not random sampling or
an independent or complete external validation cohort.

- Source: HELDOUT_CT_DIR / HELDOUT_MASK_DIR — aligned CT + lesion mask (lesion == label 2).
- 2D arm: max-lesion section (soft-tissue window, bbox-center crop, 224) → mean of 3
  trained seeds. Weights are (re)trained here from CACHE_2D_TASK2/Training with the same
  recipe as train_2D_densenet_task2.py; if CKDIR_TASK2/seed{0,1,2}.pt already exist they
  are loaded instead of retrained.
- 3D arm: case-level probabilities read from PRED3D_TASK2_CSV (locked model), same cases.
- Complete-pair records → DeLong test + bootstrap ΔAUROC CI.

Report the paired ΔAUROC with its DeLong p and bootstrap CI exactly. A CI that crosses 0
means "no significant difference detected" — it does NOT establish equivalence or
non-inferiority. This is a SENSITIVITY analysis; nothing here is deployed.

Outputs (under WORK_DIR): paired_3d_vs_2d_HELDOUT.json, preds_2d_heldout.csv
"""
import os, glob, csv, json, random
import numpy as np, nibabel as nib
from scipy import ndimage, stats
import torch, torch.nn as nn
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import sensitivity_paths as P

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = P.HELDOUT_CT_DIR
MSK = P.HELDOUT_MASK_DIR
CACHE = P.CACHE_2D_TASK2
CKDIR = P.CKDIR_TASK2
LES = 2; WL, WW = 40, 400; LO, HI = WL - WW / 2, WL + WW / 2; SZ = 224; MARGIN = 0.25
SEEDS = [0, 1, 2]; EPOCHS = 40; BS = 32


def norm(b):
    b = str(b).split('_')[0].lstrip('0')
    return b or '0'


# --- paired-subset labels + 3D probabilities (locked model) ---
r3 = list(csv.DictReader(open(P.PRED3D_TASK2_CSV)))
pk = [k for k in r3[0] if k.endswith('patient_id')][0]
held = {}; p3d = {}
for r in r3:
    # Pair-definition filter: Validation records outside the original 3D training
    # feature pool, as indicated by the existing leakage flag.
    if r['cohort'] == 'Validation' and 'bingli' not in r.get('leakage_flag', ''):
        k = norm(r[pk]); held[k] = int(r['label_malignant']); p3d[k] = float(r['probability'])

img = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(IMG + '/*.nii.gz')}
msk = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(MSK + '/*.nii.gz')}
keys = sorted(set(held) & set(img) & set(msk))
print(f"paired-candidate={len(held)} complete-pair={len(keys)}", flush=True)


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
    patch = np.pad(sl[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)], pad, mode='edge')
    patch = win(patch)
    patch = np.clip(ndimage.zoom(patch.astype(np.float32),
                    (SZ / patch.shape[0], SZ / patch.shape[1]), order=1), 0, 255).astype(np.uint8)
    return patch


X = {}
for i, k in enumerate(keys):
    p = extract(k)
    if p is not None:
        X[k] = p
    if (i + 1) % 30 == 0:
        print(f"  extract {i+1}/{len(keys)}", flush=True)
keys = [k for k in keys if k in X]
print(f"extracted={len(keys)}", flush=True)


class DS(Dataset):
    def __init__(self, split, aug):
        self.items = []; self.aug = aug
        for lab in (0, 1):
            for f in glob.glob(f"{CACHE}/{split}/{lab}/*.npy"):
                self.items.append((f, lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]; x = np.load(f).astype(np.float32) / 255.
        if self.aug:
            if random.random() < .5:
                x = x[:, ::-1].copy()
            if random.random() < .5:
                x = x[::-1, :].copy()
        x = (x - .485) / .229
        return torch.from_numpy(np.stack([x] * 3, 0)), y


def infer_held(model):
    model.eval(); Pr = {}
    with torch.no_grad():
        for k in keys:
            x = X[k].astype(np.float32) / 255.; x = (x - .485) / .229
            t = torch.from_numpy(np.stack([x] * 3, 0))[None].to(DEV)
            Pr[k] = float(torch.softmax(model(t), 1)[0, 1].cpu())
    return Pr


def make_seed_model(s):
    """Load CKDIR_TASK2/seed{s}.pt if present, else train it (same recipe as stage 2)."""
    pt = os.path.join(CKDIR, f"seed{s}.pt")
    m = M.densenet121(weights=None if os.path.exists(pt) else M.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, 2); m = m.to(DEV)
    if os.path.exists(pt):
        m.load_state_dict(torch.load(pt, map_location=DEV))
        return m.eval()
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    tr = DS("Training", True)
    yt = [y for _, y in tr.items]
    w = torch.tensor([len(yt) / (2 * yt.count(0)), len(yt) / (2 * yt.count(1))], device=DEV)
    dl = DataLoader(tr, BS, shuffle=True, num_workers=4)
    opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.CrossEntropyLoss(weight=w)
    for ep in range(EPOCHS):
        m.train()
        for x, y in dl:
            x = x.to(DEV); y = y.to(DEV); opt.zero_grad(); lf(m(x), y).backward(); opt.step()
        sch.step()
    return m.eval()


probs = {k: [] for k in keys}
for s in SEEDS:
    m = make_seed_model(s)
    Ph = infer_held(m)
    for k in keys:
        probs[k].append(Ph[k])
    print(f"seed{s} done", flush=True)

p2 = {k: float(np.mean(v)) for k, v in probs.items()}
y = np.array([held[k] for k in keys]); pa3 = np.array([p3d[k] for k in keys]); pa2 = np.array([p2[k] for k in keys])
auc3 = roc_auc_score(y, pa3); auc2 = roc_auc_score(y, pa2)
print(f"\n=== complete-pair n={len(keys)} malignant={int(y.sum())} benign={int((1-y).sum())} ===")
print(f"3D AUROC={auc3:.4f}  2D AUROC={auc2:.4f}  Δ(3D-2D)={auc3-auc2:+.4f}")


def delong(y, p1, p2):
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1; i = j
        T2 = np.empty(N); T2[J] = T; return T2

    def comp(Pm, m):
        n = Pm.shape[1] - m; k = Pm.shape[0]
        tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
        for r in range(k):
            tx[r] = midrank(Pm[r, :m]); ty[r] = midrank(Pm[r, m:]); tz[r] = midrank(Pm[r])
        a = (tz[:, :m].sum(1) / m - (m + 1) / 2.) / n
        v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
        return a, np.cov(v01) / m + np.cov(v10) / n
    o = np.argsort(-y); m = int(y.sum()); Pm = np.vstack([p1[o], p2[o]]); a, S = comp(Pm, m)
    L = np.array([[1, -1]]); z = (L @ a) / np.sqrt(L @ S @ L.T); return a, 2 * stats.norm.sf(abs(z[0, 0]))


a, pval = delong(y, pa3, pa2)
rng = np.random.RandomState(2024); d = []
for _ in range(2000):
    bi = rng.choice(len(y), len(y), True)
    if 0 < y[bi].sum() < len(y):
        d.append(roc_auc_score(y[bi], pa3[bi]) - roc_auc_score(y[bi], pa2[bi]))
d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
print(f"DeLong p={pval:.4f}  Bootstrap Δ={d.mean():+.4f} 95%CI[{lo:+.4f},{hi:+.4f}]")

json.dump(dict(n=len(keys), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
    auc_3d=round(auc3, 4), auc_2d=round(auc2, 4), delta=round(auc3 - auc2, 4),
    delong_p=round(float(pval), 4), boot_delta=round(float(d.mean()), 4),
    boot_ci=[round(float(lo), 4), round(float(hi), 4)],
    note="Post hoc complete-pair architecture-sensitivity analysis. The subset is not "
         "random sampling or an independent external validation cohort. A CI crossing 0 "
         "indicates no detected difference, NOT equivalence."),
    open(os.path.join(P.WORK_DIR, 'paired_3d_vs_2d_HELDOUT.json'), 'w'), indent=2, ensure_ascii=False)
with open(os.path.join(P.WORK_DIR, 'preds_2d_heldout.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pid', 'label', 'prob_2d', 'prob_3d'])
    for k in keys:
        w.writerow([k, held[k], f"{p2[k]:.4f}", f"{p3d[k]:.4f}"])
print("saved paired_3d_vs_2d_HELDOUT.json + preds_2d_heldout.csv")
