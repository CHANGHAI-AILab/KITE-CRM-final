#!/usr/bin/env python3
"""Task 2 2D inference on ANY cohort (Training/Validation/Test cohorts).

Frozen 2D sensitivity DenseNet-121 ensemble (seed0/1/2). Nothing trained/tuned here.
Input per case = raw CT .nii.gz + lesion mask .nii.gz (lesion voxels == label 2).
Slice rule = max-lesion-area axial section, soft-tissue window [-160,240], lesion
bbox-center crop, 224 — identical to build_2D_maxsection_task2 / paired_heldout_neibu.

Locates CT by normalized id across --img-dirs, mask across --msk-dirs (priority order).
A case is inferable only if BOTH a CT and a nonzero lesion mask are found. Directory
arguments may be absolute or relative to sensitivity_paths.DATA_ROOT. The case label
table is sensitivity_paths.TASK2_LABELS_XLSX; checkpoints come from CKDIR_TASK2.

Usage:
  python infer_2d_task2_anycohort.py \
     --img-dirs "<lesion_ct_dir>,<test_site_ct_dir>" \
     --msk-dirs "<lesion_mask_dir>,<test_site_mask_dir>" \
     --cohorts "Test 1,Test 2,Test 3,Test 4,Test 5,Test 6,Training,Validation" \
     --out preds_2d_task2_allcohorts.csv
"""
import os, glob, csv, argparse, numpy as np, nibabel as nib, torch, torch.nn as nn
import torchvision.models as M
from scipy import ndimage

import sensitivity_paths as P

XLSX = P.TASK2_LABELS_XLSX
CKDIR = P.CKDIR_TASK2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LES = 2; WL, WW = 40, 400; LO, HI = WL - WW / 2, WL + WW / 2; SZ = 224; MARGIN = 0.25


def norm(b):
    b = str(b).split('_')[0].split('.')[0].lstrip('0')
    return b or '0'


def read_xlsx_targets(cohorts):
    """Task2 gold col: 0 benign / 1 malignant / 2 normal. Keep 0/1 only."""
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, read_only=True).active
    hdr = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    bi = hdr.index('编号'); dni = hdr.index('dataset_name')
    gi = [i for i, h in enumerate(hdr) if h and '任务2' in str(h) and '金标准' in str(h)][0]
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[dni] in cohorts:
            try:
                lab = int(r[gi])
            except (TypeError, ValueError):
                lab = None
            out[norm(r[bi])] = (lab, r[dni], r[bi])
    return out


def build_index(dirs):
    idx = {}
    for d in dirs:
        dd = d if os.path.isabs(d) else os.path.join(P.DATA_ROOT, d)
        for f in glob.glob(f"{dd}/*.nii.gz") + glob.glob(f"{dd}/**/*.nii.gz", recursive=True):
            idx.setdefault(norm(os.path.basename(f)), f)
    return idx


def win(x):
    x = np.clip(x, LO, HI); return ((x - LO) / (HI - LO) * 255).astype(np.uint8)


def extract(img_path, msk_path):
    mimg = nib.load(img_path); mmsk = nib.load(msk_path)
    if mimg.shape != mmsk.shape:
        return None, "shape_mismatch"
    mk = np.asanyarray(mmsk.dataobj)                    # full mask (needed for argmax area)
    les = (mk == LES); ar = les.reshape(-1, les.shape[2]).sum(0)
    if ar.max() == 0:
        return None, "no_lesion_label2"
    z = int(ar.argmax())
    sl = np.asanyarray(mimg.dataobj[:, :, z]).astype(np.float32)   # lazy: only the max-lesion slice
    m2 = les[:, :, z]
    ys, xs = np.where(m2)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    r = max(int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1 + 2 * MARGIN) / 2), 16)
    H, W = sl.shape; Y0, Y1, X0, X1 = cy - r, cy + r, cx - r, cx + r
    pad = [[max(0, -Y0), max(0, Y1 - H)], [max(0, -X0), max(0, X1 - W)]]
    p = win(np.pad(sl[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)], pad, mode='edge'))
    p = np.clip(ndimage.zoom(p.astype(np.float32), (SZ / p.shape[0], SZ / p.shape[1]), order=1), 0, 255)
    return p.astype(np.uint8), "ok"


def load_models():
    nets = []
    for s in (0, 1, 2):
        m = M.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        m.load_state_dict(torch.load(f"{CKDIR}/seed{s}.pt", map_location=DEV))
        nets.append(m.to(DEV).eval())
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dirs", required=True)
    ap.add_argument("--msk-dirs", required=True)
    ap.add_argument("--cohorts", default="Test 1,Test 2,Test 3,Test 4,Test 5,Test 6,Training,Validation")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="preds_2d_task2_allcohorts.csv")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    cohorts = [c.strip() for c in a.cohorts.split(",")]
    targets = read_xlsx_targets(set(cohorts))
    img = build_index([d.strip() for d in a.img_dirs.split(",")])
    msk = build_index([d.strip() for d in a.msk_dirs.split(",")])
    nets = load_models()
    print(f"targets={len(targets)} imgs={len(img)} masks={len(msk)}", flush=True)

    rows = []; found = 0
    items = sorted(targets.items())
    if a.end_idx is None:
        a.end_idx = len(items)
    items = items[a.start_idx:a.end_idx]
    print(f"Processing subset [{a.start_idx}:{a.end_idx}] = {len(items)} cases", flush=True)

    for i, (k, (lab, coh, raw)) in enumerate(items):
        ip, mp = img.get(k), msk.get(k)
        if ip is None or mp is None:
            rows.append([raw, k, coh, lab, "", "MISSING_IMG" if ip is None else "MISSING_MASK"]); continue
        patch, status = extract(ip, mp)
        if patch is None:
            rows.append([raw, k, coh, lab, "", status]); continue
        x = patch.astype(np.float32) / 255.; x = (x - .485) / .229
        t = torch.from_numpy(np.stack([x] * 3, 0))[None].to(DEV)
        ps = []
        with torch.no_grad():
            for m in nets:
                ps.append(float(torch.softmax(m(t), 1)[0, 1].cpu()))
        rows.append([raw, k, coh, lab, f"{np.mean(ps):.6f}", os.path.basename(ip)]); found += 1
        if (i + 1) % 50 == 0:
            print(f"  {a.start_idx + i+1}/{a.end_idx} found={found}", flush=True)

    outp = os.path.join(P.WORK_DIR, a.out)
    with open(outp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id_raw", "id_norm", "cohort", "label", "prob_2d", "file_or_status"])
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r[2], [0, 0]); by[r[2]][0] += 1; by[r[2]][1] += 1 if r[4] else 0
    print(f"\n[DONE] -> {outp}")
    for c in cohorts:
        if c in by:
            print(f"  {c:<12} total={by[c][0]:>5} inferred={by[c][1]:>5} missing={by[c][0]-by[c][1]:>5}")


if __name__ == "__main__":
    main()
