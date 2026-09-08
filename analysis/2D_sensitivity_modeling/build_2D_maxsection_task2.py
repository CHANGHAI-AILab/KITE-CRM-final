#!/usr/bin/env python3
"""2D arm, stage 1: extract the maximal-lesion axial section as a 2D patch (Task 2).

For each case: pick the axial slice whose predicted lesion mask (label == 2) has the
largest area, crop centred on the lesion bounding box, apply a soft-tissue window
[-160, 240] HU, and resize to 224x224 (uint8). Only cases in Task 2's official
Training/Validation split that also have an aligned CT + mask are processed.

Inputs come from sensitivity_paths (env-configurable):
    LESION_CT_DIR, LESION_MASK_DIR
Split/labels come from the released Task 2 loader (classification_tasks2_4/train_LE.py
via build_items(2)); adapt the import below to your loader if needed.

Output: CACHE_2D_TASK2/<split>/<label>/<pid>.npy  +  manifest_2d_task2.csv
"""
import os, glob, csv
import numpy as np, nibabel as nib
from scipy import ndimage

import sensitivity_paths as P
# Task 2 split/labels: reuse the released classifier's item builder.
# build_items(2) -> (train_items, val_items), each a list of (feature_path, label).
from build_items_task2 import build_items, norm  # thin adapter over the released loader

OUT = P.CACHE_2D_TASK2
IMG_DIR = P.LESION_CT_DIR
MSK_DIR = P.LESION_MASK_DIR
LES = 2                       # lesion label
WL, WW = 40, 400             # soft-tissue window -> [-160, 240]
LO, HI = WL - WW / 2, WL + WW / 2
SZ = 224
MARGIN = 0.25                # bbox expansion ratio


def win(x):
    x = np.clip(x, LO, HI); x = (x - LO) / (HI - LO) * 255.0
    return x.astype(np.uint8)


def crop_center(sl, mask2d):
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    r = int(max(h, w) * (1 + 2 * MARGIN) / 2)
    r = max(r, 16)
    Y0, Y1, X0, X1 = cy - r, cy + r, cx - r, cx + r
    H, W = sl.shape
    pad = [[max(0, -Y0), max(0, Y1 - H)], [max(0, -X0), max(0, X1 - W)]]
    Y0, Y1, X0, X1 = max(0, Y0), min(H, Y1), max(0, X0), min(W, X1)
    return np.pad(sl[Y0:Y1, X0:X1], pad, mode='edge')


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, va = build_items(2)
    t2 = {}
    for p, y in tr:
        t2[norm(os.path.basename(p).split('_')[0])] = (y, 'Training')
    for p, y in va:
        t2[norm(os.path.basename(p).split('_')[0])] = (y, 'Validation')
    imgs = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(IMG_DIR + '/*.nii.gz')}
    masks = {norm(os.path.basename(f).split('_')[0]): f for f in glob.glob(MSK_DIR + '/*.nii.gz')}
    keys = sorted(set(imgs) & set(masks) & set(t2))
    print(f"[start] processing {len(keys)} cases", flush=True)
    rows = []; ok = 0; skip = 0
    for i, k in enumerate(keys):
        y, split = t2[k]
        try:
            im = nib.load(imgs[k]).get_fdata()
            mk = nib.load(masks[k]).get_fdata()
            if im.shape != mk.shape:
                skip += 1; continue
            les = (mk == LES)
            areas = les.reshape(-1, les.shape[2]).sum(0)
            if areas.max() == 0:
                skip += 1; continue
            z = int(areas.argmax())
            patch = crop_center(im[:, :, z], les[:, :, z])
            if patch is None:
                skip += 1; continue
            patch = win(patch)
            zoom = (SZ / patch.shape[0], SZ / patch.shape[1])
            patch = np.clip(ndimage.zoom(patch.astype(np.float32), zoom, order=1), 0, 255).astype(np.uint8)
            d = f"{OUT}/{split}/{y}"; os.makedirs(d, exist_ok=True)
            np.save(f"{d}/{k}.npy", patch)
            rows.append([k, split, y, z, int(areas.max())]); ok += 1
        except Exception:
            skip += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(keys)} ok={ok} skip={skip}", flush=True)
    with open(os.path.join(P.WORK_DIR, "manifest_2d_task2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "split", "label", "z_maxsec", "lesion_area_px"])
        w.writerows(rows)
    print(f"[done] ok={ok} skip={skip} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
