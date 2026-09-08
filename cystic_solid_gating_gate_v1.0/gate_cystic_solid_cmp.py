#!/usr/bin/env python3
"""Standalone HU-rule cystic-solid gate for corticomedullary-phase renal CT.

The formal KITE-CRM gate is an independent deterministic HU-rule model
(Software Registration No. 2024SR1383073). It labels each detected lesion as
cystic, solid, or none; lesions labeled cystic enter the CRM risk-support path.
This is not a learned CNN checkpoint.

The implementation uses the LightM-UNet label-2 lesion ROI and the corresponding
CMP CT volume. The locked release defaults are HU_THR=20.0 and FRAC=0.5.
Command-line overrides are for sensitivity analyses only.
"""
import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np

PHASE = "corticomedullary (CMP, contrast-enhanced)"


def lesion_hu(pred_path, ct_path):
    pred = np.asanyarray(nib.load(pred_path).dataobj)
    lesion = pred == 2
    n = int(lesion.sum())
    if n == 0:
        return 0, None
    ct = np.asanyarray(nib.load(ct_path).dataobj).astype(np.float32)
    if ct.shape != pred.shape:
        raise ValueError(f"shape mismatch: ct{ct.shape} vs pred{pred.shape}")
    return n, ct[lesion]


def main():
    parser = argparse.ArgumentParser(
        description="Standalone HU-rule cystic-solid gate for CMP CT."
    )
    parser.add_argument("pred_dir", help="LightM-UNet prediction directory")
    parser.add_argument("images_dir", help="Matching CMP CT directory")
    parser.add_argument("out_csv")
    parser.add_argument("--hu-thr", type=float, default=20.0,
                        help="HU <= threshold is cystic-density (default 20.0)")
    parser.add_argument("--frac", type=float, default=0.5,
                        help="cyst_fraction >= threshold is cystic (default 0.5)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    rows = []
    cystic_count = solid_count = none_count = missing_count = 0
    for pred_path in files:
        stem = os.path.basename(pred_path).replace(".nii.gz", "")
        ct_path = os.path.join(args.images_dir, stem + "_0000.nii.gz")
        if not os.path.exists(ct_path):
            rows.append([stem, "", "", "", "", "MISSING_CT"])
            missing_count += 1
            continue
        n, hu = lesion_hu(pred_path, ct_path)
        if n == 0:
            rows.append([stem, 0, 0, 0, "", "none"])
            none_count += 1
            continue
        cyst_voxels = int((hu <= args.hu_thr).sum())
        solid_voxels = n - cyst_voxels
        fraction = cyst_voxels / n
        gate = "cystic" if fraction >= args.frac else "solid"
        cystic_count += gate == "cystic"
        solid_count += gate == "solid"
        rows.append([stem, n, cyst_voxels, solid_voxels, f"{fraction:.4f}", gate])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "lesion_vox", "cyst_vox", "solid_vox",
                         "cyst_fraction", "gate"])
        writer.writerows(rows)
    print(f"phase = {PHASE}")
    print(f"cases={len(rows)} HU_THR={args.hu_thr} FRAC={args.frac}")
    print(f"cystic={cystic_count} solid={solid_count} none={none_count}"
          + (f" missing_ct={missing_count}" if missing_count else ""))
    print(f"output = {args.out_csv}")


if __name__ == "__main__":
    main()
