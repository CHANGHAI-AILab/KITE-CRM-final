# Standalone HU-Rule Cystic–Solid Gating Model

Version 1.0

## Scope and intended use

The formal KITE-CRM cystic–solid gate is a standalone deterministic HU-rule model (Software Registration No. 2024SR1383073). It labels each detected lesion as `cystic`, `solid`, or `none`; only lesions routed as `cystic` proceed to the CRM risk-support pathway. This is an independent rule-based model, not a CNN and not a learned checkpoint.

The implementation uses the LightM-UNet lesion mask and the corresponding corticomedullary-phase (CMP) CT volume. The defaults in this release are `HU_THR=20.0` and `FRAC=0.5`. Command-line overrides are intended for sensitivity analysis only and do not redefine the locked deployment configuration.

## Imaging phase

The gate is packaged for CMP contrast-enhanced CT. CMP is the sole imaging phase consumed by the deployed KITE-CRM modules.

## Rule

1. Use LightM-UNet `label 2` as the 3D lesion ROI. The label is used for localization only; it is not a cystic-versus-solid class and there is no deployed label-3 cyst channel.
2. Within the ROI, classify voxels by attenuation: `HU <= HU_THR` is cystic-density and `HU > HU_THR` is solid-density.
3. Compute `cyst_fraction = cystic_voxels / lesion_voxels`.
4. Return `cystic` when `cyst_fraction >= FRAC`, `solid` otherwise, and `none` when no lesion voxels are present.

## Locked deployment defaults

| Parameter | Meaning | Release default |
|---|---|---:|
| `--hu-thr` | voxel cut-off; `HU <= threshold` is cystic-density | `20.0 HU` |
| `--frac` | lesion cut-off; `cyst_fraction >= threshold` is cystic | `0.5` |

Use other values only for explicitly labelled sensitivity analyses; doing so does not reproduce the deployed gate.

## Input contract

Segmentation predictions use labels `0=background`, `1=kidney`, and `2=lesion/tumor` (one `<stem>.nii.gz` per case). Provide the matching full-size CMP CT in HU at `<images_dir>/<stem>_0000.nii.gz`. The prediction and CT must share the same voxel grid.

## Usage

```bash
python gate_cystic_solid_cmp.py <pred_dir> <images_dir> <out_csv> \
  --hu-thr 20 --frac 0.5
```

Requires Python 3, `numpy`, and `nibabel`.

The output columns are `name`, `lesion_vox`, `cyst_vox`, `solid_vox`, `cyst_fraction`, and `gate`.

## Files

| File | Description |
|---|---|
| `gate_cystic_solid_cmp.py` | HU-rule implementation |
| `README.md` | This documentation |
| `SHA256SUMS.txt` | Package checksums |
