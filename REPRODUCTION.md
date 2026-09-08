# Reproduction guide

## 1. Install

Use Python 3.9 or 3.10 and install the packages in `requirements.txt`. Running the LightM-UNet Mamba backbone requires the original CUDA-compatible stack described in `ENVIRONMENT.md`; the newer reference stack is suitable for the released classifier and sensitivity scripts but is not guaranteed to be bit-identical.

## 2. Obtain checkpoints

Download the `UPLOAD_READY` package from the Google Drive link in `paper_configs/eclinm_v1/WEIGHTS_DOWNLOAD.md`. Verify SHA256 values before loading any checkpoint. The repository does not contain patient CT, masks, or official patient-level result workbooks.

## 3. Run the formal HU gate

```bash
python cystic_solid_gating_gate_v1.0/gate_cystic_solid_cmp.py \
  <predictions_dir> <cmp_images_dir> <output_csv> \
  --hu-thr 20 --frac 0.5
```

Predictions must use the LightM-UNet label convention `0=background`, `1=kidney`, `2=lesion`. CT files must be full-size CMP volumes in HU and must share the prediction grid.

## 4. Run 3D classification

Task 1 uses `task1_abnormality/train_task1.py`, which constructs MONAI
`DenseNet121(spatial_dims=3)` for the volumetric kidney clip. Tasks 2-4 use
`classification_tasks2_4/model_LE.py` with the decoder features exported by
`segmentation/feature_extraction/dump_decoder_features.py`. Tasks 2, 3, and 4
are separate two-class checkpoints sharing the same 3D FPN/class-LE
formulation.

## 5. Reproduce the architecture sensitivity analysis

The post hoc complete-pair 2D architecture-sensitivity analysis uses the central
three-slice kidney-clip mean for Task 1 and the largest-area lesion-mask axial section
for Task 2. The complete modeling chain is documented under
`analysis/2D_sensitivity_modeling/`. The final Task 2 validation analysis uses the
full cohort of `n=235` records. The public release does not include patient-level
result workbooks or case-level outputs.
This is not random sampling and is not a complete external validation. These
comparators are not part of the deployed pathway and do not define any locked threshold.
The 2D scripts require restricted patient data, checkpoints, and the official Task 2
split/label loader. The optional 3D re-inference adapter additionally requires a
compatible external `train_fpn` module.

## 6. Data and provenance limitations

The official workbooks, patient images, and complete clinical cascade are not redistributed here. A prediction table reconstructed from source workbooks is provenance material, not a claim that every upstream gate decision is directly observable. No threshold may be re-derived on an external evaluation cohort.
