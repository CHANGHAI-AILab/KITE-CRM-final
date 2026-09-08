# KITE-CRM v1.0-eclinm

Research release of the KITE-CRM single-phase CT decision-support pipeline for cystic renal masses.

## Scope

The locked pipeline consumes corticomedullary-phase (CMP) CT and contains:

1. LightM-UNet under nnU-Net v2 for 3D kidney and lesion segmentation. The same forward pass exposes a 3D decoder feature pyramid.
2. A MONAI `DenseNet121(spatial_dims=3)` for Task 1 kidney-level abnormality screening.
3. A standalone rule-based HU gate for cystic-versus-solid lesion routing.
4. Separately trained 3D FPN/class-LE classifiers for Tasks 2–4.

The 2D DenseNet-121 arms are sensitivity analyses only. They are not deployed models and must not be used to describe the primary pipeline.

This repository is for research and retrospective reproduction. It is not a medical device and must not be used to guide clinical care.

## Repository map

- `ARCHITECTURE.md`: verified component map and dimensionality evidence.
- `MODEL_CARD.md`: intended use, limitations, and safety framing.
- `ENVIRONMENT.md`: original and reference software environments.
- `paper_configs/eclinm_v1/pipeline.yaml`: release configuration.
- `paper_configs/eclinm_v1/weights_manifest.csv`: checkpoint identities and SHA256 values.
- `RELEASE_SHA256SUMS.txt`: SHA256 values for the V3 release documents and configurations.
- `cystic_solid_gating_gate_v1.0/`: formal standalone HU-rule gating implementation.
- `analysis/2D_sensitivity_modeling/`: 2D DenseNet-121 sensitivity/ablation modeling chain;
  not part of the deployed pipeline.
- `classification_tasks2_4/`: 3D FPN/class-LE implementation.
- `segmentation/`: LightM-UNet implementation and decoder-feature extraction.
- `paper_configs/eclinm_v1/pipeline.yaml`: the locked deployment configuration;
  the 2D sensitivity arms are specified there for provenance but are not part of
  the deployed pipeline.
- `examples/`: synthetic, data-free smoke examples.

Task 1 cohort-level performance values are not included in this public release.
Task 2 uses the unrounded deployment probability with the rule
`probability >= 0.586442888`.

## Gating model

The cystic–solid gate is a formal independent rule-based model, not a CNN checkpoint. It uses the 3D lesion ROI (LightM-UNet label 2) and voxel attenuation in HU:

```text
HU <= hu_threshold  -> cystic-density voxel
HU >  hu_threshold  -> solid-density voxel
cyst_fraction >= fraction_threshold -> cystic
otherwise -> solid
```

The default release settings are `hu_threshold=20.0 HU` and `fraction_threshold=0.5`. These values must match the author-confirmed deployment configuration when reproducing reported outputs. See the gating README and `paper_configs/eclinm_v1/gating_config.yaml`.

## Weights

Checkpoints are distributed separately through the project Google Drive folder listed in `paper_configs/eclinm_v1/WEIGHTS_DOWNLOAD.md`. Verify every downloaded file against `weights_manifest.csv` and the supplied checksum file. Imaging data and patient-level tables are not included.

## Public source identity

The author-confirmed deployment provenance is documented in `RELEASE_IDENTITY.md`.
The public release is pinned by the `v1.0-eclinm` tag, which identifies the exact
repository tree to cite as the deployment source in the manuscript and response
letter.

## Reproduction

Read `REPRODUCTION.md` before running any command. The release distinguishes
locked/deployed components from sensitivity-only analyses and does not claim
bit-identical reproduction under a newer CUDA stack. The code, configuration, formal
HU gate, and separately distributed locked weights define the complete computational
pipeline; restricted patient data and official result workbooks are required to
reproduce cohort-level numerical results.
