# Release notes — v1.0-eclinm

## Included

- English release documentation.
- LightM-UNet 3D segmentation and decoder-hook feature extraction.
- MONAI 3D DenseNet-121 Task 1 implementation.
- 3D FPN/class-LE Tasks 2–4 implementation.
- Formal standalone HU-rule cystic–solid gating implementation.
- Complete 2D DenseNet-121 sensitivity/ablation modeling chain under
  `analysis/2D_sensitivity_modeling/`.
- Locked/deployed and sensitivity-only status in the weight manifest.
- Public-source identity pinned by the `v1.0-eclinm` release tag.

## Explicit non-claims

- The 2D DenseNet arms are not deployed.
- The 2D sensitivity scripts do not ship patient data, checkpoints, or official
  split/label tables; they require the corresponding restricted inputs and adapters.
- The HU gate is not a learned checkpoint and has no `.pt` weight file.
- A single CMP phase does not model cross-phase enhancement dynamics.
- The local package does not include patient imaging or official clinical workbooks.
- The `v1.0-eclinm` tag pins the exact public release tree; an unrelated
  `main` update does not change the cited release.
