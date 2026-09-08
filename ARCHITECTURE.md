# Architecture and dimensionality evidence

This document describes the author-confirmed locked/deployed KITE-CRM release. Weight
identities and SHA256 values are listed in
[`paper_configs/eclinm_v1/weights_manifest.csv`](paper_configs/eclinm_v1/weights_manifest.csv).

## Provenance and scope

The target public source is <https://github.com/CHANGHAI-AILab/KITE-CRM-final>.
LightM-UNet, Tasks 1–4, and the formal HU-rule gate are all `locked/deployed`.
Reader-study and prospective-study checkpoints have
the same SHA256 values as the corresponding locked/deployed checkpoints. The gate is
included as a standalone deterministic implementation in this repository. Code,
configuration, dependencies, and separately distributed locked weights define the
complete computational pipeline; restricted patient data and result workbooks are
still required for cohort-level numerical reproduction.

## Component map

| Component | Implementation | Dimensionality | Input and output | Status |
|---|---|---:|---|---|
| Segmentation | LightM-UNet, Mamba/SSM, nnU-Net v2 | 3D | CMP CT -> kidney and lesion masks plus decoder features | locked/deployed |
| Task 1 abnormality | MONAI `DenseNet121(spatial_dims=3)` | 3D | volumetric kidney clip -> abnormality probability | locked/deployed |
| Cystic-solid gate | deterministic HU rule, Software Registration No. 2024SR1383073 | 3D lesion-volume rule | lesion label 2 + CMP HU -> `cystic`, `solid`, or `none` | locked/deployed |
| Tasks 2–4 | `class LE`, top-down 3D FPN | 3D | LightM-UNet decoder features -> task probability | locked/deployed |
| 2D comparison arms | torchvision DenseNet-121 | 2D | central three-slice or maximal lesion section -> sensitivity result | sensitivity only |

The 2D models are not deployment models and are not used to describe the primary
pipeline. Task 1 uses the kidney mask/region to form a volumetric kidney clip; Tasks
2–4 consume the multi-scale decoder features from the same LightM-UNet forward pass.

```text
CMP CT -> LightM-UNet (3D)
          |-> kidney/lesion masks and overlays
          |-> decoder feature pyramid -> class LE (Tasks 2–4)
          `-> kidney clip -> MONAI 3D DenseNet-121 (Task 1)
```

## Evidence that the deployed classifiers are 3D

Task 1 is constructed as `DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)`.
Its first convolution has the weight tensor `(64, 1, 7, 7, 7)`, with a 3D kernel of
`7 x 7 x 7`, rather than
the four-dimensional `(64, 3, 7, 7)` kernel of a 2D ImageNet DenseNet.

Tasks 2–4 use `class LE`, whose spatial operators are `Conv3d`, `MaxPool3d`, and
`avg_pool3d`. The input is the LightM-UNet decoder feature pyramid, not an image or
an axial section:

| Feature | Shape | Archive key | `LE.forward` argument |
|---|---:|---|---|
| Bottleneck hook | `[256, 10, 16, 16]` | `feat1` | `feat0` |
| Decoder hook 1 | `[128, 20, 32, 32]` | `feat2` | `feat1` |
| Decoder hook 2 | `[64, 40, 64, 64]` | `feat3` | `feat2` |

The one-based archive keys are mapped explicitly for compatibility with the stored
feature files. The hook is captured before each decoder upsampling block, as documented
in `segmentation/README.md` and implemented in `LightMUNet.py`.

## Formal HU-rule gate

The gate uses LightM-UNet label `2` as a three-dimensional lesion ROI. Within that ROI,
voxels with `HU <= 20.0` are cystic-density and voxels with `HU > 20.0` are
solid-density. The cystic fraction is the number of cystic-density lesion voxels
divided by all lesion voxels. A fraction of at least `0.5` returns `cystic`; otherwise
the result is `solid`; an empty ROI returns `none`. There is no learned gating
checkpoint.

## 2D sensitivity analysis

The 2D arms were trained only as sensitivity analyses. Task 1 uses the mean of the
central three slices of the kidney clip. Task 2 uses the axial section with the largest
lesion-mask area. Tasks 3 and 4 had no separately trained 2D comparator in this release.
The complete 2D modeling chain is included under `analysis/2D_sensitivity_modeling/`.
These arms do not replace or define the 3D deployment architecture, and they were not
used to set any locked operating threshold.

## Separate task checkpoints

Tasks 2, 3, and 4 are separately trained two-class classifiers that share the `class LE`
formulation and upstream LightM-UNet feature source. They are not jointly optimized
heads. The earlier `LC`, `liange_v2`, nnU-Net v1, and unrelated Swin UNETR artefacts
are historical or non-deployed components and are not part of this release.

## Stated limitations

- Only the corticomedullary phase is consumed; cross-phase enhancement dynamics are
  not modeled.
- Volumetric processing addresses spatial sampling within one phase but does not
  recover temporal information across phases.
- Tasks 3 and 4 are secondary characterization outputs, not stand-alone management
  criteria.
