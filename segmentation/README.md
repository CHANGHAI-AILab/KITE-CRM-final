# Segmentation and decoder-feature extraction

The locked pipeline contains one segmentation network: LightM-UNet, a three-dimensional
Mamba/state-space U-Net implemented within nnU-Net v2. One forward pass produces the
kidney and lesion masks used for overlays and also exposes the decoder feature pyramid
consumed by the Tasks 2–4 classifiers.

```text
CMP CT -> LightM-UNet (nnU-Net v2, 3D)
          |-> kidney/lesion masks and reader overlays
          `-> decoder features feat0/feat1/feat2 -> class LE (Tasks 2–4)
```

The downstream classifiers therefore reuse a volumetric representation computed by the
segmentation network. They do not select a representative axial slice and do not run a
second segmentation network.

## LightM-UNet implementation

| Item | Release value |
|---|---|
| Trainer | `nnUNetTrainerLightMUNet` |
| Configuration | `3d_fullres`, fold 0, patch `80 x 128 x 128` |
| Implementation | `LightM-UNet-master/nets/LightMUNet.py` |
| Construction | `LightMUNet(spatial_dims=3, init_filters=32, in_channels=1, out_channels=3)` |
| Labels | `0=background`, `1=kidney`, `2=lesion/tumor` |
| Segmentation checkpoint | External `UPLOAD_READY` package; SHA256 is in the weight manifest |

The released implementation and manifest identify the segmentation component as
`locked/deployed`. The checkpoint is distributed separately because patient data and
large binary artefacts are not stored in this source repository.

## Decoder hook used by Tasks 2–4

The decoder hook is part of the same LightM-UNet forward pass. In
`nets/LightMUNet.py`, an ordered hook dictionary is created during initialization and
the decoder tensor is copied at the top of each decode iteration, before the upsampling
block. `feature_extraction/dump_decoder_features.py` writes the captured tensors for
the classifier data loader.

For the documented full-resolution patch, the measured tensors are:

| Decoder tensor | Shape | NPZ key | `class LE` argument |
|---|---:|---|---|
| Bottleneck (`hook[0]`) | `[256, 10, 16, 16]` | `feat1` | `feat0` |
| First decoder stage (`hook[1]`) | `[128, 20, 32, 32]` | `feat2` | `feat1` |
| Second decoder stage (`hook[2]`) | `[64, 40, 64, 64]` | `feat3` | `feat2` |

The one-based NPZ keys are retained for compatibility with archived feature files; the
data loader maps them explicitly to the zero-based `LE.forward(feat0, feat1, feat2)`
arguments.

## Relation to the manuscript

The manuscript descriptions "3D Mamba U-Net" and "Mamba U-Net backbone with a feature
pyramid" refer to this LightM-UNet implementation and its decoder outputs. Tasks 2–4
use the 3D FPN/class-LE formulation documented in the repository root. The 2D DenseNet
models belong only to the architecture-sensitivity analysis; they are not deployed.

## Excluded historical networks

Earlier nnU-Net v1 `generic_UNet` experiments and unrelated Swin UNETR artefacts are
not components of this locked release and are intentionally not included here.
