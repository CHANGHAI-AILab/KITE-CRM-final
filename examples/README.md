# examples/

Patient CT volumes, masks and derived feature archives cannot be redistributed
(participating-centre data-governance agreements — see README § Availability).
What is here instead is enough to verify the released architecture without any
patient data.

```bash
python examples/make_synthetic_example.py
python examples/run_LE_forward.py examples/synthetic_case.npz
```

Expected output:

```
feat0  (1, 256, 10, 16, 16)  (rank 5)
feat1  (1, 128, 20, 32, 32)  (rank 5)
feat2  (1, 64, 40, 64, 64)   (rank 5)
parameters      10569666
Conv3d modules  8
Conv2d modules  0
```

Three rank-5 input tensors, 10,569,666 trainable parameters (Gen2 `class LE`, no
SE; the `state_dict` numel including BatchNorm buffers is 10,570,634), eight 3D
convolution modules and zero 2D convolution modules. Add `--weights <path>` to
load a checkpoint from `paper_configs/eclinm_v1/weights_manifest.csv`.

The same check for Task 1 needs no script — the first convolution kernel of the
released checkpoint is a 5-D tensor:

```python
>>> torch.load("best.pt", map_location="cpu")["features.conv0.weight"].shape
torch.Size([64, 1, 7, 7, 7])
```

A 2D DenseNet-121 has `conv0.weight` of shape `(64, 3, 7, 7)`. The released
weights will not load into a 2D model.

## Not carried over from the previous snapshot

The earlier public snapshot shipped 37 `.npz` archives under
`classification/dataset/dataset/`. They are excluded here for two reasons:

1. Each contains a single `image` array of shape `(32, 32, 32)` — input for the
   toy `LC` head, not the deployed `LE` feature pyramid. Shipping them as
   "examples" would misrepresent what the deployed model consumes.
2. Their filenames are hospital identifiers plus examination-specific fields
   (`<case_id>_<side>.npz`), which is not adequate de-identification for
   public release.
