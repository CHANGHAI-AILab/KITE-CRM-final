#!/usr/bin/env python
"""Generate a shape-correct synthetic example for the Tasks 2-4 classifier.

Patient imaging cannot be redistributed (see README section Availability), so
this script writes a random-noise archive with exactly the tensor shapes and
dtypes that `class LE` expects. It lets a reader confirm that the released
architecture instantiates, accepts the documented input, and produces a
2-logit output — without any patient data.

    python examples/make_synthetic_example.py
    python examples/run_LE_forward.py examples/synthetic_case.npz

Note: the previously published snapshot shipped .npz archives containing a
single `image` array of shape (32, 32, 32). Those were inputs for the toy `LC`
head, not for the deployed `LE` model, and are not carried over here.
"""
import argparse
import os

import numpy as np

# (channels, depth, height, width) — LightM-UNet 3d_fullres decoder hook, patch 80x128x128
SHAPES = {
    "feat0": (256, 10, 16, 16),
    "feat1": (128, 20, 32, 32),
    "feat2": (64, 40, 64, 64),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "synthetic_case.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Post-ReLU decoder activations are non-negative; half-normal keeps the
    # synthetic tensors in a plausible range rather than symmetric about zero.
    arrays = {k: np.abs(rng.standard_normal(s)).astype(np.float32)
              for k, s in SHAPES.items()}
    arrays["label"] = np.array([0], dtype=np.int64)

    np.savez(args.out, **arrays)
    print("wrote", args.out)
    for k, v in arrays.items():
        print("  %-6s %-22s %s" % (k, v.shape, v.dtype))


if __name__ == "__main__":
    main()
