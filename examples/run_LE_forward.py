#!/usr/bin/env python
"""Smoke test: instantiate the deployed classifier and run one forward pass.

    python examples/make_synthetic_example.py
    python examples/run_LE_forward.py examples/synthetic_case.npz

Prints the parameter count and the two output logits. Confirms that the
released `class LE` is a 3D network taking the documented decoder feature
pyramid, without requiring any patient data or trained weights.

Pass --weights <path> to load a checkpoint from
paper_configs/eclinm_v1/weights_manifest.csv instead of random initialisation.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "classification_tasks2_4"))
from model_LE import LE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = np.load(args.npz)
    feats = [torch.as_tensor(d[k], dtype=torch.float32)[None].to(args.device)
             for k in ("feat0", "feat1", "feat2")]
    for name, t in zip(("feat0", "feat1", "feat2"), feats):
        print("%-6s %s  (rank %d)" % (name, tuple(t.shape), t.dim()))

    net = LE().to(args.device)
    if args.weights:
        state = torch.load(args.weights, map_location=args.device)
        state = state.get("state_dict", state)
        net.load_state_dict(state)
        print("loaded", args.weights)
    net.eval()

    n_params = sum(p.numel() for p in net.parameters())
    n_conv3d = sum(1 for m in net.modules() if isinstance(m, torch.nn.Conv3d))
    n_conv2d = sum(1 for m in net.modules() if isinstance(m, torch.nn.Conv2d))
    print("parameters      %d" % n_params)
    print("Conv3d modules  %d" % n_conv3d)
    print("Conv2d modules  %d" % n_conv2d)

    with torch.no_grad():
        out = net(*feats)
        prob = torch.softmax(out, 1)[0, 1].item()
    print("logits          %s" % out[0].tolist())
    print("P(positive)     %.4f%s" % (prob, "" if args.weights else
                                      "   [random init — value is meaningless]"))


if __name__ == "__main__":
    main()
