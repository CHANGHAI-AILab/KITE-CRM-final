#!/usr/bin/env python
"""
LightM-UNet decoder-feature dump — the 3D feature source consumed by Tasks 2-4.

Why this file exists
--------------------
Reviewer-facing point: Tasks 2-4 are NOT image classifiers. They consume the
multi-scale 3D decoder feature maps of the segmentation network, i.e. the
volumetric representation produced during segmentation is *reused*, not
discarded. The same LightM-UNet forward pass that produces the kidney/lesion
mask also produces the classifier's input.

The extraction is a two-line patch on LightM-UNet plus this dump step.

Patch 1 — nnunetv2/nets/LightMUNet.py
    __init__ (L213):   self.hook = OrderedDict()
    decode   (L284):   self.hook[i] = copy.deepcopy(x)   # top of the decoder
                                                         # loop, BEFORE up(x)

    def decode(self, x, down_x):
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            self.hook[i] = copy.deepcopy(x)              # <-- patch
            x = up(x) + down_x[i + 1]
            x = upl(x)

    Because the hook is taken at the *top* of the loop, hook[0] is the
    bottleneck tensor and hook[1], hook[2] are the outputs of the first and
    second decoder stages.

Patch 2 — nnunetv2/inference/predict_from_raw_data.py (L474-475), inside the
prediction loop once the logits for a case are complete: call
dump_decoder_features() below. (In the original working tree this was an inline
np.savez with a hard-coded output directory; it is refactored here into a
function with an explicit argument so the path is a configuration value, not a
source-code constant.)

Feature shapes (patch 80x128x128, 3d_fullres, init_filters=32), verified by
running a forward pass through LightMUNet and reading the tensor shapes:

    hook[0] -> npz "feat1" -> LE argument feat0   [256, 10, 16, 16]
    hook[1] -> npz "feat2" -> LE argument feat1   [128, 20, 32, 32]
    hook[2] -> npz "feat3" -> LE argument feat2   [ 64, 40, 64, 64]

Note the off-by-one between the npz key names and the `LE` argument names: the
archive keys are 1-based ("feat1".."feat3") while `LE.forward` takes
(feat0, feat1, feat2). The dataloader in classification_tasks2_4/ performs this
mapping explicitly; it is preserved here for byte-compatibility with the
archived .npz files.

Every tensor is 4-D (C, D, H, W). There is no 2-D stage anywhere in this path.

Weights: see paper_configs/eclinm_v1/weights_manifest.csv
    segmentation  nnUNetTrainerLightMUNet__nnUNetPlans__3d_fullres/fold0
"""
import os

import numpy as np

# decoder hook index -> name written into the .npz archive
HOOK_KEYS = {0: "feat1", 1: "feat2", 2: "feat3"}


def dump_decoder_features(network, case_id, out_dir, label=None,
                          overwrite=False):
    """Persist the hooked LightM-UNet decoder feature maps for one case.

    Parameters
    ----------
    network : LightMUNet
        Must expose ``network.hook`` populated by the LightMUNet patch above.
        When called from nnUNetPredictor this is ``self.network``.
    case_id : str
        De-identified case identifier, for example ``"case001_left"``. Used as the file stem.
    out_dir : str
        Destination directory for ``<case_id>.npz``.
    label : int or None
        Optional ground-truth label stored alongside the features so that the
        downstream `LE` dataloader needs no second lookup table.
    overwrite : bool
        Guard against silently clobbering a previous dump. The original code
        called ``sys.exit(1)`` on collision; raising is the same guard, minus the
        process kill.
    """
    if not hasattr(network, "hook"):
        raise RuntimeError(
            "network has no .hook — LightMUNet.py is unpatched; "
            "see Patch 1 in this file's docstring"
        )

    os.makedirs(out_dir, exist_ok=True)
    save_file = os.path.join(out_dir, case_id + ".npz")
    if os.path.exists(save_file) and not overwrite:
        raise FileExistsError(
            "refusing to overwrite %s — case ids must be unique" % save_file
        )

    arrays = {}
    for hook_idx, name in HOOK_KEYS.items():
        if hook_idx not in network.hook:
            raise KeyError(
                "hook[%d] missing; decoder recorded %d levels"
                % (hook_idx, len(network.hook))
            )
        arrays[name] = network.hook[hook_idx].cpu().numpy()

    if label is not None:
        arrays["label"] = label

    np.savez(save_file, **arrays)
    return save_file
