# ENVIRONMENT — software and hardware versions

This file records **two distinct environments** and keeps them separate on
purpose:

1. **Original training/deployment environment** — the stack the locked models
   were actually trained and run under. Values here are recovered from the
   surviving conda environment that produced the deployed LightM-UNet weights
   and from the training artefacts on disk. Where a value is *not* recoverable
   from an artefact, it is identified as unrecovered rather than presented as a
   verified deployment fact. The deployment provenance items below have been
   author-confirmed.
2. **Reference reproduction environment** — a newer, clean stack under which the
   released reproduction code (`requirements.txt`) has been exercised. This is
   **not** the original training stack; a reproduction under it will not be
   bit-identical to the locked artefacts.

Reviewer-facing point: the primary reported numbers come from the **locked
model artefacts** (probabilities stored at deployment time), not from a re-run
under either environment. See `ARCHITECTURE.md` and
the release configuration and the source files in this repository.

### Final analysis scope

The final analysis denominators are fixed as follows:

- Task 1 final validation analysis: `n=469`.
- Task 2 final validation analysis: `n=235`.
- The release metadata and 2D sensitivity documentation use the full Task 2
  validation analysis as the final scope.

---

## 1. Original training / deployment environment

### 1.1 Deployed segmentation — LightM-UNet (Mamba, nnU-Net v2)

This is the network that produced the masks and overlays radiologists saw and,
via the decoder hook, the feature pyramid for Tasks 2–4 (weight SHA256
`92af277b…`). Versions recovered from the environment that trains and
runs this backbone:

| Component | Version | How recovered |
|---|---|---|
| Python | 3.10.14 | conda env interpreter |
| PyTorch | **2.0.1+cu117** | `torch.__version__` |
| CUDA (torch build) | **11.7** | `torch.version.cuda` |
| torchvision | 0.15.2+cu117 | `torchvision.__version__` |
| MONAI | 1.3.0 | `monai.__version__` |
| nnU-Net (v2) | **2.1.1** | `importlib.metadata.version("nnunetv2")` |
| mamba-ssm | **1.1.4** | package metadata |
| causal-conv1d | **1.1.1** | package metadata |
| dynamic-network-architectures | 0.3.1 | package metadata |
| batchgenerators | 0.25 | package metadata |
| acvl-utils | 0.2 | package metadata |
| einops | 0.8.0 | package metadata |
| numpy | 1.26.4 | package metadata |
| SimpleITK | 2.3.1 | package metadata |
| pandas | 2.2.2 | package metadata |
| scikit-learn | 1.4.2 | package metadata |
| scipy | 1.13.0 | package metadata |

`mamba-ssm` and `causal-conv1d` are required to load and run the LightM-UNet
backbone; without a matching CUDA toolchain the Mamba layers will not import.

### 1.2 Tasks 2–4 classifier — unified `class LE` 3D FPN

The Tasks 2–4 classifier tree lives alongside the LightM-UNet code and consumes
the same LightM-UNet decoder-hooked `.npz` features. All three tasks use the
same unified LightM-UNet + nnU-Net v2 decoder-hook + 3D FPN formulation and run
under the **same** environment as §1.1 (torch
2.0.1+cu117, numpy 1.26.4). No SE attention; `linear1 10240→1024`; 10,569,666
trainable parameters (`state_dict` numel 10,570,634); five-class variant
The five-class pathology variant is described in `classification_tasks2_4/model_LE.py`.

### 1.3 Task 1 abnormality — MONAI 3D DenseNet-121

`monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)`.
Runs under MONAI 1.3.0 (as §1.1). The released Task-1 checkpoint is a
**locked/deployed** weight (`task1_ckpt/best.pt`, packaged as
`02_locked_deployed/task1_best.pt`; older source-directory names are retained
only in lineage metadata). The original selection rule is not
separately pinned in the surviving artefacts.

### 1.4 Early-development segmentation — nnU-Net **v1** (Task016 / Task017)

Earlier development models are standard nnU-Net **v1**
`generic_UNet`, trained before the LightM-UNet switch. They are **not** part of
the locked pipeline and are not in the released repository (see
They are not required to run this release and are not included in the public tree.
Their nnU-Net v1 stack (Python 3.7 / torch 1.12-era) is **not** required to run
anything in this repository.

### 1.5 Hardware, container, and build

| Item | Status |
|---|---|
| GPU model and count | NVIDIA RTX 3090 ×2 (author-confirmed) |
| GPU driver version | 535.104.05 (author-confirmed) |
| Container image (tag/digest) | Bare metal; no container (author-confirmed) |
| Original deployment repository | [KITE-CRM-final](https://github.com/CHANGHAI-AILab/KITE-CRM-final) — complete author-confirmed deployment code for the releasable KITE-CRM components |
| Core-code baseline before the release commit | `ca3648a702efb762904f0cc1c603f6df69902333` |
| Corrected final release commit/tag | Git tag `v1.0-eclinm` identifies the exact public release tree |
| Reader/prospective study weights | Same SHA256 hashes as the corresponding `locked/deployed` checkpoints in `paper_configs/eclinm_v1/weights_manifest.csv` (author-confirmed) |
| Operating system / kernel | Ubuntu 20.04 / kernel 5.15.0 (author-confirmed) |

The nnU-Net training logs contain nnU-Net's built-in DKFZ cluster host strings
(`e230-dgxa100-*`, A100 SXM4) — these are **library boilerplate**, not evidence
of the hardware used here. The RTX 3090 ×2, driver, bare-metal deployment, and
Ubuntu/kernel values above are author-confirmed. The original deployment
repository completeness, commit, and reader/prospective checkpoint identity are
also author-confirmed above.

---

## 2. Reference reproduction environment

The released reproduction code (`requirements.txt`) has been exercised under a
newer clean stack. This is intentionally more recent than §1 and is **not**
byte-compatible with the locked artefacts.

| Component | Version |
|---|---|
| Python | 3.9.19 |
| PyTorch | 2.8.0+cu128 |
| CUDA (torch build) | 12.8 |
| torchvision | 0.23.0 |
| MONAI | 1.3.0 |
| numpy | 1.26.4 |
| SimpleITK | 2.3.1 |
| nibabel | 5.2.1 |
| pandas | 2.2.2 |
| scipy | 1.13.1 |
| scikit-learn | 1.5.1 |
| PyYAML | 6.0.1 |
| openpyxl | 3.1.5 |
| tqdm | 4.66.4 |

The 2D architecture-sensitivity arms were trained under this reproduction stack
(torch 2.8.0+cu128). Their modeling scripts are included under
`analysis/2D_sensitivity_modeling/`; patient data, official split/label tables,
feature pools, checkpoints, and result workbooks remain restricted.

---

## 3. Why the two environments differ

The original deployment and the reference reproduction stack are distinct.
LightM-UNet's Mamba kernels (`mamba-ssm==1.1.4`, `causal-conv1d==1.1.1`) are
pinned to a CUDA 11.7 / torch 2.0.1 toolchain; the reproduction stack (torch
2.8.0 / CUDA 12.8) is what the released classifier and Task-1 code were verified
under. When the goal is to reproduce the **locked segmentation weights** rather
than the downstream classifiers, use the §1.1 stack; the Mamba packages will not
build against arbitrary newer CUDA toolchains without matching versions.

---

## 4. One-line recovery command

To re-verify the §1.1 versions against the environment that runs the deployed
backbone:

```bash
python - <<'PY'
import sys, importlib, importlib.metadata as md
print("python", sys.version.split()[0])
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
for p in ["torchvision","monai","nnunetv2","mamba-ssm","causal-conv1d",
          "dynamic-network-architectures","batchgenerators","numpy","SimpleITK"]:
    try: print(p, md.version(p))
    except Exception: pass
PY
```
