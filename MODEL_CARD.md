# Model card — KITE-CRM v1.0-eclinm

## Overview

KITE-CRM is a CT-based decision-support pathway for cystic renal masses. It
segments kidney and lesion on corticomedullary-phase CT with a 3D LightM-UNet,
screens kidneys for abnormality, gates lesions into cystic versus solid, and
produces a continuous malignancy risk estimate plus two secondary
characterisation outputs from a 3D feature-pyramid classifier.

Intended use is **research and retrospective evaluation**. This is not a medical
device and must not be used to guide clinical care.

| | |
|---|---|
| Version | `v1.0-eclinm` |
| Manuscript | eclinm-D-26-02033, eClinicalMedicine |
| Input | Single-phase (corticomedullary) contrast-enhanced abdominal CT, axial |
| Segmentation | LightM-UNet (Mamba, nnU-Net v2) — one network producing both the deployed masks/overlays and, via a decoder hook, the Tasks 2-4 feature pyramid |
| Outputs | kidney/lesion masks; P(abnormal); cystic/solid; P(malignant); P(ccRCC); P(high grade) |
| Architecture map | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Configuration | [`paper_configs/eclinm_v1/pipeline.yaml`](paper_configs/eclinm_v1/pipeline.yaml) |
| Weights and hashes | [`paper_configs/eclinm_v1/weights_manifest.csv`](paper_configs/eclinm_v1/weights_manifest.csv) |

Architecture summary: Task 1 uses the LightM-UNet kidney output as the input
to a MONAI 3D DenseNet-121. Tasks 2–4 use the LightM-UNet decoder-hook feature
shapes as input to the shared 3D FPN/class LE formulation. The two 2D
DenseNet-121 models are sensitivity-only comparators: Task 1 uses a central
three-slice kidney view and Task 2 uses the lesion-mask maximal axial section.

Available checkpoint files are distributed through the project Google Drive
folder: <https://drive.google.com/drive/folders/1-i-5Q1g9LVnywzEyLbbEYjtwvPIsbvb0?usp=drive_link>.
The accompanying manifest/checksum files identify each checkpoint and its
 status. Patient imaging data remain restricted and are not included. The
 cystic-solid gate is a formal standalone HU-rule model documented under
`cystic_solid_gating_gate_v1.0/`; it has no learned checkpoint.

## Tasks and reported performance

| Task | Question | Model | Status | Reported |
|---|---|---|---|---|
| 1 | kidney normal vs abnormal | MONAI `DenseNet121(spatial_dims=3)` | **locked/deployed** | metrics not included in this public release |
| 2 | lesion benign vs malignant | `class LE`, LightM-UNet decoder-hook features + 3D FPN | **locked/deployed** | deployment threshold `0.586442888` |
| 3 | ccRCC vs other malignant | `class LE`, LightM-UNet decoder-hook features + 3D FPN | **locked/deployed** | metrics not included in this public release |
| 4 | ccRCC high vs low grade | `class LE`, LightM-UNet decoder-hook features + 3D FPN | **locked/deployed** | metrics not included in this public release |

Tasks 3 and 4 are **secondary characterisation outputs, not stand-alone
management criteria**.

## Training and evaluation data

Development and internal validation at Changhai Hospital, Naval Medical
University. External test cohorts, the multi-reader study set,
and the prospective randomised cohort are described in Supplementary Table S1.
Cohort denominators per task are in Supplementary Table S6.

Imaging data are **not released** (participating-centre data-governance
agreements). Case-level predictions and the official result workbooks are not included
in this code-only release; the configuration files document the required model inputs.

## Model locking

After the final development phase, architecture, weights, preprocessing and operating
thresholds were frozen. The Task 2 operating threshold `0.586442888` was fixed
on the internal validation cohort before external testing and prospective
deployment. The reader study and prospective study used the same corresponding
checkpoint files as the locked/deployed release; their SHA256 hashes are
identical to the hashes in `paper_configs/eclinm_v1/weights_manifest.csv`.

> **Provenance.** The author-confirmed deployment source is hosted at
> <https://github.com/CHANGHAI-AILab/KITE-CRM-final>. The exact public commit/tag
> for this final release is the `v1.0-eclinm` tag documented in
> `RELEASE_IDENTITY.md`.
> The deployment hardware/build was
> author-confirmed as NVIDIA RTX 3090 ×2, driver 535.104.05, bare metal (no
> container), Ubuntu 20.04 / kernel 5.15.0 (see `ENVIRONMENT.md`). The
> separately registered cystic–solid gate is the standalone HU-rule model under
> `cystic_solid_gating_gate_v1.0/`; the documented release package supports
> end-to-end reproduction of the released deterministic pipeline.

The deployment rule is `probability >= 0.586442888`; probabilities are not
rounded before thresholding. This release does not infer an additional
threshold-selection rule.

## Limitations

- **Single phase.** Only the corticomedullary phase is consumed. Cross-phase
  enhancement dynamics are not modelled. Volumetric processing addresses spatial
  sampling within one phase; it does not recover temporal information across
  phases.
- **Gating.** The cystic–solid gate is a formal standalone HU-rule model
  (Software Registration No. 2024SR1383073) operating on a 3D lesion ROI and
  CMP voxel intensities. It has no learned checkpoint.
- **Deployment status.** The author-confirmed release description identifies
  Task 1, Task 2, Task 3 and Task 4, together with LightM-UNet, as
  `locked/deployed`. Historical source-directory names in the Drive package do
  not override this status.
- **One segmentation network, two uses.** A single LightM-UNet (Mamba, nnU-Net
  v2) forward pass produces the masks and overlays radiologists saw *and*, via a
  decoder hook, the 3D feature pyramid consumed by Tasks 2-4. The volumetric
  representation is reused, not recomputed by a second network. See
  `segmentation/README.md`.
- **Output-perturbation analysis is not contour robustness.** Lesion-level
  logits were perturbed with Gaussian noise (SD 0.05–0.25) to stress-test
  threshold stability. Manually edited contours were not available, so this is
  an output-stability sensitivity analysis only — not segmentation robustness,
  contour correction, or interactive-edit validation.
- **Follow-up is immature.** Reported longitudinal findings are short-horizon
  context and do not establish long-term oncologic safety or exclude
  undertreatment.
- **Grad-CAM is an attention aid**, not a validated biomarker; it was not used
  for threshold selection, recalibration, or decision rules.

## Clinical scope

This public code-only release does not define a clinical management pathway.
Clinical interpretation and management remain outside the scope of the released
code and must follow the study protocol and applicable clinical standards.

## Ethics

Retrospective phases were conducted under institutional review board approval
with waiver of informed consent where applicable; the prospective randomised
phase was conducted under its registered protocol. Details are in the
manuscript's Ethics and Registration statements.

