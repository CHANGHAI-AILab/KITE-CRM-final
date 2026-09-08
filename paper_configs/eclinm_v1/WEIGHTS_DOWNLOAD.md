# Weights download index

Checkpoint files are hosted outside GitHub in the project Google Drive folder:

<https://drive.google.com/drive/folders/1-i-5Q1g9LVnywzEyLbbEYjtwvPIsbvb0?usp=drive_link>

Download the complete `UPLOAD_READY` package and verify files with
`99_metadata/SHA256SUMS.txt` before use.

The author-confirmed deployment source is available in the GitHub repository at
<https://github.com/CHANGHAI-AILab/KITE-CRM-final>. The exact public release is
pinned by the `v1.0-eclinm` tag documented in `RELEASE_IDENTITY.md`. The locked/deployed model
set was frozen before public release. The reader study and prospective study used
the same corresponding checkpoint files; their SHA256 hashes are identical to
the `locked/deployed` entries below (checkpoint by checkpoint).
The separately registered cystic–solid gate is included as the standalone HU-rule
source under `cystic_solid_gating_gate_v1.0/` and has no learned checkpoint.

| Drive file | Role | Status | SHA256 |
|---|---|---|---|
| `seg_LightMUNet_weight/checkpoint_best.pth` | LightM-UNet segmentation; masks/overlays + decoder-hook features for Tasks 2–4 | locked/deployed | `92af277bff3778f32532bc09f502fc9f591ab7250535aae97124645460c2619a` |
| `seg_LightMUNet_weight/checkpoint_final.pth` | LightM-UNet final checkpoint | archived final | `a599457aee1f08a84b278fd8e05dd46ea4942f0776acc01eeea299682837add0` |
| `01_locked_deployed/task2_deployed_Gen2_best.pt` | Task 2 `class LE` Gen2 | locked/deployed | `f57e5e637af55ae421ca8d5ac69c35b1b6b90b0a1a8e87f55420a08e95075f58` |
| `02_locked_deployed/task1_best.pt` | Task 1 3D MONAI DenseNet-121 | locked/deployed | `c37a7ee5f1501aa6442984fc20fb26c21b5bfca16bcf83616b71a77255b9a9e3` |
| `02_locked_deployed/task3_best.pt` | Task 3 classifier；LightM-UNet hook + 3D FPN | locked/deployed | `0e92d3e2c5a655d013ce91fcbb3bb7543f41e368859a105ff7ee4047863496d7` |
| `02_locked_deployed/task4_best.pt` | Task 4 classifier；LightM-UNet hook + 3D FPN | locked/deployed | `4001630f3c7b90d6b8fc3f264c76560e2508d5ba1cb998fe0cd91f4bb6e4404d` |
| `03_sensitivity_2D/sens2D_task1_seed{0,1,2}.pt` | Task 1 2D central three-slice kidney-view sensitivity arm | sensitivity only | see checksum file |
| `03_sensitivity_2D/sens2D_task2_seed{0,1,2}.pt` | Task 2 2D lesion-mask maximal-section sensitivity arm | sensitivity only | see checksum file |

## Important compatibility notes

1. Tasks 2–4 share the same stated formulation: LightM-UNet (nnU-Net v2)
   decoder-hook features feeding a 3D FPN/class LE classifier. Task 1 uses
   the LightM-UNet kidney mask to form a volumetric clip for MONAI 3D
   DenseNet-121. The author-confirmed status of Tasks 1–4 is
   `locked/deployed`; historical source-directory names are retained only in
   lineage metadata, not as the current package paths.
2. The segmentation folder also includes `plans.json`, `dataset.json`,
   `dataset_fingerprint.json`, and the original training log; their SHA256
   values are recorded in that folder's `SHA256SUMS.txt`.
3. The cystic-solid gate sits between the LightM-UNet lesion mask and the
   downstream classification flow. It is a formal standalone HU-rule model
   (No. 2024SR1383073) with source code in
   `cystic_solid_gating_gate_v1.0/`; it has no learned checkpoint.
4. The official Task 1–Task 4 result workbooks and patient imaging data
   are separate from the weight package and are not implied to be downloadable
   from this link.

