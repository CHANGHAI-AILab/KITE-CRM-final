# Release identity

- Release: `v1.0-eclinm`
- Locked/deployed components: LightM-UNet, Task 1, Task 2, Task 3, Task 4, and the formal standalone HU-rule gate.
- Reader-study and prospective-study checkpoints use the same SHA256 values as the corresponding locked/deployed checkpoints.
- Target repository: <https://github.com/CHANGHAI-AILab/KITE-CRM-final>

## Public commit status

This release tree was prepared from the core-code baseline
`ca3648a702efb762904f0cc1c603f6df69902333`. The public release is pinned by
the Git tag `v1.0-eclinm`, which identifies this exact release tree.

Final corrected release reference: **`v1.0-eclinm`**.
Final tag: **`v1.0-eclinm`**.

Do not cite a commit copied from another repository as the release commit. The
release commit must contain this exact tree, including
`analysis/2D_sensitivity_modeling/`, `cystic_solid_gating_gate_v1.0/`, the model
manifests, and the English documentation.

## Final analysis scope

- Task 1 final validation analysis: `n=469`.
- Task 2 final validation analysis: `n=235`.
- The public release uses the full Task 2 validation analysis as its final scope.

## Upload checklist

1. Upload this directory as the repository root.
2. The single corrected release commit is tagged `v1.0-eclinm`.
3. The public tree was verified without authentication.
4. Use the `v1.0-eclinm` release reference consistently in the response letter and release metadata.
5. Confirm that the GitHub tree and the weight manifest use identical component names and SHA256 values.
