# Public upload checklist

Use this directory as the repository root when updating
`https://github.com/CHANGHAI-AILab/KITE-CRM-final`.

1. Upload the complete directory, including
   `analysis/2D_sensitivity_modeling/`, without adding patient images, masks,
   result workbooks, or checkpoint binaries.
2. Preserve the final analysis scope in the release documents: Task 1 `n=469`
   and Task 2 `n=235`; use the full Task 2 validation analysis as the final
   scope.
3. Create the single corrected release commit and the selected public tag (for
   example, `v1.0-eclinm`). Do not create separate metadata-only commits.
4. Verify the public tree while signed out or from an unauthenticated request.
   Confirm that `cystic_solid_gating_gate_v1.0/`,
   `analysis/2D_sensitivity_modeling/`, the manifest, and all English
   documentation are present.
5. Record the final verified commit SHA consistently in `RELEASE_IDENTITY.md`,
   `ENVIRONMENT.md`, and the response-letter metadata.
6. Confirm that the public tree and the external `UPLOAD_READY` weight package
   use the same component names, deployment status, and SHA256 values.
7. Verify the V3 release documents and configurations against
   `RELEASE_SHA256SUMS.txt`.
8. Only after these checks cite the commit or tag as the deployment source in the
   manuscript and response letter.

The source baseline before the single release commit is
`ca3648a702efb762904f0cc1c603f6df69902333`. Cite the final verified commit
only after this exact tree is pushed and independently re-verified anonymously.
