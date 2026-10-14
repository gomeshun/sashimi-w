# Test review for the WDM power correction

Author comparison baseline: upstream/main
`99dfc3632eec0126080c0273ddf78f84fe09216c`. Owner main baseline:
`685bcfde7bf3ac5805492b56070151ba0415a9ca`. This review concerns the approved
c=2.5 and shared top-hat amplitude correction, not empirical calibration.

No test is retired, consolidated or given a larger comparison tolerance.
New reference files identify the changed physical target; historical arrays
and sidecars remain unchanged. New helpers and modified checks document
their numerical setup, units and limitations in NumPy-style docstrings.

| Test | Decision and necessity | Passing evidence and limits |
| --- | --- | --- |
| [test_top_hat_zero_and_bessel_reference](../tests/test_power_prescription.py) | Retain; prevents cancellation and a wrong smoothing window. | Exact W(0)=1 and independent spherical-Bessel values within absolute 2e-12; floating-point window evaluation only. |
| [test_eps_power_has_independent_top_hat_sigma8](../tests/test_power_prescription.py) | Retain all three particle masses and two integration resolutions; these check the actual normalized EPS spectrum. | Independent Simpson/Bessel sigma8 within relative 2e-8; Gauss order 16/32 within 2e-12. Neither identity proves population calibration. |
| [test_invalid_normalization_cannot_create_a_physical_spectrum](../tests/test_power_prescription.py) | Retain positive control and all four rejection cases; avoids silent undefined or negative variance. | Valid input gives finite N; negative/zero power, unordered k and zero target fail. This is input protection, not an observational prior. |
| [test_variance_is_independent_sharp_k_integral](../tests/test_numerics.py) | Retain; update the analytic upper limit for the approved c=2.5 mass assignment. | Independent adaptive integral within the existing 2e-11 tolerance, with amplitude separately checked. |
| [test_full_catalog_independent_reference](../tests/test_numerics.py) | Retain all 0.5/2/5 keV full catalogues; select distinctly named independently generated references. | Original rtol=2e-8 and exact survival masks. This is reproduction with historical ODE controls, not a total integration-error bound. |
| [test_public_dispatcher_candidate_and_legacy_paths](../tests/test_picard_solver.py) | Retain Picard, LSODA and option-rejection coverage. Resolve the concentration lookup features explicitly when constructing the numerical reference. | Original relative gates 1e-3/2e-7 remain. DOP853 reference refinement must agree within 2e-8; LSODA limits its maximum redshift step to 2e-4. |
| [test_default_catalog_and_wrapper_forwarding_against_frozen_reference](../tests/test_picard_solver.py) | Retain default dispatch, no-fallback, all tuple columns and wrapper options; update reference selection only. | Original 1e-3 mass and 5e-3 other-column gates, exact survival. |

The helper `_wdm_host_breakpoints` locates nearest-mass switches and the entry
of finite future-formation trials. `_segmented_log_reference` restarts only
the integration, leaving the physical rate unchanged. The controls are needed
because an adaptive solver can step across a narrow concentration feature.
They do not establish the accuracy of the default ODE on every host. Removing
this feature-resolution check would again let nominally tiny local tolerances
stand in for measured integration convergence.

The ablation tool and historical reference controller support reproducibility.
The former records controlled physical changes; the latter's original bytes
and source hashes are retained as evidence. Neither has an invented physical
interpretation or grants a new license to author-maintained code.
