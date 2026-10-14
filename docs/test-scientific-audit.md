# Scientific test review: sashimi-w — 2026-10-14

Reviewed **51 test definitions** individually. This count is not the number of parametrized cases. The review proposes future changes; all test bodies, assertions and tolerances are retained.

Executable-code baseline before this documentation change: `619f6c9eee398992be20b9d48663943de481c172`.
Comparison baseline for added definitions: author upstream/main, `99dfc3632eec0126080c0273ddf78f84fe09216c`.

## Decisions

| Decision | Definitions |
| --- | ---: |
| Retain | 35 |
| Consolidate without losing conditions | 9 |
| Relax only incidental assertions | 6 |
| Retire or replace, conditionally | 1 |

A consolidation retains every independent scenario and positive/negative control. A relaxation removes an implementation constraint only after preserving the physical or provenance invariant. A retirement is conditional on the replacement/surviving coverage stated below; no candidate is an instruction to delete a test.

Exact equality remains appropriate for lossless serialization, unchanged input identity, deterministic weight factorization and algebraic contracts. A near-roundoff comparison with a frozen fixture protects reproduction of a specified calculation; it is not an independent physical accuracy requirement. Analytic solutions, independent quadrature, unit identities and controlled refinement have a separate scientific role.

## Validation of this documentation change

- NumPy docstrings parsed for 170 in-scope definitions; all explicit parameters are documented.
- All 25 tracked Python files have identical executable ASTs after removing only module/class/function docstrings. The test bodies, decorators and assertions are unchanged.
- Preserved reference and historical analysis files remain byte-for-byte unchanged.
- Python 3.11.15: 126 existing pytest cases passed, 0 failures/errors and 0 skips.
- Ruff checks pass; ITAMAE additionally passes its full format and mypy checks.
- The saved numerical sweeps/catalogues retain their original calculation revisions. This audit does not relabel them as runs of a later documentation commit.

## Prescription checks versus physical validation

The current combined sharp-k cutoff scale `2.5*(9*pi/2)**(1/3)` and sigma8 renormalization are explicit retained choices. Identifier/literal assertions pin that prescription; they do not independently justify it. Preserve direct integral, moving-cutoff derivative and small-positive variance-gap checks when simplifying prescription identity tests. The fixed-barrier-gap EPS normalization remains an approximation for a mass-dependent WDM barrier.

## Candidate index

| Test | Proposal | Reason and retained coverage |
| --- | --- | --- |
| [test_archive_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L212) | Consolidate without losing conditions | Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios. |
| [test_exact_checkout_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L241) | Consolidate without losing conditions | Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources. |
| [test_malformed_explicit_revision_does_not_silently_fall_back](../tests/test_build_source_identity.py#L269) | Consolidate without losing conditions | Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required. |
| [test_matching_explicit_archive_revision_is_allowed](../tests/test_build_source_identity.py#L297) | Consolidate without losing conditions | Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build. |
| [test_concentration_scalar_and_matrix_use_valid_background_inputs](../tests/test_concentration_domain.py#L33) | Consolidate without losing conditions | Consolidate scalar/vector/matrix cases with the concentration boundary test in migration coverage; retain shape-independent physical values. |
| [test_zero_limit_on_tiny_variance_gap_and_unrepresentable_exponent](../tests/test_eps_stability.py#L88) | Relax only incidental assertions | Keep analytic-limit checks, but extreme stress values may be separated from the validated physical-domain suite. |
| [test_catalog_records_numerical_backport](../tests/test_eps_stability.py#L451) | Consolidate without losing conditions | Consolidate with full catalogue metadata coverage; preserve distinguishable numerical prescriptions without requiring redundant label checks. |
| [test_standard_import_and_removed_mode](../tests/test_itamae_migration.py#L45) | Consolidate without losing conditions | Consolidate alias/default checks with q10 policy tests while preserving explicit old-mode rejection. |
| [test_variance_factories_default_to_q10_and_reject_old_models](../tests/test_itamae_variance.py#L65) | Consolidate without losing conditions | Consolidate with q10 policy tests while retaining rejection of an already-created incompatible model, not only a string option. |
| [test_callable_adapter_uses_same_continuous_derivative](../tests/test_itamae_variance.py#L101) | Retire or replace, conditionally | Retirement candidate: test_callable_variance_adapter_preserves_standard_model already checks derivative forwarding at several masses/redshifts; merge any unique z=2 coverage there. |
| [test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous](../tests/test_itamae_variance.py#L121) | Relax only incidental assertions | Retain numerical/unit checks, but move hard-coded filter_scale and current normalization to historical-prescription coverage; they are not an independent validation of the physical sharp-k or sigma8 convention. |
| [test_standard_q10_integrated_variance_formula_and_identifier_are_pinned](../tests/test_itamae_variance.py#L156) | Consolidate without losing conditions | Consolidate with the preceding normalization and q10 provenance tests; duplicated string checks add no independent scientific evidence and should not block an approved prescription correction. |
| [test_variance_and_derivative_use_one_continuous_integral](../tests/test_itamae_variance.py#L182) | Relax only incidental assertions | Keep derivative consistency; exact cache-object identity and absence of a named private attribute are implementation details that can be removed independently. |
| [test_public_default_is_q10_and_old_constant_is_removed](../tests/test_q10_policy.py#L34) | Relax only incidental assertions | Keep default and immutability safeguards; absence of an obsolete symbol is API policy that can be separated from scientific tests. |
| [test_mass_threshold_total_and_exact_cumulative_axis](../tests/test_surviving_observables.py#L118) | Relax only incidental assertions | Keep selection/axis consistency; requiring exactly 10000 plotting points is unnecessary for scientific counts and can be removed or made a UI/API test. |
| [test_velocity_selection_and_cumulative_current_velocity](../tests/test_surviving_observables.py#L159) | Relax only incidental assertions | Keep physical selection checks; the fixed 10000-point display grid is incidental and can be relaxed independently. |

## Individual assessment of every test

### tests/test_build_source_identity.py

#### 1. `test_archive_in_unrelated_repository_keeps_embedded_revision`

[Source](../tests/test_build_source_identity.py#L140) · **Retain** · Role: reproducibility.

**Purpose.** Keep a published catalogue attributable to the code inside its source archive.
**Pass.** An exported package inside an unrelated checkout resolves to its embedded source SHA rather than the enclosing repository SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one integration case per distinct build hook; otherwise a valid numerical result may cite the enclosing project's revision.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 2. `test_unidentified_export_rejects_enclosing_repository`

[Source](../tests/test_build_source_identity.py#L164) · **Retain** · Role: reproducibility.

**Purpose.** Reject an export whose scientific source revision cannot be established.
**Pass.** A source export lacking embedded identity raises RuntimeError instead of using the surrounding repository's SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This prevents false reproducibility claims; it is independent of physical accuracy.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 3. `test_exact_source_checkout_uses_own_revision`

[Source](../tests/test_build_source_identity.py#L190) · **Retain** · Role: reproducibility.

**Purpose.** Identify the actual source checkout used to produce a catalogue.
**Pass.** A package at its own Git root resolves to that checkout's exact revision.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an end-to-end provenance case for each packaging system.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 4. `test_archive_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L212) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent a build override from relabelling a preserved scientific source archive.
**Pass.** A supplied revision that conflicts with the archive's embedded SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 5. `test_exact_checkout_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L241) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent an environment override from relabelling the checked-out calculation.
**Pass.** A supplied revision different from the checkout's SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 6. `test_malformed_explicit_revision_does_not_silently_fall_back`

[Source](../tests/test_build_source_identity.py#L269) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Reject an unusable source identifier before a scientific package is built.
**Pass.** Malformed explicit revision values raise RuntimeError even when another source identity is available.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 7. `test_matching_explicit_archive_revision_is_allowed`

[Source](../tests/test_build_source_identity.py#L297) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Permit an explicitly confirmed archive identity without changing its provenance.
**Pass.** An explicit SHA equal to the embedded archive SHA is accepted and recorded unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

### tests/test_concentration_domain.py

#### 8. `test_nonreal_formation_trials_are_removed_before_growth`

[Source](../tests/test_concentration_domain.py#L9) · **Retain** · Role: numerical.

**Purpose.** Evaluate the WDM concentration relation only on real formation-redshift branches.
**Pass.** At z=0, trials at or below the density boundary are excluded; retained trials have the expected values and finite z>-1.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain to avoid feeding complex or undefined formation epochs into the growth factor.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 9. `test_concentration_scalar_and_matrix_use_valid_background_inputs`

[Source](../tests/test_concentration_domain.py#L33) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Give a physical concentration independently of scalar or matrix query form.
**Pass.** The 1e9-solar-mass, z=0.5 concentration is finite and positive and the one-element matrix query equals the scalar.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate scalar/vector/matrix cases with the concentration boundary test in migration coverage; retain shape-independent physical values.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_eps_stability.py

#### 10. `test_kernel_matches_independent_normalization_integral`

[Source](../tests/test_eps_stability.py#L29) · **Retain** · Role: numerical.

**Purpose.** Recover the normalized WDM EPS kernel through the zero-barrier-gap limit.
**Pass.** For gaps spanning zero and the small-gap branch boundary, the kernel agrees with independent definite quadrature to rtol=3e-14.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain direct normalization evidence; small-gap branching must not change physical accretion probability.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 11. `test_fixed_gap_kernel_integrates_to_one`

[Source](../tests/test_eps_stability.py#L58) · **Retain** · Role: numerical.

**Purpose.** Normalize conditional WDM accretion probability over its full variance support.
**Pass.** After mapping the infinite variance interval to [0,1], the integrated kernel equals one to rtol=2e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This independent measure normalization is stronger than comparing a few local kernel values.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 12. `test_zero_limit_on_tiny_variance_gap_and_unrepresentable_exponent`

[Source](../tests/test_eps_stability.py#L88) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Avoid undefined arithmetic in extreme normalized EPS limits.
**Pass.** A zero barrier gap at variance 1e-250 gives 5e249 to rtol=1e-13, while an overwhelming exponential suppression returns zero without invalid/divide/overflow errors.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep analytic-limit checks, but extreme stress values may be separated from the validated physical-domain suite.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 13. `test_active_domain_errors_are_explicit`

[Source](../tests/test_eps_stability.py#L109) · **Retain** · Role: numerical.

**Purpose.** Reject physically invalid active EPS barrier or variance gaps.
**Pass.** Negative barrier gap, nonpositive variance support or NaN raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Silently replacing these with zero would erase invalid population contributions.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 14. `test_variance_gap_matches_independent_interval_integral`

[Source](../tests/test_eps_stability.py#L219) · **Retain** · Role: numerical.

**Purpose.** Resolve the variance difference that controls WDM EPS accretion.
**Pass.** Four widely separated or nearly equal mass pairs give positive gaps matching independent interval quadrature to rtol=3e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because subtracting two saturated full variances can lose a physically positive interval.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 15. `test_saturated_tail_preserves_positive_gap_and_ordering`

[Source](../tests/test_eps_stability.py#L250) · **Retain** · Role: numerical.

**Purpose.** Keep low-mass EPS support even when full variances round to the same float.
**Pass.** The 100/200-solar-mass interval remains positive despite equal rounded full variances; equal masses give zero, reversed ordering fails and broadcasting agrees with scalar calls.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain this targeted cancellation regression without imposing a positive floor or clipping.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 16. `test_eps_shapes_support_and_finite_low_mass_tails`

[Source](../tests/test_eps_stability.py#L285) · **Retain** · Role: numerical.

**Purpose.** Preserve EPS support across host quadrature resolution and mass-grid representations.
**Pass.** Orders 1,4,64,200 give finite nonnegative, correctly shaped rates; matrix and vector mass queries agree and forbidden high-mass nodes have zero weight.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain independent physical support and shape checks at both degenerate and refined quadrature orders.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 17. `test_deterministic_rows_do_not_depend_on_redshift_grid`

[Source](../tests/test_eps_stability.py#L320) · **Retain** · Role: numerical.

**Purpose.** Keep a local accretion rate independent of unrelated later query epochs.
**Pass.** Appending extra redshifts leaves the original rates exactly unchanged for three host-scatter offsets.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because a query-grid-dependent host-scatter anchor changes the physical prediction.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 18. `test_exact_z1_deterministic_scatter_anchor`

[Source](../tests/test_eps_stability.py#L349) · **Retain** · Role: numerical.

**Purpose.** Use the specified z=1 host-history scatter anchor exactly.
**Pass.** The variance-gap upper masses match the independently evaluated z=1 anchored host formula to rtol=2e-15.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain the model-prescription check; observing an internal call is secondary to the independent host-mass formula.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 19. `test_active_zero_gap_is_not_silently_erased`

[Source](../tests/test_eps_stability.py#L419) · **Retain** · Role: numerical.

**Purpose.** Retain a finite accretion contribution when the collapse barrier gap is zero.
**Pass.** For an exactly zero barrier prescription, active rates remain finite and strictly positive.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This distinguishes a normalized zero-gap limit from an empty physical support.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 20. `test_catalog_records_numerical_backport`

[Source](../tests/test_eps_stability.py#L451) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Identify the EPS numerical prescription used in a saved WDM result.
**Pass.** Metadata record the exact-z=1 anchor and direct-variance-gap normalized-zero-limit implementation.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with full catalogue metadata coverage; preserve distinguishable numerical prescriptions without requiring redundant label checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 21. `test_wholly_suppressed_catalog_preserves_zero_population`

[Source](../tests/test_eps_stability.py#L475) · **Retain** · Role: numerical.

**Purpose.** Represent a physically empty WDM sample without NaN normalization.
**Pass.** The strongly suppressed 0.5-keV case has zero base/final weights and finite physical columns under strict floating-point handling.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain: a zero population is a legitimate result, unlike invalid active rates.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 22. `test_invalid_population_normalization_raises`

[Source](../tests/test_eps_stability.py#L506) · **Retain** · Role: numerical.

**Purpose.** Reject a catalogue whose EPS rates cannot define a nonnegative finite measure.
**Pass.** Injected negative, NaN or infinite accretion rates raise a normalization error.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain so numerical failures cannot masquerade as a small or empty physical population.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_migration.py

#### 23. `test_standard_import_and_removed_mode`

[Source](../tests/test_itamae_migration.py#L45) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Select the declared q10 WDM prescription through every supported import.
**Pass.** Aliases identify the standard model, q=10 is selected and obsolete physics_mode arguments fail.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate alias/default checks with q10 policy tests while preserving explicit old-mode rejection.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 24. `test_frozen_records_keep_original_mode_and_source_identity`

[Source](../tests/test_itamae_migration.py#L66) · **Retain** · Role: numerical.

**Purpose.** Keep old WDM catalogue fixtures interpretable after prescription changes.
**Pass.** Both historical fixture files retain their mode records, source SHAs and provenance schema.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain historical identity; it is evidence about reference origin, not physical accuracy.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 25. `test_full_catalog_matches_independent_reference`

[Source](../tests/test_itamae_migration.py#L89) · **Retain** · Role: numerical.

**Purpose.** Reproduce the preserved q10/odeint WDM population and its physical invariants.
**Pass.** Sixteen nodes match the independent reference with documented tolerances; survival uses c_t>0.77, weights are nonnegative, NFW masses reconstruct, and concentration weights normalize.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one complete historical regression plus independent invariants; 5e-10/5e-12 parity is not a required accuracy for different solvers or revised physics.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 26. `test_calculation_metadata_and_serialization`

[Source](../tests/test_itamae_migration.py#L148) · **Retain** · Role: numerical.

**Purpose.** Record the WDM particle, background, variance and solver choices with each saved population.
**Pass.** Metadata match the executed configuration and source identities, and all columns, final weights and metadata round-trip exactly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain provenance and lossless storage; fixed specification strings describe the current prescription rather than independently validating it.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 27. `test_tuple_is_only_a_format_conversion`

[Source](../tests/test_itamae_migration.py#L198) · **Retain** · Role: numerical.

**Purpose.** Keep legacy WDM tuples equivalent to the named catalogue without rerunning evolution.
**Pass.** Conversion from the supplied catalogue and back preserves all physical fields to rtol=3e-16 and final weights exactly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain unit-aware format equivalence; the tolerance is conversion roundoff, not physical model accuracy.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 28. `test_native_astropy_tuple_units`

[Source](../tests/test_itamae_migration.py#L234) · **Retain** · Role: numerical.

**Purpose.** Interpret historical WDM tuple units identically with native and Astropy backends.
**Pass.** Both backends produce equal canonical columns and weights from the same historical-unit tuple.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because WDM's legacy kpc and density scales differ greatly from canonical Mpc units.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 29. `test_growth_derivative_and_physical_mass_boundary`

[Source](../tests/test_itamae_migration.py#L271) · **Retain** · Role: numerical.

**Purpose.** Use a normalized, differentiably consistent WDM background and mass convention.
**Pass.** D(0)=1, dD/dz matches a finite difference to 3e-9, relative growth matches D ratios and scalar/vector concentration agrees.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain background identities; concentrate duplicate concentration-shape assertions into one focused test if simplifying.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 30. `test_constructor_errors`

[Source](../tests/test_itamae_migration.py#L308) · **Retain** · Role: numerical.

**Purpose.** Reject unsupported WDM particle masses, power conventions or cosmologies.
**Pass.** Zero/infinite particle mass, unknown convention and incompatible Omega_m or h raise ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because fitted WDM ingredients are not interchangeable across arbitrary cosmologies.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 31. `test_catalog_rejects_invalid_inputs_before_work`

[Source](../tests/test_itamae_migration.py#L350) · **Retain** · Role: numerical.

**Purpose.** Reject an unusable WDM population grid before physical execution starts.
**Pass.** Invalid host mass, step, epoch, node counts, mass bounds or profile flag raises an error without calling the population executor.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain early physical-domain and numerical-control validation.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_variance.py

#### 32. `test_callable_variance_adapter_preserves_standard_model`

[Source](../tests/test_itamae_variance.py#L33) · **Retain** · Role: numerical.

**Purpose.** Preserve the WDM variance and derivative through the callable ITAMAE interface.
**Pass.** Sigma and derivative equal the standard model, variance equals sigma squared, and metadata distinguish particle mass, q10 and half-mode convention.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain adapter behavior; repeated literal identifier substrings can be consolidated with provenance tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 33. `test_variance_factories_default_to_q10_and_reject_old_models`

[Source](../tests/test_itamae_variance.py#L65) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Prevent either variance factory from silently accepting the retired q5 prescription.
**Pass.** Both factories identify q10 defaults and reject a supplied q5-like model.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with q10 policy tests while retaining rejection of an already-created incompatible model, not only a string option.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 34. `test_callable_adapter_uses_same_continuous_derivative`

[Source](../tests/test_itamae_variance.py#L101) · **Retire or replace, conditionally** · Role: numerical.

**Purpose.** Forward the same WDM variance slope used by the population model.
**Pass.** The callable derivative equals the model derivative at two masses and z=2.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retirement candidate: test_callable_variance_adapter_preserves_standard_model already checks derivative forwarding at several masses/redshifts; merge any unique z=2 coverage there.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 35. `test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous`

[Source](../tests/test_itamae_variance.py#L121) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Reproduce the current WDM sharp-k normalization and continuous negative slope.
**Pass.** The implemented reference mass gives sigma_8, unit-converted integration bounds/density agree, and all tested derivatives are finite, distinct and negative.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain numerical/unit checks, but move hard-coded filter_scale and current normalization to historical-prescription coverage; they are not an independent validation of the physical sharp-k or sigma8 convention.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 36. `test_standard_q10_integrated_variance_formula_and_identifier_are_pinned`

[Source](../tests/test_itamae_variance.py#L156) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Keep the current WDM prescription label and normalization identifiable.
**Pass.** Integrated variance reproduces the current sigma_8 normalization and both adapters contain the specified q10/half-mode labels.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the preceding normalization and q10 provenance tests; duplicated string checks add no independent scientific evidence and should not block an approved prescription correction.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 37. `test_variance_and_derivative_use_one_continuous_integral`

[Source](../tests/test_itamae_variance.py#L182) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Use one continuous WDM density variance for both sigma and the EPS slope.
**Pass.** A centered mass finite difference agrees with dS/dM to rtol=4e-8 and callable/native sigma agrees; repeated queries retain the current cache behavior.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep derivative consistency; exact cache-object identity and absence of a named private attribute are implementation details that can be removed independently.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 38. `test_lighter_wdm_has_more_small_scale_power_suppression`

[Source](../tests/test_itamae_variance.py#L214) · **Retain** · Role: numerical.

**Purpose.** Recover the expected particle-mass ordering of thermal WDM suppression.
**Pass.** At 1e8 solar masses a 1-keV particle gives smaller sigma than 5 keV; at 1e14 solar masses they agree within 0.3 percent.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a qualitative physical limit that complements exact normalization checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 39. `test_variance_cache_binds_wdm_particle_mass_and_round_trips`

[Source](../tests/test_itamae_variance.py#L240) · **Retain** · Role: numerical.

**Purpose.** Avoid reusing variance data for a different WDM mass or physical prescription.
**Pass.** Changing particle mass or historical prescription changes the cache key; exact arrays reload and mismatched keys fail.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain physical cache identity; exact old-label reconstruction is historical compatibility evidence only.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 40. `test_integrated_power_is_a_snapshot_of_constructor_parameters`

[Source](../tests/test_itamae_variance.py#L310) · **Retain** · Role: numerical.

**Purpose.** Keep a saved variance object's spectrum fixed after the source model mutates.
**Pass.** Changing the model suppression scale leaves the constructed power values and identifier exactly unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain snapshot semantics so an identifier never describes a different spectrum later in a calculation.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_population_components.py

#### 41. `test_canonical_transport_matches_half_mass_solution_and_chunking`

[Source](../tests/test_population_components.py#L18) · **Retain** · Role: numerical.

**Purpose.** Transport an analytic half-mass WDM evolution through canonical structure stages.
**Pass.** The bound mass is half the input to 1e-7, radii use the correct units, NFW mass reconstruction agrees to 3e-13, and repeated batches preserve all node values.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because this isolates unit/node transport from the actual WDM stripping law.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_q10_policy.py

#### 42. `test_retired_q5_is_rejected_instead_of_reinterpreted`

[Source](../tests/test_q10_policy.py#L11) · **Retain** · Role: numerical.

**Purpose.** Prevent an old power-suppression choice from silently becoming q10.
**Pass.** The model and both variance factories reject published-q5 explicitly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain while old inputs exist; silent reinterpretation would mislabel a physical comparison.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 43. `test_public_default_is_q10_and_old_constant_is_removed`

[Source](../tests/test_q10_policy.py#L34) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Expose the adopted WDM power prescription as fixed model state.
**Pass.** The default reports q10/standard-t2-q10, the old constant is absent and prescription fields cannot be reassigned.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep default and immutability safeguards; absence of an obsolete symbol is API policy that can be separated from scientific tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 44. `test_transfer_and_power_are_distinct_and_both_half_modes_are_explicit`

[Source](../tests/test_q10_policy.py#L57) · **Retain** · Role: numerical.

**Purpose.** Distinguish transfer amplitude from power suppression and their half-mode scales.
**Pass.** P_ratio=T**2, both tend to one at k=0, the amplitude-half point has P_ratio=0.25, and the power-half point lies at smaller k.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain these algebraic/physical definitions; they detect the factor-of-two exponent mistake directly.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 45. `test_half_mode_threshold_must_be_a_real_scalar_between_zero_and_one`

[Source](../tests/test_q10_policy.py#L85) · **Retain** · Role: numerical.

**Purpose.** Require a meaningful suppression level for half-mode inversion.
**Pass.** Endpoints, outside-range values, NaN/infinity, arrays and Boolean thresholds are rejected.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a well-defined inversion domain; exact scalar-versus-length-one-array policy can be revisited separately.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_shared_picard.py

#### 46. `test_wdm_adapter_preserves_physical_rhs_and_matches_direct_reference`

[Source](../tests/test_shared_picard.py#L29) · **Retain** · Role: numerical.

**Purpose.** Feed the original WDM loss law into the shared logarithmic Picard solver.
**Pass.** At three epochs, reconstructed dm/dz agrees with the WDM RHS to 4e-15; Picard endpoints agree with DOP853 to 1e-3 with no fallback in this case.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain both the physical coefficient identity and numerical comparison; sharing an integrator alone does not prove the WDM adapter is correct.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 47. `test_catalog_default_and_explicit_ode_have_identical_accretion_weights`

[Source](../tests/test_shared_picard.py#L61) · **Retain** · Role: numerical.

**Purpose.** Change only tidal integration when selecting WDM Picard by default.
**Pass.** Initial columns and base/concentration weights equal historical odeint; final masses match DOP853 to 1e-3; metadata record Picard.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain controlled same-population comparison; this does not imply Picard is more accurate than the old ODE.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 48. `test_tuple_conversion_has_no_invented_solver_identity`

[Source](../tests/test_shared_picard.py#L90) · **Retain** · Role: numerical.

**Purpose.** Avoid assigning an unevidenced tidal solver to an imported legacy tuple.
**Pass.** Bare tuple conversion records unknown:tuple-conversion and no Picard diagnostics.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain honest provenance: a format conversion cannot infer how the supplied masses were calculated.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_surviving_observables.py

#### 49. `test_mass_distribution_uses_survivors_in_both_mass_displays`

[Source](../tests/test_surviving_observables.py#L85) · **Retain** · Role: numerical.

**Purpose.** Count surviving subhaloes whether the horizontal mass is accreted or current.
**Pass.** For full, partial, empty and single-node populations, integrating the histogram recovers survivor weight and preserves the requested profile-change option.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because displaying accretion mass must not reintroduce disrupted nodes.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 50. `test_mass_threshold_total_and_exact_cumulative_axis`

[Source](../tests/test_surviving_observables.py#L118) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Apply the same strict peak-mass cut to total and cumulative surviving counts.
**Pass.** Totals and every cumulative point equal direct weighted selections for several thresholds and survival patterns; the axis is increasing.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep selection/axis consistency; requiring exactly 10000 plotting points is unnecessary for scientific counts and can be removed or made a UI/API test.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 51. `test_velocity_selection_and_cumulative_current_velocity`

[Source](../tests/test_surviving_observables.py#L159) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Distinguish peak-velocity selection from current-velocity cumulative counts.
**Pass.** Totals use the requested peak/current threshold, cumulative values equal direct current-velocity selections and profile options are forwarded.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep physical selection checks; the fixed 10000-point display grid is incidental and can be relaxed independently.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.
