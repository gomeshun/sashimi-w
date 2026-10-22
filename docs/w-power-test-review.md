# Test review for the WDM power correction

Author comparison baseline: upstream/main
`99dfc3632eec0126080c0273ddf78f84fe09216c`. Owner migration baseline:
`62b923f81c690c369a97cca567bbe32f1e32d6ba`; pre-correction candidate:
`465db7a0f24def6ebe065718fe1f5f5fc543c048`.

This is an addendum to the earlier documentation-only audit. No test is
retired or consolidated and no comparison tolerance is enlarged. Earlier
retirement candidates remain proposals. Old fixtures remain byte-for-byte
unchanged; new filenames identify the approved physical prescription.

| Test | Decision and necessity | Passing evidence and limits |
| --- | --- | --- |
| [test_top_hat_zero_and_bessel_reference](../tests/test_power_prescription.py) | Retain; prevents cancellation and an incorrect smoothing window. | W(0)=1 and independent Bessel values within absolute 2e-12; numerical window evaluation only. |
| [test_eps_power_has_independent_top_hat_sigma8](../tests/test_power_prescription.py) | Retain all three masses and both quadratures; measures the actual canonical EPS power, including h conversion. | Independent Simpson/Bessel sigma8 within relative 2e-8 and Gauss order 16/32 within 2e-12. It does not calibrate abundances. |
| [test_invalid_normalization_cannot_create_a_physical_spectrum](../tests/test_power_prescription.py) | Retain valid control and all four invalid conditions; protects against silently undefined variance. | Finite N for valid input, ValueError for negative/zero power, unordered k or zero target. |
| [test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous](../tests/test_itamae_variance.py) | Retain unit/slope checks and add independent adaptive integrals for three halo masses. Replace only the old physical cutoff/normalization expectations. | c=2.5 integral agrees within relative 2e-11. New top-hat amplitude tests replace the obsolete assertion that sharp-k sigma(M8)=sigma8. |
| [test_standard_q10_integrated_variance_formula_and_identifier_are_pinned](../tests/test_itamae_variance.py) | Retain prescription/half-mode identity and explicit rejection of forced sharp-k normalization. | Both adapters identify v3; a different smoothing window has a different rms. This is provenance, not independent physical validation. |
| [test_variance_cache_binds_wdm_particle_mass_and_round_trips](../tests/test_itamae_variance.py) | Retain exact serialization and changed-prescription rejection. | v3 and a synthetic stale-v2 identifier produce different keys. The synthetic identifier is not claimed to reconstruct an old model. |
| [test_full_catalog_matches_independent_reference](../tests/test_itamae_migration.py) | Retain every column, weight, survival and NFW invariant. Select the independently patched c=2.5/top-hat reference. | Original 5e-10 catalogue and 5e-12 weight relative gates. Both sides use explicit ODE rtol=1e-11/atol=1e-13; independent tenfold refinement changes bound masses by at most 7.2e-11. |
| [test_calculation_metadata_and_serialization](../tests/test_itamae_migration.py) | Retain exact archive identity and add physical amplitude/mass-assignment fields. | N, sigma8 convention and c identify the spectrum used to calculate the catalogue. |

The original 51-test-definition audit describes its earlier executable
baseline; the three new test definitions above add eight parametrized cases.
The public active scientific notebook now selects the new B reference with
explicit solver controls. Archived q5/q10 convergence runs keep their original
source revisions and are labeled historical v2 evidence.

Independent quadrature, complete catalogue comparison, provenance and
serialization protect different failure modes. None is removed merely because
another check passes. A green suite does not establish grid convergence,
simulation calibration or new WDM constraints. In particular, the inherited
concentration lookup can create narrow features; the standalone solver review
records the additional controls needed to resolve them in its strict numerical
comparison. See [the physical prescription](w-power-prescription.md).
