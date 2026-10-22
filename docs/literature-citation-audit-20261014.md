# Literature citation audit: sashimi-w — started 2026-10-14 UTC

Inventoried **179 active definitions**, including inherited/moved functions, private/nested helpers, constructors and tests. Updated **63 docstrings**, of which **63** receive or complete local paper references. The remaining edits clarify units, assumptions or NumPy structure.

The executable baseline before this citation edit is `8baf091948f8216cfad14998772e49053bd633b7`. Comparison baseline: author upstream/main `99dfc3632eec0126080c0273ddf78f84fe09216c`. Moving a definition does not make it new; this citation pass also includes inherited definitions.

The audit screens numerical literals and existing attributions, then matches adopted values to primary equations/tables. Generic algorithms, numerical controls, unit conversions and frozen regression outputs do not receive invented physical calibrations. The full [definition index and hash evidence](literature-citation-audit-20261014.json) distinguish edited definitions, unchanged references and archived evidence. An inventory entry is not proof of physical calibration.

## Findings and limits

- Viel thermal power uses the square of the density transfer (q10). Schneider c=2.5 sharp-k mass assignment is distinct from the top-hat R=8 Mpc/h normalization. References are also attached to independent test formulas.
- The Ludlow model uses C=650 and f=0.02. Its original profile calibration is Einasto-based; this code uses NFW enclosed masses and retains C. This is an adaptation, not a new calibration.
- Dekker Appendix C prints rounded tidal coefficients, e.g. -0.31, while exact author upstream/main 99dfc3632eec0126080c0273ddf78f84fe09216c uses -0.313 and additional precision in zeta. Both sources are identified; the retained coefficients are unchanged.
- Benson writes x=log(M/MJ) without an explicit base at that equation. The retained implementation uses natural log. This audit does not establish equality of those conventions; the ambiguity is documented in delc_Y11.
- The WMAP7-labelled stored CAMB header is an input record. A Komatsu citation does not recover the exact undocumented original CAMB configuration.

## Verification

The dependency/CI lock advances from ITAMAE `c452c5bbbbf8276870a8ff1f61f77d8bff0d8d7c` to its citation-only revision `4aeb13ed4e3e3e87171fe800e08f74b95c1244a2`. Its executable AST is unchanged.

- All 63 changed docstrings parse as NumPy style, cover explicit arguments and contain their assigned paper identifiers.
- All 30 inventoried tracked Python files have identical executable ASTs after removing only docstrings; assertions, tolerances and equations are unchanged. 2 historical Python files remain byte-identical.
- 134 distinct existing local pytest cases pass on Python 3.11.15. This is source-checkout testing, not a new scientific catalogue or a new calibration.
- The repository CI Ruff scope passes.
- Historical numerical results keep their original calculation revisions. The docstring edit changes source bytes and build identity; executable equivalence does not permit relabeling old outputs. No tests are deleted or weakened.

## Changed definitions and paper mapping

| Definition | References / change |
| --- | --- |
| [`cases`](../scripts/validate_eps_backport.py#L29) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691) |
| [`WDMHostHistory.mass_virial`](../src/sashimi_w/_itamae_components.py#L68) | [yang](https://arxiv.org/abs/1104.1757), [correa](https://arxiv.org/abs/1502.00391) |
| [`WDMTidalMassLoss`](../src/sashimi_w/_itamae_components.py#L119) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMTidalMassLoss.rhs`](../src/sashimi_w/_itamae_components.py#L159) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMTidalSolver`](../src/sashimi_w/_itamae_components.py#L223) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMTidalSolver.__init__`](../src/sashimi_w/_itamae_components.py#L263) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMTidalSolver.exponent`](../src/sashimi_w/_itamae_components.py#L373) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMTidalSolver.rate`](../src/sashimi_w/_itamae_components.py#L414) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMInitialStructure.initialize`](../src/sashimi_w/_itamae_components.py#L620) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`WDMProfileEvolution.evolve`](../src/sashimi_w/_itamae_components.py#L716) | [tracks](https://arxiv.org/abs/1002.3376), [structure](https://arxiv.org/abs/1803.07691), [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`WDMSurvival`](../src/sashimi_w/_itamae_components.py#L814) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`WDMSurvival.select`](../src/sashimi_w/_itamae_components.py#L847) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`Subhalos`](../src/sashimi_w/_itamae_migration.py#L24) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos.__init__`](../src/sashimi_w/_itamae_migration.py#L62) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos.transfer_amplitude`](../src/sashimi_w/_itamae_migration.py#L214) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos.power_ratio`](../src/sashimi_w/_itamae_migration.py#L247) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos._wdm_power_ratio`](../src/sashimi_w/_itamae_migration.py#L282) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos.half_mode_wavenumber`](../src/sashimi_w/_itamae_migration.py#L320) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`Subhalos.growthD`](../src/sashimi_w/_itamae_migration.py#L492) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`Subhalos.dDdz`](../src/sashimi_w/_itamae_migration.py#L553) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`Subhalos.conc200`](../src/sashimi_w/_itamae_migration.py#L730) | [formation](https://arxiv.org/abs/1601.02624), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`Subhalos.linear_growth_factor`](../src/sashimi_w/_itamae_migration.py#L777) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`Subhalos.rs_rhos_catalog_calc`](../src/sashimi_w/_itamae_migration.py#L866) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`Subhalos.rs_rhos_calc`](../src/sashimi_w/_itamae_migration.py#L965) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`Subhalos._execute_population`](../src/sashimi_w/_itamae_migration.py#L1024) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`make_variance_model`](../src/sashimi_w/_itamae_variance.py#L71) | [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`make_integrated_variance_model`](../src/sashimi_w/_itamae_variance.py#L118) | [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`make_integrated_variance_model.wdm_power_ratio`](../src/sashimi_w/_itamae_variance.py#L198) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`make_integrated_variance_model.normalized_wdm_power_ratio`](../src/sashimi_w/_itamae_variance.py#L245) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`_normalized_eps_kernel`](../src/sashimi_w/_physics.py#L95) | [yang](https://arxiv.org/abs/1104.1757) |
| [`WDMPhysics`](../src/sashimi_w/_physics.py#L176) | [viel](https://arxiv.org/abs/astro-ph/0501562), [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf), [wmap](https://arxiv.org/abs/1001.4538) |
| [`WDMPhysics.__init__`](../src/sashimi_w/_physics.py#L213) | [viel](https://arxiv.org/abs/astro-ph/0501562), [sharp](https://doi.org/10.1093/mnras/stv1169), [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf), [wmap](https://arxiv.org/abs/1001.4538) |
| [`WDMPhysics._power_normalization`](../src/sashimi_w/_physics.py#L270) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`WDMPhysics.SigmaIntegrand`](../src/sashimi_w/_physics.py#L334) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`WDMPhysics.conc200`](../src/sashimi_w/_physics.py#L413) | [formation](https://arxiv.org/abs/1601.02624), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`WDMPhysics._real_formation_trials`](../src/sashimi_w/_physics.py#L542) | [formation](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics.fc`](../src/sashimi_w/_physics.py#L597) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`WDMPhysics.Mvir_from_M200`](../src/sashimi_w/_physics.py#L651) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`WDMPhysics.Mvir_from_M200_fit`](../src/sashimi_w/_physics.py#L696) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`WDMPhysics.Mvir_from_M200_fit.ffunc`](../src/sashimi_w/_physics.py#L735) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`WDMPhysics.Mvir_from_M200_fit.xfunc`](../src/sashimi_w/_physics.py#L767) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`WDMPhysics.xi`](../src/sashimi_w/_physics.py#L811) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics.Mzi`](../src/sashimi_w/_physics.py#L843) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics._host_history_parameters`](../src/sashimi_w/_physics.py#L885) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics._host_history_mass`](../src/sashimi_w/_physics.py#L932) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics.Mzzi`](../src/sashimi_w/_physics.py#L975) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics.Delc`](../src/sashimi_w/_physics.py#L1016) | [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`WDMPhysics.dMdz`](../src/sashimi_w/_physics.py#L1047) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`WDMPhysics.delc_Y11`](../src/sashimi_w/_physics.py#L1102) | [benson](https://arxiv.org/abs/1209.3018) |
| [`WDMPhysics.Ffunc_Yang`](../src/sashimi_w/_physics.py#L1179) | [yang](https://arxiv.org/abs/1104.1757) |
| [`WDMPhysics.Na_calc`](../src/sashimi_w/_physics.py#L1227) | [yang](https://arxiv.org/abs/1104.1757) |
| [`WDMPhysics.subhalo_distr`](../src/sashimi_w/_physics.py#L1324) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`WDMPhysics.N_sat`](../src/sashimi_w/_physics.py#L1440) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`WDMPhysics.N_sat_Vthres`](../src/sashimi_w/_physics.py#L1555) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [scatter](https://arxiv.org/abs/1101.2020), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`sigma8_normalization`](../src/sashimi_w/_power.py#L42) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_full_catalog_matches_independent_reference`](../tests/test_itamae_migration.py#L93) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`test_variance_factories_default_to_q10_and_reject_old_models`](../tests/test_itamae_variance.py#L66) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_variance_factories_default_to_q10_and_reject_old_models.OldModel`](../tests/test_itamae_variance.py#L92) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous`](../tests/test_itamae_variance.py#L142) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous.integrand`](../tests/test_itamae_variance.py#L174) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_standard_q10_integrated_variance_formula_and_identifier_are_pinned`](../tests/test_itamae_variance.py#L231) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_public_default_is_q10_and_old_constant_is_removed`](../tests/test_q10_policy.py#L34) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`population_model`](../tests/test_surviving_observables.py#L10) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |

## Bibliography

- **benson**: [Benson et al. (2013), Dark Matter Halo Merger Histories Beyond Cold Dark Matter: I - Methods and Application to Warm Dark Matter, Eqs. (7)-(10).](https://arxiv.org/abs/1209.3018)
- **constants**: [Mohr & Taylor (2005), CODATA recommended values of the fundamental physical constants: 2002, Table XXVI.](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf)
- **conversion**: [Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.](https://arxiv.org/abs/astro-ph/0203169)
- **correa**: [Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.](https://arxiv.org/abs/1502.00391)
- **formation**: [Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Section 4.3, Eqs. (6)-(7).](https://arxiv.org/abs/1601.02624)
- **growth**: [Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5.](https://doi.org/10.1146/annurev.aa.30.090192.002435)
- **ludlow**: [Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Appendix C.](https://arxiv.org/abs/1601.02624)
- **nfw**: [Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.](https://arxiv.org/abs/astro-ph/9611107)
- **scatter**: [Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.](https://arxiv.org/abs/1101.2020)
- **sharp**: [Schneider (2015), Structure formation with suppressed small-scale perturbations, Eqs. (10)-(12).](https://doi.org/10.1093/mnras/stv1169)
- **structure**: [Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).](https://arxiv.org/abs/1803.07691)
- **survival**: [Hayashi et al. (2003), The Structural Evolution of Substructure.](https://arxiv.org/abs/astro-ph/0203004)
- **time**: [Jiang & van den Bosch (2016), Statistics of Dark Matter Substructure: I. Model and Universal Fitting Functions, Eq. (2).](https://arxiv.org/abs/1403.6827)
- **tracks**: [Penarrubia et al. (2010), The impact of dark matter cusps and cores on the satellite galaxy population around spiral galaxies, Eq. (8), Fig. 6.](https://arxiv.org/abs/1002.3376)
- **viel**: [Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).](https://arxiv.org/abs/astro-ph/0501562)
- **virial**: [Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).](https://arxiv.org/abs/astro-ph/9710107)
- **wmap**: [Komatsu et al. (2011), Seven-Year Wilkinson Microwave Anisotropy Probe Observations: Cosmological Interpretation.](https://arxiv.org/abs/1001.4538)
- **wtidal**: [Dekker et al. (2022), Warm dark matter constraints using Milky-Way satellite observations and subhalo evolution modeling, Appendix C.](https://arxiv.org/abs/2111.13137)
- **yang**: [Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).](https://arxiv.org/abs/1104.1757)
