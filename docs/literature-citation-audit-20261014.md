# Literature citation audit: sashimi-w-main — started 2026-10-14 UTC

Inventoried **134 active definitions**, including inherited/moved functions, private/nested helpers, constructors and tests. Updated **51 docstrings**, of which **47** receive or complete local paper references. The remaining edits clarify units, assumptions or NumPy structure.

The executable baseline before this citation edit is `345a8d02b3c64c552fb818e29df9acc2c441e520`. Comparison baseline: author upstream/main `99dfc3632eec0126080c0273ddf78f84fe09216c`. Moving a definition does not make it new; this citation pass also includes inherited definitions.

The audit screens numerical literals and existing attributions, then matches adopted values to primary equations/tables. Generic algorithms, numerical controls, unit conversions and frozen regression outputs do not receive invented physical calibrations. The full [definition index and hash evidence](literature-citation-audit-20261014.json) distinguish edited definitions, unchanged references and archived evidence. An inventory entry is not proof of physical calibration.

## Findings and limits

- Viel thermal power uses the square of the density transfer (q10). Schneider c=2.5 sharp-k mass assignment is distinct from the top-hat R=8 Mpc/h normalization. References are also attached to independent test formulas.
- The Ludlow model uses C=650 and f=0.02. Its original profile calibration is Einasto-based; this code uses NFW enclosed masses and retains C. This is an adaptation, not a new calibration.
- Dekker Appendix C prints rounded tidal coefficients, e.g. -0.31, while exact author upstream/main 99dfc3632eec0126080c0273ddf78f84fe09216c uses -0.313 and additional precision in zeta. Both sources are identified; the retained coefficients are unchanged.
- Benson writes x=log(M/MJ) without an explicit base at that equation. The retained implementation uses natural log. This audit does not establish equality of those conventions; the ambiguity is documented in delc_Y11.
- The WMAP7-labelled stored CAMB header is an input record. A Komatsu citation does not recover the exact undocumented original CAMB configuration.

## Verification

- All 51 changed docstrings parse as NumPy style, cover explicit arguments and contain their assigned paper identifiers.
- All 11 inventoried tracked Python files have identical executable ASTs after removing only docstrings; assertions, tolerances and equations are unchanged. 2 historical Python files remain byte-identical.
- 46 distinct existing local pytest cases pass on Python 3.11.15. This is source-checkout testing, not a new scientific catalogue or a new calibration.
- Historical numerical results keep their original calculation revisions. The docstring edit changes source bytes and build identity; executable equivalence does not permit relabeling old outputs. No tests are deleted or weakened.

## Changed definitions and paper mapping

| Definition | References / change |
| --- | --- |
| [`subhalos`](../sashimi_w.py#L85) | [viel](https://arxiv.org/abs/astro-ph/0501562), [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf), [wmap](https://arxiv.org/abs/1001.4538) |
| [`subhalos.__init__`](../sashimi_w.py#L118) | [viel](https://arxiv.org/abs/astro-ph/0501562), [sharp](https://doi.org/10.1093/mnras/stv1169), [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf), [wmap](https://arxiv.org/abs/1001.4538) |
| [`subhalos._prepare_variance`](../sashimi_w.py#L179) | [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`subhalos.power_ratio`](../sashimi_w.py#L225) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`subhalos.transfer_amplitude`](../sashimi_w.py#L255) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`subhalos.TopHat`](../sashimi_w.py#L289) | Units, calibration wording, or NumPy structure clarified. |
| [`subhalos.SigmaIntegrand`](../sashimi_w.py#L315) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`subhalos.integratePk_th`](../sashimi_w.py#L352) | Units, calibration wording, or NumPy structure clarified. |
| [`subhalos.linear_growth_factor`](../sashimi_w.py#L397) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`subhalos.conc200`](../sashimi_w.py#L444) | [formation](https://arxiv.org/abs/1601.02624), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`subhalos._real_formation_trials`](../sashimi_w.py#L534) | [formation](https://arxiv.org/abs/1601.02624) |
| [`subhalos.fc`](../sashimi_w.py#L595) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`subhalos.rhocrit`](../sashimi_w.py#L629) | Units, calibration wording, or NumPy structure clarified. |
| [`subhalos.Mvir_from_M200`](../sashimi_w.py#L652) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`subhalos._Mvir_from_M200_grid`](../sashimi_w.py#L695) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`subhalos.Mvir_from_M200_fit`](../sashimi_w.py#L736) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`subhalos.Mvir_from_M200_fit.ffunc`](../sashimi_w.py#L774) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`subhalos.Mvir_from_M200_fit.xfunc`](../sashimi_w.py#L805) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`subhalos.growthD`](../sashimi_w.py#L841) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`subhalos.xi`](../sashimi_w.py#L874) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`subhalos.dDdz`](../sashimi_w.py#L913) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`subhalos.Mzi`](../sashimi_w.py#L947) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`subhalos.Mzzi`](../sashimi_w.py#L989) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`subhalos.Delc`](../sashimi_w.py#L1043) | [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`subhalos.dMdz`](../sashimi_w.py#L1074) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`subhalos.delc_Y11`](../sashimi_w.py#L1131) | [benson](https://arxiv.org/abs/1209.3018) |
| [`subhalos.s_Y11`](../sashimi_w.py#L1179) | Units, calibration wording, or NumPy structure clarified. |
| [`subhalos.Ffunc_Yang`](../sashimi_w.py#L1201) | [yang](https://arxiv.org/abs/1104.1757) |
| [`subhalos.Na_calc`](../sashimi_w.py#L1245) | [yang](https://arxiv.org/abs/1104.1757) |
| [`subhalos.rs_rhos_calc`](../sashimi_w.py#L1368) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [scatter](https://arxiv.org/abs/1101.2020), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`subhalos.subhalo_distr`](../sashimi_w.py#L1546) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`subhalos.N_sat`](../sashimi_w.py#L1679) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`subhalos.N_sat_Vthres`](../sashimi_w.py#L1809) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [scatter](https://arxiv.org/abs/1101.2020), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`TidalStrippingSolver`](../sashimi_w.py#L1946) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`TidalStrippingSolver._Mzvir_uncached`](../sashimi_w.py#L1984) | [yang](https://arxiv.org/abs/1104.1757), [correa](https://arxiv.org/abs/1502.00391) |
| [`TidalStrippingSolver.AMz`](../sashimi_w.py#L2029) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`TidalStrippingSolver.zetaMz`](../sashimi_w.py#L2066) | [wtidal](https://arxiv.org/abs/2111.13137), [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`TidalStrippingSolver.tdynz`](../sashimi_w.py#L2103) | [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`SharpKVariance`](../sashimi_w_numerics.py#L32) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`SharpKVariance.__init__`](../sashimi_w_numerics.py#L69) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`SharpKVariance._integrand`](../sashimi_w_numerics.py#L116) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`SharpKVariance._cutoff`](../sashimi_w_numerics.py#L153) | [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`nfw_mass_function`](../sashimi_w_numerics.py#L242) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`invert_nfw_mass_function`](../sashimi_w_numerics.py#L280) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`sigma8_normalization`](../sashimi_w_power.py#L42) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_vectorized_inverse_spans_radii_and_the_disruption_threshold`](../tests/test_nfw_vectorized.py#L8) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`test_variance_is_independent_sharp_k_integral`](../tests/test_numerics.py#L30) | [viel](https://arxiv.org/abs/astro-ph/0501562), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`test_variance_is_independent_sharp_k_integral.raw`](../tests/test_numerics.py#L62) | [viel](https://arxiv.org/abs/astro-ph/0501562), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`test_variance_is_independent_sharp_k_integral.raw.integrand`](../tests/test_numerics.py#L96) | [viel](https://arxiv.org/abs/astro-ph/0501562), [sharp](https://doi.org/10.1093/mnras/stv1169) |
| [`test_top_hat_and_sharp_k_both_use_power_q10`](../tests/test_numerics.py#L150) | [viel](https://arxiv.org/abs/astro-ph/0501562) |
| [`test_exact_nfw_boundary`](../tests/test_numerics.py#L187) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |

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
- **viel**: [Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).](https://arxiv.org/abs/astro-ph/0501562)
- **virial**: [Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).](https://arxiv.org/abs/astro-ph/9710107)
- **wmap**: [Komatsu et al. (2011), Seven-Year Wilkinson Microwave Anisotropy Probe Observations: Cosmological Interpretation.](https://arxiv.org/abs/1001.4538)
- **wtidal**: [Dekker et al. (2022), Warm dark matter constraints using Milky-Way satellite observations and subhalo evolution modeling, Appendix C.](https://arxiv.org/abs/2111.13137)
- **yang**: [Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).](https://arxiv.org/abs/1104.1757)
