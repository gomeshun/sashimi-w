<p align="center"><img src="assets/logo.svg" alt="SASHIMI-W logo" width="440"></p>

# SASHIMI-W

Semi-Analytical SubHalo Inference ModelIng for warm dark matter.
This migration candidate provides one standard ITAMAE-backed calculation.
The explicit WDM power choice, WMAP7 calibration, tidal prescriptions and
survival threshold belong to W; shared numerical mechanisms belong to ITAMAE.

## Physical walkthrough

The [executed physical walkthrough](notebooks/usage_walkthrough.ipynb) computes
subhalo mass functions and weighted Vmax–rmax distributions for a Milky Way scale
host, together with satellite observables and a joint grid refinement. See
[package layout and setup](docs/package-layout.md) for the `src/` structure.

## Standard API

```python
from sashimi_w import Subhalos

model = Subhalos(mass_wdm=2.0)
catalog = model.rs_rhos_catalog_calc(
    M0=1e10, dz=0.5, zmax=1.0, N_ma=4, N_herm=2,
    N_hermNa=3, logmamin=6, logmamax=8,
)
print(catalog.weight_final.sum())
catalog.to_npz("wdm-catalog.npz")
```

This small grid demonstrates the API; it is not a converged abundance estimate.
The named catalog uses physical Msun, Mpc and Msun/Mpc³. It retains independent
`weight_base`, `weight_concentration` and `weight_survival` factors. The final
weight is their product. Column order and node identity are preserved.

`subhalos` is an alias for `Subhalos`. The previous `sashimi_w_itamae` import
is an alias too. `physics_mode` has been removed and is rejected. Reproducing
an older published calculation requires the independent frozen reference
workflow, not a product mode. This is a breaking API and numerical change;
see [the migration table](docs/standard-api-migration.md).

`rs_rhos_calc` is a tuple-format conversion of the same standard calculation:

```text
m200_acc, z_acc, r_s_acc, rho_s_acc, m_bound, r_s, rho_s, c_t, weight, survive
```

Tuple radii use kpc, densities Msun/pc³, and tuple `weight` excludes survival.
Use the named catalog's final weight for the expected surviving population.
The existing `subhalo_distr`, `N_sat`, and `N_sat_Vthres` functions now consistently
count current survivors and forward `profile_change`. The accretion mass display
uses the same survivors. See [thresholds and cumulative axes](docs/surviving-observables.md).
The current-survivor definition was adopted after the independent review on
2026-09-10 and is verified against complete named catalogs.

## Power and background specification

The current Viel coefficients and thermal-particle mass definition use the
transfer amplitude `T=[1+(alpha k)^(2 nu)]^(-5/nu)` and power ratio `T²`
(exponent `-10/nu`). q10 is the sole standard path after the 2026-09-11 adoption.
`Subhalos(2.0)` and explicit `wdm_power_convention="standard-t2-q10"` select it.
The old `published-q5` selector raises a descriptive error; `PUBLISHED_Q5` is
removed. Frozen q5 references remain available for historical reproduction.

Both sharp-k EPS variance and top-hat Ludlow concentration use q10.
`transfer_amplitude(k)` and `power_ratio(k)` distinguish the two quantities
(k in h/Mpc). `half_mode_wavenumber()` retains amplitude half (T=0.5, power=0.25);
`half_mode_wavenumber(power_ratio=0.5)` explicitly reports power half. The
threshold diagnostic does not change the population or concentration calculation.
[Adoption and controlled comparison](docs/q10-adoption.md) records the evidence,
old-input/cache rejection and unchanged coefficients. New fitting formulas and
observational limits are separate work; no old mass bound is rescaled here.

The standard background is flat WMAP7 (Omega_m=0.27, h=0.7), with D(0)=1 and
S(M,z)=D(z)² S(M,0). Nonmatching cosmology parameters are rejected. The table
uses k in h/Mpc and P in (Mpc/h)³. The shared integral receives physical
k=h k_table and P=P_table/h³, physical mean density rho=rho_table h², and
alpha=alpha_table/h. The filter is R=R_th/2.5 with
k_cut=(9 pi/2)^(1/3)/R. Every accretion redshift uses its own virial-mass row.

The shared fixed-cell integral splits at tabulated spectrum knots. Its
moving-boundary derivative and variance describe the same continuous integral;
there is no coarse mass interpolation, variance projection or signed-weight
clipping. Top-hat concentration retains the original numerical calibration.
The default mass solver retains odeint/LSODA, SciPy tolerance defaults and
100 redshift output points. The strict survival criterion is c_t > 0.77.

## Reproducible use and validation

[Usage walkthrough](notebooks/usage_walkthrough.ipynb) runs from outside the
source tree. [Scientific validation](notebooks/scientific_validation.ipynb)
uses the validation source bundle and compares independent B/C full catalogs,
historical A provenance, and the numerical derivative. These are separate
from broad grid/solver convergence and simulation validation.

The existing `dz=0.1` default is retained by explicit scope agreement.
[Measured resolution dependence and review procedure](docs/resolution-and-review.md)
report the fixed historical q5 and adopted q10 sweep: `.1` to `.0125` changes representative catalog
counts by about 5% and bound mass fractions by 17–18%. These finite-grid effects
are not universal error estimates for the full default configuration.

The [family reference workflow](https://github.com/gomeshun/sashimi-family/tree/codex/migration-release-20260910/validation/references/sashimi-w)
freezes A and separately applies each correction as B without importing the
current product. Historical signed weights and original fixtures remain
unchanged. [Shared-stage boundaries](docs/population-components.md) document
physical ownership, units and independent weights.

Python 3.11–3.13 is the target range. The candidate depends on `sashimi-itamae`
(import `itamae`), supplied from a pinned local wheelhouse during preparation.
The development lock/CI uses an exact reviewed core source revision:

```bash
uv sync --extra test
uv run --no-sync python -m pytest -q
```

Source tree tests cover independent full catalogs, formula/derivative checks,
unit conversion, nonnegative weights, NFW mass identity, serialization and
invalid inputs. A final exact five-package artifact matrix remains required
before the candidate is ready for peer review. No release has been published.

## References

Please cite the relevant primary papers:

- [Dekker et al., SASHIMI-W](https://arxiv.org/abs/2111.13137)
- [Hiroshima, Ando & Ishiyama](https://arxiv.org/abs/1803.07691)
- [Ludlow et al., concentration prescription](https://arxiv.org/abs/1601.02624)

## Review candidate

The prepared version is `0.2.0rc1`, with a versioned `sashimi-itamae` dependency.
See [release preparation](docs/release-preparation.md), [changelog](CHANGELOG.md)
and [citation metadata](CITATION.cff). The full artifact matrix is recorded in the
family review handoff after verification; no public upload is implied.
