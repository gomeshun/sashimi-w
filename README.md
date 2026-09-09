<p align="center">
  <img src="assets/logo.svg" alt="SASHIMI-W logo" width="440">
</p>

## Hands-on usage walkthrough

Start with [the executable usage walkthrough](notebooks/usage_walkthrough.ipynb): setup, main APIs, plots, catalogue export and checks in one notebook for this package. Its coverage table records remaining gaps. The **Usage walkthrough** CI runs every cell against the candidate package and uploads an executed notebook. This is a mandatory migration deliverable tracked in [sashimi-family #28](https://github.com/gomeshun/sashimi-family/issues/28).

# Semi-Analytical SubHalo Inference ModelIng for WDM (SASHIMI-W)
[![arXiv](https://img.shields.io/badge/arXiv-2111.13137%20-green.svg)](https://arxiv.org/abs/2111.13137)

The code calculates weighted warm-dark-matter (WDM) subhalo catalogs with a
semi-analytical accretion and tidal-evolution model. Its WDM transfer function,
sharp-\(k\) filtering, concentration calibration, and profile-evolution
prescription remain specific to SASHIMI-W.

## Legacy API

The established public API remains available and retains its historical
numerical conventions:

```python
from sashimi_w import subhalos

model = subhalos(mass_wdm=1.5)  # WDM particle mass in keV
result = model.rs_rhos_calc(
    1.0e12,
    redshift=0.0,
    dz=0.1,
)
```

The returned tuple contains

```text
m200_acc, z_acc, r_s_acc, rho_s_acc,
m_bound, r_s, rho_s, c_t, weight, survive
```

The legacy radii are in kpc and densities are in
\(M_\odot\,\mathrm{pc}^{-3}\). Its `weight` excludes the binary `survive`
factor. Existing callers do not opt into ITAMAE merely by upgrading or
importing `sashimi_w`.

## ITAMAE opt-in API

SASHIMI-W is being migrated to the shared
[ITAMAE](https://github.com/gomeshun/itamae) numerical toolkit. Select the
parallel API explicitly:

```python
from sashimi_w_itamae import subhalos

model = subhalos(
    mass_wdm=1.5,
    physics_mode="consistent",  # opt-in default
    wdm_power_convention="standard-t2-q10",
)
catalog = model.rs_rhos_catalog_calc(
    1.0e12,
    redshift=0.0,
    dz=0.1,
)

print(catalog.weight_final)
print(catalog.metadata["model_identifier"])
```

The migrated call returns an ITAMAE `WeightedSubhaloCatalog`. Mass, radius,
and density are converted to its canonical units of \(M_\odot\), Mpc, and
\(M_\odot\,\mathrm{Mpc}^{-3}\). Population, concentration-quadrature, and
survival weights are stored independently:

- `weight_base`: accretion-population weight before concentration quadrature;
- `weight_concentration`: Gauss-Hermite concentration weight;
- `weight_survival`: binary tidal-survival factor;
- `weight_final`: the product of the three factors.

Metadata records the WDM particle mass, power convention and exponent,
half-mode definition, physics mode, cosmology and unit backend identifiers,
variance-growth convention, schema and ITAMAE versions, and both legacy and
canonical units.

For a development installation from the pinned public ITAMAE source:

```bash
uv sync --extra test
```

The repository's `uv.lock` and CI resolve ITAMAE from a reviewed, pinned public
Git commit. A sibling checkout can instead be installed directly before this
project.

### Executable migration demo

[`itamae_migration_demo.ipynb`](itamae_migration_demo.ipynb) is a compact,
executed comparison of:

- the unchanged public `sashimi_w.subhalos` tuple API;
- `physics_mode="legacy"` with `wdm_power_convention="published-q5"`;
- consistent growth and mass units with the same published q5 spectrum; and
- consistent physics with the standard transfer-squared q10 spectrum.

It checks legacy tuple parity, displays a catalog summary table, and plots the
subhalo mass function and cumulative satellite abundance. Install the
notebook-only dependencies and rerun it from a clean kernel with:

```bash
uv sync --extra demo
uv run --no-sync jupyter nbconvert \
  --to notebook --execute --inplace itamae_migration_demo.ipynb
```

The original [`Examples.ipynb`](Examples.ipynb) demonstrates the historical
high-resolution API defaults and is correspondingly much slower. Its imports
are explicit so it remains compatible with current NumPy and Matplotlib.

## Physics modes and known legacy differences

The opt-in class requires one of two explicit modes. They never change the
behavior of `sashimi_w.subhalos`.

- `physics_mode="consistent"` is the opt-in default. It uses the flat WMAP7
  background \(\Omega_{\rm m}=0.27\), \(\Omega_\Lambda=1-\Omega_{\rm m}=0.73\),
  normalizes the linear growth factor to \(D(0)=1\), and implements
  \(S(M,z)=D(z)^2S(M,0)\), hence
  \(\mathrm{d}S/\mathrm{d}M(M,z)=D(z)^2
  \mathrm{d}S/\mathrm{d}M(M,0)\). All masses passed through this mode are
  physical \(M_\odot\).
- `physics_mode="legacy"` reproduces the original SASHIMI-W equations and is
  provided for regression and published-result reproduction.

Six coupled legacy inconsistencies motivated the separate consistent mode:

1. The historical expression defines
   `OmegaL = 1 - OmegaC - Omegar`, omitting the baryon contribution when
   subtracting the total matter density. With the bundled WMAP7 table this is
   \(\Omega_\Lambda=0.7769\), so
   \(\Omega_{\rm m}+\Omega_\Lambda=1.0469\), rather than a flat background.
2. The legacy growth approximation is consequently not normalized at the
   present epoch (\(D(0)\simeq0.938\)). This normalization propagates into
   collapse thresholds, host histories, and abundance weights.
3. Although the implementation defines
   \(\sigma(M,z)=D(z)\sigma(M,0)\), its historical derivative scales
   \(\mathrm{d}S/\mathrm{d}M\) with one power of \(D\). Since
   \(S=\sigma^2\), the self-consistent derivative requires two powers.
4. The historical catalog passes physical \(M_\odot\) values directly to a
   variance table whose numerical mass coordinate is \(M_\odot/h\). The
   correct table coordinate is \(M_{\rm table}=hM_{\rm physical}\).
5. The historical concentration boundary converts a physical CGS mass using
   \(M/M_\odot/h\), rather than the same
   \(hM/M_\odot\) table coordinate.
6. The catalog loop overwrites the virial-mass grid at each accretion
   redshift, then passes only the final redshift's grid to `Na_calc` for every
   redshift. Consistent mode instead evaluates the accretion factors with the
   matching virial-mass row at each redshift.

These corrections are physically linked, so `consistent` applies them
together and catalog metadata prevents results from the two conventions from
being mixed silently. They can materially change abundance normalization;
`legacy` should therefore be selected explicitly when reproducing an earlier
SASHIMI-W result.

The historical sharp-\(k\) implementation evaluates each mass with a separate
fixed-node quadrature. On the WDM variance plateau, integration noise can make
the tabulated \(S(M)\) locally increase and hence produce negative
\(\mathrm{d}S/\mathrm{d}M\), accretion rates, and population weights for some
otherwise valid grids. The tuple API retains those signed values for exact
reproduction. `WeightedSubhaloCatalog` deliberately rejects them because its
weights represent nonnegative effective counts; the migration does not clip
or silently renormalize them. The consistent mode uses ITAMAE's
moving-boundary derivative and does not show this sign failure. The compact
demo uses the reviewed nonnegative golden grid.

The canonical WMAP7 matter density and Hubble parameter are fixed during this
migration. A custom cosmology backend is rejected unless it matches them,
because replacing only expansion or growth while retaining the variant's
WMAP7-calibrated transfer and concentration equations would create another
mixed model.

## WDM power conventions

The ITAMAE API requires `wdm_power_convention` explicitly. It is independent
of `physics_mode`: changing expansion, growth, or mass-unit behavior must not
silently select a different primordial power spectrum.

Viel et al. define the q=5 expression as the transfer amplitude

\[
T(k)=\sqrt{\frac{P_{\rm WDM}(k)}{P_{\rm CDM}(k)}}
=\left[1+(\alpha k)^{2\nu}\right]^{-5/\nu}.
\]

SASHIMI-W's published implementation instead inserted that q=5 expression
directly as the power ratio. The two explicit choices are therefore:

- `wdm_power_convention="published-q5"`:
  \(P_{\rm WDM}/P_{\rm CDM}=[1+(\alpha k)^{2\nu}]^{-5/\nu}\).
  This reproduces arXiv:2111.13137 and is the convention permanently retained
  by the public `sashimi_w.subhalos` API.
- `wdm_power_convention="standard-t2-q10"`:
  \(P_{\rm WDM}/P_{\rm CDM}=T^2
  =[1+(\alpha k)^{2\nu}]^{-10/\nu}\).
  This is the standard transfer-squared interpretation and the candidate
  convention for future physical results after dedicated validation.

The q5/q10 choice feeds both consumers of the WDM spectrum: the sharp-\(k\)
variance used by EPS and the top-hat variance used by the Ludlow
concentration model. A mixed q5/q10 catalog is rejected rather than being
assembled implicitly.

The historical q5 result calls the scale where its power ratio is 0.5 the
half-mode scale. For standard q10, this repository follows the later
amplitude convention \(T(k_{\rm hm})=0.5\), for which the power ratio is
0.25. These definitions happen to give the same numerical
\(k_{\rm hm}=\alpha^{-1}(2^{\nu/5}-1)^{1/(2\nu)}\), but they are not
semantically interchangeable. Catalog metadata and variance identifiers
record the convention, formula role, q, and half-mode definition.

The q10 option is intentionally not an implicit default. It materially changes
the EPS derivative and full-catalog abundance, and a half-mode mass conversion
does not reproduce the q5 catalog. Ono et al. (2025) document a q10 simulation
spectrum but do not identify the exact SASHIMI-W source commit used for their
semi-analytical curves, while the public upstream implementation remains q5.
The migration therefore preserves q5 provenance and treats q10 as an explicit
validation path until the simulation curves and complete satellite likelihood
have been reproduced.

## ITAMAE integration boundary

SASHIMI-W continues to own the selected WDM power suppression, the WMAP7 spectrum,
sharp-\(k\) filter radius \(R_{\rm th}/2.5\), the historical cutoff
\(k_c=(9\pi/2)^{1/3}/R\), and WDM concentration prescription. ITAMAE owns
generic spectrum/window protocols, variance
integration and cache helpers, cosmology/unit interfaces, NFW mass inversion,
and the weighted-catalog schema.

At the integration boundary, the bundled table is converted from its native
\(k_h\,[h\,\mathrm{Mpc}^{-1}]\) and
\(P_h\,[(\mathrm{Mpc}/h)^3]\) coordinates to
\(k=hk_h\,[\mathrm{Mpc}^{-1}]\) and
\(P=P_h/h^3\,[\mathrm{Mpc}^3]\). The corresponding mean density and WDM
length scale are converted as
\(\bar\rho=\bar\rho_hh^2\,[M_\odot\,\mathrm{Mpc}^{-3}]\) and
\(\alpha=\alpha_h/h\,[\mathrm{Mpc}]\). The sigma8 normalization mass is
likewise interpreted as a physical mass, \(M_8=M_{8,h}/h\).

Consistent catalogs use ITAMAE's analytic sharp-\(k\) moving-boundary
derivative. Their \(\sigma(M,0)\) values are evaluated once on SASHIMI-W's
physical mass grid and interpolated in log mass during the catalog
calculation. The legacy-callable variance adapter remains available for exact
published-result regression. Select `physics_mode="legacy"` together with
`wdm_power_convention="published-q5"` to reproduce the complete historical
model; neither setting implies the other.

## Validation

With the pinned ITAMAE package installed, run:

```bash
python -m pytest -q
```

Migration regressions cover both physics modes, compact full-catalog golden
fixtures, weight factorization, canonical-unit conversion, WDM transfer and
variance invariants, serialization, import isolation, and clean-wheel imports
from outside the source tree. CI runs the suite on supported Python versions
and builds both wheel and source distributions.

## References

If you use the SASHIMI script to write a paper, please cite:

- A. Dekker, S. Ando, C.A. Correa and K.C.Y. Ng, arXiv: 2111.13137
- N. Hiroshima, S. Ando and T. Ishiyama, arXiv: 1803.07691
- A.D. Ludlow, S. Bose, R.E. Angulo, L. Wang, W.A. Hellwing, J.F. Navarro, S. Cole and C.S. Frenk, arXiv: 1601.02624
