# WDM sharp-k mass assignment and common power amplitude

This correction uses author-maintained `shinichiroando/sashimi-w` upstream/main
at `99dfc3632eec0126080c0273ddf78f84fe09216c` as the scientific source baseline.
The owner main baseline is `685bcfde7bf3ac5805492b56070151ba0415a9ca`;
the migration baseline is `62b923f81c690c369a97cca567bbe32f1e32d6ba`, with the
Picard/documentation candidate at `465db7a0f24def6ebe065718fe1f5f5fc543c048`.
The physical changes are applied to both implementations.

## Declared prescription

For physical halo mass M, define R_th by M=4*pi*rho_mean*R_th**3/3.
The sharp-k window is W(k R)=Theta(1-k R), with R=R_th/c and c=2.5.
Consequently k_cut=2.5/R_th. The extra factor (9*pi/2)**(1/3) in the
previous combined cutoff is removed. This follows the mass assignment in
[Schneider (2015), equations 10–12](https://academic.oup.com/mnras/article/451/3/3117/1195205).

The q10 WDM spectrum is
P_raw(k)=P_CDM(k)*[1+(alpha*k)**(2*1.12)]**(-10/1.12).
One amplitude divisor N is computed from

```text
N**2 = integral dln(k) k**3 P_raw(k) W_TH(8 k)**2 / (2*pi**2*sigma8**2)
W_TH(x) = 3*(sin(x)-x*cos(x))/x**3
P_WDM(k) = P_raw(k)/N**2
```

Here k is in h/Mpc, P in (Mpc/h)**3, and the top-hat radius is 8 Mpc/h
at redshift zero. This is the sigma8 definition discussed in
[Benson et al. (2013), section 2.1](https://arxiv.org/pdf/1209.3018).
Both the EPS sharp-k integral and the concentration top-hat calculation
receive this same spectrum. Sharp-k sigma(M8) is not top-hat sigma8.
The migration converts k, P, rho and alpha to canonical Mpc units before
using ITAMAE; the standalone model converts physical Msun to the table's
Msun/h coordinate with h*M.

Normalization integrates every original interpolation interval with a
16-point Gauss rule in log(k). A small-x series evaluates W_TH without
cancellation. The supplied finite table is not extrapolated by this helper.
This numerical rule is separate from the unchanged 500-point concentration
power integral, 100 mass nodes and 100 concentration trials.

WMAP7, q10 transfer coefficients, the thermal barrier, Ludlow A=650 and f=0.02,
the survival threshold and tidal rate remain the selected physical inputs.
The fixed-barrier-gap EPS normalizer remains an approximation. No empirical
calibration, satellite constraint or WDM mass bound has been recomputed.

The migration calculation identifier advances to
`sashimi-w:thermal-wdm-q10:2026-10-14:v3`. Power/variance identities include
the top-hat convention and numerical amplitude; saved catalogues additionally
record sigma8, N and c. Old caches therefore cannot masquerade as v3 results.

## Independent numerical evidence

- B is exported from the exact author source, then patched independently;
  it never imports the current product or ITAMAE. Its normalization uses
  adaptive QUADPACK on original spectrum knots and the spherical-Bessel
  window. Source, input, patch, controller and output hashes accompany it.
- Product tests integrate the actual EPS power with a Bessel window and
  Simpson quadrature on 50,001 and 100,001 log-k points for 0.5, 2 and 5 keV.
  They recover sigma8=0.82 within 2e-8 relatively. Gauss orders 16/32 agree
  within 2e-12; these are numerical checks of the declared finite spectrum.
- Separate adaptive sharp-k integrals check the c=2.5 cutoff and unit
  conversions. Existing moving-boundary derivative and positive-gap checks
  remain in place.
- Standalone full-catalogue regressions retain rtol=2e-8 and historical
  odeint defaults for all three masses. Migration retains catalogue
  rtol=5e-10 and weight rtol=5e-12 for its 2 keV catalogue, with explicit
  odeint rtol=1e-11 and atol=1e-13 Msun on both sides. Refining the independent
  ODE controls by ten changes bound masses by at most 7.2e-11 relatively.

The former implicit ODE controls gave a 7.2e-9 migration/reference bound-mass
difference at 2 keV. Explicit refinement resolves that comparison without
enlarging its tolerance. It is not a universal error bound: at 5 keV, a
tighter ODE run differs across the implementations by about 3.4e-6 in bound
mass, despite much closer default-regression agreement. The inherited
nearest-mass concentration lookup and finite future-formation trial selection
can contain narrow jumps. The physical prescription correction does not
establish global solver or concentration convergence.

All earlier catalogue arrays and sidecars remain unchanged. New references
have distinct filenames. See [regeneration inputs](../tests/reference-generation/README.md)
and [the test review](w-power-test-review.md).

## Controlled effects at one fixed population grid

For a 1.5 keV particle and 1e12 Msun host, fix dz=0.2, zmax=7, 128 accretion
masses from 1e6 to 1e11 Msun, three concentration nodes, twenty host-history
nodes and the Picard solver. The table counts surviving nodes with strict
bound mass greater than 1e8 Msun, using their catalogue weights.

| Change | EPS amplitude divisor N | c200(1e10 Msun, z=0) | N(m_bound > 1e8 Msun) |
| --- | ---: | ---: | ---: |
| Owner main baseline | 1.8908692881 | 6.603276509 | 3.429548376 |
| Cutoff only, baseline amplitudes held fixed | 1.8908692881 | 6.603276509 | 19.352656015 |
| Common top-hat amplitude only, old cutoff | 0.9633008077 | 6.613366282 | 2.795588918 |
| Both corrections | 0.9633008077 | 6.613366282 | 14.151885395 |

This separates the two causes without secretly renormalizing the filter-only
case. It is a finite-grid diagnostic, not a converged prediction or comparison
with simulation data. The amplitude-only row also replaces concentration's
former coarse-mass interpolation normalization with the common direct R=8
integral. [Inputs and output records](validation/w-prescription-20261014/)
retain the exact numerical settings and executable source hashes.
