# Thermal WDM q10 adoption — 2026-09-11

The user adopted the independent power review's q10 policy after comparison of
Viel's transfer-amplitude definition, the published SASHIMI-W expression, upstream
code and independent CLASS calculations. With the **existing** Viel alpha, nu
and thermal-particle mass definition, power is T squared: exponent -10/nu.
The supplied evidence and hashes are preserved in the family repository at
`validation/independent-w-power-review-20260911`. Completed CLASS runs were not
repeated. This adoption does not introduce new fitted coefficients, cosmology,
filters, collapse or tidal prescriptions, redshift steps, or observational limits.

## API and identities

`Subhalos(mass_wdm=2.0)` uses q10. Explicit `standard-t2-q10` still works.
`published-q5` raises ValueError identifying its retirement; the public
PUBLISHED_Q5 constant is removed. Historical q5 sources, patches and fixtures
remain unchanged outside the runtime. Variance factories reject old q5 models
as well as old selectors. Read-only convention fields cannot be reassigned to
silently mix the concentration and variance prescriptions.

`transfer_amplitude(k)` returns T and `power_ratio(k)` returns T squared,
with k in h/Mpc. The latter retains the pre-adoption q10 arithmetic order.
`half_mode_wavenumber()` keeps the exact previous q10 amplitude-half factor
(T=1/2, power=1/4); explicit `power_ratio=0.5` returns power half. Metadata
records both thresholds/scales. This diagnostic choice changes no calibration.

The calculation specification is `sashimi-w:thermal-wdm-q10:2026-09-11:v2`.
Power and variance identifiers include it, so old cache keys cannot be reused
under the new specification, including old q10 caches. Historical catalog
metadata is not relabeled by generic catalog deserialization. Applications
loading old catalogs must inspect their recorded specification. Transfer
callbacks snapshot alpha and q to keep values bound to their content identity.

## Validation

Twelve targeted adoption tests failed before the change and passed after it:
old-selector rejection, default/export contract, immutable prescription fields,
T/T-squared equality and explicit valid/invalid half-mode thresholds. Existing
independent B-q10 full-catalog reference tests retain the original tolerances;
old q5 arrays and provenance remain unchanged. A cache test reconstructs the
prior q10 identifier and confirms key mismatch is rejected.

The separate family runner `scripts/check_w_q10_adoption.py` exports exact
source revisions into fresh processes and retains full arrays and hashes. It
fixes old q10 at 09322feb1e348f40fa93d0514f42dd02171a7282 and core at
dfa083d0d46181c376947ac7b2da829facaf2e8b. Controls cover 0.5, 2 and 5 keV,
81 physical masses (1e6–1e14 Msun), four redshifts, 101 wavenumbers, concentration,
derivatives, the default half-mode, all catalog columns and weight factors.
Catalog settings: M0=1e12 Msun, zmax=1, dz=.25, N_ma=16, N_herm=3,
N_hermNa=4, logmamin=7, logmamax=10. At candidate e3017d93b4dfdcb975bca71f1a00535e8faf4ec2, all 21 arrays per
particle mass are bitwise identical to old q10, with zero warnings on both sides.
See `validation/q10-adoption/comparison.json`. The full local suite passes
87 tests. The controlled run retains an explicit note that installed core
metadata still said 0.1.0a4, while the verified editable source/runtime was
0.2.0rc1 at the exact SHA above; both sides used the same environment.

The existing q10 finite-grid convergence evidence remains distinct from
formula adoption. Its remaining dz sensitivity is not removed by choosing q10;
see [resolution limits](resolution-and-review.md). No mass limit is inferred
by converting the old q5 half-mode scale.
