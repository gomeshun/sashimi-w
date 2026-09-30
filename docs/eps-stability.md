# EPS numerical stability and deterministic scatter anchor

This is a narrow backport of the standalone W implementation at
[`5de09c2`](https://github.com/gomeshun/sashimi-w/commit/5de09c2c1bc7e7c58eb2467aa230bb590941df0b),
checked against main at `685bcfde7bf3ac5805492b56070151ba0415a9ca`.
The migration baseline is `dcef1910d42cab940567be448ffc79b42d436802`;
ITAMAE remains pinned to `23d01e8758a88b061b87de9e488c38ec89fd8e4f`.

## Retained physical approximation

The Yang EPS kernel has the fixed-gap form
`F = delta / sqrt(2*pi) / ds**1.5 * exp(-delta**2/(2*ds))`, with
`B = erf(delta/sqrt(2*dsmin))` and strict support `ma < mmax`.
Here `ds = S(ma)-S(Mmax)` and `dsmin = S(mmax)-S(Mmax)`.
The retained WDM collapse threshold depends on mass. Therefore using this
analytic fixed-gap normalizer is the existing WDM approximation, not an exact
moving-barrier first-crossing solution or a newly calibrated physical model.
The normalization test integrates a genuinely fixed gap only.

Changes are limited to:

- Avoid evaluating the discarded overflow-prone branch of the collapse threshold
- Evaluate the normalized kernel only on strict active mass support
- Integrate positive variance intervals directly rather than subtracting two
  saturated WDM variances; nonpositive gaps on active support are errors
- Evaluate regular kernels in log space and retain the normalized limit
  `sqrt(dsmin)/(2*ds**1.5)` at zero barrier gap. The small-gap series is also
  evaluated in log space to avoid intermediate underflow
- Preserve zero population weights for a wholly suppressed grid, while keeping
  the existing normalization operation order for positive populations
- Use the host mass at exactly `z=1` to anchor deterministic (`N_hermNa=1`)
  scatter width, rather than the first sampled redshift above one. The existing
  low-redshift taper and positive-offset host-mass cap are retained

The scatter correction affects nonzero deterministic `sigmafac`. It also removes
the missing-index failure when an input grid has no redshift above one.
Tidal host history already used exact `z=1` and is unchanged. Multi-node
Gauss–Hermite scatter and the existing unused `dMdz` `sigmafac` argument are
unchanged. The explicit host-node axis also supports nonsquare redshift/mass
arrays without a length-based reshape.

## Variance ownership and provenance

The pinned ITAMAE version has no variance-interval method. The W-side helper
uses its already normalized canonical power spectrum, density, filter scale,
k-range, fixed logarithmic cells, original spectrum knots, and quadrature order.
Same-cell intervals are integrated once; split intervals combine two partial
cells with forward or reverse whole-cell sums, preserving positive suppressed
high-k tails. Standard variance and its derivative remain ITAMAE calculations.

Transfer `q10`, WMAP7 cosmology, sigma8 normalization, mass/length units,
concentration calibration, tidal evolution, survival threshold, and integration
controls are unchanged. Catalog metadata adds `eps_numerics` and
`deterministic_scatter_anchor`; the physical variance specification is unchanged.
Source revision remains part of catalog provenance.

## Validation

`tests/test_eps_stability.py` checks independent adaptive-quadrature normalization,
normalization to one for a fixed gap, zero/small-gap continuity, strict support,
active-domain failures, direct interval quadrature (including saturated tails),
scalar/broadcast/nonsquare grids, host orders 1/4/64/200, exact-z=1 anchoring and
redshift-grid independence for deterministic offsets, and zero populations.
Existing immutable A/B reference catalogs remain unchanged.

The controlled comparison script uses isolated processes for pinned old migration,
standalone main and candidate sources. It compares all ten catalog columns,
counts and bound mass fractions across the six compact native-API transport
cases, representative higher-order populations, deterministic offsets and an
empty population. Its [machine-readable results](validation/eps-backport.json)
include inputs, source hashes, failures, and absolute errors scaled by each
reference column's peak. These are selected numerical regression checks, not
simulation validation, broad convergence guarantees, or new observational bounds.

Reproduce with the candidate environment, plus matplotlib required by standalone
main, and clean checkouts of the revisions above:

```sh
python scripts/validate_eps_backport.py \
  --baseline-source /path/to/pinned-migration \
  --main-source /path/to/pinned-main \
  --output docs/validation/eps-backport.json
python -m pytest -q
```

## Native API PR integration

This backport targets `itamae-migration` independently of native API PR #18.
Both touch deterministic `Na_calc` shape handling. When integrating that PR,
preserve its input-shape validation and this explicit host-node axis; do not
restore its old length-based reshape or first-sampled-redshift scatter anchor.
Native transport fixtures remain records of the old migration calculation;
nonzero deterministic scatter weights intentionally change under this correction.

### Measured effects on selected grids

- The three representative `N_hermNa=200` populations (0.5/2/5 keV, host
  `1e12 Msun`, masses `1e8–1e11 Msun`, `dz=0.5`, `zmax=3`) retain counts to
  within `1.3e-14` relative and bound fractions to `1.4e-14`
- For the same 2-keV grid with one host history, deterministic `sigmafac=-0.5`
  changes survivor count by `+0.7075%` and bound fraction by `+0.7749%`;
  `sigmafac=+0.5` changes them by `-0.1436%` and `-0.1578%`. Zero offset agrees
  to roundoff. These changes isolate the corrected scatter-width anchor
- The compact native-API `sigma_offset` case changes count by `-2.2618%` and
  bound fraction by `-2.2623%`; the other five compact cases agree within
  `6.9e-13` relative. Frozen fixtures are intentionally not overwritten
- All eight structural columns and the survival flags remain bit-identical
  to the old migration in every successful old/candidate comparison
- Twelve rate grids covering 0.5/2/5 keV and orders 1/4/64/200 agree with
  standalone main within `3.65e-15` of each grid peak. Catalog accretion weights
  agree within `6.52e-15` of their peak. Standalone main's tidal structural
  outputs are not a strict parity target: its pre-existing solver path differs
  from migration at roughly `1e-5` peak-scaled in these cases, while this
  backport leaves migration's structural outputs exactly unchanged
- The zero-population grid previously failed the finite-weight contract;
  it now returns finite structural fields and exactly zero base/final weights
