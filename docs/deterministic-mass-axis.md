# Deterministic mass-axis correction and EPS integration

The original PR18 shape repair addressed a known issue already fixed on fork
main by [5de09c2](https://github.com/gomeshun/sashimi-w/commit/5de09c2c1bc7e7c58eb2467aa230bb590941df0b).
Migration dcef191 reshaped using `(len(zacc), len(ma))`, even when `ma` was a
2D per-redshift virial-mass array. A 3-redshift × 4-mass grid therefore tried to
reshape 12 values to (3,3).

PR18 initially used `ma.shape[-1]` while preserving the old EPS arithmetic.
PR19 subsequently backported stable EPS and exact-z=1 scatter anchoring, merged
as 65f25390f7d745d671818ba904c59b133ce1aa29. The integrated implementation keeps
PR18's explicit 1D/2D mass-grid validation and leading-axis alignment check,
then uses PR19's `(N_herm, n_redshift, 1)` host axis and `Phi[0]` selection.
It does not restore the obsolete reshape or sampled-z>1 anchor requirement.

The original independent one-dimensional row fixture from dcef191 is preserved
with its original hash. Stable variance gaps and log-space kernels change two
extremely small row rates by at most 2.94e-11 relative (1.35e-39 absolute) on the
recorded environment. A separately installed corrected migration at 65f2539
provides new independent single-row references. The test retains rtol=5e-12,
atol=0 against that corrected file and additionally requires exact equality
against a separately generated corrected-source reference on every CI runner.
No historical file or comparison tolerance was overwritten.

Nonzero deterministic scatter weights intentionally change because both native
and compatibility paths now use the physical z=1 host mass. Both native anchor
metadata fields are 1.0. All-at-or-below-one domains succeed, and common
accretion-rate rows are independent of extending the redshift grid. Population
weights themselves may legitimately change when the integrated domain expands.

See [EPS numerical scope](eps-stability.md) and
[native integration validation](native-eps-integration.md). This remains the
existing fixed-gap EPS approximation with unchanged q10, WMAP7 and calibration;
it is not a new moving-barrier first-crossing model.
