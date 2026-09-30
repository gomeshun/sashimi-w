# Migration backport of the deterministic mass-axis correction

This is a migration-specific minimal backport of a **known issue already fixed
on main**, not a newly discovered unfixed main bug. Upstream commit
[5de09c2](https://github.com/gomeshun/sashimi-w/commit/5de09c2c1bc7e7c58eb2467aa230bb590941df0b),
merged by [PR17](https://github.com/gomeshun/sashimi-w/pull/17), also changed the
scatter anchor and EPS active-support/variance-gap arithmetic.

The pinned migration's non-default `N_hermNa=1` branch reshaped the accretion kernel
using `(len(zacc), len(ma))`. The current population executor supplies a 2D
per-redshift virial-mass grid, so `len(ma)` is the redshift count, not the mass
count. A 3-redshift × 4-mass grid failed by trying to reshape 12 values to (3,3).

The correction uses `ma.shape[-1]` and explicitly validates 2D leading-axis
alignment. It preserves the 1D mass-axis behavior, the ordinary quadrature
branch, and all kernel/normalization arithmetic. No physical formula or default
solver changes.

The small independent reference was generated before editing from installed
migration commit `dcef1910d42cab940567be448ffc79b42d436802`: each redshift row was
calculated separately through its unchanged 1D mass-grid path with `sigmafac=0`.
The repaired nonsquare 2D path is compared against those stored rows. The
native API's square deterministic catalogs also retain pre-change parity.

A separate existing convention remains: deterministic accretion scatter uses
the first sampled redshift above one as its anchor, while the tidal host
history uses redshift one. The native API records both and rejects domains with
no sampled z>1. This patch does not silently change either convention.

## Comparison with upstream and remaining divergence

Upstream normalizes M200 to `(N_herm, n_redshift, 1)` and selects `Phi[0]` for
the deterministic branch after replacing the EPS kernel/active-domain handling.
The migration-only repair instead uses `ma.shape[-1]` in its existing reshape,
with leading-axis validation. The saved square-grid and independent 1D-row
references establish equivalence to the intended migration arithmetic.

The exact-z1 scatter anchor and finite active-support/variance-gap/log-kernel
changes from upstream are **not** included here. The sampled-z>1 requirement is
a preserved limitation of this pinned migration branch, not a universal WDM
requirement or a limitation of current main. No whole-commit cherry-pick or
scientific-default synchronization is implied; main and migration remain
different reviewed scientific specifications.
