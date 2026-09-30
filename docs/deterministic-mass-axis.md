# Deterministic host-history mass-axis correction

The migration's non-default `N_hermNa=1` branch reshaped the accretion kernel
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
