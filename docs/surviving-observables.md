# Current surviving populations and cumulative observables

The user approved this interpretation after independent review on 2026-09-10.
All three observables use the survivors at the requested observation redshift.
`subhalo_distr(accretion=True)` displays those same survivors by accretion mass;
`False` displays their current bound mass. Both retain tidal evolution and the
requested `profile_change`. This agrees with the actual C/F observable
constructors, which select survivors before changing the displayed mass.

| API | Additional selection | Returned curve axis and meaning |
| --- | --- | --- |
| `subhalo_distr` | none | logarithmic mass-bin centers in Msun, dN/dm of survivors |
| `N_sat` | strict ma200 > Mpeak when enabled | x in Msun; exact sum of selected weights with ma200 > x |
| `N_sat_Vthres` | strict Vpeak > threshold, or Vmax > threshold when Vpeak_thres=False | x in km/s; exact sum of selected weights with current Vmax > x |

`N_sat` already used an accretion-mass horizontal axis in the source; it did
not use Vmax. The independent review's general sentence about both axes being
Vmax did not match that implementation. This correction preserves each actual
observable coordinate and states it explicitly.

The cumulative APIs return `(total, x, N_above_x)`. Total is the direct sum of
selected weights, independent of the display bins. x contains 10000 equally
spaced histogram **right edges**, while counts are computed from individual
selected rows with the strict inequality, not from bin centers or binned sums.
Thus the total may exceed the first curve value when its threshold excludes
rows. An empty selection returns total zero and a finite zero curve on the
default [0, 1] threshold grid. These thresholds have no inferred physical range
when no objects were selected.

Previously the methods ignored `survive`; both cumulative totals also omitted
the entire first bin. With weights [2,3,5], the corrected totals are 10 for all
survivors, 7 when the middle row is destroyed, and 0 when all rows are destroyed.
The old cumulative API returned 8 in every case. Its two errors could cancel
for some threshold selections. Thirty-five new regression cases failed before
the correction and pass afterward, covering both mass displays, all-destroyed,
single/empty selections, both velocity selectors, exact threshold equality,
strict cumulative counts and forwarding `profile_change=False`.

Thresholds are compared in the documented public units. In particular velocity
thresholds use computed km/s values directly, avoiding a unit round-trip that
could include an exactly equal value by a last-bit difference. Input arrays are
not modified during conversion. The population's existing strict `c_t > 0.77`
survival rule is unchanged; this does not decide the physical equality boundary.

The present-day mass function's survival factor is explicit in Eq. (1) and
Appendix D of [Dekker et al. (2022)](https://arxiv.org/pdf/2111.13137).
This aggregation correction introduces no new calibration or default change.
The separate [resolution limits](resolution-and-review.md) remain applicable.

The installed API was also checked against named catalogs for q5/q10 and
`profile_change=True/False` at the recorded 32-row configuration. All four
counts and strict cumulative curves agree. See
`validation/convergence/surviving-observables.json` and
`python scripts/check_observable_counts.py --output <new-report.json>`.
Both walkthrough (4 code cells) and scientific notebook (3 code cells) execute
in fresh kernels. The first local notebook launch required local socket
permission; the successful runs exercised every cell.
