# Continuous WDM variance and redshift-resolved mass integration

Update 2026-09-11: q10 is now the sole supported thermal-WDM product prescription.
The q5 results below are historical; see [adoption](q10-adoption.md).

The previous corrected path interpolated 100 sigma nodes and projected them
with a cumulative minimum, while using the exact moving-boundary derivative.
Thus sigma and its derivative represented different functions. Sigma now uses
the same finite-domain sharp-k integral as dS/dM, composed with ITAMAE's fixed
positive quadrature and interpolation-knot partition. No variance or weight is
clipped, projected or renormalized to remove a sign discrepancy.

The accretion mass integral now uses each redshift's virial-mass row, matching
the existing corrected accretion-rate input. The historical tuple path still
uses its final row at this review stage; removing that path belongs to the
subsequent standard-API change.

The corrected model identity advances from v5 to v6 and records the complete
variance identifier. ITAMAE is pinned to ceb38eaf6ee57efb43ccb60005c68c1c4cf044ac.
WMAP7, the historical sigma8 normalization and explicit q5/q10 choices remain
unchanged. Same-valued half-mode scales do not imply equivalent spectra.

Independent B catalogs use frozen source, high-precision/adaptive formulas and
a separate historical dependency environment. Both complete 16-node catalogs
agree within the original tolerances (catalog rtol 5e-10, weight rtol 5e-12).
No historical fixture was overwritten; see tests/references/README.md. Before
updating the reference selection, 34 tests passed and four failed because of
the intentionally changed catalogs, identifier and deleted projected cache.
The updated suite has 39 passing tests. The new q5/q10 finite-difference check
requires relative error below 4e-8; the former mass interpolation fails it.

These compact comparisons are numerical acceptance evidence. Higher-resolution
convergence, the complete observable inventory and final packaged API checks
remain part of the family release-preparation gate.
