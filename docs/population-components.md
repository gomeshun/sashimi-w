# WDM stages through the shared executor

Update 2026-09-11: q10 is now the sole supported thermal-WDM product prescription.
The q5 results below are historical; see [adoption](q10-adoption.md).

WDM now owns explicit host history, tidal mass loss, accretion slices, initial
NFW structure, evolved profiles, survival and named columns. ITAMAE owns stage
ordering, validation, weight transport and concatenation. Accretion mass and
base population weight are prepared before concentration quadrature; independent
weights are never recovered by division of the final product.

The pipeline's batches and stage arrays use physical Msun, Mpc and Msun/Mpc^3.
Historical CGS constants occur only inside WDM kernel calls and local contexts;
conversion happens explicitly before arrays cross the common stage boundary.
The mass evolution coordinate is redshift. `solve_evolution(method="odeint")`
preserves W's original LSODA tolerance defaults and 100 output points.
The q5/q10 choice, concentration calibration, profile response and strict
c_t > 0.77 survival rule are unchanged. Population quadrature retains the
original global normalization before splitting the redshift slices.

The named catalog is now the execution result. The ten-column method only
converts that result to the historical tuple format for the corrected model.
Removal of the remaining legacy mode/public classes is the next API review
unit, not a claim made by this structural commit.

Validation on 2026-09-10: the existing 39 tests pass, including immutable
independent B q5/q10 full-catalog comparisons and their original 5e-10 / 5e-12
tolerances. No golden files were changed. A new independent exponential
mass-loss case verifies half-mass evolution, the canonical NFW mass identity,
node/weight identity and repeated-batch equivalence. It passes separately,
bringing the covered cases to 40. This is a regression and unit-contract result;
the broad W convergence audit and observable-count decision remain open.
