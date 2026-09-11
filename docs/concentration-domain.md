# Concentration candidate domain

The historical Ludlow concentration inversion evaluates 100 trial concentrations.
At low redshift several trials have a nonpositive radicand in the formation-time
relation. The fractional power produced NaN (or z=-1), which was passed into
growth before the resulting non-finite interpolation candidates were dropped.
ITAMAE's strict public cosmology boundary correctly rejects those inputs.

The W-owned concentration kernel now removes the same unsupported trial points
before taking the fractional power and calling growth. The exponent 0.3333,
trial concentration grid, valid candidate values/order, final interpolation and
all physical choices are preserved. Finite future redshifts -1 < z < 0 remain
supported; this is not a new physical formation-redshift cut.

Before the fix, the existing W suite with the strict core had one failure and
eight setup errors. After the fix, all 37 tests pass, including full independent
q5/q10 catalogs at unchanged rtol=5e-10 for structures and 5e-12 for weights,
exact survival masks, scalar/matrix concentration, units and serialization.
New direct-domain tests retain exactly the analytically supported candidates.
The measured suite runtime was 78.58 seconds in the recorded local environment.
No count/survival policy, q default, ODE solver or physics parameter changed.
