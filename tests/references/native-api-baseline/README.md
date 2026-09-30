# Independent native-API regression records

These reduced-grid records were generated before implementation from separately
built and installed unchanged migration sources. Each JSON records constructor
and population inputs, exact variant/core source revisions and output SHA256.
NPZ archives contain named columns and weight factors (including both states
where the model returns paired catalogs). They are not raw production run
archives or independent new physical calibration.

Tests compare the new and old paths exactly within one runtime. Comparisons
against saved float arrays use rtol5e-12 with only a subnormal absolute tolerance;
boolean masks are exact. Original independent scientific references are retained.


After PR19 integration these files remain historical. The unchanged migration
is still an exact same-runner structural reference; EPS weight equality uses a
separately installed corrected migration at 65f2539. The deterministic offset
has a new corrected fixture alongside these records, not in place of them.
Original field-specific transport allowances are retained; see
`docs/native-eps-integration.md` for the current gates.
