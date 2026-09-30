# Native WDM integration with the corrected EPS migration

PR19 merged into `itamae-migration` as
`65f25390f7d745d671818ba904c59b133ce1aa29`. This native API branch incorporates
that merge while preserving its existing configuration, solver controls,
component execution, shape validation and immutable settings. It uses an ordinary
merge; the earlier native history and historical fixtures remain intact.

## Runtime changes

- Keep the 1D/2D mass-grid validation from the earlier native shape repair
- Retain PR19's strict EPS support, direct variance gaps, normalized zero-gap
  limit, log-space kernel, explicit host-node axis and zero-population guard
- Remove the native wrapper's obsolete sampled-z>1 rejection
- Report the deterministic accretion scatter anchor as exactly 1.0, separately
  from the unchanged exact-z=1 tidal anchor

Deterministic domains entirely at or below z=1 now run. Tests cover nonzero
negative and positive scatter offsets as well as zero. Common accretion-rate
rows are invariant when the redshift grid is extended; integrated catalog
weights are not expected to remain unchanged when the integration domain grows.
Quadrature behavior and its sigmafac=0 policy are unchanged.

## Three distinct reference checks

1. Current native vs current compatibility API: exact field/shape/array equality
   for all six compact cases
2. Independently installed corrected migration `65f2539` on each CI runner:
   exact columns, all weight factors and independently calculated nonsquare
   rate rows. The generator asserts the installed source revision, unchanged
   ITAMAE pin and installation path
3. Historical migration `dcef191` and saved fixtures: original files/hashes and
   transport evidence are preserved. The old source remains an exact same-runner
   structural reference. Its changed EPS weights are not an exact current oracle

Existing saved-file transport tolerances are unchanged. Two small new references
live in `tests/references/native-api-eps-baseline`: corrected deterministic
`sigma_offset` and the original nonsquare inputs evaluated independently one row
at a time by the corrected source. The row test retains rtol=5e-12 and atol=0;
its CI runner reference additionally requires exact equality.

[The integration record](validation/native-eps-integration.json) quantifies the
old-to-corrected change. The deterministic offset case changes total survivor
count by -2.2618% and bound mass fraction by -2.2623%. Its structural fields and
survival decisions are unchanged. The other five compact counts differ by at
most 6.9e-13 relative in the recorded environment. Two very small nonsquare
rates change by at most 2.94e-11 relative, or 1.35e-39 absolute. These are explicit
corrections, not tolerance relaxation or revised physical calibration.

The [earlier EPS comparison](validation/eps-backport.json) remains the record
of PR19's runtime source hashes and controlled before/after checks. The native
branch adds mass-input validation, so that historical report is not presented
as a hash manifest for the entire integrated native branch.

## Reproducing the runner gates

Install old migration into `.baseline-install`, corrected migration into
`.corrected-install`, and the candidate separately. With the pinned ITAMAE:

```sh
PYTHONPATH="$PWD/.baseline-install" python scripts/check_native_baseline_transport.py --output /tmp/w-old
PYTHONPATH="$PWD/.corrected-install" python scripts/generate_corrected_native_reference.py --output /tmp/w-corrected
SASHIMI_W_BASELINE_REFERENCE_DIR=/tmp/w-old SASHIMI_W_CORRECTED_REFERENCE_DIR=/tmp/w-corrected python -m pytest -q
```

This does not merge or release the native API PR. The numerical scope remains
[the retained fixed-gap WDM EPS approximation](eps-stability.md), with unchanged
transfer/cosmology/calibration and no new moving-barrier model.
