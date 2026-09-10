# Standard API migration

| Previous entry | Standard behavior |
| --- | --- |
| sashimi_w.subhalos without a convention | Subhalos/subhalos defaults to the adopted q10 thermal WDM prescription |
| sashimi_w_itamae.subhalos | Alias of the standard Subhalos class |
| physics_mode=legacy or consistent | Removed; raises TypeError |
| rs_rhos_catalog_calc | Primary named catalog via PopulationComponents |
| rs_rhos_calc | Same calculation, ten-element tuple format |
| catalog_from_legacy | catalog_from_tuple, format conversion only |
| make_variance_model | Callable adapter of the standard continuous variance |
| make_integrated_variance_model | Same physical integral, configurable numerical resolution |

The duplicate population loop and old growth/derivative calculations are removed.
WDM physical kernels are explicitly owned by WDMPhysics; the public class never
inherits an old population class. The now-unused 500-step historical sharp-k
initialization and interpolated sigma arrays are removed. Concentration's
separate top-hat calibration and its numerical inputs remain unchanged.
Old source archives and their dependencies belong to independent A/B validation.

New provenance uses itamae:calculation:v2 with the calculation specification,
input grid, solver settings, source revisions, power/variance identity and
physical units. Historical fixture mode fields are kept exactly as generated.
New cache identities cannot be confused with historical mode-based artifacts.

On 2026-09-10, all 35 standard-API tests pass, including both independent q5/q10
full catalogs at unchanged 5e-10 / 5e-12 tolerances. Removed test cases exercised
product legacy coexistence; equivalent historical arrays remain independently
archived. The maintained suite checks standard imports, removed-argument
errors, tuple conversion, units, serialization, physical invariants and the
common executor's node/weight transport. No golden data was overwritten.

Still open: user selection for existing observable counts (survival and the
first cumulative bin), broad convergence, final versions and distribution
validation. These are required before release-preparation completion.

Both usage and scientific notebooks were executed in fresh Python 3.11 kernels.
The first science execution exposed a source-bundle lookup assumption: Hatch's
editable force-included modules can live in site-packages. The science runner
now starts in the explicit validation bundle; the usage runner still executes
in an unrelated temporary directory. The plots were visually inspected.

The 2026-09-11 q10 adoption supersedes the earlier two-choice API. See [q10 migration](q10-adoption.md). Historical tests and q5 artifacts below describe their original revisions.
