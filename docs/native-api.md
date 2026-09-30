# Native WDM API

`WDM` owns immutable process settings and creates fresh model, variance,
concentration and execution state for each population.

```python
from sashimi_w import WDM

model = WDM().configure(
    dark_matter={"mass_keV": 2.0, "power_convention": "standard-t2-q10"},
    accretion={"mass_nodes": 40, "redshift_step": 0.1,
               "host_history_mode": "quadrature", "host_history_nodes": 20},
    concentration={"scatter_dex": 0.128, "quadrature_nodes": 5},
    stripping={"solver": "odeint", "solver_options": {"rtol": 1e-8}},
    disruption={"ct_threshold": 0.77},
)
catalog = model.population(
    host_mass_msun=1e12, host_mass_definition="200c", host_mass_redshift=0.0,
    redshift=0.5, accretion_mass_range_msun=(1e6, 1e10),
    accretion_mass_definition="200c", accretion_redshift_range=(0.5, 3.0),
)
```

The result is the existing `WeightedSubhaloCatalog`, using Msun/Mpc/Msun per
Mpc3. The old tuple still uses Msun/kpc/Msun per pc3, and its weight excludes
survival. Existing aliases and calculation defaults are retained.

## Physical coupling and accepted model

One particle-mass/q10 specification controls both sharp-k variance and the
Ludlow top-hat concentration calculation. There is no independently adjustable
concentration transfer, retired q5 mode, or arbitrary relation callback. The
calibrated WMAP7 cosmology (.27, .7) is fixed. `host_mass_redshift` must be zero;
this adapter does not pretend to implement a target-epoch host inversion.
Input mass definitions are `200c`; the existing redshift-dependent virial
conversion is used in the EPS measure and evolved nodes.

The unchanged defaults are particle mass1.5 keV, q10, mass_nodes100,
redshift_step0.1, host_history_nodes200, sigmafac0, concentration scatter0.128
and quadrature_nodes5, odeint with SciPy defaults and100 output points,
profile_change=True, and strict c_t>0.77 survival. Default accretion mass support
is 10 Msun to0.1 times host M200c(z=0).

## Host-history mode is a physical choice

`host_history_mode="quadrature"` requires at least two host_history_nodes and
sigmafac=0. `"deterministic"` requires exactly one node and permits a finite
sigmafac offset. A mode-only change that leaves incompatible inherited values
fails with an explicit instruction to change/reset them. No ignored sigmafac
is silently retained in quadrature mode.

The preserved deterministic accretion law anchors its low-z scatter at the
**first sampled z>1**; the tidal history uses z=1. Both are recorded separately.
On this pinned migration branch, a deterministic population without a sampled
z>1 is rejected before execution,
even for sigmafac0. These are existing scientific/numerical conventions, not
silently unified prescriptions.

The pre-existing nonsquare 2D mass-axis reshape defect in this non-default
branch is corrected separately: see [dimension-contract correction](deterministic-mass-axis.md).
Its 1D/square-grid outputs and the default quadrature branch remain unchanged.
This is a minimal migration backport of a known main/PR17 correction; upstream
also changes exact-z1 anchoring and EPS active-support/gap arithmetic. Those
scientific changes remain explicitly outstanding, so this adapter does not
claim main-equivalence.

## Solver and disruption controls

The only supported native stripping solver is odeint. Scalar rtol/atol/h0/hmax/
hmin and integer mxstep/mxhnil/mxordn/mxords are routed to the existing ITAMAE
controller, not stored as unused labels. Callbacks, reserved options, positive
h0, invalid orders/tolerances, or hmin>positive hmax are rejected. Defaults retain
the original arithmetic. Explicit-control results are checked against
independent DOP853 integration of the same W-owned physical RHS.

The configured ct_threshold reaches the existing WDMSurvival component; its
mask and factorized final weight are tested directly. This exposes an existing
physical selection boundary without changing the default threshold.

## Immutable configuration and integration domains

configure returns a new specification, partially overriding process dictionaries.
Inputs and nested options are copied/frozen; resolved_settings is read-only.
Empty solver_options clears old options; nonempty options partially override.
All caches belong to freshly constructed per-run objects.

Explicit redshift support is (lower,upper], with lower>=target and at least two
nodes; no sampled node exceeds upper. Omitted support retains the historical
arange-to7 behavior, including off-grid endpoint behavior. Actual nodes, requested
bounds and policy are recorded alongside settings/configuration hash, solver
controls, host branch/anchors, coupled power/variance identifiers and units.

## Verification

Independent pre-edit records use W dcef1910d42cab940567be448ffc79b42d436802 and
core23d01e8758a88b061b87de9e488c38ec89fd8e4f. They cover nonzero output epoch,
particle mass, both host branches, sigmafac offset and profile-off. The
nonsquare fix has a separate per-redshift 1D reference from the untouched
installed baseline. Old/new same-runtime comparisons are exact; frozen output
comparisons allow the existing reference tolerance. Configuration changes are
also checked for simultaneous variance/concentration propagation and state
isolation. These W semantics are not interchangeable with every family variant.

### ODE reference diagnostic: discontinuous concentration sampling

The historical concentration routine picks the nearest of 100 mass nodes.
Consequently the host-dependent RHS can jump when that index changes. In the
small host1e10 Msun/particle2keV/z0.5-to0 diagnostic, knots occur near z0.370986
and0.0013365, with RHS jumps around7e-5. Unsegmented DOP853 comparisons were
nonmonotonic under tighter tolerances/max-step values; tighter rtol alone is
not evidence of global accuracy across these jumps.

A separate reference split exactly at these known knots is recorded with
refinement and default-solver comparisons in
[the diagnostic record](validation/native-ode-boundaries.json). The API
option-routing test uses an independently checked knot-free z0.1-to0.3 interval
and retains its strict2e-7 comparison threshold. It verifies controlled solver
agreement and option routing, not convergence over all physical domains. No
historical concentration smoothing, physical law or default solver is changed.

The reduced-mass, omitted-full-z-domain regression emits existing RuntimeWarning
categories from intermediate historical kernels (overflow/invalid/division).
The test checks native/legacy warning category/message parity and finite final
columns/weights. Runtime warnings are not suppressed by the API, and this
regression is not described as warning-free physical-domain validation.


### Frozen-reference transport across runners

The unchanged pinned migration was independently installed and executed on each
CI matrix runner before testing the native API. On some runners it is exactly elementwise
equal to the original fixtures; on others its maximum relative deviations are
3.42e-10 for evolved mass, 1.42e-10 for scale radius, 8.97e-11 for scale density,
8.33e-12 for truncation ratio and 1.66e-9 for EPS weights (the last has a maximum
absolute difference of 4.66e-21 in the affected sigma-offset case). Mass/redshift
inputs, masks, zero support and quadrature/survival factors remain exact.
[Per-field measurements and software versions](validation/native-reference-transport.json)
record the unchanged source/core pins and the three runner reports.

The saved cross-environment comparisons therefore use bounded field-specific
relative tolerances: 5e-10 for evolved mass, 2e-10 for evolved radius/density,
2e-11 for truncation ratio and 3e-9 for EPS weights; remaining fields retain
5e-12. This is a fixture-transport allowance, not a scientific accuracy claim.
No physical calculation or solver tolerance was changed. The exact low-level
hardware/math-library cause is unresolved; Python version alone did not explain
which runners differed.

CI additionally requires exact elementwise equality to independently calculated pinned
old-source arrays from that same runner, alongside exact current legacy/native
parity. Missing requested runner references fail rather than silently skipping
the comparison. Original checked-in fixtures and their hashes are unchanged.
