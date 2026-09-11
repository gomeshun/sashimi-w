# Standalone main maintenance, 2026-09-11

This work is on `minor-updates`, based on main `99dfc3632eec0126080c0273ddf78f84fe09216c`. It keeps the original module import, public classes, tuple order, units and positional arguments. It introduces no ITAMAE runtime dependency, package-layout migration, main merge or release.

## Corrections

- W-0: replace removed NumPy/SciPy APIs and explicitly use the modern even-sample Simpson endpoint rule. The independent quadrature patch separates that numerical change from the other corrections.
- W-1: restore the flat WMAP7 background, normalized growth and its matching derivative.
- W-2/3: use physical mass consistently in the native sharp-k variance, evaluate the same finite-domain integral and its moving-boundary derivative, and include the correct growth square. Quadrature resolves every original linear spectrum-interpolation knot; no sign projection is used.
- W-4: use each accretion-redshift virial mass row in both EPS and its logarithmic mass integral.
- W-5: evaluate only real concentration-formation trials while preserving finite future-redshift trials, and use an accurate NFW inverse.
- W-6: all displayed populations contain current survivors. Counts use exact strict `>` thresholds, include the first bin, forward `profile_change`, and leave caller arrays unchanged.
- W-7: apply the approved Viel q10 **power** ratio to both sharp-k and concentration top-hat calculations, retaining the original coefficients and WMAP7. Amplitude-half means power one quarter; power-half means one half. This branch adds no q5 product mode and performs no observational-limit conversion.

Empty cutoff populations have exactly zero weights. EPS support and very small variance gaps are evaluated directly; a physically invalid host/concentration still raises instead of being clipped.

## Tidal solver

The default is `method="picard_table"`; `method="odeint"` explicitly selects the previous solver. Other existing named solvers remain available. `method="dop853"` integrates log mass directly. Unknown solver options raise rather than being discarded.

The standalone numerical helper derives from SASHIMI-C PR #5, commit `88ae730762fb153be7a7433bb563b0b8ab3ec2c2`, with its MIT notice retained. Each variant supplies its own host history, background and stripping coefficients. The four copies have matching numerical code; no variant imports another.

Endpoint tables use 48 accretion-redshift nodes, 32 log-mass-ratio nodes, 129 integration points, cubic interpolation and three nonlinear updates plus a fourth convergence check. The saved initial linear-table study and resolution/iteration comparisons remain available. Table build, first lookup and reuse are measured separately from the whole catalog.

Caches belong to the solver and include host/background state, final redshift, numerical options and particle settings where applicable. Stale tables raise; wrapper calls rebuild after state changes. Queries never silently extrapolate. The automatic envelope is `0 <= z_obs <= z_acc <= 7` and `-24 <= log10(ma/Mvir(z_acc)) <= 3`. It was extended above the initial +2 probe after inspecting the high-redshift native W catalog; the error gates were unchanged. Outside that envelope or after a failed iteration check, `PicardFallbackWarning` and `solver._picard_events` record a direct-ODE fallback. Invalid physical input raises.

## Numerical acceptance

Gates were recorded before the pilot in [`picard-gates.json`](../validation/maintenance/picard-gates.json). Final decisions are machine checked by [`gate_audit.py`](../validation/maintenance/tools/gate_audit.py), with results in [`acceptance.json`](../validation/maintenance/acceptance.json).

| Measurement | Result | Gate |
| --- | ---: | ---: |
| Maximum solver mass relative error | 0.000266883 | 0.001 |
| Refined independent-reference disagreement | 3.18919e-09 | 1e-8 |
| Largest relative smooth-observable difference, representative catalog | 1.49605e-05 | 0.01 |
| Whole-catalog median speedup | 2.921x | 1.05x |

The independent reference solves delta log mass using DOP853 at `rtol=1e-11, atol=1e-12`, checked against `3e-13, 3e-14`. Known concentration branch boundaries are segmented; original unsegmented failures remain in the historical reports. The mass gate also covers SI's intermediate times. Host probes include 1e8/1e12/1e15 Msun, observed redshifts 0/0.5/2, accretion boundaries and the full required mass-ratio envelope. Native unsupported domains are recorded explicitly.

The same-grid catalog checks cover mass functions, survivor/satellite counts, bound mass fraction, Vmax, available n=0 boost, and SI core/collapse indicators. Particle/cross-section variations and a simultaneous mass/redshift/concentration-grid and solver refinement are saved. [`picard-boundaries.json`](../validation/maintenance/picard-boundaries.json) lists changed boundary nodes and the closest positive-weight nodes. No boundary mask changed in these catalog comparisons; this is a finite-grid observation, not a general proof about all thresholds.

The benchmark uses three fresh processes per solver on the same host and environment. Whole-catalog time includes construction and table build; peak RSS is measured before a separate first/reused solver call. Configurations and all repetitions are in [`picard-benchmark.json`](../validation/maintenance/picard-benchmark.json). They are representative full catalogs at the disclosed resolution, not timings of every public default setting. Ordinary benchmark catalogs use no fallback.

## Evidence and reproduction

- [`three-stage-comparison.json`](../validation/maintenance/three-stage-comparison.json) and arrays separate frozen main, corrected legacy solver, and corrected Picard. Differences from the old perturbative result can exceed the independent-ODE error gate; that gate compares Picard to the refined ODE on the corrected prescription.
- `A-config.json`, `A.json`, `A.npz`, input/patch digests and the isolated reference controller retain the frozen source and dependency provenance. Bug-fix tests use independently patched main, not an imported migration product.
- `picard-complete-range-domain.json`, `picard-final-catalog.json` / `picard-accepted-catalog.json`, scenario records and `picard-grid-refinement-result.json` hold the final numerical comparisons. Earlier pilot and failed-refinement outputs are retained as historical evidence.
- `example-execution.json` records execution of original main notebook cells with disclosed grid reductions while retaining imports, tuple unpacking and observable calls. C's recursive boost-table generation is excluded; n=0 boost is exercised. F's example includes its existing explicit Colossus comparison and CDM sentinel.

From the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
PYTHONPATH=. python validation/maintenance/tools/run_example.py --variant w
PYTHONPATH=. python validation/maintenance/tools/gate_audit.py --variant w
PYTHONPATH=. python validation/maintenance/tools/benchmark.py --variant w --output validation/maintenance/new-benchmark.json
```

Python 3.10–3.13 are covered by standalone CI. Local dependency versions and test results are recorded with the final validation artifacts. The historical A/B environment is separately pinned; it is not a runtime requirement. Numerical regression, convergence and speed do not establish simulation calibration, observational constraints or downstream inference validity.

The 2 keV, 1e8 Msun host does not support the original native concentration/history over the requested range. Its failed domain is retained; it is not filled by Picard. The unchanged concentration algorithm has finite future-formation/search-branch transitions. The independent reference locates those transitions and checks integration refinement without changing the physical prescription. The 0.5 keV small-host/empty-population reference failures are likewise retained. Fourteen independent correction ablations (including q5 vs q10 and Simpson endpoint rules) are in `validation/maintenance/ablations` and `ablation-effects.json`.
