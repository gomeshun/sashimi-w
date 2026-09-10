# Resolution limits carried into peer review

On 2026-09-10 the user approved retaining the existing redshift step (`dz=0.1`).
Changing that default is not a completion condition for this release preparation.
The observed resolution dependence and the following convergence procedure remain
part of the review evidence. This is a scope decision, not an accuracy certificate.

The measured sweep fixes M0=1e12 Msun at z=0, WDM particle mass=2 keV, zmax=7,
log10 accretion mass/Msun from 5 to 10, N_ma=16, N_herm=3, N_hermNa=4,
sigmalogc=.128, sigmafac=0 and profile_change=True. q5 and q10 are independent
explicit prescriptions. The data use clean SASHIMI-W
`dcc4379186174a016924d88b362fbe40d1ac8f88` and ITAMAE
`d752306e62bf295b7a96100b9cc022f8f9c61071`, Python 3.11.15,
NumPy 2.4.6 and SciPy 1.17.1. Each full catalog and its input/output hashes are
preserved in the [frozen family bundle](https://github.com/gomeshun/sashimi-family/tree/62388f2/validation/references/sashimi-w).

| dz | q5 surviving count | q5 bound mass fraction | q10 surviving count | q10 bound mass fraction |
|---:|---:|---:|---:|---:|
| .1 | 15.18764449 | .01082776482 | 3.79071667 | .00456969998 |
| .05 | 15.61600465 | .01186202742 | 3.89828945 | .00501605815 |
| .025 | 15.82940567 | .01240881362 | 3.95259344 | .00525538203 |
| .0125 | 15.93793800 | .01269002232 | 3.98023699 | .00537685086 |

Surviving count is the sum of catalog weight times survival; bound mass fraction
is sum(m_bound × surviving weight)/M0. The subsequently adopted [observable correction](surviving-observables.md)
uses this same surviving population; the stored convergence catalogs are unchanged.

Changing only dz=.1 to .0125 increases count by 4.940%/5.000% and bound mass
fraction by 17.199%/17.663% for q5/q10. The final .025→.0125 refinement still
changes count by about .7% and mass fraction by about 2.3%. The finest run is a
finite-grid comparator, not continuum truth. Because the other coordinates use
representative reduced settings, these numbers are not estimates of the error
of the full default configuration, and must not be used as universal tolerances.

Separate one-factor runs at the coarser dz=.5 also vary mass and host-history
quadrature. N_ma=256→500 changes count by -0.0131%/-0.0670% and mass fraction by
-0.7242%/-0.8728%. N_hermNa=64→200 changes count by +0.0412%/-0.1950% and mass
fraction by -0.0476%/-0.0934%. These effects cannot simply be added to the
redshift effects, since all factors were not refined jointly. The original
coarse sweep additionally varies concentration quadrature; its results and
conditions are in the same frozen bundle. No model setting has been changed.

## Reproduction and use

1. Recover the exact clean W/core SHAs and the pinned numerical environment from
   the configuration/sidecar. Install that core alone as the worker's `itamae`.
2. From the family validation bundle, run `scripts/product_worker.py` with
   `--source` pointing to the clean W checkout, `--config` selecting a
   `C-dcc4379-convergence-q*-dz*.json`, and a new `--output` location. The worker
   rejects dirty or incorrect source revisions. Repeat all four steps for each
   q convention without modifying other settings.
3. Run `scripts/check_w_convergence.py --root validation/references/sashimi-w
   --output <new-audit.json>` to verify the stored full-catalog hashes and
   independently recompute all reported metrics. The committed audit verifies
   all eight redshift catalogs. The scientific notebook plots their checked
   summary; it does not rerun the expensive sweep.
4. Before a scientific application, select its observable-specific accuracy
   target, halve dz successively at fixed other settings, then refine mass and
   both quadratures independently. Finally verify the selected settings jointly
   and compare at least two refinements. State residual finite-grid changes;
   do not rename agreement with a coarse fixture as convergence.

The observed numerical trends are suitable to share with these caveats. They do
not establish agreement with simulations or determine a preferred q convention.
