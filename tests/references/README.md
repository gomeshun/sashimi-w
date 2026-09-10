# Independent corrected reference B

These q5/q10 fixtures were generated from frozen SASHIMI-W
`99dfc3632eec0126080c0273ddf78f84fe09216c`, in a separate Python 3.10.20 process
with NumPy 1.22.4 and SciPy 1.10.1. Full patches, immutable export controller,
dependency/input hashes, configurations and single-correction ablations live in
`sashimi-family/validation/references/sashimi-w`. Each sidecar records its role,
patch hashes, source/input hashes and complete parameters. No current W or
ITAMAE implementation is imported by B.

The six physical/unit corrections are followed by independently integrated
continuous sharp-k variance and its moving-boundary derivative, the analytic
principal-branch Lambert-W NFW inverse at 50 digits, and the modern Simpson
last-interval rule derived as a quadratic integral. q5/q10 are explicit choices.

Tests retain the existing catalog rtol 5e-10 and weight rtol 5e-12. Maximum
observed differences are 7.75e-11 for bound mass (different SciPy ODE versions),
3.33e-11 for evolved profile scales, and 9.94e-13 for weights (cancellation of
nearby variances in EPS). Both survival masks agree. The historical consistent
fixtures in `tests/golden` are not overwritten or relabeled; their old numerical
specification is superseded by B for current product regression.
