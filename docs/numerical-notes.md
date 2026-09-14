# Numerical notes for SASHIMI-W

The original module imports, public classes, tuple order, units and positional
arguments remain supported. See the example notebook for catalog usage and
[tests](../tests/README.md) for installation checks.

## Changes affecting results

- The Viel transfer amplitude is squared to obtain the power ratio:
  `P_WDM / P_CDM = (1 + (alpha*k)**(2*nu))**(-10/nu)`. Both the sharp-k variance
  and concentration top-hat calculation use this ratio, with the existing
  coefficients and WMAP7 parameters. A transfer amplitude of one half means a
  power ratio of one quarter. Earlier q5 results are not interchangeable.
- The background satisfies flatness using total matter, including baryons.
  Growth and its derivative are consistent; `dS/dM` scales with growth squared.
- Physical mass is converted consistently to the spectrum table's mass units.
  Sharp-k variance, its moving-boundary derivative and small variance differences
  use the same finite integral, resolving the original spectrum knots.
- EPS and its mass integral use the evolved accretion mass at each redshift.
- Concentration evaluates real formation trials, including supported finite
  future times. A safeguarded NFW inverse preserves the disruption threshold.
- Displayed populations contain current survivors. Counts use strict `>`
  thresholds, include the first bin, honor `profile_change`, and preserve input
  arrays. `accretion=True` classifies current survivors by accretion mass.
- Integration uses the modern SciPy Simpson endpoint rule. This can differ from
  the removed `simps` rule for an even number of samples.

Empty populations below the cutoff have zero weights. Unsupported physical
hosts raise errors. In particular, the native concentration/history failed for
a 2 keV particle and a `1e8 Msun` host over the tested redshift range; changing
the stripping solver does not supply that missing host history. Concentration
search branches and future-formation boundaries can remain discontinuous.
These changes do not recalibrate the transfer function or observational limits.

## Tidal stripping solver

The default is `method="picard_table"`. Use `method="odeint"` to select the
previous default explicitly, or `method="dop853"` for direct log-mass integration.
The supported names are `picard_table`, `dop853` and `odeint`.
Unknown methods or options raise errors.

Endpoint tables use 48 accretion-redshift nodes, 32 log-mass-ratio nodes,
129 integration points, cubic interpolation, three nonlinear updates and a fourth
convergence check.

The automatic table envelope is `0 <= z_obs <= z_acc <= 7` and
`-24 <= log10(ma/Mvir(z_acc)) <= 3`. Outside it, or after a failed convergence
check, `PicardFallbackWarning` and `solver._picard_events` record direct-ODE
fallback. Queries do not silently extrapolate; invalid physical input raises.
Caches include the host/background state, final redshift, numerical options and
particle settings where applicable. Wrapper calls rebuild after state changes.

Each variant supplies its own host history, background and stripping coefficients.
The standalone helper retains the MIT notice from SASHIMI-C PR #5, commit
`88ae730762fb153be7a7433bb563b0b8ab3ec2c2`.

## Validation scope

The checked domains met a `1e-3` relative mass-error gate against refined
independent ODE integrations. This is a finite-domain numerical check, not a
bound on changes from older approximations or a physical calibration.
Simulation agreement and downstream inference require separate validation.

The [historical validation report](https://github.com/gomeshun/sashimi-w/blob/0a997abbf031b9630769b81d286b504a3c59bae3/docs/standalone-maintenance.md)
records the evaluated grids, reference methods, timings and limitations and
links to the full development evidence at that fixed commit. Those generated
reports and intermediate arrays are not needed in a working checkout.
