# Repeated WDM calculations

The tidal ODE right-hand side previously evaluated the identical host virial
mass five times at the same redshift. Each evaluation includes concentration
integration and a mass conversion. It now evaluates that mass once per call,
preserving arithmetic grouping. The fitted mass conversion also evaluates its
identical concentration once instead of twice. No value is cached across calls;
changes to model parameters or interpolation inputs remain visible immediately.

Independent q5/q10 full-catalog references, exact before/after source comparisons
and runtime measurements are recorded by the parent product-comparison worker.
This changes execution cost only; it does not change the WDM prescription,
quadrature resolution, solver, survival or the pending count-policy decision.
