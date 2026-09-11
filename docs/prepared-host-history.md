# Prepare the fixed WDM host history once

Profiling the small q5 catalog found about 6.7 of 8.1 instrumented seconds in
repeated variance evaluations for the same host mass and reference redshift.
The WDM history component now prepares those coefficients once at construction
and evaluates their original expression at each ODE time. The physical formula
and arithmetic grouping stay in WDMPhysics; the component owns only the
prepared inputs for its fixed host.

A new history component is constructed for every population calculation, so
this is not a persistent model cache. Model/configuration changes between
catalog calls receive fresh coefficients. As with the rest of one calculation,
mutating its scientific model concurrently during execution is unsupported.

Full q5/q10 catalog comparisons use the clean d5b7f17 product as the before
reference, with identical ITAMAE d752306 and original numerical tolerances.
The one-factor convergence experiment is recorded separately in the parent.
A scratch top-hat table-cache prototype yielded only a modest improvement and
was not adopted; no table caching or interpolation change is included here.
