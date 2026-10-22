"""Separate the WDM cutoff and power-amplitude changes on one fixed grid.

Select the baseline or changed standalone checkout with PYTHONPATH. The
filter-only case retains the baseline EPS and concentration amplitudes;
the normalization-only case retains the old cutoff. This is a diagnostic
ablation, not a supported alternative product prescription.
"""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import sashimi_w as w


def diagnostic_cutoff(self, mass):
    """Set one diagnostic mass assignment while keeping the power amplitude fixed.

    Parameters
    ----------
    self : SharpKVariance
        Already normalized variance object with a diagnostic_factor attribute.
    mass : array_like
        Physical halo mass in Msun.

    Returns
    -------
    numpy.ndarray
        Wavenumber in h/Mpc from diagnostic_factor/R_th.

    Notes
    -----
    Only the cutoff changes. The precomputed raw power integrals and both
    concentration and EPS amplitude divisors retain their selected values.
    """
    radius = (3 * np.asarray(mass) * self.h / (4 * np.pi * self.density))**(1 / 3)
    return self.diagnostic_factor / radius


def main():
    """Measure changes in surviving counts with identical population controls.

    Notes
    -----
    The fixed 1.5 keV, 1e12 Msun host calculation uses 128 accretion masses,
    dz=0.2 to z=7, three concentration nodes and twenty host-history nodes.
    Counts are summed catalogue weights for strict bound-mass thresholds.
    No simulation matching or grid convergence is inferred from this grid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("old", "filter", "normalization", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = w.subhalos(1.5)
    if args.variant in ("filter", "normalization"):
        model._variance.diagnostic_factor = 2.5 if args.variant == "filter" else 2.5 * (9 * np.pi / 2)**(1 / 3)
        model._variance._cutoff = MethodType(diagnostic_cutoff, model._variance)
    parameters = dict(M0=1e12, redshift=0.0, dz=0.2, zmax=7.0, N_ma=128,
                      N_herm=3, N_hermNa=20, logmamin=6, logmamax=11,
                      method="picard_table")
    catalog = model.rs_rhos_calc(**parameters)
    weight = catalog[8] * catalog[9]
    report = {"variant": args.variant, "module": str(Path(w.__file__).resolve()),
              "parameters": parameters, "mass_wdm_keV": 1.5,
              "eps_amplitude_divisor": float(model.normalise),
              "c200_M1e10_z0": float(model.conc200(1e10 * w.Msolar, 0.0)),
              "counts_bound_mass_gt": {str(m): float(np.sum(weight[catalog[4] > m]))
                                       for m in (1e6, 1e7, 1e8, 1e9)}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
