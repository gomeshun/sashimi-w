"""Common WDM power amplitude for EPS and concentration calculations.

The bundled spectrum is linear in its tabulated k coordinate. Its q10 WDM
suppression is normalized once by a real-space top-hat of radius 8 Mpc/h.
Changing the halo smoothing window must not change this physical amplitude.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

SHARP_K_MASS_ASSIGNMENT = 2.5
POWER_NORMALIZATION = "wdm-top-hat:R=8-Mpc/h:z=0:v1"


def top_hat_window(x):
    """Evaluate the spherical top-hat Fourier window, including its origin.

    Parameters
    ----------
    x : array_like
        Dimensionless product of wavenumber and smoothing radius, k R.

    Returns
    -------
    float or numpy.ndarray
        W(x)=3[sin(x)-x*cos(x)]/x**3, with W(0)=1 and the input shape.

    Notes
    -----
    A series through x**6 avoids cancellation for abs(x)<0.01. This is a
    stable evaluation of the same window, not a different smoothing model.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 0.01
    squared = x[small] ** 2
    result[small] = 1 - squared / 10 + squared**2 / 280 - squared**3 / 15120
    result[~small] = 3 * (np.sin(x[~small]) - x[~small] * np.cos(x[~small])) / x[~small] ** 3
    return float(result) if result.ndim == 0 else result


def sigma8_normalization(k, power, alpha, sigma8, *, radius=8.0, order=16):
    r"""Return the single amplitude divisor for the q10 WDM power spectrum.

    Parameters
    ----------
    k : array_like, shape (n,)
        Strictly increasing positive table wavenumbers in h/Mpc.
    power : array_like, shape (n,)
        Finite nonnegative CDM table power in (Mpc/h)**3, interpolated
        linearly in k between the supplied knots.
    alpha : float
        Nonnegative Viel suppression length in Mpc/h; alpha=0 is CDM.
    sigma8 : float
        Positive target rms density contrast at redshift zero.
    radius : float, optional
        Top-hat radius in Mpc/h. The WDM model uses 8, independently of the
        sharp-k halo mass assignment.
    order : int, optional
        Gauss-Legendre nodes in log(k) in each original table interval.
        Increase this to check quadrature convergence, not to change sigma8.

    Returns
    -------
    float
        N=sqrt(integral[dln(k) k**3 P_CDM(k) T(k)**2 W(k R)**2 /
        (2*pi**2)])/sigma8. Both EPS and concentration use P_WDM/N**2.

    Raises
    ------
    ValueError
        If the table, physical inputs, quadrature order or integrated power
        are invalid. No power outside the supplied table is invented.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    T(k)**2=[1+(alpha*k)**(2*1.12)]**(-10/1.12). The radius belongs to
    the conventional sigma8 definition, not to M=(4*pi/3)*rho*(2.5 R)**3.
    The finite table and fixed quadrature have numerical limits; agreement
    with an independent integral does not calibrate the halo model.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """
    k, power = np.asarray(k, dtype=float), np.asarray(power, dtype=float)
    if (k.ndim != 1 or k.size < 2 or power.shape != k.shape
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(power))
            or np.any(k <= 0) or np.any(np.diff(k) <= 0) or np.any(power < 0)):
        raise ValueError("Power table requires increasing positive k and aligned nonnegative power.")
    if (not np.isfinite(alpha) or alpha < 0 or not np.isfinite(sigma8)
            or sigma8 <= 0 or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Require alpha>=0, sigma8>0 and radius>0, all finite.")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("Quadrature order must be an integer of at least two.")
    nodes, weights = leggauss(order)
    edges = np.log(k)
    half = np.diff(edges) / 2
    logk = (edges[:-1] + edges[1:])[:, None] / 2 + half[:, None] * nodes
    wave = np.exp(logk)
    suppression = (1 + (alpha * wave) ** (2 * 1.12)) ** (-10 / 1.12)
    integrand = wave**3 * np.interp(wave, k, power) * suppression
    integrand *= top_hat_window(wave * radius) ** 2 / (2 * np.pi**2)
    variance = np.sum(half * np.sum(integrand * weights, axis=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The top-hat normalization requires positive finite variance.")
    return float(np.sqrt(variance) / sigma8)
