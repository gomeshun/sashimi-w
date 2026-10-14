"""Standalone W variance helper and NFW inversion from ITAMAE 23d01e8.

MIT License

Copyright (c) 2026 Shunichi Horigome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from sashimi_w_power import SHARP_K_MASS_ASSIGNMENT, sigma8_normalization


class SharpKVariance:
    r"""Finite sharp-k integral on every original, linearly interpolated P(k) cell.

    Sixteen-point Gauss-Legendre integration in log(k) resolves each input
    spectrum knot. Partial cells use the same rule; no mass interpolation or
    projection is applied. Physical masses are in Msun, k in h/Mpc.
    The cutoff is 2.5/R_th, where R_th encloses the physical halo mass.
    The amplitude is set independently by a top-hat at 8 Mpc/h.

    Parameters
    ----------
    k, power : array_like
        Original k and CDM power table in h/Mpc and (Mpc/h)**3.
    alpha : float
        Viel thermal suppression length in Mpc/h.
    density : float
        Numerical mean density in Msun*h**2/Mpc**3.
    hubble_h : float
        Reduced Hubble parameter used to convert physical halo masses.
    normalizing_mass : float
        Physical mass in Msun enclosed by the sigma8 top-hat sphere.
    sigma8 : float
        Target top-hat rms contrast at that radius, at redshift zero.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """

    def __init__(self, k, power, alpha, density, hubble_h, normalizing_mass, sigma8):
        r"""Tabulate raw sharp-k cells and set the independent top-hat amplitude.

        Parameters
        ----------
        k, power : array_like
            Wavenumbers in h/Mpc and CDM power in (Mpc/h)**3.
        alpha : float
            WDM suppression length in Mpc/h.
        density : float
            Mean density in numerical Msun*h**2/Mpc**3 units.
        hubble_h : float
            Dimensionless reduced Hubble parameter.
        normalizing_mass : float
            Physical top-hat mass in Msun, corresponding to radius 8 Mpc/h
            for the supplied cosmology. It does not set a sharp-k cutoff.
        sigma8 : float
            Positive target top-hat rms density contrast at redshift zero.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        Forward and reverse cell sums preserve small positive EPS variance
        gaps; only the amplitude divisor and mass assignment change here.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        self.k = np.asarray(k, dtype=float)
        self.power = np.asarray(power, dtype=float)
        self.logk = np.log(self.k)
        self.alpha, self.density, self.h = alpha, density, hubble_h
        self.nodes, self.weights = leggauss(16)
        cells = self._partial(self.logk[:-1], self.logk[1:])
        self.cumulative = np.r_[0., np.cumsum(cells)]
        self.tail = np.r_[np.cumsum(cells[::-1])[::-1], 0.]
        radius8 = (3 * normalizing_mass * hubble_h / (4 * np.pi * density)) ** (1 / 3)
        self.normalization2 = sigma8_normalization(
            self.k, self.power, alpha, sigma8, radius=radius8
        ) ** 2

    def _integrand(self, logk):
        r"""Evaluate the sharp-k variance integrand per logarithmic wavenumber.

        Parameters
        ----------
        logk : array_like
            Natural logarithm of k in h/Mpc.

        Returns
        -------
        integrand : numpy.ndarray
            Dimensionless k**3*P_CDM*T**2/(2*pi**2), before dlnk integration.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The enclosing table supplies CDM power in (Mpc/h)**3; the Viel transfer is squared before integration.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        k = np.exp(logk)
        suppression = (1+(self.alpha*k)**(2*1.12))**(-10/1.12)
        return k**3*np.interp(k, self.k, self.power)*suppression/(2*np.pi**2)

    def _partial(self, left, right):
        left, right = np.broadcast_arrays(left, right)
        half = (right-left)/2
        nodes = ((left+right)/2)[...,None]+half[...,None]*self.nodes
        return half*np.sum(self._integrand(nodes)*self.weights, axis=-1)

    def _cutoff(self, mass):
        r"""Map physical mass to the Schneider sharp-k upper wavenumber.

        Parameters
        ----------
        mass : array_like
            Finite positive physical halo mass in Msun.

        Returns
        -------
        numpy.ndarray
            Cutoff 2.5/R_th in h/Mpc with the input shape, where
            R_th**3=3*(mass*h)/(4*pi*density).

        Raises
        ------
        ValueError
            If mass is nonfinite or nonpositive.

        Notes
        -----
        The sharp-k mass assignment M=(4*pi/3)*rho*(c*R)**3 with c=2.5 follows Schneider. This
        calibrated mass mapping is distinct from the real-space top-hat radius used to define
        sigma8.

        M=(4*pi/3)*rho*(c*R)**3 with W=Theta(1-k*R) gives this
        cutoff for c=2.5. No additional (9*pi/2)**(1/3) is applied.

        References
        ----------

        .. [cite-sharp] Schneider (2015), Structure formation with suppressed small-scale perturbations, Eqs. (10)-(12).
            https://doi.org/10.1093/mnras/stv1169
        """
        mass = np.asarray(mass, dtype=float)
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("Mass must be finite and positive.")
        radius = (3*(mass*self.h)/(4*np.pi*self.density))**(1/3)
        return SHARP_K_MASS_ASSIGNMENT/radius

    def _raw(self, mass):
        cutoff = self._cutoff(mass)
        upper = np.log(np.clip(cutoff, self.k[0], self.k[-1]))
        cell = np.clip(np.searchsorted(self.logk,upper,side='right')-1,0,len(self.k)-2)
        return self.cumulative[cell]+self._partial(self.logk[cell],upper)

    def variance(self, mass):
        return self._raw(mass)/self.normalization2

    def derivative(self, mass):
        mass = np.asarray(mass, dtype=float)
        cutoff = self._cutoff(mass)
        active = (cutoff>self.k[0]) & (cutoff<self.k[-1])
        result = np.zeros_like(mass)
        result[active] = -self._integrand(np.log(cutoff[active]))/(3*mass[active]*self.normalization2)
        return result

    def gap(self, smaller_mass, larger_mass):
        """S(smaller)-S(larger), without subtracting saturated variances."""
        small, large = np.broadcast_arrays(smaller_mass,larger_mass)
        if np.any(small>large):
            raise ValueError("Variance gap requires ordered masses.")
        left = np.log(np.clip(self._cutoff(large),self.k[0],self.k[-1]))
        right = np.log(np.clip(self._cutoff(small),self.k[0],self.k[-1]))
        i = np.clip(np.searchsorted(self.logk,left,side='right')-1,0,len(self.k)-2)
        j = np.clip(np.searchsorted(self.logk,right,side='right')-1,0,len(self.k)-2)
        out = np.empty(small.shape)
        same = i==j
        out[same] = self._partial(left[same],right[same])
        split = ~same
        a,b=i[split],j[split]
        # Use forward sums in the rising spectrum and reverse sums in the
        # suppressed tail, preserving positive integrals beyond saturation.
        middle = np.where(self.cumulative[b]<self.tail[a+1],
                          self.cumulative[b]-self.cumulative[a+1],
                          self.tail[a+1]-self.tail[b])
        out[split] = (self._partial(left[split],self.logk[a+1])+middle
                      +self._partial(self.logk[b],right[split]))
        return out/self.normalization2


def _real_numeric(value, name):
    """Reject non-real coordinates before any float cast can discard data."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    return np.asarray(array, dtype=float)


def nfw_mass_function(x):
    r"""Evaluate the dimensionless enclosed-mass factor of an NFW halo.

    Parameters
    ----------
    x : array_like
        Nonnegative dimensionless radius r/rs.

    Returns
    -------
    enclosed : numpy.ndarray
        ln(1+x)-x/(1+x), using a stable small-radius evaluation.

    Notes
    -----
    The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
    inversion tolerances concern this profile identity, not agreement with simulated halos.

    Physical enclosed mass is 4*pi*rhos*rs**3 times this factor. Small-radius series coefficients are algebraic, not empirical fit parameters.

    References
    ----------

    .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
        https://arxiv.org/abs/astro-ph/9611107
    """
    x = _real_numeric(x, "NFW radius ratio")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("NFW radius ratio must be finite and nonnegative.")
    result = np.empty_like(x)
    small = x < 0.01
    # f(x) = sum_{n=2}^infinity (-1)^n (n-1)/n x^n.
    coefficient = [(-1.0) ** n * (n - 1.0) / n for n in range(2, 11)]
    result[small] = x[small] ** 2 * np.polynomial.polynomial.polyval(x[small], coefficient)
    result[~small] = np.log1p(x[~small]) - x[~small] / (1.0 + x[~small])
    return result


def invert_nfw_mass_function(y):
    r"""Recover a nonnegative NFW truncation radius from dimensionless enclosed mass.

    Parameters
    ----------
    y : array_like
        Nonnegative m/(4*pi*rhos*rs**3).

    Returns
    -------
    radius : numpy.ndarray
        Dimensionless radius r/rs with the input shape.

    Notes
    -----
    The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
    inversion tolerances concern this profile identity, not agreement with simulated halos.

    The scalar root solve in logarithmic radius preserves the NFW mass equation. Root tolerances describe numerical inversion, not physical profile accuracy.

    References
    ----------

    .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
        https://arxiv.org/abs/astro-ph/9611107
    """
    y = _real_numeric(y, "Enclosed-mass function")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Enclosed-mass function values must be finite and nonnegative.")
    max_radius = np.finfo(float).max
    max_log_radius = np.log(max_radius)
    max_enclosed = float(nfw_mass_function(max_radius))
    if np.any(y > max_enclosed):
        raise ValueError(
            "The inverse NFW radius is outside the representable floating-point range."
        )

    def one(value: float) -> float:
        if value == 0.0:
            return 0.0
        if value == max_enclosed:
            return max_radius
        # f(x) <= x^2/2 and f(exp(y+1)) >= y bound the positive root.
        lower = 0.5 * (np.log(2.0) + np.log(value)) - np.log(2.0)
        upper = min(value + 1.0, max_log_radius)

        def residual(log_radius):
            radius = max_radius if log_radius == max_log_radius else np.exp(log_radius)
            enclosed = float(nfw_mass_function(radius))
            # The upper bracket can be far above a subnormal target. Preserve
            # its sign without overflowing a ratio that is used only to bracket.
            if value < 1.0 and enclosed > max_radius * value:
                return max_radius
            return enclosed / value - 1.0

        root = brentq(residual, lower, upper, xtol=5.0e-14, rtol=4.0 * np.finfo(float).eps)
        return float(np.exp(root))

    # A safeguarded vector Newton solve avoids a Python/Brent call for every
    # catalog node. The scalar log-radius solver remains the fallback for
    # extreme floating-point values or any unconverged element.
    out = np.zeros_like(y)
    active = (y >= 1e-200) & (y <= 600.)
    values = y[active]
    if values.size:
        lower = .5*(np.log(2.)+np.log(values))-np.log(2.)
        upper = values+1.
        radius_guess = np.sqrt(2*values)+4*values/3
        log_radius = np.where(values<.5,np.log(radius_guess),upper)
        converged = np.zeros(values.shape,dtype=bool)
        for _ in range(30):
            radius = np.exp(log_radius)
            residual = nfw_mass_function(radius)-values
            derivative = (radius/(1+radius))**2
            correction = residual/derivative
            converged |= np.abs(correction) <= 2e-13
            if np.all(converged):
                break
            lower = np.where(residual<0,log_radius,lower)
            upper = np.where(residual>0,log_radius,upper)
            proposed = log_radius-correction
            inside = (proposed>lower) & (proposed<upper)
            proposed = np.where(inside,proposed,.5*(lower+upper))
            log_radius = np.where(converged,log_radius,proposed)
        solved = np.exp(log_radius)
        if np.any(~converged):
            solved[~converged] = [one(float(value)) for value in values[~converged]]
        out[active] = solved
    extreme = (y>0) & ~active
    if np.any(extreme):
        out[extreme] = [one(float(value)) for value in y[extreme]]
    return float(out) if out.ndim == 0 else out
