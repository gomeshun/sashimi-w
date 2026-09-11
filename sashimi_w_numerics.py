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


class SharpKVariance:
    """Finite sharp-k integral on every original, linearly interpolated P(k) cell.

    Sixteen-point Gauss-Legendre integration in log(k) resolves each input
    spectrum knot. Partial cells use the same rule; no mass interpolation or
    projection is applied. Physical masses are in Msun, k in h/Mpc.
    """

    def __init__(self, k, power, alpha, density, hubble_h, normalizing_mass, sigma8):
        self.k = np.asarray(k, dtype=float)
        self.power = np.asarray(power, dtype=float)
        self.logk = np.log(self.k)
        self.alpha, self.density, self.h = alpha, density, hubble_h
        self.nodes, self.weights = leggauss(16)
        cells = self._partial(self.logk[:-1], self.logk[1:])
        self.cumulative = np.r_[0., np.cumsum(cells)]
        self.tail = np.r_[np.cumsum(cells[::-1])[::-1], 0.]
        self.normalization2 = self._raw(normalizing_mass)/sigma8**2

    def _integrand(self, logk):
        k = np.exp(logk)
        suppression = (1+(self.alpha*k)**(2*1.12))**(-10/1.12)
        return k**3*np.interp(k, self.k, self.power)*suppression/(2*np.pi**2)

    def _partial(self, left, right):
        left, right = np.broadcast_arrays(left, right)
        half = (right-left)/2
        nodes = ((left+right)/2)[...,None]+half[...,None]*self.nodes
        return half*np.sum(self._integrand(nodes)*self.weights, axis=-1)

    def _cutoff(self, mass):
        mass = np.asarray(mass, dtype=float)
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("Mass must be finite and positive.")
        radius = (3*(mass*self.h)/(4*np.pi*self.density))**(1/3)/2.5
        return (9*np.pi/2)**(1/3)/radius

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
    """Return ``ln(1+x)-x/(1+x)`` for finite nonnegative ``x``.

    A small-radius series avoids subtracting two nearly equal terms. Values
    smaller than the floating-point subnormal range can still underflow.
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
    """Invert the NFW enclosed-mass function over representable positive radii.

    Solve in log-radius so an absolute tolerance in radius cannot erase a
    small, positive solution. The output retains the input shape.
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

    out = np.vectorize(one, otypes=[float])(y)
    return float(out) if out.ndim == 0 else out
