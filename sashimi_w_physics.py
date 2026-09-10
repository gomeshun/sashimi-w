"""WDM physical kernels and existing observable definitions.

The shared population controller lives in the standard Subhalos API.
"""

import numpy as np
from pathlib import Path
from scipy import optimize
from scipy import special
from scipy.interpolate import interp1d
from scipy.special import cbrt, erf
from scipy.interpolate import splrep, splev
from numpy.polynomial.hermite import hermgauss

cm = 1.0
km = 100000.0 * cm
s = 1.0
gram = 1.0
c = 29979200000.0 * cm / s
G = 6.6742e-08 * cm**3 / gram / s**2
Mpc = 3.086e24 * cm
kpc = Mpc / 1000.0
pc = kpc / 1000.0
Msolar = 1.988435e33 * gram
GeV = 1.7827e-24 * gram
keV = 1e-06 * GeV
" WMAP7 "
filename_PS = Path(__file__).resolve().with_name("WMAP7_camb_matterpower_z0_extrapolated.dat")
PowerSpectrum = np.genfromtxt(filename_PS, skip_header=5)
Pk_file, k_file = (PowerSpectrum[:, 0], PowerSpectrum[:, 1])
k_min = k_file.min() * 1.15
k_max = k_file.max() * 0.86
Pk_interp = interp1d(k_file, Pk_file)
" Cosmology from WMAP7 "
PS_cosmology = np.genfromtxt(filename_PS, max_rows=5)
Omegar = 0.0
Omega0 = 1.0
OmegaB = PS_cosmology[0]
OmegaM = PS_cosmology[1]
OmegaC = OmegaM - OmegaB
OmegaL = 1.0 - OmegaM
pOmega = [OmegaC + OmegaB, Omegar, OmegaL]
h = PS_cosmology[2]
H0 = h * 100 * km / s / Mpc
rhocrit0 = 3 * pow(H0, 2) * pow(8.0 * np.pi * G, -1)
sigma_8 = PS_cosmology[4]
WDM_TRANSFER_NU = 1.12
PUBLISHED_WDM_POWER_Q = 5.0


class WDMPhysics:
    def __init__(self, mass_wdm=1.5):
        self.mass_wdm = mass_wdm
        self.G_units = G * (Msolar / Mpc) * (s**2 / km**2)
        self.Rhocrit_z = 3.0 / (8.0 * np.pi * self.G_units) * 10000.0
        self.Omz = OmegaM
        self.Rhomean_z = self.Rhocrit_z * self.Omz
        self.MassMin = 1e-12
        self.MassMax = 1e18
        self.dlogm = (np.log10(self.MassMax) - np.log10(self.MassMin)) / (100 - 1)
        self.logM0 = np.log10(self.MassMin) + np.arange(100) * self.dlogm + 0.5 * self.dlogm
        self.filter_Mass = 10**self.logM0
        self.R = cbrt(self.filter_Mass / (4 / 3 * np.pi * self.Rhomean_z)) / 2.5
        self.MassIn8Mpc = 4 / 3 * np.pi * 8**3 * self.Rhomean_z
        self.a = 0.049 * (1 / self.mass_wdm) ** 1.11 * (OmegaC / 0.25) ** 0.11 * (h / 0.7) ** 1.22

    def TopHat(self, k, r):
        return 3.0 / (k * r) ** 2 * (np.sin(k * r) / (k * r) - np.cos(k * r))

    def SigmaIntegrand(self, k, r):
        om_wdm = OmegaM - OmegaB
        a = 0.049 * (1 / self.mass_wdm) ** 1.11 * (om_wdm / 0.25) ** 0.11 * (h / 0.7) ** 1.22
        return k**2 * Pk_interp(k) * self.TopHat(k, r) ** 2 * self._wdm_power_ratio(k, alpha=a)

    " Integration function "

    def integratePk_th(self, kmin, kmax, r):
        log_k_min_th = np.log10(kmin)
        log_k_max_th = np.log10(kmax)
        dlogk_th = (log_k_max_th - log_k_min_th) / (500 - 1)
        tot_sum_th = 0.0
        logk_th = log_k_min_th + np.arange(500) * dlogk_th
        logk_th = np.reshape(logk_th, (500, 1))
        sum_rect_th = self.SigmaIntegrand(10**logk_th, r) * 10**logk_th * np.log(10)
        tot_sum_th = np.sum(sum_rect_th, axis=0)
        log_k_min_th = log_k_min_th - dlogk_th
        sum_rect_min_th = self.SigmaIntegrand(10**log_k_min_th, r) * 10**log_k_min_th * np.log(10)
        log_k_max_th = log_k_max_th + dlogk_th
        sum_rect_max_th = self.SigmaIntegrand(10**log_k_max_th, r) * 10**log_k_max_th * np.log(10)
        sigma_sq_th = (tot_sum_th + 0.5 * sum_rect_min_th + 0.5 * sum_rect_max_th) * dlogk_th
        sigma_sq_th /= 2 * np.pi**2
        return sigma_sq_th

    def conc200(self, M, z):
        M = M / Msolar / h
        redshiftvect = np.linspace(0, 7, 8)
        R_th = cbrt(self.filter_Mass / (4 / 3 * np.pi * self.Rhomean_z))
        Sigma_Sq_th = np.zeros(len(self.filter_Mass))
        Sigma_Sq_th = self.integratePk_th(k_min, k_max, R_th)
        Sigma_th = np.sqrt(Sigma_Sq_th)
        sig_interp_th = interp1d(self.filter_Mass, Sigma_th)
        MassIn8Mpc = 4 / 3 * np.pi * 8**3 * self.Rhomean_z
        sig_8_th = sig_interp_th(MassIn8Mpc)
        normalise_th = sig_8_th / sigma_8
        Sigma_th /= normalise_th
        Sigma_Sq_th = Sigma_th**2
        " free model parameters, Ludlow et al. (2016) "
        A = 650.0 / 200
        f = 0.02
        delta_sc = 1.686
        delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.0 - OmegaM, redshiftvect)
        OmegaL = 1.0 - OmegaM
        sig2_interp_th = splrep(self.logM0 - 10.0, Sigma_Sq_th, k=1)
        if np.shape(M) == ():
            index = np.abs(10**self.logM0 - M).argmin()
            c_array = 10 ** (np.arange(100) * 4.0 / 99.0)
            M2 = (np.log(2.0) - 0.5) / (np.log(1.0 + c_array) - c_array / (1.0 + c_array))
            rho_2 = 200.0 * c_array**3 * M2
            rhoc = rho_2 / (200.0 * A)
            with np.errstate(invalid="ignore"):
                z2 = (
                    1.0 / OmegaM * (rhoc * (OmegaM * (1 + z) ** 3 + OmegaL) - OmegaL)
                ) ** 0.3333 - 1.0
            delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
            delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.0 - OmegaM, z)
            sig2fM_th = splev(self.logM0[index] - 10.0 + np.log10(f), sig2_interp_th)
            sig2M_th = Sigma_Sq_th[index]
            arg = A * rhoc / c_array**3 - (
                1.0 - erf((delta_sc_z2 - delta_sc_0_vect) / np.sqrt(2.0 * (sig2fM_th - sig2M_th)))
            )
            mask = np.isinf(arg) | np.isnan(arg)
            arg = arg[~mask]
            c_array = c_array[~mask]
            c_nfw = np.interp(0, arg, c_array)
        elif M.ndim == 1:
            M_reshaped = M.flatten()
            c_nfw = np.zeros(len(M_reshaped))
            for i in range(len(M_reshaped)):
                index = np.abs(10**self.logM0 - M_reshaped[i]).argmin()
                c_array = 10 ** (np.arange(100) * 4.0 / 99.0)
                M2 = (np.log(2.0) - 0.5) / (np.log(1.0 + c_array) - c_array / (1.0 + c_array))
                rho_2 = 200.0 * c_array**3 * M2
                rhoc = rho_2 / (200.0 * A)
                with np.errstate(invalid="ignore"):
                    z2 = (
                        1.0 / OmegaM * (rhoc * (OmegaM * (1 + z) ** 3 + OmegaL) - OmegaL)
                    ) ** 0.3333 - 1.0
                delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
                delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.0 - OmegaM, z)
                sig2fM_th = splev(self.logM0[index] - 10.0 + np.log10(f), sig2_interp_th)
                sig2M_th = Sigma_Sq_th[index]
                arg = A * rhoc / c_array**3 - (
                    1.0
                    - erf((delta_sc_z2 - delta_sc_0_vect) / np.sqrt(2.0 * (sig2fM_th - sig2M_th)))
                )
                mask = np.isinf(arg) | np.isnan(arg)
                arg = arg[~mask]
                c_array = c_array[~mask]
                c_nfw[i] = np.interp(0, arg, c_array)
            c_nfw = np.reshape(c_nfw, np.shape(M))
        elif M.ndim == 2:
            M_reshaped = M.flatten()
            z_reshaped = z.flatten()
            c_nfw = np.zeros(len(M_reshaped))
            for i in range(len(M_reshaped)):
                index = np.abs(10**self.logM0 - M_reshaped[i]).argmin()
                c_array = 10 ** (np.arange(100) * 4.0 / 99.0)
                M2 = (np.log(2.0) - 0.5) / (np.log(1.0 + c_array) - c_array / (1.0 + c_array))
                rho_2 = 200.0 * c_array**3 * M2
                rhoc = rho_2 / (200.0 * A)
                with np.errstate(invalid="ignore"):
                    z2 = (
                        1.0
                        / OmegaM
                        * (rhoc * (OmegaM * (1 + z_reshaped[i]) ** 3 + OmegaL) - OmegaL)
                    ) ** 0.3333 - 1.0
                delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
                delta_sc_0_vect = delta_sc / self.linear_growth_factor(
                    OmegaM, 1.0 - OmegaM, z_reshaped[i]
                )
                sig2fM_th = splev(self.logM0[index] - 10.0 + np.log10(f), sig2_interp_th)
                sig2M_th = Sigma_Sq_th[index]
                arg = A * rhoc / c_array**3 - (
                    1.0
                    - erf((delta_sc_z2 - delta_sc_0_vect) / np.sqrt(2.0 * (sig2fM_th - sig2M_th)))
                )
                mask = np.isinf(arg) | np.isnan(arg)
                arg = arg[~mask]
                c_array = c_array[~mask]
                c_nfw[i] = np.interp(0, arg, c_array)
            c_nfw = np.reshape(c_nfw, np.shape(M))
        return c_nfw

    def fc(self, x):
        return np.log(1 + x) - x * pow(1 + x, -1)

    def rhocrit(self, z):
        return 3.0 * pow(self.Hz(z), 2) * pow(np.pi * 8.0 * G, -1)

    def Mvir_from_M200(self, M200, z):
        gz = self.g(z)
        c200 = self.conc200(M200, z)
        r200 = (3.0 * M200 / (4 * np.pi * 200 * rhocrit0 * gz)) ** (1.0 / 3.0)
        rs = r200 / c200
        fc200 = self.fc(c200)
        rhos = M200 / (4 * np.pi * rs**3 * fc200)
        Dc = self.Delc(self.Omegaz(pOmega, z) - 1.0)
        rvir = optimize.fsolve(
            lambda r: 3.0 * (rs / r) ** 3 * self.fc(r / rs) * rhos - Dc * rhocrit0 * gz, r200
        )
        Mvir = 4 * np.pi * rs**3 * rhos * self.fc(rvir / rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -0.00313
        a4 = -3.52e-05
        Oz = self.Omegaz(pOmega, z)

        def ffunc(x):
            return np.power(x, 3.0) * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))

        def xfunc(f):
            p = a2 + a3 * np.log(f) + a4 * np.power(np.log(f), 2.0)
            return np.power(a1 * np.power(f, 2.0 * p) + (3.0 / 4.0) ** 2, -0.5) + 2.0 * f

        return (
            self.Delc(Oz - 1)
            / 200.0
            * M200
            * np.power(
                self.conc200(M200, z)
                * xfunc(self.Delc(Oz - 1) / 200.0 * ffunc(1.0 / self.conc200(M200, z))),
                -3.0,
            )
        )

    def xi(self, M):
        return pow(M * pow(10000000000.0 * pow(h, -1), -1), -1)

    def Mzi(self, M0, z):
        a = 1.686 * np.sqrt(2.0 / np.pi) * self.dDdz(0) + 1.0
        zf = -0.0064 * pow(np.log10(M0), 2) + 0.0237 * np.log10(M0) + 1.8837
        q = 4.137 * pow(zf, -0.9476)
        fM0 = pow(pow(self.sigmaMz(M0 / q, 0), 2) - pow(self.sigmaMz(M0, 0), 2), -0.5)
        return M0 * pow(1 + z, a * fM0) * np.exp(-fM0 * z)

    def Mzzi(self, M0, z, zi):
        Mzi0 = self.Mzi(M0, zi)
        zf = -0.0064 * pow(np.log10(M0), 2) + 0.0237 * np.log10(M0) + 1.8837
        q = 4.137 * pow(zf, -0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0 / q, zi), 2) - pow(self.sigmaMz(Mzi0, zi), 2), -0.5)
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) * pow(self.growthD(zi), -2) * self.dDdz(zi) + 1
        )
        beta = -fMzi
        return Mzi0 * pow(1 + z - zi, alpha) * np.exp(beta * (z - zi))

    def Delc(self, x):
        return 18 * pow(np.pi, 2) + 82.0 * x - 39 * pow(x, 2)

    def dMdz(self, M0, z, zi, sigmafac=0):
        Mzi0 = self.Mzi(M0, zi)
        zf = -0.0064 * pow(np.log10(M0), 2) + 0.0237 * np.log10(M0) + 1.8837
        q = 4.137 * pow(zf, -0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0 / q, zi), 2) - pow(self.sigmaMz(Mzi0, zi), 2), -0.5)
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) * pow(self.growthD(zi), -2) * self.dDdz(zi) + 1
        )
        beta = -fMzi
        Mzzidef = Mzi0 * pow(1 + z - zi, alpha) * np.exp(beta * (z - zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef * Msolar, z)
        return (beta + alpha * pow(1 + z - zi, -1)) * Mzzivir / Msolar

    def delc_Y11(self, M, z):
        """Critical overdensity for collapse for WDM, Benson et al. (2012): Eq.7)"""
        gx = 1.5
        z_eq = 3600 * (OmegaM * pow(h, 2) / 0.15) - 1
        Mj = (
            3.06
            * 10**8
            * pow((1 + z_eq) / 3000, 1.5)
            * pow(OmegaM * pow(h, 2) / 0.15, 0.5)
            * pow(gx / 1.5, -1)
            * pow(self.mass_wdm, -4)
        )
        x = np.log(M / Mj)
        hh = 1.0 / (1 + np.exp((x + 2.4) / 0.1))
        return (
            1.686
            * pow(self.growthD(z), -1)
            * (hh * (0.04 / np.exp(2.3 * x)) + (1 - hh) * np.exp(0.31687 / np.exp(0.809 * x)))
        )

    def s_Y11(self, M):
        return pow(self.sigmaMz(M, 0), 2)

    def Ffunc_Yang(self, delc1, delc2, sig1, sig2):
        """Returns Eq. (14) of Yang et al. (2011)"""
        return (
            pow(2 * np.pi, -0.5)
            * (delc2 - delc1)
            * pow(sig2 - sig1, -1.5)
            * np.exp(-pow(delc2 - delc1, 2) * pow(2 * (sig2 - sig1), -1))
        )

    def Na_calc(self, ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000, sigmafac=0):
        """Returns Na, Eq. (3) of Yang et al. (2011)"""
        zacc_2d = zacc.reshape(len(zacc), 1)
        M200_0 = self.Mzzi(Mhost, zacc_2d, z0)
        logM200_0 = np.log10(M200_0)
        if N_herm == 1:
            sigmalogM200_0 = 0.12 + 0.15 * np.log10(Mhost / M200_0)
            sigmalogM200_1 = (
                sigmalogM200_0[zacc_2d > 1.0][0]
                / np.log10(M200_0[zacc_2d > 1.0][0] / Mhost)
                * np.log10(M200_0 / Mhost)
            )
            sigmalogM200 = np.where(zacc_2d > 1.0, sigmalogM200_0, sigmalogM200_1)
            logM200 = logM200_0 + sigmafac * sigmalogM200
            M200 = 10**logM200
            if sigmafac > 0.0:
                M200 = np.where(M200 < Mhost, M200, Mhost)
        else:
            xxi, wwi = hermgauss(N_herm)
            xxi = xxi.reshape(len(xxi), 1, 1)
            wwi = wwi.reshape(len(wwi), 1, 1)
            " eq. (21) in Yang et al. (2011) "
            sigmalogM200 = 0.12 - 0.15 * np.log10(M200_0 / Mhost)
            logM200 = np.sqrt(2) * sigmalogM200 * xxi + logM200_0
            M200 = 10**logM200
        mmax = np.minimum(M200, Mhost / 2.0)
        Mmax = np.minimum(M200_0 + mmax, Mhost)
        zlist = zacc_2d * np.linspace(1, 0, Nrand)
        iMmax = np.argmin(np.abs(self.Mzzi(Mhost, zlist, z0) - Mmax), axis=-1)
        z_Max = zlist[np.arange(len(zlist)), iMmax]
        z_Max_3d = z_Max.reshape(N_herm, len(zlist), 1)
        delcM = self.delc_Y11(Mmax, z_Max_3d)
        delca = self.delc_Y11(ma, zacc_2d)
        sM = self.s_Y11(Mmax)
        sa = self.s_Y11(ma)
        xmax = pow(delca - delcM, 2) * pow(2 * (self.s_Y11(mmax) - sM), -1)
        normB = special.gamma(0.5) * special.gammainc(0.5, xmax) / np.sqrt(np.pi)
        " those reside in the exponential part of eq.14 "
        Phi = self.Ffunc_Yang(delcM, delca, sM, sa) / normB * np.heaviside(mmax - ma, 0)
        if N_herm == 1:
            F2t = np.nan_to_num(Phi)
            F2 = F2t.reshape((len(zacc_2d), len(ma)))
        else:
            F2 = np.sum(np.nan_to_num(Phi) * wwi / np.sqrt(np.pi), axis=0)
        Na = F2 * self.dsdm(ma, 0) * self.dMdz(Mhost, zacc_2d, z0, sigmafac) * (1 + zacc_2d)
        return Na

    def subhalo_distr(
        self,
        M0,
        accretion=False,
        redshift=0.0,
        dz=0.1,
        zmax=7.0,
        N_ma=500,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=1,
        logmamax=None,
        sigmafac=0,
        N_hermNa=200,
        profile_change=True,
    ):
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(
            M0,
            redshift,
            dz,
            zmax,
            N_ma,
            sigmalogc,
            N_herm,
            logmamin,
            logmamax,
            sigmafac,
            N_hermNa,
            profile_change=True,
        )
        if not accretion:
            N, lnm_edges = np.histogram(np.log(m0), weights=weight, bins=100)
        else:
            N, lnm_edges = np.histogram(np.log(ma200), weights=weight, bins=100)
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        dlnm = lnm_edges[1:] - lnm_edges[:-1]
        m = np.exp(lnm)
        dNdlnm = N / dlnm
        dNdm = dNdlnm / m
        return (m, dNdm)

    def N_sat(
        self,
        M0,
        Mpeak=None,
        Mpeak_thres=False,
        redshift=0.0,
        dz=0.1,
        zmax=7.0,
        N_ma=500,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=1,
        logmamax=None,
        sigmafac=0,
        N_hermNa=200,
        profile_change=True,
    ):
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(
            M0,
            redshift,
            dz,
            zmax,
            N_ma,
            sigmalogc,
            N_herm,
            logmamin,
            logmamax,
            sigmafac,
            N_hermNa,
            profile_change=True,
        )
        if Mpeak_thres:
            N, x_edges = np.histogram(
                ma200[ma200 > Mpeak], weights=weight[ma200 > Mpeak], bins=10000
            )
        else:
            N, x_edges = np.histogram(ma200, weights=weight, bins=10000)
        x = (x_edges[1:] + x_edges[:-1]) / 2.0
        Ncum = np.cumsum(N)
        Ncum = Ncum[-1] - Ncum
        return (Ncum[0], x, Ncum)

    def N_sat_Vthres(
        self,
        M0,
        Vpeak_max,
        Vpeak_thres=True,
        redshift=0.0,
        dz=0.1,
        zmax=7.0,
        N_ma=500,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=1,
        logmamax=None,
        sigmafac=0,
        N_hermNa=200,
        profile_change=True,
    ):
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(
            M0,
            redshift,
            dz,
            zmax,
            N_ma,
            sigmalogc,
            N_herm,
            logmamin,
            logmamax,
            sigmafac,
            N_hermNa,
            profile_change=True,
        )
        ma200 *= Msolar
        m0 *= Msolar
        rs_a *= kpc
        rs0 *= kpc
        rhos_a *= Msolar / pc**3
        rhos0 *= Msolar / pc**3
        Vpeak = np.sqrt(4.0 * np.pi * G * rhos_a / 4.625) * rs_a
        Vmax = np.sqrt(4.0 * np.pi * G * rhos0 / 4.625) * rs0
        if Vpeak_thres:
            N, x_edges = np.histogram(
                Vmax[Vpeak > Vpeak_max * km / s] / (km / s),
                weights=weight[Vpeak > Vpeak_max * km / s],
                bins=10000,
            )
        else:
            N, x_edges = np.histogram(
                Vmax[Vmax > Vpeak_max * km / s] / (km / s),
                weights=weight[Vmax > Vpeak_max * km / s],
                bins=10000,
            )
        x = (x_edges[1:] + x_edges[:-1]) / 2.0
        Ncum = np.cumsum(N)
        Ncum = Ncum[-1] - Ncum
        return (Ncum[0], x, Ncum)
