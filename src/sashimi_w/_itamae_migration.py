"""Standard ITAMAE-backed WDM calculation and named catalog API."""

from __future__ import annotations
from typing import Any
import numpy as np
from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import build_calculation_metadata
from itamae.protocols import CosmologyBackend
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
from . import _physics as _physics
from ._physics import OmegaM, WDM_TRANSFER_NU, h, WDMPhysics
from ._power import POWER_NORMALIZATION, SHARP_K_MASS_ASSIGNMENT

_CANONICAL_SCALE = {"Msun": 1.0, "kpc": 0.001, "Msun/pc3": 1e18, "dimensionless": 1.0}
_CANONICAL_ASTROPY_UNIT = {"mass": "Msun", "length": "Mpc", "density": "Msun / Mpc3"}
STANDARD_T2_Q10 = "standard-t2-q10"
CALCULATION_SPECIFICATION = "sashimi-w:thermal-wdm-q10:2026-10-14:v3"
_CONSISTENT_OMEGA_L = 1.0 - float(OmegaM)


class Subhalos(WDMPhysics):
    r"""Thermal WDM with the Viel transfer amplitude squared and a fixed specification.

    Canonical catalogs use Msun, Mpc and Msun/Mpc^3. Tuple conversion uses the
    historical Msun, kpc and Msun/pc^3 format. No legacy calculation is provided.

    Parameters
    ----------
    mass_wdm : float
        Positive thermal-relic WDM particle mass, numerical value in keV.
    wdm_power_convention : str, optional
        Supported standard-t2-q10 convention: the Viel transfer amplitude is squared in
        power. Historical published-q5 selectors are rejected.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.
    unit_backend : UnitBackend, optional
        Explicit converter to canonical Msun, Mpc and Msun/Mpc^3 catalogue arrays;
        NativeUnits is the default.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
    Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
    kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
    specification.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """

    def __init__(
        self,
        mass_wdm: float = 1.5,
        *,
        wdm_power_convention: str = STANDARD_T2_Q10,
        cosmology_backend: Any | None = None,
        unit_backend: Any | None = None,
    ) -> None:
        r"""Select the q10 thermal WDM model, compatible WMAP7 background and catalogue unit
        boundary.

        Parameters
        ----------
        mass_wdm : float
            Positive thermal-relic WDM particle mass, numerical value in keV.
        wdm_power_convention : str, optional
            Supported standard-t2-q10 convention: the Viel transfer amplitude is squared in
            power. Historical published-q5 selectors are rejected.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.
        unit_backend : UnitBackend, optional
            Explicit converter to canonical Msun, Mpc and Msun/Mpc^3 catalogue arrays;
            NativeUnits is the default.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        try:
            particle_mass = float(mass_wdm)
        except (TypeError, ValueError) as error:
            raise ValueError("mass_wdm must be finite and positive.") from error
        if not np.isfinite(particle_mass) or particle_mass <= 0.0:
            raise ValueError("mass_wdm must be finite and positive.")
        if wdm_power_convention == "published-q5":
            raise ValueError(
                "wdm_power_convention='published-q5' is retired. Reproduce q5 with "
                "the frozen historical reference; it is not converted to q10."
            )
        if wdm_power_convention != STANDARD_T2_Q10:
            raise ValueError("wdm_power_convention must be 'standard-t2-q10'.")
        backend = cosmology_backend or NativeFlatLCDM(omega_m0=float(OmegaM), h=float(h))
        self._validate_cosmology(backend)
        self.itamae_cosmology = backend
        self.itamae_units = unit_backend or NativeUnits()
        super().__init__(mass_wdm=particle_mass)
        self._variance_model = None

    @property
    def wdm_power_convention(self) -> str:
        """Identify the active thermal WDM power-suppression prescription.

        Returns
        -------
        str
            standard-t2-q10.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return STANDARD_T2_Q10

    @property
    def wdm_power_q(self) -> float:
        """Return the exponent numerator applied to the power ratio.

        Returns
        -------
        float
            10.0, corresponding to the square of the q=5 transfer amplitude.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return 10.0

    @property
    def wdm_power_formula_role(self) -> str:
        """Identify that the Viel expression is used as a squared transfer amplitude.

        Returns
        -------
        str
            Formula-role label recorded in scientific provenance.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return "viel-transfer-amplitude-squared"

    @property
    def half_mode_definition(self) -> str:
        """State which suppression observable defines the standard half-mode scale.

        Returns
        -------
        str
            Amplitude half T=0.5, equivalent to power ratio 0.25.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return "T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25"

    @property
    def half_mode_power_ratio(self) -> float:
        """Return the power threshold corresponding to amplitude half.

        Returns
        -------
        float
            0.25, not power half.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return 0.25

    def transfer_amplitude(self, k: Any):
        r"""Viel transfer amplitude T=sqrt(P_WDM/P_CDM), for k in h/Mpc.

        Parameters
        ----------
        k : float or array_like
            Nonnegative wavenumber in h/Mpc; alpha is correspondingly in Mpc/h.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless transfer amplitude T(k) with the wavenumber shape.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        wavenumber = np.asarray(k, dtype=float)
        return (1.0 + (self.a * wavenumber) ** (2.0 * WDM_TRANSFER_NU)) ** (-5.0 / WDM_TRANSFER_NU)

    def power_ratio(self, k: Any):
        r"""Thermal WDM/CDM power ratio T^2, for k in h/Mpc.

        Evaluate exponent -10/nu directly to retain the existing q10 floating
        point operation order. It equals transfer_amplitude(k)**2 to roundoff.

        Parameters
        ----------
        k : float or array_like
            Nonnegative wavenumber in h/Mpc; alpha is correspondingly in Mpc/h.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless power ratio P_WDM/P_CDM=T(k)**2.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        return self._wdm_power_ratio(k)

    def _wdm_power_ratio(self, k: Any, *, alpha: Any | None = None):
        r"""Single q10 power hook for sharp-k variance and top-hat concentration.

        Parameters
        ----------
        k : float or array_like
            Nonnegative wavenumber in h/Mpc; alpha is correspondingly in Mpc/h.
        alpha : float, optional
            Explicit transfer scale in Mpc/h; None selects the configured particle-mass value.

        Returns
        -------
        float or numpy.ndarray
            q10 power suppression using the selected alpha, with query shape.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        selected_alpha = self.a if alpha is None else alpha
        wavenumber = np.asarray(k, dtype=float)
        return (1.0 + (selected_alpha * wavenumber) ** (2.0 * WDM_TRANSFER_NU)) ** (
            -10.0 / WDM_TRANSFER_NU
        )

    def half_mode_wavenumber(self, *, power_ratio: float = 0.25) -> float:
        r"""Return k in h/Mpc at an explicit power-ratio threshold.

        The default is amplitude half (T=1/2, power=1/4), preserving the old
        q10 numerical scale. Use power_ratio=0.5 for power half. This diagnostic
        threshold does not alter the concentration or population prescription.

        Parameters
        ----------
        power_ratio : float
            Threshold strictly between zero and one; 0.25 means amplitude half, while 0.5 means
            power half.

        Returns
        -------
        float
            Positive k in h/Mpc where the specified power ratio is reached.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        value = np.asarray(power_ratio)
        if value.ndim != 0 or value.dtype.kind not in "iuf" or not np.isfinite(value):
            raise ValueError("power_ratio must be a real scalar strictly between 0 and 1.")
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            raise ValueError("power_ratio must be a real scalar strictly between 0 and 1.")
        if threshold == 0.25:
            factor = (2.0 ** (WDM_TRANSFER_NU / 5.0) - 1.0) ** (1.0 / (2.0 * WDM_TRANSFER_NU))
        else:
            factor = (threshold ** (-WDM_TRANSFER_NU / 10.0) - 1.0) ** (
                1.0 / (2.0 * WDM_TRANSFER_NU)
            )
        return float(factor / self.a)

    @staticmethod
    def _validate_cosmology(backend: Any) -> None:
        """Require a complete backend matching the WMAP7 migration model.

        Parameters
        ----------
        backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        if not isinstance(backend, CosmologyBackend):
            raise TypeError("cosmology_backend must implement the ITAMAE cosmology protocol.")
        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        backend_h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, float(OmegaM), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"SASHIMI-W migration requires OmegaM={float(OmegaM)}; received {omega_m0}."
            )
        if not np.isclose(backend_h, float(h), rtol=0.0, atol=1e-12):
            raise ValueError(f"SASHIMI-W migration requires h={float(h)}; received {backend_h}.")

    @property
    def omega_lambda(self) -> float:
        """Flat WMAP7 dark-energy density fraction.

        Returns
        -------
        float
            Flat-background fraction 1-OmegaM for the bundled WMAP7 parameters.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return _CONSISTENT_OMEGA_L

    def g(self, z: Any):
        """Return ``H(z)^2/H0^2`` under the selected physics convention.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless squared expansion H(z)**2/H0**2.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        redshift = np.asarray(z, dtype=float)
        return float(OmegaM) * (1.0 + redshift) ** 3 + self.omega_lambda

    def Hz(self, z: Any):
        """Return the WMAP7 Hubble rate in historical inverse-second units.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Expansion rate in inverse seconds using the retained WDM H0 normalization.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        ratio = np.asarray(self.itamae_cosmology.H(z), dtype=float) / float(
            np.asarray(self.itamae_cosmology.H(0.0))
        )
        return _physics.H0 * ratio

    def Omegaz(self, parameters: Any, redshift: Any):
        """Return matter density while retaining the historical signature.

        Parameters
        ----------
        parameters : object
            Ignored historical density-parameter argument; the selected backend owns the actual
            background.
        redshift : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Dimensionless matter fraction Omega_m(z); parameters is retained only for signature
            compatibility.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return np.asarray(self.itamae_cosmology.omega_m(redshift), dtype=float)

    def growthD(self, z: Any):
        r"""Return the normalized WMAP7 growth factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Linear growth D(z)/D(0), dimensionless.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """
        growth = np.asarray(self.itamae_cosmology.growth_factor(z), dtype=float)
        growth0 = float(np.asarray(self.itamae_cosmology.growth_factor(0.0), dtype=float))
        result = growth / growth0
        return float(result) if result.ndim == 0 else result

    def dOdz(self, z: Any):
        """Return the derivative of the flat WMAP7 dark-energy fraction.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            d Omega_Lambda(z)/dz in the flat WMAP7 background.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        redshift = np.asarray(z, dtype=float)
        denominator = self.omega_lambda + float(OmegaM) * (1.0 + redshift) ** 3
        return -self.omega_lambda * 3.0 * float(OmegaM) * (1.0 + redshift) ** 2 / denominator**2

    def dDdz(self, z: Any):
        r"""Differentiate the selected Carroll-Press-Turner growth factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            dD/dz from the same normalized Carroll-Press-Turner relation.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """
        redshift = np.asarray(z, dtype=float)
        omega_lz = self.omega_lambda / (self.omega_lambda + float(OmegaM) * (1.0 + redshift) ** 3)
        omega_mz = 1.0 - omega_lz
        phi_z = (
            omega_mz ** (4.0 / 7.0) - omega_lz + (1.0 + omega_mz / 2.0) * (1.0 + omega_lz / 70.0)
        )
        phi0 = (
            float(OmegaM) ** (4.0 / 7.0)
            - self.omega_lambda
            + (1.0 + float(OmegaM) / 2.0) * (1.0 + self.omega_lambda / 70.0)
        )
        domega_l_dz = self.dOdz(redshift)
        dphi_dz = domega_l_dz * (
            -4.0 / 7.0 * omega_mz ** (-3.0 / 7.0)
            + (omega_mz - omega_lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        result = (
            phi0
            / float(OmegaM)
            * (
                -domega_l_dz / (phi_z * (1.0 + redshift))
                - omega_mz
                * (dphi_dz * (1.0 + redshift) + phi_z)
                / (phi_z**2 * (1.0 + redshift) ** 2)
            )
        )
        return float(result) if result.ndim == 0 else result

    def variance_model(self):
        """Lazily construct the canonical-unit analytic sharp-k variance.

        Returns
        -------
        IntegratedVarianceModel
            Cached canonical sharp-k variance and its moving-boundary derivative for this WDM
            model.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        if self._variance_model is None:
            from ._itamae_variance import make_integrated_variance_model

            self._variance_model = make_integrated_variance_model(self)
        return self._variance_model

    def _variance_gap(self, smaller_mass, larger_mass):
        """Integrate S(smaller)-S(larger) directly with the configured spectrum.

        Parameters
        ----------
        smaller_mass : float or array_like
            Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
            nonnegative sharp-k variance interval.
        larger_mass : float or array_like
            Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
            nonnegative sharp-k variance interval.

        Returns
        -------
        numpy.ndarray
            S(smaller,0)-S(larger,0), evaluated as a positive spectrum interval.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        from ._itamae_variance import _sharp_k_variance_gap

        return _sharp_k_variance_gap(self.variance_model(), smaller_mass, larger_mass)

    def sigmaMz(self, mass: Any, z: Any):
        """Evaluate the physical sharp-k integral in the model's mass domain.

        Sigma and dS/dM use the same continuous finite-domain integral; no
        coarse mass interpolation or monotonic projection is applied.

        Parameters
        ----------
        mass : float or array_like
            Positive halo mass. Use solar masses with the canonical backend; a supplied variance
            prescription must use the same mass convention and factors of h.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless rms sigma(M,z), restricted to the declared physical mass grid.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        mass_array, redshift = np.broadcast_arrays(
            np.asarray(mass, dtype=float), np.asarray(z, dtype=float)
        )
        if not np.all(np.isfinite(mass_array)) or np.any(mass_array <= 0.0):
            raise ValueError("Masses must be finite and positive.")
        physical_grid = np.asarray(self.filter_Mass, dtype=float) / float(h)
        if np.any(mass_array < physical_grid[0]) or np.any(mass_array > physical_grid[-1]):
            raise ValueError("Mass lies outside the canonical SASHIMI-W variance grid.")
        result = np.asarray(self.variance_model().sigma(mass_array, redshift), dtype=float)
        return float(result) if result.ndim == 0 else result

    def dsdm(self, mass: Any, z: Any):
        """Return ``dS/dM`` under the selected growth convention.

        Parameters
        ----------
        mass : float or array_like
            Positive halo mass. Use solar masses with the canonical backend; a supplied variance
            prescription must use the same mass convention and factors of h.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            dS/dM in inverse physical Msun, using the continuous moving-cutoff formula and
            D(z)**2.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        result = np.asarray(self.variance_model().dvariance_dmass(mass, z), dtype=float)
        return float(result) if result.ndim == 0 else result

    def conc200(self, mass_cgs: Any, z: Any):
        r"""Ludlow concentration for CGS physical mass.

        The kernel uses M/Msolar/h; multiplying by h squared before that
        boundary implements M_filter[Msun/h] = h M_physical[Msun].

        Parameters
        ----------
        mass_cgs : float or array_like
            Physical halo M200 in grams at the retained WDM concentration-kernel boundary.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless c200 from the retained top-hat formation prescription after the
            physical-to-filter mass mapping.

        Notes
        -----
        The formation model adopts C=650, f=0.02 and spherical barrier 1.686 from Ludlow et al.
        The implementation evaluates NFW enclosed masses while the original calibration uses
        Einasto profiles; retaining C is a model adaptation, not a recalibration. Trial counts
        and root-search limits are numerical choices.

        The sharp-k mass assignment M=(4*pi/3)*rho*(c*R)**3 with c=2.5 follows Schneider. This
        calibrated mass mapping is distinct from the real-space top-hat radius used to define
        sigma8.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-formation] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Section 4.3, Eqs. (6)-(7).
            https://arxiv.org/abs/1601.02624

        .. [cite-sharp] Schneider (2015), Structure formation with suppressed small-scale perturbations, Eqs. (10)-(12).
            https://doi.org/10.1093/mnras/stv1169
            Full text: https://arxiv.org/abs/1412.2133
        """
        return super().conc200(np.asarray(mass_cgs, dtype=float) * float(h) ** 2, z)

    def linear_growth_factor(self, Omega_m0: float, Omega_l0: float, z: Any):
        r"""Return the established interval-aware linear growth factor using ITAMAE.

        Parameters
        ----------
        Omega_m0 : float
            Present-day matter density fraction.
        Omega_l0 : float
            Present-day cosmological-constant density fraction.
        z : float or numpy.ndarray
            Redshift. A two-element array is interpreted as an interval.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless growth factor or interval growth ratio.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """
        if not np.isclose(Omega_m0 + Omega_l0, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("The initial ITAMAE WDM adapter supports flat cosmologies only.")
        backend = self.itamae_cosmology
        backend_omega_m = getattr(backend, "omega_m0", Omega_m0)
        if not np.isclose(backend_omega_m, Omega_m0, rtol=0.0, atol=1e-12):
            backend = NativeFlatLCDM(omega_m0=float(Omega_m0), h=float(h))
        redshift = np.asarray(z, dtype=float)
        flattened = np.atleast_1d(redshift)
        if flattened.size == 2:
            z1, z2 = (flattened[0], flattened[1])
            return np.asarray(backend.growth_factor(z2)) / np.asarray(backend.growth_factor(z1))
        result = np.asarray(backend.growth_factor(redshift)) / np.asarray(
            backend.growth_factor(0.0)
        )
        return float(result) if result.ndim == 0 else result

    def _to_canonical(self, value, physical_type: str, tuple_unit: str) -> np.ndarray:
        """Convert a tuple array to the active ITAMAE unit backend.

        Parameters
        ----------
        value
            Legacy scalar or array.
        physical_type : str
            ITAMAE physical-type identifier.
        tuple_unit : {"Msun", "kpc", "Msun/pc3", "dimensionless"}
            Unit attached implicitly by the tuple return contract.

        Returns
        -------
        numpy.ndarray
            Plain floating values in ITAMAE canonical units.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        try:
            scale = _CANONICAL_SCALE[tuple_unit]
        except KeyError as error:
            raise ValueError(f"Unsupported tuple unit: {tuple_unit}") from error
        canonical = np.asarray(value, dtype=float) * scale
        if getattr(self.itamae_units, "identifier", "").startswith("astropy"):
            import astropy.units as u

            if physical_type == "dimensionless":
                quantity = canonical * u.dimensionless_unscaled
            else:
                quantity = canonical * u.Unit(_CANONICAL_ASTROPY_UNIT[physical_type])
            return self.itamae_units.to_internal(quantity, physical_type)
        return self.itamae_units.to_internal(canonical, physical_type)

    def rs_rhos_catalog_calc(
        self,
        M0,
        redshift=0.0,
        dz=0.1,
        zmax=7.0,
        N_ma=100,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=1,
        logmamax=None,
        sigmafac=0,
        N_hermNa=200,
        profile_change=True,
        method="picard_table",
        **solver_options,
    ) -> WeightedSubhaloCatalog:
        r"""Calculate one named physical WDM catalog with explicit numerical inputs.

        Parameters
        ----------
        M0 : float
            Positive z=0 host M200 in physical Msun.
        redshift : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        dz : float
            Positive nominal spacing of the accretion-redshift quadrature grid.
        zmax : float
            Largest requested accretion redshift, above the target epoch; the actual grid
            convention belongs to this variant.
        N_ma : int
            Number of logarithmically spaced accretion-mass nodes.
        sigmalogc : float
            Nonnegative standard deviation of log10 concentration, in dex.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        logmamin : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        logmamax : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        sigmafac : float
            Deterministic host-history offset in units of log-mass scatter when N_hermNa=1;
            low-redshift scatter is anchored at exactly z=1.
        N_hermNa : int
            Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
        profile_change : bool
            Apply the variant tidal response of scale radius and density in addition to
            bound-mass loss.
        method : str
            picard_table (default), dop853 or odeint for population evolution; the Picard
            adapter itself accepts only the first two.
        **solver_options : dict
            Explicit numerical keyword controls for the selected tidal method; stored in
            calculation provenance.

        Returns
        -------
        WeightedSubhaloCatalog
            Canonical physical columns with separate population, concentration and strict
            c_t>0.77 survival weights.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        Host M0 is physical M200 at z=0. All accretion nodes are retained and weight_final
        applies the strict survival mask. The retained arange(target+dz,zmax+dz,dz) convention
        can exceed a non-aligned zmax by less than one nominal step. A wholly zero-measure WDM
        population remains zero rather than being renormalized to NaN.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure, Section 2.1.
            https://arxiv.org/abs/astro-ph/0203004
        """
        parameters = dict(locals())
        parameters.pop("self")
        self._validate_catalog_inputs(parameters)
        execution, diagnostics = self._execute_population(parameters)
        return execution.to_catalog(self._catalog_metadata(parameters, diagnostics))

    def rs_rhos_calc(self, *args: Any, **kwargs: Any):
        r"""Return the historical tuple format of the current population calculation.

        Parameters
        ----------
        *args : tuple
            Positional settings of rs_rhos_catalog_calc, beginning with host M0 in Msun.
        **kwargs : dict
            Named population and tidal-solver settings forwarded to rs_rhos_catalog_calc.

        Returns
        -------
        tuple of numpy.ndarray
            Ten historical-format arrays: ma200, z_acc, r_s_acc, rho_s_acc, m_bound, r_s, rho_s,
            c_t, generation weight and survival.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure, Section 2.1.
            https://arxiv.org/abs/astro-ph/0203004
        """
        catalog = self.rs_rhos_catalog_calc(*args, **kwargs)
        columns = catalog.columns
        return (
            columns["m200_acc"],
            columns["z_acc"],
            columns["r_s_acc"] * 1000.0,
            columns["rho_s_acc"] / 1e18,
            columns["m_bound"],
            columns["r_s"] * 1000.0,
            columns["rho_s"] / 1e18,
            columns["c_t"],
            catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
            columns["survive"],
        )

    def _execute_population(self, parameters):
        r"""Assemble EPS slices and evolve the WDM NFW population through the common executor.

        Parameters
        ----------
        parameters : Mapping or tuple
            Explicit population settings or prepared host-history coefficients, as specified by
            the owning routine.

        Returns
        -------
        tuple
            PopulationExecution and actual Picard diagnostics, or None diagnostics for other
            methods.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure, Section 2.1.
            https://arxiv.org/abs/astro-ph/0203004
        """
        from scipy.integrate import simpson
        from itamae.execution import PopulationComponents
        from ._itamae_components import (
            WDMAccretionSlices,
            WDMCatalogColumns,
            WDMHostHistory,
            WDMInitialStructure,
            WDMProfileEvolution,
            WDMSurvival,
            WDMTidalMassLoss,
            WDMTidalSolver,
        )

        p = parameters
        zdist = np.arange(p["redshift"] + p["dz"], p["zmax"] + p["dz"], p["dz"])
        logmax = np.log10(0.1 * p["M0"]) if p["logmamax"] is None else p["logmamax"]
        ma200 = np.logspace(p["logmamin"], logmax, p["N_ma"])
        ma_by_redshift = np.array(
            [self.Mvir_from_M200(ma200 * _physics.Msolar, za) / _physics.Msolar for za in zdist]
        )
        accretion = self.Na_calc(
            ma_by_redshift,
            zdist,
            p["M0"],
            z0=0,
            N_herm=p["N_hermNa"],
            Nrand=1000,
            sigmafac=p["sigmafac"],
        )
        total = simpson(simpson(accretion, x=np.log(ma_by_redshift)), x=np.log(1 + zdist))
        population = accretion / (1.0 + zdist[:, None])
        population_sum = np.sum(population)
        if population_sum == 0.0 and total == 0.0:
            # A grid wholly below the WDM cutoff carries zero population.
            population = np.zeros_like(population)
        elif (
            not np.isfinite(population_sum)
            or population_sum <= 0.0
            or not np.isfinite(total)
            or total < 0.0
        ):
            raise ValueError(
                "WDM accretion population requires a finite nonnegative normalization."
            )
        else:
            population = population / population_sum * total
        slices = WDMAccretionSlices(
            self, ma200, ma_by_redshift, zdist, population, p["sigmalogc"], p["N_herm"]
        )
        history = WDMHostHistory(self, p["M0"], p["N_hermNa"], p["sigmafac"])
        law = WDMTidalMassLoss(self, history)
        solver = WDMTidalSolver(law, z_min=p["redshift"], z_max=p["zmax"])
        components = PopulationComponents(
            initializer=WDMInitialStructure(self, p["N_herm"]),
            evolver=WDMProfileEvolution(
                self,
                law,
                p["redshift"],
                p["N_herm"],
                p["profile_change"],
                method=p["method"],
                solver_options=p["solver_options"],
                solver=solver,
            ),
            survival=WDMSurvival(),
            columns=WDMCatalogColumns(),
        )
        batches, contexts = zip(*(slices.build(i) for i in range(zdist.size)), strict=True)
        execution = components.execute(
            batches,
            contexts=contexts,
            diagnostics={"variant": "sashimi-w", "accretion_integral": total},
        )
        return execution, solver.backend.diagnostics() if p["method"] == "picard_table" else None

    @staticmethod
    def _validate_catalog_inputs(parameters: dict[str, Any]) -> None:
        """Reject invalid physical and numerical catalog inputs early.

        Parameters
        ----------
        parameters : Mapping or tuple
            Explicit population settings or prepared host-history coefficients, as specified by
            the owning routine.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        if parameters["method"] not in {"picard_table", "dop853", "odeint"}:
            raise ValueError(f"Invalid stripping method: {parameters['method']}")
        for name in ("M0", "redshift", "dz", "zmax", "sigmalogc", "logmamin", "sigmafac"):
            value = np.asarray(parameters[name])
            if value.ndim != 0 or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite scalar.")
        if parameters["logmamax"] is not None:
            value = np.asarray(parameters["logmamax"])
            if value.ndim != 0 or not np.isfinite(float(value)):
                raise ValueError("logmamax must be None or a finite scalar.")
        if float(parameters["M0"]) <= 0.0:
            raise ValueError("M0 must be positive.")
        if float(parameters["redshift"]) < 0.0:
            raise ValueError("redshift must be nonnegative.")
        if float(parameters["dz"]) <= 0.0:
            raise ValueError("dz must be positive.")
        if float(parameters["zmax"]) <= float(parameters["redshift"]):
            raise ValueError("zmax must be greater than redshift.")
        if float(parameters["sigmalogc"]) < 0.0:
            raise ValueError("sigmalogc must be nonnegative.")
        for name in ("N_ma", "N_herm", "N_hermNa"):
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(parameters["N_ma"]) < 2:
            raise ValueError("N_ma must be at least two for mass integration.")
        if not isinstance(parameters["profile_change"], (bool, np.bool_)):
            raise TypeError("profile_change must be boolean.")
        effective_logmamax = (
            np.log10(0.1 * float(parameters["M0"]))
            if parameters["logmamax"] is None
            else float(parameters["logmamax"])
        )
        if float(parameters["logmamin"]) >= effective_logmamax:
            raise ValueError("logmamin must be smaller than logmamax.")

    def _invert_nfw_mass_fraction(self, enclosed_fraction: Any):
        """Exact positive NFW mass inverse from ITAMAE.

        Parameters
        ----------
        enclosed_fraction : array_like
            Dimensionless NFW enclosed-mass value f(c)=m/(4*pi*rho_s*r_s**3), nonnegative; this
            is not the fraction m/M_host.

        Returns
        -------
        float or numpy.ndarray
            Positive tidal concentration r_t/r_s reproducing the specified dimensionless
            enclosed mass.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        return invert_nfw_mass_function(enclosed_fraction)

    def catalog_from_tuple(
        self, result, *, weight_base: Any | None = None, weight_concentration: Any | None = None
    ) -> WeightedSubhaloCatalog:
        """Convert a ten-element tuple format; does not execute an old model.

        Parameters
        ----------
        result : tuple
            Historical ten-array population tuple with mass in Msun, radii in kpc, density in
            Msun/pc^3, and separate generation weight/survival mask.
        weight_base : array_like, optional
            Accretion-abundance factor. If absent, the supplied tuple weight is retained as the
            base factor.
        weight_concentration : array_like, optional
            Independent concentration probability, or ones if it was not supplied. Its product
            with weight_base must reconstruct the tuple generation weight.

        Returns
        -------
        WeightedSubhaloCatalog
            Canonical-unit representation of the supplied tuple, with survival factored
            separately.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        if len(result) != 10:
            raise ValueError(f"Expected 10 tuple outputs, received {len(result)}.")
        survive = np.asarray(result[9], dtype=bool)
        columns = {
            "m200_acc": self._to_canonical(result[0], "mass", "Msun"),
            "z_acc": self._to_canonical(result[1], "dimensionless", "dimensionless"),
            "r_s_acc": self._to_canonical(result[2], "length", "kpc"),
            "rho_s_acc": self._to_canonical(result[3], "density", "Msun/pc3"),
            "m_bound": self._to_canonical(result[4], "mass", "Msun"),
            "r_s": self._to_canonical(result[5], "length", "kpc"),
            "rho_s": self._to_canonical(result[6], "density", "Msun/pc3"),
            "c_t": self._to_canonical(result[7], "dimensionless", "dimensionless"),
            "survive": survive,
        }
        tuple_weight = np.asarray(result[8], dtype=float)
        if weight_base is None:
            weight_base = tuple_weight
        if weight_concentration is None:
            weight_concentration = np.ones(tuple_weight.shape, dtype=float)
        weight_base = np.asarray(weight_base, dtype=float)
        weight_concentration = np.asarray(weight_concentration, dtype=float)
        if (
            weight_base.shape != tuple_weight.shape
            or weight_concentration.shape != tuple_weight.shape
        ):
            raise ValueError("Factorized weights must match the tuple catalog shape.")
        if not np.allclose(weight_base * weight_concentration, tuple_weight, rtol=2e-15, atol=0.0):
            raise ValueError("Factorized weights do not reconstruct the tuple catalog weight.")
        negative_weight = tuple_weight < 0.0
        if np.any(negative_weight):
            count = int(np.count_nonzero(negative_weight))
            minimum = float(np.min(tuple_weight))
            raise ValueError(
                f"Consistent SASHIMI-W produced negative population weights ({count} entries; minimum={minimum:.6e}); this violates the corrected catalog contract."
            )
        return WeightedSubhaloCatalog(
            columns=columns,
            weights={
                "weight_base": weight_base,
                "weight_concentration": weight_concentration,
                "weight_survival": survive.astype(float),
            },
            metadata=self._catalog_metadata(),
        )

    def _catalog_metadata(self, parameters=None, picard_diagnostics=None):
        """Record the transfer, filter, growth, unit and evolution choices of this WDM calculation.

        Parameters
        ----------
        parameters : Mapping or tuple
            Explicit population settings or prepared host-history coefficients, as specified by
            the owning routine.
        picard_diagnostics : dict, optional
            Actual shared-Picard settings and fallback records for the population run; None when
            that backend was not used.

        Returns
        -------
        CatalogMetadata
            Versioned calculation provenance; tuple-only conversion records an unknown solver
            rather than inventing a run identity.

        Notes
        -----
        The supported power ratio is T(k)**2=[1+(alpha*k)**(2*nu)]**(-10/nu), with nu=1.12.
        Canonical catalogues use Msun, Mpc and Msun/Mpc^3; the compatibility tuple uses Msun,
        kpc and Msun/pc^3. No historical q5 calculation is silently converted to this
        specification.
        """
        backend = BackendConfig(self.itamae_cosmology, self.itamae_units)
        variance = self.variance_model()
        return build_calculation_metadata(
            variant="sashimi-w",
            distribution_name="sashimi-w",
            module_file=__file__,
            calculation_specification=CALCULATION_SPECIFICATION,
            model_identifier=f"sashimi-w:wdm:m_wdm_keV={self.mass_wdm:g}:power={self.wdm_power_convention}:v3",
            backend_identifier=backend.identifier,
            source_identifier="sashimi-w:itamae-migration",
            variance_identifier=variance.identifier,
            power_identifier=variance.power.identifier,
            solver_identifier=(
                "unknown:tuple-conversion"
                if parameters is None
                else f"sashimi-w:tidal-stripping:{parameters['method']}:itamae:v3"
            ),
            extra={
                "power_normalization": POWER_NORMALIZATION,
                "sigma8_target": float(_physics.sigma_8),
                "power_amplitude_divisor": self._power_normalization(),
                "sharp_k_mass_assignment": SHARP_K_MASS_ASSIGNMENT,
                "default_stripping_method": "picard_table",
                "picard_diagnostics": picard_diagnostics,
                "eps_numerics": "active-support:direct-variance-gap:normalized-zero-limit:v1",
                "deterministic_scatter_anchor": "exact-z=1",
                "mass_wdm_keV": float(self.mass_wdm),
                "wdm_power_convention": self.wdm_power_convention,
                "wdm_power_formula_role": self.wdm_power_formula_role,
                "wdm_power_q": self.wdm_power_q,
                "wdm_transfer_nu": float(WDM_TRANSFER_NU),
                "half_mode_definition": self.half_mode_definition,
                "half_mode_power_ratio": self.half_mode_power_ratio,
                "half_mode_wavenumber_h_per_mpc": self.half_mode_wavenumber(),
                "half_mode_scales": {
                    "amplitude_half": {
                        "power_ratio": 0.25,
                        "wavenumber_h_per_mpc": self.half_mode_wavenumber(),
                    },
                    "power_half": {
                        "power_ratio": 0.5,
                        "wavenumber_h_per_mpc": self.half_mode_wavenumber(power_ratio=0.5),
                    },
                },
                "cosmology_parameters": {
                    "omega_m0": float(OmegaM),
                    "omega_lambda0": self.omega_lambda,
                    "h": float(h),
                },
                "growth_normalized_at_z0": True,
                "variance_growth_power": 2,
                "variance_mass_unit": "Msun",
                "variance_power_units": {
                    "wavenumber": "1/Mpc",
                    "power": "Mpc^3",
                    "density": "Msun/Mpc^3",
                },
                "physical_to_filter_mass": "M_filter[Msun/h] = h * M_physical[Msun]",
                "accretion_mass_redshift_mapping": "per-redshift-virial-mass-grid",
                "population_weight_contract": "nonnegative-corrected-counts",
                "unit_backend": self.itamae_units.identifier,
                "canonical_units": {"mass": "Msun", "length": "Mpc", "density": "Msun/Mpc^3"},
                "tuple_units": {"mass": "Msun", "length": "kpc", "density": "Msun/pc^3"},
                "tuple_weight_excludes_survival": True,
                "calculation_parameters": {} if parameters is None else parameters,
                "survival": "strict c_t > 0.77",
                "profile": "tidally-evolved NFW",
                "concentration": "Ludlow-2016:100-mass-nodes:500-tophat-nodes",
            },
        )


ItamaeSubhalos = Subhalos
__all__ = ["Subhalos", "ItamaeSubhalos", "STANDARD_T2_Q10"]
