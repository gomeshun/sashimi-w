import numpy as np
from pathlib import Path
from picard_tidal_stripping import endpoint_mass, direct_log_mass, cached_host_mass
from sashimi_w_numerics import SharpKVariance, invert_nfw_mass_function
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import special
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.special import cbrt, gammainc, erf, erfc, hyp2f1
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from numpy.polynomial.hermite import hermgauss
import warnings





###############################
#  Constants
############################### 
cm       = 1.
km       = 1.e5*cm
s        = 1.
gram     = 1.
c        = 2.99792e+10*cm/s
G        = 6.6742e-8*cm**3/gram/s**2
Mpc      = 3.086e+24*cm
kpc      = Mpc/1000.
pc       = kpc/1000.
Msolar   = 1.988435e+33*gram
GeV      = 1.7827e-24*gram
keV      = 1.0e-6*GeV




###############################
#  Matter Power Spectrum
############################### 
""" WMAP7 """ 
filename_PS      = Path(__file__).resolve().parent / "WMAP7_camb_matterpower_z0_extrapolated.dat"
PowerSpectrum    = np.genfromtxt(filename_PS, skip_header = 5)
Pk_file, k_file  = PowerSpectrum[:,0], PowerSpectrum[:,1]
k_min            = k_file.min() * 1.15
k_max            = k_file.max() * 0.86
Pk_interp        = interp1d(k_file, Pk_file)


""" Cosmology from WMAP7 """ 
PS_cosmology  = np.genfromtxt(filename_PS,max_rows=5)
Omegar        = 0.0
Omega0        = 1.0
OmegaB        = PS_cosmology[0]
OmegaM        = PS_cosmology[1]
OmegaC        = OmegaM - OmegaB
OmegaL        = Omega0 - OmegaM - Omegar
pOmega        = [OmegaC+OmegaB,Omegar,OmegaL]
h             = PS_cosmology[2]
H0            = h*100*km/s/Mpc 
rhocrit0      = 3*pow(H0,2)*pow(8.0*np.pi*G,-1)
sigma_8       = PS_cosmology[4]




def _cumulative_above(values, weights, bins=10000):
    """Return total weight and exact N(value > x) at histogram right edges.

    Bins select display thresholds only; totals and the strict cumulative counts
    are evaluated from the selected samples, so no first-bin count is lost.
    Empty selections retain a finite zero curve on NumPy's default [0, 1] grid.
    """
    values, weights = np.asarray(values), np.asarray(weights)
    thresholds = np.histogram_bin_edges(values, bins=bins)[1:]
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    tail = np.concatenate((np.cumsum(sorted_weights[::-1])[::-1], [0.0]))
    above = np.searchsorted(sorted_values, thresholds, side="right")
    return float(np.sum(weights)), thresholds, tail[above]


class subhalos:

    def __init__(self, mass_wdm=1.5):
        if not np.isfinite(mass_wdm) or mass_wdm <= 0:
            raise ValueError("mass_wdm must be finite and positive.")
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

        self._prepare_variance()

    def _prepare_variance(self):
        key = (float(self.mass_wdm), float(OmegaM), float(OmegaC), float(h), float(sigma_8))
        if getattr(self, '_variance_key', None) != key:
            if not np.isfinite(self.mass_wdm) or self.mass_wdm <= 0:
                raise ValueError("mass_wdm must be finite and positive.")
            self.a = 0.049*(1/self.mass_wdm)**1.11*(OmegaC/0.25)**0.11*(h/0.7)**1.22
            self._variance = SharpKVariance(k_file, Pk_file, self.a, self.Rhomean_z, h,
                                          self.MassIn8Mpc/h, sigma_8)
            self._variance_key = key
            self.normalise = np.sqrt(self._variance.normalization2)
            self.sig_8 = self.normalise*sigma_8
            self.Sigma = np.sqrt(self._variance.variance(self.filter_Mass/h))
            self.Sigma_Sq = self.Sigma**2
            self.sigma_sq = self.Sigma_Sq*self.normalise**2

    def Sigma_interp(self, mass_in_msun_over_h):
        """Compatibility entry point: the numerical mass coordinate is Msun/h."""
        return self.sigmaMz(np.asarray(mass_in_msun_over_h)/h, 0.)

    def sig_interp(self, mass_in_msun_over_h):
        return self.Sigma_interp(mass_in_msun_over_h)*self.normalise

    def power_ratio(self, k):
        """Viel power suppression, T(k)^2: the adopted exponent is q=10."""
        self._prepare_variance()
        return (1+(self.a*np.asarray(k))**(2*1.12))**(-10./1.12)

    def transfer_amplitude(self, k):
        """Transfer amplitude; amplitude-half means power ratio one quarter."""
        return np.sqrt(self.power_ratio(k))


    def dlnSigmadlnM_interp(self, log_mass_in_msun_over_h):
        mass = np.exp(np.asarray(log_mass_in_msun_over_h))/h
        return self.dsdm(mass, 0.)*mass/(2*self.sigmaMz(mass, 0.)**2)
    def TopHat(self,k, r):
        return 3.0/(k*r)**2 * (np.sin(k*r)/(k*r) - np.cos(k*r))  

    def SigmaIntegrand(self, k, r):
        om_wdm = OmegaM - OmegaB
        a = 0.049 * (1 / self.mass_wdm) ** 1.11 * (om_wdm / 0.25) ** 0.11 * (h / 0.7) ** 1.22
        return k**2 * Pk_interp(k) * self.TopHat(k, r) ** 2 * (1+(a*k)**(2*1.12))**(-10./1.12)

    """ Integration function """ 
    def integratePk_th(self,kmin, kmax, r):

        log_k_min_th   = np.log10(kmin) 
        log_k_max_th   = np.log10(kmax)
        dlogk_th       = (log_k_max_th - log_k_min_th)/(500 - 1)
        tot_sum_th     = 0. 

        logk_th = log_k_min_th+np.arange(500)*dlogk_th
        logk_th = np.reshape(logk_th,(500,1))
        sum_rect_th = self.SigmaIntegrand(10**logk_th, r) * 10**logk_th * np.log(10)
        tot_sum_th  = np.sum(sum_rect_th,axis=0)

        log_k_min_th    = log_k_min_th - dlogk_th
        sum_rect_min_th = self.SigmaIntegrand(10**log_k_min_th, r) * 10**log_k_min_th * np.log(10)
        log_k_max_th    = log_k_max_th + dlogk_th
        sum_rect_max_th = self.SigmaIntegrand(10**log_k_max_th, r) * 10**log_k_max_th * np.log(10)
        
        sigma_sq_th     = (tot_sum_th + 0.5*sum_rect_min_th + 0.5*sum_rect_max_th) * dlogk_th
        sigma_sq_th    /= (2*np.pi**2)
        return sigma_sq_th            

    def linear_growth_factor(self, Omega_m0, Omega_l0, z):
        if len(np.atleast_1d(z)) == 2:
            z1    = z[0]
            z2    = z[1] # z2 > z1                                                                                            

        if (len(np.atleast_1d(z)) == 1) or (len(np.atleast_1d(z)) > 2):
            z1    = 0.
            z2    = z    # z2 > z1                                                                                           
                 
        Omega_lz1 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z1)**3)
        Omega_mz1 = 1. - Omega_lz1
        gz1       = (5./2.) * Omega_mz1 / (Omega_mz1**(4./7.) - Omega_lz1 + (1. + Omega_mz1/2.) * (1. + Omega_lz1/70.))
        Omega_lz2 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z2)**3)
        Omega_mz2 = 1. - Omega_lz2
        gz2       = (5./2.) * Omega_mz2 / (Omega_mz2**(4./7.) - Omega_lz2 + (1. + Omega_mz2/2.) * (1. + Omega_lz2/70.))
        return (gz2 / (1.+z2)) / (gz1 / (1+z1))



    def conc200(self, M, z):
        """Ludlow concentration on the unchanged 100-mass top-hat table.

        Cache only the spectrum integral. Evaluate all real formation trials
        in arrays, retaining the original nearest-mass and c interpolation.
        """
        mass, redshift = np.broadcast_arrays(np.asarray(M)/Msolar*h,np.asarray(z))
        if np.any(~np.isfinite(mass)) or np.any(mass<=0) or np.any(~np.isfinite(redshift)) or np.any(redshift<=-1):
            raise ValueError("Concentration requires finite M>0 and z>-1.")
        if mass.size==0:
            return np.empty(mass.shape)
        cache_key=(float(self.mass_wdm),OmegaM,OmegaL,h,sigma_8,self.filter_Mass.tobytes())
        if getattr(self,'_concentration_key',None)!=cache_key:
            radius=cbrt(self.filter_Mass/(4/3*np.pi*self.Rhomean_z))
            sigma=np.sqrt(self.integratePk_th(k_min,k_max,radius))
            normalization=interp1d(self.filter_Mass,sigma)(self.MassIn8Mpc)/sigma_8
            variance=(sigma/normalization)**2
            self._concentration_variance=variance
            self._concentration_spline=splrep(self.logM0-10.,variance,k=1)
            self._concentration_key=cache_key
        variance=self._concentration_variance
        spline=self._concentration_spline
        index=np.argmin(np.abs(self.filter_Mass[:,None]-mass.ravel()[None,:]),axis=0)
        gap=splev(self.logM0[index]-10.+np.log10(.02),spline)-variance[index]
        c=10**(np.arange(100)*4./99.)
        fraction=(np.log(2.)-.5)/(np.log(1.+c)-c/(1.+c))
        rho_2=200.*c**3*fraction
        ratio=rho_2/(200.*(650./200.))
        radicand=(ratio[None,:]*(OmegaM*(1+redshift.ravel()[:,None])**3+OmegaL)-OmegaL)/OmegaM
        valid=radicand>0
        delta_formation=np.zeros_like(radicand)
        z2=radicand[valid]**.3333-1.
        # Pointwise growth avoids the public two-element interval convention.
        delta_formation[valid]=1.686/self.growthD(z2)
        delta_now=1.686/self.growthD(redshift.ravel())
        if np.any(gap<=0):
            raise ValueError("Native top-hat concentration has no positive variance gap at this mass.")
        arg=(650./200.)*ratio[None,:]/c[None,:]**3-(1.-erf((delta_formation-delta_now[:,None])/np.sqrt(2*gap[:,None])))
        valid &= np.isfinite(arg)
        result=np.empty(mass.size)
        for i in range(mass.size):
            if not np.any(valid[i]):
                raise ValueError("No finite real concentration-formation trials remain.")
            result[i]=np.interp(0.,arg[i,valid[i]],c[valid[i]])
        result=result.reshape(mass.shape)
        return float(result) if result.ndim==0 else result

    @staticmethod
    def _real_formation_trials(concentration, density_ratio, redshift):
        """Drop the same non-real trials before evaluating background growth.

        The original fractional exponent is preserved. Trials with nonpositive
        radicand have no supported finite z > -1 and were already discarded as
        non-finite values by the final concentration interpolation.
        """
        omega_lambda = 1.0 - OmegaM
        radicand = (
            1.0
            / OmegaM
            * (density_ratio * (OmegaM * (1 + redshift) ** 3 + omega_lambda) - omega_lambda)
        )
        if not np.all(np.isfinite(radicand)):
            raise ValueError("Concentration trial radicands must be finite.")
        valid = radicand > 0.0
        if not np.any(valid):
            raise ValueError("No real concentration-formation trials remain.")
        return concentration[valid], density_ratio[valid], radicand[valid] ** 0.3333 - 1.0
       


    ###############################
    #  Functions for subhalo model
    ###############################    

    def fc(self, x):
        return np.log(1+x)-x*pow(1+x,-1)

    def g(self, z):
        return (OmegaB+OmegaC)*(1.+z)**3+Omegar*(1+z)**4+OmegaL

    def rhocrit(self, z):
        return 3.0*pow(self.Hz(z),2)*pow(np.pi*8.0*G,-1)

    def Mvir_from_M200(self, M200, z):
        gz = self.g(z)
        c200 = self.conc200(M200,z)
        r200 = (3.0*M200/(4*np.pi*200*rhocrit0*gz))**(1./3.)
        rs = r200/c200
        fc200 = self.fc(c200)
        rhos = M200/(4*np.pi*rs**3*fc200)
        Dc = self.Delc(self.Omegaz(pOmega,z)-1.)
        rvir = optimize.fsolve(lambda r: 3.*(rs/r)**3*self.fc(r/rs)*rhos-Dc*rhocrit0*gz,r200)
        Mvir = 4*np.pi*rs**3*rhos*self.fc(rvir/rs)
        return Mvir

    def _Mvir_from_M200_grid(self, M200, z):
        """Solve the same NFW density equation independently for each halo."""
        mass, redshift=np.broadcast_arrays(np.asarray(M200),np.asarray(z))
        concentration=self.conc200(mass,redshift)
        overdensity=self.Delc(self.Omegaz(pOmega,redshift)-1.)
        c200=np.broadcast_to(concentration,mass.shape).ravel()
        target=(overdensity/200.*self.fc(concentration)/np.asarray(concentration)**3).ravel()
        cvir=np.empty_like(c200)
        for i,(c0,density) in enumerate(zip(c200,target)):
            lower,upper=c0/10.,c0*10.
            cvir[i]=optimize.brentq(lambda x:self.fc(x)/x**3-density,
                                     lower,upper,xtol=1e-13,rtol=1e-13)
        result=mass*self.fc(cvir.reshape(mass.shape))/self.fc(concentration)
        return float(result) if result.ndim==0 else result

    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.Omegaz(pOmega,z)
        def ffunc(x):
            return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
        def xfunc(f):
            p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
            return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
        return self.Delc(Oz-1)/200.0*M200 \
            *np.power(self.conc200(M200,z) \
            *xfunc(self.Delc(Oz-1)/200.0*ffunc(1.0/self.conc200(M200,z))),-3.0)

    def growthD(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
        return (Omega_Mz/OmegaM)*(phi0/phiz)*pow(1+z,-1)

    def xi(self, M):
        return pow(M*pow((1e+10)*pow(h,-1),-1),-1)

    def sigmaMz(self, M, z):
        self._prepare_variance()
        return np.sqrt(self._variance.variance(M))*self.growthD(z)

    def dOdz(self, z):
        return -OmegaL*3*OmegaM*pow(1+z,2)*pow(OmegaL+OmegaM*pow(1+z,3),-2)

    def dDdz(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
        dphidz = self.dOdz(z)*((-4.0/7.0)*pow(Omega_Mz,-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.0+(1.0/70.0)-(3.0/2.0))
        return (phi0/OmegaM)*(-self.dOdz(z)*pow(phiz*(1+z),-1)-Omega_Mz*(dphidz*(1+z)+phiz)*pow(phiz,-2)*pow(1+z,-2))

    def Mzi(self, M0, z):
        a = 1.686*np.sqrt(2.0/np.pi)*self.dDdz(0)+1.0
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fM0 = pow(pow(self.sigmaMz(M0/q,0),2)-pow(self.sigmaMz(M0,0),2),-0.5)
        return M0*pow(1+z,a*fM0)*np.exp(-fM0*z)

    def Mzzi(self, M0, z, zi):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        return Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))

    def Hz(self, z):
        return H0*np.sqrt(OmegaL+OmegaM*pow(1+z,3))

    def Omegaz(self, p, x):
        E=p[0]*pow(1+x,3)+p[1]*pow(1+x,2)+p[2]
        return p[0]*pow(1+x,3)*pow(E,-1)

    def Delc(self, x):
        return 18*pow(np.pi,2)+(82.*x)-39*pow(x,2)

    def dMdz(self, M0, z, zi, sigmafac=0):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        Mzzidef = Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef*Msolar,z)
        return (beta+alpha*pow(1+z-zi,-1))*Mzzivir/Msolar

    def dsdm(self, M, z):
        self._prepare_variance()
        return self._variance.derivative(M)*self.growthD(z)**2

    def delc_Y11(self,M, z):
        """ Critical overdensity for collapse for WDM, Benson et al. (2012): Eq.7) """ 
        gx = 1.5
        z_eq = 3600*(OmegaM*pow(h,2)/0.15)-1
        Mj = 3.06*10**8* pow((1+z_eq)/3000,1.5) * pow((OmegaM*pow(h,2)/0.15),0.5)* pow(gx/1.5,-1) * pow(self.mass_wdm,-4) 
        x = np.log(M/Mj)
        hh = special.expit(-(x+2.4)/0.1)
        # The discarded (1-hh)=0 branch overflows below the WDM cutoff.
        x, hh = np.broadcast_arrays(x, hh)
        correction = hh*(0.04/np.exp(2.3*x))
        active = hh != 1.
        correction[active] += (1-hh[active])*np.exp(0.31687/np.exp(0.809*x[active]))
        return 1.686/self.growthD(z)*correction

    def s_Y11(self, M):
        return pow(self.sigmaMz(M,0),2)

    def Ffunc_Yang(self, delc1, delc2, sig1, sig2):
        """ Returns Eq. (14) of Yang et al. (2011) """
        return pow(2*np.pi,-0.5)*(delc2-delc1)*pow(sig2-sig1,-1.5) \
            *np.exp(-pow(delc2-delc1,2)*pow(2*(sig2-sig1),-1))

    def Na_calc(self, ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000, sigmafac=0):
        """ Returns Na, Eq. (3) of Yang et al. (2011) """ 
        zacc_2d = np.asarray(zacc).reshape(-1,1)
        M200_0 = self.Mzzi(Mhost,zacc_2d,z0)
        logM200_0 = np.log10(M200_0)
        if N_herm==1:
            sigmalogM200_0 = 0.12+0.15*np.log10(Mhost/M200_0)
            Mz1 = self.Mzzi(Mhost, 1., z0)
            sigma1 = 0.12-0.15*np.log10(Mz1/Mhost)
            sigmalogM200_1 = sigma1/np.log10(Mz1/Mhost)*np.log10(M200_0/Mhost)
            sigmalogM200 = np.where(zacc_2d>1.,sigmalogM200_0,sigmalogM200_1)
            logM200=logM200_0+sigmafac*sigmalogM200
            M200=10**logM200
            if(sigmafac>0.):
                M200 = np.where(M200<Mhost,M200,Mhost)
        else:
            xxi,wwi = hermgauss(N_herm)
            xxi = xxi.reshape(len(xxi),1,1)
            wwi = wwi.reshape(len(wwi),1,1)
            """ eq. (21) in Yang et al. (2011) """ 
            sigmalogM200 = 0.12-0.15*np.log10(M200_0/Mhost)
            logM200 = np.sqrt(2)*sigmalogM200*xxi+logM200_0
            M200 = 10**logM200
        M200 = np.asarray(M200).reshape(N_herm, len(zacc_2d), 1)
        mmax=np.minimum(M200,Mhost/2.0)
        Mmax=np.minimum(M200_0+mmax,Mhost)
        zlist = zacc_2d*np.linspace(1,0,Nrand)
        iMmax = np.argmin(np.abs(self.Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
        z_Max = zlist[np.arange(len(zlist)),iMmax]
        z_Max_3d = z_Max.reshape(N_herm,len(zlist),1)
        delcM = self.delc_Y11(Mmax,z_Max_3d)
        delca = self.delc_Y11(ma,zacc_2d)
        d1,d2,small,big,minimum,allowed = np.broadcast_arrays(
            delcM,delca,ma,Mmax,mmax,mmax>ma)
        Phi = np.zeros(allowed.shape)
        # Integrate variance gaps directly; subtracting two saturated WDM
        # variances loses the positive support in high-order host tails.
        ds = self._variance.gap(small[allowed],big[allowed])
        dsmin = self._variance.gap(minimum[allowed],big[allowed])
        gap = d2[allowed]-d1[allowed]
        if np.any(gap < 0) or np.any(ds <= 0) or np.any(dsmin <= 0):
            raise ValueError("WDM EPS requires nonnegative barrier and positive variance gaps.")
        # Both erf factors vanish at zero barrier gap, leaving this limit.
        ratio = gap/np.sqrt(2*dsmin)
        values = np.empty_like(gap)
        small_gap = ratio < 1e-4
        t = ratio[small_gap]**2
        values[small_gap] = (np.sqrt(dsmin[small_gap])/(2*ds[small_gap]**1.5)
                            * np.exp(-gap[small_gap]**2/(2*ds[small_gap]))
                            /(1-t/3+t*t/10-t*t*t/42))
        regular = ~small_gap
        exponent = gap[regular]/np.sqrt(ds[regular])
        # Evaluate the complete kernel in log space. Only omit exponents
        # whose square cannot affect any representable floating-point rate.
        active = exponent < 1e150
        ordinary = np.zeros(exponent.shape)
        ordinary[active] = np.exp(np.log(gap[regular][active])
            -1.5*np.log(ds[regular][active])-.5*np.log(2*np.pi)
            -.5*exponent[active]**2-np.log(special.erf(ratio[regular][active])))
        values[regular] = ordinary
        Phi[allowed] = values
        if not np.all(np.isfinite(Phi)):
            raise ValueError("Non-finite WDM EPS kernel in the active domain.")
        F2 = Phi[0] if N_herm==1 else np.sum(Phi*wwi/np.sqrt(np.pi),axis=0)
        Na = F2*self.dsdm(ma,0)*self.dMdz(Mhost,zacc_2d,z0,sigmafac)*(1+zacc_2d)
        return Na

    ###############################
    # Calculate subhalo properties at accretion and after tidal stripping
    ###############################         

    def rs_rhos_calc(self, M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=100, sigmalogc=0.128,
                     N_herm=5, logmamin=1, logmamax=None, sigmafac=0,
                     N_hermNa=200, profile_change=True, method="picard_table", **kwargs):

        zdist = np.arange(redshift+dz,zmax+dz,dz)
        if logmamax==None:
            logmamax = np.log10(0.1*M0)
        ma200 = np.logspace(logmamin,logmamax,N_ma)
        rs_acc = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_acc = np.zeros((len(zdist),N_herm,len(ma200)))
        rs_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        ct_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        survive=np.zeros((len(zdist),N_herm,len(ma200)))
        m0_matrix = np.zeros((len(zdist),N_herm,len(ma200)))
        Oz_0 = self.Omegaz(pOmega,redshift)

        solver = TidalStrippingSolver(self, M0, redshift, zmax, N_hermNa, sigmafac)
        self.stripping_solver = solver

        ma_by_redshift = np.stack([
            self.Mvir_from_M200(ma200*Msolar,z)/Msolar for z in zdist
        ])
        for iz in range(len(zdist)):
            ma = ma_by_redshift[iz]
            Oz = self.Omegaz(pOmega,zdist[iz])
            m0 = solver.subhalo_mass_stripped(ma,zdist[iz],redshift,method=method,**kwargs)
            c200sub = self.conc200(ma200*Msolar,zdist[iz])
            rvirsub = pow(3*ma*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *self.Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
            r200sub = pow(3*ma200*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *200*4*np.pi,-1),1.0/3.0)
            c_mz = c200sub*rvirsub/r200sub
            x1,w1 = hermgauss(N_herm)
            x1 = x1.reshape(len(x1),1)
            w1 = w1.reshape(len(w1),1)
            log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
            c_sub = pow(10.0,log10c_sub)
            rs_acc[iz] = rvirsub/c_sub
            rhos_acc[iz] = ma*Msolar/(4*np.pi*rs_acc[iz]**3*self.fc(c_sub))
            if(profile_change==True):
                rmax_acc = rs_acc[iz]*2.163
                Vmax_acc = np.sqrt(rhos_acc[iz]*4*np.pi*G/4.625)*rs_acc[iz]
                Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
                rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
                rs_z0[iz] = rmax_z0/2.163
                rhos_z0[iz] = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0[iz],2)
            else:
                rs_z0[iz] = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]
            ct_z0[iz] = invert_nfw_mass_function(m0*Msolar/(4*np.pi*rhos_z0[iz]*rs_z0[iz]**3))
            survive[iz] = np.where(ct_z0[iz]>0.77,1,0)
            m0_matrix[iz] = m0*np.ones((N_herm,1))

        Na = self.Na_calc(ma_by_redshift,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,
                          sigmafac=sigmafac)
        Na_total = integrate.simpson(integrate.simpson(Na,x=np.log(ma_by_redshift)),x=np.log(1+zdist))
        weight = Na/(1.0+zdist.reshape(len(zdist),1))
        weight_sum = np.sum(weight)
        if weight_sum == 0 and Na_total == 0:
            # A grid wholly below the WDM cutoff represents no population.
            weight = np.zeros_like(weight)
        elif not np.isfinite(weight_sum) or weight_sum <= 0 or Na_total < 0:
            raise ValueError("WDM accretion normalization must be finite and nonnegative.")
        else:
            weight = weight/weight_sum*Na_total
        weight = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
        z_acc = (zdist.reshape(len(zdist),1,1))*np.ones((1,N_herm,N_ma))
        z_acc = z_acc.reshape(len(zdist)*N_herm*N_ma)
        ma200_matrix = ma200*np.ones((len(zdist),N_herm,1))
        ma200_matrix = ma200_matrix.reshape(len(zdist)*N_herm*len(ma200))
        m0_matrix = m0_matrix.reshape(len(zdist)*N_herm*len(ma200))
        rs_acc = rs_acc.reshape(len(zdist)*N_herm*len(ma200))
        rhos_acc = rhos_acc.reshape(len(zdist)*N_herm*len(ma200))
        rs_z0 = rs_z0.reshape(len(zdist)*N_herm*len(ma200))
        rhos_z0 = rhos_z0.reshape(len(zdist)*N_herm*len(ma200))
        ct_z0 = ct_z0.reshape(len(zdist)*N_herm*len(ma200))
        weight = weight.reshape(len(zdist)*N_herm*len(ma200))
        survive = (survive==1).reshape(len(zdist)*N_herm*len(ma200))

        return ma200_matrix, z_acc, rs_acc/kpc, rhos_acc/(Msolar/pc**3), \
            m0_matrix, rs_z0/kpc, rhos_z0/(Msolar/pc**3), ct_z0, \
            weight, survive
            
            




    #################################################################################
    # Subhalo distribution with host halo M0
    # Input: M0 in units [M_solar]. Optional: Distribution at accretion (accretion=True) instead of at present (redshift = 0)
    # Output: subhalo masses in units [M_solar], subhalo distribution dN/dm
    #################################################################################
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
        method="picard_table",
        **kwargs,
    ):
        """Return dN/dm of current survivors using bound or accretion mass.

        Both displays apply tidal evolution and survival at the observation
        redshift. accretion=True changes the mass coordinate to ma200 only.
        Returned m is the logarithmic bin center in Msun.
        """
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
            profile_change=profile_change,
            method=method,
            **kwargs,
        )
        selected = np.array(survive, dtype=bool, copy=True)
        mass = ma200 if accretion else m0
        N, lnm_edges = np.histogram(np.log(mass[selected]), weights=weight[selected], bins=100)
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        dlnm = lnm_edges[1:] - lnm_edges[:-1]
        m = np.exp(lnm)
        dNdlnm = N / dlnm
        dNdm = dNdlnm / m
        return (m, dNdm)



    #################################################################################
    # Calculate expected number of satellites for given host halo.
    # Input: M0, Mpeak in units [M_solar]
    # Optional: Satellite forming condition with threshold on subhalo peak mass, Mpeak, in units [M_solar] (Mpeak_thres=True)
    # Output: Total number of satellites, subhalo masses [M_solar], cumulative distribution subhalo mass
    #################################################################################
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
        method="picard_table",
        **kwargs,
    ):
        """Return survivor count and N(ma200 > x), with x in Msun.

        Optional Mpeak selection is strict ma200 > Mpeak. x contains the
        10000 histogram right edges; cumulative counts are exact sample sums.
        The total counts all selected weights, including the first bin.
        """
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
            profile_change=profile_change,
            method=method,
            **kwargs,
        )
        selected = np.array(survive, dtype=bool, copy=True)
        if Mpeak_thres:
            selected &= ma200 > Mpeak
        return _cumulative_above(ma200[selected], weight[selected])


    #################################################################################
    # Calculate expected number of satellites for given host halo with threshold on Vpeak (Vpeak_thres=True) or Vmax (Vpeak_thres=False)
    # Input: M0 in units [M_solar], Vpeak/Vmax in units [km/s]
    # Output: Total number of satellites, subhalo Vmax or Vpeak [km/s], cumulative distribution subhalo Vmax or Vpeak
    #################################################################################
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
        method="picard_table",
        **kwargs,
    ):
        """Return survivor count and N(Vmax > x), with x in km/s.

        The strict selection uses accretion Vpeak when Vpeak_thres=True,
        otherwise current Vmax. The curve always uses current Vmax and
        10000 histogram right-edge thresholds, with exact sample sums.
        """
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
            profile_change=profile_change,
            method=method,
            **kwargs,
        )
        rs_a = rs_a * kpc
        rs0 = rs0 * kpc
        rhos_a = rhos_a * (Msolar / pc**3)
        rhos0 = rhos0 * (Msolar / pc**3)
        Vpeak = np.sqrt(4.0 * np.pi * G * rhos_a / 4.625) * rs_a / (km / s)
        Vmax = np.sqrt(4.0 * np.pi * G * rhos0 / 4.625) * rs0 / (km / s)
        selected = np.array(survive, dtype=bool, copy=True)
        selected &= (Vpeak if Vpeak_thres else Vmax) > Vpeak_max
        return _cumulative_above(Vmax[selected], weight[selected])


class TidalStrippingSolver:
    """Standalone WDM coefficients and host history for either tidal solver.

    M0 and returned masses use physical Msun. The original W coefficients,
    host-quantile setting, dynamical time, and exact virial conversion remain
    the provider for both direct odeint and Picard.
    """
    def __init__(self, model, M0, z_min=0., z_max=7., N_hermNa=200, sigmafac=0.):
        self.model, self.M0 = model, float(M0)
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.N_hermNa, self.sigmafac = N_hermNa, sigmafac

    def picard_physics_key(self):
        return (self.model.mass_wdm, OmegaM, OmegaB, OmegaL, h, H0, rhocrit0, sigma_8)

    def Mzvir(self, z):
        return cached_host_mass(self, z)


    def _Mzvir_uncached(self, z):
        z = np.asarray(z, dtype=float)
        Mz200 = self.model.Mzzi(self.M0,z,0.)
        if self.N_hermNa==1:
            sig0 = .12-.15*np.log10(Mz200/self.M0)
            Mz1 = self.model.Mzzi(self.M0,1.,0.)
            sigma1 = .12-.15*np.log10(Mz1/self.M0)
            sig1 = sigma1/np.log10(Mz1/self.M0)*np.log10(Mz200/self.M0)
            Mz200 = 10**(np.log10(Mz200)+self.sigmafac*np.where(z>1.,sig0,sig1))
            if self.sigmafac>0:
                Mz200 = np.minimum(Mz200,self.M0)
        if z.ndim>0 and z.size>1:
            return self.model._Mvir_from_M200_grid(Mz200*Msolar,z)/Msolar
        return self.model.Mvir_from_M200(Mz200*Msolar,z)/Msolar

    def AMz(self, z):
        loghost = np.log10(self.Mzvir(z))
        return 10**((-.0019*loghost+.045)*z+(.0097*loghost-.313))

    def zetaMz(self, z):
        loghost = np.log10(self.Mzvir(z))
        return (-5.55e-5*loghost+1.43e-3)*z+(3.34e-4*loghost-8.11e-3)

    def tdynz(self, z):
        omega = self.model.Omegaz(pOmega,z)
        return 1.628/h*(self.model.Delc(omega-1)/178.)**-.5/(self.model.Hz(z)/H0)*(86400*365*1e9)

    def Phi(self, z):
        return self.AMz(z)/self.tdynz(z)/self.model.Hz(z)/(1+z)

    def msolve(self, m, z):
        return self.AMz(z)*(m/self.tdynz(z))*(m/self.Mzvir(z))**self.zetaMz(z)/(self.model.Hz(z)*(1+z))

    def subhalo_mass_stripped(self, ma, za, z, method="picard_table", **kwargs):
        if method=="picard_table":
            return endpoint_mass(self,ma,za,z,**kwargs)
        if method=="dop853":
            return direct_log_mass(self,ma,za,z,**kwargs)
        if method=="odeint":
            return odeint(self.msolve,ma,np.linspace(za,z,100),**kwargs)[-1]
        raise ValueError(f"Invalid stripping method: {method}")
