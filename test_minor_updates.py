"""Independent checks for the standalone W maintenance corrections."""
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

import sashimi_w as w


@pytest.fixture(scope="module")
def model():
    return w.subhalos(2.)


def test_flat_background_and_growth(model):
    assert w.OmegaM + w.OmegaL == 1.
    assert model.Hz(0.) == w.H0
    assert model.growthD(0.) == 1.
    z = np.array([0., .5, 1., 3.])
    step = 1e-5
    fd = (model.growthD(z+step)-model.growthD(z-step))/(2*step)
    np.testing.assert_allclose(model.dDdz(z), fd, rtol=2e-9)


@pytest.mark.parametrize("mass", [1e7, 1e9, 1e12])
def test_variance_is_independent_sharp_k_integral(model, mass):
    def raw(m):
        radius = (3*(m*w.h)/(4*np.pi*model.Rhomean_z))**(1/3)/2.5
        cutoff = min((9*np.pi/2)**(1/3)/radius, w.k_file[-1])
        knots = np.log(w.k_file[w.k_file < cutoff])
        knots = np.r_[knots, np.log(cutoff)]
        def integrand(logk):
            k = np.exp(logk)
            alpha = .049*(1/model.mass_wdm)**1.11*(w.OmegaC/.25)**.11*(w.h/.7)**1.22
            ratio = (1+(alpha*k)**(2*1.12))**(-10/1.12)
            return k**3*np.interp(k,w.k_file,w.Pk_file)*ratio/(2*np.pi**2)
        return sum(quad(integrand,a,b,epsabs=0,epsrel=2e-12)[0] for a,b in zip(knots[:-1],knots[1:]))
    expected = raw(mass)/raw(model.MassIn8Mpc/w.h)*w.sigma_8**2
    np.testing.assert_allclose(model.sigmaMz(mass,0.)**2,expected,rtol=2e-11)


def test_variance_derivative_and_growth_square(model):
    mass = np.logspace(8,14,21)
    step = 1e-4
    fd = (model.sigmaMz(mass*np.exp(step),0.)**2-model.sigmaMz(mass*np.exp(-step),0.)**2)/(2*step*mass)
    # At the WDM plateau the finite difference subtracts almost equal S.
    # Carry the floating-point subtraction floor, in dS/dM units, explicitly.
    roundoff = 4*np.finfo(float).eps*model.sigmaMz(mass,0.)**2/(step*mass)
    assert np.all(np.abs(model.dsdm(mass,0.)-fd) <= 2e-5*np.abs(fd)+roundoff)
    np.testing.assert_allclose(model.dsdm(mass,2.),model.dsdm(mass,0.)*model.growthD(2.)**2,rtol=2e-14)
    assert np.all(model.dsdm(mass,0.) < 0)


def test_top_hat_and_sharp_k_both_use_power_q10(model):
    k = np.array([.01, 1., 30.])
    ratio = (1+(model.a*k)**(2*1.12))**(-10/1.12)
    np.testing.assert_allclose(model.SigmaIntegrand(k,1.)/(k*k*w.Pk_interp(k)*model.TopHat(k,1.)**2),ratio,rtol=2e-14)
    np.testing.assert_allclose(model.power_ratio(k),ratio,rtol=2e-14)
    np.testing.assert_allclose(model.transfer_amplitude(k)**2,ratio,rtol=2e-14)


def test_concentration_trials_keep_real_future_times(model):
    with np.errstate(invalid="raise"):
        c,r,z = model._real_formation_trials(np.array([1.,2.,3.]),np.array([.5,.9,2.]),0.)
        actual = model.conc200(np.array([1e8,1e10])*w.Msolar,0.)
    assert np.any(z < 0) and np.all(z > -1)
    assert np.all(np.isfinite(actual))


def test_exact_nfw_boundary(model):
    from sashimi_w_numerics import invert_nfw_mass_function
    radii = np.array([0.,.77-1e-8,.77+1e-8,1e-5,500.])
    with mp.workdps(65):
        mass = np.array([float(mp.log1p(float(x))-mp.mpf(float(x))/(1+mp.mpf(float(x)))) for x in radii])
    actual = invert_nfw_mass_function(mass)
    np.testing.assert_allclose(actual,radii,rtol=1e-11)
    np.testing.assert_array_equal(actual>.77,radii>.77)


def synthetic_catalog(alive=(True,False,True)):
    ma=np.array([10.,20.,30.]); z=np.zeros(3); rs=np.ones(3)*.1; rho=np.ones(3)*.01
    return ma,z,rs,rho,ma/2,rs*.7,rho*1.2,np.ones(3),np.array([2.,3.,5.]),np.array(alive)


@pytest.mark.parametrize("alive,total", [((True,False,True),7.),((False,False,False),0.),((True,False,False),2.)])
def test_survivor_counts_exact_cumulative_and_forwarding(monkeypatch,alive,total):
    model=w.subhalos(2.)
    arrays=synthetic_catalog(alive); originals=tuple(x.copy() for x in arrays)
    def fake(*args,**kwargs):
        assert kwargs.get('profile_change', args[-1] if args else None) is False
        return arrays
    monkeypatch.setattr(model,'rs_rhos_calc',fake)
    count,x,curve=model.N_sat(1e12,profile_change=False)
    assert count==total
    expected=[np.sum(arrays[8][arrays[9] & (arrays[0]>threshold)]) for threshold in x]
    np.testing.assert_array_equal(curve,expected)
    assert model.N_sat(1e12,Mpeak=10.,Mpeak_thres=True,profile_change=False)[0] == (5. if alive[2] else 0.)
    assert model.N_sat_Vthres(1e12,0.,profile_change=False)[0] == total
    for accretion in (False,True):
        mass,dndm=model.subhalo_distr(1e12,accretion=accretion,profile_change=False)
        # Logarithmic bins have a constant width; reconstruct integrated count.
        np.testing.assert_allclose(np.sum(dndm*mass)*(np.log(mass[1])-np.log(mass[0])),total,atol=1e-12)
    for old,new in zip(originals,arrays):
        np.testing.assert_array_equal(old,new)


def test_cumulative_empty_and_single():
    from sashimi_w import _cumulative_above
    for value,weight,total in [([],[],0.),([3.],[2.],2.)]:
        n,x,y=_cumulative_above(value,weight)
        assert n==total and np.all(np.isfinite(x))
        for threshold,actual in zip(x,y):
            assert actual==sum(v for a,v in zip(value,weight) if a>threshold)


@pytest.mark.parametrize('order',[1,4,64,200])
def test_eps_shape_support_and_finite_default_low_masses(model,order):
    z=np.array([.25,.5,1.])
    mass=np.broadcast_to(np.logspace(1,11,5),(3,5))
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        result=model.Na_calc(mass,z,1e12,N_herm=order)
    assert result.shape==(3,5)
    assert np.all(np.isfinite(result)) and np.all(result>=0)


def test_redshift_grid_passes_each_evolved_mass(monkeypatch,model):
    observed=[]; original=model.Na_calc
    def spy(ma,z,*args,**kwargs):
        observed.append((np.array(ma),np.array(z),original(ma,z,*args,**kwargs)))
        return observed[-1][2]
    monkeypatch.setattr(model,'Na_calc',spy)
    options=dict(M0=1e10,dz=.5,N_ma=4,logmamin=6.,logmamax=8.,N_herm=2,N_hermNa=3)
    model.rs_rhos_calc(zmax=1.,**options)
    model.rs_rhos_calc(zmax=1.5,**options)
    for ma,z,rate in observed:
        expected=np.stack([model.Mvir_from_M200(np.logspace(6,8,4)*w.Msolar,zi)/w.Msolar for zi in z])
        np.testing.assert_array_equal(ma,expected)
    np.testing.assert_allclose(observed[0][2],observed[1][2][:2],rtol=1e-13)


@pytest.mark.parametrize('particle_mass',[.5,2.,5.])
def test_full_catalog_independent_reference(particle_mass):
    directory=Path(__file__).parent/'validation/maintenance'
    config=json.loads((directory/f'B-q10-{particle_mass}.json').read_text())
    expected=np.load(directory/f'B-q10-{particle_mass}-reference.npz')
    model=w.subhalos(particle_mass)
    actual=model.rs_rhos_calc(**config['parameters'],method='odeint')
    for i,value in enumerate(actual):
        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(expected[f'tuple_{i}']))
        assert value.shape==expected[f'tuple_{i}'].shape
        if i==9:
            np.testing.assert_array_equal(value,expected[f'tuple_{i}'])
        else:
            np.testing.assert_allclose(value,expected[f'tuple_{i}'],rtol=2e-8,atol=0,err_msg=f'column {i}')


def test_grid_entirely_below_wdm_cutoff_has_zero_weight():
    model=w.subhalos(.5)
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        catalog=model.rs_rhos_calc(M0=1e12,zmax=1.,dz=.5,N_ma=4,
            logmamin=6.,logmamax=8.,N_herm=2,N_hermNa=3)
    assert all(np.all(np.isfinite(x)) for x in catalog)
    assert np.all(catalog[8]==0)
