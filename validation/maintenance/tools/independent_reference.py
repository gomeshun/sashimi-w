"""Independent segmented log-mass DOP853 reference.

The Correa concentration fit changes branch at z=4. W's unchanged nearest
mass concentration table jumps at arithmetic midpoints of its mass nodes.
Segment those known coefficient boundaries so tolerance refinement cannot
miss a narrow jump. No Picard helper is imported.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, toms748
from scipy.special import erf
from scipy.interpolate import splev
from functools import lru_cache


def boundaries(solver,za,target):
    points=[float(za),float(target)]
    if not hasattr(solver,'model'):
        if target<4.<za: points.append(4.)
    else:
        import sashimi_w as w
        # Native formation trials are dropped when their radicand reaches
        # zero. The low-concentration interpolation can jump at those points.
        c=10**(np.arange(100)*4./99.)
        fraction=(np.log(2.)-.5)/(np.log1p(c)-c/(1+c))
        ratio=c**3*fraction/(650./200.)
        active=ratio<1.
        transitions=np.cbrt((w.OmegaL/ratio[active]-w.OmegaL)/w.OmegaM)-1.
        points.extend(float(z) for z in transitions if target<z<za)
        table=solver.model.filter_Mass
        mids=(table[1:]+table[:-1])/2
        def mass(z):
            m=solver.model.Mzzi(solver.M0,z,0.)
            if solver.N_hermNa==1 and solver.sigmafac!=0:
                m1=solver.model.Mzzi(solver.M0,1.,0.)
                scatter=(.12-.15*np.log10(m/solver.M0)) if z>1 else (.12-.15*np.log10(m1/solver.M0))/np.log10(m1/solver.M0)*np.log10(m/solver.M0)
                m=10**(np.log10(m)+solver.sigmafac*scatter)
                if solver.sigmafac>0: m=min(m,solver.M0)
            return m*w.h
        low,high=sorted([mass(za),mass(target)])
        for mid in mids[(mids>low)&(mids<high)]:
            points.append(brentq(lambda z:mass(z)-mid,target,za,xtol=1e-14))
        # np.interp's legacy argument table need not be monotonic near the
        # future-formation branch. Any search-branch jump occurs at arg_i=0.
        # Locate all such roots on each fixed mass-bin / valid-trial interval.
        solver.model.conc200(solver.M0*w.Msolar,target)
        coarse=sorted(set(points))
        c=10**(np.arange(100)*4./99.)
        fraction=(np.log(2.)-.5)/(np.log(1.+c)-c/(1.+c))
        ratio=c**3*fraction/(650./200.)
        for lower,upper in zip(coarse[:-1],coarse[1:]):
            margin=min((upper-lower)*1e-6,1e-11)
            if upper-lower<1e-12:continue
            mid=(upper+lower)/2
            index=np.argmin(abs(table-mass(mid)))
            gap=splev(solver.model.logM0[index]-10.+np.log10(.02),solver.model._concentration_spline)-solver.model._concentration_variance[index]
            valid=(ratio*(w.OmegaM*(1+mid)**3+w.OmegaL)-w.OmegaL)>0
            def arguments(z):
                z=np.atleast_1d(z)
                rad=(ratio[valid][None,:]*(w.OmegaM*(1+z[:,None])**3+w.OmegaL)-w.OmegaL)/w.OmegaM
                formation=np.maximum(rad,np.finfo(float).tiny)**.3333-1.
                growth=solver.model.growthD(formation)
                delta=1.686/growth-1.686/solver.model.growthD(z[:,None])
                return (650./200.)*ratio[valid][None,:]/c[valid][None,:]**3-(1.-erf(delta/np.sqrt(2*gap)))
            samples=np.linspace(lower+margin,upper-margin,65)
            values=arguments(samples)
            rows,columns=np.nonzero(values[:-1]*values[1:]<0)
            for i,j in zip(rows,columns):
                root=brentq(lambda z:arguments(z)[0,j],samples[i],samples[i+1],xtol=1e-13)
                points.append(root)
    return sorted(set(points),reverse=True)



def warm_coefficients(solver):
    """Independent, dimensionless density root; avoid scalar fsolve noise."""
    import sashimi_w as w
    @lru_cache(maxsize=16384)
    def evaluate(z):
        model=solver.model
        m200=model.Mzzi(solver.M0,z,0.)
        if solver.N_hermNa==1 and solver.sigmafac!=0:
            m1=model.Mzzi(solver.M0,1.,0.)
            scatter=(.12-.15*np.log10(m200/solver.M0)) if z>1 else (.12-.15*np.log10(m1/solver.M0))/np.log10(m1/solver.M0)*np.log10(m200/solver.M0)
            m200=10**(np.log10(m200)+solver.sigmafac*scatter)
            if solver.sigmafac>0:m200=min(m200,solver.M0)
        c=model.conc200(m200*w.Msolar,z)
        fc=lambda r: np.log1p(r)-r/(1+r)
        overdensity=model.Delc(model.Omegaz(w.pOmega,z)-1.)
        goal=overdensity/200*fc(c)/c**3
        radius=toms748(lambda r:fc(r)/r**3-goal,c/10,c*10,xtol=1e-13,rtol=1e-13)
        host=m200*fc(radius)/fc(c)
        loghost=np.log10(host)
        rate=10**((-.0019*loghost+.045)*z+(.0097*loghost-.313))
        dyn=1.628/w.h*(overdensity/178)**-.5/(model.Hz(z)/w.H0)*(86400*365*1e9)
        return rate/dyn/model.Hz(z)/(1+z),(-5.55e-5*loghost+1.43e-3)*z+(3.34e-4*loghost-8.11e-3),np.log(host)
    return evaluate

def reference(solver,mass,za,times,refined=False,*,max_step=None):
    times=np.asarray(times,dtype=float)
    mass=np.atleast_1d(mass)
    if times[-1]==za:
        return np.broadcast_to(mass,(len(times),len(mass))).copy()
    edges=boundaries(solver,za,float(times[-1]))
    warm=warm_coefficients(solver) if hasattr(solver,"model") else None
    log_initial=np.log(mass)
    # Integrate delta ln(m), so relative tolerances cannot depend on mass units.
    state=np.zeros_like(mass)
    out=np.empty((len(times),len(mass)))
    out[times==za]=log_initial
    for upper,lower in zip(edges[:-1],edges[1:]):
        def rhs(z,y):
            # Evaluate the appropriate one-sided value at a coefficient jump.
            z=np.clip(z,np.nextafter(lower,upper),np.nextafter(upper,lower))
            if warm is not None:
                phi,zeta,loghost=warm(float(z))
                return phi*np.exp(zeta*(y+log_initial-loghost))
            return solver.Phi(z)*np.exp(solver.zetaMz(z)*(y+log_initial-np.log(solver.Mzvir(z))))
        result=solve_ivp(rhs,(upper,lower),state,dense_output=True,
                         method='DOP853',rtol=3e-13 if refined else 1e-11,
                         atol=3e-14 if refined else 1e-12,
                         max_step=max_step if max_step is not None else (.05 if hasattr(solver,'model') else .2))
        if not result.success or np.any(~np.isfinite(result.y)):
            raise RuntimeError(result.message)
        selected=(times<upper)&(times>=lower)
        if np.any(selected):
            out[selected]=result.sol(times[selected]).T+log_initial
        state=result.y[:,-1]
    return np.exp(out)
