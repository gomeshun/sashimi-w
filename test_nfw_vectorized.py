"""Independent precision audit of the vectorized NFW inverse."""
from pathlib import Path
import importlib
import mpmath as mp
import numpy as np


def test_vectorized_inverse_spans_radii_and_the_disruption_threshold():
    variant=next(v for v in ('c','si','w','f') if Path(__file__).with_name('sashimi_'+v+'.py').exists())
    numerics=importlib.import_module('sashimi_'+variant+'_numerics')
    radii=np.r_[np.logspace(-100,100,61),.77-1e-10,.77+1e-10,0.]
    with mp.workdps(250):
        values=np.array([float(mp.log1p(mp.mpf(float(x)))-mp.mpf(float(x))/(1+mp.mpf(float(x)))) for x in radii])
    solved=numerics.invert_nfw_mass_function(values.reshape(8,8))
    np.testing.assert_allclose(solved.ravel(),radii,rtol=2e-11,atol=0)
    np.testing.assert_array_equal(solved.ravel()[-3:-1]>.77,radii[-3:-1]>.77)
    assert numerics.invert_nfw_mass_function([]).shape==(0,)
