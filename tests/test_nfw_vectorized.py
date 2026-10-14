"""Independent precision audit of the vectorized NFW inverse."""
from pathlib import Path
import importlib
import mpmath as mp
import numpy as np


def test_vectorized_inverse_spans_radii_and_the_disruption_threshold():
    r"""Notes
    -----
    The literature threshold rt/rs > 0.77 follows the Hayashi et al. disruption prescription
    adopted by Hiroshima et al. A caller-supplied threshold takes precedence in variants
    that expose it (C and SI currently default to 0). This is a model selection rule; an
    exact-boundary test does not calibrate disruption physics.

    References
    ----------

    .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
        https://arxiv.org/abs/astro-ph/0203004

    .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
        https://arxiv.org/abs/1803.07691
    """
    variant=next(v for v in ('c','si','w','f') if (Path(__file__).resolve().parents[1]/('sashimi_'+v+'.py')).exists())
    numerics=importlib.import_module('sashimi_'+variant+'_numerics')
    radii=np.r_[np.logspace(-100,100,61),.77-1e-10,.77+1e-10,0.]
    with mp.workdps(250):
        values=np.array([float(mp.log1p(mp.mpf(float(x)))-mp.mpf(float(x))/(1+mp.mpf(float(x)))) for x in radii])
    solved=numerics.invert_nfw_mass_function(values.reshape(8,8))
    np.testing.assert_allclose(solved.ravel(),radii,rtol=2e-11,atol=0)
    np.testing.assert_array_equal(solved.ravel()[-3:-1]>.77,radii[-3:-1]>.77)
    assert numerics.invert_nfw_mass_function([]).shape==(0,)
