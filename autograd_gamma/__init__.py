from autograd.extend import primitive, defvjp
from autograd import numpy as np
from autograd.scipy.special import gamma
from autograd.numpy.numpy_vjps import unbroadcast_f
from scipy.special import gammainc as _scipy_gammainc, gammaincc as _scipy_gammaincc

__all__ = [
    'gammainc', # regularized lower incomplete gamma function
    'gammaincc', # regularized upper incomplete gamma function
    'gamma', # gamma function
    'reg_lower_inc_gamma',
    'reg_upper_inc_gamma'
]


DELTA = 1e-7


gammainc = primitive(_scipy_gammainc)
gammaincc = primitive(_scipy_gammaincc)


defvjp(
    gammainc,
    lambda ans, a, x: unbroadcast_f(
        a,
        lambda g: g
        * (
            -gammainc(a + 2 * DELTA, x)
            + 8 * gammainc(a + DELTA, x)
            - 8 * gammainc(a - DELTA, x)
            + gammainc(a - 2 * DELTA, x)
        )
        / (12 * DELTA),
    ),
    lambda ans, a, x: unbroadcast_f(x, lambda g: g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)


defvjp(
    gammaincc,
    lambda ans, a, x: unbroadcast_f(
        a,
        lambda g: g
        * (
            -gammaincc(a + 2 * DELTA, x)
            + 8 * gammaincc(a + DELTA, x)
            - 8 * gammaincc(a - DELTA, x)
            + gammaincc(a - 2 * DELTA, x)
        )
        / (12 * DELTA),
    ),
    lambda ans, a, x: unbroadcast_f(x, lambda g: -g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)


reg_lower_inc_gamma = gammainc
reg_upper_inc_gamma = gammaincc
