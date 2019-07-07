import pytest
from autograd import grad, jacobian
from autograd.scipy.special import gammainc as scipy_gammainc, gammaincc as scipy_gammaincc, gamma
from autograd_gamma import gammainc, gammaincc
import numpy as np
import numpy.testing as npt
from scipy.special import expi


def test_inc_gamma_second_argument():
    for a in np.logspace(-5, 2):
        for x in np.logspace(-5, 2):
            npt.assert_allclose(grad(gammainc, argnum=1)(a, x), grad(scipy_gammainc, argnum=1)(a, x))
            npt.assert_allclose(grad(gammaincc, argnum=1)(a, x), grad(scipy_gammaincc, argnum=1)(a, x))


def test_a_special_case_of_the_derivative():
    """
    We know a specific to test against:

    dUIG(s, x) / ds at (s=1, x) = ln(x) * UIG(1, x) + E_1(x)

    where E_1(x) is the exponential integral
    """

    # incomplete upper gamma
    IUG = lambda s, x: gammaincc(s, x) * gamma(s)


    def analytical_derivative(x):
        dIUG = np.log(x) * IUG(1., x) - expi(-x)
        return dIUG

    def approx_derivative(x):
        return jacobian(IUG, argnum=0)(1., x)


    x = np.linspace(1, 12)
    npt.assert_allclose(analytical_derivative(x), approx_derivative(x))

    x = np.logspace(-25, 25, 100)
    npt.assert_allclose(analytical_derivative(x), approx_derivative(x))


def test_large_x():
    npt.assert_allclose(grad(gammainc, argnum=1)(100., 10000.), 0)
    npt.assert_allclose(grad(gammaincc, argnum=1)(100., 10000.), 0)
