import pytest
from autograd import grad, jacobian
from autograd.scipy.special import (
    gammainc as scipy_gammainc,
    gammaincc as scipy_gammaincc,
    gamma,
)
from autograd_gamma import gammainc, gammaincc, betainc
import numpy as np
import numpy.testing as npt
from scipy.special import expi


def test_inc_gamma_second_argument():
    for a in np.logspace(-5, 2):
        for x in np.logspace(-5, 2):
            npt.assert_allclose(
                grad(gammainc, argnum=1)(a, x), grad(scipy_gammainc, argnum=1)(a, x)
            )
            npt.assert_allclose(
                grad(gammaincc, argnum=1)(a, x), grad(scipy_gammaincc, argnum=1)(a, x)
            )


def test_a_special_case_of_the_derivative():
    """
    We know a specific to test against:

    dUIG(s, x) / ds at (s=1, x) = ln(x) * UIG(1, x) + E_1(x)

    where E_1(x) is the exponential integral
    """

    # incomplete upper gamma
    IUG = lambda s, x: gammaincc(s, x) * gamma(s)

    def analytical_derivative(x):
        dIUG = np.log(x) * IUG(1.0, x) - expi(-x)
        return dIUG

    def approx_derivative(x):
        return jacobian(IUG, argnum=0)(1.0, x)

    x = np.linspace(1, 12)
    npt.assert_allclose(analytical_derivative(x), approx_derivative(x))

    x = np.logspace(-25, 25, 100)
    npt.assert_allclose(analytical_derivative(x), approx_derivative(x))


def test_large_x():
    npt.assert_allclose(grad(gammainc, argnum=1)(100.0, 10000.0), 0)
    npt.assert_allclose(grad(gammaincc, argnum=1)(100.0, 10000.0), 0)


def test_betainc_to_known_values():

    d_beta_inc_0 = jacobian(betainc, argnum=0)

    # I_0(a, b) = 0
    npt.assert_allclose(d_beta_inc_0(0.3, 1.0, 0), 0)

    # I_1(a, b) = 1
    npt.assert_allclose(d_beta_inc_0(0.3, 1.0, 1), 0)

    # I_x(a, 1) = x ** a
    x = np.linspace(0, 1)
    a = np.linspace(0.1, 10)
    npt.assert_allclose(d_beta_inc_0(a, 1.0, x), jacobian(lambda a, x: x ** a)(a, x))

    d_beta_inc_1 = jacobian(betainc, argnum=1)

    # I_0(a, b) = 0
    npt.assert_allclose(d_beta_inc_1(0.3, 1.0, 0), 0)

    # I_1(a, b) = 1
    npt.assert_allclose(d_beta_inc_1(0.3, 1.0, 1), 0)

    x = np.linspace(0.001, 0.5)
    b = np.linspace(0.1, 2.0)
    # I_x(1, b) = 1 - (1-x) ** b
    npt.assert_allclose(
        d_beta_inc_1(1.0, b, x), jacobian(lambda b, x: 1 - (1 - x) ** b)(b, x)
    )
