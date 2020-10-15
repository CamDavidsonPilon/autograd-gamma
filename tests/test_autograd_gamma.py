import pytest
from autograd import grad, jacobian
from autograd.scipy.special import (
    gammainc as scipy_gammainc,
    gammaincc as scipy_gammaincc,
    gamma,
)
from autograd_gamma import gammainc, gammaincc, betainc, gammaincln, gammaincinv, gammainccinv, betaincinv
import numpy as np
import numpy.testing as npt
from scipy.special import expi
from scipy.optimize import check_grad


def test_inc_gamma_second_argument():
    for a in np.logspace(-5, 2):
        for x in np.logspace(-5, 2):
            npt.assert_allclose(
                grad(gammainc, argnum=1)(a, x), grad(scipy_gammainc, argnum=1)(a, x)
            )
            npt.assert_allclose(
                grad(gammaincc, argnum=1)(a, x), grad(scipy_gammaincc, argnum=1)(a, x)
            )

def test_log_gamma():
    gammaincln(1., 1.)
    grad(gammaincln)(1., 1.)
    grad(gammaincln, argnum=1)(1., 1.)


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



def test_gammainc():

    for a in np.logspace(-0.1, 2, 10):
        gammainc_1 = lambda x: gammainc(a, x)
        gammainc_2 = lambda x: grad(gammainc, argnum=1)(a, x)

        assert check_grad(gammainc_1, gammainc_2, 1e-4) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e-3) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e-2) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e-1) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e-0) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e1) < 0.0001
        assert check_grad(gammainc_1, gammainc_2, 1e2) < 0.0001

def test_gammaincinv():
    for a in np.logspace(-1, 1, 10):
        for y in np.linspace(0.001, 0.99, 10):
            gammaincinv_1 = lambda x: gammaincinv(a, x)
            gammaincinv_2 = lambda x: grad(gammaincinv, argnum=1)(a, x)
            assert check_grad(gammaincinv_1, gammaincinv_2, y) < 0.01, (a, y)

def test_gammainccinv():
    for a in np.logspace(-1, 1, 10):
        for y in np.linspace(0.01, 0.99, 10):
            gammainccinv_1 = lambda x: gammainccinv(a, x)
            gammainccinv_2 = lambda x: grad(gammainccinv, argnum=1)(a, x)
            assert check_grad(gammainccinv_1, gammainccinv_2, y) < 0.0005, (a, y)

def test_betaincinv():
    for a in np.logspace(-1, 2, 10):
        for b in np.logspace(-1, 2, 10):
            for y in np.linspace(0.01, 0.99, 10):
                betaincinv_1 = lambda x: betaincinv(a, b, x)
                betaincinv_2 = lambda x: grad(betaincinv, argnum=2)(a, b, x)
                assert check_grad(betaincinv_1, betaincinv_2, y) < 0.0005, (a, b, y)



@pytest.mark.xfail
def test_gammainc_fails():
    a = 0.1
    gammainc_1 = lambda x: gammainc(a, x)
    gammainc_2 = lambda x: grad(gammainc, argnum=1)(a, x)
    assert not check_grad(gammainc_1, gammainc_2, 1e-4) < 0.0001

