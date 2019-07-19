# autograd-gamma
[![PyPI version](https://badge.fury.io/py/autograd-gamma.svg)](https://badge.fury.io/py/autograd-gamma)


autograd compatible approximations to the derivatives of the Gamma-family of functions.


# Tutorial

```python
from autograd import grad
from autograd_gamma import gammainc, gammaincc, gamma, gammaincln, gammainccln


grad(gammainc, argnum=0)(1., 2.)
grad(gammaincc, argnum=0)(1., 2.)

# logarithmic functions too.
grad(gammaincln, argnum=0)(1., 2.)
grad(gammainccln, argnum=0)(1., 2.)



from autograd import grad
from autograd_gamma import betainc, betaincln

grad(betainc, argnum=0)(1., 2., 0.5)
grad(betainc, argnum=1)(1., 2., 0.5)

# logarithmic functions too.
grad(betaincln, argnum=0)(1., 2., 0.5)
grad(betaincln, argnum=1)(1., 2., 0.5)

```


# Long-term goal

Build and improve upon the derivative of the upper and lower incomplete gamma functions. Eventually, if we have a fast analytical solution, we will merge into the autograd library.