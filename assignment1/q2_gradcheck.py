#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at

    confirmed it's a bit different from eval_numerical_gradient on
    http://cs231n.github.io/optimization-1/

    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)             # Evaluate function value at original point
    h = 1e-4                    # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.
        old_x = x[ix]
        x[ix] = old_x + h       # increment by h
        random.setstate(rndstate)
        fxph, _ = f(x)             # evalute f(x + h)
        x[ix] = old_x - h
        random.setstate(rndstate)
        fxmh, _ = f(x)             # evaluate f(x - h)
        numgrad = (fxph - fxmh) / (2 * h)  # the slope
        x[ix] = old_x

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        # print('numerical gradient: {0}; reldiff: {1}'.format(numgrad, reldiff))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index {0}".format(str(ix)))
            print("Your gradient: {0} \t Numerical gradient: {1}".format(
                grad[ix], numgrad))
            return
        it.iternext()           # Step to next dimension

    print("Gradient check passed!")


def quad(x):
    return (np.sum(x ** 2), x * 2)


def sanity_check():
    """
    Some basic sanity checks.
    """

    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


def test_gradcheck_naive():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    sanity_check()


if __name__ == "__main__":
    test_gradcheck_naive()
