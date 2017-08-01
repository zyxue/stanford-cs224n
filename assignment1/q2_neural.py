#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))  # Dx x H
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))        # 1 x H
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))  # H x Dy
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))      # 1 x Dy

    # forward propagation
    # data.shape:               # M x Dx
    z1 = data.dot(W1) + b1      # M x H
    a1 = sigmoid(z1)            # M x H
    z2 = a1.dot(W2) + b2        # M x Dy
    a2 = softmax(z2)            # M x Dy
    # labels.shape:             # M x Dy
    cost = - np.multiply(labels, np.log(a2)).sum()

    d1 = a2 - labels            # M x Dy
    gradW2 = a1.T.dot(d1)       # H x Dy
    # gradb2 = np.ones(b2.shape) * a2.shape[0]  # 1 x Dy
    gradb2 = d1.sum(axis=0, keepdims=True)                # 1 x Dy
    # dz2 / dh
    d2 = d1.dot(W2.T)                       # M x H
    # dh / dz1
    d3 = np.multiply(d2, sigmoid_grad(a1))  # M x H
    # dz1 / dW1
    gradW1 = data.T.dot(d3)    # Dx x H
    gradb1 = d3.sum(axis=0, keepdims=True)

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(
        lambda params: forward_backward_prop(
            data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    sanity_check()


if __name__ == "__main__":

    your_sanity_checks()
