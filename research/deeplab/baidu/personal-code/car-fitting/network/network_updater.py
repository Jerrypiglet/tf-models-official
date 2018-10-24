
import math
from mxnet.ndarray import square
from mxnet.ndarray import sqrt

def adam_updater(weight, grad, state,
        lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=0):

    mean, variance = state

    mean *= beta1
    mean += grad * (1. - beta1)

    variance *= beta2
    variance += (1 - beta2) * square(grad)

    coef1 = 1. - beta1
    coef2 = 1. - beta2
    lr *= math.sqrt(coef2) / coef1

    weight -= lr * mean / (sqrt(variance) + epsilon)
    if wd > 0.:
        weight[:] -= (lr * wd) * weight
