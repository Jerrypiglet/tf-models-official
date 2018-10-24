import numpy as np
import numpy.random as nr
import numpy.linalg as linalg
from scipy import constants
import pdb


def mvnlogpdf(X, mu, Sigma):
    """
    Multivariate Normal Log PDF

    Args:
        X      : NxD matrix of input data. Each ROW is a single sample.
        mu     : Dx1 vector for the mean.
        PrecMat: DxD precision matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    PrecMat = linalg.inv(Sigma)
    D = PrecMat.shape[0]
    N = len(X)

    _, neglogdet = linalg.slogdet(PrecMat)
    normconst = -0.5 * (D * np.log(2 * constants.pi) - neglogdet)

    logpdf = np.zeros((N, 1))
    for n, x in enumerate(X):
        d = x[:, None] - mu
        logpdf[n] = normconst - 0.5 * d.transpose().dot(PrecMat.dot(d))

    return logpdf



def uniform_sample(ranges):
    dim = len(ranges)
    action = np.zeros((1, dim), dtype=np.float32)
    for i, ran in enumerate(ranges):
        action[0, i] = np.random.uniform(ran[0], ran[1])

    return action


class OUNoise(object):
    """use multiple gaussian noisy for action noise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.1):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self, sigma=None):
        if sigma is None:
            sigma = self.sigma

        x = self.state
        dx = self.theta * (self.mu - x) + nr.randn(len(x)) * self.sigma
        self.state = x + dx
        return self.state[None, :]

    def noise_multi(self, sigma=None, num=1, activate=None):
        if activate is None:
            activate = np.ones(self.action_dimension)

        if (sigma is None) or isinstance(sigma, float):
            sigma = np.ones(self.action_dimension) * sigma

        dx = nr.randn(num, self.action_dimension) * sigma * activate

        return dx





