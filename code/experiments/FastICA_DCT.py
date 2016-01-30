# uniqueness noisy DCT test
# using FastICA

import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from utils import sample_from_bernoulli

DCT = np.load('../../data/DCT.npz')['DCT']  # cols are DCT images
A = DCT

M = 640; K = 4; N = 64
noises = np.linspace(0, 10)
num_trials = 1

rel_error = np.zeros((len(noises), num_trials))
for n, noise in enumerate(noises):
    print "Noise: %1.4f" % noise
    for trial in xrange(num_trials):
        a = sample_from_bernoulli(N * [1. * K / N], M).T  # sparse codes
        # a = np.vstack([a, np.identity(N)])
        C = np.random.randn(M, N); D = np.random.randn(M, N)  # coefficients / noise term
        X = np.dot(a * C, A.T) + noise * D
        # tack on ei's to figure out mapping
        X = np.vstack([X, A.T])

        ica = FastICA(max_iter=100000, tol=0.000001, algorithm='deflation')
        b = ica.fit_transform(X)  # Reconstruct signals
        B = ica.mixing_  # Get estimated mixing matrix

        PD = b[-64:,:].T

        Ahat = np.dot(B, PD)

        rel_error[n, trial] = np.linalg.norm(A - Ahat, axis=0, ord=inf).max()  # max inf norm over cols
    print "Dictionary Reconstruction Error: %1.3f" % rel_error[n, :].mean()

