# Experiment 1:  Synthetic Data
# set K = 4, 5, ...?
# let A = DCT matrix (j-th column is j-th orthonormnal 8x8 DCT basis vector)
# pick 640 points in 64 dimensions as follows
# choose K nonzero coefficients in a (of the 64) ~ N(0,1) and make linear combination
# x = Aa + noise * N(0,1); 0 <= noise <= MAX_NOISE
# do fastICA to recover x = Bb; find P,D, compute ||A - BPD|| as function of noise

import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from utils import sample_from_bernoulli
import os


DCT = np.load('../../data/DCT_8x8.npz')['DCT']  # cols are DCT images
A = DCT

M = 640; N = 64; Ks = [1, 2, 4, 8, 16, 32]
noises = np.linspace(0, 10)[:10]
num_trials = 2

#save_dir = 'figs/pFastICA_M%d_K%d_Trials%d/' % (M, K, num_trials)
save_dir = 'figs/tst_pFastICA_M%d_Ks%d_Trials%d/' % (M, len(Ks), num_trials)
#save_dir = 'figs/FastICA_M%d_K%d_Trials%d/' % (M, K, num_trials)
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'imgs/'): os.makedirs(save_dir + 'imgs/')

plt.figure()
plt.clf()
plt.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
plt.title('DCT Basis (as columns)')
plt.colorbar()
plt.savefig(save_dir + 'DCT.png', bbox_inches='tight', pad_inches=.05)

rel_error = np.zeros((len(Ks), len(noises), num_trials))
for k, K in enumerate(Ks):
    for n, noise in enumerate(noises):
        print "Noise: %1.4f" % noise
        for trial in xrange(num_trials):
            a = sample_from_bernoulli(N * [1. * K / N], M).T  # sparse codes
            # a = np.vstack([a, np.identity(N)])
            C = np.sign(np.random.randn(M, N)) + np.random.randn(M, N); D = np.random.randn(M, N)  # coefficients / noise term
            X = np.dot(a * C, A.T) + noise * D
            # tack on ei's to figure out mapping
            X = np.vstack([X, A.T])

            ica = FastICA(max_iter=10000, tol=0.0001, algorithm='parallel')  # maybe params could be more optimial
            b = ica.fit_transform(X)  # Reconstruct signals
            B = ica.mixing_  # Get estimated mixing matrix
            PD = b[-64:,:].T
            Ahat = np.dot(B, PD)
            diffm = A - Ahat
            rel_error[k, n, trial] = (diffm ** 2).mean()  # np.max([np.abs(diffm[:,i]).sum() for i in xrange(64)])
            # np.linalg.norm(A - Ahat, axis=0, ord='fro').max()  # max inf norm over cols
        print "Dictionary Reconstruction Error (K=%d): %1.3f" % (K, rel_error[k, n, :].mean())
        plt.figure()
        plt.clf()
        plt.imshow(Ahat, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
        plt.title('Recovery (as columns)')
        plt.colorbar()
        plt.savefig(save_dir + 'imgs/Ahat_FastICA_M%d_K%d_noise%1.4f.png' % (M, K, noise), bbox_inches='tight', pad_inches=.05)

        plt.figure()
        plt.clf()
        plt.imshow(PD, interpolation='nearest')
        plt.title('PD Recovery (Ahat = BPD)')
        plt.colorbar()
        plt.savefig(save_dir + 'imgs/PD_recovery_FastICA_M%d_K%d_noise%1.4f.png' % (M, K, noise), bbox_inches='tight', pad_inches=.05)

        plt.figure()
        plt.clf()
        plt.imshow(B, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
        plt.title('Learned dictionary B (y = Bb)')
        plt.colorbar()
        plt.savefig(save_dir + 'imgs/B_FastICA_M%d_K%d_noise%1.4f.png' % (M, K, noise), bbox_inches='tight', pad_inches=.05)
        plt.close('all')


np.save(save_dir + 'error_table_M%d_Trials%d_Ks%d' % (M, num_trials, len(Ks)), rel_error)

plt.figure()
#plt.plot(noises, rel_error.mean(axis=1), linewidth=2)
for k, K in enumerate(Ks):
    plt.errorbar(noises, rel_error[k].mean(axis=1), yerr=rel_error[k].std(axis=1), linewidth=2, label='K=%d' % K)

plt.title('MSE of Ahat vs A relative to noise in samples (trials=%d; sd errors)' % num_trials)
#plt.title('||A-Ahat||_1 relative to noise in samples (trials=%d; sd errors)' % num_trials)
plt.xlabel('Noise in input')
plt.ylabel('Dictionary recovery error (max absolute sum)')
plt.legend(frameon=False, loc='best')
plt.savefig(save_dir + 'error_FastICA_M%d_K%d.pdf' % (M, K), bbox_inches='tight', pad_inches=.05)
