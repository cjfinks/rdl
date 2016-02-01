"""
Plotting reconstruction error vs noise for synthetic data
- Let A = DCT basis (j-th column is j-th orthonormnal 8x8 DCT basis vector)
- Pick N vectors a in 64 dimensions as follows:
- Set k to 1, 2, 4, ...
- For each vector a, set k nonzero coefficients in a to be in [0,1]
- Let y = Aa + noise with noise in [-epsilon, epsilon]
- Do FastICA to recover y = Bb
- Find PD as the transform of A.T
- Compute ||A - BPD|| as function of epsilon
"""

import numpy as np
import matplotlib.pyplot as pp
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA, DictionaryLearning
from utils import sample_from_bernoulli, corrupt_binary_data
import os
import scipy.io

TEST = False
PLOT = True
COMPUTE = True

""" Set generating dictionary """
A = scipy.io.loadmat('../../data/DCT_8x8.npz')['DCT_8x8']  # cols are rastered DCT8x8
#A = np.load('../../data/DCT_12x12.npy')['DCT_12x12']  # cols are rastered DCT12x12
#A = np.random.randn(64, 64) # Random basis
#A = np.dot(np.random.randn(64, 64), np.dot(np.random.randn(64, 2), np.random.randn(2, 64))) # Bad basis
A = normalize(A, axis = 0, norm = 'l2')
n, m = A.shape

if TEST:
	""" Learn the canonical basis vectors """
	ks = [1]; 
	A = np.eye(64,64)
	epsilons = [0.01]
	num_trials = 10
	coeff_ivl = [0.9, 1]
	N = 2 * m
else:
	N = 10 * m; ks = [4]
	epsilons = np.linspace(.0001, 0.5, 20)
	num_trials = 20
	coeff_ivl = [0.5,1]

""" Initialize directory for saving results and figures """
param_string = 'k%dto%d_N%d_trials%d/' % (min(ks), max(ks), N, num_trials)
save_dir = os.path.abspath( os.path.dirname(__file__) ) + '/figs_ica'
if not os.path.isdir(save_dir): os.makedirs(save_dir)
save_dir = save_dir + '/' + param_string 
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'imgs/'): os.makedirs(save_dir + 'imgs/')

if COMPUTE:
    error_mse = np.zeros((len(ks), len(epsilons), num_trials))
    error_mat1norm = np.zeros((len(ks), len(epsilons), num_trials))

    for kID, k in enumerate(ks):
    	print 'k = %d' % k
        for epsilonID, epsilon in enumerate(epsilons):
            print "noise: %1.4f" % epsilon
            for trial in xrange(num_trials):
                support = corrupt_binary_data( np.zeros((N,m)), bits_corrupted=k ) # sparse support indicator
                coeffs = coeff_ivl[0] + np.diff(coeff_ivl) *  np.random.rand(N, m) # coefs uniformly distributed in coeff_ivl
                a = support * coeffs # k-sparse codes with values in coeff_ivl
                noise = epsilon * (1 - 2 * np.random.rand(N, n))  # noise in [-epsilon, epsilon]
                Y = np.dot(support * coeffs, A.T) + noise
                ica = FastICA(max_iter=1000, tol=0.0001, algorithm='parallel')  # maybe params could be more optimial
                ica.fit(Y)  # fit the ICA model to infer mixing matrix
                B = ica.mixing_  # Get estimated mixing matrix
                PD = ica.transform(A.T).T # transform canonical basis vectors to get PD
                PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # Map to 'closest' permutation matrix
                if not np.all( np.sum(PD, axis=0)  == 1 ): # check that this method worked
                	print '(k = %d, epsilon = %1.4f, trial = %d) Estimated P not a permutation matrix. Need a better method to find closest PD.' % (k, epsilon, trial)
                Ahat = np.dot(B, PD)
                err = A - Ahat
                error_mse[kID, epsilonID, trial] = (err ** 2).mean() / (A ** 2).mean() 
                error_mat1norm[kID, epsilonID, trial] = np.max([np.abs(err[:,i]).sum() for i in xrange(n)]) / np.max([np.abs(A[:,i]).sum() for i in xrange(n)])
            print "Dictionary MSE/matrix-1-norm reconstruction error (k=%d): %1.3f/%1.3f" % (k, error_mse[kID, epsilonID, :].mean(), error_mat1norm[kID, epsilonID, :].mean())
            if PLOT:
				""" Plot noiseless vs noisy first sample """
				fig, axes = pp.subplots(1,2); pp.suptitle('Datum (k = %d)' % k)
				Ynoiseless = np.dot( a, A.T )
				pp.sca(axes[0]); pp.title('Noiseless')
				pp.imshow( Ynoiseless[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=Y[0].min(), vmax=Y[0].max())
				pp.sca(axes[1]); pp.title('Noisy (epsilon = %1.4f)' % epsilon)
				pp.imshow(Y[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=Y[0].min(), vmax=Y[0].max())
				pp.colorbar()
				pp.savefig(save_dir + 'imgs/sample_k%d_epsilon%1.4f.png' % (k, epsilon), bbox_inches='tight', pad_inches=.05)

				""" Plot B, PD, Ahat and A for the last trial """
				titles = ['B', 'PD', 'Ahat', 'A']
				matrices = [B, PD, Ahat, A]
				fig, axes = pp.subplots(1,len(matrices)); pp.suptitle('Original and recovered dictionaries (k = %d, eps = %1.4f)' % (k, epsilon))
				for i, M in enumerate( matrices ):
					pp.sca(axes[i]); pp.title(titles[i])
					pp.imshow(M, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
					pp.savefig(save_dir + 'imgs/B_PD_Ahat_A_k%d_epsilon%1.4f.png' % (k, epsilon), bbox_inches='tight', pad_inches=.05)
				pp.colorbar()
    np.savez(save_dir + 'error', error_mse=error_mse, error_mat1norm=error_mat1norm)

if PLOT:
	error = np.load(save_dir + 'error.npz')['error_mse']
	error_mat1norm = np.load(save_dir + 'error.npz')['error_mat1norm']
	
	""" Plot A """
	pp.figure()
	pp.clf()
	pp.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
	pp.title('Basis (as columns)')
	pp.colorbar()
	pp.savefig(save_dir + 'Basis.png', bbox_inches='tight', pad_inches=.05)
	
	""" Plot MSE reconstruction error as a function of epsilon for each k """
	fig, axes = pp.subplots(1,2)
	pp.sca(axes[0])
	for kID, k in enumerate(ks):
		pp.errorbar(epsilons, error[kID].mean(axis=1), yerr=error[kID].std(axis=1), linewidth=2, label='k=%d' % k)
	pp.title('MSE of Ahat vs A relative to sample epsilon (trials=%d; sd errors)' % num_trials)
	pp.xlabel('Noise in input')
	pp.ylabel('Dictionary recovery error (mean square error / mean square of A)')
	pp.legend(frameon=False, loc='best')
	pp.savefig(save_dir + 'error_mse.pdf', bbox_inches='tight', pad_inches=.05)
	
	""" Plot matrix 1-norm reconstruction error as a function of epsilon for each k """
	pp.sca(axes[1])
	for kID, k in enumerate(ks):
		pp.errorbar(epsilons, error_mat1norm[kID].mean(axis=1), yerr=error_mat1norm[kID].std(axis=1), linewidth=2, label='k=%d' % k)
	pp.title('||A-Ahat||_1 relative to sample epsilon (trials=%d; sd errors)' % num_trials)
	pp.xlabel('Noise in input')
	pp.ylabel('Recovery error (max abs col sum A-Ahat / max abs col sum A)')
	pp.legend(frameon=False, loc='best')
	pp.savefig(save_dir + 'error_mat1norm.pdf', bbox_inches='tight', pad_inches=.05)
