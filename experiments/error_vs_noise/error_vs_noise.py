"""
Experiment 1:  Synthetic Data
- Let A = DCT basis (j-th column is j-th orthonormnal 8x8 DCT basis vector)
- Pick N vectors a in 64 dimensions as follows:
- Set k to 1, 2, 4, ...
- For each vector a, set k nonzero coefficients in a to be in [0,1]
- Let y = Aa + noise with noise in [-epsilon, epsilon]
- Do FastICA to recover y = Bb
- Find PD as the transform of I 
- Compute ||A - BPD|| as function of epsilon
"""

import numpy as np
import matplotlib.pyplot as pp
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA, DictionaryLearning
from utils import sample_from_bernoulli, corrupt_binary_data
import os

TEST = False
PLOT = True
COMPUTE = True
BASIS = 'DCT' # 'DCT' # 'RAND', 'BAD'
METHOD = 'ICA' # 'ICA', 'SCA'

if BASIS == 'DCT':
	A = np.load('../../data/DCT_8x8.npz')['DCT']  # cols are DCT images
	#A = np.load('../../data/DCT_12x12.npy')  # cols are DCT images
if BASIS == 'RAND':
    A = np.random.randn(64, 64)
if BASIS == 'BAD':
    A = np.dot(np.random.randn(64, 64), np.dot(np.random.randn(64, 2), np.random.randn(2, 64)))
A = normalize(A, axis = 0, norm = 'l2')
n, m = A.shape

N = 5000; ks = [2]
epsilons = np.linspace(.0001, 1, 10)
num_trials = 1

if TEST:
	n = 4; m = 4; ks = [1,2]; N = 1000
	A = np.random.rand(n, m); A = normalize(A,axis=0)
	epsilons = [0.0001]
	num_trials = 1

save_dir = 'figs/' + BASIS + '_p' + METHOD + '_N%d_ks%d_Trials%d/' % (N, len(ks), num_trials)
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'imgs/'): os.makedirs(save_dir + 'imgs/')

pp.figure()
pp.clf()
pp.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
pp.title('Basis (as columns)')
pp.colorbar()
pp.savefig(save_dir + 'Basis.png', bbox_inches='tight', pad_inches=.05)

if COMPUTE:
    rel_error = np.zeros((len(ks), len(epsilons), num_trials))
    rel_error_matnorm = np.zeros((len(ks), len(epsilons), num_trials))

    for kID, k in enumerate(ks):
    	print 'k = %d' % k
        for epsilonID, epsilon in enumerate(epsilons):
            print "Noise: %1.4f" % epsilon
            for trial in xrange(num_trials):
                support = corrupt_binary_data( np.zeros((N,m)), bits_corrupted=k ) #support = sample_from_bernoulli(n * [1. * k / n], N).T  # sparse support indicator functions
                coeffs = 0.5 + 0.5 * np.random.rand(N, m) #np.sign(np.random.randn(N, n)) + np.random.randn(N, n); 
                a = support * coeffs # sparse codes in [0.5, 1]
                noise = epsilon * (1 - 2 * np.random.rand(N, n))  # noise in [-epsilon, epsilon]
                X = np.dot(support * coeffs, A.T) + noise
                if METHOD == 'ICA':
                	ica = FastICA(max_iter=10000, tol=0.0001, algorithm='parallel')  # maybe params could be more optimial
                	ica.fit(X)  # Reconstruct signals
                	B = ica.mixing_  # Get estimated mixing matrix
                	PD = ica.transform(A.T).T # transform canonical basis vectors to get PD
                if METHOD == 'SCA':
					if trial == 0 and epsilonID == 0: # search for best lambda to get k-sparse reconstruction
						alpha_min, alpha_max = 0.0, 1.0
						alpha = (alpha_max - alpha_min) / 2
						GOOD_ALPHA = 0
						while GOOD_ALPHA == 0:
							print 'Trying alpha = %1.4f' % alpha
							sca = DictionaryLearning( n_components = m, alpha = alpha, max_iter = 10000,
									                  transform_alpha = alpha, tol = epsilon)
							sca.fit(X)  # Reconstruct signals
							B = sca.components_.T  # Get estimated mixing matrix
							ntest = 100
							b = sca.transform(X[:100])
							Xhat = np.dot(b, B.T)
							err = np.mean( np.sqrt( np.sum( (Xhat-X[:ntest])**2, axis=1 ) ) )
							khat = np.sum( abs( b ) > 0.5 - epsilon, axis = 1)
							avgk = np.mean(khat); maxk = np.max(khat); mediank = np.median(khat)
							if mediank == k and abs( avgk - k ) < 0.1: 
								GOOD_ALPHA = 1
							elif mediank > k+1:
								alpha_min = alpha
								alpha = alpha +  (alpha_max - alpha) / 2
							elif mediank < k+1:
								alpha_max = alpha
								alpha = alpha - (alpha - alpha_min) / 2
							print 'mean ||Xhat - X|| = %1.2f, mean k = %1.2f, median k = %1.2f, max k = %1.2f' % (err, avgk, mediank, maxk)
					else:
						sca.fit(X) 
					B = sca.components_.T  # Get estimated mixing matrix
					PD = sca.transform(A.T).T
                PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # keep only the largest indices in each column
                #PD *= np.sign( np.diag(np.dot(A.T, np.dot(B, PD) )) )[None,:]
                Ahat = np.dot(B, PD)
                pp.ion(); pp.figure(); pp.imshow(Ahat); pp.show()
                diffm = A - Ahat
                rel_error[kID, epsilonID, trial] = (diffm ** 2).mean() / (A ** 2).mean() # np.max([np.abs(diffm[:,i]).sum() for i in xrange(64)])
                rel_error_matnorm[kID, epsilonID, trial] = np.max([np.abs(diffm[:,i]).sum() for i in xrange(n)]) / np.max([np.abs(A[:,i]).sum() for i in xrange(n)])
                # np.linalg.norm(A - Ahat, axis=0, ord='fro').max()  # max inf norm over cols
            print "Dictionary MSE/matrix-1-norm Reconstruction Error (k=%d): %1.3f/%1.3f" % (k, rel_error[kID, epsilonID, :].mean(), rel_error_matnorm[kID, epsilonID, :].mean())

            pp.figure()
            pp.clf()
            pp.imshow(np.dot(support * coeffs, A.T)[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=X[0].min(), vmax=X[0].max())
            pp.title('Sample x')
            pp.colorbar()
            pp.savefig(save_dir + 'imgs/x_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

            pp.figure()
            pp.clf()
            pp.imshow(X[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=X[0].min(), vmax=X[0].max())
            pp.title('Noisy x')
            pp.colorbar()
            pp.savefig(save_dir + 'imgs/xnoisy_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

            pp.figure()
            pp.clf()
            pp.imshow(Ahat, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
            pp.title('Recovery (as columns)')
            pp.colorbar()
            pp.savefig(save_dir + 'imgs/Ahat_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

            pp.figure()
            pp.clf()
            pp.imshow(PD, interpolation='nearest')
            pp.title('PD Recovery (Ahat = BPD)')
            pp.colorbar()
            pp.savefig(save_dir + 'imgs/PD_recovery_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

            pp.figure()
            pp.clf()
            pp.imshow(B, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
            pp.title('Learned dictionary B (y = Bb)')
            pp.colorbar()
            pp.savefig(save_dir + 'imgs/B_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)
            pp.close('all')

    np.savez(save_dir + 'error_table_N%d_Trials%d_ks%d' % (N, num_trials, len(ks)), rel_error=rel_error, rel_error_matnorm=rel_error_matnorm)

if PLOT:
    rel_error = np.load(save_dir + 'error_table_N%d_Trials%d_ks%d.npz' % (N, num_trials, len(ks)))['rel_error']
    rel_error_matnorm = np.load(save_dir + 'error_table_N%d_Trials%d_ks%d.npz' % (N, num_trials, len(ks)))['rel_error_matnorm']
    
    pp.figure()
    for kID, k in enumerate(ks):
        pp.errorbar(epsilons, rel_error[kID].mean(axis=1), yerr=rel_error[kID].std(axis=1), linewidth=2, label='k=%d' % k)
    pp.title('MSE of Ahat vs A relative to sample epsilon (trials=%d; sd errors)' % num_trials)
    pp.xlabel('Noise in input')
    pp.ylabel('Dictionary recovery error (mean square error / mean square of A)')
    pp.legend(frameon=False, loc='best')
    pp.savefig(save_dir + 'error_MSE_FastICA_N%d_k%d.pdf' % (N, k), bbox_inches='tight', pad_inches=.05)

    pp.figure()
    for kID, k in enumerate(ks):
        pp.errorbar(epsilons, rel_error_matnorm[kID].mean(axis=1), yerr=rel_error_matnorm[kID].std(axis=1), linewidth=2, label='k=%d' % k)
    pp.title('||A-Ahat||_1 relative to sample epsilon (trials=%d; sd errors)' % num_trials)
    pp.xlabel('Noise in input')
    pp.ylabel('Recovery error (max abs col sum A-Ahat / max abs col sum A)')
    pp.legend(frameon=False, loc='best')
    pp.savefig(save_dir + 'error_mat1norm_FastICA_N%d_k%d.pdf' % (N, k), bbox_inches='tight', pad_inches=.05)
