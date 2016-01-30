# Experiment 1N:  Synthetic Data with growing # samples N

import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA, DictionaryLearning
from utils import sample_from_bernoulli, corrupt_binary_data
import os

TEST = False
PLOT = False
COMPUTE = True
BASIS = 'DCT' # 'DCT' # 'RAND', 'BAD'
METHOD = 'ICA' # 'ICA', 'SCA'
MAX_ITER_ICA = 100000
error_tol = .00001

if TEST:
	n = 64; m = 64; ks = [1,2]
	A = np.random.rand(n, m); A = normalize(A,axis=0)
	epsilon = 0.0001
	num_trials = 3
	Ns = range(100,300000)[::30000]
else:
	ks = [2, 4]  # , , 8, 16]
	Ns = range(100,300000)[::30000]
	epsilon = .0001
	num_trials = 3
	if BASIS == 'DCT':
		A = np.load('./DCT_8x8.npz')['DCT']  # cols are DCT images
		#A = np.load('./DCT_12x12.npy')  # cols are DCT images
	if BASIS == 'RAND':
	    A = np.random.randn(64, 64)
	if BASIS == 'BAD':
	    A = np.dot(np.random.randn(64, 64), np.dot(np.random.randn(64, 2), np.random.randn(2, 64)))

A = normalize(A, axis = 0, norm = 'l2')
n, m = A.shape

save_dir = 'figs/' + BASIS + '_p' + METHOD + '_Ns%d_ks%d_Trials%d_exp1N/' % (len(Ns), len(ks), num_trials)
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'imgs/'): os.makedirs(save_dir + 'imgs/')

if PLOT:
	plt.figure()
	plt.clf()
	plt.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
	plt.title('Basis (as columns)')
	plt.colorbar()
	plt.savefig(save_dir + 'Basis.png', bbox_inches='tight', pad_inches=.05)

if COMPUTE:
    rel_error = np.zeros((len(ks), len(Ns), num_trials))
    rel_error_matnorm = np.zeros((len(ks), len(Ns), num_trials))

    for kID, k in enumerate(ks):
    	print 'k = %d' % k
    	for Nid, N in enumerate(Ns):
#        for epsilonID, epsilon in enumerate(epsilons):
            print "N: %d" % N
            for trial in xrange(num_trials):
                support = corrupt_binary_data( np.zeros((N,m)), bits_corrupted=k ) #support = sample_from_bernoulli(n * [1. * k / n], N).T  # sparse support indicator functions
                coeffs = 0.5 + 0.5 * np.random.rand(N, m) #np.sign(np.random.randn(N, n)) + np.random.randn(N, n); 
                a = support * coeffs # sparse codes in [0.5, 1]
                noise = epsilon * (1 - 2 * np.random.rand(N, n))  # noise in [-epsilon, epsilon]
                X = np.dot(support * coeffs, A.T) + noise
                if METHOD == 'ICA':
                	ica = FastICA(max_iter=MAX_ITER_ICA, tol=0.0001, algorithm='parallel')  # maybe params could be more optimial
                	ica.fit(X)  # Reconstruct signals
                	B = ica.mixing_  # Get estimated mixing matrix
                	PD = ica.transform(A.T).T # transform canonical basis vectors to get PD
                PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # keep only the largest indices in each column
                #PD *= np.sign( np.diag(np.dot(A.T, np.dot(B, PD) )) )[None,:]
                Ahat = np.dot(B, PD)
                Ahat = normalize(Ahat, axis = 0, norm = 'l2')
                # plt.ion(); plt.figure(); plt.imshow(Ahat); plt.show()
                diffm = A - Ahat
                #rel_error[kID, epsilonID, trial] = (diffm ** 2).mean() / (A ** 2).mean() # np.max([np.abs(diffm[:,i]).sum() for i in xrange(64)])
                #rel_error_matnorm[kID, epsilonID, trial] = np.max([np.abs(diffm[:,i]).sum() for i in xrange(n)]) / np.max([np.abs(A[:,i]).sum() for i in xrange(n)])
                rel_error[kID, Nid, trial] = (diffm ** 2).mean() / (A ** 2).mean() # np.max([np.abs(diffm[:,i]).sum() for i in xrange(64)])
                rel_error_matnorm[kID, Nid, trial] = np.max([np.abs(diffm[:,i]).sum() for i in xrange(n)]) / np.max([np.abs(A[:,i]).sum() for i in xrange(n)])
                # np.linalg.norm(A - Ahat, axis=0, ord='fro').max()  # max inf norm over cols
            print "Dictionary MSE/matrix-1-norm Reconstruction Error (k=%d): %1.3f/%1.3f" % (k, rel_error[kID, Nid, :].mean(), rel_error_matnorm[kID, Nid, :].mean())

	    # if PLOT:
		   #  plt.figure()
		   #  plt.clf()
		   #  plt.imshow(np.dot(support * coeffs, A.T)[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=X[0].min(), vmax=X[0].max())
		   #  plt.title('Sample x')
		   #  plt.colorbar()
		   #  plt.savefig(save_dir + 'imgs/x_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

		   #  plt.figure()
		   #  plt.clf()
		   #  plt.imshow(X[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=X[0].min(), vmax=X[0].max())
		   #  plt.title('Noisy x')
		   #  plt.colorbar()
		   #  plt.savefig(save_dir + 'imgs/xnoisy_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

		   #  plt.figure()
		   #  plt.clf()
		   #  plt.imshow(Ahat, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
		   #  plt.title('Recovery (as columns)')
		   #  plt.colorbar()
		   #  plt.savefig(save_dir + 'imgs/Ahat_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

		   #  plt.figure()
		   #  plt.clf()
		   #  plt.imshow(PD, interpolation='nearest')
		   #  plt.title('PD Recovery (Ahat = BPD)')
		   #  plt.colorbar()
		   #  plt.savefig(save_dir + 'imgs/PD_recovery_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)

		   #  plt.figure()
		   #  plt.clf()
		   #  plt.imshow(B, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
		   #  plt.title('Learned dictionary B (y = Bb)')
		   #  plt.colorbar()
		   #  plt.savefig(save_dir + 'imgs/B_FastICA_N%d_k%d_epsilon%1.4f.png' % (N, k, epsilon), bbox_inches='tight', pad_inches=.05)
		   #  plt.close('all')

    np.savez(save_dir + 'error_table_N%d_Trials%d_ks%d' % (N, num_trials, len(ks)), rel_error=rel_error, rel_error_matnorm=rel_error_matnorm)

if PLOT:
    rel_error = np.load(save_dir + 'error_table_N%d_Trials%d_ks%d.npz' % (N, num_trials, len(ks)))['rel_error']
    rel_error_matnorm = np.load(save_dir + 'error_table_N%d_Trials%d_ks%d.npz' % (N, num_trials, len(ks)))['rel_error_matnorm']
    
    plt.figure()
    for kID, k in enumerate(ks):
        plt.errorbar(Ns, rel_error[kID].mean(axis=1), yerr=rel_error[kID].std(axis=1), linewidth=2, label='k=%d' % k)
    plt.title('MSE of Ahat vs A relative to sample epsilon (trials=%d; sd errors)' % num_trials)
    plt.xlabel('# Samples N')
    plt.ylabel('Dictionary recovery error (mean square error / mean square of A)')
    plt.legend(frameon=False, loc='best')
    plt.savefig(save_dir + 'error_MSE_FastICA_N%d_k%d.pdf' % (N, k), bbox_inches='tight', pad_inches=.05)

    plt.figure()
    for kID, k in enumerate(ks):
        plt.errorbar(Ns, rel_error_matnorm[kID].mean(axis=1), yerr=rel_error_matnorm[kID].std(axis=1), linewidth=2, label='k=%d' % k)
    plt.title('||A-Ahat||_1 relative to sample epsilon (trials=%d; sd errors)' % num_trials)
    plt.xlabel('# Samples N')
    plt.ylabel('Recovery error (max abs col sum A-Ahat / max abs col sum A)')
    plt.legend(frameon=False, loc='best')
    plt.savefig(save_dir + 'error_mat1norm_FastICA_N%d_k%d.pdf' % (N, k), bbox_inches='tight', pad_inches=.05)
