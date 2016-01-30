# Experiment 1:  Synthetic Data
# set K = 4, 5, ...?
# let A = DCT matrix (j-th column is j-th orthonormnal 8x8 DCT basis vector)
# pick 640 points in 64 dimensions as follows
# choose K nonzero coefficients in a (of the 64) ~ n(0,1) and make linear combination
# x = Aa + noise * n(0,1); 0 <= noise <= NAX_nOISE
# do fastICA to recover x = Bb; find P,D, compute ||A - BPD|| as function of noise

import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from utils import sample_from_bernoulli
import os
import theano
from sparse_model import SparseModel
from theano import tensor as T

from scipy.fftpack import dct, idct
A = np.load('../../data/DCT_12x12.npy')  # cols are DCT images
A = A[:64,:]

n, m = A.shape
N = 10000 
Ks = [1, 2, 4]
noises = np.linspace(0, 10)[:10]
num_trials = 2

save_dir = 'figs/tst_pFastICA_N%d_Ks%d_Trials%d/' % (N, len(Ks), num_trials)
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'imgs/'): os.makedirs(save_dir + 'imgs/')

plt.figure()
plt.clf()
plt.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
plt.title('DCT Basis (as columns)')
plt.colorbar()
plt.savefig(save_dir + 'DCT.png', bbox_inches='tight', pad_inches=.05)

import pdb
pdb.set_trace()
rel_error = np.zeros((len(Ks), len(noises), num_trials))
for k, K in enumerate(Ks):
    for nidx, noise in enumerate(noises):
        print "noise: %1.4f" % noise
        for trial in xrange(num_trials):
            a = sample_from_bernoulli(m * [1. * K / m], N).T  # sparse codes
            # a = np.vstack([a, np.identity(n)])
            C = np.sign(np.random.randn(N, m)) + np.random.randn(N, m); # coefficients
            D = np.random.randn(N, n)  # noise term 
            X = np.dot(a * C, A.T) + noise * D
            # tack on ei's to figure out mapping
            X = np.vstack([X, A.T])
            
            print 'Initializing sparse coding'
            lrate = 3e-6
            nb = 3 # batch size
            data = theano.shared(X.astype(np.float32))
            model = SparseModel(n, m, nb, xvar = 0.001, sparsity = 100.0)
            x = T.fmatrix()
            lr = T.fscalar()
            idxs = T.lscalar()
            objs, ss, learn_updates = model.update_params(x, lr)
            train_model = theano.function([idxs, lr], [objs, ss], updates = learn_updates, givens = {x: data[ idxs:idxs + nb, : ]}, allow_input_downcast = True )
            print "Run dictionary learning algorithm "
            for i in range(200):
            	idx = np.random.randint( N - nb ) # starting index of batch (batch is idx:idx+nb)
            	var_exp, ssh = train_model( idx, lrate ) # train over one batch
            	print var_exp #pp.plot(var_exp); pp.show() # print progress
            	
            	
            B = model.W.get_value() # cols are dict elements
            b = model.map_estimate( X )
            PD = b[-64:,:].T
            Ahat = np.dot(B, PD)
            diffm = A - Ahat
            rel_error[k, nidx, trial] = (diffm ** 2).mean()  # np.max([np.abs(diffm[:,i]).sum() for i in xrange(64)])
            # np.linalg.norm(A - Ahat, axis=0, ord='fro').max()  # max inf norm over cols

            print "Dictionary Reconstruction Error (K=%d): %1.3f" % (K, rel_error[k, n, :].mean())
            plt.figure()
            plt.clf()
            plt.imshow(Ahat, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
            plt.title('Recovery (as columns)')
            plt.colorbar()
            plt.savefig(save_dir + 'imgs/Ahat_FastICA_N%d_K%d_noise%1.4f.png' % (N, K, noise), bbox_inches='tight', pad_inches=.05)
            
            plt.figure()
            plt.clf()
            plt.imshow(PD, interpolation='nearest')
            plt.title('PD Recovery (Ahat = BPD)')
            plt.colorbar()
            plt.savefig(save_dir + 'imgs/PD_recovery_FastICA_N%d_K%d_noise%1.4f.png' % (N, K, noise), bbox_inches='tight', pad_inches=.05)

            plt.figure()
            plt.clf()
            plt.imshow(B, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
            plt.title('Learned dictionary B (y = Bb)')
            plt.colorbar()
            plt.savefig(save_dir + 'imgs/B_FastICA_N%d_K%d_noise%1.4f.png' % (N, K, noise), bbox_inches='tight', pad_inches=.05)
            plt.close('all')
            
            np.save(save_dir + 'error_table_N%d_Trials%d_Ks%d' % (N, num_trials, len(Ks)), rel_error)
            
            plt.figure()
            #plt.plot(noises, rel_error.mean(axis=1), linewidth=2)
            for k, K in enumerate(Ks):
            	plt.errorbar(noises, rel_error[k].mean(axis=1), yerr=rel_error[k].std(axis=1), linewidth=2, label='K=%d' % K)
            	plt.title('NSE of Ahat vs A relative to noise in samples (trials=%d; sd errors)' % num_trials)
            	#plt.title('||A-Ahat||_1 relative to noise in samples (trials=%d; sd errors)' % num_trials)
            	plt.xlabel('noise in input')
            	plt.ylabel('Dictionary recovery error (max absolute sum)')
            	plt.legend(frameon=False, loc='best')
            	plt.savefig(save_dir + 'error_FastICA_N%d_K%d.pdf' % (N, K), bbox_inches='tight', pad_inches=.05)
