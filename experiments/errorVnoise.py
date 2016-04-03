"""
Plots reconstruction error vs noise for synthetic data. Complete dictionary learned using Fast ICA.

1) Fix dictionary A with unit norm columns.
2) Set k = 1, 2, ...
3) Randomly generate  k-sparse a_i. 
4) Set noise level 0.01, 0.1, ... 
5) Generate noisy y_i = Aa_i + n_i
6) Learn alternate dictionary B and k-sparse b_i to minimize ||y_i - Bb_i||
7) Find PD by transforming the columns of A.
8) Compute max_i ||(A-BPD)e_i|| as a function of noise
9) Repeat for n trials
10) Plot results

"""

import numpy as np
import matplotlib.pyplot as pp
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA
import os
import scipy.io
from scipy.misc import comb
from data.data import Dictionary
from utils import random_sparse_vectors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sparse_model import SparseModel

""" TOGGLE PROGRAM FLOW """
PLOT = True #False
COMPUTE = True #True

""" Initialize Simulation """
ks = [1, 2, 3]
epsilons = np.linspace(0.0001, 2.0 / np.sqrt(2), 20) #max-out error at known error threshold for canonical basis
num_trials = 1
algo = 'ICA' #'SCA'

param_string = 'k%dto%d_trials%d/' % (min(ks), max(ks), num_trials)
save_dir = os.path.abspath( os.path.dirname(__file__) ) + '/results'
save_dir = save_dir + '/' + param_string 
if not os.path.isdir(save_dir): os.makedirs(save_dir)
if not os.path.isdir(save_dir + 'figs/'): os.makedirs(save_dir + 'figs/')

""" Fix dictionary A with unit norm columns"""
#Adict = Dictionary( np.eye(16,32) )
Adict = Dictionary('alphabet')#'DCT_8x8.mat') # 'DCT_8x8.mat', 'DCT_12x12.mat', 'alphabet'
Adict.matrix = Adict.matrix[:,:16]
Adict.normalize()
A = np.array( Adict.matrix )
n, m = A.shape

if COMPUTE:
    max_mse = np.zeros((len(ks), len(epsilons)))
    max_mat1norm = np.zeros((len(ks), len(epsilons)))

    for i_k, k in enumerate(ks):
    	#N = int(m * k * comb(m,k))
    	N = k * int(comb(m,k))

        for i_epsilon, epsilon in enumerate(epsilons):
            print "noise: %1.4f" % epsilon

            for trial in xrange(num_trials):
            	# Generate random k-sparse vectors
            	a = random_sparse_vectors( N, m, k, support_set = 'consecutive', RANDOM_SUPPORTS = False) # random m-dim k-sparse a in [0,1]^m
            	a = normalize(a, axis=1, norm='l2')  # make them have unit norm

            	# Make noisy Data
                noise = np.random.randn(a.shape[0], n)   # circ-sym n-dimensional noise. N rows of n-dim vectors.
                noise = epsilon * normalize( noise, axis=1, norm='l2' ) # put noise on epsilon-sphere
                Y = np.dot( a, A.T) + noise # N rows of n-dim vectors

                # Run inference algorithm
                if algo == 'ICA':
                	ica = FastICA( max_iter=1000, tol=0.0001, algorithm='parallel' )  # initialize ica algo. could params  be more optimial?
                	ica.fit(Y)  # fit the ICA model to infer B and vectors b
                	B = ica.mixing_  # Get estimated mixing matrix B
                	PD = ica.transform(A.T).T # transform original dictionary elements themselves to get permutation
                	PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # Map to 'closest' permutation-scaling matrix
                elif algo == 'SCA':
					sca = SparseModel( nx = Y.shape[1], ns = m, nb = 3 )
					B = sca.fit( Y, n_iter = 1000 )
					""" NEED TO FIND PERMUTATION """
					PD = np.eye( m )

                # Find permutation-scaling matrix
                if not np.all( np.sum(abs(PD)>0, axis=0)  == 1 ): # check that this method worked
                	print '(trial = %d) Estimated P not a permutation matrix.' % trial

                # Determine error
                Ahat = np.dot( B, PD)
                err = A - Ahat
                mse =  (err ** 2).mean() / (A ** 2).mean() 
                if mse > max_mse[i_k, i_epsilon]:
                	max_mse[i_k, i_epsilon] = mse
                mat1norm = np.max([np.abs(err[:,i]).sum() for i in xrange(n)]) / np.max([np.abs(A[:,i]).sum() for i in xrange(n)])
                if mat1norm > max_mat1norm[i_k, i_epsilon]:
                	max_mat1norm[i_k, i_epsilon] = mat1norm
            print "Dictionary max MSE / max matrix-1-norm reconstruction error (k=%d): %1.3f/%1.3f" % (k, max_mse[i_k, i_epsilon], max_mat1norm[i_k, i_epsilon])

            if PLOT:
				""" Plot B, PD, Ahat and A for the last trial """
				Bdict = Dictionary( B ); Ahatdict = Dictionary( Ahat ); 
				dictionaries = {'A': Adict, 'Ahat': Ahatdict, 'B': Bdict}
				titles = ['A', 'Ahat', 'B']
				fig, axes = pp.subplots(2,2); pp.suptitle('Original and recovered dictionaries (k = %d, eps = %1.4f)' % (k, epsilon))
				for ax, key in zip( axes.ravel()[:-1], dictionaries.keys() ):
					ax.set_title( key )
					dict = dictionaries[key]; 
					img = dict._tile_atoms()
					if key != 'Ahat':
						im = ax.imshow(img, interpolation='none')
					else:
						ax.imshow(img, interpolation='none')
					# creates a new axis, cax, located 0.05 inches to the right of ax, whose width is 15% of ax
					# cax is used to plot a colorbar for each subplot
					div = make_axes_locatable(ax)
					cax = div.append_axes("right", size="10%", pad=0.05)
					cbar = pp.colorbar(im, cax=cax, format="%.2g")
					ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
				ax = axes.ravel()[-1]
				ax.set_title( 'PD' )
				im = ax.imshow(PD, interpolation='none')
				ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
				div = make_axes_locatable(ax)
				cax = div.append_axes("right", size="10%", pad=0.05)
				cbar = pp.colorbar(im, cax=cax, format="%.2g")
				pp.pause(1)
    np.save(save_dir + 'error', error_mse = max_mse, error_mat1norm = max_mat1norm)

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
	for i_k, k in enumerate(ks):
		#pp.errorbar(epsilons, error[i_k].mean(axis=1), yerr=error[i_k].std(axis=1), linewidth=2, label='k=%d' % k)
		pp.plot(epsilons, max_mse[i_k], linewidth=2, label='k=%d' % k)
	pp.title('max MSE of Ahat vs A relative to sample epsilon (trials=%d)' % num_trials)
	pp.xlabel('Noise in input')
	pp.ylabel('Dictionary recovery error (mean square error / mean square of A)')
	pp.legend(frameon=False, loc='best')
	#pp.savefig(save_dir + 'error_mse.pdf', bbox_inches='tight', pad_inches=.05)
	
	""" Plot matrix 1-norm reconstruction error as a function of epsilon for each k """
	pp.sca(axes[1])
	for i_k, k in enumerate(ks):
		#pp.errorbar(epsilons, error_mat1norm[i_k].mean(axis=1), yerr=error_mat1norm[i_k].std(axis=1), linewidth=2, label='k=%d' % k)
		pp.plot(epsilons, max_mat1norm[i_k], linewidth=2, label='k=%d' % k)
	pp.title('max ||A-Ahat||_1 relative to sample epsilon (trials=%d)' % num_trials)
	pp.xlabel('Noise in input')
	pp.ylabel('Recovery error (max abs col sum A-Ahat / max abs col sum A)')
	pp.legend(frameon=False, loc='best')
	#pp.savefig(save_dir + 'error_mat1norm.pdf', bbox_inches='tight', pad_inches=.05)


""" SPARE CODE """
""" Inside loop: plots noiseless vs noisy first sample """
"""
fig, axes = pp.subplots(1,2); pp.suptitle('Datum (k = %d)' % k)
Ynoiseless = np.dot( a, A.T )
pp.sca(axes[0]); pp.title('Noiseless')
pp.imshow( Ynoiseless[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=Y[0].min(), vmax=Y[0].max())
pp.sca(axes[1]); pp.title('Noisy (epsilon = %1.4f)' % epsilon)
pp.imshow(Y[0].reshape(np.sqrt(n),np.sqrt(n)), cmap='gray', interpolation='nearest', vmin=Y[0].min(), vmax=Y[0].max())
pp.colorbar()
#pp.savefig(save_dir + 'imgs/sample_k%d_epsilon%1.4f.png' % (k, epsilon), bbox_inches='tight', pad_inches=.05)
"""
