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
from numpy.linalg import norm
import matplotlib.pyplot as pp
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
import os
import scipy.io
from scipy.misc import comb
import data.data as data
from utils import random_sparse_vectors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sparse_model import SparseModel
import pickle
import itertools

class RandomSparseSources:
	""" Generates sparse data with coefficients in [0,1] """
	def __init__( self, m, k ):
		self.m = m
		self.k = k
	
	def consecutive_cyclic_interval_generator(self, m, k):
		cycle = range(m) + range(k)
		for i in range(m):
			yield cycle[i:i+k]

	def sample_nPerSupport( self, n, supports = 'all' ):
		m = self.m; k = self.k
		if supports == 'all':
			Ss = itertools.combinations( range(m), k )
			N = int(comb(m, k)) * n
		elif supports == 'consecutive cyclic intervals':
			Ss = self.consecutive_cyclic_interval_generator( m, k )
			N = m * n
		a = np.zeros( (N , m) ) # N is now the number of vectors *per support*
		for i in xrange( N/n ):
			S = list(Ss.next())
			for j in xrange(n):
				a[i*n + j][S] = np.random.rand(1, k)
		return a

	def sample( self, n, supports = 'all' ):
		m = self.m; k = self.k
		a = np.zeros( (n, m) ) 
		if support_set == 'consecutive cyclic intervals':
			Ss = [ range(i,i+k) for i in xrange(m-k+1) ] + [ range(k-i-1) + range(m-i-1,m) for i in xrange(k) ]
			for ai in a:
				S = random.choice( Ss )
				ai[S] = np.random.rand(1,k)
		elif support_set == 'all':
			for ai in a:
				S = np.random.choice( m, k, replace = 0 )
				ai[S] = np.random.rand(1,k)
		return a

class SCAModel:
	def __init__( self, mixing_matrix ):
		self.mixingMatrix = mixing_matrix
	
	def sample_nPerSubspace( self, N, k, eps = 0, subspaces = 'all' ):
		# Generate random k-sparse vectors
		n, m = self.mixingMatrix.shape
		randomSparseSources = RandomSparseSources( m, k )
		sources = randomSparseSources.sample_nPerSupport( N, supports = subspaces )
		samples = np.dot( sources, self.mixingMatrix.T)
		N = samples.shape[0]
		if eps > 0: 
			noise = eps * normalize( np.random.randn(N, n), axis=1, norm='l2' ) # noise on eps-sphere
			samples += noise
		return samples, sources

class Experiment:
	def __init__(self, parameters=None):
		self.parameters = parameters
		self.resultsDirectory = os.path.abspath( os.path.dirname(__file__) ) + '/results/' + os.path.splitext(os.path.basename(__file__))[0] + '/'
		self.new_run()

	def new_run( self ):
		# Make directory to store the results of this run
		iRun=1
		while os.path.isdir(self.resultsDirectory + 'run' + str(iRun)): 
			iRun+=1
		runDirectory = self.resultsDirectory + 'run' + str(iRun) + '/'
		os.makedirs(runDirectory)
		with open( runDirectory + 'parameters.txt', 'wb') as f:
			for key in self.parameters.keys():
				parameter = self.parameters[key]
				if isinstance(parameter, list) or isinstance(parameter, np.ndarray):
					listOfValues = '\n'.join( str(val) for val in parameter ) 
					f.write( '%s = ...\n%s\n\n' % (key, listOfValues ) )
				else:
					f.write( '%s = %s\n\n' % (key, str(parameter) ) )
			with open( runDirectory + 'parameters.pkl', 'wb') as f:
				pickle.dump(self.parameters, f)
		self.runDirectory = runDirectory

	def set_run( self, run ):
		pass

if __name__ == '__main__':
	# Set experimental parameters
	params = {'sparsities': [1, 2, 3], 
			  'noise bounds':  np.linspace(0.001, np.sqrt(2), 10), 
			  'trials per configuration': 3, 
			  'sparse supports': 'consecutive cyclic intervals',
			  'algo': 'ICA' }
	experiment = Experiment( params )

	# Choose Dictionary 
	dictionary = data.Dictionary( 'DCT_8x8.mat' ); dictionary.normalize()
	A = dictionary.matrix; m = A.shape[1]
	fig = pp.figure(); pp.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
	pp.title('Dictionary'); pp.colorbar()
	pp.savefig(experiment.runDirectory + 'dictionary.png', bbox_inches='tight', pad_inches=.05)
	pp.close(fig)

	# Run experiment
	scaModel = SCAModel( A )
	max_error = np.zeros( (len(params['sparsities']), len(params['noise bounds'])) )
	for i_k, k in enumerate( params['sparsities'] ):
		samplesPerSupport = k * int(comb(m,k))
		for i_eps, eps in enumerate( params['noise bounds'] ):
			print 'k = %d' % k
			print 'eps = %1.3f' % eps
			for trial in xrange(params['trials per configuration']):
				Y, a = scaModel.sample_nPerSubspace( samplesPerSupport, k, eps, subspaces = params['sparse supports'] )
				if params['algo'] == 'ICA':
					ica = FastICA( max_iter=1000, tol=0.0001, algorithm='parallel' )  # initialize ica algo. could params  be more optimial?
					ica.fit(Y)  # fit the ICA model to infer B and vectors b
					B = ica.mixing_  # Get estimated mixing matrix B
					PD = ica.transform(A.T).T # transform original dictionary elements themselves to get permutation
				elif params['algo'] == 'SCA':
					sca = SparseModel( nx = Y.shape[1], ns = m, nb = 3 )
					B = sca.fit( Y, n_iter = 1000 )
					PD = sca.map_estimate( A.T ).T
				PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # To make permutation, set all but the largest value in each column to zero
				if not np.all( np.sum(abs(PD)>0, axis=0)  == 1 ): # check that this method worked
					print '(trial = %d) Estimated P not a permutation matrix.' % trial

				# Determine max mean-square and max mat1norm errors over all trials
				BPD = np.dot( B, PD)
				error = np.mean( norm(A - BPD, axis=0) / norm(A, axis=0) )
				max_error[i_k, i_eps] = max( error, max_error[i_k, i_eps] )
			print " maximum mean_i( ||(A-BPD)e_i||_2 ) over %d trials: %1.3f" % (params['trials per configuration'], max_error[i_k, i_eps])

			# For each value of the experimental parameters we plot/save the results of the last trial
			pp.ioff()
			fig, axes = pp.subplots(2,2);
			fig.suptitle('Original and recovered dictionaries (k = %d, eps = %1.4f)' % (k, eps))

			# Plot B, PD, BPD and A for the last trial
			dictionaries = { 'A': dictionary, 'BPD': data.Dictionary(BPD), 'B': data.Dictionary(B) }
			for ax, key in zip( axes.ravel()[:-1], dictionaries.keys() ):
				ax.set_title( key )
				dict = dictionaries[key]; 
				img = dict._tile_atoms()
				im = ax.imshow(img, interpolation='none', vmin = dict.matrix.min(), vmax = dict.matrix.max())
				# create a new axis, cax, located 0.05 inches to the right of ax, whose width is 10% of ax. cax is used to plot a colorbar for each subplot
				div = make_axes_locatable(ax)
				cax = div.append_axes("right", size="10%", pad=0.05)
				cbar = pp.colorbar(im, cax=cax, format="%.2g")
				ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
			ax = axes.ravel()[-1]
			ax.set_title( 'PD' )
			im = ax.imshow(PD, interpolation='none', vmin = PD.min(), vmax = PD.max())
			ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
			div = make_axes_locatable(ax)
			cax = div.append_axes("right", size="10%", pad=0.05)
			cbar = pp.colorbar(im, cax=cax, format="%.2g")
			# Save progress so far
			np.savez( experiment.runDirectory + 'ABPD', A=A, B=B, PD=PD )
			np.save( experiment.runDirectory + 'error', max_error )
			# Save A, BPD, B and PD for the last trial of this setting for k and eps
			fig_dir = experiment.runDirectory + 'A_BPD_B_PD/'
			if not os.path.isdir(fig_dir): os.makedirs(fig_dir)
			pp.savefig(fig_dir + 'k=%d_eps=%1.3f.png' % (k, eps))
			pp.close(fig)

	# Plot end-result
	with open( experiment.runDirectory + 'parameters.pkl', 'r') as f:
		parameters = pickle.load(f)
	error = np.load(experiment.runDirectory + 'error.npy')
		
	""" Plot MSE reconstruction error as a function of epsilon for each k """
	fig = pp.figure()
	for i_k, k in enumerate(parameters['sparsities']):
		pp.plot(parameters['noise bounds'], error[i_k], linewidth=2, label='k=%d' % k)
	pp.title('Maximum dictionary recovery error over %d trials vs. noise bound' % parameters['trials per configuration'])
	pp.xlabel('Noise in input')
	pp.ylabel('max mean_i ||(A-BPD)e_i||_2 over %d trials ' % parameters['trials per configuration'])
	pp.legend(frameon=False, loc='best')
	pp.savefig(experiment.runDirectory + 'max_error.png', bbox_inches='tight', pad_inches=.05)
	pp.close(fig)
