"""
--CAPTION--
Provided the signal-noise ratio is large enough, the maximum column-wise difference between the learned dictionary and the generating dictionary grows linearly with the noise once the columns of the learned dictionary are properly re-ordered and scaled. The figure was generated as follows. First, a random (n x n) dictionary $A$ and $N = m[(k-1)mCk+1]$ random $k$-sparse vectors $\mathbf{a}_i$ were drawn by independently sampling their non-zero coefficients from a uniform distribution over [-1,1], with an equal number of $k$-sparse vectors being supported on each length-$k$ interval in $[m]$. Then, a noisy dataset $\mathbf{y}_i = A\mathbf{a}_i + \mathbf{n}_i$ was generated for each value of $\varepsilon$ by drawing $\mathbf{n}_i$ from a uniform distribution over the $\varepsilon$-sphere. For each of these datasets, we ran FastICA until convergence to a dictionary $B$ and sources $\mathbf{b}_i$ and then thresholded each source to be $k$-sparse by setting all but their $k$ largest absolute cofficients to zero. To each such matrix $B$ a corresponding permutation-scaling matrix $PD$ was attributed, estimated to be the matrix obtained by setting all but the largest element in every column of $B^{-1}A$ to 0 (a check was performed to confirm that every row of PD had only one nonzero element as well). The plot shows, for each value of $\varepsilon$, the maximum value of $\max_i |(A-BPD)_i|_2$ over all dictionaries $B$ and corresponding $k$-sparse sources $\mathbf{b}_i$ obtained by this procedure which satisfied $|\mathbf{y}_i - B\mathbf{b}_i|_2 \leq \varepsilon$. 

"""
import os
import numpy as np
import data, data.datatypes
from scipy.misc import comb
import itertools
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA
import matplotlib.pyplot as pp
import bottleneck

class SparseVectorGenerator:
	""" Generates sparse vectors as the rows of an array, with coefficients sampled uniformly from [-1,1] """
	def __init__( self, m, k, supports = 'all', coef_distr = np.random.rand  ):
		"""
		'all': all possible supports of size k
		'cyclic': consecutive cyclic intervals, i.e. all intervals of length k in some cyclic permutation of 1, ... , m
		"""
		self.m = m
		self.k = k
		self.supports = supports
		self.coef_distr = coef_distr

		if supports == 'cyclic':
			self.num_supports = m
			self.support_generator = self._cyclic_ivl_generator(m, k)
		elif supports == 'all':
			self.num_supports = int(comb(m,k))
			self.support_generator = itertools.combinations( range(m), k )

	def _cyclic_ivl_generator(self, m, k):
		" Generates the consecutive cyclic intervals over [0, ..., m-1] "
		cycle = list(range(m)) + list(range(k))
		for i in range(m):
			yield cycle[i:i+k]

	def sample( self, nSamples, PER_SUPPORT = False ):
		""" 
		Samples vectors with supports self.supports
		"""
		self.__init__(self.m, self.k, self.supports) # reset generators

		# Generate a list of the number of samples to draw for each support
		if PER_SUPPORT is True:
			a = np.zeros( (self.num_supports * nSamples, self.m) )
			nPerSupport = [nSamples] * self.num_supports
		else:
			a = np.zeros( (nSamples, self.m) )
			probs = [1./self.num_supports] * self.num_supports # uniform probability over supports
			nPerSupport = np.random.multinomial( nSamples, probs )
			
		# draw nPerSupport samples from each support
		for (i,S) in enumerate( self.support_generator ): 
			n = nPerSupport[i]
			start = sum(nPerSupport[:i])
			a[start : start + n, S] = 2 * self.coef_distr(n,self.k) - 1
		return a

def keep_top_k( arr, k ):
	""" For every row in the array, sets all but the top k entries to 0 """
	mset = set(range(arr.shape[1]))
	for i, row in enumerate( abs(arr) ): 
		topk = bottleneck.argpartsort( -row, k )[:k] # partially sorts the array so that smallest k entries are at the front
		setto0 = mset - set(topk)
		for j in setto0:
			arr[i,j] = 0
	return arr

def random_orthogonal_matrix( shape ):
	X = np.random.randn( *shape )
	Y = np.dot( X, X.T )
	U, S, V = np.linalg.svd( Y )
	return U

if __name__ == '__main__':
	np.random.seed(0)
	k = 3 # sparsity
	support_set = 'all' # 'cyclic' or 'all'
	nTrialsPerEpsilon = 1
	# (16,16)-ortho k=2, (9,9)-ortho k=3,

	" Load dictionary from file or generate random dictionary "
	filename = os.path.join( data.__path__[0], 'dictionaries/alphabet.npy' ) # available dictionaries: 'alphabet', 'dct8x8', 'dct12x12'
	with open( filename, 'rb' ) as file:
		dictionary = data.datatypes.Dictionary( np.load(file) ) 
	#dictionary = data.datatypes.Dictionary( random_orthogonal_matrix((16,16)) ); 
	#dictionary = data.datatypes.Dictionary( np.random.rand(9,9) ); 
	#dictionary = data.datatypes.Dictionary( np.eye(9,9) ); 
	dictionary.normalize()
	A = dictionary.matrix

	" Draw N random k-sparse vectors a_i with values in [-1, 1]"
	n, m = dictionary.matrix.shape
	#nSamplesPerSupport = (k-1) * int(comb(m,k)) + 1 #NOTE: for k = 1 ICA algo needs more than one sample per support
	nSamplesPerSupport = m * k #NOTE: for k = 1 ICA algo needs more than one sample per support
	sparseVectorGenerator = SparseVectorGenerator( m, k, support_set ) # supports can be 'cyclic' or 'all'
	X = sparseVectorGenerator.sample( nSamplesPerSupport, PER_SUPPORT = True ) # (nSamples, m)

	" Run Experiment "
	eps_0 = 1.5/np.sqrt(2) # 1/sqrt(2) for orthogonal matrix
	epsilons = np.linspace(1e-5, eps_0, 120) 
	maxColErrors = -np.inf * np.ones((len(epsilons), len(epsilons))) # For each epsilon, store the worst result over all trials of max_i ||(A-BPD)_i||_2
	dictionaries = [ np.zeros( A.shape) ] * len(epsilons) # save one reconstruction dictionary for each epsilon
	ica = FastICA( algorithm='parallel' )  
	Ynoiseless = np.dot( X, A.T) # rows are vectors
	for i, eps in enumerate(epsilons): # For each value of epsilon we'll generate noisy data a bunch of times
		print('eps = %1.3f' %  eps)
		for trial in range(nTrialsPerEpsilon): # 
			Y = Ynoiseless + eps * normalize( np.random.randn(*Ynoiseless.shape), axis=1, norm='l2' ) # add noise from the eps-ball
			ica.fit(Y)  
			PD = ica.transform( A.T ).T # transform original dictionary elements themselves to get permutation
			PD *= ( abs(PD) >= np.max(abs(PD), axis=0)[None,:] ) # To make permutation, set all but the largest value in each column to zero
			#TODO: Should test that P is a true permutation, i.e. that all rows have exactly one nonzero entry as well
			bSupports = np.dot( X, PD.T ) != 0
			Xhat = ica.transform(Y) * bSupports #Xhat = keep_top_k( Xhat, k ) # enforce hard k-sparsity
			Yhat = ica.inverse_transform(Xhat) # np.dot( Xhat, ica.mixing_.T )
			maxReconError = np.max( np.linalg.norm( Yhat - Y, axis = 1 ) )
			iEpsilon = next((i for i, x in enumerate(maxReconError < epsilons) if x), None) # find smallest epsilon this is less than
			if iEpsilon is not None: # if maxReconError is bigger than all epsilons, forget it
				Ahat = np.dot( ica.mixing_, PD )
				maxColErrorThisTrial = max( np.linalg.norm( A - Ahat, axis=1 ) )
				#print('max_j ||(A-BPD)_j||_2  = %1.3f' % maxColumnErrorThisTrial)
				for idx in range(iEpsilon, len(epsilons)):
					if maxColErrorThisTrial > maxColErrors[i,idx]:
						maxColErrors[i,idx] = maxColErrorThisTrial
						dictionaries[idx] = Ahat
			if iEpsilon == i:
				break # Do repeat trials only if no answer was found within eps

	" Plot results "
	pp.ion()
	pp.figure;
	maxColErrorsAllEps = np.max(maxColErrors, axis=0)
	pp.plot( epsilons, maxColErrorsAllEps ) # merge results over all generating epsilons
	lim = np.max( [np.max(maxColErrorsAllEps), eps_0] )
	pp.xlim(0, lim); pp.ylim(0, lim) # make axes start at 0
	pp.xlabel('$\max_i \|y_i - Aa_i\|_2$')
	pp.ylabel(r'$\max_i \|(A - BPD)_i\|_2$')
	pp.title('Done Computing. This wassup.')

	#dictionary.matshow()
