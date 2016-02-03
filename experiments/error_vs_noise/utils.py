""" Draw a 3D vector """
import numpy as np
import random

def sample_from_bernoulli(p, M=1):
    """
    Returns N x M numpy array with M Bernoulli(p) N-bit samples
    
    Parameters
    ----------
    p : Type
        Description
    M : int, optional
        Description (default 1)
    
    Returns
    -------
    Value : Type
        Description
    """
    N = len(p)
    p = np.array(p)
    v_cp = np.array([p, ] * M).transpose()
    rand_vect = np.random.random((N, M))
    outcome = v_cp > rand_vect
    data = outcome.astype("int")
    if M == 1:
        return data[:, 0]
    return data

def random_sparse_vectors( N, m, k, support_set = 'random' ):
	""" Generates random k-sparse m-dimensional vectors """
	
	a = np.zeros( (N, m) ) # initialize k-sparse vectors
	if support_set == 'random':
		for ai in a:
			S = np.random.choice( m, k, replace = 0 )
			ai[S] = np.random.rand(1,k)
	elif support_set == 'consecutive':
		Ss = [ range(i,i+k) for i in xrange(m-k+1) ]
		Ss += [ range(k-i-1) + range(m-i-1,m) for i in xrange(k) ]
		for ai in a:
			S = random.choice( Ss )
			ai[S] = np.random.rand(1,k)

	return a

def closest_permutation_scaling( X ):
	""" Finds the closest permutation-scaling matrix to M through a shitty heuristic """

	PD = np.zeros(X.shape)
	M = abs(X)
	x = [] # stores max indices of previously scanned columns
	for j in xrange(len(M.T)):
		i = np.argmax(M[:,j])
		while i in x:
			M[i,j] = 0 # this i already taken by previous column
			i = np.argmax(M[:,j])
		PD[i,j] = X[i,j]
		x.append(i)
	return PD
