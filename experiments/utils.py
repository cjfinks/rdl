import numpy as np

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
