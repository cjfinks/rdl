" Give intuition for why m(k+1) vectors distributed over m 'consecutive' supports should have a unique code with probability 1 "

import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

" Initialize plot "
fig = pp.figure()
ax = fig.add_subplot( 111, projection = '3d' )

" Set dimensionality "
m = 5 # number of dictionary elements
k = 2 # every data point lies on a 2D-plane
n = 3 # ambient space of data (3d)

""" Initialize a dictionary of norm 1 vectors that satisfies spark condition """
A = 2*np.random.rand( n, m ) - 1 # values between -1 and 1. satisfies spark condition w/ prob 1
A /= np.sqrt(np.sum( A**2, axis = 0 ))[None, :] # scale columns to have norm 1

""" Generate dataset of k-sparse m-dimensional vectors and measure with A"""
Ss = [range(i, min(i+k,m) ) + range( k - (m-i) ) for i in xrange(m)]
N = m * (k+1)
a = np.zeros( (N, m) )
for i, S in enumerate(Ss):
	a[i*(k+1):(i+1)*(k+1), S] = np.random.rand(k+1, k)
y = np.dot( a, A.T )

""" Plot measurements in 3-space """
ax.scatter( y.T[0], y.T[1], y.T[2], s = 20 )
pp.show()

""" Draw dictionary elements """
from utils import Arrow3D
for vec in A.T:
	arrow = Arrow3D([0,vec[0]],[0,vec[1]],[0,vec[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
	ax.add_artist(arrow)
pp.draw()

"""
" Do dictionary learning "
from sparse_model import SparseModel
import theano
import theano.tensor as T
lrate = 3e-2
nb = 16
data = theano.shared(y.astype(np.float32))
model = SparseModel(nx = n, ns = m, nb = nb, xvar = 0.001, sparsity = 100.0)
x = T.fmatrix()
lr = T.fscalar()
idxs = T.lscalar()
objs, ss, learn_updates = model.update_params(x, lr)
train_model = theano.function([idxs, lr],
		                      [objs, ss],
		                      updates = learn_updates,
							  givens = {x: data[ idxs:idxs + nb, : ]},
							  allow_input_downcast = True )
for i in range(1000):
	idx = np.random.randint( N - nb ) # starting index of batch (batch is idx:idx+nb)
	var_exp, ssh = train_model( idx, lrate ) # train over one batch
	print var_exp #pp.plot(var_exp); pp.show() # print progress
Ahat = model.W.get_value() # cols are dict elements
#Ahat *= np.sign( np.sum( Ahat, axis=0 ) )
Ahat *= np.sign( Ahat[2] )

" Draw estimated dictionary elements "
from utils import Arrow3D
for vec in Ahat.T:
	arrow = Arrow3D([0,vec[0]],[0,vec[1]],[0,vec[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
	ax.add_artist(arrow)
pp.draw()
"""
