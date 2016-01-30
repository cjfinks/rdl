""" Runs sparse coding algorithm using Theano on image patches drawn from an image """

import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
from utils import sample_from_bernoulli
from sparse_model import SparseModel

""" Import dictionary """
A = np.load('../../data/DCT_12x12.npy')[:64,:]  # cols are DCT images
n, m = A.shape
N = 1000
ks = [1, 2, 4, 8, 16, 32]
epsilons = np.linspace(0, 10)[:10]

""" Initialize plot """
pp.ion()
pp.figure()
pp.clf()
pp.imshow(A, cmap='gray', interpolation='nearest', vmin=A.min(), vmax=A.max())
pp.title('DCT Basis (as columns)')
pp.colorbar()

""" Generate dataset """
k = ks[0]; eps = epsilons[1]
supports = sample_from_bernoulli(m * [1. * k / m], N).T  # sparse codes
coeffs = sign(np.random.randn(N, m)) + np.random.randn(N, m); 
noise = np.random.randn(N, n)  # noise term
X = np.dot(supports * coeffs, A.T ) + eps * noise
X = np.vstack([X, A.T])

print 'Initializing sparse coding'
lrate = 3e-3
nb = 10 # batch size
data = theano.shared(X.astype(np.float32))
model = SparseModel(n, m, nb, xvar = 0.001, sparsity = 100.0)
x = T.fmatrix()
lr = T.fscalar()
idxs = T.lscalar()
objs, ss, learn_updates = model.update_params(x, lr)
train_model = theano.function([idxs, lr],
		                      [objs, ss],
		                      updates = learn_updates,
							  givens = {x: data[ idxs:idxs + nb, : ]},
							  allow_input_downcast = True )

print " Initialize plotting "
plot_ivl = 50 # plotting interval
pp.ion()
fig, ax = pp.subplots(1,1)
B = model.W.get_value() # columns are dictionary elements
img = ax.matshow(B, cmap = 'gray')

print "Run dictionary learning algorithm "
for i in range(3000):
	idx = np.random.randint( N - nb ) # starting index of batch (batch is idx:idx+nb)
	var_exp, ssh = train_model( idx, lrate ) # train over one batch
	print var_exp #pp.plot(var_exp); pp.show() # print progress

	# Plot
	B = model.W.get_value() 
	img.set_data(B)
	img.autoscale()
	fig.canvas.draw()
	plot_title = 'iteration %d' % i
	pp.title( plot_title )

	b, void = model.map_estimate(X)
	B = model.W.get_value()
	PD = b[-64:,:].T
	Ahat = np.dot(B, PD)
	diffm = A - Ahat
