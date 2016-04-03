""" Runs sparse coding algorithm using Theano on image patches drawn from an image """

import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import skimage.io, os
from sklearn.feature_extraction.image import PatchExtractor
from utils import tile_raster_images, canonically_preprocess
from sparse_model import SparseModel

print 'loading image patches'
imageDirectory = '/Users/Charles/Dropbox/rdl/data/images/'
image = skimage.io.imread( os.path.join(imageDirectory, 'boat.png') )
patchShape = (8,8)
patchExtractor = PatchExtractor( patchShape, max_patches = 100000 ) # automatically extracts all the patches if no max_patches argument
X = patchExtractor.transform( image[None,:,:] ) # requires (n_samples, *shape) as input 
X = X.reshape(X.shape[0], np.prod(X.shape[1:]) )

print 'Whiten and reduce dimensionality'
Xwhite, unwhiten = canonically_preprocess( X, reduce_dim = 0.50 ) # np.dot( Xwhite, Ux ) to map back to original space
Xwhite = np.random.permutation(Xwhite)
nt, nx = Xwhite.shape # nt: number of trials, nx: dimensionality (reduced, nor original number of pixels)
print 'Sample variance: %0.2f' % np.var( Xwhite )

print 'Initializing sparse coding'
lrate = 3e-2
ns = 2 * X.shape[1] # number of features. specify by overcompleteness
nb = 3 # batch size
data = theano.shared(Xwhite.astype(np.float32))
model = SparseModel(nx, ns, nb,
		            xvar = 0.001, 
		            sparsity = 100.0)
x = T.fmatrix()
lr = T.fscalar()
idxs = T.lscalar()
objs, ss, learn_updates = model.update_params(x, lr)
train_model = theano.function([idxs, lr],
		                      [objs, ss],
		                      updates = learn_updates,
							  givens = {x: data[ idxs:idxs + nb, : ]},
							  allow_input_downcast = True )

# Shall we plot as we go? If so, initialize.
plotW = True
plot_ivl = 50 # plotting interval
if plotW:
	print " Initialize plotting "
	pp.ion()
	fig, ax = pp.subplots(1,1)
	sz = int(np.sqrt(X.shape[1]))
	sqrt_ns = int(np.sqrt(ns))
	W = model.W.get_value().T # columns are dictionary elements
	#W *= np.sign( np.sum( np.sign(W), axis=1 ) + 0.1 )[:,None]
	#order = range(ns)
	#order.sort( key = lambda i: np.sum( abs(W[i]) ) )
	#W = W[order]
	Worig = unwhiten( W ) # mapped back to original space
	wpic = tile_raster_images(Worig, (sz, sz) , (sqrt_ns, sqrt_ns), tile_spacing = (1,1))
	imgW = ax.matshow(wpic, cmap = 'gray')

print "Run dictionary learning algorithm "
for i in range(3000):
	idx = np.random.randint( nt - nb ) # starting index of batch (batch is idx:idx+nb)
	var_exp, ssh = train_model( idx, lrate ) # train over one batch
	print var_exp #pp.plot(var_exp); pp.show() # print progress

	# Plot?
	if not plotW:
		pp.plot(ssh[:,0,:]); pp.show() #what's this??
	elif i % plot_ivl == 0:
		W = model.W.get_value().T # rows are dict elements
		#W *= np.sign( np.sum( np.sign(W), axis = 1 ) + 0.1 )[:,None]
		#order = range(ns)
		#order.sort( key = lambda i: np.sum( abs(W[i]) ) )
		#W = W[order]
		Worig = unwhiten( W )
		wpic = tile_raster_images(Worig, (sz, sz) , (sqrt_ns, sqrt_ns), tile_spacing = (1,1))
		imgW.set_data(wpic)
		imgW.autoscale()
		fig.canvas.draw()
		plot_title = 'iteration %d' % i
		pp.title( plot_title )
