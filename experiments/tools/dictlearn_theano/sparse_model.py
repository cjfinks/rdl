""" Contains class 'SparseModel' which learns a sparse dictionary """

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class SparseModel():
	def __init__(self, nx, ns, nb, W_init=None, 
			     xvar = 0.001, sparsity = 1.0 ):
		self.nx = nx # dimensionality of input data
		self.ns = ns # dimensionality of latent variable
		self.nb = nb # mini-batch size during training

		self.sigma = np.sqrt(xvar) # stdev of gaussian noise
		self.regularizer = sparsity
		
		# Initialize basis functions
		if W_init==None:
			W_init=np.random.randn(nx,ns)
		W_init=W_init/np.sqrt(np.sum(W_init**2,axis=0)) # constrain to unit length
		W_init=W_init.astype(np.float32)
		
		self.W=theano.shared(W_init)
		self.dW=theano.shared(W_init*0.0)
		self.params=[self.W]
		self.momentums=[self.dW] # each parameter gets its own momentum term
	
	def log_likelihood( self, s, x ):
		""" This function returns log P( x | s ) over a batch.
		i.e. the log-likelihood that the nb-by-ns block of samples of
		the latent variable came from the nb-by-nx block of inputs
		(Up to a constant...gaussian normalization coefficient ignored)"""

		recon = T.dot(s, self.W.T) # sparse reconstruction of input
		err = (recon - x)**2 # reconstruction error
		xterms = - T.sum( err, axis=1) / ( 2.0 * self.sigma**2 ) # error summed over pixels (assume noise indep. bw pixels)
		return xterms

	def log_posterior( self, s, x ):
		""" This function returns log P( s | x ) over a batch.
		i.e. the log-posterior that the nb-by-ns block of samples of
		the latent variable came from the nb-by-nx block of inputs """

		xterms = self.log_likelihood( s, x )
		sparse_prior = T.sqrt( s**2 + 1e-6 )
		sterms = - self.regularizer * T.sum( sparse_prior , axis=1 ) # sum over coeffs (assume indep. coeffs)
		return xterms + sterms

	def map_step( self, s, ds, x, stepsize ):
		""" Does a step of gradient ascent on the log-posterior and
		returns the new value of the objective (-ve log posterior) """

		# Use Nesterov momentum
		mu = 0.95
		s_test = s + mu*ds
		objective = - self.log_posterior( s_test, x) # objective
		s_grad = T.grad( T.sum( objective ), s_test )
		ds_new = mu * ds - stepsize * s_grad
		s_new = s + ds_new
		return s_new, ds_new, objective

	def map_estimate(self, x, nsteps = 200, stepsize = 1e-5):
		""" This function takes an nb-by-nx block of training samples and 
		returns the MAP estimate for each of them as an nb-by-ns matrix """

		s0 = T.dot( x, self.W ) * 0.1 # shitty initial guess
		ds0 = s0 * 0.0 # initialize ds
		[ss, dss, objs], updates = theano.scan(fn = self.map_step,
								               outputs_info = [s0, ds0, None],
								               non_sequences = [x, T.cast(stepsize,'float32')],
								               n_steps = nsteps)
		return ss, objs # list of values at every step of gradient ascent
		
	def update_params(self, x, lrate):
		# TODO: Umm...why don't we just do linear regression?
		""" This function takes a nb-by-nx block of training samples,
		finds the MAP estimate for each of them, and does a step of gradient
		ascent on the log-likelihood """

		# First find the sparse coefficients for the batch
		ss, objs = self.map_estimate(x, 200, 1e-5)
		s_map = ss[-1]
		logl = self.log_likelihood(s_map, x)
		objective = T.mean( -logl )

		# Now we optimize the dictionary given these coefficients
		mu=0.99 # The momentum coefficient for the model parameters (dictionary)
		updates=OrderedDict()

		# To compute the learning update for each of the parameters, loop over 
		# self.params and self.momentums, calculating the gradients etc.
		for i in range( len(self.params) ):
			param = self.params[i]
			mom = self.momentums[i]
			gparam = T.grad( objective, param, consider_constant = [s_map] )
			step = mu * mom - lrate * gparam
			new_param = param + step
			# TODO: trying this out...closed for lin reg. why not this?
			#new_param = T.dot( T.dot( T.inv( T.dot( s_map.T, s_map ) ), s_map.T ), x ) 

			if param==self.W:
				#Each column of W is constrained to be unit-length
				new_param=new_param / T.sqrt( T.sum( new_param**2, axis=0) ).dimshuffle('x',0)

			updates[param] = T.cast(new_param,'float32')
			updates[mom] = T.cast(step,'float32')

		variance_explained = (2.0 * self.sigma**2) * T.mean( -logl / T.sum(x**2,axis=1) )

		return variance_explained, ss, updates

	def decompose(self, x):
		# TODO: is this actually necessary...try just using map_estimate
		""" This function takes a nb-by-nx block of training samples,
		finds the MAP estimate for each of them, and does a step of gradient
		ascent on the log-likelihood """

		# First find the sparse coefficients for the batch
		ss, objs = self.map_estimate(x, 200, 1e-5)
		s_map = ss[-1]

		return s_map
	
	def fit( X, n_iter = 3000, batch_size = 3, lrate = 3e-2, PLOT = False ):
		print "Running dictionary learning algorithm "
		nt, nx = X.shape
		ns = self.ns

		data = theano.shared(X.astype(np.float32))
		self.__init__(nx, ns, nb, xvar = 0.001, sparsity = 100.0)
		x = T.fmatrix()
		lr = T.fscalar()
		idxs = T.lscalar()
		objs, ss, learn_updates = self.update_params(x, lr)
		train_model = theano.function([idxs, lr],
									[objs, ss],
									updates = learn_updates,
									givens = {x: data[ idxs:idxs + nb, : ]},
									allow_input_downcast = True )

		for i in xrange(n_iter):
			idx = np.random.randint( nt - nb ) # starting index of batch (batch is idx:idx+nb)
			var_exp, ssh = train_model( idx, lrate ) # train over one batch
			print var_exp #pp.plot(var_exp); pp.show() # print progress
	
		return self.W.get_value().T
