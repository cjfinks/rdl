""" Contains classes for manipulating data """

import numpy as np
from sklearn.preprocessing import normalize
import os
import scipy.io
import matplotlib.pyplot as pp
#import pytest
import itertools
import random

class Dictionary:
	""" Wraps a numpy matrix (stored in self.matrix) and contains methods for loading a particular dictionary from the list below, calculating relevant functions thereof, and plotting the dictionary matrix.

		Stored dictionaries, saved in the 'dictionaries' directory, are:
	    - dct8x8: 8x8 discrete cosine transform basis 
		- dct12x12: 12x12 discrete cosine transform basis
		- alphabet: letters from the alphabet
	"""
	def __init__(self, array ):
		self.matrix = np.matrix( array )
	
	def phi(self, k):
		""" Returns phi_k of the dictionary matrix """
		pass

	def restricted_lower_bound(self, k):
		""" Returns the k-restricted lower-bound of the dictionary matrix """
		pass

	def coherence(self):
		""" Calculate mutual coherence of columns """
		corr = matrix.T * matrix
		mu = np.max( np.abs( corr - np.diag(np.diag(corr)) ) )
		return mu
	
	def normalize( self, norm = 'l2' ):
		""" Sets columns of dictionary matrix to have unit norm """
		self.matrix = normalize( self.matrix, norm, axis=0 )
		return self # TODO: check this accomplishes normalizedDictionary = Dictionary().normalize()
	
	def matshow( self, tile_atoms = True, image_shape=None, tile_shape=None ):
		""" Plots dictionary raw or as tiled images """
		if tile_atoms:
			img = self._tile_atoms(image_shape, tile_shape)
			fig = pp.matshow( img )
		else:
			fig = pp.matshow(self.matrix)
		return fig

	def plot(self):
		""" Plots dictionary elements as functions over the integers """
		pass

	def _tile_atoms( self, image_shape=None, tile_shape=None ):
		""" Displays the dictionary as tiled images """
		if image_shape is None:
			# Assume image is square
			shp = int( np.sqrt( self.matrix.shape[0] ) )
			image_shape = (shp,shp)

		if tile_shape is None:
			# Calculate tile_shape closest to square root
			m = self.matrix.shape[1]
			i = np.floor( np.sqrt(m) )
			while m % i != 0:
				i-=1
			n_rows = int(i)
			n_cols = int(m / n_rows)
			tile_shape = (n_rows, n_cols)
		
		def tile_raster_images(X, img_shape , tile_shape, tile_spacing = (1,1), output_pixel_vals = False ):
			out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
								in zip(img_shape, tile_shape, tile_spacing)]
			H, W = img_shape
			Hs, Ws = tile_spacing

			# generate a matrix to store the output
			dt = X.dtype
			if output_pixel_vals:
				dt = 'uint8'
			out_array = np.zeros(out_shape, dtype=dt)

			for tile_row in range(tile_shape[0]):
				for tile_col in range(tile_shape[1]):
					if tile_row * tile_shape[1] + tile_col < X.shape[0]:
						this_x = X[tile_row * tile_shape[1] + tile_col]
						this_img = this_x.reshape(img_shape)
						# add the slice to the corresponding position in the output array
						c = 1
						if output_pixel_vals:
							c = 255
						out_array[
							tile_row * (H + Hs): tile_row * (H + Hs) + H,
							tile_col * (W + Ws): tile_col * (W + Ws) + W
							] = this_img * c
			return out_array

		img = tile_raster_images( np.array( self.matrix.T ), image_shape, tile_shape )
		return img

class DataTable:
	def __init__( self, dataTable, batchSize = None ):
		if isinstance( dataTable, DataTable ):
			self.all = dataTable.all
		elif isinstance( dataTable, np.ndarray ) or isinstance( dataTable, np.matrix ):
			self.all = np.array(dataTable)

		if batchSize is None:
			self.batchSize = self.all.shape[0]

		self.batch = self.__batch_iterator()

	def __batch_iterator( self ):
		""" Returns a batch of data """
		numberOfSamples = self.all.shape[0]
		indexOfLastSample = numberOfSamples - 1
		iter = itertools.cycle( range(0, numberOfSamples, self.batchSize) )
		while True:
			firstIndexOfBatch = iter.next()
			lastIndexOfBatch = firstIndexOfBatch + self.batchSize - 1
			if lastIndexOfBatch > indexOfLastSample:
				overflow = np.array( random.sample( self.all[:firstIndexOfBatch], lastIndexOfBatch - indexOfLastSample ) )
				yield np.vstack( ( self.all[firstIndexOfBatch:], overflow ) )
			else:
				yield self.all[firstIndexOfBatch:lastIndexOfBatch+1]

""" Dictionary Tests """
def test_Dictionary_coherence():
	A = np.eye( 5, 5 )
	d = Dictionary( matrix = A )
	assert d.get_coherence() == 0
	
if __name__ == '__main__':
	pytest.main()
