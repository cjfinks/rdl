import numpy as np
from sklearn.preprocessing import normalize
import os
import scipy.io
import matplotlib.pyplot as pp
import pytest

class Dictionary:
	def __init__(self, arr_or_string ):
		if isinstance( arr_or_string, str):
			self._load_dictionary( arr_or_string )
		else:
			self.matrix = np.matrix( arr_or_string )
	
	def get_coherence(self):
		""" Calculat mutual coherence of columns """
		corr = matrix.T * matrix
		return np.max( np.abs( corr - np.diag(np.diag(corr)) ) )
	
	def normalize( self, norm = 'l2' ):
		""" Sets columns of dictionary matrix to have unit norm """
		self.matrix = normalize( self.matrix, norm, axis=0 )
	
	def matshow( self, tile_atoms = True, image_shape=None, tile_shape=None ):
		""" Plots dictionary raw or as tiled images """
		if tile_atoms:
			img = self._tile_atoms(image_shape, tile_shape)
			pp.matshow( img )
		else:
			pp.matshow(self.matrix)

	def plot(self):
		""" Plots dictionary elements as functions over the integers """
		pass

	def _load_dictionary( self, name ):
		""" Loads one of the saved dictionaries in the data directory """
		datadir = os.path.abspath( os.path.dirname(__file__) )
		if name == 'alphabet':
			struct = scipy.io.loadmat( os.path.join(datadir, 'alphabet.mat' ) )[ 'letters' ]  # cols are rastered DCT8x8
			letters = ['A', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'X', 'V', 'Y']
			D = np.zeros((16,len(letters)))
			for j, letter in enumerate(letters):
				D[:,j] = struct[letter][0,0].ravel()
		else:
			filename = name
			name, ext = os.path.splitext( filename )
			if ext == '.mat':
				datadir = os.path.abspath( os.path.dirname(__file__) )
				D = scipy.io.loadmat( os.path.join(datadir, filename) )[ name ]  # cols are rastered DCT8x8
			elif ext == '.npy':
				D = np.load( filename )
		self.matrix = np.matrix( D )

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

			for tile_row in xrange(tile_shape[0]):
				for tile_col in xrange(tile_shape[1]):
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

""" Dictionary Tests """
def test_Dictionary_coherence():
	A = np.eye( 5, 5 )
	d = Dictionary( matrix = A )
	assert d.get_coherence() == 0
	
if __name__ == '__main__':
	pytest.main()
