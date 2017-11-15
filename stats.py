from __future__ import print_function, division
import numpy as np, astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os,sys
import psf_marginalise as pm
import scipy.ndimage as nd
def weighted_mean(cube):
	"""Find weighted mean of images in a cube where the weight is defined as 
	1/rms for each image.

	Parameters
	----------
	cube: (nImages,sz,sz) array
		  Data cube in the form of 3d array.
	
	Returns
	----------
	mean_im: (sz,sz) array
			 The weighted mean of the cube.
	"""
	weights = np.zeros(cube.shape[0])
	for ii in range(0,len(weights)):
		rms = np.sqrt(np.mean(cube[ii]**2))
		weights[ii] = 1./rms
	mean_im = np.zeros(cube[0].shape)
	for ii in range(0,len(weights)):
		mean_im+=weights[ii]*cube[ii]
	mean_im/=sum(weights)
	return mean_im
	
def bootstrap(cube,niter):
	"""Find standard deviation map by bootstrapping.

	Parameters
	----------
	cube: (nImages,sz,sz) array
		  Data cube in the form of 3d array.
	niter: int
		   The number of steps in the bootstrapping process

	Returns
	----------
	result: (sz,sz) array
			The standard deviation map.
	"""
	out_cube = np.zeros((niter,cube.shape[1],cube.shape[2]))
	n_ims = cube.shape[0]
	sz = cube.shape[1]
	for ii in range(0,niter):
		temp_cube = np.zeros((n_ims,sz,sz))
		for jj in range(0,n_ims):
			element = int(np.floor(n_ims*np.random.random()))
			temp_cube[jj] = cube[element]
		mean_im = weighted_mean(temp_cube)
		out_cube[ii] = mean_im
	result = np.std(out_cube,axis=0)
	return result