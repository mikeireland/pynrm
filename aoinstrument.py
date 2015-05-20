# -*- coding: utf-8 -*-
"""Useful utilities that are not telescope-dependent.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pyfits

class AOInstrument:
 def make_dark(self,in_files, out_file):
	"""This is a basic method to make a dark from several files. It is
	generally expected to be over-ridden in derived classes.
	
	Parameters
	----------
	in_files : array_like (dtype=string). A list if input filenames.
	out_file: string
		The file to write to.
	"""
	nf = len(in_files)
	in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
	adark = in_fits.data
	in_fits.close()
	s = adark.shape
	darks = np.zeros((nf,s[0],s[1]))
	for i in range(nf):
		#Read in the data, linearizing as a matter of principle, and also because
		#this routine is used for 
		in_fits = pyfits.open(in_files[i], ignore_missing_end=True)
		adark = in_fits.data
		in_fits.close()
		darks[i,:,:] = adark
	med_dark = np.median(darks, axis=0)
	pyfits.writeto(out_file,med_dark)
	
		
 def mod2pi(self,angle):
	""" Convert an angle to the range (-pi,pi)
	
	Parameters
	----------
	angle: float
		input angle
	
	Returns
	-------
	angle: float
		output angle after the mod2pi operation
	"""
	return np.remainder(angle + np.pi,2*np.pi) - np.pi
	
 def make_flat(self,in_files, dark_file, out_file):
	"""Create a flat frame and save to a fits file, 
	with an attached bad pixel map as the first fits extension.

	Parameters
	----------
	in_files : array_like (dtype=string). A list if input filenames.
	dark_file: string
		The dark file, previously created with make_dark
	out_file: string
		The file to write to
	
	Returns
	-------
	Nothing.
	"""
	#We use the make_dark function to average our flats
	self.make_dark(in_files, out_file, subtract_median=False, destripe=False, med_threshold=15.0)
	#Now extract the key parts.
	h = pyfits.getheader(out_file)
	flat = pyfits.getdata(out_file,0) - pyfits.getdata(dark_file,0)
	bad = np.logical_or(pyfits.getdata(out_file,1),pyfits.getdata(dark_file,1))
	flat[np.where(bad)] = np.median(flat)
	flat /= np.median(flat)
	#Write this to a file
	hl = pyfits.HDUList()
	hl.append(pyfits.ImageHDU(flat,h))
	hl.append(pyfits.ImageHDU(np.uint8(bad)))
	hl.writeto(out_file,clobber=True)
	plt.figure(1)
	plt.imshow(flat, cmap=cm.gray, interpolation='nearest')
	plt.title('Flat')
	
 def fix_bad_pixels(self,im,bad,fmask):
	"""Fix the bad pixels, using a Fourier technique that adapts to the
	sampling of each particular pupil/filter combination.

	Parameters
	----------
	im : (N,N) array (dtype=float) 
		An image, already chopped to the subarr x subarr size.
	bad: (N,N) array (dtype=int)
		A bad pixel map
	fmask: (N,N) array (dtype=int)
		A mask containing the region in the Fourier plane where there is
		no expected signal.
		
	Returns
	-------
	The image with bad pixel values optimally corrected.
	
	"""
	wft = np.where(fmask)
	w = np.where(bad)
	badmat = np.zeros((len(w[0]),len(wft[0])),dtype='complex')
	print("Bad matrix shape: " + str(badmat.shape))
	xy = np.meshgrid(2*np.pi*np.arange(im.shape[1]/2 + 1)/float(im.shape[1]),
					2*np.pi*np.arange(im.shape[0])/float(im.shape[0]))
	for i in range(len(w[0])):
		# Avoiding the fft is marginally faster here...
		bft = np.exp(-1j*(w[0][i]*xy[1] + w[1][i]*xy[0]))
		badmat[i,:] = bft[wft]
	#A dodgy pseudo-inverse that needs an "invert" is faster than the la.pinv function
	#Unless things are really screwed, the matrix shouldn't be singular.
	hb = np.transpose(np.conj(badmat))
	ibadmat = np.dot(hb,np.linalg.inv(np.dot(badmat,hb)))
	#Now find the image Fourier transform on the "zero" region in the Fourier plane
	#To minimise numerical errors, set the bad pixels to zero at the start.
	im[w]=0	
	ftimz = (np.fft.rfft2(im))[wft]
	# Now compute the bad pixel corrections. (NB a sanity check here is
	# that the imaginary part really is 0)
	addit = -np.real(np.dot(ftimz,ibadmat))
#	plt.clf()
#	plt.plot(np.real(np.dot(ftimz,ibadmat)), np.imag(np.dot(ftimz,ibadmat)))
#	raise UserWarning
	im[w] += addit
	return im

 def regrid_fft(self,im,new_shape, fmask=[]):
	"""Regrid onto a larger number of pixels using an fft. This is optimal
	for Nyquist sampled data.
	
	Parameters
	----------
	im: array
		The input image.
	new_shape: (new_y,new_x)
		The new shape
	
	Notes
	------
	TODO: This should work with an arbitrary number of dimensions
	"""
	ftim = np.fft.rfft2(im)
	if len(fmask) > 0:
		ftim[np.where(fmask)] = 0
	new_ftim = np.zeros((new_shape[0], new_shape[1]/2 + 1),dtype='complex')
	new_ftim[0:ftim.shape[0]/2,0:ftim.shape[1]] = \
		ftim[0:ftim.shape[0]/2,0:ftim.shape[1]]
	new_ftim[new_shape[0]-ftim.shape[0]/2:,0:ftim.shape[1]] = \
		ftim[ftim.shape[0]/2:,0:ftim.shape[1]]
	return np.fft.irfft2(new_ftim)
	
 def rebin(self,a, shape):
	"""Re-bins an image to a new (smaller) image with summing	
	
	Originally from:
	http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
	
	Parameters
	----------
	a: array
		Input image
	shape: (xshape,yshape)
		New shape
	"""
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	return a.reshape(sh).sum(-1).sum(1)

 def shift_and_ft(self,im):
	"""Sub-pixel shift an image to the origin and Fourier-transform it

	Parameters
	----------
	im: (ny,nx) float array

	Returns
	----------
	ftim: (ny,nx/2+1)  complex array
	"""
	ny = im.shape[0]
	nx = im.shape[1]
	im = self.regrid_fft(im,(3*ny,3*nx))
	shifts = np.unravel_index(im.argmax(), im.shape)
	im = np.roll(np.roll(im,-shifts[0]+1,axis=0),-shifts[1]+1,axis=1)
	im = self.rebin(im,(ny,nx))
	return np.fft.rfft2(im)