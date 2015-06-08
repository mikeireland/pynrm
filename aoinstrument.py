# -*- coding: utf-8 -*-
"""Useful utilities that are not telescope-dependent.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pyfits
import csv

class AOInstrument:
 """The AOInstrument Class
 """
 #A blank dictionary on startup.
 csv_dict = dict()
 #Blank reduction, cube. and data directories on startup.
 rdir = ''
 ddir = ''
 cdir = ''
 
 def read_summary_csv(self, filename='datainfo.csv',ddir=''):
	"""Read the data from local file into a csv_dict structure.
	
	Notes
	-----
	At the moment, all data types are strings. It would work better if
	the second line of the CSV file was the data type.
	"""
	#Allow over-riding default data directory.
	if (ddir == ''):
		ddir = self.ddir
	try:
		f = open(ddir + filename)
	except:
		print "Error: file doesn't exist " + ddir + filename
		raise UserWarning
	r = csv.DictReader(f, delimiter=',')
	#Read first line to initiate the dictionary
	line = r.next()
	d = dict()
	for k in line.keys():
		d[k] = [line[k]]
	#Read the rest of the file
	for line in r:
		for k in line.keys():
			d[k].append(line[k])
	#Convert to numpy arrays
	for k in line.keys():
		d[k] = np.array(d[k])
	f.close()
	self.csv_dict = d
 
 def make_all_darks(self, ddir='', rdir=''):
 	"""Make all darks in a current directory. This skeleton routine assumes that
 	keywords "SHRNAME", "NAXIS1" and "NAXIS2" exist.
 	"""
 	#Allow over-riding default reduction and data directories.
	if (rdir == ''):
		rdir = self.rdir
	if (ddir == ''):
		ddir = self.ddir
 	darks = np.where(np.array(n2.csv_dict['SHRNAME']) == 'closed')[0]
 	#Now we need to find unique values of the following:
 	#NAXIS1, NAXIS2 (plus for nirc2... ITIME, COADDS, MULTISAM)
 	codes = []
 	for d in darks:
 		codes.append(self.csv_dict['NAXIS1'][d] + self.csv_dict['NAXIS2'][d])
 	codes = np.array(codes)
 	#For each unique code, find all dark files and call make_dark.
 	for c in np.unique(codes):
 		w = np.where(codes == c)[0]
 		#Only bother if there are at least 3 files.
 		if (len(w) >= 3):
	 		files = [ddir + self.csv_dict['FILENAME'][darks[ww]] for ww in w]
	 		self.make_dark(files, rdir=rdir)
 	
 def make_dark(self,in_files, out_file='dark.fits', rdir=''):
	"""This is a basic method to make a dark from several files. It is
	generally expected to be over-ridden in derived classes.
	
	Parameters
	----------
	in_files : array_like (dtype=string). A list if input filenames.
	out_file: string
		The file to write to.
	"""
	#Allow over-riding default reduction directory.
	if (rdir == ''):
		rdir = self.rdir
	nf = len(in_files)
	in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
	adark = in_fits[0].data
	in_fits.close()
	s = adark.shape
	darks = np.zeros((nf,s[0],s[1]))
	for i in range(nf):
		#Read in the data, linearizing as a matter of principle, and also because
		#this routine is used for 
		in_fits = pyfits.open(in_files[i], ignore_missing_end=True)
		adark = in_fits[0].data
		in_fits.close()
		darks[i,:,:] = adark
	med_dark = np.median(darks, axis=0)
	pyfits.writeto(rdir + out_file,med_dark)
	
 def info_from_header(self, h):
	"""Find important information from the fits header and store in a common format
	Prototype function only - to be over-ridden in derived classes.
	
	Parameters
	----------
	h:	The fits header
	
	Returns
	-------
	(dark_file, flat_file, filter, wave, rad_pixel)
	"""
	try: filter = h['FILTER']
	except:
		print "No FILTER in header"
	try: wave = h['WAVE']
	except:
		print "No WAVE in header"	
	try: rad_pix = h['RAD_PIX']
	except:
		print "No RAD_PIX in header"
	try: targname = h['TARGET']
	except:
		print "No TARGET in header"
	return {'dark_file':'dark.fits', 'flat_file':'flat.fits', 'filter':filter, 
		'wave':wave, 'rad_pixel':rad_pixel,'targname':targname,'pupil_type':'circ','pupil_params':dict()}
		
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
	
 def make_flat(self,in_files, rdir='', out_file='', dark_file=''):
	"""Create a flat frame and save to a fits file, 
	with an attached bad pixel map as the first fits extension.

	Parameters
	----------
	in_files : array_like (dtype=string). A list if input filenames.
	dark_file: string
		The dark file, previously created with make_dark
	out_file: string
		The file to write to
	rdir: Over-writing the default reduction directory.
	
	Returns
	-------
	Nothing.
	"""
	#Allow over-riding default reduction directory.
	if (rdir == ''):
		rdir = self.rdir
	#Create a default flat filename from the input files
	if (out_file == ''):
		in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
		h = in_fits[0].header
		hinfo = self.info_from_header(h)
		out_file = hinfo['flat_file']
	#We use the make_dark function to average our flats. NB we don't destripe.
	self.make_dark(in_files, rdir=rdir, out_file=out_file, subtract_median=False, destripe=False, med_threshold=15.0)
	#Now extract the key parts.
	h = pyfits.getheader(rdir + out_file)
	if (dark_file ==''):
		dark_file=self.get_dark_filename(h)
	flat = pyfits.getdata(rdir + out_file,0) - pyfits.getdata(rdir + dark_file,0)
	bad = np.logical_or(pyfits.getdata(rdir + out_file,1),pyfits.getdata(rdir + dark_file,1))
	flat[np.where(bad)] = np.median(flat)
	flat /= np.median(flat)
	#Write this to a file
	hl = pyfits.HDUList()
	hl.append(pyfits.ImageHDU(flat,h))
	hl.append(pyfits.ImageHDU(np.uint8(bad)))
	hl.writeto(rdir + out_file,clobber=True)
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

 def hexagon(self, dim, width):
		"""This function creates a hexagon.
	
		Parameters
		----------
		dim: int
			Size of the 2D array
		width: int
			flat-to-flat width of the hexagon
		
		Returns
		-------
		pupil: float array (sz,sz)
			2D array hexagonal pupil mask
		"""
		x = np.arange(dim)-dim/2.0
		xy = np.meshgrid(x,x)
		xx = xy[1]
		yy = xy[0]
		w = np.where( (yy < width/2) * (yy > (-width/2)) * \
		 (yy < (width-np.sqrt(3)*xx)) * (yy > (-width+np.sqrt(3)*xx)) * \
		 (yy < (width+np.sqrt(3)*xx)) * (yy > (-width-np.sqrt(3)*xx)))
		hex = np.zeros((dim,dim))
		hex[w]=1.0
		return hex

	
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

 def shift_and_ft(self,im, ftpix=()):
	"""Sub-pixel shift an image to the origin and Fourier-transform it

	Parameters
	----------
	im: (ny,nx) float array
	ftpix: optional ( (nphi) array, (nphi) array) of Fourier sampling points. 
		If included, the mean square Fourier phase will be minimised.

	Returns
	----------
	ftim: (ny,nx/2+1)  complex array
	"""
	ny = im.shape[0]
	nx = im.shape[1]
	if len(ftpix)==0:
		im = self.regrid_fft(im,(3*ny,3*nx))
		shifts = np.unravel_index(im.argmax(), im.shape)
		im = np.roll(np.roll(im,-shifts[0]+1,axis=0),-shifts[1]+1,axis=1)
		im = self.rebin(im,(ny,nx))
		ftim = np.fft.rfft2(im)
	else:
		shifts = np.unravel_index(im.argmax(), im.shape)
		im = np.roll(np.roll(im,-shifts[0]+1,axis=0),-shifts[1]+1,axis=1)
		ftim = np.fft.rfft2(im)
		#Project onto the phase ramp in each direction...
		arg_ftpix = np.angle(ftim[ftpix])
		#Compute phase in radians per Fourier pixel
		vcoord = ((ftpix[0] + ny/2) % ny)- ny/2
		ucoord = ftpix[1]
		vphase = np.sum(arg_ftpix * vcoord)/np.sum(vcoord**2)
		uphase = np.sum(arg_ftpix * ucoord)/np.sum(ucoord**2)
		uv = np.meshgrid(np.arange(nx/2 + 1), ((np.arange(ny) + ny/2) % ny) - ny/2 )
		ftim = ftim*np.exp(-1j * uv[0] * uphase - 1j*uv[1]*vphase)
	return ftim