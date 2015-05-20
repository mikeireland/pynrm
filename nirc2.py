import astropy.io.fits as pyfits
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from pysam_util import AOInstrument 

class NIRC2(AOInstrument):
 #A global definition, for error-checking downstream
 instrument = 'NIRC2'	
	
 def destripe_nirc2(self,im, subtract_edge=True, subtract_median=False):
	"""Destripe an image from the NIRC2 camera.
	
	The algorithm is:
	1) Subtract the mode from each quadrant.
	2) For each pixel, find the 24 pixels in other quadrants corresponding to 
	reflections about the chip centre.
	3) Subtract the median of these pixels.
	
	Parameters
	----------
	im: array_like
		The input image.
	subtract_median: bool, optional
		Whether or not to subtract the median from each quadrant.	
	subtract_edge: bool, optional
		Whether or not to adjust the means of each quadrant by the edge pixels.
	
	Returns
	-------
	im: array_like
		The corrected image.
	"""
	s = im.shape
	quads = [im[0:s[0]/2,0:s[0]/2],im[s[0]:s[0]/2-1:-1,0:s[0]/2],
			 im[0:s[0]/2,s[0]:s[0]/2-1:-1],im[s[0]:s[0]/2-1:-1,s[0]:s[0]/2-1:-1]]
	quads = np.array(quads, dtype='float')
	#Work through the quadrants, modifying based on the edges.
	if subtract_edge:
		quads[1] += np.median(quads[3][:,s[0]/2-8:s[0]/2])- np.median(quads[1][:,s[0]/2-8:s[0]/2]) 
		quads[2] += np.median(quads[3][s[0]/2-8:s[0]/2,:])- np.median(quads[2][s[0]/2-8:s[0]/2,:])  
		delta = 0.5*(np.median(quads[3][s[0]/2-8:s[0]/2,:]) + np.median(quads[3][:,s[0]/2-8:s[0]/2])
			   - np.median(quads[0][s[0]/2-8:s[0]/2,:]) - np.median(quads[0][:,s[0]/2-8:s[0]/2]))
		quads[0] += delta
	#Subtract the background
	if subtract_median:
		print "Subtracting Median..."
		med = np.median(quads)
		dispersion = np.median(np.abs(quads - med))
		MED_DIFF_MULTIPLIER = 4.0
		goodpix = np.where(np.abs(quads - med) < MED_DIFF_MULTIPLIER*dispersion)
		med = np.median(quads[goodpix])
		quads -= med
	quads = quads.reshape((4,s[0]/2,s[1]/16,8))
	stripes = quads.copy()
	for i in range(4):
		for j in range(s[0]/2): #The -1 on  line is because of artifacts
			for k in range(s[0]/16):
				pix = np.array([stripes[(i+1)%4,j,k,:],stripes[(i+2)%4,j,k,:],stripes[(i+3)%4,j,k,:]])
				quads[i,j,k,:] -= np.median(pix)
	quads = quads.reshape((4,s[0]/2,s[0]/2))
	im[0:s[0]/2,0:s[0]/2] = quads[0]
	im[s[0]:s[0]/2-1:-1,0:s[0]/2] = quads[1]
	im[0:s[0]/2,s[0]:s[0]/2-1:-1] = quads[2]
	im[s[0]:s[0]/2-1:-1,s[0]:s[0]/2-1:-1] = quads[3]
	return im
	
 def make_dark(self,in_files, out_file, subtract_median=True, destripe=True, med_threshold=15.0):
	"""Create a dark frame and save to a fits file, 
	with an attached bad pixel map as the first fits extension.

	Parameters
	----------
	in_files : array_like (dtype=string). A list if input filenames.
	out_file: string
		The file to write to.
	subtract_median: bool, optional
		Whether or not to subtract the median from each frame (or quadrants)
	destripe: bool, optional
		Whether or not to destripe the images.
	med_threshold: float, optional
		The threshold for pixels to be considered bad if their absolute
		value differs by more than this multiple of the median difference
		of pixel values from the median.
	
	Returns
	-------
	Nothing.
	"""
	VAR_THRESHOLD = 10.0
	nf = len(in_files)
	if (nf < 3):
		print "At least 3 dark files sre needed for reliable statistics"
		raise UserWarning
	# Read in the first dark to check the dimensions.
	in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
	h = in_fits.header
	instname = ''
	try: instname=h['CURRINST']
	except:
		print "Unknown Header Type"
	s = in_fits.data.shape
	in_fits.close()
	darks = np.zeros((nf,s[0],s[1]))
	plt.clf()
	for i in range(nf):
		#Read in the data, linearizing as a matter of principle, and also because
		#this routine is used for 
		adark = self.linearize_nirc2(in_files[i])
		if (instname == 'NIRC2' and destripe):
			adark = self.destripe_nirc2(adark, subtract_median=subtract_median)
		if (subtract_median):
			plt.imshow(np.minimum(adark,1e2))
		else:
			plt.imshow(adark)
			print "Median: " + str(np.median(adark))
		plt.draw()
		darks[i,:,:] = adark
	#Now look for weird pixels. 
	med_dark = np.median(darks, axis=0)
	max_dark = np.max(darks, axis=0)
	var_dark = np.zeros((s[0],s[1]))
	for i in range(nf):
		var_dark += (darks[i,:,:] - med_dark)**2
	var_dark -= (max_dark - med_dark)**2
	var_dark /= nf-2
	med_diff = np.median(np.abs(med_dark - np.median(med_dark)))
	print "Median difference: " + str(med_diff)
	med_var_diff = np.median(np.abs(var_dark - np.median(var_dark)))
	bad_med = np.abs(med_dark - np.median(med_dark)) > med_threshold*med_diff
	bad_var = np.abs(var_dark) > np.median(var_dark) + VAR_THRESHOLD*med_var_diff
	print "Pixels with bad mean: " + str(np.sum(bad_med))
	print "Pixels with bad var: " + str(np.sum(bad_var))
	bad = np.logical_or(bad_med, bad_var)
	med_dark[bad] = 0.0
	#Copy the original header to the dark file.
	hl = pyfits.HDUList()
	hl.append(pyfits.ImageHDU(med_dark,h))
	hl.append(pyfits.ImageHDU(np.uint8(bad)))
	hl.writeto(out_file,clobber=True)
	plt.figure(1)
	plt.imshow(med_dark,cmap=cm.gray, interpolation='nearest')
	plt.title('Median Frame')
	plt.figure(2)
	plt.imshow(bad,cmap=cm.gray, interpolation='nearest')
	plt.title('Bad Pixels')
	plt.draw()

 def linearize_nirc2(self,in_file, out_file=''):
	"""Procedure to linearize NIRC2 (treated as a single detector).
    Run on all images nominally before running anything else.
	Originally from IDL code by Stan Metchev with Adam Kraus modifications.

	Parameters
	----------
	in_file: string
		The input fits file
	out_file: string, optional
		An output fits file
		
	Returns
	-------
	im: (N,N) array
		The linearized image.
	
	Notes
	-----
	This procedure takes COADDS into account, but does not take subreads into account.
	i.e. in MCDS sampling, there is slightly more nonlinearity than accounted for, 
	because the detector is more saturated by the time the last read is made.
	"""

	coeff = np.array([1.001,-6.9e-6,-0.70e-10])
	#Get key parameters from the header and get the data
	in_fits = pyfits.open(in_file)
	z = in_fits.header
	fitsarr = in_fits.data
	in_fits.close()
	#See if we've already updated this header
	try:
		lindate = z['LINHIST']
	except:
		print 'Linearizing: ',in_file
		xsub=z['NAXIS1']
		ysub=z['NAXIS2']
		norm=np.array((xsub,ysub))
		coadds = z['COADDS']
		norm = coeff[0]+coeff[1]*fitsarr/coadds+coeff[2]*(fitsarr/coadds)**2
		fitsarr = fitsarr / norm
		z['LINHIST'] = time.asctime()
		if (len(out_file) > 0):
			hl = pyfits.HDUList()
			hl.append(pyfits.ImageHDU(fitsarr,z))
			hl.writeto(out_file,clobber=True)
	return fitsarr
	
 def clean_no_dither(self, in_files, fmask_file='',dark_file='', flat_file='', fmask=[],\
	subarr=128,extra_threshold=7,out_file='',median_cut=0.7):
	"""Clean a series of fits files, including: applying the dark, flat, 
	removing bad pixels and cosmic rays.

	Parameters
	----------
	in_files : array_like (dtype=string). 
		A list if input filenames.
	dark_file: string
		The dark file, previously created with make_dark
	flat_file: string
		The flat file, previously created with make_flat
	ftpix: ( (N) array, (N) array)
		The pixels in the data's Fourier Transform that include all non-zero
		values (created using pupil_sampling)
	subarr: int, optional
		The width of the subarray.
	extra_threshold: float, optional
		A threshold for identifying additional bad pixels and cosmic rays.
	outfile: string,optional
		A filename to save the cube as, including the header of the first
		fits file in the cube plus extra information.
	
	
	Returns
	-------
	The cube of cleaned frames.
	"""
	#If an fmask_file is given, get details from this:
	if (len(fmask_file) > 0):
		fmask = pyfits.getdata(fmask_file,1)
		h = pyfits.getheader(fmask_file)
		if (len(dark_file) == 0):
			dark_file = h['DARK']
		if (len(flat_file) == 0):
			flat_file = h['FLAT']
	#If we still don't have an fmask, something is wrong
	if (len(fmask) == 0):
		print "Error: A Fourier plane mask is necessary... (input or file)"
		raise UserWarning
	#Allocate data for the cube
	nf = len(in_files)
	cube = np.zeros((nf,subarr,subarr))
	#Decide on the image size from the first file. !!! are x and y around the right way?
	in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
	h = in_fits.header
	in_fits.close()
	szx = h['NAXIS1']
	szy = h['NAXIS2']
	#Estimate the expected readout noise directly from the header.
	if h['SAMPMODE'] == 2:
		multisam = 1
	else:
		multisam = h['MULTISAM']
	#The next line comes from the NIRC2 manual home page.
	rnoise = 50.0/4.0/np.sqrt(multisam)
	gain = 4.0
	#Chop out the appropriate part of the flat, dark, bad arrays
	if len(flat_file) > 0:
		flat = pyfits.getdata(flat_file,0)
		flat = flat[flat.shape[0]/2 - szy/2:flat.shape[0]/2 + szy/2,flat.shape[1]/2 - szx/2:flat.shape[1]/2 + szx/2]
		bad = pyfits.getdata(flat_file,1)
		bad = bad[bad.shape[0]/2 - szy/2:bad.shape[0]/2 + szy/2,bad.shape[1]/2 - szx/2:bad.shape[1]/2 + szx/2]
	else:
		flat = np.ones((szy,szx))
		bad = np.zeros((szy,szx))
	if len(dark_file) > 0:
		dark = pyfits.getdata(dark_file,0)
		if (szy != dark.shape[0]):
			print("Warning - Dark is of the wrong shape!")
			dark = dark[dark.shape[0]/2 - szy/2:dark.shape[0]/2 + szy/2, \
					   dark.shape[1]/2 - szx/2:dark.shape[1]/2 + szx/2]
	else:
		dark = np.zeros((szy,szx))
	#Go through the files, cleaning them one at a time
	xpeaks = np.zeros(nf)
	ypeaks = np.zeros(nf)
	maxs = np.zeros(nf)
	pas = np.zeros(nf)
	for i in range(nf):
		#First, find the position angles from the header keywords. NB this is the Sky-PA of chip vertical.
		in_fits = pyfits.open(in_files[i], ignore_missing_end=True)
		h = in_fits.header
		in_fits.close()
		pas[i]=360.+h['PARANG']+h['ROTPPOSN']-h['EL']-h['INSTANGL'] 
		#Read in the image - making a nonlinearity correction
		im = self.linearize_nirc2(in_files[i])
		#Destripe, then clean the data using the dark and the flat
		im = self.destripe_nirc2(im)
		im = im/flat #(im - dark)/flat
		#Find the star... 
		im *= 1.0 - bad
		im_filt = nd.filters.median_filter(im,size=5)
		max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
		print("Maximum x,y: " + str(max_ix[1])+', '+ str(max_ix[0]))
		xpeaks[i] = max_ix[1]
		ypeaks[i] = max_ix[0]
		maxs[i] = im[max_ix[0],max_ix[1]]
		subim = np.roll(np.roll(im,subarr/2-max_ix[0],axis=0),subarr/2-max_ix[1],axis=1)
		subim = subim[0:subarr,0:subarr]
		subbad = np.roll(np.roll(bad,subarr/2-max_ix[0],axis=0),subarr/2-max_ix[1],axis=1)
		subbad = subbad[0:subarr,0:subarr]
		new_bad = subbad.copy()	
		subim[np.where(subbad)] = 0
		plt.clf()
		plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
		plt.draw()
		for ntry in range(1,15):
			#Correct the known bad pixels
			self.fix_bad_pixels(subim,new_bad,fmask)
			#Search for more bad pixels. Lets use a Fourier technique here...
			extra_bad_ft = np.fft.rfft2(subim)*fmask
			extra_bad = np.real(np.fft.irfft2(extra_bad_ft))
			mim = nd.filters.median_filter(subim,size=5)
			#NB The next line *should* take experimentally determined readout noise into account !!!
			extra_bad = np.abs(extra_bad/np.sqrt(np.maximum(gain*mim + rnoise**2,rnoise**2)))
			unsharp_masked = extra_bad-nd.filters.median_filter(extra_bad,size=3)
			current_threshold = np.max([0.3*np.max(unsharp_masked[new_bad == 0]), extra_threshold*np.median(extra_bad)])
			extra_bad = unsharp_masked > current_threshold
#			plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
#			plt.draw()
#			dummy = plt.ginput(1)
			n_extra_bad = np.sum(extra_bad)
#			if (n_extra_bad < 2):
#				unsharp_masked = extra_bad-nd.filters.median_filter(extra_bad,size=5)
#				extra_bad = unsharp_masked > extra_threshold*np.median(extra_bad)
#				n_extra_bad = np.sum(extra_bad)
#				print "Widened unsharp mask window..."
			print(str(n_extra_bad)+" extra bad pixels or cosmic rays identified. Attempt: "+str(ntry))
			subbad += extra_bad
			if (ntry == 1):
				new_bad = extra_bad
			else:
				new_bad += extra_bad
				new_bad = extra_bad>0
			if (n_extra_bad == 0):
				break
		
		#Now re-correct both the known and new bad pixels at once.
		self.fix_bad_pixels(subim,subbad,fmask)
		plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
		plt.draw()
		
		#Save the data and move on!
		cube[i,:,:]=subim
	#Fine bad frames based on low peak count.
	good = np.where(maxs > median_cut*np.median(maxs))
	good = good[0]
	if (len(good) < nf):
		print nf-len(good), "  frames rejected due to low peak counts."
	cube = cube[good,:,:]
	nf = np.shape(cube)[0]
	#If a filename is given, save the file.
	if (len(out_file) > 0):
		hl = pyfits.HDUList()
		h['RNOISE'] = rnoise
		h['SZX'] = szx
		h['SZY'] = szy
		for i in range(nf):
			h['HISTORY'] = 'Input: ' + in_files[i]
		hl.append(pyfits.ImageHDU(cube,h))
		#Add in the original peak pixel values, forming the image centers in the cube.
		#(this is needed for e.g. undistortion)
		col1 = pyfits.Column(name='xpeak', format='E', array=xpeaks)
		col2 = pyfits.Column(name='ypeak', format='E', array=ypeaks)
		col3 = pyfits.Column(name='pa', format='E', array=pas)
		col4 = pyfits.Column(name='max', format='E', array=maxs)
		cols = pyfits.ColDefs([col1, col2,col3,col4])
		hl.append(pyfits.new_table(cols))
		hl.writeto(out_file,clobber=True)
	return cube
	

if(0):
	n2 = NIRC2()

#Testing destripe only.
if (0):
	f = pyfits.open(file, ignore_missing_end=True)
	im = f.data
	f.close()
	plt.imshow(np.minimum(np.maximum(n2.destripe_nirc2(im),-50),50), interpolation='nearest', cmap=cm.gray)
	f.close()

#Testing darks and flats.
if (0):
	dir =  '/Users/mireland/data/nirc2/131115/n'
	extn = '.fits.gz'
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(40,45)]
	#files.extend([(dir + '{0:04d}' + extn).format(i) for i in range(56,61)])
	n2.make_dark(files,'dark.fits')
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(29,40)]
	n2.make_flat(files,'dark.fits','flat.fits')