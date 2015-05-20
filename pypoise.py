"""The main module to implement the Phase Observationally Independent of
Systematic Errors (POISE) algorithm.

"""

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from pysam_nirc2 import NIRC2

#By default, just assume the instrument is NIRC2. I *think* this can be
#changed by:
# import pypoise
# pypoise.aoinst = NIRC2()
#... however, it is much neater if pypoise becomes a class, which has to be
#passed an AOInstrument derived class in its constructor.
aoinst = NIRC2()
	
def pmask_to_ft(pmask):
	"""Create Fourier sampling for kerphase based on a pupil mask.
	
	Parameters
	----------
	pmask: (subarr, subarr) array
	
	Returns
	-------
	RR: (subarr,subarr) array
		Two-dimensional redundancy array, computing how many baselines
		contribute to each Fourier pixel.
	AA: (subarr,subarr,npsi) array
		A two-dimensional array for each pupil position, showing how phase
		at that position contributes to the particular Fourier component phase
		(+1 or -1)
	ftpix: ( (nphi) array, (nphi) array)
		Co-ordinates defining the Fourier sampling.
	"""
	if (pmask.shape[0] != pmask.shape[1]):
		print "Error: Only a square pupil mask is allowed."
		raise UserWarning
	subarr = pmask.shape[0]
	ww= np.where(pmask)
	npsi = len(ww[0])
	RR = np.zeros((subarr,subarr),dtype='int')
	AA = np.zeros((subarr,subarr,npsi),dtype='int')
	for i in range(0,npsi-1):
		for j in range(i+1,npsi):
			dx = ww[0][j]-ww[0][i]+subarr/2
			dy = ww[1][j]-ww[1][i]+subarr/2
			AA[dy,dx,j] += 1
			AA[dy,dx,i] -= 1
			RR[dy,dx] += 1
	RR = np.roll(np.roll(RR,-subarr/2,axis=0),-subarr/2,axis=1)
	AA = np.roll(np.roll(AA,-subarr/2,axis=0),-subarr/2,axis=1)
	ftpix = np.where(RR)
	return RR,AA,ftpix

def pupil_sampling(in_files, subarr=128, ignore_data=False, ignore_dark=False, dark_file='', flat_file='', out_file='', rdir=''):
	"""Define the Fourier-Plane sampling based on Pupil geometry
	
	This function finds the pupil sampling and the matrix that maps phase to 
	closure-quantities, i.e. kernel-phases.
	
	Parameters
	----------
	h : A fits header dictionary-like structure returned from pyfits or
		astropy.io.fits. Must be from a known camera in a known mode.
	subarr : int
		The sub-array size 
	ignore_data : bool
		Whether to ignore the data in finding the precise Fourier sampling.
	
	Returns
	-------
	ftpix: ( (N) array, (N) array)
	fmask: (subarr,subarr) array
	ptok: (M,N) array
		A matrix that maps pupil-phases to kernel-phases.
	"""
	in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
	h = in_fits.header
	in_fits.close()

	#First, see which telescope is in use, then decide on the mask and wavelength.
	try: inst=h['CURRINST']
	except: inst=''
	if (len(inst)==0):
		print "Error: could not find instrument in header..."
		raise UserWarning
	if (aoinst.instrument != inst):
		print "Error: software expecting: ", aoinst.instrument, " but instrument is: ", inst
		raise UserWarning
	#Start with NIRC2
	if (inst == 'NIRC2'):
		try: fwo = h['FWONAME']
		except:
			print "No FWONAME in NIRC2 header"
			raise UserWarning
		try: fwi = h['FWINAME']
		except:
			print "No FWINAME in NIRC2 header"
			raise UserWarning
		if (fwo == '9holeMsk'):
			hole_xy = [[3.44,  4.57,  2.01,  0.20, -1.42, -3.19, -3.65, -3.15,  1.18],
 	  				[-2.22, -1.00,  2.52,  4.09,  4.46,  0.48, -1.87, -3.46, -3.01]]
			#For some reason... the y-axis is reversed. 
			#NB as we aren't doing (u,v) coordinates but only chip coordinates here,
			#there is no difference between reversing the x- and y- axes.
 	  		hole_xy = np.array(hole_xy)
			hole_xy[1,:] = -hole_xy[1,:]
 	  		hole_diam = 1.1
 	  		hole_shape = 'circ'
 	  		mask_rotation = -0.01
 	  	elif (fwi == '18holeMsk'):
 	  		print "Still to figure out 18 hole mask..."
 	  		raise UserWarning
		else:
			print "Assuming full pupil..."
			hole_shape = 'annulus'
			inner_diam = 2.0
			outer_diam = 10.0
 	  	if (fwi=='Kp'):
 	  		wave = 2.12e-6
			if (len(rdir) > 0):
				flat_file = rdir + 'flat_Kp.fits'
 	  	elif (fwi=='CH4_short'):
 	  		wave = 1.60e-6
			if (len(rdir) > 0):
				flat_file = rdir + 'flat_CH4_short.fits'
		elif (fwo=='Kcont'):
			wave = 2.27e-6
			if (len(rdir) > 0):
				flat_file = rdir + 'flat_Kp.fits'
 	  	else:
 	  		print "Unknown Filter!"
 	  		raise UserWarning
		print "Using Flat file: ", flat_file
 	  	try: camname = h['CAMNAME']
 	  	except:
 	  		print "No CAMNAME in header"
 	  	if (camname == 'narrow'):
			#This comes from the Yelda (2010) paper.
 	  		rad_pixel = 0.009952*(np.pi/180.0/3600.0)
 	  	else:
 	  		print "Unknown Camera!"
 	  		raise UserWarning
		#Find the appropriate dark file if needed.
		if (len(rdir) > 0):
			if h['SAMPMODE'] == 2:
				multisam = 1
			else:
				multisam = h['MULTISAM']
			dark_file = rdir + 'dark_' + str(h['NAXIS1']) +'_'+str(h['COADDS']) +'_' +str(multisam)+'_'+ str(int(h['ITIME'])) + '.fits'
			print "Using Dark file: ", dark_file
		targname = h['TARGNAME']
	else:
		print "Unknown instrument!"
		raise UserWarning 
	#Create the pupil mask the same size as the subarray...
	pmask = np.zeros((subarr,subarr))
	#The "bighole" pupil mask is 1.3 times larger, to take into account scattered
	#light from broad filters.
	pmask_bighole = pmask.copy()
	#ix[0] is x on the screen, i.e.the *second* index.
	ix = np.meshgrid(np.arange(subarr)-subarr/2,np.arange(subarr)-subarr/2)
	if (hole_shape == 'circ'):
		badpmask_threshold=0.2
		nbad_threshold = 4
		rmat = np.array([[np.cos(mask_rotation),np.sin(mask_rotation)],
					[-np.sin(mask_rotation), np.cos(mask_rotation)]])
		hole_xy = np.dot(rmat,hole_xy)
		pxy = hole_xy/wave*rad_pixel*subarr
		for i in range(hole_xy.shape[1]):
			pmask += ( (ix[0] - pxy[0,i])**2 + (ix[1] - pxy[1,i])**2 < (hole_diam/wave*rad_pixel*subarr/2.0)**2 )
			pmask_bighole += ( (ix[0] - pxy[0,i])**2 + (ix[1] - pxy[1,i])**2 < (1.3*hole_diam/wave*rad_pixel*subarr/2.0)**2 )
		#The following two lines seem to be needed for NIRC2
		pmask = np.transpose(pmask)
		pmask_bighole = np.transpose(pmask_bighole)
	elif (hole_shape == 'annulus'):
		badpmask_threshold = 0.2
		nbad_threshold = 12
		pmask += ( ix[0]**2 + ix[1]**2 < (outer_diam/wave*rad_pixel*subarr/2.0)**2 )
		pmask_bighole += ( ix[0]**2 + ix[1]**2 < (1.1*outer_diam/wave*rad_pixel*subarr/2.0)**2 )
		pmask -= ( ix[0]**2 + ix[1]**2 < (inner_diam/wave*rad_pixel*subarr/2.0)**2 )
		pmask_bighole -= ( ix[0]**2 + ix[1]**2 < (inner_diam/wave*rad_pixel*subarr/2.0)**2 )
	#Now, we create the pupil to kernel-phase array from this pupil mask.
	#Start with intrinsically 2D quantities, and the bighole version first
	RR_bighole,AA_bighole,ftpix_bighole = pmask_to_ft(pmask_bighole)
	#Create the Fourier transform mask for bad pixel rejection and bias
	#subtraction. It looks a little complex because we are using rfft2 to 
	#save time.
	fmask = np.ones((subarr,subarr/2+1),dtype='int')
	fmask[ftpix_bighole] = 0
	#Even though we're in half the Fourier plane, there are some doubled-up pixels.
	fmask[:,0] *= np.roll(fmask[::-1,0],1)
	#Now the normal version. 
	RR,AA,ftpix = pmask_to_ft(pmask)
	npsi = AA.shape[2]
	ww = np.where(pmask)
	#We can now compute ftpix,then tweak the pmask to match the actual Fourier
	#power
	if not ignore_data:
		if ignore_dark:
			dark_file=''
		cube = aoinst.clean_no_dither(in_files, dark_file=dark_file, flat_file=flat_file, fmask=fmask,subarr=subarr)
		for i in range(cube.shape[0]):
			ftim = aoinst.shift_and_ft(cube[i])
			if (i == 0):
				ps = np.abs(ftim)**2
				ftim_sum = ftim
			else:
				ps += np.abs(ftim)**2
				ftim_sum += ftim
		ps -= np.mean(ps[np.where(fmask)])
		phases = np.angle(ftim_sum)
		#We now find any power spectrum values that are unusually low, or any phases
		#that are unusually high.
		medrat = np.median(ps[ftpix]/RR[ftpix]**2)
		medps = np.median(ps[ftpix])
		badft = np.where( np.logical_or( np.abs(phases[ftpix]) > 2, \
			np.logical_and(ps[ftpix]/RR[ftpix]**2/medrat < badpmask_threshold,ps[ftpix]/medps < 0.1)) )
		
		badft = badft[0]		
		print len(badft), " Fourier positions of low power identified"
		nbadft = np.zeros(npsi)
		some_ones = np.ones(len(badft),dtype='int')
		for i in range(0,npsi):
			nbadft[i] = np.sum(np.abs(AA[(ftpix[0][badft],ftpix[1][badft],i*some_ones)]))
		badpmask = np.where(nbadft > nbad_threshold)[0]
		oldpmask = pmask.copy()
		pmask[ww[0][badpmask],ww[1][badpmask]] = 0
		print len(badpmask), " low signal pupil positions eliminated."
		#The next 2 lines are pretty useful for demonstrating how this works happens.
		plt.clf()
		plt.imshow(pmask + oldpmask,interpolation='nearest')
		plt.draw()
		#plt.plot(nbadft,'o')
		#plt.semilogy(ps[ftpix]/RR[ftpix])
		#RRsmall = RR[0:subarr,0:subarr/2+1]
		#mask = np.zeros((subarr,subarr/2+1))
		#mask[ftpix]=1
		#plt.imshow( (mask*np.maximum(ps,0)/np.maximum(RRsmall,0.01)**2)**0.3, interpolation='nearest') 
		#raise UserWarning
	#With a (potentially) updated pmask, re-compute the ftpix variables
	RR,AA,ftpix = pmask_to_ft(pmask)
	npsi = AA.shape[2]
	#Now convert to 1D quantities, indexed by the number of non-zero Fourier points, nphi
	nphi = len(ftpix[0])
	A = np.zeros((nphi,npsi),dtype='int')
	one_ix = np.ones(nphi,dtype='int')
	#Extend the ftpix index to include each of the psi variables, one at a time.
	for j in range(0,npsi):
		A[:,j] = AA[ftpix + (j*one_ix,)]

	#At this point, we have:
	#R: The reduncancy vector
	#xf: The x Fourier component.
	#yf: The y Fourier component.
	#A: A matrix relating the npsi pupil phases to
	#    the nphi Fourier components.

	print("Finally doing the matrix stuff!")
	
	#As overall piston doesn't matter, we can remove one row (column?) 
	#of A, i.e. setting one Fourier component to zero phase. Frantz did this anyway.
	A = A[:,0:npsi]
	
	U, W, V = np.linalg.svd(A)

	print('SVDC Complete.')

	# Now the matrix A is equal to U#diag_matrix(W)#V, or

	ptok = np.transpose(U[:,len(W):])

	#If a savefile name is given, save the file!
	if (len(out_file) > 0):
		hl = pyfits.HDUList()
		header = pyfits.Header()
		header['RAD_PIX'] = rad_pixel
		header['SUBARR'] = subarr
		header['TARGNAME'] = targname
		header['DARK'] = dark_file
		header['FLAT'] = flat_file
		for i in range(len(in_files)):
			header['HISTORY'] = 'Input: ' + in_files[i]
		hl.append(pyfits.ImageHDU(np.array(ftpix),header))
		hl.append(pyfits.ImageHDU(fmask))
		hl.append(pyfits.ImageHDU(ptok))
		hl.writeto(out_file,clobber=True)

	return ftpix, fmask, ptok
	
def extract_kerphase(ptok_file='',ftpix=([],[]),ptok=[],cube=[],cube_file='',
	add_noise=0,rnoise=10.0,gain=4.0,out_file='',recompute_ptok=False,
	use_poise=False,systematic=[],cal_kp=[],summary_file='',pas=[0],
	ptok_out='',window_edges=True):
	"""Extract the kernel-phases.
	
	Parameters
	----------
	cube: (Nframes,Nx,Ny) array (dtype=float)
		The cleaned input data
	ftpix: (K array,K array) 
		The Fourier pixel sampling points, from pupil_sampling
	ptok: (M,K) array
		The phase to kernel-phase matrix, from pupil_sampling
	add_noise: int
		Multiplier for the number of frames to add fake noise to. 0 turns
		this feature off.
	rnoise: float
		Readout noise in DN
	gain: float
		Gain in electrons/DN. Photon variance in DN is N/gain
	out_file: string, optional
		File to save the kernel-phases to.
	cube_file: string, optional
		Optional file input.
	
	Returns
	-------
	kp: (Nframes*extra_noisy_frames,M)
		Kernel phase array.
		
	Issues
	------
	Some Fourier components can end up wrapping around. This obviously destroys 
	the kernel-phases. There are two improvements needed to fix this...
	a) The bright calibrator mean phases should be subtracted prior to 
	target phase extraction.
	b) High-Variance phases for the target have to be weighted to zero by
	re-diagonalizing the covariance matrix and saving a new ptok file.
	c) Weighted averaging by amplitude should be added.
	"""
	#Input the kernel-phase extraction parameters
	if (len(ptok_file) > 0):
		ftpix = pyfits.getdata(ptok_file,0)
		ptok = pyfits.getdata(ptok_file,2)
	if (len(ptok.shape) != 2):
		print "Error: pupil to kerphase matrix doesn't have 2 dimensions! specify kp_file or ptok"
		raise UserWarning
	if (len(ftpix[0]) == 0):
		print "Error: no Fourier pixels specified! Specify ftpix or kp_file"
		raise UserWarning
	#Extract the POISE variables if needed
	if (use_poise):
		if (len(ptok_file) > 0):
			systematic = pyfits.getdata(ptok_file,3)
			cal_kp = pyfits.getdata(ptok_file,4)
		if (len(systematic) == 0):
			print "Error: systematic error component must be input if use_poise is set"
			return UserWarning
	#Put ftpix into the appropriate tuple format.
	ftpix = (ftpix[0],ftpix[1])
	targname = ''
	#Input the cube (or extract from file)
	if (len(cube_file) > 0):
		cube = pyfits.getdata(cube_file,0)
		pas = pyfits.getdata(cube_file,1)['pa']
		h = pyfits.getheader(cube_file)
		targname = h['TARGNAME']
	if (len(cube.shape) != 3):
		print "Error: cube doesn't have 3 dimensions! specify cube or cube_file"
		raise UserWarning
	nf = cube.shape[0]*np.max((1,add_noise))
	nf_nonoise = cube.shape[0]
	npsi =  ptok.shape[0]
	kp = np.zeros((nf,npsi))
	kp_nonoise = np.zeros((nf_nonoise,npsi))
	plt.clf()
	plt.axis([0,len(ftpix[0]),-3,3])
	for i in range(nf):
		if (add_noise > 0):
			im_raw = cube[i//add_noise,:,:]
			if (i % add_noise == 0):
				sig = 0.0
			else:
				sig = np.sqrt(np.maximum(im_raw,0)/gain + rnoise**2)
			im = im_raw + sig*np.random.normal(size=im_raw.shape)
		else:
			im = cube[i,:,:]
		if (window_edges):
			for j in range(9):
				im[j,:] *= (j + 1.0)/10.0
				im[:,j] *= (j + 1.0)/10.0
				im[-j-1,:] *= (j + 1.0)/10.0
				im[:,-j-1] *= (j + 1.0)/10.0
		#We really want to minimise the phase variance, because
		#phase wraps kill kernel-phases.
		ftim = aoinst.shift_and_ft(im)
		phases = np.angle(ftim[ftpix])
		plt.plot(phases,'.')
		kp[i,:] = np.dot(ptok,phases)
		#If use_poise is set, then we need to subtract the calibrator kp
		if (use_poise):
			kp[i,:] -= cal_kp
		if (add_noise > 0):
			if (i % add_noise == 0):
				kp_nonoise[i//add_noise,:] = kp[i,:]
		if (i % 10 == 9):
			print("Done file: " + str(i))
#			plt.plot(kp[i,:])
			plt.draw()
			plt.clf()
			plt.axis([0,len(ftpix[0]),-3,3])
			
	#Compute statistics, to save to the summary file. If use_poise is set, 
	#then we also need to add the systematic error component.
	if len(summary_file) > 0:
		#Compute the mean kernel-phase. !!! Some bad frame rejection should
		#have occured ...
		kp_mn = kp.mean(0)	
		kp_cov = np.dot(np.transpose(kp),kp)/nf - np.outer(kp_mn,kp_mn)
		kp_mn = kp_nonoise.mean(0)
		#Don't count the add_noise frames in computing the standard error of
		#the mean.
		kp_cov /= (cube.shape[0]-1.0)
		diag_ix = (range(npsi),range(npsi))
		if (use_poise):
			kp_cov[diag_ix] += systematic
		#Now we only save the diagonal of the covariance matrix
		#(as we've already computed a new ptok as required)
		kp_sig = np.sqrt(np.diagonal(kp_cov))
		hl = pyfits.HDUList()
		header = pyfits.Header()
		header['PTOKFILE'] = ptok_file
		header['CUBEFILE'] = cube_file
		header['TARGNAME'] = targname
		header['PA'] = np.mean(pas) #!!! This assumes that all frames are used
		col1 = pyfits.Column(name='kp_mn', format='E', array=kp_mn)
		col2 = pyfits.Column(name='kp_sig', format='E', array=kp_sig)
		cols = pyfits.ColDefs([col1, col2])
		hl.append(pyfits.PrimaryHDU(header=header))
		hl.append(pyfits.new_table(cols))
		hl.writeto(summary_file,clobber=True)

	if (len(out_file) > 0):
		hl = pyfits.HDUList()
		header = pyfits.Header()
		header['CUBEFILE'] = cube_file
		header['PTOKFILE'] = ptok_file
		hl.append(pyfits.ImageHDU(kp,header))
		hl.writeto(out_file,clobber=True)
	return kp

def poise_kerphase(kp_files,ptok=[],ftpix=([],[]),fmask=[],beta=0.4,
		out_file='',rad_pixel=0.0,subarr=128):
	"""Extract the POISE kernel-phases, returning a new ptok matrix
	
	Parameters
	----------
	kp_files: list of strings
		List of kp filenames to turn in to kernel phases.
	ptok: (K,M) array
		The phase to kernel-phase array.
	
	Returns
	-------
	ptok_poise: (K_new,M) array
	
	Notes
	-----
	This works on files rather than all the data, because it could take up
	too much memory. As bad files can't be rejected until kernel-phases are
	computed, saving kp files is the only way to only compute them once. 
	"""
	hkp = pyfits.getheader(kp_files[0])
	npsi = hkp['NAXIS1']
	#Hopefully, the ptok variables are stored as part of the kp_file.
	ptok_file = hkp['PTOKFILE']
	if len(ptok_file) > 0:
		print "Extracting ptok and other variables from file"
		ftpix = pyfits.getdata(ptok_file,0)
		fmask = pyfits.getdata(ptok_file,1)
		ptok = pyfits.getdata(ptok_file,2)
		hptok = pyfits.getheader(ptok_file)
		rad_pixel = hptok['RAD_PIX']
		subarr = hptok['SUBARR']
	#Let's sanity-check
	if len(ptok.shape) != 2:
		print "Error: no valid ptok matrix. ptok or a kp_file with embedded ptok fits filename must be input."
		return UserWarning
	if len(fmask) == 0:
		print "Error: A Fourier-plane mask fmask must set or embedded in the ptok fits file."
		return UserWarning
	ftpix = (ftpix[0],ftpix[1])
	ncubes = len(kp_files)
	kp_cube_covs = np.zeros((ncubes,npsi,npsi))
	kp_cov = np.zeros((npsi,npsi))
	kp_mns = np.zeros((ncubes,npsi))
	kp_mn = np.zeros(npsi)
	nf_all = 0 #Total number of frames
	for i in range(ncubes):
		kp = pyfits.getdata(kp_files[i])
		kp_mns[i,:] = kp.sum(0)
		kp_mn += kp_mns[i,:]
		nf = kp.shape[0]
		kp_mns[i,:] /= nf
		nf_all += nf
		#Create the overall covariance ignoring the mean...
		kp_cov += np.dot(np.transpose(kp),kp)
		#Now correct for the mean...
		for j in range(nf):
			kp[j,:] -= kp_mns[i,:]
		cov = np.dot(np.transpose(kp),kp)
		kp_cube_covs[i,:,:] = cov/(nf - 1.0)
	kp_mn /= nf_all
#	kp_cov -= np.outer(kp_mn,kp_mn)
	kp_cov /= (nf_all - 1)
	
	#Diagonalise kp_cov, and project the cube kernel-phases onto this
	#new space. Not that the transpose of W is a projection matrix, but the
	#rows of kp_mns are the kernel_phases
	D, V = np.linalg.eigh(kp_cov)
	kp_mns = np.dot(kp_mns, V)
	kp_mn = np.dot(kp_mn,V)
	for i in range(ncubes):
		kp_cube_covs[i,:,:] = np.dot(np.transpose(V),np.dot(kp_cube_covs[i,:,:],V))
#This should work but is way slow! 	kp_cube_covs = np.dot(kp_cube_covs,V)
	#Now lets compare the eigenvalues to the internal variance to discover
	#the "systematic" errors
	cov_internal = np.mean(kp_cube_covs, axis=0)
	var_internal = np.diagonal(cov_internal)
	var_sys = np.maximum(D-var_internal,0)
	#Make a plot that shows these variances
	plt.clf()
	plt.semilogy(D)
	plt.semilogy(var_internal)
	good = np.where(var_sys/var_internal < beta)[0]
	bad = np.where(var_sys/var_internal >= beta)[0]
	plt.semilogy(bad,D[bad],'o')
	plt.xlabel('Kernel Phase Number')
	plt.ylabel('Variance')
	plt.title('Rejected Kernel-phases')
	plt.show()
	print('Num Kernel-phases rejected for calibration: ' + str(len(bad)))
	print('Num Good Kernel-phases: ' + str(len(good)))
	print V.shape
	print ptok.shape
	ptok_poise = np.dot(np.transpose(V),ptok)
	#Now show the Fourier response of the rejected kernel-phases
	if len(ftpix)>0:
		for i in range(len(bad)):
			print "Click for next figure..."
			dummy = plt.ginput(1)
			ysz = np.max(ftpix[0])+1
			xsz= np.max(ftpix[1])+1
			ft_temp = np.zeros((ysz,xsz))
			ft_temp[ftpix] = 0.1
			ft_temp[ftpix] += ptok_poise[bad[i],:]
			ft_temp = np.roll(ft_temp,ysz/2,axis=0)
			delta_y = int(np.min([xsz*1.1,ysz/2]))
			ft_temp = ft_temp[ysz/2-delta_y:ysz/2+delta_y,:]
			plt.clf()
			plt.imshow(ft_temp,interpolation='nearest')
			plt.title('Bad kerphase ' + str(bad[i]))
			plt.draw()
	ptok_poise = ptok_poise[good,:]
	systematic = var_sys[good]
	kp_mn = kp_mn[good]
	#If a savefile name is given, save the file!
	if (len(out_file) > 0):
		if (rad_pixel == 0):
			print "Missing rad_pixel variable!"
			return UserWarning
		hl = pyfits.HDUList()
		header = pyfits.Header()
		header['RAD_PIX'] = rad_pixel
		header['SUBARR'] = subarr
		for i in range(ncubes):
			header['HISTORY'] = 'Input: ' + kp_files[i]
		hl.append(pyfits.ImageHDU(ftpix,header))
		hl.append(pyfits.ImageHDU(fmask))
		hl.append(pyfits.ImageHDU(ptok_poise))
		hl.append(pyfits.ImageHDU(systematic))
		hl.append(pyfits.ImageHDU(kp_mn))
		hl.writeto(out_file,clobber=True)
	return ptok_poise,systematic,kp_mn

def kp_to_implane(ptok_file=[], pas=[0], summary_files=[], pxscale=10.0, sz=128,out_file=''):
	"""Here we convert kernel-phases to their image-plane representation,
	in sky coordinates.
	
	Parameters
	----------
	ptok_file: string
		The input fits file containing the ptok matrix
	pas: float, optional
		The position angles of vertical
	summary_files: string array, optional
		The list of filenames to get position angles and summary data from
	pxscale: float
		Output pixel scale in arcsec per pixel
	"""
	# Read in details from summary_files
	if len(summary_files) > 0:
		pas = np.zeros(len(summary_files))
		kp_mn = np.array([])
		kp_sig = np.array([])
		for i in range(len(summary_files)):
			h = pyfits.getheader(summary_files[i])
			d = pyfits.getdata(summary_files[i])
			pas[i] = h['PA']
			kp_mn = np.append(kp_mn,d['kp_mn'])
			kp_sig = np.append(kp_sig,d['kp_sig'])
		ptok_file = h['PTOKFILE']
		all_data = np.array([kp_mn,kp_sig])
		print all_data.shape
	ftpix = pyfits.getdata(ptok_file,0)
	ptok = pyfits.getdata(ptok_file,2)
	# The ptok_file gives us additional header information...
	header = pyfits.getheader(ptok_file)
	rad_pixel = header['RAD_PIX'] 
	subarr = header['SUBARR'] 
	
			
	# From here, we work through each element of ftpix one at a time for each
	# target position angle. First, sort out some preliminaries...
	nt = len(pas)
	nphi = ptok.shape[1]
	npsi = ptok.shape[0]
	x = (np.arange(sz) - sz/2)*pxscale/1000.0*np.pi/180.0/3600.0
	xy = np.meshgrid(x,x)
	# Now allocate the giant memory array.
	kp_implane = np.zeros((nt,npsi,sz,sz))
	fty = ((ftpix[0] + subarr/2) % subarr) - subarr/2
	ftx = ftpix[1]
	for k in range(nt):
		#Create the xf and yf (x and y Fourier vectors) in units of cycles
		#per FOV. !!! sign of pa to be checked.
		yf = fty*np.cos(np.radians(pas[k])) + ftx*np.sin(np.radians(pas[k]))
		xf = ftx*np.cos(np.radians(pas[k])) - fty*np.sin(np.radians(pas[k]))
		#Now convert to physical units of radians^-1
		xf = xf/(subarr*rad_pixel)*2*np.pi
		yf = yf/(subarr*rad_pixel)*2*np.pi
		for i in range(nphi):
			#Sign of xf and yf below have calibrated by injecting fake binaries.
			sine = np.sin(-xf[i]*xy[0] - yf[i]*xy[1])
			for j in range(npsi):
				kp_implane[k,j,:,:] += ptok[j,i]*sine
	#Reshaping makes the array compatible with the imaging code.
	kp_implane=kp_implane.reshape((nt*npsi,sz,sz))
	#If a savefile name is given, save the file!
	if (len(out_file) > 0):
		hl = pyfits.HDUList()
		header = pyfits.Header()
		header['PXSCALE'] = pxscale
		header['SUBARR'] = subarr
		header['RAD_PIX'] = rad_pixel
		hl.append(pyfits.ImageHDU(kp_implane,header))
		#Now add in the data. This was originally an image in IDL, so we keep
		#it as an image, not a table.
		if len(summary_files) > 0:
			header['PAMODE'] = 'Sky'
			hl.append(pyfits.ImageHDU(all_data))
		else:
			header['PAMODE'] = 'Det'
		hl.writeto(out_file,clobber=True)
	return kp_implane


def implane_fit_binary(kp_implane_file, summary_file='', out_file='',pa_vertical=0,to_sky_pa=False):
	"""A binary grid search to a kp_implane file.
	
	Parameters
	----------
	kp_implane_file: string
		The fits file containing the kernel-phase to implane data
		(and, optionally, the data also)
	summary_file: string
		The kernel-phase results file.		
		
	Returns
	-------
	p, crat, crat_sig, chi2
	
	"""
	kp_implane = pyfits.getdata(kp_implane_file,0)
	h = pyfits.getheader(kp_implane_file)
	pxscale = h['PXSCALE']
	sz = kp_implane.shape[1]
	if len(summary_file) > 0:
		d = pyfits.getdata(summary_file)
		header = pyfits.getheader(summary_file)
		pa_vertical = header['PA']
		kp_mn = d['kp_mn']
		kp_sig = d['kp_sig']
	else:
		d = pyfits.getdata(kp_implane_file,1)
		kp_mn = d[0,:]
		kp_sig = d[1,:]
	crat = np.zeros((sz,sz))
	crat_sig = np.zeros((sz,sz))
	chi2 = np.zeros((sz,sz))
	for i in range(sz):
		for j in range(sz):
			mod_kp = kp_implane[:,i,j]
			crat[i,j] = np.sum(mod_kp*kp_mn/kp_sig**2)/np.sum(mod_kp**2/kp_sig**2)
			crat_sig[i,j] = np.sqrt(1.0/np.sum(mod_kp**2/kp_sig**2))
			if (np.isnan(crat[i,j])): 
				crat[i,j] = 0.0
				crat_sig[i,j]=1.0
			chi2[i,j] = np.sum((crat[i,j]*mod_kp - kp_mn)**2/kp_sig**2)
	modified_chi2 = chi2.copy()
	modified_chi2[crat < 0] = np.max(modified_chi2)
	min_ix = np.unravel_index(modified_chi2.argmin(), modified_chi2.shape)
	best_rchi2 = chi2[min_ix[0], min_ix[1]]/(len(kp_sig) - 3)
	print "Minimum reduced chi2: ", best_rchi2
	print "Significance (in sigma): ", crat[min_ix[0], min_ix[1]]/crat_sig[min_ix[0], min_ix[1]]
	sep = pxscale*np.sqrt((min_ix[1]-sz/2)**2 + (min_ix[0]-sz/2)**2)
	pa = np.degrees(np.arctan2( -(min_ix[1]-sz/2), min_ix[0]-sz/2))
	contrast = crat[min_ix[0], min_ix[1]]
	#Adjust to on-sky units if needed.
	if (to_sky_pa): 
		pa += pa_vertical
	
	#Some plotting
	extent = [sz/2*pxscale,-sz/2*pxscale,-sz/2*pxscale,sz/2*pxscale]
	plt.clf()
	plt.imshow(modified_chi2[::-1,:], interpolation='nearest', extent=extent)
	plt.axis(extent)
	plt.title('Chi^2 map for positive contrasts')
	plt.xlabel('Delta RA (milli-arcsec)', fontsize='large')
	plt.ylabel('Delta Dec (milli-arcsec)', fontsize='large')
	plt.plot(0,0,'w*', ms=10)
	ax = plt.axes()
	ax.arrow(-0.45*sz*pxscale, -0.45*sz*pxscale, 0.1*sz*pxscale, 0, head_width=0.02*sz*pxscale, head_length=0.02*sz*pxscale, fc='k', ec='k')
	ax.arrow(-0.45*sz*pxscale, -0.45*sz*pxscale, 0, 0.1*sz*pxscale, head_width=0.02*sz*pxscale, head_length=0.02*sz*pxscale, fc='k', ec='k')
	plt.text(-0.45*sz*pxscale, -0.3*sz*pxscale,'N')
	plt.text(-0.3*sz*pxscale,-0.45*sz*pxscale, 'E')
	plt.draw()
	print "sep, pa, contrast, sig: ", sep, pa, contrast, crat_sig[min_ix[0], min_ix[1]]
	return (sep,pa,contrast), crat, crat_sig, chi2, best_rchi2

def 	kp_binary_fitfunc_onefile(p, rad_pixel, subarr, ftpix, ptok, kp_mn, kp_sig):
	"""This function finds the fit residuals based on input model parameters in
	chip-coorindates as (sep,PA,contrast). 
	
	Parameters
	----------
	p: [sep in mas, chip position angle in degrees, contrast secondary/primary]
	rad_pixel: float
		Radians per pixel in the original image.
	subarr: float
		Subarray size in the original image and ftpix definition.
	ftpix: ( (N) array, (N) array )
		Sampling points in the Fourier domain
	ptok: 
		Pupil to kernel-phase matrix
	kp_mn: 
		Kernel-phases from the data
	kp_sig:
		Uncertainties in kernel-phases
	"""
	xy = np.meshgrid(2*np.pi*np.arange(subarr/2 + 1)/float(subarr),
					2*np.pi*np.arange(subarr)/float(subarr))
	dy_in_pix = p[0]*np.pi/180./3600./1000.*np.cos(np.radians(p[1]))/rad_pixel
	dx_in_pix = -p[0]*np.pi/180./3600./1000.*np.sin(np.radians(p[1]))/rad_pixel
	# Avoiding the fft is marginally faster here, just like for bad pixel 
	# rejection...
	ft = 1 + p[2]*np.exp(-1j*(dy_in_pix*xy[1] + dx_in_pix*xy[0]))
	modkp = np.dot(ptok,np.angle(ft[ftpix]))
	return (kp_mn - modkp)/kp_sig

def 	kp_binary_fitfunc(p, rad_pixels, subarrs, ftpixs, ptoks, kp_mns, kp_sigs, pas,ptok_files):
	"""This function finds the fit residuals based on input model parameters in
	on-sky coorindates as (sep,PA,contrast). It can have multiple input files.
	
	Parameters
	----------
	p: [sep in mas, sky position angle in degrees, contrast secondary/primary]
	rad_pixels: float list
		Radians per pixel in the original image.
	subarrs: float list
		Subarray size in the original image and ftpix definition.
	ftpixs: ( (N) array, (N) array ) list
		Sampling points in the Fourier domain
	ptoks:  
		Pupil to kernel-phase matrices
	kp_mns: 
		Kernel-phases from the data
	kp_sigs:
		Uncertainties in kernel-phases
	
	Returns
	-------
	resid: (K) array
		normalised residuals to model kernel phases.
	"""	
	resid=np.array([])
	for i in range(len(pas)):
		p_i = p.copy()
		p_i[1] -= pas[i]
		ftpix = ftpixs[ptok_files[i]] 
		ptok = ptoks[ptok_files[i]] 
		resid = np.append(resid,kp_binary_fitfunc_onefile(p_i, rad_pixels[i], subarrs[i], ftpix, ptok, kp_mns[i], kp_sigs[i]))
	return resid
	
def 	kp_binary_fit(summary_files, initp):
	"""Fit a binary model to kernel-phases
	
	Parameters
	----------
	summary_files: string array
		List of input summary files.
	initp: (sep in mas, pa, contrast sec/primary)
		Initial fit parameters. These can be obtained from e.g. implane_fit_binary
		
	Returns
	-------
	(p_fit, p_sig, p_cov)
	p_fit: [float,float,float]
	p_sig: [float,float,float]
	p_cov: (3,3) array
		Best fit (sep, pa, contrast) parameters, standard deviation and 
		covariance. 
	"""
	#Read in the summary files, adding to appropriate variables as we go
	kp_mns = []
	kp_sigs = []
	pas = []
	ptok_files = []
	#Allow the different summary_files to use different ptok definition files,
	#but if they are all the same, only store one copy in memory. This is 
	#easily accomplised with a "dict"
	rad_pixels = []
	subarrs = []
	ftpixs = dict()
	ptoks = dict()	
	for i in range(len(summary_files)):
		d = pyfits.getdata(summary_files[i])
		kp_mns.append(d['kp_mn'])
		kp_sigs.append(d['kp_sig'])
		header = pyfits.getheader(summary_files[i])
		pas.append(header['PA'])
#		pas[i]=0.0
		ptok_files.append(header['PTOKFILE'])
		header = pyfits.getheader(ptok_files[i])
		rad_pixels.append(header['RAD_PIX'])
		subarrs.append(header['SUBARR'])
		#!!! Could save time here by only reading in once.
		ftpix = pyfits.getdata(ptok_files[i],0)
		ftpixs[ptok_files[i]] = (ftpix[0], ftpix[1])
		ptoks[ptok_files[i]] = pyfits.getdata(ptok_files[i],2)
	p = op.leastsq(kp_binary_fitfunc, initp, args=(rad_pixels, subarrs, ftpixs, \
		ptoks, kp_mns, kp_sigs, pas,ptok_files), full_output=True, diag = [1,.5,0.01],factor=0.1)
	#, factor = factorTry, diag = diagTry, maxfev = maxfev)
	ndf = len(p[2]['fvec'])
	rchi2 = np.sum(p[2]['fvec']**2)/(ndf-3)
	print "Best Reduced Chi^2: ", rchi2
	print "Parameters: {0:6.1f} {1:7.2f} {2:6.2f}".format(p[0][0], p[0][1], -2.5*np.log10(p[0][2]))
	errs = np.sqrt(np.diag(p[1])*rchi2)
	print "Errors:     {0:6.1f} {1:7.2f} {2:6.2f}".format(errs[0], errs[1], 2.5*np.log10(np.e)*errs[2]/p[0][2])
	return p        

file = '/Users/mireland/data/nirc2/131020/n0641.fits.gz'
file = '/Users/mireland/data/nirc2/131020/n0204.fits.gz'
extn = '.fits.gz'
	
#Dark for the main file
if(0):
	dir =  '/Users/mireland/data/nirc2/131020/n'
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(617,627)]
	#files.extend([(dir + '{0:04d}' + extn).format(i) for i in range(56,61)])
	aoinst.make_dark(files,'dark512.fits')

#Testing pupil mask generation etc...
if(1):
	dir =  '/Users/mireland/data/nirc2/131020/n'
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(201,211)]
	run = '2'
	(ftpix, fmask, ptok) = pupil_sampling(files[0:3],dark_file='dark512.fits',flat_file='flat.fits',out_file='kp_Kp9h.fits')
	cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
	kp = extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(179,189)]
	run = '1'
	cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
	kp = extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(157,167)]
	run = '0'
	cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
	kp = extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(225,237)]
	run = '3'
	cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
	kp = extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')

#Turn these observables into POISE observables, and find the image-plane representation.
if(0):
	ptok_poise,systematic,cal_kp = poise_kerphase(['test_kp0.fits','test_kp1.fits','test_kp2.fits','test_kp3.fits'],out_file='ptok_poise_FPTauCals.fits')
	kp_implane = kp_to_implane(ptok_file='ptok_poise_FPTauCals.fits', out_file='ptok_poise_implane.fits')


#The target cube files
if(1):
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(168,178)]
	cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube_FPTau.fits')
	extract_kerphase(ptok_file='ptok_poise_FPTauCals.fits',cube_file='test_cube_FPTau.fits',use_poise=True,summary_file='poise_FPTau.fits',add_noise=16)
	kp_implane = kp_to_implane(summary_files=['poise_FPTau.fits'], out_file='FPTau_implane.fits')

if (0):
	files = [(dir + '{0:04d}' + extn).format(i) for i in range(144,154)]
#	cube = clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube_V892Tau.fits')
#	extract_kerphase(ptok_file='ptok_poise_hd283477.fits',cube_file='test_cube_V892Tau.fits',use_poise=True,summary_file='poise_V892Tau.fits',add_noise=16)
	d = pyfits.getdata('poise_V892Tau.fits')	
#	kp_implane = pyfits.getdata('ptok_poise_implane.fits')
	kp_implane = kp_to_implane(summary_files=['poise_V892Tau.fits'], out_file='V892_implane.fits')
	im = np.zeros((128,128))
	for i in range(2896): 
		im += d['kp_mn'][i]*kp_implane[i,:,:]
	plt.clf()
	plt.imshow(im[::-1,:], interpolation='nearest')
	plt.title('Simplified reconstructed image (vertical reversed)')
	
if (0):
	#Now lets create a fake binary...
	acube = pyfits.getdata('test_cube2.fits')
	h = pyfits.getheader('test_cube2.fits')
	for i in range(acube.shape[0]):
		acube[i,:,:] += np.roll(np.roll(acube[i,:,:],0,axis=1),7,axis=0)*0.1
	hl = pyfits.HDUList()
	hl.append(pyfits.ImageHDU(acube,h))
	hl.append(pyfits.new_table(pyfits.getdata('test_cube2.fits',1)))
	hl.writeto('fakebin_cube.fits',clobber=True)
	extract_kerphase(ptok_file='ptok_poise_hd283477.fits',cube_file='fakebin_cube.fits',use_poise=True,summary_file='fakebin_summary.fits',add_noise=16)
	d = pyfits.getdata('fakebin_summary.fits')	
	kp_implane = pyfits.getdata('ptok_poise_implane.fits')
	im = np.zeros((128,128))
	for i in range(2850): 
		im += d['kp_mn'][i]*kp_implane[i,:,:]
	plt.clf()
	plt.imshow(im[::-1,:], interpolation='nearest')
	plt.title('Simplified reconstructed image (vertical reversed)')

if (1):
#	summary_file = 'fakebin_summary.fits'
#	implane_file = 'ptok_poise_implane.fits'
#	summary_file = 'poise_V892Tau.fits'
#	implane_file = 'V892_implane.fits'
	summary_file = 'poise_FPTau.fits'
	implane_file = 'FPTau_implane.fits'
	pgrid, crat, crat_sig, chi2 = implane_fit_binary(implane_file, summary_file=summary_file, to_sky_pa=False)
	print "Grid Fit: ", pgrid
	pgrid = np.array(pgrid)
	if (pgrid[2] > 0.5):
		print "Contrast too high to use kerphase for fitting (i.e. near-equal binary)."
	else:
		p = kp_binary_fit([summary_file],pgrid)
	