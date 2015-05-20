# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:49:11 2014

@author: mireland

A script for testing...  Change this to try out your own analysis.
"""

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from azimuthalAverage import *
from pysam_nirc2 import *
from pysam_util import *

rdir = '/Users/mireland/tel/nirc2/redux/130717/'
ddir =  '/Users/mireland/data/nirc2/130717/n'
extn = '.fits.gz'

if(0):
 #Make darks... (manually for now)
 files = [(ddir + '{0:04d}' + extn).format(i) for i in range(66,76)]
 make_dark(files,rdir + 'dark_512_1_64_20.fits')
 files = [(ddir + '{0:04d}' + extn).format(i) for i in range(378,389)]
 make_dark(files,rdir + 'dark_512_4_16_5.fits')  

 #Make a dark for the flat
 files = [(ddir + '{0:04d}' + extn).format(i) for i in range(21,26)]
 make_dark(files,rdir + 'dark_1024_1_64_45.fits')
 files = [(ddir + '{0:04d}' + extn).format(i) for i in range(26,41)]
 make_flat(files,rdir + 'dark_1024_1_64_45.fits',rdir + 'flat_Kp.fits')

 #Create the Fourier sampling and ptok matrix. This needs a moderately high S/N
 #data set to tweak the Fourier sampling.
 files = [(ddir + '{0:04d}' + extn).format(i) for i in range(372,378)]
 (ftpix, fmask, ptok) = pupil_sampling(files,rdir=rdir,out_file='kp_Kp9h.fits')

if(1): 
 #Cube/clean the files. Lets start with doing this manually over a limited elevation range
 fnums = (range(93,99),range(102,108),range(112,118),range(121,127),range(130,136),\
	range(142,148),range(151,157),range(160,166),range(169,175),range(178,184))
 cube_files = []
 kp_files = []
 kp_mn_files = []
 all_maxs = []
 mn_maxs = []
 mn_kps = []
 for j in range(len(fnums)):
  files = [(ddir + '{0:04d}' + extn).format(i) for i in fnums[j]]
  cube_file = rdir + 'cube' + str(fnums[j][0]) + '.fits'
  kp_file = rdir + 'kp' + str(fnums[j][0]) + '.fits'
  kp_mn_file = rdir + 'kp_mn' + str(fnums[j][0]) + '.fits'
  cube_files.append(cube_file)
  kp_files.append(kp_file)
  cube = clean_no_dither(files, fmask_file = 'kp_Kp9h.fits', out_file=cube_file)
  d = pyfits.getdata(cube_file,1)
  all_maxs.append(d['max'])
  mn_maxs.append(all_maxs[j].mean)
  extract_kerphase(ptok_file='kp_Kp9h.fits',cube_file=cube_file,\
		add_noise=50,out_file=kp_file, rnoise=2.0, summary_file=kp_mn_file)
  d=pyfits.getdata(kp_mn_file)
  mn_kps.append(d['kp_mn'])

if(0):
 #Go through the files and find the ones that look best for poise...
 mn_kps = np.array(mn_kps)
 kp_files = np.array(kp_files)
 med_cal_kps = []
 med_kp = np.median(mn_kps, axis=0)
 for j in range(len(fnums)):
  cal_kp = mn_kps[j,:] - med_kp
  med_cal_kps.append(np.abs(np.median(cal_kp)))
 # Find the good calibrators from the cals with the lowest scatter.
 good_cals = np.where(med_cal_kps < 1.5*np.median(med_cal_kps))[0]
 poise_kerphase(kp_files[good_cals], out_file=rdir + 'poise.fits')
	
if (0):
 targname = 'KOI2704'
 summary_file = rdir + targname + '_kp_poise.fits'
 implane_file = rdir + targname + '_implane.fits'
 pxscale = 5.0
 #Readout noise on the following line is slightly higher because of the weird systematics associated with 
 #phase wrapping.
 extract_kerphase(ptok_file=rdir + 'poise.fits',cube_file=rdir + 'cube142.fits',\
	use_poise=True,summary_file=summary_file,add_noise=16,rnoise=3.0) 
 #Automatic from here...
 kp_implane = kp_to_implane(summary_files=[summary_file], out_file=implane_file, sz=200, pxscale=pxscale)

if (0):
 pgrid, crat, crat_sig, chi2 = implane_fit_binary(implane_file, summary_file=summary_file, to_sky_pa=False)
 print "Grid Fit: ", pgrid
 pgrid = np.array(pgrid)
 if (pgrid[2] > 0.5):
	print "Contrast too high to use kerphase for fitting (i.e. near-equal binary)."
 else:
	p = kp_binary_fit([summary_file],pgrid)
 a = azimuthalAverage(crat_sig, returnradii=True,binsize=1)
 sep_null = a[0]*pxscale
 contrast_null = -2.5*np.log10(5*a[1])
 plt.clf()
 plt.plot(sep_null, contrast_null)
 plt.title(targname)
 plt.xlabel("Separation (milli-arcsec)")
 plt.ylabel("5-sigma contrast (mags)")
 sep_out = np.arange(20,301,10)
 contrast_out = np.interp(sep_out, sep_null, contrast_null)
 for i in range(len(sep_out)):
  print '{0:4d} {1:5.1f}'.format(int(sep_out[i]),contrast_out[i])
 plt.axis((0,300,2,4.5))
