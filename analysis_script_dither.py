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
# This includes an AO Instrument called "aoinst"
import pypoise
import nirc2
#Create a pypoise instance with a nirc2 AO instrument
pp = pypoise.PYPOISE(nirc2.NIRC2())
plt.ion()

#Reduction Directory - Lp full pupil
pp.aoinst.rdir = '/Users/mireland/tel/nirc2/redux/140610/'
pp.aoinst.cdir = '/Users/mireland/tel/nirc2/redux/140610/'
#Data directory
pp.aoinst.ddir =  '/Users/mireland/data/nirc2/140610/'


extn = '.fits.gz'
pp.aoinst.read_summary_csv()

if(False):
 pp.aoinst.make_all_darks()

if(False):
 files = [(pp.aoinst.ddir + 'n{0:04d}' + extn).format(i) for i in range(33,87)]
 pp.aoinst.make_flat(files, dark_file='dark_1024_10_1_18.fits')
    
if(False):
 #Create the Fourier sampling and ptok matrix. This needs a moderately high S/N
 #data set to tweak the Fourier sampling.
 files = [('n{0:04d}' + extn).format(i) for i in range(558,566)]
 pp.pupil_sampling(files)

if(False): 
 #Cube/clean the files. Lets start with doing this manually over a limited elevation range
 fnums = (range(541+8,557),range(558,566),range(567,575),range(576,582),range(582,588))
 cube_files = []
 kp_files = []
 kp_mn_files = []
 all_maxs = []
 mn_maxs = []
 mn_kps = []
 for j in range(len(fnums)):
  files = [('n{0:04d}' + extn).format(i) for i in fnums[j]]
  cube_file = 'cube{0:04d}'.format(fnums[j][0]) + '.fits'
  #These next two are manual for now... Maybe they should be based by default
  #on something in the cube_file header?
  kp_file = pp.aoinst.rdir + 'kp' + str(fnums[j][0]) + '.fits'
  kp_mn_file = pp.aoinst.rdir + 'kp_mn' + str(fnums[j][0]) + '.fits'
  
  cube_files.append(cube_file)
  kp_files.append(kp_file)
  cube = pp.aoinst.clean_dithered(files, out_file=cube_file)#, manual_click=True)
  d = pyfits.getdata(pp.aoinst.cdir + cube_file,1)
  all_maxs.append(d['max'])
  mn_maxs.append(all_maxs[j].mean)
  pp.extract_kerphase(cube_file=cube_file, add_noise=50,out_file=kp_file, summary_file=kp_mn_file)
  d=pyfits.getdata(kp_mn_file)
  mn_kps.append(d['kp_mn'])
 kp_files = np.array(kp_files)

if(True):
 #Go through the files and find the ones that look best for poise...
#  cals = [0,2,4]
#  mn_kps = np.array(mn_kps)
#  kp_files = np.array(kp_files)
#  med_cal_kps = []
#  med_kp = np.median(mn_kps[, axis=0)
#  for j in range(len(fnums)):
#   cal_kp = mn_kps[j,:] - med_kp
#   med_cal_kps.append(np.abs(np.median(cal_kp)))
#  # Find the good calibrators from the cals with the lowest scatter.
#  good_cals = np.where(med_cal_kps < 1.5*np.median(med_cal_kps))[0]
 good_cals = [0,2,4]
 pp.poise_kerphase(kp_files[good_cals], out_file=pp.aoinst.cdir + 'poise.fits')
    
if (True):
 targname = 'HD169142'
 summary_file1 = pp.aoinst.cdir + targname + '_0558_kp_poise.fits'
 summary_file2 = pp.aoinst.cdir + targname + '_0576_kp_poise.fits'
 implane_file = pp.aoinst.cdir + targname + '_implane.fits'
 #Readout noise on the following line is slightly higher because of the weird systematics associated with 
 #phase wrapping.
 pp.extract_kerphase(ptok_file='poise.fits',cube_file='cube0558.fits',\
    use_poise=True,summary_file=summary_file1,add_noise=16) 
 pp.extract_kerphase(ptok_file='poise.fits',cube_file='cube0576.fits',\
    use_poise=True,summary_file=summary_file2,add_noise=16) 
 #Automatic from here...
 pxscale = 5.0
 kp_implane = pp.kp_to_implane(summary_files=[summary_file1,summary_file2], out_file=implane_file, sz=100, pxscale=pxscale)

if (True):
 pgrid, crat, crat_sig, chi2, best_rchi2 = pp.implane_fit_binary(implane_file)
 print "Grid Fit: ", pgrid
 pgrid = np.array(pgrid)
 if (pgrid[2] > 0.5):
    print "Contrast too high to use kerphase for fitting (i.e. near-equal binary)."
 else:
    p = pp.kp_binary_fit([summary_file1, summary_file2],pgrid)
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
