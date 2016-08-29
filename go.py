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
import glob
import pdb

#Create a pypoise instance with a nirc2 AO instrument
pp = pypoise.PYPOISE(nirc2.NIRC2())
plt.ion()

#Reduction Directory - Lp full pupil
pp.aoinst.rdir = '/Users/mireland/tel/nirc2/redux/generic2015/'
pp.aoinst.cdir = '/Users/mireland/tel/nirc2/redux/TauPAH15/'
#Data directory
pp.aoinst.ddir =  '/Users/mireland/data/nirc2/151128/'


pp.aoinst.read_summary_csv()

if(False): 
  pp.process_block(fstart='n1251.fits',fend='n1293.fits', dither=True)

if(False): 
  pp.process_block(fstart='n1493.fits',fend='n1517.fits', dither=True)

targname = 'AB Aur'
targname = 'SU Aur'
targname = 'RY Tau'

if(True):
 #The argument "target_file" is just there to determine which object is the target.
 summary_files = pp.poise_process(target=targname, use_powerspect=False)
 print(summary_files)
 
if(True): 
# summary_files = glob.glob('*LkCa*poise_cube*.fits')
 implane_file = pp.aoinst.cdir + targname + '_implane.fits'
 pxscale = 5.0

#pdb.set_trace()

if (True):
 kp_implane = pp.kp_to_implane(summary_files=summary_files, 
 	out_file=implane_file, sz=141, pxscale=pxscale, use_powerspect=False)

if (True):
 #Automatic from here...
 pgrid, crat, crat_sig, chi2, best_rchi2 = pp.implane_fit_binary(implane_file, maxrad=250)
 print "Grid Fit: ", pgrid
 pgrid = np.array(pgrid)
 if (pgrid[2] > 0.5):
    print "Contrast too high to use kerphase for fitting (i.e. near-equal binary)."
 else:
    p,errs,cov = pp.kp_binary_fit(summary_files,pgrid)
    fitfile = open(targname + '_binaryfit.txt','w')
    fitfile.write('Separation (mas) & Position angle (degs) & Contrast \\\\\n')
    fitfile.write('{0:5.2f} $\pm$ {1:5.2f} & {2:5.2f} $\pm$ {3:5.2f} & {4:6.4f} $\pm$ {5:6.4f} \\\\ \n'.format(\
            p[0],errs[0], p[1],errs[1], p[2],errs[2]))
    fitfile.write('Contrast (mags) & Separation (mas) & Position angle (degs) \\\\\n')
    fit_crat = -2.5*np.log10(p[2])
    fit_crat_sig = 2.5/np.log(10)*errs[2]/p[2]
    fitfile.write('{0:5.2f} $\pm$ {1:5.2f} & {2:5.2f} $\pm$ {3:5.2f} & {4:5.3f} $\pm$ {5:5.3f} \\\\ \n'.format(\
            fit_crat, fit_crat_sig, p[0],errs[0], p[1],errs[1] ))
    fitfile.close()
 a = azimuthalAverage(crat_sig*np.sqrt(best_rchi2), returnradii=True,binsize=1)
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
 plt.axis((0,300,2,7))
 plt.savefig(pp.aoinst.cdir + targname + '_contrast_curve.png')
