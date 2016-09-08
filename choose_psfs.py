"""This takes some cleaned cubes (from pynrm) and turns them into a single fits
cube containing target and calibrator images. Bad data can be manually rejected
by clicking less than x pixel number 20. """

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

outfile = 'good_ims.fits' #Change this name for your star!

tgt_cubes = ['cube333.fits']
cal_cubes = ['cube306.fits', 'cube322.fits','cube352.fits']

REJECT_COLUMN = 20

tgt_ims = []
cal_ims = []

#Reject target data
for fname in tgt_cubes:
    dd = pyfits.getdata(fname)
    for i in range(dd.shape[0]):
        plt.imshow(dd[i,:,:], interpolation='nearest')
        xy = plt.ginput(1)
        if (xy[0][0] > REJECT_COLUMN):
            tgt_ims.append(dd[i,:,:])
        else:
            print("Frame Rejected!")
            
#Reject calibrator data            
for fname in cal_cubes:
    dd = pyfits.getdata(fname)
    for i in range(dd.shape[0]):
        plt.imshow(dd[i,:,:], interpolation='nearest')
        xy = plt.ginput(1)
        if (xy[0][0] > REJECT_COLUMN):
            cal_ims.append(dd[i,:,:])
        else:
            print("Frame Rejected!")
 
tgt_ims = np.array(tgt_ims)
cal_ims = np.array(cal_ims)

#Now save the file!
hdu1 = pyfits.PrimaryHDU(tgt_ims)
hdu2 = pyfits.ImageHDU(cal_ims)
hdulist = pyfits.HDUList([hdu1,hdu2])
hdulist.writeto(outfile, clobber=True)