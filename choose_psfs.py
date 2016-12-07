"""This takes some cleaned cubes (from pynrm) and turns them into a single fits
cube containing target and calibrator images. Bad data can be manually rejected
by clicking less than x pixel number 20. """

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

outfile = 'good_ims_LkCa15_1.fits' #Change this name for your star!

tgt_cubes = ['cube333.fits']
cal_cubes = ['cube306.fits', 'cube322.fits','cube352.fits']

dir = '/Users/mireland/tel/nirc2/redux/TauL15_4/'
tgt_cubes  = ['cube_n1352.fits'] # LkCa 15
cal_cubes = ['cube_n1293.fits','cube_n1310.fits','cube_n1316.fits','cube_n1322.fits',\
    'cube_n1334.fits','cube_n1340.fits','cube_n1346.fits','cube_n1358.fits','cube_n1370.fits']

dir = '/Users/mireland/tel/nirc2/redux/TauL15_2/'
tgt_cubes = ['cube_n0409.fits','cube_n0484.fits']#[,'cube_n0587.fits']
cal_cubes = ['cube_n0361.fits','cube_n0373.fits','cube_n0385.fits','cube_n0397.fits',\
'cube_n0436.fits','cube_n0448.fits','cube_n0424.fits','cube_n0472.fits','cube_n0460.fits']
#'cube_n0507.fits','cube_n0575.fits','cube_n0581.fits','cube_n0513.fits',\
#'cube_n0593.fits','cube_n0599.fits','cube_n0605.fits','cube_n0569.fits',]
						
REJECT_COLUMN = 20

tgt_ims = []
cal_ims = []

#Reject target data
bintab = []
header = pyfits.getheader(dir + tgt_cubes[0])
for fname in tgt_cubes:
    dd = pyfits.getdata(dir + fname)
    newtab = pyfits.getdata(dir + fname,1)
    for i in range(dd.shape[0]):
        plt.clf()
        plt.imshow(dd[i,:,:], interpolation='nearest')
        xy = plt.ginput(1)
        if (xy[0][0] > REJECT_COLUMN):
            tgt_ims.append(dd[i,:,:])
            if len(bintab):
                bintab = np.append(bintab, newtab[i:i+1])
            else:
                bintab =  newtab[i:i+1]
        else:
            print("Frame Rejected!")
            
#Reject calibrator data            
for fname in cal_cubes:
    dd = pyfits.getdata(dir + fname)
    for i in range(dd.shape[0]):
        plt.clf()
        plt.imshow(dd[i,:,:], interpolation='nearest')
        xy = plt.ginput(1)
        if (xy[0][0] > REJECT_COLUMN):
            cal_ims.append(dd[i,:,:])
        else:
            print("Frame Rejected!")
 
tgt_ims = np.array(tgt_ims)
cal_ims = np.array(cal_ims)

#Now save the file!
hdu1 = pyfits.PrimaryHDU(tgt_ims, header)
hdu2 = pyfits.ImageHDU(cal_ims)
hdu3 = pyfits.BinTableHDU(bintab)
hdulist = pyfits.HDUList([hdu1,hdu2,hdu3])
hdulist.writeto(dir + outfile, clobber=True)