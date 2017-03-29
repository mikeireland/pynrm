from __future__ import division
import numpy as np,opticstools,matplotlib.pyplot as plt,math,scipy.ndimage,copy
from skimage.measure import block_reduce
import astropy.io.fits as pyfits
import os,sys,contratio as crat
filenames = []
data = []
pas = []
tgt = []
cal = []
hdr = pyfits.getheader(sys.argv[1])
object = hdr['OBEJECT']
if len(sys.argv)<3:
	print('Useage: fake_companion.py target1 [target2 ...] calibrator1 [calibrator2 ...]')
	sys.exit()
for ii in range(1,len(sys.argv)):
	hdr = pyfits.getheader(sys.argv[ii])
	current_object = hdr['OBEJECT']
	if current_object==object:
		tgt.append(sys.argv[ii])
	else:
		cal.append(sys.argv[ii])
		
for ii in range(0,len(tgt)):
	filenames.append(tgt[ii])
	cube = pyfits.getdata(tgt[ii])
	extra = pyfits.getdata(tgt[ii],1)
	for jj in range(0,cube.shape[0]):
		pas.append(extra['pa'][jj])
		data.append(cube[jj])
		
data = np.array(data)
pas = np.array(pas)
sep = 40 #Companion Separation in pixels
contrast = 0.01 #Contrast Ratio
newIms = np.zeros(data.shape)
for ii in range(0,data.shape[0]):
	im = data[ii]
	angle = pas[ii]
	#Put companion directly east of centre
	xShift = -int(sep*np.cos((math.pi/180)*angle))
	yShift = int(sep*np.sin((math.pi/180)*angle))
	newIms[ii] = im + contrast*np.roll(np.roll(im,xShift,axis=1),yShift,axis=0)
	
outfile = 'cube_with_companion.fits'
oldHeader = pyfits.getheader(filenames[0])
header = pyfits.Header(oldHeader)
hdu = pyfits.PrimaryHDU(newIms,header)
col1 = pyfits.Column(name='pa', format='E', array=pas)
col2 = pyfits.Column(name='paDiff', format='E', array=paDiff)
hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2]))
hdulist = pyfits.HDUList([hdu,hdu2])
hdulist.writeto(outfile,clobber=True)
good_ims = crat.choose_psfs([outfile],cal,'./')
crat_file = crat.best_psf_subtract(good_ims,'./')
