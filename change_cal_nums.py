from __future__ import division
import numpy as np,opticstools as ot,matplotlib.pyplot as plt,math,scipy.ndimage as nd
import astropy.io.fits as pyfits
import os,sys,aplpy
from matplotlib import ticker
import matplotlib.cm as cm
numCals = [4,8,12]
name = 'HL Tau'
objNoSpaces = name.split(' ')
objName = ''.join(objNoSpaces)
separations = []
curves = []
path_to_pynrm = '/Users/awallace/Documents/'
ddir = '/Users/awallace/Documents/161107'
cdir = '/Users/awallace/Documents/161107_cubes'
savedir = '.'
for ii in range(0,len(numCals)):
	num = numCals[ii]
	#For each number of calibrators, run code which finds contrast ratio
	os.system('python '+path_to_pynrm+'pynrm/crat_from_object.py '+ddir+' '+cdir+' '+savedir+' '+str(num)+' '+name)
	newfile = savedir+'/ave_crat_'+objName+'.fits'
	hdr = pyfits.getheader(newfile)
	data = pyfits.getdata(newfile)
	azAve = ot.azimuthalAverage(data, returnradii=True, binsize=1, center=[64,64])
	separations.append(0.01*azAve[0])
	curves.append(azAve[1])
	fig = aplpy.FITSFigure(newfile)
	fig.show_colorscale(cmap=cm.cubehelix, vmin=-0.0005, vmax=0.8*np.max(data))
	fig.add_colorbar()
	#fig.add_grid()
	fig.savefig('image'+objName+'Cal'+str(num)+'.png')
	plt.plot(separations[ii],curves[ii],label=str(numCals[ii])+' Calibrators')

plt.ylim(0,0.002)
plt.xlabel('Separation (arcsec)')
plt.ylabel('Average Contrast Ratio')
plt.legend()
plt.title('Contrast Curves for '+name)
plt.savefig('calibration_nums.png')