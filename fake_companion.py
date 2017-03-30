from __future__ import division
import numpy as np,opticstools,matplotlib.pyplot as plt,math,scipy.ndimage,copy
from skimage.measure import block_reduce
import astropy.io.fits as pyfits
import os,sys,contratio as crat
nameList = sys.argv[5:len(sys.argv)]
if len(sys.argv)<4:
    print('Useage: crat_from_object.py raw_directory cube_directory plot_directory num_cals object_name (with spaces)')
    sys.exit()

#Combine name into single string
name = ''
if len(nameList)>1:
    for ii in range(0,len(nameList)):
        name+=nameList[ii]
        if ii<len(nameList)-1:
            name+=' '
#Remove Spaces From Object Name
objNoSpaces = name.split(' ')
objName = ''.join(objNoSpaces)
rawDir = sys.argv[1]
plotDir = sys.argv[3]
infoFile = open(rawDir+'/blockinfo.txt','r')
elements = []
ii = 0
lineNums = []
all_elements = []
cubeDir = sys.argv[2]
"""Code to select target cubes based on object name and calibration cubes close to the target cubes"""
#Find line and frame numbers where the name appears
for line in infoFile:
    ii+=1
    if ii==1:
        continue
    entry = line.split(' ')
    if name in line and os.path.isfile(cubeDir+'/cube'+str(int(entry[0]))+'.fits'):
        elements.append(int(entry[0]))
        lineNums.append(ii)
    all_elements.append(int(entry[0]))
cal_els = []
tgt_cubes = []
cal_cubes = []
#Create target cube list
for ii in range(0,len(elements)):
    tgt_cubes.append(cubeDir+'/cube'+str(elements[ii])+'.fits')
ii = 0
numCals = int(sys.argv[4])
#Find calibrators from objects nearby in the list.  Go one step in both directions and add frame to
#list of calibrators.
for kk in range(0,len(elements)):
    ii = lineNums[kk]
    jj = lineNums[kk]
    while ii>=0 or jj<len(all_elements):
        if ii>=0 and os.path.isfile(cubeDir+'/cube'+str(all_elements[ii-2])+'.fits') and ii not in lineNums:
            cal_els.append(all_elements[ii-2])
            ii-=1
        elif not os.path.isfile(cubeDir+'/cube'+str(all_elements[ii-2])+'.fits') or ii in lineNums:
            ii-=1
        if len(cal_els)==numCals:
            break
        if jj<len(all_elements) and os.path.isfile(cubeDir+'/cube'+str(all_elements[jj-2])+'.fits') and jj not in lineNums:
            cal_els.append(all_elements[jj-2])
            jj+=1
        elif jj>=len(all_elements) or not os.path.isfile(cubeDir+'/cube'+str(all_elements[jj-2])+'.fits') or jj in lineNums:
            jj+=1
        if len(cal_els)==numCals:
            break
    if len(cal_els)==numCals:
        break

tgt_cubes = []
cal_cubes = []
tgt_ims = []
cal_ims = []
pas = []
#Create target cube list
for ii in range(0,len(elements)):
    tgt_cubes.append(cubeDir+'/cube'+str(elements[ii])+'.fits')
    cube = pyfits.getdata(tgt_cubes[ii])
    pa = pyfits.getdata(tgt_cubes[ii],1)['pa']
    for jj in range(0,len(cube)):
        tgt_ims.append(cube[jj])
        pas.append(pa[jj])
#Create calibrator list
cal_objects = []
for ii in range(0,len(cal_els)):
    cal_cubes.append(cubeDir+'/cube'+str(cal_els[ii])+'.fits')
    cube = pyfits.getdata(cal_cubes[ii])
    cal = pyfits.getheader(cal_cubes[ii])['OBJECT']
    for jj in range(0,len(cube)):
        cal_ims.append(cube[jj])
        cal_objects.append(cal)
    
tgt_ims = np.array(tgt_ims)
cal_ims = np.array(cal_ims)

pas = np.array(pas)
"""Code to create artificial companion"""
sep = 20 #Companion Separation in pixels
contrast = 0.1 #Contrast Ratio
newIms = np.zeros(tgt_ims.shape)
for ii in range(0,tgt_ims.shape[0]):
	im = tgt_ims[ii]
	angle = pas[ii]
	#Put companion directly east of centre
	xShift = -int(sep*np.cos((math.pi/180)*angle))
	yShift = int(sep*np.sin((math.pi/180)*angle))
	newIms[ii] = im + contrast*np.roll(np.roll(im,xShift,axis=1),yShift,axis=0)
	
outfile = plotDir+'/cube_with_companion.fits'
oldHeader = pyfits.getheader(tgt_cubes[0])
header = pyfits.Header(oldHeader)
hdu = pyfits.PrimaryHDU(newIms,header)
col1 = pyfits.Column(name='pa', format='E', array=pas)
#col2 = pyfits.Column(name='paDiff', format='E', array=paDiff)
hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1]))
hdulist = pyfits.HDUList([hdu,hdu2])
hdulist.writeto(outfile,clobber=True)
good_ims = crat.choose_psfs([outfile],cal_cubes,plotDir)
crat_file = crat.best_psf_subtract(good_ims,plotDir)
