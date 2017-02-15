import os,sys,os.path,numpy as np
import rlmodule as rl
import psf_marginalise as pm
import scipy.ndimage as nd
import astropy.io.fits as pyfits
nameList = sys.argv[4:len(sys.argv)]
if len(sys.argv)<5:
    print('Useage: rl_from_object.py raw_directory cube_directory plot_directory object_name (with spaces)')
    sys.exit()
#Combine name into single string
name = ''
if len(nameList)>1:
    for ii in range(0,len(nameList)):
        name+=nameList[ii]
        if ii<len(nameList)-1:
            name+=' '
rawDir = sys.argv[1]
plotDir = sys.argv[3]
infoFile = open(rawDir+'/blockinfo.txt','r')
elements = []
ii = 0
lineNums = []
all_elements = []
cubeDir = sys.argv[2]
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
numCals = 3*len(elements)
#Find calibrators from objects nearby in the list.  Go one step in both directions and add frame to
#list of calibrators.  Continue until there are three times as many calibrator cubes as target cubes
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
#Create list of calibrator cubes
for ii in range(0,len(cal_els)):
    cal_cubes.append(cubeDir+'/cube'+str(cal_els[ii])+'.fits')

deconv_file = rl.deconvolve(tgt_cubes,cal_cubes,plotDir)

