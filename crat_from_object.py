import os,sys,os.path,numpy as np
import contratio as crat
import astropy.io.fits as pyfits
import scipy.ndimage as nd
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
#cal_els = [299]
for ii in range(0,len(cal_els)):
    cal_cubes.append(cubeDir+'/cube'+str(cal_els[ii])+'.fits')
    cube = pyfits.getdata(cal_cubes[ii])
    cal = pyfits.getheader(cal_cubes[ii])['OBJECT']
    for jj in range(0,len(cube)):
        cal_ims.append(cube[jj])
        cal_objects.append(cal)
    
tgt_ims = np.array(tgt_ims)
cal_ims = np.array(cal_ims)
all_ims = []
tempFile = plotDir+'/temp.fits'
header = pyfits.getheader(tgt_cubes[0])
radec = [header['RA'],header['DEC']]
hdu1 = pyfits.PrimaryHDU(tgt_ims, header)
hdu2 = pyfits.ImageHDU(cal_ims)
col1 = pyfits.Column(name='pa', format='E', array=pas)
col2 = pyfits.Column(name='cal_objects', format='A40', array=cal_objects)
hdu3 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2]))
hdulist = pyfits.HDUList([hdu1,hdu2,hdu3])
hdulist.writeto(tempFile, clobber=True)

crat_file = crat.best_psf_subtract(tempFile,plotDir)
os.system('rm -rf '+tempFile)

crat_im = pyfits.getdata(crat_file)
pas = pyfits.getdata(crat_file,1)['pa']
result = np.mean(crat_im,axis=0)
newPA = np.mean(pas)
#Set up rotation angle so image is approximately aligned
#Make sure north is never more than 60 degrees from vertical
newRot = newPA
while newRot>180:
    newRot-=360
while newRot<-180:
    newRot+=360
newRot = 90-newRot
while newRot>60 or newRot<-60:
    if newRot<0:
        newRot+=90
    elif newRot>0:
        newRot-=90
    
result = nd.rotate(result,newRot,reshape=True)
size = result.shape[0]
sz = 128
result = result[size//2-sz//2:size//2+sz//2,size//2-sz//2:size//2+sz//2]
hdu = pyfits.PrimaryHDU(result)
costerm = np.cos(np.radians(newRot))*0.01/3600.
sinterm = np.sin(np.radians(newRot))*0.01/3600.
header = pyfits.getheader(tgt_cubes[0])
hdu.header = header
hdu.header['CRVAL1']=radec[0]
hdu.header['CRVAL2']=radec[1]
hdu.header['CTYPE1']='RA---TAN'
hdu.header['CTYPE2']='DEC--TAN'
hdu.header['CRPIX1']=sz//2
hdu.header['CRPIX2']=sz//2
hdu.header['CD1_1']=-costerm
hdu.header['CD2_2']=costerm
hdu.header['CD1_2']=sinterm
hdu.header['CD2_1']=sinterm
hdu.header['OBJECT']=name
#hdu.header['RADECSYS']='FK5'
hdulist = pyfits.HDUList([hdu])
hdulist.writeto(plotDir+'/ave_crat_'+objName+'.fits', clobber=True)
