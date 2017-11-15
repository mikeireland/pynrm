import os,sys,os.path,numpy as np,copy
import contratio as crat
import astropy.io.fits as pyfits
import scipy.ndimage as nd,matplotlib.pyplot as plt
sys.path.append('/Users/awallace/Documents/pynrm/')
import stats
nameList = sys.argv[4:len(sys.argv)]
if len(sys.argv)<4:
    print('Useage: crat_all_images.py data_directory_location plot_directory num_cals_per_target object_name (with spaces)')
    sys.exit()
#Combine name into single string
name = ''
if len(nameList)>1:
    for ii in range(0,len(nameList)):
        name+=nameList[ii]
        if ii<len(nameList)-1:
            name+=' '
else:
	name = nameList[0]
#Remove Spaces From Object Name
objNoSpaces = name.split(' ')
objName = ''.join(objNoSpaces)
if sys.argv[1][0]=='/':
	dataDir = sys.argv[1]
else:
	dataDir = os.getcwd()+'/'+sys.argv[1]
if sys.argv[2][0]=='/':
	plotDir = sys.argv[2]
else:
	plotDir = os.getcwd()+'/'+sys.argv[2]
	
objDir = plotDir+'/'+objName
if not os.path.isdir(objDir):
	os.makedirs(objDir)
outfiles = []
#cycle through all existing data files
for year in range(11,20):
	for month in range(1,13):
		for day in range(1,32):
			directory = '%02d'%(year,)+'%02d'%(month,)+'%02d'%(day,)
			rawDir = dataDir+'/'+directory
			cubeDir = rawDir+'_cubes'
			if not os.path.isfile(rawDir+'/blockinfo.txt'):
				continue
			if not os.path.isdir(cubeDir):
				continue
			lineNums = []
			all_elements = []
			elements = []
			cal_els = []
			tgt_cubes = []
			cal_cubes = []
			total = -1
			allDirs = []
			infoFile = open(rawDir+'/blockinfo.txt','r')
			mm = 0
			count = 0
			#Only proceed if the given object was observed on this particular night
			for line1 in infoFile:
				if name in line1:
					count+=1
			if count<1:
				continue
			allDirs.append(rawDir)
			infoFile = open(rawDir+'/blockinfo.txt','r')
			#Find line and frame numbers where the name appears
			for line in infoFile:
				mm+=1
				if mm==1:
					continue
				total+=1
				entry = line.split(' ')
				newEntry = []
				for nn in range(0,len(entry)):
					if not entry[nn]=='':
						newEntry.append(entry[nn])
				#Find object name
				cubeFile = cubeDir+'/cube'+str(int(entry[0]))+'.fits'
				if os.path.isfile(cubeFile) and pyfits.getheader(cubeFile)['OBJECT']==name:
					elements.append([int(entry[0]),rawDir])
					lineNums.append([total,rawDir])
				all_elements.append([int(entry[0]),rawDir])
			#Create target cube list
			for ii in range(0,len(elements)):
				cubeDir = elements[ii][1]+'_cubes'
				tgt_cubes.append(cubeDir+'/cube'+str(elements[ii][0])+'.fits')

			#Find calibrators from objects nearby in the list.  Go one step in both directions and add frame to
			#list of calibrators.
			els = [ii for ii,jj in enumerate(tgt_cubes) if cubeDir in jj]
			#print(lineNums)
			numCals = int(sys.argv[3])*len(els)
			for kk in els:
				ii = lineNums[kk][0]
				jj = lineNums[kk][0]
				cubeDir = elements[kk][1]+'_cubes'
				while ii>=0 or jj<len(all_elements):
					if ii>=0 and os.path.isfile(cubeDir+'/cube'+str(all_elements[ii][0])+'.fits'):
						header = pyfits.getheader(cubeDir+'/cube'+str(all_elements[ii][0])+'.fits')
						if not all_elements[ii][1]==rawDir:
							break
						if not header['OBJECT']==name:
							cal_els.append(all_elements[ii])
						ii-=1
					elif not os.path.isfile(cubeDir+'/cube'+str(all_elements[ii][0])+'.fits'):
						ii-=1
					if len(cal_els)>=int(sys.argv[3])*(kk+1):
						break
					if jj<len(all_elements) and os.path.isfile(cubeDir+'/cube'+str(all_elements[jj][0])+'.fits'):
						header = pyfits.getheader(cubeDir+'/cube'+str(all_elements[jj][0])+'.fits')
						if not all_elements[jj][1]==rawDir:
							continue
						if not header['OBJECT']==name:
							cal_els.append(all_elements[jj])
						jj+=1
					elif jj>=len(all_elements) or not os.path.isfile(cubeDir+'/cube'+str(all_elements[jj][0])+'.fits'):
						jj+=1
					if len(cal_els)>=int(sys.argv[3])*(kk+1):
						break			
			tgt_ims = []
			cal_ims = []
			pas = []
			#Make sure there are no duplicates
			cal_els.sort()
			new_els = []
			for ii in range(0,len(cal_els)):
				if ii==0:
					new_els.append(cal_els[ii])
				elif not cal_els[ii]==cal_els[ii-1]:
					new_els.append(cal_els[ii])
			cal_els = new_els
			#Create target cube list
			new_cubes = []
			tgt_els = []
			for ii in range(0,len(tgt_cubes)):
				cube = pyfits.getdata(tgt_cubes[ii])
				pa = pyfits.getdata(tgt_cubes[ii],1)['pa']
				for jj in range(0,len(cube)):
					image = cube[jj]
					if (np.max(image)-np.mean(image))/np.std(image)>19:
						tgt_ims.append(image)
						pas.append(pa[jj])
						new_cubes.append(tgt_cubes[ii])
						tgt_els.append(jj)
			#Create calibrator list
			cal_objects = []
			for ii in range(0,len(cal_els)):
				cubeDir = cal_els[ii][1]+'_cubes'
				cal_cubes.append(cubeDir+'/cube'+str(cal_els[ii][0])+'.fits')
				cube = pyfits.getdata(cal_cubes[ii])
				num = 0
				for jj in range(0,len(cube)):
					image = cube[jj]
					if (np.max(image)-np.mean(image))/np.std(image)>19:
						num+=1
				for jj in range(0,len(cube)):
					image = cube[jj]
					cal = [pyfits.getheader(cal_cubes[ii])['OBJECT'],cubeDir+'/cube'+str(cal_els[ii][0])+'.fits',str(num),str(jj)]
					if (np.max(image)-np.mean(image))/np.std(image)>19:
						cal_ims.append(image)
						cal_objects.append(cal)

			tgt_ims = np.array(tgt_ims)
			cal_ims = np.array(cal_ims)
			all_ims = []
			#Restructure tgt_cubes vector so there is an entry for each image
			#When this is stored in the FITS file, it shows where each image came from
			tgt_cubes = new_cubes
			#Store target and calibration images in one file
			tempFile = plotDir+'/temp.fits'
			header = pyfits.getheader(tgt_cubes[0])
			radec = [header['RA'],header['DEC']]
			hdu1 = pyfits.PrimaryHDU(tgt_ims, header)
			hdu2 = pyfits.ImageHDU(cal_ims)
			col1 = pyfits.Column(name='pa', format='E', array=pas)
			col2 = pyfits.Column(name='cal_objects', format='A40', array=[row[0] for row in cal_objects])
			col3 = pyfits.Column(name='cal_cubes', format='A200', array=[row[1] for row in cal_objects])
			col4 = pyfits.Column(name='cal_lengths', format='A40', array=[row[2] for row in cal_objects])
			col5 = pyfits.Column(name='cal_els', format='A40', array=[row[3] for row in cal_objects])
			col6 = pyfits.Column(name='tgt_cubes', format='A200', array=tgt_cubes)
			col7 = pyfits.Column(name='tgt_els', format='E', array=tgt_els)
			hdu3 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7]))
			hdulist = pyfits.HDUList([hdu1,hdu2,hdu3])
			hdulist.writeto(tempFile, clobber=True)
			#Create contrast ratio plots (one crat file for each night this object was observed)
			crat_file = crat.best_psf_subtract(tempFile,plotDir)
			#Clean up and rename crat file to something unique
			os.system('rm -rf '+tempFile)
			os.system('mv '+crat_file+' crat'+str(year)+str(month)+str(day)+'.fits')
			outfiles.append('crat'+str(year)+str(month)+str(day)+'.fits')

crat_ims = []
pas = []
cal_objects = []
cal_cubes = []
cal_lengths = []
cal_elements = []
tgt_cubes = []
oldHeader = pyfits.getheader(outfiles[0])
sizes = []
tgt_elements = []
#Extract information from crat files creating image array
for ii in range(0,len(outfiles)):
	infile = outfiles[ii]
	ims = pyfits.getdata(infile)
	extra = pyfits.getdata(infile,1)
	for jj in range(0,len(ims)):
		crat_ims.append(ims[jj])
		sizes.append(ims[jj].shape[0])
		pas.append(extra['pa'][jj])
		cal_objects.append(extra['cal_objects'][jj])
		cal_cubes.append(extra['cal_cubes'][jj])
		cal_elements.append(extra['cal_elements'][jj])
		tgt_cubes.append(extra['tgt_cubes'][jj])
		tgt_elements.append(extra['tgt_elements'][jj])
	os.system('rm -rf '+infile)
bigSize = max(sizes)
#Make all crat images same size by zooming out if necessary
for ii in range(0,len(crat_ims)):
	if crat_ims[ii].shape[0]==bigSize:
		continue
	smallSize = crat_ims[ii].shape[0]
	newArray = np.zeros((bigSize,bigSize))
	newArray[bigSize//2-smallSize//2:bigSize//2+smallSize//2,bigSize//2-smallSize//2:bigSize//2+smallSize//2] = crat_ims[ii]
	crat_ims[ii] = newArray
	
dates = []
#Create array showing the date each image was taken
for ii in range(0,len(tgt_cubes)):
	for jj in range(0,len(tgt_cubes[ii])-11):
		if tgt_cubes[ii][jj:jj+11]=='_cubes/cube':
			break
	dateString = tgt_cubes[ii][jj-6:jj]
	months = 'JANFEBMARAPRMAYJUNJULAUGSEPOCTNOVDEC'
	date = str(int(dateString[4:6])) + '-' + months[3*(int(dateString[2:4])-1):3*int(dateString[2:4])] + '-20' + dateString[0:2]
	dates.append(date)

#Store all crat information in one file
sz = crat_ims[0].shape[0]
header = pyfits.Header(oldHeader)
header['CRVAL1']=header['RA']
header['CRVAL2']=header['DEC']
header['CRPIX1']=sz//2
header['CRPIX2']=sz//2
header['CDELT1']=-1./(3600*1024)
header['CDELT2']=1./(3600*1024)
header['CTYPE1']='RA---TAN'
header['CTYPE2']='DEC--TAN'
header['CD1_1']=-0.01/3600.
header['CD2_2']=0.01/3600.
header['CD1_2']=0
header['CD2_1']=0
header['OBJECT']=oldHeader['OBJECT']
hdu = pyfits.PrimaryHDU(crat_ims,header)
col1 = pyfits.Column(name='pa', format='E', array=pas)
col2 = pyfits.Column(name='cal_objects', format='A40', array=cal_objects)
col3 = pyfits.Column(name='cal_cubes', format='A200', array=cal_cubes)
col4 = pyfits.Column(name='cal_elements', format='E', array=cal_elements)
col5 = pyfits.Column(name='tgt_cubes', format='A200', array=tgt_cubes)
col6 = pyfits.Column(name='tgt_indices', format='E', array=tgt_elements)
col7 = pyfits.Column(name='observation_dates', format='A40', array=dates)
hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7]))
hdulist = pyfits.HDUList([hdu,hdu2])
hdulist.writeto(crat_file,clobber=True)
crat_im = pyfits.getdata(crat_file)
pas = pyfits.getdata(crat_file,1)['pa']
#Take average over frames
result = np.mean(crat_im,axis=0)
oldSize = pyfits.getdata(tgt_cubes[0]).shape[1]
"""newPA = np.mean(pas)
#Set up rotation angle so image is approximately aligned
#Make sure north is never more than 45 degrees from vertical
newRot = newPA
while newRot>180:
    newRot-=360
while newRot<-180:
    newRot+=360
newRot = 90-newRot
while newRot>45 or newRot<-45:
    if newRot<0:
        newRot+=90
    elif newRot>0:
        newRot-=90
    
result = nd.rotate(result,newRot,reshape=True)
size = result.shape[0]
sz = pyfits.getdata(tgt_cubes[0]).shape[1]
#Resize the result so it is the same size as original image
result = result[size//2-sz//2:size//2+sz//2,size//2-sz//2:size//2+sz//2]"""
#Store average plot in FITS file
hdu = pyfits.PrimaryHDU(result)
#costerm = np.cos(np.radians(newRot))*0.01/3600.
#sinterm = np.sin(np.radians(newRot))*0.01/3600.
header = pyfits.getheader(tgt_cubes[0])
hdu.header = header
hdu.header['CRVAL1']=radec[0]
hdu.header['CRVAL2']=radec[1]
hdu.header['CTYPE1']='RA---TAN'
hdu.header['CTYPE2']='DEC--TAN'
hdu.header['CRPIX1']=sz//2
hdu.header['CRPIX2']=sz//2
hdu.header['CD1_1']=-0.01/3600.
hdu.header['CD2_2']=0.01/3600.
hdu.header['CD1_2']=0
hdu.header['CD2_1']=0
hdu.header['OBJECT']=name
#hdu.header['RADECSYS']='FK5'
hdulist = pyfits.HDUList([hdu])
hdulist.writeto(objDir+'/ave_crat_'+objName+'_'+str(oldSize)+'.fits', clobber=True)
#Find weighted mean
mean_im = stats.weighted_mean(crat_im)
std_im = stats.bootstrap(crat_im,100)
std_im[np.where(std_im==0)] = 1e9
significance = mean_im/std_im

"""mean_im = nd.rotate(mean_im,newRot,reshape=True)
size = mean_im.shape[0]
mean_im = mean_im[size//2-sz//2:size//2+sz//2,size//2-sz//2:size//2+sz//2]
"""
hdu2 = pyfits.PrimaryHDU(mean_im)
hdu2.header = hdu.header
hdulist = pyfits.HDUList([hdu2])
hdulist.writeto(objDir+'/weighted_mean_'+objName+'_'+str(oldSize)+'.fits', clobber=True)
"""significance = nd.rotate(significance,newRot,reshape=True)
size = significance.shape[0]
significance = significance[size//2-sz//2:size//2+sz//2,size//2-sz//2:size//2+sz//2]"""
#significance[np.where(significance>100*np.median(significance))] = 0
#significance[np.where(np.isnan(significance))] = 0
hdu3 = pyfits.PrimaryHDU(significance)
hdu3.header = hdu.header
hdulist = pyfits.HDUList([hdu3])
hdulist.writeto(objDir+'/significance_'+objName+'_'+str(oldSize)+'.fits', clobber=True)