from __future__ import print_function, division
import numpy as np, astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os,sys
import psf_marginalise as pm
import scipy.ndimage as nd
def choose_psfs(tgt_cubes,cal_cubes,save_dir):
    objName = pyfits.getheader(tgt_cubes[0])['TARGNAME']
    #Remove Spaces From Object Name
    objNoSpaces = objName.split(' ')
    objName = ''.join(objNoSpaces)
    outfile = 'good_ims_'+objName+'.fits'
    REJECT_COLUMN = 20
    header = pyfits.getheader(tgt_cubes[0])
    tgt_ims = []
    cal_ims = []
    pas = []
    #Reject target data
    bintab = []
    header = pyfits.getheader(tgt_cubes[0])
    for fname in tgt_cubes:
        dd = pyfits.getdata(fname)
        newtab = pyfits.getdata(fname,1)
        for ii in range(0,dd.shape[0]):
        	tgt_ims.append(dd[ii,:,:])
        	pas.append(newtab['pa'][ii])
    pas = np.array(pas)
    #Reject calibrator data
    cal_objects = []  
    cal_lengths = []     
    cal_els = []    
    cubes = []
    jj = 0
    for fname in cal_cubes:
        dd = pyfits.getdata(fname)
        cal = pyfits.getheader(fname)['TARGNAME']
        for ii in range(0,dd.shape[0]):
        	cal_ims.append(dd[ii,:,:])
        	cal_lengths.append(str(dd.shape[0]))
        	cal_els.append(jj)
        	cal_objects.append(cal)
        	cubes.append(fname)
        jj+=1
    
    tgt_ims = np.array(tgt_ims)
    cal_ims = np.array(cal_ims)
    #Now save the file!
    col1 = pyfits.Column(name='pa', format='E', array=pas)
    col2 = pyfits.Column(name='cal_objects', format='A40', array=cal_objects)
    col3 = pyfits.Column(name='cal_cubes', format='A40', array=cubes)
    col4 = pyfits.Column(name='cal_lengths', format='A40', array=cal_lengths)
    col5 = pyfits.Column(name='cal_els', format='A40', array=cal_els)
    col6 = pyfits.Column(name='tgt_cubes', format='A40', array=tgt_cubes)
    hdu1 = pyfits.PrimaryHDU(tgt_ims, header)
    hdu2 = pyfits.ImageHDU(cal_ims)
    hdu3 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4,col5,col6]))
    hdulist = pyfits.HDUList([hdu1,hdu2,hdu3])
    hdulist.writeto(save_dir+'/'+outfile, clobber=True)
    return save_dir+'/'+outfile

def best_psf_subtract(filename,plotDir):
	#Create our target model and object model.
	psfs = pm.Psfs(cubefile=filename)
	object_model = pm.PtsrcObject()
	tgt = pm.Target(psfs, object_model, cubefile=filename)
	cal_objects = pyfits.getdata(filename,2)
	if not 'cal_objects' in cal_objects.names:
		cal_objects = pyfits.getdata(filename,3)
	best_ixs, best_fit_ims = tgt.find_best_psfs([])
	used_cals = []
	for ii in range(0,len(best_ixs)):
		used_cals.append([cal_objects['cal_objects'][best_ixs[ii]],cal_objects['cal_cubes'][best_ixs[ii]],int(cal_objects['cal_lengths'][best_ixs[ii]]),int(cal_objects['cal_els'][best_ixs[ii]])])
	all_sizes = []
	for ii in range(0,len(cal_objects['cal_objects'])):
		if len(all_sizes)>0 and cal_objects['cal_cubes'][ii] in all_sizes[len(all_sizes)-1]:
			continue
		all_sizes.append([cal_objects['cal_cubes'][ii],cal_objects['cal_lengths'][ii]])
	for ii in range(0,len(used_cals)):
		size = used_cals[ii][2]
		num = used_cals[ii][3]
		count = 0
		for jj in range(0,num):
			count+=int(all_sizes[jj][1])
		element = best_ixs[ii]-count
		used_cals[ii] = [used_cals[ii][0],used_cals[ii][1],element]
	
	objName = pyfits.getheader(filename)['TARGNAME']
	#Remove Spaces From Object Name
	objNoSpaces = objName.split(' ')
	objName = ''.join(objNoSpaces)
	imageBreaks = []
	#tgt_ims = [tgt.ims]
	#best_ims = [best_fit_ims]
	#tgt_pas = [tgt.pas]

	subims = tgt.ims - best_fit_ims
	subim_sum = np.sum(subims, axis=0) 
	#We want to compute: \Sum (subims[i] * shift(best_fit_ims[i])) for 
	#every possible shift value in 2D. This is just a cross-correlation!
	#We can use the power of numpy by putting this in a single line.
	#We are computing best_fit_ims [cross-correlation *] subims
	crats = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(best_fit_ims)) * np.fft.rfft2(subims) ), axes=[1,2])
	#Now rotate and normalise
	rot_crats = []
	for i in range(len(crats)):
		rot_crats.append(nd.rotate(crats[i], -tgt.pas[i])[::-1,:]/np.sum(best_fit_ims[i]**2))

	sizes = np.zeros(len(rot_crats))
	for ii in range(0,len(rot_crats)):
		sizes[ii] = int(len(rot_crats[ii][0]))
	newSize = int(np.min(sizes))
	for ii in range(0,len(rot_crats)):
		rot_crats[ii] = np.array(rot_crats[ii])
		start = int(sizes[ii]//2-newSize//2)
		end = int(sizes[ii]//2+newSize//2)
		rot_crats[ii] = rot_crats[ii][start:end,start:end]
	crats = np.array(rot_crats)
	stds = np.empty(len(tgt.ims))
	for i, im in enumerate(tgt.ims):
		stds[i] = np.std(im[tgt.corner_pix])
	crat_std = tgt.ims
	crat_errors = stds/np.sqrt(np.sum(best_fit_ims**2,axis=(1,2)))
	crat_stds = np.std(crats,axis=0)/len(crat_errors)
	sz = len(crats[0])
	outfile = plotDir+'/crats_'+objName+'.fits'
	oldHeader = pyfits.getheader(filename)
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
	header['OBJECT']=oldHeader['TARGNAME']
	hdu = pyfits.PrimaryHDU(crats,header)
	col1 = pyfits.Column(name='pa', format='E', array=tgt.pas[0:len(crats)])
	col2 = pyfits.Column(name='cal_objects', format='A40', array=[row[0] for row in used_cals])
	col3 = pyfits.Column(name='cal_cubes', format='A40', array=[row[1] for row in used_cals])
	col4 = pyfits.Column(name='cal_elements', format='E', array=[row[2] for row in used_cals])
	col5 = pyfits.Column(name='tgt_cubes', format='A40', array=cal_objects['tgt_cubes'])
	col6 = pyfits.Column(name='tgt_elements', format='E', array=cal_objects['tgt_els'])
	hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4,col5,col6]))
	hdulist = pyfits.HDUList([hdu,hdu2])
	hdulist.writeto(outfile,clobber=True)
	return outfile

def marginalise_image(filename,plotDir):
	#Create our target model and object model.
	psfs = pm.Psfs(cubefile=filename)
	object_model = pm.PtsrcObject()
	tgt = pm.Target(psfs, object_model, cubefile=filename)
	
	cal_objects = pyfits.getdata(filename,2)
	if not 'cal_objects' in cal_objects.names:
		cal_objects = pyfits.getdata(filename,3)
	best_ixs, best_fit_ims = tgt.find_best_psfs([])
	used_cals = []
	for ii in range(0,len(best_ixs)):
		used_cals.append([cal_objects['cal_objects'][best_ixs[ii]],cal_objects['cal_cubes'][best_ixs[ii]],int(cal_objects['cal_lengths'][best_ixs[ii]]),int(cal_objects['cal_els'][best_ixs[ii]])])
	all_sizes = []
	for ii in range(0,len(cal_objects['cal_objects'])):
		if len(all_sizes)>0 and cal_objects['cal_cubes'][ii] in all_sizes[len(all_sizes)-1]:
			continue
		all_sizes.append([cal_objects['cal_cubes'][ii],cal_objects['cal_lengths'][ii]])
	for ii in range(0,len(used_cals)):
		size = used_cals[ii][2]
		num = used_cals[ii][3]
		count = 0
		for jj in range(0,num):
			count+=int(all_sizes[jj][1])
		element = best_ixs[ii]-count
		used_cals[ii] = [cal_objects['cal_objects'][num],cal_objects['cal_cubes'][num],element]
	
	objName = pyfits.getheader(filename)['TARGNAME']
	#Remove Spaces From Object Name
	objNoSpaces = objName.split(' ')
	objName = ''.join(objNoSpaces)
	#Embed the PSFs
	psfs.lle(ndim=3) #For Tau15_4, ndim=3 is roughly enough.
	starting_lnprob = tgt.lnprob(np.zeros(psfs.ndim*tgt.n_ims))

	#Find the best matching PSFs.
	best_x, sampler = tgt.marginalise(nchain=400, use_threads=False)

	#Find the resulting PSFs
	best_fit_ims = tgt.lnprob(best_x, return_mod_ims=True)
	subims = tgt.ims - best_fit_ims

	#Find the rotated residuals
	rot_resid = np.empty_like(subims)
	for rr, si, pa in zip(rot_resid, subims, tgt.pas):
		rr = nd.rotate(si, -pa, reshape=False)[::-1,:]
	rot_resid_sum = np.sum(rot_resid,axis=0)/np.max(np.sum(tgt.ims, axis=0))

	crats = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(best_fit_ims)) * np.fft.rfft2(subims) ), axes=[1,2])
	#Now rotate and normalise
	rot_crats = []
	for i in range(len(crats)):
		rot_crats.append(nd.rotate(crats[i], -tgt.pas[i])[::-1,:]/np.sum(best_fit_ims[i]**2))
	
	sizes = np.zeros(len(rot_crats))
	for ii in range(0,len(rot_crats)):
		sizes[ii] = int(len(rot_crats[ii][0]))
	newSize = int(np.min(sizes))
	for ii in range(0,len(rot_crats)):
		rot_crats[ii] = np.array(rot_crats[ii])
		start = int(sizes[ii]//2-newSize//2)
		end = int(sizes[ii]//2+newSize//2)
		rot_crats[ii] = rot_crats[ii][start:end,start:end]
	crats = np.array(rot_crats)
	sz = len(crats[0])
	outfile = plotDir+'/crats_'+objName+'.fits'
	oldHeader = pyfits.getheader(filename)
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
	header['OBJECT']=oldHeader['TARGNAME']
	hdu = pyfits.PrimaryHDU(crats,header)
	col1 = pyfits.Column(name='pa', format='E', array=tgt.pas[0:len(crats)])
	col2 = pyfits.Column(name='cal_objects', format='A40', array=[row[0] for row in used_cals])
	col3 = pyfits.Column(name='cal_cubes', format='A40', array=[row[1] for row in used_cals])
	col4 = pyfits.Column(name='cal_elements', format='E', array=[row[2] for row in used_cals])
	col5 = pyfits.Column(name='tgt_cubes', format='A40', array=cal_objects['tgt_cubes'])
	hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4,col5]))
	hdulist = pyfits.HDUList([hdu,hdu2])
	hdulist.writeto(outfile,clobber=True)
	return outfile