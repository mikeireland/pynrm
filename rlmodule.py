from __future__ import print_function, division
import numpy as np, astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os,sys
import psf_marginalise as pm
import scipy.ndimage as nd

def deconvolve(tgt_cubes,cal_cubes,savedir):
    header = pyfits.getheader(tgt_cubes[0])
    radec = [header['RA'],header['DEC']]
    #pa = np.mean(pyfits.getdata(filename,1)['pa'])
    name = header['OBJECT']
    objNoSpaces = name.split(' ')
    objName = ''.join(objNoSpaces)
    #dir = '/Users/mireland/tel/nirc2/redux/HD169142/2016/'
    #filename = dir + 'good_ims.fits'
    #pa = 329.0 # From np.mean(pyfits.getdata('cube333.fits',1)['pa'])

    subtract_median=True

    #Should be automatic from here. Things to play with include niter, and the initial
    #model
    niter = 100
    tgt_ims = []
    cal_ims = []
    pas = []
    for ii in range(0,len(tgt_cubes)):
        cube = pyfits.getdata(tgt_cubes[ii])
        for jj in range(0,len(cube)):
            tgt_ims.append(cube[jj])
            pas.append(pyfits.getdata(tgt_cubes[ii],1)['pa'][jj])
    cal_objects = []
    cal_elements = []
    cubes = []
    for ii in range(0,len(cal_cubes)):
        cube = pyfits.getdata(cal_cubes[ii])
        header = pyfits.getheader(cal_cubes[ii])
        for jj in range(0,len(cube)):
            cal_ims.append(cube[jj])
            cal_elements.append(jj)
            cal_objects.append(header['OBJECT'])
            cubes.append(cal_cubes[ii])
    tgt_ims = np.array(tgt_ims)
    cal_ims = np.array(cal_ims)
    sz = tgt_ims.shape[1]
    best_models = np.zeros( tgt_ims.shape )
    best_rms = np.zeros( tgt_ims.shape[0] )
    if subtract_median:
        for i in range(len(cal_ims)):
            for j in range(len(cal_ims[i])):
                cal_ims[i][j] -= np.median(cal_ims[i])
        for i in range(len(tgt_ims)):
            for j in range(len(tgt_ims[i])):
                tgt_ims[i][j] -= np.median(tgt_ims[i])

    #Loop through all target images, and make the best deconvolution possible for each image.
    result = []
    used_elements = []
    used_cals = []
    used_cubes = []
    for i in range(tgt_ims.shape[0]):
        print("Working on image {0:d}".format(i))
        #Create a blank model image
        model_ims = np.zeros( cal_ims.shape )
        #Create a blank array of RMS of the model fits to the data
        rms = np.zeros( cal_ims.shape[0] )
        #Extract the data image from the cube and normalise it
        data = tgt_ims[i,:,:]
        data /= np.sum(data)
        #In order for RL deconvolution to work, we need to put in a background offset for 
        #flux. We'll subtract this at the end.
        data += 1.0/data.size
        #Find the peak pixel in the data.
        max_ix_data = np.argmax(data)
        max_ix_data = np.unravel_index(max_ix_data,data.shape)
        #Try to deconvolve with each calibrator image one at a time.
        for j in range(cal_ims.shape[0]):
            #Extract and normalise the Point-Spread Function
            psf = cal_ims[j,:,:]
            psf /= np.sum(psf)
            #Find the maximum pixel for the PSF, and roll the PSF so that (0,0) is the
            #peak pixel.
            max_ix = np.argmax(psf)
            max_ix = np.unravel_index(max_ix, psf.shape)
            psf = np.roll(np.roll(psf, -max_ix[0], axis=0), -max_ix[1],axis=1)
            #To save computational time, pre-compute the Fourier transform of the PSF
            psf_ft = np.fft.rfft2(psf)
            #The initial model just has a "star" at the location of the data maximum
            model = np.zeros(data.shape)
            model += 1.0/data.size
            model[max_ix_data] = 1.0
            #Do the RL magical algorithm. See 
            for k in range(niter):
                # u (convolved) p is our model of the data. Compute this first.
                model_convolved = np.fft.irfft2(np.fft.rfft2(model)*psf_ft)
                # Update the model according to the RL algorithm
                model *= np.fft.irfft2(np.fft.rfft2(data / model_convolved)*np.conj(psf_ft))
            model_convolved = np.fft.irfft2(np.fft.rfft2(model)*psf_ft)
            #Record the RMS difference between the model and the data.
            rms[j] = np.sqrt(np.mean( (model_convolved - data)**2)) * data.size 
            #Subtract off our offset.   
            model -= 1.0/data.size   
            #Shift the final model to the middle, so we can add together target images on
            #different pixel coordinates 
            model_ims[j,:,:] = np.roll(np.roll(model,sz//2-max_ix_data[0], axis=0), sz//2-max_ix_data[1], axis=1)
        #Only use the calibrator with the best RMS. i.e. we assume this is the best PSF for our data.
        best_cal = np.argmin(rms)
        best_models[i,:,:] = model_ims[best_cal,:,:]
        used_elements.append(cal_elements[best_cal])
        used_cubes.append(cubes[best_cal])
        used_cals.append(cal_objects[best_cal])
        best_models[i,sz//2,sz//2]=0
        result.append(nd.rotate(best_models[i,:,:],-pas[i],reshape=True)[::-1,:])
        best_rms[i] = rms[best_cal]
    sizes = np.zeros(len(result))
    for ii in range(0,len(result)):
        sizes[ii] = int(len(result[ii][0]))
    newSize = int(np.min(sizes))
    for ii in range(0,len(result)):
        result[ii] = np.array(result[ii])
        start = int(sizes[ii]//2-newSize//2)
        end = int(sizes[ii]//2+newSize//2)+1
        result[ii] = result[ii][start:end,start:end]

    best_models = np.array(result)
    sz = best_models.shape[1]
    ptsrc_fluxes = best_models[:,sz//2,sz//2].copy()
    #set the central pixel to zero.
    best_models[:,sz//2,sz//2]=0
    final_image = np.mean(best_models,axis=0)

    image = final_image/np.max(final_image)
    image_sub = image - np.roll(np.roll(image[::-1,::-1],1,axis=0),1,axis=1)

    image[sz//2,sz//2]=1.0
    image_sub[sz//2,sz//2]=1.0

    tic_min = np.min(image)
    tic_max = np.max(image)
    tics = np.arcsinh(tic_min/0.1) + np.arange(8)/7.0*(np.arcsinh(tic_max/0.1) - np.arcsinh(tic_min/0.1))
    tics = np.sinh(tics)*0.1
    used_elements = np.array(used_elements)
    #used_cubes = np.array(used_cubes)
    #used_cals = np.array(used_cals)
    hdu = pyfits.PrimaryHDU(best_models)
    hdu.header['CRVAL1']=radec[0]
    hdu.header['CRVAL2']=radec[1]
    hdu.header['CTYPE1']='RA---TAN'
    hdu.header['CTYPE2']='DEC--TAN'
    hdu.header['CRPIX1']=sz//2
    hdu.header['CRPIX2']=sz//2
    hdu.header['CDELT1']=-1./(3600*1024)
    hdu.header['CDELT2']=1./(3600*1024)
    #hdu.header['RADECSYS']='FK5'
    col1 = pyfits.Column(name='pa', format='E', array=pas)
    col2 = pyfits.Column(name='cal_cubes', format='A40', array=used_cubes)
    col3 = pyfits.Column(name='cal_ims', format='I', array=used_elements)
    col4 = pyfits.Column(name='cal_objects', format='A40', array=used_cals)
    hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1,col2,col3,col4]))
    hdulist = pyfits.HDUList([hdu,hdu2])
    hdu.data = best_models
    outfile = savedir+'/rl_deconvolve_'+objName+'.fits'
    hdulist.writeto(outfile, clobber=True)
    return outfile
