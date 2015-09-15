"""This makes a simple Richardson-Lucy deconvolution on a cleaned data cube, with
some reference calibrator images. Input data have to be neatly packaged in a single
data cube. """

from __future__ import print_function, division

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import aplpy
plt.ion()

#Inputs here...
dir = '/Users/mireland/tel/nirc2/redux/IRS48/'
filename = dir + 'good_ims.fits'
pa = 317.4 # From np.mean(pyfits.getdata('cube333.fits',1)['pa'])

#Should be automatic from here. Things to play with include niter, and the initial
#model
niter = 50
tgt_ims = pyfits.getdata(filename)
cal_ims = pyfits.getdata(filename,1)

sz = tgt_ims.shape[1]
best_models = np.zeros( tgt_ims.shape )
best_rms = np.zeros( tgt_ims.shape[0] )

#Loop through all target images, and make the best deconvolution possible for each image.
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
    best_rms[i] = rms[best_cal]
    
ptsrc_fluxes = best_models[:,sz//2,sz//2].copy()
#set the central pixel to zero.
best_models[:,sz//2,sz//2]=0
final_image = np.mean(best_models,axis=0)

image = final_image/np.max(final_image)
plt.imshow(np.arcsinh(image/0.1), interpolation='nearest', cmap=cm.cubehelix)
plt.plot(sz//2,sz//2, 'r*', markersize=20)
tic_min = np.min(image)
tic_max = np.max(image)
tics = np.arcsinh(tic_min/0.1) + np.arange(8)/7.0*(np.arcsinh(tic_max/0.1) - np.arcsinh(tic_min/0.1))
tics = np.sinh(tics)*0.1

hdu = pyfits.PrimaryHDU(image)
costerm = np.cos(np.radians(pa))*0.01/3600.
sinterm = np.sin(np.radians(pa))*0.01/3600.
hdu.header['CRVAL1']=246.9049584
hdu.header['CRVAL2']=-23.49026944
hdu.header['CTYPE1']='RA---TAN'
hdu.header['CTYPE2']='DEC--TAN'
hdu.header['CRPIX1']=sz//2
hdu.header['CRPIX2']=sz//2
hdu.header['CD1_1']=-costerm
hdu.header['CD2_2']=costerm
hdu.header['CD1_2']=sinterm
hdu.header['CD2_1']=sinterm
#hdu.header['RADECSYS']='FK5'
hdulist = pyfits.HDUList([hdu])
hdulist.writeto('deconv_image.fits', clobber=True)
fig = aplpy.FITSFigure('deconv_image.fits')
fig.show_colorscale(cmap=cm.cubehelix, stretch='arcsinh',vmax=1, vmid=0.05)
fig.add_colorbar()
fig.add_grid()


