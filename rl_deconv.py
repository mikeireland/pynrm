"""This makes a simple Richardson-Lucy deconvolution on a cleaned data cube, with
some reference calibrator images. Input data have to be neatly packaged in a single
data cube. 

To make a "good_ims.fits" file, run "choose_psfs.py" after cleaning the data
(e.g. with process_block called in a script go.py or run_clean)."""

from __future__ import print_function, division

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import aplpy
import opticstools as ot
plt.ion()

#Inputs here...
dir = '/Users/mireland/tel/nirc2/redux/IRS48/'
filename = dir + 'good_ims.fits'
pa = 317.4 # From np.mean(pyfits.getdata('cube333.fits',1)['pa'])
radec = [246.9049584,-23.49026944]

radec = [276.124,-29.780]
radec = [15*(18 + 24/60. + 29.76/3600.),-29 - 46/60. - 47.7/3600]
dir = '/Users/mireland/tel/nirc2/redux/HD169142/2014/'
filename = dir + 'good_ims.fits'
pa = 337.5 # From np.mean(pyfits.getdata('cube333.fits',1)['pa'])

#dir = '/Users/mireland/tel/nirc2/redux/HD169142/2016/'
#filename = dir + 'good_ims.fits'
#pa = 329.0 # From np.mean(pyfits.getdata('cube333.fits',1)['pa'])

subtract_median=True

#Should be automatic from here. Things to play with include niter, and the initial
#model
niter = 50
tgt_ims = pyfits.getdata(filename)
cal_ims = pyfits.getdata(filename,1)

sz = tgt_ims.shape[1]
best_models = np.zeros( tgt_ims.shape )
best_rms = np.zeros( tgt_ims.shape[0] )

if subtract_median:
    for i in range(len(cal_ims)):
        cal_ims[i] -= np.median(cal_ims[i])
    for i in range(len(tgt_ims)):
        tgt_ims[i] -= np.median(tgt_ims[i])

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
image_sub = image - np.roll(np.roll(image[::-1,::-1],1,axis=0),1,axis=1)

image[sz//2,sz//2]=1.0
image_sub[sz//2,sz//2]=1.0

plt.imshow(np.arcsinh(image/0.1), interpolation='nearest', cmap=cm.cubehelix)
plt.plot(sz//2,sz//2, 'r*', markersize=20)
tic_min = np.min(image)
tic_max = np.max(image)
tics = np.arcsinh(tic_min/0.1) + np.arange(8)/7.0*(np.arcsinh(tic_max/0.1) - np.arcsinh(tic_min/0.1))
tics = np.sinh(tics)*0.1

hdu = pyfits.PrimaryHDU(image)
costerm = np.cos(np.radians(pa))*0.01/3600.
sinterm = np.sin(np.radians(pa))*0.01/3600.
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
#hdu.header['RADECSYS']='FK5'
hdulist = pyfits.HDUList([hdu])

hdu.data = image
hdulist.writeto('deconv_image.fits', clobber=True)
fig = aplpy.FITSFigure('deconv_image.fits')
fig.show_colorscale(cmap=cm.cubehelix, stretch='arcsinh',vmax=1, vmid=0.05)
fig.add_colorbar()
fig.add_grid()

hdu.data=image_sub
hdulist.writeto('deconv_image_sub.fits', clobber=True)
fig2 = aplpy.FITSFigure('deconv_image_sub.fits')
fig2.show_colorscale(cmap=cm.cubehelix, stretch='arcsinh',vmax=1, vmid=0.05)
fig2.add_colorbar()
fig2.add_grid()

fig3 = aplpy.FITSFigure('deconv_image.fits')
fig3.show_colorscale(cmap=cm.cubehelix, stretch='linear',vmax=1, vmin=0.0)
fig3.add_colorbar()
fig3.add_grid()

plt.figure(1)
plt.clf()
rr, ii = ot.azimuthalAverage(image,returnradii=True,center=[64,64],binsize=0.7)
plt.plot(rr*0.01,ii)
plt.axis([0,.3,-0.05,0.8])
plt.xlabel('Radius (arcsec)')
plt.ylabel('Azi. Ave. Intensity (rel. to disk peak)')
plt.plot([0.11,0.11],[-0.1,1],'r')
plt.plot([0.17,0.17],[-0.1,1],'r')
plt.annotate("Companion Radius", [0.11,0.6],[0.18,0.6],arrowprops={"arrowstyle":"->"})
plt.annotate("Wall Radius", [0.17,0.3],[0.2,0.3],arrowprops={"arrowstyle":"->"})