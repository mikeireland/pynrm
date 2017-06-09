"""This makes a simple Richardson-Lucy deconvolution on a cleaned data cube, with
some reference calibrator images. Input data have to be neatly packaged in a single
data cube. 

To make a "good_ims.fits" file, run "choose_psfs.py" after cleaning the data
(e.g. with process_block called in a script go.py or run_clean)."""

from __future__ import print_function, division

import astropy.io.fits as pyfits
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import aplpy
import opticstools as ot
plt.ion()

def rl_deconv(tgt_fn=None, cal_fn=None, good_ims_fn=None, niter=50):
    """Deconvolve a target using RL deconvolution and either a pair of target and 
    calibrator cubes, or a set of manually selected "good" files
    
    Parameters
    ----------
    tgt_fn: string
        Name of the target cube file
        
    cal_fn: string
        Name of the calibrator cube file
        
    good_ims_fn: string
        Name of the "good_ims" filename, from choose_psfs.
    """
    if good_ims_fn is None:
        header = pyfits.getheader(tgt_fn)
        radec = [header['RA'],header['DEC']]
        pa = np.mean(pyfits.getdata(tgt_fn,1)['pa'])
        tgt_ims = pyfits.getdata(tgt_fn)
        cal_ims = pyfits.getdata(cal_fn)
    else:
        header = pyfits.getheader(good_ims_fn)
        radec = [header['RA'],header['DEC']]
        pas = pyfits.getdata(good_ims_fn,2)['pa']
        #Check for too much sky rotation.
        pa_diffs = pas - pas[0]
        pa_diffs = ((pa_diffs + 180) % 360) - 180
        if np.max(np.abs(pa_diffs)) > 30:
            raise UserWarning("Too much sky rotation! Re-write code or reduce number of files.")
        #Average the pas modulo 360
        pa = pas[0] + np.mean(pa_diffs)
        tgt_ims = pyfits.getdata(good_ims_fn, 0)
        cal_ims = pyfits.getdata(good_ims_fn, 1)

    subtract_median=True

    
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
