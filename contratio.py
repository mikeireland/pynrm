from __future__ import print_function, division
import numpy as np, astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os,sys
import psf_marginalise as pm
import scipy.ndimage as nd
class CONTRATIO():
    def choose_psfs(self,tgt_cubes,cal_cubes):
        objName = pyfits.getheader(tgt_cubes[0])['OBJECT']
        #Remove Spaces From Object Name
        objNoSpaces = objName.split(' ')
        objName = ''.join(objNoSpaces)
        outfile = 'good_ims_'+objName+'.fits'			
        REJECT_COLUMN = 20
        
        tgt_ims = []
        cal_ims = []
        
        #Reject target data
        bintab = []
        header = pyfits.getheader(tgt_cubes[0])
        for fname in tgt_cubes:
            dd = pyfits.getdata(fname)
            newtab = pyfits.getdata(fname,1)
            for i in range(dd.shape[0]):
                plt.clf()
                plt.imshow(dd[i,:,:], interpolation='nearest')
                xy = plt.ginput(1)
                if (xy[0][0] > REJECT_COLUMN):
                    tgt_ims.append(dd[i,:,:])
                    if len(bintab):
                        bintab = np.append(bintab, newtab[i:i+1])
                    else:
                        bintab =  newtab[i:i+1]
                else:
                    print("Frame Rejected!")
                    
        #Reject calibrator data            
        for fname in cal_cubes:
            dd = pyfits.getdata(fname)
            for i in range(dd.shape[0]):
                plt.clf()
                plt.imshow(dd[i,:,:], interpolation='nearest')
                xy = plt.ginput(1)
                if (xy[0][0] > REJECT_COLUMN):
                    cal_ims.append(dd[i,:,:])
                else:
                    print("Frame Rejected!")
        
        tgt_ims = np.array(tgt_ims)
        cal_ims = np.array(cal_ims)
        
        #Now save the file!
        hdu1 = pyfits.PrimaryHDU(tgt_ims, header)
        hdu2 = pyfits.ImageHDU(cal_ims)
        hdu3 = pyfits.BinTableHDU(bintab)
        hdulist = pyfits.HDUList([hdu1,hdu2,hdu3])
        hdulist.writeto(outfile, clobber=True)
        return outfile
    
    def best_psf_subtract(self,filename,plotDir):
        #Create our target model and object model.
        psfs = pm.Psfs(cubefile=filename)
        object_model = pm.PtsrcObject()
        tgt = pm.Target(psfs, object_model, cubefile=filename)

        best_ixs, best_fit_ims = tgt.find_best_psfs([])
        objName = pyfits.getheader(filename)['OBJECT']
        #Remove Spaces From Object Name
        objNoSpaces = objName.split(' ')
        objName = ''.join(objNoSpaces)
        #Subtracted image - oriented so that N is up and E is left.
        subims = tgt.ims - best_fit_ims
        subim_sum = np.sum(subims, axis=0) 
        plt.figure(1)
        plt.clf()
        plt.imshow(nd.rotate(subim_sum, -np.mean(tgt.pas), reshape=False)[::-1,:],'cubehelix',extent=[640,-640,-640,640])
        plt.colorbar()
        plt.xlabel('RA Offset (mas)')
        plt.ylabel('Dec Offset (mas)')
        plt.savefig(plotDir+'/subimSum'+objName+'.png')
        plt.clf()
        #We want to compute: \Sum (subims[i] * shift(best_fit_ims[i])) for 
        #every possible shift value in 2D. This is just a cross-correlation!
        #We can use the power of numpy by putting this in a single line.
        #We are computing best_fit_ims [cross-correlation *] subims
        crats = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(best_fit_ims)) * np.fft.rfft2(subims) ), axes=[1,2])
        #Now rotate and normalise
        for i in range(len(crats)):
            crats[i] = nd.rotate(crats[i], -tgt.pas[i], reshape=False)[::-1,:]/np.sum(best_fit_ims[i]**2)
        plt.imshow(np.average(crats,axis=0),'cubehelix',vmin=0,extent=[640,-640,-640,640])
        plt.colorbar()
        plt.xlabel('RA Offset (mas)')
        plt.ylabel('Dec Offset (mas)')
        plt.title(objName+' Average Contrast Ratio')
        plt.savefig(plotDir+'/cratMean'+objName+'.png')
        plt.clf()
        #We can also compute a background-limited standard deviation, and the crat uncertainty due to
        #this. 
        stds = np.empty(len(tgt.ims))
        for i, im in enumerate(tgt.ims):
            stds[i] = np.std(im[tgt.corner_pix])
        crat_std = tgt.ims
        crat_errors = stds/np.sqrt(np.sum(best_fit_ims**2,axis=(1,2)))
        crat_stds = np.std(crats,axis=0)/len(crat_errors)
        plt.imshow(crat_stds,'cubehelix',vmin=0,extent=[640,-640,-640,640])
        plt.colorbar()
        plt.title(objName+' Contrast Ratio Standard Deviation')
        plt.xlabel('RA Offset (mas)')
        plt.ylabel('Dec Offset (mas)')
        plt.savefig(plotDir+'/cratStd'+objName+'.png')
        plt.clf()
        
