"""
Notes: all27.pkl didn't seem to work, so was put in the trash.
all28.pkl comes from TauL15_4, which was from 151128, before transit (1 data set)
There is also TauL15_5 (after transit) and also 

Questions (for Alex?):
1) How important is the *time* axis in determining PSFs? Is it good enough to become a
prior?
2) Why did psf_marginalise choose a blurry prior for LkCa15? Was this because there 
was resolved structure? Or was 

#--------
rot_resid_sum = np.sum(rot_resid,axis=0)/np.max(np.sum(tgt.ims, axis=0))
plt.clf()
#plt.imshow(rot_resid_sum, extent=[-.64,.64,-.64,.64], interpolation='nearest', cmap=cm.cubehelix)
plt.imshow(0.5*(rot_resid_sum + rot_resid_sum1), extent=[-.64,.64,-.64,.64], vmin=-0.002, interpolation='nearest', cmap=cm.cubehelix)
plt.axis([-0.4,0.4,-0.4,0.4])
plt.xlabel('Delta RA (arcsec)')
plt.ylabel('Deta Dec (arcsec)')
plt.colorbar()

tgt.marginalise()

psfs.display_lle_space()
psfs.mcmc_explore()


pickle.dump((mod_ims, rot_resid, best_x, sampler.chain, sampler.lnprobability, tgt.ims), open('all27.pkl', 'w'))
pickle.dump((mod_ims, rot_resid, best_x, sampler.chain, sampler.lnprobability, tgt.ims), open('all28.pkl', 'w'))

best_fit_ims, rot_resid, best_x, chain, lnprobability, ims = pickle.load(open('all28.pkl', 'r'))
"""
from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import psf_marginalise as pm
import scipy.ndimage as nd
import astropy.io.fits as pyfits
import scipy.ndimage as nd

cubefile='/Users/mireland/tel/nirc2/redux/TauL15_2/good_ims_LkCa15.fits' #30 target images.
cubefile='/Users/mireland/tel/nirc2/redux/TauL15_4/good_ims_LkCa15.fits' #5 target images (in use)

#Create our target model and object model.
psfs = pm.Psfs(cubefile=cubefile)
object_model = pm.PtsrcObject()
tgt = pm.Target(psfs, object_model, cubefile=cubefile)

#Embed the PSFs
psfs.lle(ndim=3) #For Tau15_4, ndim=3 is roughly enough.
starting_lnprob = tgt.lnprob(np.zeros(psfs.ndim*tgt.n_ims))

#Find the best matching PSFs.
best_x, sampler = tgt.marginalise(nchain=400, use_threads=False)

#Find the resulting PSFs
best_fit_ims = tgt.lnprob(best_x, return_mod_ims=True)
subims = tgt.ims - mod_ims

#Find the rotated residuals
rot_resid = np.empty_like(subims)
for rr, si, pa in zip(rot_resid, subims, tgt.pas):
    rr = nd.rotate(si, -pa, reshape=False)[::-1,:]
rot_resid_sum = np.sum(rot_resid,axis=0)/np.max(np.sum(tgt.ims, axis=0))

crats = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(best_fit_ims)) * np.fft.rfft2(subims) ), axes=[1,2])

#Now rotate and normalise
for i in range(len(crats)):
    crats[i] = nd.rotate(crats[i], -tgt.pas[i], reshape=False)[::-1,:]/np.sum(best_fit_ims[i]**2)

