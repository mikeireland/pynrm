"""This module provides tools for finding the best fit object model by marginalising
over the space of possible PSFs.

For LkCa15 testing, see ./marginalise_image.py.

"""

from __future__ import print_function, division
import mdp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage as nd
import astropy.io.fits as pyfits
import scipy.optimize as op
import opticstools as ot
import emcee
import multiprocessing
from scipy.ndimage import zoom


#!!! Removed pi from the following two functions.
def optimize_tilt_derivative(p, mod_ft, im_ft, uv):
    """Helper function for optimize_tilt."""
    
    #The Jacobian, i.e. the derivatives of the model with respect to tilt
    #complex_jac = [uv[0]*mod_ft*np.exp(1j*(p[0]*uv[0] + p[1]*uv[1])),\
    #               uv[1]*mod_ft*np.exp(1j*(p[0]*uv[1] + p[1]*uv[1]))]
    #retarray = np.array([complex_jac[0].imag,-complex_jac[0].real,complex_jac[1].imag,-complex_jac[1].real])
    complex_jac = [1j*uv[0]*mod_ft*np.exp(1j*(p[0]*uv[0] + p[1]*uv[1])),\
                   1j*uv[1]*mod_ft*np.exp(1j*(p[0]*uv[1] + p[1]*uv[1]))]
    retarray = np.array([complex_jac[0].real,complex_jac[0].imag,complex_jac[1].real,complex_jac[1].imag])
    return retarray.reshape( (2,2*np.prod(uv[0].shape)) )

def make_uv_grid(sz, diam, wave, pscale):
    """Helper function to make a 2D grid in the UV plane. 
    
    Parameters
    ----------
    sz: int
        (square) array size in pixels
        
    diam: float
        Telescope maximum diameter in m
    
    wave: float
        Wavelength in m of the shortest wavelength in the filter.
        
    pscale: float
        Pixel scale in arcsec/pix.
    
    Returns
    -------
    sampled_uv: Numpy array
        The pixel coordinates in the 2D Fourier transform of the image.
        
    uv:
        The Fourier coordinates of these sample points.
    """
    uv = np.meshgrid(2*np.pi*np.arange(sz//2 + 1)/float(sz),
        2*np.pi*(((np.arange(sz) + sz//2) % sz) - sz//2)/float(sz))
    
    #A variable that is 2*pi for 1 cycle per pixel.
    rr = np.sqrt(uv[0]**2 + uv[1]**2)
    sampled_uv = np.where(rr < 2*np.pi*diam/wave*np.radians(pscale/3600.))
    
    return sampled_uv, np.array([uv[0][sampled_uv],uv[1][sampled_uv]])

def optimize_tilt_function(p, mod_ft, im_ft, uv, return_model=False):
    """Helper function for optimize_tilt. This is used  as a function to
    input into leastsq 
    
    Parameters
    ----------
    
    p: numpy array (2): [xtilt,ytilt]
        The wavefront tilt image-plane pixels.
    
    mod_ft: array
        Model Fourier transform that we want to fit to the image
    
    im_ft: array
        Image Fourier transform
        
    uv: array
        Sampling points for mod_ft and im_ft
        
    Returns
    -------
    resid: numpy array
        Array of residuals to the fit.
    """
    new_model = mod_ft*np.exp(1j*(p[0]*uv[0] + p[1]*uv[1]))
    if return_model:
        return new_model
    retval_complex = im_ft - new_model
    return np.append(retval_complex.real.flatten(), retval_complex.imag.flatten())

def optimize_tilt(mod_ft, im_ft, uv, scale_flux=False, check_fit=False):
    """Given a PSF and an image Fourier transform sampled at points u and
    v, tilt and scale the model so that it matches the image.
    
    We do this fitting in the Fourier domain rather than the image-plane, because
    sub-pixel tilts can be more precise in the Fourier domain. A shift is a convolution
    with a delta-function centered at the pixel (xshift, yshift). So we can shift
    by multiplying the Fourier transform of the image by the Fourier transform of this
    shift operator, which we can call the shift kernel.
    
    Parameters
    ----------
    
    u,v: float array
        Cycles per pixel times 2 np.pi.
    
    Returns
    -------
    The shifted model Fourier transform."""
    if scale_flux:
        im_ft_out = im_ft/im_ft[0]
    else:
        im_ft_out = im_ft
    #retvals = op.leastsq(optimize_tilt_function, [0,0], args=(mod_ft,im_ft_out,uv), \
    #    Dfun=optimize_tilt_derivative, col_deriv=True, xtol=1e-3, ftol=1e-6)
    retvals = op.leastsq(optimize_tilt_function, [0,0], args=(mod_ft,im_ft_out,uv), \
        xtol=1e-4, ftol=1e-7)
    if check_fit:
        pdb.set_trace()
    if retvals[1] <= 0:
        print("Error in finding tilts!")
        raise UserWarning
    else:
        return retvals[0]
    
def prepare_im(im, ref_ft, uv, sampled_uv, corner_pix, center_ft = True, scale = 1.0):
    """A helper function to shift an image, optimize its tilt and subtract the
    background.
    
    Parameters
    ----------
    im: numpy array
        Image that we want to prepare.
    
    ref_ft: numpy array
        Fourier transform of the reference image, that defines a "centered" position.
        
    uv, sampled_uv: numpy array
        The (u,v) coordinates and pixel values in the UV plane.
        
    corner_pix: numpy array
        The image corner pixels, which defined the background.
        
    center_ft: (optional) bool
        Do we return a Fourier transform with sub-pixel sampling?
        
    Returns
    -------
    (a_psf, a_psf_ft):
        Roughly centered image, and precisely centered imaged Fourier transform (unless
        center_ft is False)
    """
    sz = im.shape[0]
    maxpix = np.unravel_index(np.argmax(im), im.shape)
    a_psf = np.roll(np.roll(im, sz//2 - maxpix[0], axis=0), sz//2 - maxpix[1],axis=1)
    a_psf -= np.median(a_psf[corner_pix])
    if scale == 1.0:
        mod_ft = np.fft.rfft2(a_psf)[sampled_uv]
    else:
        if scale > 1:
            scaled_psf = zoom(a_psf, scale)[a_psf.shape[0]//2-sz//2:a_psf.shape[0]//2+sz//2,\
                                            a_psf.shape[1]//2-sz//2:a_psf.shape[1]//2+sz//2]
        else:
            scaled_psf = np.zeros_like(a_psf)
            scaled_psf[sz//2-a_psf.shape[0]//2:sz//2+a_psf.shape[0]//2,\
                       sz//2-a_psf.shape[1]//2:sz//2+a_psf.shape[1]//2] = zoom(a_psf, scale)
        mod_ft = np.fft.rfft2(scaled_psf)[sampled_uv]
    if center_ft:
        tilt = optimize_tilt(mod_ft/mod_ft[0], ref_ft/ref_ft[0], uv)
        a_psf_ft = optimize_tilt_function(tilt, mod_ft, ref_ft, uv, return_model=True)
    else:
        a_psf_ft = mod_ft
    return a_psf, a_psf_ft


class Psfs(object):
    """A set of reference PSFs, which creates the space to marginalise over"""
    def __init__(self, psfs=[], psf_files=[], wave=3.5e-6, diam=10.0,pscale=0.01, \
            cubefile=None, cube_extn=1, hyperparams=[], subtract_outer_median=True, scale=1.0):
        """Initialise the reference PSFs. This includes reading them in, cleaning
        and shifting to the origin. Cleaning here includes Fourier filtering, and 
        shifting to the origin is done in a least squares sense, i.e. a sub-pixel shift
        that refers PSFs to a master (i.e. mean) PSF.
        
        Given that the PSFs have limited support in the Fourier domain, we will store them
        as complex Fourier component vectors on this support. Then the process of fitting
        to a linear combination of PSFs is just making a linear combination on this support.
        
        Note that the (uv) co-ordinates are stored in a way most convenient for the
        tilt_function, going from 0 to 2pi over the full Fourier domain.
        
        Parameters
        ----------
        wave: float
            Wavelength in m
            
        diam: float
            Telescope diameter in m
            
        pscale: float
            Pixel scale in arcsec. 
        """
        self.ndim = 0 #0 until we embed the PSFs in a lower dimensional space!
        self.use_this_psf = 0 #i.e. just use the first PSF until we're told otherwise.
        if cubefile:
            psfs = pyfits.getdata(cubefile,cube_extn)
        else:
            print("Not implemented quite yet...")
            raise UserWarning
            
        sz = psfs.shape[1]
        self.sz = sz
        self.npsfs = len(psfs)
        
        outer_pix = (1-ot.circle(sz,2*sz/3)) > 0
        for i in range(len(psfs)):
            if subtract_outer_median:
                psfs[i] -= np.median(psfs[i][outer_pix])
            psfs[i] /= np.sum(psfs[i])
        
        #While sampled_uv is an integer array of uv pixel coordinates, uv is an
        #array of Fourier frequency in inverse pixel units
        self.sampled_uv, self.uv = make_uv_grid(sz, diam, wave, pscale)
        
        psf_mn = np.sum(psfs,0)/psfs.shape[0]
        psf_mn_ft = np.fft.rfft2(psf_mn)[sampled_uv]
        psf_fts = []
        #NB This should probably be run twice - once to get a better psf_mn_ft.
        corner_pix = np.where(1 - ot.circle(self.sz, self.sz))
        for i in range(len(psfs)):
            centered_psf, a_psf_ft = prepare_im(psfs[i], psf_mn_ft, self.uv, self.sampled_uv, corner_pix, scale=scale)
            psf_fts.append(a_psf_ft)
                
        self.psf_fts = np.array(psf_fts)
        self.psf_fts_vect = np.array([np.append(psf_ft.real, psf_ft.imag) for psf_ft in psf_fts])
        self.psf_mn_ft = psf_mn_ft
    
    def psf_im(self,ix):
        """Helper function to return a point-spread function from a library MTF
        
        Parameters
        ----------
        ix: int
            Index of the point spread function to return
        """
        if (ix >= len(self.psf_fts)):
            print("ERROR: index out of range")
            raise UserWarning
        else:
            return self.im_from_ft(self.psf_fts[ix])
    
    def im_from_ft(self,im_ft_sampled):
        """Return a full image based on the subsampled Fourier plane."""
        im_ft = np.zeros( (self.sz,self.sz//2+1), dtype=np.complex)
        im_ft[self.sampled_uv] = im_ft_sampled
        return np.fft.irfft2(im_ft)
        
    def lle(self,ndim=2,nk=None, length_nsigma=2.0):
        """Embed the PSFs onto an ndim dimensional space
        
        Parameters
        ----------
        ndim: int
            Number of LLE dimensions
            
        nk: int
            Number of nearest neighbors to check.
            
        """
        self.ndim=ndim
        #The following 
        if not nk:
            nk = int(1.5*np.sqrt(self.psf_fts_vect.shape[0]))
        self.nk = nk
        self.lle_proj = mdp.nodes.LLENode(nk,output_dim=ndim,verbose=True)(self.psf_fts_vect)
        lengths = np.empty(ndim)
        for d in range(ndim):
            one_sigmas = np.percentile(self.lle_proj[:,d],[16,84])
            lengths[d] = length_nsigma*(one_sigmas[1]-one_sigmas[0])
        print("Axis lengths: " + str(lengths) )
        self.h_density = (np.prod(lengths)/self.lle_proj.shape[0])**(1.0/ndim)
        self.tri = Delaunay(self.lle_proj)
        self.point_lengths = np.sum(self.tri.points**2,1)
    
    def display_lle_space(self, nsamp=100, return_density=False):
        """Display the space of PSFs as a density in LLE space plus the points
        from which it was constructed.
        
        Parameters
        ----------
        nsamp: int (optional)
            number of samples in each dimension in the 2D image
            
        Returns
        -------
        density: numpy array
            The probability density that is plotted, if return_dentiy=True.
        extent: numpy array
            The extent of the plot, if return_dentiy=True.
        """
        if self.ndim != 2:
            print("A 2D display only works for ndim=2")
        sz = self.sz #Short-hand
        uv = np.meshgrid(2*np.pi*np.arange(sz//2 + 1)/float(sz),
                2*np.pi*(((np.arange(sz) + sz//2) % sz) - sz//2)/float(sz))
        x=np.linspace(-0.5,0.5,nsamp)
        density = np.empty( (nsamp, nsamp) )
        xy = np.meshgrid(x,x)
        for i in range(nsamp): 
            for j in range(nsamp):
                density[i,j] = self.lle_density([xy[0][i,j],xy[1][i,j]])
        extent = [-0.5,0.5,-0.5,0.5]
        density = density[::-1,:]
        if return_density:
            return density, extent
        else:
            plt.clf()
            plt.imshow(density, extent=extent, cmap=cm.gray)
            plt.plot(self.tri.points[:,0], self.tri.points[:,1], '.')
            plt.axis(extent)
        
    def augment_zernike(self,naugment=3, amps=np.ones(7)*0.1):
        """Augment the library of reference PSFs by adding zernike's to them
        (neglecting tilt)"""
        
    def find_lle_psf(self,x, return_image=True):
        """Return the unique interpolated PSF from ndim+1 library PSFs, by
        interpolating within the smallest enclosing simplex where all 
        angles. If outside the convex hull, find the nearest edge/faces and
        the simplex that includes one of these and (if possible) extends the furthest 
        in the opposite direction. """
        
        #If we havnen't embedded out PSFs into some abstract space, this is simple!
        if self.ndim==0:
            if return_image:
                return self.psf_im(self.use_this_psf)
            else:
                return self.psf_fts[self.use_this_psf]
        
        #Otherwise, we have to find nearby LLE co-ordinates (an enclosing simplex) and
        #interpolate between PSFs.
        x = np.array(x)
        enclosing_simplex = self.tri.find_simplex(x)
        if enclosing_simplex<0:
            #Distances between x and the points
            dists = self.point_lengths - 2*np.dot(self.tri.points,x) + np.sum(x**2)
            nearest = np.argmin(dists)
            possible_simplices = np.where(np.sum(self.tri.simplices==nearest,axis=1))[0]
            #Given a simplex and reference vertex r we can find c such that.
            # T . c = x-v2
            #... then x = v2 + c0*(v0-v2) + c1*(v1-v2)
            #           = c0*v0 + c1*v1 + (1-c0-c1)*v2
            min_coeffs=[]
            for simplex in possible_simplices:
                coeffs = np.dot(self.tri.transform[simplex][:self.ndim,:self.ndim], \
                                x - self.tri.transform[simplex][-1])
                coeffs = np.append(coeffs, 1-np.sum(coeffs))
                min_coeffs.append(np.min(coeffs))
            #The best simplex is the one with the least negative coefficient.
            simplex = possible_simplices[np.argmax(min_coeffs)]
        else:
            simplex = enclosing_simplex
        #Now that we know which simplex to use, get the coefficients and find the PSF
        coeffs = np.dot(self.tri.transform[simplex][:self.ndim,:self.ndim], \
                        x - self.tri.transform[simplex][-1])
        coeffs = np.append(coeffs, 1-np.sum(coeffs))
        interp_psf_ft = np.dot(coeffs,self.psf_fts[self.tri.simplices[simplex]])
        if return_image:
            return self.im_from_ft(interp_psf_ft)
        else:
            return interp_psf_ft
    
    def trunc_gauss(self, q):
        """ Compute the truncated Gaussian probability density"""
        wl = np.where( (q>0) * (q<0.5) )[0]
        wh = np.where( (q>0.5) * (q<1) )[0]
        the_sum = (np.sum(1-6*q[wl]**2+6*q[wl]**3) + \
                   np.sum(2*(1-q[wh])**3))
        if self.ndim==2:
            return 40/np.pi/7*the_sum
        else:
            return 8/np.pi*the_sum
    
    def lle_density(self,x):
        """Return the local probability density of a given LLE coordinate
        
        Normalise to a total integral of 1.0 over all LLE parameter space."""
        #Brute force here... KDTree will help as we only need to consider
        #12-ish nearest neighbors in 2D and 33-ish nearest neighbors in 3D.
        if self.ndim==0:
            print("Density zero - must compute the LLE first!")
            return 0
        dists = np.array([np.sqrt(np.sum((x - y)**2)) for y in self.lle_proj])
        ww = np.where(dists < 2*self.h_density)[0]
        if len(ww)==0:
            return 0.0
        else:
            return self.trunc_gauss(dists[ww]/2.0/self.h_density)/len(self.lle_proj)/(2*self.h_density)**2 
    
    def mcmc_explore(self,niter=30,stepsize=0.5):
        """Explore the space of PSFs."""
        current_pos = self.lle_proj[0]
        current_density = self.lle_density(current_pos)
        print("Computing overall background density for plotting...")
        density, extent = self.display_lle_space(return_density=True)
        for i in range(niter):
            plt.clf()
            plt.subplot(121)
            jump = np.random.normal(size=self.ndim)*stepsize*self.h_density
            trial = current_pos + jump
            new_density = self.lle_density(trial)
            if new_density/current_density > np.random.random():
                current_density = new_density
                current_pos = trial
            psf = self.find_lle_psf(current_pos, return_image=True)
            plt.imshow(np.arcsinh(psf/np.max(psf)/0.01),interpolation='nearest', cmap=cm.cubehelix)
            plt.title("lnprob: {0:5.1f}".format(current_density))
            plt.subplot(122)
            plt.imshow(density,extent=extent,cmap=cm.gray)
            plt.plot(self.tri.points[:,0],self.tri.points[:,1],'b.')
            plt.title("Pos: {0:5.2f} {1:5.2f}".format(current_pos[0], current_pos[1]))
            plt.plot(current_pos[0], current_pos[1], 'ro') 
            plt.axis(extent)
            plt.draw()
            #!!! Problem: Current matplotlib does not draw here. !!!
            dummy = plt.ginput(1)
        return None 
        
    def hyperparam_prob(self,x, hyperparams=None):
        """Return the hyperparameter probability for a given set of LLE coordinates 
        and hyperparameters. Uses the same density kernel as lle_density."""  
        return self.lle_density(x)
        
class PtsrcObject(object):
    def __init__(self,initp = []):
        """A model for the object on sky, consisting of a single point source.
        
        Other objects can inherit this. Generally, there will be some fixed parameters
        and some variable parameters. The model parameters are *not* imaging parameters, 
        i.e. do not include x, y, flux variables. 
        """
        self.p = initp
        self.np = len(initp)

    def model_uv(self, p_in, uv):
        """Return a model of the Fourier transform of the object given a set of
        points in the uv plane
        
        Parameters
        ----------
        p_in: array-like
            model parameters. Can be None if if the model has no parameters!
            
        uv: array-like
            Coordinates in the uv plane """
        return np.ones(uv.shape[1])

class ModelObject(object):
    def __init__(self,initp = [], infits=''):
        """A model for the object on sky, consisting of an input fits files.
        
        Parameters
        ----------
        initp: array-like
            Unused: The input parameters
            
        infits: string 
            A filename for an input model fits files.
        """
        if len(infits)==0:
            raise ValueError("Must set keyword infits to a filename!")
        self.p = initp
        self.np = len(initp)
        
        #Read in the fits file.
        im = pyfits.getdata(infits)
        
        if im.shape[0] != im.shape[1]:
            raise ValueError("Model Image must be square")
        
        #Take the Fourier transform and make sure we have coordinate arrays ready
        #for interpolation
        self.mod_ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im)))
        uv_coord = 2*np.pi*(np.arange(im.shape[0]) - im.shape[0]//2)/float(im.shape[0])
        #x = np.arange(im.shape[0]) - im.shape[0]//2 #XXX
        #y = np.arange(im.shape[1]) - im.shape[1]//2
        #self.mod_ft_func = RectBivariateSpline(uv_coord, uv_coord, self.mod_ft, kx=1, ky=1)
        self.mod_ft_realfunc = RectBivariateSpline(uv_coord, uv_coord, self.mod_ft.real, kx=1, ky=1)
        self.mod_ft_imagfunc = RectBivariateSpline(uv_coord, uv_coord, self.mod_ft.imag, kx=1, ky=1)
        
        #self.yx = np.meshgrid(x, y)

    def model_uv(self, p_in, uv):
        """Return a model of the Fourier transform of the object given a set of
        points in the uv plane
        
        Parameters
        ----------
        p_in: array-like
            model parameters. Can be None if if the model has no parameters!
            
        uv: array-like
            Coordinates in the uv plane """
        ret_array = self.mod_ft_realfunc(uv[0], uv[1], grid=False).astype(np.complex)
        ret_array += 1j*self.mod_ft_imagfunc(uv[0], uv[1], grid=False)
        return ret_array

class ResidObject(object):
    def __init__(self,initp = [], resid_in=None, psf_in=None):
        """A model for the object on sky, consisting of a point source and a 
        map that has been convolved with the PSF map. 
        
        The idea is that, iteratively, the fit residuals can be added to to the input 
        residuals to last model residuals, and the final problem is a standard 
        deconvolution problem with a known PSF.
        
        Parameters
        ----------
        initp: array-like
            A single parameter, the relative flux of the resolved part of the image.
            
        resid_in: numpy array
            Residuals from the previous iteration. Same size and format as the input 
            image, but with N down and E left when displayed with imshow.
            
        psf_in: numpy array
            The PSF that should be used for the residuals, weighted in the same way.
        """
        self.p = initp
        self.np = len(initp)
        
        #Normalise the input image and PSF
        im = resid_in.copy()
        im /= np.sum(im)
        mean_psf = psf_in.copy()
        mean_psf /= np.sum(mean_psf)
        
        #Error checking
        if im.shape[0] != im.shape[1]:
            raise ValueError("Model Image must be square")
        if im.shape != mean_psf.shape:
            raise ValueError("PSF must have the same shape as input residuals.")
        
        #Take the Fourier transform and make sure we have coordinate arrays ready
        #for interpolation. The line below could have a divide by zero - but not where
        #the Fourier transform has non-zero support.
        self.mod_ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im))) / \
            np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mean_psf)))
        
        uv_coord = 2*np.pi*(np.arange(im.shape[0]) - im.shape[0]//2)/float(im.shape[0])
        self.mod_ft_realfunc = RectBivariateSpline(uv_coord, uv_coord, self.mod_ft.real, kx=1, ky=1)
        self.mod_ft_imagfunc = RectBivariateSpline(uv_coord, uv_coord, self.mod_ft.imag, kx=1, ky=1)


    def model_uv(self, p_in, uv):
        """Return a model of the Fourier transform of the object given a set of
        points in the uv plane
        
        Parameters
        ----------
        p_in: array-like
            model parameters. Can be None if if the model has no parameters!
            
        uv: array-like
            Coordinates in the uv plane """
        ret_array = self.mod_ft_realfunc(uv[0], uv[1], grid=False).astype(np.complex)
        ret_array += 1j*self.mod_ft_imagfunc(uv[0], uv[1], grid=False)
        return p_in[0]*ret_array + (1-p_in[0])


class BinaryObject(PtsrcObject):
    def __init__(self, initp=[]):
        """A Model with two point-sources
        
        Parameters
        ----------
        init_p: numpy array(3)
            North Separation in pix, East separation in pix, contrast secondary/primary
        """
        super(BinaryObject, self).__init__(initp)
    
    def model_uv(self, p_in, uv):
        """Return a model of the Fourier transform of a binary given a set of points
        in the uv plane
        
        Parameters
        ----------
        p_in: array-like
            North Separation in pix, East Separation in pix, Contrast secondary/primary
            
        uv: array-like
            Coordinates in the uv plane """
        #The Fourier transform of 2 delta functions is the sum of the Fourier transform
        #of each delta function. Lets make the primary star be at co-ordinate (0,0)
        ft_object = np.ones(uv.shape[1]) + p_in[2]*np.exp(1j*(p_in[0]*uv[0] + p_in[1]*uv[1]))
        ft_object /= 1+p_in[2]
        return ft_object
        
class Target(object):
    """A set of target images"""
    def __init__(self,psfs,object_model,ims=[], im_files=[],hyperparams=[], 
                 cubefile=None, cube_extn=0, pas_extn=2, pas=[], gain=4.0):
        """Initialise the reference PSFs. This includes reading them in, cleaning,
        shifting to the origin, normalising, creating variance arrays, and chopping
        out the uv component.
        
        Parameters
        ----------
        psfs: Psfs instance
            The PSF library to go with the target
        
        object: PtsrcObject instance
            The object model
        
        ims: float numpy array
            A set of target images
            
        Notes
        -----
        The input gain should come from pynrm!!! (or better: be set to 1 by scaling)
        """
        self.psfs= psfs
        self.object = object_model
        if cubefile:
            ims = pyfits.getdata(cubefile,cube_extn)
            self.pas = pyfits.getdata(cubefile,pas_extn)['pa']
        else:
            print("Not implemented quite yet...")
            raise UserWarning
        self.n_ims = len(ims)
        self.sz = ims.shape[1]
        if self.sz != self.psfs.sz:
            print("Error: PSFs and target images must be the same size")
            raise UserWarning
        self.tgt_uv = np.empty( (self.n_ims, self.psfs.uv.shape[0], self.psfs.uv.shape[1]) )
        #!!! NB Sign of rotation not checked below !!!
        for i in range(self.n_ims):
            self.tgt_uv[i,0] = self.psfs.uv[0]*np.cos(np.radians(self.pas[i])) + \
                               self.psfs.uv[1]*np.sin(np.radians(self.pas[i]))
            self.tgt_uv[i,1] = self.psfs.uv[1]*np.cos(np.radians(self.pas[i])) - \
                               self.psfs.uv[0]*np.sin(np.radians(self.pas[i]))
        self.corner_pix = np.where(1 - ot.circle(self.sz, self.sz))
        self.read_var = 0.0
        #Assumption: all target images have the same readout and/or background
        #variance !!!
        #!!! Bad pixels have to be added in separately here !!!
        #!!! There should be read noise and variance from pynrm !!!
        im_fts = []
        for i in range(len(ims)):
            centered_im, a_psf_ft = prepare_im(ims[i], self.psfs.psf_mn_ft, \
                self.psfs.uv, self.psfs.sampled_uv, self.corner_pix, center_ft = False)
            ims[i] = centered_im
            self.read_var += np.var(ims[i][self.corner_pix])
            im_fts.append(a_psf_ft)
        
        self.ims = ims      
        self.im_fts = np.array(im_fts)      
        self.read_var /= len(ims)
        self.ivar = 1.0/(self.read_var + ims/gain)
    
    def lnprob(self, x, tgt_use=[], return_mod_ims=False, return_mod_psfs=False):
        """Compute the log probability of a model.
        
        Parameters
        ----------
        
        x: numpy array
            The input LLE coordinates, followed by the model parameters, in the order:
            [lle[0],lle[1],,,,lle[len(tgt_use)-1],p_in[0],,,,p_in[n_params-1], where
            each of lle[0] etc is a list of length psfs.ndim.
            
        p_use: numpy int array (optional)
            The list of model parameters to use, e.g. [0,1,3], in case we want to fix
            some of them. 
            
        tgt_use: numpy int array (optional)
            The list of target PSFs to fit to, in case we don't want to fit to all of them.
            
        return_mod_ims: bool (optional)
            Optionally return the model images rather than the log probability.
        """
        x = np.array(x)
        #If tgt_use is not given, use all targets.
        if len(tgt_use)==0:
            tgt_use = np.arange(self.n_ims)
        
        if self.psfs.ndim > 0:
            x_lle = x[:self.psfs.ndim * len(tgt_use)].reshape( (len(tgt_use), self.psfs.ndim) )
        else:
            x_lle=[]
        x_p = x[self.psfs.ndim * self.psfs.npsfs:]
        prior_prob=1.0
        chi2 = 0.0
        
        mod_ims = []
        mod_psfs = []
        
        #Loop through the image and add to the chi-squared
        for i in range(len(tgt_use)):
            prior_prob *= self.psfs.hyperparam_prob(x_lle[i])
            
            #What is our object model?
            obj_ft = self.object.model_uv(x_p, self.tgt_uv[tgt_use[i]])
            
            #Convolve the object with the PSF model to form an image model.
            psf_ft = self.psfs.find_lle_psf(x_lle[i], return_image=False)
            if return_mod_psfs:
                mod_psfs.append(self.psfs.im_from_ft(psf_ft))
            mod_ft = obj_ft * psf_ft
            
            #Find the tilt that best matches the image model, and form an image model,
            #and scale the image to match the total flux.
            scale_factor = self.im_fts[tgt_use[i]][0].real/mod_ft[0].real
            tilt = optimize_tilt(mod_ft, self.im_fts[tgt_use[i]]/scale_factor, self.psfs.uv)#, check_fit=True)
            mod_ft = optimize_tilt_function(tilt, mod_ft*scale_factor, self.im_fts[i], self.psfs.uv, return_model=True)
            mod_im = self.psfs.im_from_ft(mod_ft)
            
            #Do we want to return the image?
            if return_mod_ims:
                mod_ims.append(mod_im)
            
            #Compute chi-squared
            chi2 += np.sum((mod_im - self.ims[tgt_use[i]])**2*self.ivar[tgt_use[i]])
            
        #Returning multiple things is a little messy, but it saves code duplication, or
        #un-necessary computation.
        if return_mod_ims and return_mod_psfs:
            return np.array(mod_ims), np.array(mod_psfs)
        elif return_mod_ims:
            return np.array(mod_ims)
        elif return_mod_psfs:
            return np.array(mod_psfs)
        if prior_prob==0:
            return -np.inf
        else:
            return np.log(prior_prob) - chi2/2.0
    
    def lle_simplex_interp(self,x):
        """!!!What is this ???"""
        return None
    
    def marginalise(self,init_par=[],walker_sdev=[],nchain=1000, use_threads=True, start_one_at_a_time=True):
        """Use the affine invariant Monte-Carlo Markov chain technique to marginalise
        over all PSFs. We cheat a little by not marginalising over the model parameters 
        simultaneously - the parameters are expected to have Gaussian errors
        that come out of a least squares process that fits to PSFs from a point source fit 
        (at least this is what I think I meant).
        
        WARNING: This doesn't actually marginalise over the model parameters yet, it only
        marginalises over the LLE parameters for the point source model.
        """
        if len(init_par) != len(walker_sdev):
            raise UserWarning("Require same number of parameters (init_par) as walker standard deviations (walker_sdev)!")
        threads = multiprocessing.cpu_count()
        
        if start_one_at_a_time:
            #Try optimising one image at at time... (no parameters)
            ndim = self.psfs.ndim 
            #Make an even number of walkers.
            nwalkers = (3*ndim//2)*2
        
            #Initialise the chain to random psfs.
            p0 = np.empty( (nwalkers, ndim) )
            init_lle_par = []
            for i in range(nwalkers):
                p0[i,:] = self.psfs.tri.points[int(np.random.random()*self.psfs.npsfs)]
            for j in range(self.n_ims):
                kwargs = {"tgt_use":[j]}
                if use_threads:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=threads, kwargs=kwargs)
                else:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, kwargs=kwargs)
                sampler.run_mcmc(p0,nchain)
                init_lle_par.append(sampler.flatchain[np.argmax(sampler.flatlnprobability)])
                print("Done initial model for chain {0:d}".format(j))
            init_lle_par = np.array(init_lle_par).flatten()
        
        #Minimum number of walkers
        ndim = self.psfs.ndim * self.n_ims  + len(init_par)
        nwalkers = 2*ndim
        if use_threads:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=threads, kwargs=kwargs)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, kwargs=kwargs)
            
        #Initialise the chain to random psfs.
        p0 = np.empty( (nwalkers, ndim) )
        if start_one_at_a_time:
            for i in range(nwalkers):
                p0[i, :self.psfs.ndim * self.n_ims] = init_lle_par + 0.01*np.random.normal(size=ndim)*self.psfs.h_density
                #Add in a Gaussian distribution of model parameters.
                p0[i, self.psfs.ndim * self.n_ims:] = init_par + np.random.normal(size=len(init_par))*walker_sdev
        else:
            for i in range(nwalkers):
                for j in range(self.n_ims):
                    p0[i,j*self.psfs.ndim:(j+1)*self.psfs.ndim] = \
                        self.psfs.tri.points[int(np.random.random()*self.psfs.npsfs)]
                #Add in a Gaussian distribution of model parameters.
                p0[i, self.psfs.ndim * self.n_ims:] = init_par + np.random.normal(size=len(init_par))*walker_sdev
        sampler.run_mcmc(p0,nchain)
        print("Best lnprob: {0:5.2f}".format(np.max(sampler.lnprobability)))
        best_x = sampler.flatchain[np.argmax(sampler.flatlnprobability)] 
        return best_x, sampler
     #Uncomment the following line for FunnelWeb line_profile.

#kernprof -l best_psf_binary
#python -m line_profiler best_psf_binary.lprof
#    @profile   
    def find_best_psfs(self, p_fix, return_lnprob=False):
        """Make a simple fit of every target image, for fixed model parameters.
        
        Parameters
        ----------
        
        p_fix: array-like
            Model parameters, that are fixed when finding the best PSF. Note that for a 
            point source model, this should be [].
        
        """
        
        best_fit_ims = np.empty( (self.n_ims, self.ims[0].shape[0], self.ims[0].shape[1]) )
        best_ixs = np.empty( self.n_ims, dtype=np.int )
        chi2_total = 0.0
        for i in range(self.n_ims):
            #What is our object model?
            obj_ft = self.object.model_uv(p_fix, self.tgt_uv[i])
        
            #Chi-squared
            chi2s = np.empty(self.psfs.npsfs)
            mod_ims = np.empty( (self.psfs.npsfs, self.ims[0].shape[0], self.ims[0].shape[1]) )
            for j in range(self.psfs.npsfs):
                #Convolve the object with the PSF model to form an image model.
                mod_ft = obj_ft * self.psfs.psf_fts[j]
                
                #Find the tilt that best matches the image model, and form an image model,
                #and scale the image to match the total flux.
                #NB This is copied from lnprob, which is a little messy. They should be
                #consolidated!
                scale_factor = self.im_fts[i][0].real/mod_ft[0].real
                tilt = optimize_tilt(mod_ft, self.im_fts[i]/scale_factor, self.psfs.uv)
                mod_ft = optimize_tilt_function(tilt, mod_ft*scale_factor, self.im_fts[i], self.psfs.uv, return_model=True)
                mod_ims[j] = self.psfs.im_from_ft(mod_ft)
            
                #Find the mean square uncertainty of this fit
                chi2s[j] = np.sum( (mod_ims[j] - self.ims[i])**2 * self.ivar[i] )
                #pdb.set_trace()
            
            #Find the best image
            best_ixs[i] = np.argmin(chi2s)
            chi2_total += np.min(chi2s)
            best_fit_ims[i] = mod_ims[best_ixs[i]]
        if return_lnprob:
            return -chi2_total/2.0
        else:
            return best_ixs, best_fit_ims    
            
    def marginalise_best_psf(self, init_par=[],walker_sdev=[],nchain=100, nburnin=50, use_threads=True):
        """Use the affine invariant Monte-Carlo Markov chain technique to marginalise
        over all PSFs. 
        
        This brute force algorithm takes a long time, because for every parameter it fits, 
        it runs a monte-carlo chain which requires nwalkers * nchain evaluations of 
        find_best_psfs, which in turn requires N_psfs * N_target_frames evaluations of
        optimize_tilt.
        
        e.g. if running this over a 100 x 100 grid, fitting for 1 parameter with 6 walkers
        and a chain length of 100, 100 psfs and 50 target frames, this is 3 x 10^10 
        evaluations of optimize_tilt.
        
        An alternative to this would be to just add 2 model parameters per target image,
        i.e. the tilt of each image, which e.g. could be 50 parameters for 50 target 
        images. The problem with this is that it would then require nwalkers to be
        50 times larger in the case of fitting to only 1 parameter (e.g. contrast).
        
        Parameters
        ----------
        
        """
        threads = multiprocessing.cpu_count()
        
        #Minimum number of walkers
        ndim = len(init_par)
        nwalkers = 2*ndim
        kwargs = {"return_lnprob":True}
        if use_threads:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.find_best_psfs, threads=threads, kwargs=kwargs)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.find_best_psfs, kwargs=kwargs)
            
        #Initialise the chain to random parameters.
        p0 = np.empty( (nwalkers, ndim) )
        for i in range(nwalkers):
            p0[i] = init_par + np.random.normal(size=len(init_par))*walker_sdev
        import time
        then = time.time()
        best_ixs, best_fit_ims = self.find_best_psfs(p0[0])
        now = time.time()
        print(now-then)
        print("Running Chain... (burn in)")
        pos, prob, state = sampler.run_mcmc(p0,nburnin)
        sampler.reset()
        print("Running Chain... ")
        sampler.run_mcmc(pos,nchain)
        print("Best lnprob: {0:5.2f}".format(np.max(sampler.lnprobability)))
        best_x = sampler.flatchain[np.argmax(sampler.flatlnprobability)] 
        return best_x, sampler
          