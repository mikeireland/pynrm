"""The main module to implement the Phase Observationally Independent of
Systematic Errors (POISE) algorithm.

"""

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.ndimage as nd
import pdb
import glob
import os
plt.ion()
    
class PYPOISE():
    """The PYPOISE Class, that enables the POISE algorithm.
    The class must be constructed with an AO instrument. """
    def __init__(self, aoinst):
        self.aoinst = aoinst
    def pmask_to_ft(self,pmask):
        """Create Fourier sampling for kerphase based on a pupil mask.
    
        Parameters
        ----------
        pmask: (subarr, subarr) array
    
        Returns
        -------
        RR: (subarr,subarr) array
            Two-dimensional redundancy array, computing how many baselines
            contribute to each Fourier pixel.
        AA: (subarr,subarr,npsi) array
            A two-dimensional array for each pupil position, showing how phase
            at that position contributes to the particular Fourier component phase
            (+1 or -1)
        ftpix: ( (nphi) array, (nphi) array)
            Co-ordinates defining the Fourier sampling.
        """
        if (pmask.shape[0] != pmask.shape[1]):
            print "Error: Only a square pupil mask is allowed."
            raise UserWarning
        subarr = pmask.shape[0]
        #Transpose pmask so that we have the correct ordering of pixels
        #for our real Fourier transform half-plane.
        ww= np.where(np.transpose(pmask))
        npsi = len(ww[0])
        RR = np.zeros((subarr,subarr),dtype='int')
        AA = np.zeros((subarr,subarr,npsi),dtype='int')
        for i in range(0,npsi-1):
            for j in range(i+1,npsi):
                dx = ww[0][j]-ww[0][i]+subarr/2
                dy = ww[1][j]-ww[1][i]+subarr/2
                AA[dy,dx,j] += 1
                AA[dy,dx,i] -= 1
                RR[dy,dx] += 1
        RR = np.roll(np.roll(RR,-subarr/2,axis=0),-subarr/2,axis=1)
        AA = np.roll(np.roll(AA,-subarr/2,axis=0),-subarr/2,axis=1)
        ftpix = np.where(RR)
        return RR,AA,ftpix

    def pupil_sampling(self,in_files, subarr=128, ignore_data=False, ignore_dark=False, 
        dark_file='', flat_file='', out_file='', rdir='', ddir='', 
        destripe=False, dither=False):
        """Define the Fourier-Plane sampling based on Pupil geometry
    
        This function finds the pupil sampling and the matrix that maps phase to 
        closure-quantities, i.e. kernel-phases.
    
        Parameters
        ----------
        h : A fits header dictionary-like structure returned from pyfits or
            astropy.io.fits. Must be from a known camera in a known mode.
        subarr : int
            The sub-array size 
        ignore_data : bool
            Whether to ignore the data in finding the precise Fourier sampling.
    
        Returns
        -------
        ftpix: ( (N) array, (N) array)
        fmask: (subarr,subarr) array
        ptok: (M,N) array
            A matrix that maps pupil-phases to kernel-phases.
        """
        #Allow over-riding default reduction and data directories.
        if (rdir == ''):
            rdir = self.aoinst.rdir
        if (ddir == ''):
            ddir = self.aoinst.ddir
        try:
            in_fits = pyfits.open(ddir + in_files[0], ignore_missing_end=True)
        except:
            in_fits = pyfits.open(ddir + in_files[0] + '.gz', ignore_missing_end=True)
        h = in_fits[0].header
        in_fits.close()

        #Based on which telescope is in use, decide on the mask and wavelength.
        hinfo = self.aoinst.info_from_header(h)
        wave = hinfo['wave']
        rad_pixel = hinfo['rad_pixel']
        if (len(dark_file) == 0):
            dark_file = hinfo['dark_file']
        print "Using Dark file: ", dark_file
        if (len(flat_file) == 0):
            flat_file = hinfo['flat_file']
        #Fill in the filename...
        if (out_file == ''):
            out_file = hinfo['ftpix_file']
        print "Using Flat file: ", flat_file
        #Short-hand
        p = hinfo['pupil_params']
    
        #Create the pupil mask the same size as the subarray...
        pmask = np.zeros((subarr,subarr))
        #The "bighole" pupil mask is 1.3 times larger, to take into account scattered
        #light from broad filters.
        pmask_bighole = pmask.copy()
        #ix[0] is x on the screen, i.e.the *second* index.
        ix = np.meshgrid(np.arange(subarr)-subarr/2,np.arange(subarr)-subarr/2)
        if (hinfo['pupil_type'] == 'circ_nrm'):
            mask_rotation = p['mask_rotation']
            hole_diam = p['hole_diam']
            badpmask_threshold=0.2
            lowps_threshold=0.1
            nbad_threshold = 4
            rmat = np.array([[np.cos(mask_rotation),np.sin(mask_rotation)],
                        [-np.sin(mask_rotation), np.cos(mask_rotation)]])
            hole_xy = np.dot(rmat,p['hole_xy']) #!!! Correct rotation direction ???
            pxy = hole_xy/wave*rad_pixel*subarr
            for i in range(hole_xy.shape[1]):
                pmask += ( (ix[0] - pxy[0,i])**2 + (ix[1] - pxy[1,i])**2 < (hole_diam/wave*rad_pixel*subarr/2.0)**2 )
                pmask_bighole += ( (ix[0] - pxy[0,i])**2 + (ix[1] - pxy[1,i])**2 < (1.3*hole_diam/wave*rad_pixel*subarr/2.0)**2 )
            #The following two lines seem to be needed for NIRC2
            pmask = np.transpose(pmask)
            pmask_bighole = np.transpose(pmask_bighole)
        elif (hinfo['pupil_type'] == 'annulus'):
            outer_diam = p['outer_diam']
            inner_diam = p['inner_diam']
            badpmask_threshold = 0.4
            lowps_threshold=0.1 #Some tweaking needed !!! Was 0.05
            nbad_threshold = 12
            pmask += ( ix[0]**2 + ix[1]**2 < (outer_diam/wave*rad_pixel*subarr/2.0)**2 )
            pmask_bighole += ( ix[0]**2 + ix[1]**2 < (1.1*outer_diam/wave*rad_pixel*subarr/2.0)**2 )
            pmask -= ( ix[0]**2 + ix[1]**2 < (inner_diam/wave*rad_pixel*subarr/2.0)**2 )
            pmask_bighole -= ( ix[0]**2 + ix[1]**2 < (inner_diam/wave*rad_pixel*subarr/2.0)**2 )
        elif (hinfo['pupil_type'] == 'keck'):
            badpmask_threshold = 0.4
            lowps_threshold = 0.1
            nbad_threshold = 4
            #This is a special pupil geometry for Keck, consisting of lots of hexagons.
            hex = self.aoinst.hexagon( subarr, np.ceil(p['segment_size']/wave*rad_pixel*subarr))
            ys = p['segment_size']/wave*rad_pixel*subarr
            xs = p['segment_size']/wave*rad_pixel*subarr*np.sqrt(3)/2.0
            p4 = nd.interpolation.shift(hex,(0,-1.5*ys),order=1) + nd.interpolation.shift(hex,(0,-0.5*ys),order=1) + \
                 nd.interpolation.shift(hex,(0,0.5*ys),order=1) + nd.interpolation.shift(hex,(0,1.5*ys),order=1)
            p5 = nd.interpolation.shift(hex,(0,-2*ys),order=1) + nd.interpolation.shift(hex,(0,ys),order=1) + \
                 nd.interpolation.shift(hex,(0,2*ys),order=1) + nd.interpolation.shift(hex,(0,-ys),order=1) + hex
            pmask = nd.interpolation.shift(p4,(-3*xs,0),order=1) + nd.interpolation.shift(p4,(-xs,0),order=1) + \
                nd.interpolation.shift(p4,(xs,0),order=1)+nd.interpolation.shift(p4,(3*xs,0),order=1) + \
                nd.interpolation.shift(p5,(-2*xs,0),order=1)+p5 + nd.interpolation.shift(p5,(2*xs,0),order=1) + \
                nd.interpolation.shift(hex,(-xs,2.5*ys),order=1)+nd.interpolation.shift(hex,(-xs,-2.5*ys),order=1) + \
                nd.interpolation.shift(hex,(xs,2.5*ys),order=1)+nd.interpolation.shift(hex,(xs,-2.5*ys),order=1) + \
                nd.interpolation.shift(hex,(0,-3*ys),order=1)+ nd.interpolation.shift(hex,(0,3*ys),order=1)
            pmask *= (1 - self.aoinst.hexagon(subarr, np.ceil(p['obstruction_size']/wave*rad_pixel*subarr)))
            pmask = np.minimum(pmask,1)
            #Next rotate 
            #!!! Possibly, the sign should be positive on the next line and the pupil image
            #transposed... not sure-but it works for vertang=-44 !!!
            pmask = nd.interpolation.rotate(pmask,-hinfo['vertang_pa'],order=1,reshape=False)
            pmask = pmask > 0.75
            pmask_bighole = pmask > 0.1

        #Now, we create the pupil to kernel-phase array from this pupil mask.
        #Start with intrinsically 2D quantities, and the bighole version first
        RR_bighole,AA_bighole,ftpix_bighole = self.pmask_to_ft(pmask_bighole)
        #Create the Fourier transform mask for bad pixel rejection and bias
        #subtraction. It looks a little complex because we are using rfft2 to 
        #save time.
        fmask = np.ones((subarr,subarr/2+1),dtype='int')
        fmask[ftpix_bighole] = 0
        #Even though we're in half the Fourier plane, there are some doubled-up pixels.
        fmask[:,0] *= np.roll(fmask[::-1,0],1)
        fmask[0,0]=0
        #Now the normal version. 
        RR,AA,ftpix = self.pmask_to_ft(pmask)
        npsi = AA.shape[2]
        ww = np.where(pmask)
        #We can now compute ftpix,then tweak the pmask to match the actual Fourier
        #power
        if not ignore_data:
            if ignore_dark:
                dark_file=''
            if dither:
                cube = self.aoinst.clean_dithered(in_files, dark_file=dark_file, flat_file=flat_file,
                    fmask=fmask, subarr=subarr, destripe=destripe, ddir='', rdir='')
            else:
                cube = self.aoinst.clean_no_dither(in_files, dark_file=dark_file, flat_file=flat_file,
                    fmask=fmask, subarr=subarr, destripe=destripe, ddir='', rdir='')
            for i in range(cube.shape[0]):
                ftim = self.aoinst.shift_and_ft(cube[i])
                if (i == 0):
                    ps = np.abs(ftim)**2
                    ftim_sum = ftim
                else:
                    ps += np.abs(ftim)**2
                    ftim_sum += ftim
            ps -= np.mean(ps[np.where(fmask)])
            phases = np.angle(ftim_sum)
            #We now find any power spectrum values that are unusually low, or any phases
            #that are unusually high.
            medrat = np.median(ps[ftpix]/RR[ftpix]**2)
            medps = np.median(ps[ftpix])
            badft = np.where( np.logical_or( np.abs(phases[ftpix]) > 2.5, \
                np.logical_and(ps[ftpix]/RR[ftpix]**2/medrat < badpmask_threshold,ps[ftpix]/medps < lowps_threshold)) )
            badft = badft[0]        
            print len(badft), " Fourier positions of low power identified"
            nbadft = np.zeros(npsi)
            some_ones = np.ones(len(badft),dtype='int')
            for i in range(0,npsi):
                nbadft[i] = np.sum(np.abs(AA[(ftpix[0][badft],ftpix[1][badft],i*some_ones)]))
            badpmask = np.where(nbadft > nbad_threshold)[0]
            oldpmask = pmask.copy()
            pmask[ww[0][badpmask],ww[1][badpmask]] = 0
            print len(badpmask), " low signal pupil positions eliminated."
            #The next 2 lines are pretty useful for demonstrating how this works happens.
            plt.clf()
            plt.imshow(pmask + oldpmask,interpolation='nearest')
            plt.draw()
            #import pdb; pdb.set_trace()
            #plt.plot(nbadft,'o')
            #plt.semilogy(ps[ftpix]/RR[ftpix])
            #RRsmall = RR[0:subarr,0:subarr/2+1]
            #mask = np.zeros((subarr,subarr/2+1))
            #mask[ftpix]=1
            #plt.imshow( (mask*np.maximum(ps,0)/np.maximum(RRsmall,0.01)**2)**0.3, interpolation='nearest') 
            #raise UserWarning
        #With a (potentially) updated pmask, re-compute the ftpix variables
        RR,AA,ftpix = self.pmask_to_ft(pmask)
        npsi = AA.shape[2]
        #Now convert to 1D quantities, indexed by the number of non-zero Fourier points, nphi
        nphi = len(ftpix[0])
        A = np.zeros((nphi,npsi),dtype='int')
        one_ix = np.ones(nphi,dtype='int')
        #Extend the ftpix index to include each of the psi variables, one at a time.
        for j in range(0,npsi):
            A[:,j] = AA[ftpix + (j*one_ix,)]

        #At this point, we have:
        #R: The reduncancy vector
        #xf: The x Fourier component.
        #yf: The y Fourier component.
        #A: A matrix relating the npsi pupil phases to
        #    the nphi Fourier components.

        print("Finally doing the matrix stuff!")
    
        #As overall piston doesn't matter, we can remove one row (column?) 
        #of A, i.e. setting one Fourier component to zero phase. Frantz did this anyway.
        A = A[:,0:npsi]
    
        U, W, V = np.linalg.svd(A)

        print('SVDC Complete.')

        # Now the matrix A is equal to U#diag_matrix(W)#V, or

        ptok = np.transpose(U[:,len(W):])

        #If a savefile name is given, save the file!
        if (len(out_file) > 0):
            hl = pyfits.HDUList()
            header = pyfits.Header()
            header['RAD_PIX'] = rad_pixel
            header['SUBARR'] = subarr
            header['TARGNAME'] = hinfo['targname']
            header['DARK'] = dark_file
            header['FLAT'] = flat_file
            header['PFILTER'] = hinfo['filter']
            header['PWAVE'] = wave 
            header['PUPTYPE'] = hinfo['pupil_type']
            if(hinfo['pupil_type'] == 'circ_nrm'):
                header['PMASK'] = p['mask']
            header['DDIR'] = ddir
            for i in range(len(in_files)):
                header['HISTORY'] = 'Input: ' + in_files[i]
            hl.append(pyfits.ImageHDU(np.array(ftpix),header))
            hl.append(pyfits.ImageHDU(fmask))
            hl.append(pyfits.ImageHDU(ptok))
            hl.writeto(rdir + out_file,clobber=True)

        return ftpix, fmask, ptok, pmask
    
    def extract_kerphase(self,ptok_file='',ftpix=([],[]),fmask=[],ptok=[],cube=[],cube_file='',
        add_noise=0,rnoise=5.0,gain=4.0,out_file='',recompute_ptok=False,
        use_poise=False,systematic=[],cal_kp=[],summary_file='',pas=[0],
        ptok_out='',window_edges=True, rdir='', cdir='', use_powerspect=True):
        """Extract the kernel-phases.
    
        Parameters
        ----------
        cube: (Nframes,Nx,Ny) array (dtype=float)
            The cleaned input data
        ftpix: (K array,K array) 
            The Fourier pixel sampling points, from pupil_sampling
        ptok: (M,K) array
            The phase to kernel-phase matrix, from pupil_sampling
        add_noise: int
            Multiplier for the number of frames to add fake noise to. 0 turns
            this feature off.
        rnoise: float
            Readout noise in DN
        gain: float
            Gain in electrons/DN. Photon variance in DN is N/gain
        out_file: string, optional
            File to save the kernel-phases to.
        cube_file: string, optional
            Optional file input.
        recompute_ptok: bool, optional
            Recompute the covariance matrix based on the data - systematic errors are 
            added then the covariance matrix re-diagonalised. Important for highly 
            resolved objects.
            WARNING: Not implemented yet.
    
        Returns
        -------
        kp: (Nframes*extra_noisy_frames,M)
            Kernel phase array.
        ps: (Nframes*extra_noisy_frames,K)
            Normalised power spectrum array.
        
        Issues
        ------
        Some Fourier components can end up wrapping around. This obviously destroys 
        the kernel-phases. There are two improvements needed to fix this...
        a) The bright calibrator mean phases should be subtracted prior to 
        target phase extraction.
        b) High-Variance phases for the target have to be weighted to zero by
        re-diagonalizing the covariance matrix and saving a new ptok file.
        c) Weighted averaging by amplitude should be added.
        """
        # ******  Sanity check the input data and allow the caller to over-write just about anything ******
        #Allow the user to over-write the default reduction and cube directories
        if (rdir == ''):
            rdir = self.aoinst.rdir
        if (cdir == ''):
            cdir = self.aoinst.cdir
        targname = ''
        #Input the cube (or extract from file)
        if (len(cube_file) > 0):
            cube = pyfits.getdata(cdir + cube_file,0)
            cube_extn = pyfits.getdata(cdir + cube_file,1)
            pas = cube_extn['pa']
            backgrounds = cube_extn['background']
            h = pyfits.getheader(cdir + cube_file)
            targname = h['TARGNAME']
            rnoise = h['RNOISE']*2 #!!! This should be removed, or put in as an option !!!
            gain = h['PGAIN']
            hinfo = self.aoinst.info_from_header(h)
            #If we don't over-write the ptok_file, use the default
            if len(ptok_file)==0:
                ptok_file = hinfo['ftpix_file']
        if (len(cube.shape) != 3):
            print "Error: cube doesn't have 3 dimensions! specify cube or cube_file"
            raise UserWarning

        #Input the kernel-phase extraction parameters. By default, we get this from 
        #a file, unless there are additional inputs.
        if (len(ptok_file) != 0):
            #First... sort out the path. The ptok_file should be in 
            #the cdir if it is specific to a data set (e.g. a POISE file)
            #and should be in the rdir otherwise.
            if os.path.isfile(cdir + ptok_file):
                ptok_file_full = cdir + ptok_file
            else:
                ptok_file_full = rdir + ptok_file
            if len(ftpix[0]) == 0:
                ftpix = pyfits.getdata(ptok_file_full,0)
            if len(fmask) == 0:
                fmask = pyfits.getdata(ptok_file_full,1)
            if len(ptok) == 0:
                ptok = pyfits.getdata(ptok_file_full,2)
        #Now some double-checking that we have all our variables.
        if (len(ptok.shape) != 2):
            print "Error: pupil to kerphase matrix doesn't have 2 dimensions! specify ftpix_file or ptok"
            raise UserWarning
        if (len(ftpix[0]) == 0):
            print "Error: no Fourier pixels specified! Specify ftpix or ptok_file"
            raise UserWarning
        if (len(fmask) == 0):
            print "Error: no Fourier zero mask specified! Specify fmask or ptok_file"
            raise UserWarning
        #Extract the POISE variables if needed
        if (use_poise):
            if (len(ptok_file) > 0):
                systematic = pyfits.getdata(ptok_file_full,3)
                cal_kp = pyfits.getdata(ptok_file_full,4)
                if (use_powerspect):
                    ps_proj_matrix = pyfits.getdata(ptok_file_full, 5)
                    ps_systematic = pyfits.getdata(ptok_file_full, 6)
                    cal_ps = pyfits.getdata(ptok_file_full,7)
            if (len(systematic) == 0):
                print "Error: systematic error component must be input if use_poise is set"
                return UserWarning
        #Put ftpix into the appropriate tuple format.
        ftpix = (ftpix[0],ftpix[1])
        
        #***** Initialise Arrays *****
        nf = cube.shape[0]*np.max((1,add_noise))
        #nf_nonoise is the number of frames from the input data, i.e. before adding noise
        #to explore the statistics of the kernel-phases etc.
        nf_nonoise = cube.shape[0]
        npsi = ptok.shape[0]
        nphi = ptok.shape[1]
        if use_powerspect and use_poise:
            #Number of power spectra combinations.
            nps_comb = ps_proj_matrix.shape[0]

        kp = np.zeros((nf,npsi))
        kp_nonoise = np.zeros((nf_nonoise,npsi))
        ps = np.zeros((nf,nphi))
        ps_nonoise = np.zeros((nf_nonoise,nphi))
        if use_powerspect and use_poise:
            ps_comb = np.zeros((nf,nps_comb))
            ps_comb_nonoise = np.zeros((nf_nonoise,nps_comb))
        plt.clf()
        plt.axis([0,len(ftpix[0]),-3,3])
        for i in range(nf):
            if (add_noise > 0):
                im_raw = cube[i//add_noise,:,:]
                if (i % add_noise == 0):
                    sig = 0.0
                else:
                    sig = np.sqrt(np.maximum(im_raw + backgrounds[i//add_noise],0)/gain + rnoise**2)
                im = im_raw + sig*np.random.normal(size=im_raw.shape)
            else:
                im = cube[i,:,:]
            if (window_edges):
                for j in range(9):
                    im[j,:] *= (j + 1.0)/10.0
                    im[:,j] *= (j + 1.0)/10.0
                    im[-j-1,:] *= (j + 1.0)/10.0
                    im[:,-j-1] *= (j + 1.0)/10.0
            #NB: We really want to minimise the phase variance, because
            #phase wraps kill kernel-phases.
            ftim = self.aoinst.shift_and_ft(im, ftpix=ftpix)
            #Compute power spectrum (should the bias in the next line be a median or a mean?)
            ps_bias = np.mean(np.abs(ftim[np.where(fmask)])**2)
            ps[i,:] = (np.abs(ftim[ftpix])**2 - ps_bias)/abs(ftim[0,0])**2
            #... and compute the phases.
            phases = np.angle(ftim[ftpix])
            #Need 3 plots here (phase, kernel-phase and calibrated power spectra)
            plt.plot(phases,'.')
            plt.xlabel('Fourier Index')
            plt.ylabel('Phase (radians) or Calibrated PS')
            kp[i,:] = np.dot(ptok,phases)
            #If use_poise is set, then we need to subtract the calibrator kp
            if (use_poise):
                kp[i,:] -= cal_kp
                if use_powerspect:
                    ps[i,:] /= cal_ps
                    plt.plot(ps[i,:], '.')
                    ps_comb[i,:] = np.dot(ps_proj_matrix, ps[i,:]-1)
            if (add_noise > 0):
                if (i % add_noise == 0):
                    kp_nonoise[i//add_noise,:] = kp[i,:]
                    if use_powerspect:
                        ps_nonoise[i//add_noise,:] = ps[i,:]
                        if use_poise:
                            ps_comb_nonoise[i//add_noise,:] = ps_comb[i,:]
            if (i % 10 == 9):
                print("Done file: " + str(i))
    #            plt.plot(kp[i,:])
                plt.draw()
                plt.clf()
                plt.axis([0,len(ftpix[0]),-3,3])
            
        #Compute statistics, to save to the summary file. If use_poise is set, 
        #then we also need to add the systematic error component.
        if len(summary_file) > 0:
            #Compute the mean kernel-phase. !!! Some bad frame rejection should
            #have occured ...
            kp_mn = kp.mean(0)    
            kp_cov = np.dot(np.transpose(kp),kp)/nf - np.outer(kp_mn,kp_mn)
            if (add_noise > 0):
                kp_mn = kp_nonoise.mean(0)
            #Don't count the add_noise frames in computing the standard error of
            #the mean.
            kp_cov /= (cube.shape[0]-1.0)
            diag_ix = (range(npsi),range(npsi))
            if (use_poise):
                kp_cov[diag_ix] += systematic
            #Now we only save the diagonal of the covariance matrix
            kp_sig = np.sqrt(np.diagonal(kp_cov))
            hl = pyfits.HDUList()
            header = pyfits.Header()
            header['PTOKFILE'] = ptok_file
            header['CUBEFILE'] = cube_file
            header['TARGNAME'] = targname
            if len(out_file) > 0:
                header['KPFILE'] = out_file
            header['PA'] = np.mean(pas) #!!! This assumes that all frames are used
            col1 = pyfits.Column(name='kp_mn', format='E', array=kp_mn)
            col2 = pyfits.Column(name='kp_sig', format='E', array=kp_sig)
            cols = pyfits.ColDefs([col1, col2])
            hl.append(pyfits.PrimaryHDU(header=header))
            hl.append(pyfits.new_table(cols))
            #Given that there are different numbers of rows, we need another extension for
            #power spectra.
            if use_powerspect:
                #Compute the mean power spectra
                # !!! Some bad frame rejection should
                #have occured ...
                ps_mn = ps.mean(0)    
                ps_cov = np.dot(np.transpose(ps),ps)/nf - np.outer(ps_mn,ps_mn)
                ps_mn = ps_nonoise.mean(0)
                #Don't count the add_noise frames in computing the standard error of
                #the mean.
                ps_cov /= (cube.shape[0]-1.0)
                #Now we only save the diagonal of the covariance matrix
                ps_sig = np.sqrt(np.diagonal(ps_cov))
                col1 = pyfits.Column(name='ps_mn', format='E', array=ps_mn)
                col2 = pyfits.Column(name='ps_sig', format='E', array=ps_sig)
                cols = pyfits.ColDefs([col1, col2])
                hl.append(pyfits.new_table(cols))
                if (use_poise):
                    ps_comb_mn = ps_comb.mean(0)
                    ps_comb_cov = np.dot(np.transpose(ps_comb),ps_comb)/nf - np.outer(ps_comb_mn,ps_comb_mn)
                    #import pdb; pdb.set_trace()
                    ps_comb_mn = ps_comb_nonoise.mean(0)
                    #Don't count the add_noise frames in computing the standard error of
                    #the mean.
                    ps_comb_cov /= (cube.shape[0]-1.0)
                    diag_ix = (range(nps_comb),range(nps_comb))
                    ps_comb_cov[diag_ix] += ps_systematic
                    #Now we only save the diagonal of the covariance matrix
                    ps_comb_sig = np.sqrt(np.diagonal(ps_comb_cov))
                    col1 = pyfits.Column(name='ps_comb_mn', format='E', array=ps_comb_mn)
                    col2 = pyfits.Column(name='ps_comb_sig', format='E', array=ps_comb_sig)
                    cols = pyfits.ColDefs([col1, col2])
                    hl.append(pyfits.new_table(cols))                
                
            hl.writeto(cdir + summary_file,clobber=True)

        if (len(out_file) > 0):
            hl = pyfits.HDUList()
            header = pyfits.Header()
            header['CUBEFILE'] = cube_file
            header['PTOKFILE'] = ptok_file
            hl.append(pyfits.ImageHDU(kp,header))
            hl.append(pyfits.ImageHDU(ps))
            hl.writeto(cdir + out_file,clobber=True)
        return kp, ps

    def poise_kerphase(self,kp_files,ptok=[],ftpix=([],[]),fmask=[],beta=0.4,
            out_file='',rad_pixel=0.0,subarr=128, use_powerspect=True, rdir='',cdir=''):
        """Extract the POISE kernel-phases, returning a new ptok matrix. Also extract the
        POISE power spectrum.
    
        Parameters
        ----------
        kp_files: list of strings
            List of kp filenames to turn in to kernel phases.
        ptok: (K,M) array
            The phase to kernel-phase array.
    
        Returns
        -------
        ptok_poise: (K_new,M) array
    
        Notes
        -----
        This works on files rather than all the data, because it could take up
        too much memory. As bad files can't be rejected until kernel-phases are
        computed, saving kp files is the only way to only compute them once. 
        """
        if (rdir == ''):
            rdir = self.aoinst.rdir
        if (cdir == ''):
            cdir = self.aoinst.cdir
        hkp = pyfits.getheader(kp_files[0])
        #Hopefully, the ptok variables are stored as part of the kp_file.
        ptok_file = hkp['PTOKFILE']
        if len(ptok_file) > 0:
            print "Extracting ptok and other variables from file"
            ftpix = pyfits.getdata(rdir + ptok_file,0)
            fmask = pyfits.getdata(rdir + ptok_file,1)
            ptok = pyfits.getdata(rdir + ptok_file,2)
            hptok = pyfits.getheader(rdir + ptok_file)
            rad_pixel = hptok['RAD_PIX']
            subarr = hptok['SUBARR']
        
        #Let's sanity-check
        if len(ptok.shape) != 2:
            print "Error: no valid ptok matrix. ptok or a kp_file with embedded ptok fits filename must be input."
            return UserWarning
        npsi = ptok.shape[0]
        nphi = ptok.shape[1]
        if len(fmask) == 0:
            print "Error: A Fourier-plane mask fmask must set or embedded in the ptok fits file."
            return UserWarning
        ftpix = (ftpix[0],ftpix[1])
        ncubes = len(kp_files)
    
        #----- First, process the phases -----
        kp_cube_covs = np.zeros((ncubes,npsi,npsi))
        kp_cov = np.zeros((npsi,npsi))
        kp_mns = np.zeros((ncubes,npsi))
        kp_mn = np.zeros(npsi)
        nf_all = 0 #Total number of frames
        for i in range(ncubes):
            kp = pyfits.getdata(kp_files[i])
            kp_mns[i,:] = kp.sum(0)
            kp_mn += kp_mns[i,:]
            nf = kp.shape[0]
            kp_mns[i,:] /= nf
            nf_all += nf
            #Create the overall covariance
            kp_cov += np.dot(np.transpose(kp),kp)
            #Now correct for the mean...
            for j in range(nf):
                kp[j,:] -= kp_mns[i,:]
            cov = np.dot(np.transpose(kp),kp)
            kp_cube_covs[i,:,:] = cov/(nf - 1.0)
        kp_mn /= nf_all
        #Correct for the kp_mn (which is also subtracted from target phases)
        kp_cov -= nf_all * np.outer(kp_mn,kp_mn)
        kp_cov /= (nf_all - 1)
    
        #Diagonalise kp_cov, and project the cube kernel-phases onto this
        #new space. Not that the transpose of W is a projection matrix, but the
        #rows of kp_mns are the kernel_phases
        D, V = np.linalg.eigh(kp_cov)
        kp_mns = np.dot(kp_mns, V)
        kp_mn = np.dot(kp_mn,V)
        for i in range(ncubes):
            kp_cube_covs[i,:,:] = np.dot(np.transpose(V),np.dot(kp_cube_covs[i,:,:],V))
        #This should work but is way slow!     kp_cube_covs = np.dot(kp_cube_covs,V)
    
        #Now lets compare the eigenvalues to the internal variance to discover
        #the "systematic" errors
        cov_internal = np.mean(kp_cube_covs, axis=0)
        var_internal = np.diagonal(cov_internal)
        var_sys = np.maximum(D-var_internal,0)
    
        #Make a plot that shows these variances
        plt.clf()
        plt.semilogy(D)
        plt.semilogy(var_internal)
        minvar = 1e-8
        good = np.where( (var_sys/var_internal < beta) | (D < minvar) )[0]
        bad = np.where( (var_sys/var_internal >= beta) & (D >= minvar))[0]
        plt.semilogy(bad,D[bad],'o')
        plt.xlabel('Kernel Phase Number')
        plt.ylabel('Variance')
        plt.title('Rejected Kernel-phases')
        plt.axis( (0,len(var_internal),minvar,np.max(D)) )
        plt.show()
        print('Num Kernel-phases rejected for calibration: ' + str(len(bad)))
        print('Num Good Kernel-phases: ' + str(len(good)))
        print V.shape
        print ptok.shape
        ptok_poise = np.dot(np.transpose(V),ptok)
        #Show the Fourier response of the rejected kernel-phases
        if len(ftpix)>0:
            for i in range(len(bad)):
                print "Click for next figure..."
                dummy = plt.ginput(1)
                ysz = np.max(ftpix[0])+1
                xsz= np.max(ftpix[1])+1
                ft_temp = np.zeros((ysz,xsz))
                ft_temp[ftpix] = 0.1
                ft_temp[ftpix] += ptok_poise[bad[i],:]
                ft_temp = np.roll(ft_temp,ysz/2,axis=0)
                delta_y = int(np.min([xsz*1.1,ysz/2]))
                ft_temp = ft_temp[ysz/2-delta_y:ysz/2+delta_y,:]
                plt.clf()
                plt.imshow(ft_temp,interpolation='nearest')
                plt.title('Bad kerphase ' + str(bad[i]))
                plt.draw()
        ptok_poise = ptok_poise[good,:]
        systematic = var_sys[good]
        kp_mn = kp_mn[good]
    
        #---- Repeat for amplitudes: slightly different at all steps but mostly a cut and past of above
        #     with "kp" replaced with "ps" and "npsi" with "nphi", ptok_poise with ps_proj_matrix.
        #      We also divide by the mean power spectrum without doing anything.
        #     The inputs are the kp_files, the extension number, beta.
        #     For plotting, ftpix and a variable type string are needed.  ----
        if (use_powerspect):
            #Free memory
            kp_cube_covs = []
            #Initialise variables
            ps_cube_covs = np.zeros((ncubes,nphi,nphi))
            ps_cov = np.zeros((nphi,nphi))
            ps_mns = np.zeros((ncubes,nphi))
            ps_mn = np.zeros(nphi)
            nf_all = 0 #Total number of frames
            for i in range(ncubes):
                ps = pyfits.getdata(kp_files[i],1)
                ps_mn += ps.sum(axis=0)
                nf = ps.shape[0]
                nf_all += nf
            ps_mn /= nf_all
            ps_mn_corrected = np.zeros(nphi)
            for i in range(ncubes):
                ps = pyfits.getdata(kp_files[i],1)
                nf = ps.shape[0]
                #Calibrate by the mean power spectrum, and remove the
                #unity offset (i.e. equivalent to subtracting a point source)
                for j in range(nf):
                    ps[j,:] /= ps_mn
                    ps[j,:] -= 1
                ps_mns[i,:] = ps.sum(axis=0)
                ps_mn_corrected += ps_mns[i,:]
                ps_mns[i,:] /= nf
                #Create the overall covariance ignoring the mean...
                ps_cov += np.dot(np.transpose(ps), ps)
                #Now correct for the mean...
                for j in range(nf):
                    ps[j,:] -= ps_mns[i,:]
                cov = np.dot(np.transpose(ps), ps)
                ps_cube_covs[i,:,:] = cov/(nf - 1.0)
            #These next 2 lines subtract 0. But I am not 100% certain that it is
            #exactly zero...
            ps_mn_corrected /= nf_all
            ps_cov -= nf_all * np.outer(ps_mn_corrected,ps_mn_corrected)
            ps_cov /= (nf_all - 1)
        
            #Diagonalise ps_cov, and project the cube kernel-phases onto this
            #new space. Not that the transpose of W is a projection matrix, but the
            #rows of kp_mns are the kernel_phases
            D, V = np.linalg.eigh(ps_cov)
            ps_mns = np.dot(ps_mns, V)
            #Note that we keep ps_mn without multiplying by V, because we need to apply
            #this before calculating the powerspectrum linear combinations.
            for i in range(ncubes):
                ps_cube_covs[i,:,:] = np.dot(np.transpose(V),np.dot(ps_cube_covs[i,:,:],V))
    
            #Now lets compare the eigenvalues to the internal variance to discover
            #the "systematic" errors
            cov_internal = np.mean(ps_cube_covs, axis=0)
            var_internal = np.diagonal(cov_internal)
            var_sys = np.maximum(D-var_internal,0)

            #This next line is where we need to find why the first element of the ps_proj_matrix looks like
            #Strehl. Surely, this has to have a large systematic component!
            #import pdb; pdb.set_trace()

    
            #Make a plot that shows these variances
            # !!! This is the point where things slightly change
            # (but not if a "ptok" is an identity matrix) !!!
            ps_proj_matrix = np.transpose(V)
        
            #Now display diagnostics.
            plt.clf()
            plt.semilogy(D)
            plt.semilogy(var_internal)
            good = np.where(var_sys/var_internal < beta)[0]
            bad = np.where(var_sys/var_internal >= beta)[0]
            plt.semilogy(bad,D[bad],'o')
            plt.xlabel('Power Spectrum Number')
            plt.ylabel('Variance')
            plt.title('Rejected Power Spectrum combinations')
            plt.show()
            print('Num Power Spectrum combinations rejected for calibration: ' + str(len(bad)))
            print('Num Good Power Spectrum combinations: ' + str(len(good)))
            print V.shape
            print ps_proj_matrix.shape
            #Show the Fourier response of the rejected power spectra
            if len(ftpix)>0:
                for i in range(len(bad)):
                    print "Click for next figure..."
                    dummy = plt.ginput(1)
                    ysz = np.max(ftpix[0])+1
                    xsz= np.max(ftpix[1])+1
                    ft_temp = np.zeros((ysz,xsz))
                    ft_temp[ftpix] = 0.1
                    ft_temp[ftpix] += ps_proj_matrix[bad[i],:]
                    ft_temp = np.roll(ft_temp,ysz/2,axis=0)
                    delta_y = int(np.min([xsz*1.1,ysz/2]))
                    ft_temp = ft_temp[ysz/2-delta_y:ysz/2+delta_y,:]
                    plt.clf()
                    plt.imshow(ft_temp,interpolation='nearest')
                    plt.title('Bad power spectrum combination ' + str(bad[i]))
                    plt.draw()
            ps_proj_matrix = ps_proj_matrix[good,:]
            ps_systematic = var_sys[good]
        else:
            ps_proj_matrix=[]
            ps_systematic=[]
            ps_mn=[] 
    
        #If a savefile name is given, save the file!
        if (len(out_file) > 0):
            if (rad_pixel == 0):
                print "Missing rad_pixel variable!"
                return UserWarning
            hl = pyfits.HDUList()
            header = pyfits.Header()
            header['RAD_PIX'] = rad_pixel
            header['SUBARR'] = subarr
            for i in range(ncubes):
                header['HISTORY'] = 'Input: ' + kp_files[i]
            hl.append(pyfits.ImageHDU(ftpix,header))
            hl.append(pyfits.ImageHDU(fmask))
            hl.append(pyfits.ImageHDU(ptok_poise))
            hl.append(pyfits.ImageHDU(systematic))
            hl.append(pyfits.ImageHDU(kp_mn))
            if (use_powerspect):
                hl.append(pyfits.ImageHDU(ps_proj_matrix))
                hl.append(pyfits.ImageHDU(ps_systematic))
                hl.append(pyfits.ImageHDU(ps_mn))
            hl.writeto(cdir + out_file,clobber=True)
        return ptok_poise,systematic,kp_mn, ps_proj_matrix, ps_systematic, ps_mn

    def kp_to_implane(self,ptok_file=[], pas=[0], summary_files=[], pxscale=10.0, sz=128,
            out_file='',rdir='', cdir='',use_powerspect=True):
        """Here we convert kernel-phases (and power spectra linear combinations) to their image-plane representation,
        in sky coordinates. 
    
        Parameters
        ----------
        ptok_file: string
            The input fits file containing the ptok matrix
        pas: float, optional
            The position angles of vertical
        summary_files: string array, optional
            The list of filenames to get position angles and summary data from
        pxscale: float
            Output pixel scale in arcsec per pixel
        """
        if (rdir == ''):
            rdir = self.aoinst.rdir
        if (cdir == ''):
            cdir = self.aoinst.cdir
        # Read in details from summary_files
        if len(summary_files) > 0:
            pas = np.zeros(len(summary_files))
            kp_mn = np.array([])
            kp_sig = np.array([])
            for i in range(len(summary_files)):
                h = pyfits.getheader(summary_files[i])
                d = pyfits.getdata(summary_files[i])
                pas[i] = h['PA']
                kp_mn = np.append(kp_mn,d['kp_mn'])
                kp_sig = np.append(kp_sig,d['kp_sig'])
                #If we're using the power spectra, then just append the power spectrum
                #combinations to the kernel-phase combinations, after converting power
                #spectra to amplitudes by dividing by 2.
                if use_powerspect:
                    d = pyfits.getdata(summary_files[i],3)
                    kp_mn = np.append(kp_mn,0.5*d['ps_comb_mn'])
                    kp_sig = np.append(kp_sig,0.5*d['ps_comb_sig'])
            ptok_file = h['PTOKFILE']
            # all_data is simply 2 columns: means and errors for variables.
            # It makes imaging easy, as where each measurement came from is
            # abstracted.
            all_data = np.array([kp_mn,kp_sig])
            print all_data.shape
        if os.path.isfile(cdir + ptok_file):
            ptok_file_full = cdir + ptok_file
        else:
            ptok_file_full = rdir + ptok_file
        ftpix = pyfits.getdata(ptok_file_full,0)
        ptok = pyfits.getdata(ptok_file_full,2)
        if use_powerspect:
            ps_proj_matrix = pyfits.getdata(ptok_file_full, 5)
        # The ptok_file gives us additional header information...
        header = pyfits.getheader(ptok_file_full)
        rad_pixel = header['RAD_PIX'] 
        subarr = header['SUBARR'] 
            
        # From here, we work through each element of ftpix one at a time for each
        # target position angle. First, sort out some preliminaries...
        nt = len(pas)
        #!!! Testing !!!
        #pas[:]=0
        nphi = ptok.shape[1]
        npsi = ptok.shape[0]
        x = (np.arange(sz) - sz/2)*pxscale/1000.0*np.pi/180.0/3600.0
        xy = np.meshgrid(x,x)
        # Now allocate the giant memory array.
        if use_powerspect:
            nps_comb = ps_proj_matrix.shape[0]
            kp_implane = np.zeros((nt,npsi+nps_comb,sz,sz))
        else:
            kp_implane = np.zeros((nt,npsi,sz,sz))
        fty = ((ftpix[0] + subarr/2) % subarr) - subarr/2
        ftx = ftpix[1]
        for k in range(nt):
            #Create the xf and yf (x and y Fourier vectors) in units of cycles
            #per FOV. 
            yf = fty*np.cos(np.radians(pas[k])) + ftx*np.sin(np.radians(pas[k]))
            xf = ftx*np.cos(np.radians(pas[k])) - fty*np.sin(np.radians(pas[k]))
            #Now convert to physical units of radians^-1
            xf = xf/(subarr*rad_pixel)*2*np.pi
            yf = yf/(subarr*rad_pixel)*2*np.pi
            for i in range(nphi):
                #Sign of xf and yf below have calibrated by injecting fake binaries.
                sine = np.sin(-xf[i]*xy[0] - yf[i]*xy[1])
                for j in range(npsi):
                    kp_implane[k,j,:,:] += ptok[j,i]*sine
                if use_powerspect:
                    for j in range(nps_comb):
                        cosine = np.cos(-xf[i]*xy[0] - yf[i]*xy[1])
                        kp_implane[k,j+npsi,:,:] += ps_proj_matrix[j,i]*cosine
        #Reshaping makes the array compatible with the imaging code.
        if use_powerspect:
            kp_implane=kp_implane.reshape((nt*(npsi+nps_comb),sz,sz))
        else:
            kp_implane=kp_implane.reshape((nt*npsi,sz,sz))
        #If a savefile name is given, save the file!
        if (len(out_file) > 0):
            hl = pyfits.HDUList()
            header = pyfits.Header()
            header['PXSCALE'] = pxscale
            header['SUBARR'] = subarr
            header['RAD_PIX'] = rad_pixel
            hl.append(pyfits.ImageHDU(kp_implane,header))
            #Now add in the data. This was originally an image in IDL, so we keep
            #it as an image, not a table.
            if len(summary_files) > 0:
                header['PAMODE'] = 'Sky'
                hl.append(pyfits.ImageHDU(all_data))
            else:
                header['PAMODE'] = 'Det'
            hl.writeto(out_file,clobber=True)
        return kp_implane


    def implane_fit_binary(self,kp_implane_file, summary_file='', out_file='',pa_vertical=0.0,to_sky_pa=False):
        """A binary grid search to a kp_implane file.
    
        Parameters
        ----------
        kp_implane_file: string
            The fits file containing the kernel-phase to implane data
            (and, optionally, if summary_file isn't given, the data also)
        to_sky_pa: bool, optional
            Set to true if the position angles in the kp_implane_file are chip position
            angles, and need to be rotated to sky coordinates.
        pa_vertical: float 
            Position angle of vertical, over-written by a summary file and only
            needed of to_sky_pa=True
        summary_file: string
            The kernel-phase results file.    This could be an array, enabling easier
            application to situations where to_sky_pa is True    
        
        Returns
        -------
        p, crat, crat_sig, chi2
    
        """
        kp_implane = pyfits.getdata(kp_implane_file,0)
        h = pyfits.getheader(kp_implane_file)
        pxscale = h['PXSCALE']
        sz = kp_implane.shape[1]
        if len(summary_file) > 0:
            d = pyfits.getdata(summary_file)
            header = pyfits.getheader(summary_file)
            pa_vertical = header['PA']
            kp_mn = d['kp_mn']
            kp_sig = d['kp_sig']
        else:
            d = pyfits.getdata(kp_implane_file,1)
            kp_mn = d[0,:]
            kp_sig = d[1,:]
        crat = np.zeros((sz,sz))
        crat_sig = np.zeros((sz,sz))
        chi2 = np.zeros((sz,sz))
        for i in range(sz):
            for j in range(sz):
                #Model kernel-phases for pixel (i,j)
                mod_kp = kp_implane[:,i,j]
                #For a binary at pixel (i,j), we determine the best contrast ratio by a weighted
                #projection of kp_mn onto the mod_kp axis.
                crat[i,j] = np.sum(mod_kp*kp_mn/kp_sig**2)/np.sum(mod_kp**2/kp_sig**2)
                crat_sig[i,j] = np.sqrt(1.0/np.sum(mod_kp**2/kp_sig**2))
                if (np.isnan(crat[i,j])): 
                    crat[i,j] = 0.0
                    crat_sig[i,j]=1.0
                chi2[i,j] = np.sum((crat[i,j]*mod_kp - kp_mn)**2/kp_sig**2)
        #Don't count negative contrast ratios.
        modified_chi2 = chi2.copy()
        modified_chi2[crat < 0] = np.max(modified_chi2)
        min_ix = np.unravel_index(modified_chi2.argmin(), modified_chi2.shape)
        best_rchi2 = chi2[min_ix[0], min_ix[1]]/(len(kp_sig) - 3)
        print "Minimum reduced chi2: ", best_rchi2
        print "Significance (in sigma): ", crat[min_ix[0], min_ix[1]]/crat_sig[min_ix[0], min_ix[1]]
        print "Significance (scaling by sqrt(chi2)): ", crat[min_ix[0], min_ix[1]]/crat_sig[min_ix[0], min_ix[1]]/np.sqrt(best_rchi2)

        sep = pxscale*np.sqrt((min_ix[1]-sz/2)**2 + (min_ix[0]-sz/2)**2)
        pa = np.degrees(np.arctan2( -(min_ix[1]-sz/2), min_ix[0]-sz/2))
        contrast = crat[min_ix[0], min_ix[1]]
        #Adjust to on-sky units if needed.
        if (to_sky_pa): 
            pa += pa_vertical
    
        #Some plotting
        extent = [sz/2*pxscale,-sz/2*pxscale,-sz/2*pxscale,sz/2*pxscale]
        plt.clf()
        plt.imshow(modified_chi2[::-1,:], interpolation='nearest', extent=extent)
        plt.axis(extent)
        plt.title('Chi^2 map for positive contrasts')
        plt.xlabel('Delta RA (milli-arcsec)', fontsize='large')
        plt.ylabel('Delta Dec (milli-arcsec)', fontsize='large')
        plt.plot(0,0,'w*', ms=10)
        ax = plt.axes()
        ax.arrow(-0.45*sz*pxscale, -0.45*sz*pxscale, 0.1*sz*pxscale, 0, head_width=0.02*sz*pxscale, head_length=0.02*sz*pxscale, fc='k', ec='k')
        ax.arrow(-0.45*sz*pxscale, -0.45*sz*pxscale, 0, 0.1*sz*pxscale, head_width=0.02*sz*pxscale, head_length=0.02*sz*pxscale, fc='k', ec='k')
        plt.text(-0.45*sz*pxscale, -0.3*sz*pxscale,'N')
        plt.text(-0.3*sz*pxscale,-0.45*sz*pxscale, 'E')
        plt.draw()
        print "sep, pa, contrast, sig: ", sep, pa, contrast, crat_sig[min_ix[0], min_ix[1]]
        return (sep,pa,contrast), crat, crat_sig, chi2, best_rchi2

    def kp_binary_fitfunc_onefile(self,p, rad_pixel, subarr, ftpix, ptok, kp_mn, kp_sig):
        """This function finds the fit residuals based on input model parameters in
        chip-coorindates as (sep,PA,contrast). 
    
        Parameters
        ----------
        p: [sep in mas, chip position angle in degrees, contrast secondary/primary]
        rad_pixel: float
            Radians per pixel in the original image.
        subarr: float
            Subarray size in the original image and ftpix definition.
        ftpix: ( (N) array, (N) array )
            Sampling points in the Fourier domain
        ptok: 
            Pupil to kernel-phase matrix
        kp_mn: 
            Kernel-phases from the data
        kp_sig:
            Uncertainties in kernel-phases
        """
        xy = np.meshgrid(2*np.pi*np.arange(subarr/2 + 1)/float(subarr),
                        2*np.pi*(((np.arange(subarr) + subarr/2) % subarr) - subarr/2)/float(subarr))
        dy_in_pix = p[0]*np.pi/180./3600./1000.*np.cos(np.radians(p[1]))/rad_pixel
        dx_in_pix = -p[0]*np.pi/180./3600./1000.*np.sin(np.radians(p[1]))/rad_pixel
        # Avoiding the fft is marginally faster here, just like for bad pixel 
        # rejection... 
        ft = 1 + p[2]*np.exp(-1j*(dy_in_pix*xy[1] + dx_in_pix*xy[0]))
        modkp = np.dot(ptok,np.angle(ft[ftpix]))
        return (kp_mn - modkp)/kp_sig

    def kp_binary_fitfunc(self,p, rad_pixels, subarrs, ftpixs, ptoks, kp_mns, kp_sigs, pas,ptok_files):
        """This function finds the fit residuals based on input model parameters in
        on-sky coorindates as (sep,PA,contrast). It can have multiple input files.
    
        Parameters
        ----------
        p: [sep in mas, sky position angle in degrees, contrast secondary/primary]
        rad_pixels: float list
            Radians per pixel in the original image.
        subarrs: float list
            Subarray size in the original image and ftpix definition.
        ftpixs: ( (N) array, (N) array ) list
            Sampling points in the Fourier domain
        ptoks:  
            Pupil to kernel-phase matrices
        kp_mns: 
            Kernel-phases from the data
        kp_sigs:
            Uncertainties in kernel-phases
    
        Returns
        -------
        resid: (K) array
            normalised residuals to model kernel phases.
        """    
        resid=np.array([])
        for i in range(len(pas)):
            p_i = p.copy()
            p_i[1] -= pas[i]
            ftpix = ftpixs[ptok_files[i]] 
            ptok = ptoks[ptok_files[i]] 
            resid = np.append(resid,self.kp_binary_fitfunc_onefile(p_i, rad_pixels[i], subarrs[i], ftpix, ptok, kp_mns[i], kp_sigs[i]))
        return resid
    
    def kp_binary_fit(self,summary_files, initp, cdir=''):
        """Fit a binary model to kernel-phases
    
        Parameters
        ----------
        summary_files: string array
            List of input summary files.
        initp: (sep in mas, pa, contrast sec/primary)
            Initial fit parameters. These can be obtained from e.g. implane_fit_binary
        
        Returns
        -------
        (p_fit, p_sig, p_cov)
        p_fit: [float,float,float]
        p_sig: [float,float,float]
        p_cov: (3,3) array
            Best fit (sep, pa, contrast) parameters, standard deviation and 
            covariance. 
        """
        if (cdir == ''):
            cdir = self.aoinst.cdir
        #Read in the summary files, adding to appropriate variables as we go
        kp_mns = []
        kp_sigs = []
        pas = []
        ptok_files = []
        #Allow the different summary_files to use different ptok definition files,
        #but if they are all the same, only store one copy in memory. This is 
        #easily accomplised with a "dict"
        rad_pixels = []
        subarrs = []
        ftpixs = dict()
        ptoks = dict()    
        for i in range(len(summary_files)):
            d = pyfits.getdata(summary_files[i])
            kp_mns.append(d['kp_mn'])
            kp_sigs.append(d['kp_sig'])
            header = pyfits.getheader(summary_files[i])
            pas.append(header['PA'])
    #        pas[i]=0.0
            ptok_files.append(header['PTOKFILE'])
            header = pyfits.getheader(cdir + ptok_files[i])
            rad_pixels.append(header['RAD_PIX'])
            subarrs.append(header['SUBARR'])
            #!!! Could save time here by only reading in once.
            ftpix = pyfits.getdata(cdir + ptok_files[i],0)
            ftpixs[ptok_files[i]] = (ftpix[0], ftpix[1])
            ptoks[ptok_files[i]] = pyfits.getdata(cdir + ptok_files[i],2)
        initial_resid = self.kp_binary_fitfunc(initp, rad_pixels, subarrs, ftpixs, ptoks, kp_mns, kp_sigs, pas,ptok_files)
        p = op.leastsq(self.kp_binary_fitfunc, initp, args=(rad_pixels, subarrs, ftpixs, \
            ptoks, kp_mns, kp_sigs, pas,ptok_files), full_output=True, diag = [1,.5,0.01],factor=0.1)
        if p[0][2]<0:
            p[0][2] *= -1
            p[0][1] = (p[0][1] + 180) % 360
        #, factor = factorTry, diag = diagTry, maxfev = maxfev)
        ndf = len(p[2]['fvec'])
        print "Initial Chi^2: ", np.sum(initial_resid**2)/(ndf-3)
        rchi2 = np.sum(p[2]['fvec']**2)/(ndf-3)
        print "Best Reduced Chi^2: ", rchi2
        print "Parameters: {0:6.1f} {1:7.2f} {2:6.2f}".format(p[0][0], p[0][1], -2.5*np.log10(p[0][2]))
        errs = np.sqrt(np.diag(p[1])*rchi2)
        print "Errors:     {0:6.1f} {1:7.2f} {2:6.2f}".format(errs[0], errs[1], 2.5*np.log10(np.e)*errs[2]/p[0][2])
        return p[0], errs, p[1]*rchi2        

    def process_block(self, fstart='', fend='', min_files=3, dither=True, add_noise=50):
        """Process all files in a block (or a directory). The output is
        a bunch of kp files in the cdir.
        
        Parameters
        ----------
        min_files: int, optional
            The minimum number of files needed for this to run.
        
        Notes
        -----
        This may *not* be ideal software for e.g. NACO.
        
        TODO:
        1) Add manual_click
        """
        if len(self.aoinst.csv_dict) == 0:
            print("Error: Run read_summary_csv first. No darks made.")
            return
        if len(fstart)>0:
            wstart = np.where(self.aoinst.csv_dict['FILENAME'] == fstart)[0]
            if len(wstart)==0:
                print("Error: No file named " + fstart)
                return
            wstart = wstart[0]
        else:
            wstart = 0
        if len(fend)>0:
            wend = np.where(self.aoinst.csv_dict['FILENAME'] == fend)[0]
            if len(wend)==0:
                print("Error: No file named " + fend)
                return
            wend = wend[0]
        else:
            wend = len(csv_dict)
        
        i = wstart
        blocks = []
        while i<wend:
            #A new block - start the block with this current file.
            block = []
            block_string = self.aoinst.csv_block_string(i)
            #Find files that are similar...
            while (block_string == self.aoinst.csv_block_string(i) and i<wend):
                block.append(i)
                i += 1
            if len(block) >= min_files:
                blocks.append(block)
        cube_files = []
        kp_files = []
        kp_mn_files = []
        for j in range(len(blocks)):
            files = self.aoinst.csv_dict['FILENAME'][blocks[j]]
            fits_root = files[0][0:files[0].find('.fit')]
            cube_file = 'cube_' + fits_root + '.fits'
            kp_file = 'kp_' + fits_root + '.fits'
            kp_mn_file = 'kpmn_' + fits_root + '.fits'
            if dither:
                cube = self.aoinst.clean_dithered(files, out_file=cube_file, destripe=False)
            else:
                cube = self.aoinst.clean_no_dither(files, out_file=cube_file, destripe=False)
            self.extract_kerphase(cube_file=cube_file, add_noise=add_noise,out_file=kp_file, summary_file=kp_mn_file)
            cube_files.append(cube_file)
            kp_files.append(kp_file)
            kp_mn_files.append(kp_mn_file)    
        cube_files = np.array(cube_files)
        kp_files = np.array(kp_files)
        kp_mn_files = np.array(kp_mn_files)
            
        return cube_files, kp_files, kp_mn_files

    def poise_process(self, target_file='', target='', kp_files=[], use_powerspect=True,
            add_noise=16, beta=0.4):
        """Process a block of files using poise. By default,
        all files with the same settings as the target file are used.
        
        Parameters
        ----------
        target_file: string, optional
            A file containing the target. Either this or "target_file" should be used.
        target: string, optional
            The official name of the target. The first kp file with this name is used.
        kp_files: string list, optional
            A list of kernel-phase files.
        """
        if len(target_file)==0:
            print("Target name not implemented yet!")
            raise UserWarning
        if len(kp_files)==0:
            kpmn_files = glob.glob(self.aoinst.cdir + 'kpmn_*fits')
        h_target = pyfits.getheader(self.aoinst.cdir + target_file)
        targname = h_target['TARGNAME']
        cals = []
        src = []
        kp_files = []
        cube_files = []
        for i,filename in enumerate(kpmn_files):
            header = pyfits.getheader(filename)
            if header['TARGNAME'] != targname:
                cals.append(i)
            else:
                src.append(i)
            try: 
                kpfile = header['KPFILE']
            except:
                print("ERROR: Could not find KPFILE in header!")
                raise UserWarning
            kp_files.append(kpfile)
            cube_files.append(header['CUBEFILE'])
        cube_files = np.array(cube_files)
        kp_files = np.array(kp_files)
        if len(cals)==0:
            print("ERROR: No calibrators!")
            raise UserWarning
        if len(src)==0:
            print("ERROR: No target files!")
            raise UserWarning
        poise_filename = 'poise_' + target_file
        if len(cals)<5:
            good_cals = cals
        else:
            cal_kp_mns = np.array( [np.mean(pyfits.getdata(kp_files[cals[0]]),axis=0)] )
            for i in range(1,len(cals)):
                cal_kp_mns = np.append(cal_kp_mns,[np.mean(pyfits.getdata(kp_files[cals[i]]),axis=0)],0)
            cal_kp_mns -= np.tile(np.median(cal_kp_mns,0),cal_kp_mns.shape[0]).reshape(cal_kp_mns.shape)
            cal_kp_mads = np.median(np.abs(cal_kp_mns),1)
            #Reject calibrators with more than double the median deviation from the median.
            ww = np.where(cal_kp_mads < 2*np.median(cal_kp_mads))[0]
            good_cals = np.array(cals)[ww]
        self.poise_kerphase(kp_files[good_cals], out_file=poise_filename, use_powerspect=use_powerspect, beta=beta)
        summary_files = [targname + '_poise_' + cube for cube in cube_files[src]]
        for ix, cubename in enumerate(cube_files[src]):
            self.extract_kerphase(ptok_file=poise_filename,cube_file=cubename,\
                use_poise=True,summary_file=summary_files[ix],add_noise=add_noise, use_powerspect=use_powerspect)
         
        return summary_files

file = '/Users/mireland/data/nirc2/131020/n0641.fits.gz'
file = '/Users/mireland/data/nirc2/131020/n0204.fits.gz'
extn = '.fits.gz'
    
#Dark for the main file
if(0):
    dir =  '/Users/mireland/data/nirc2/131020/n'
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(617,627)]
    #files.extend([(dir + '{0:04d}' + extn).format(i) for i in range(56,61)])
    pp.aoinst.make_dark(files,'dark512.fits')

#Testing pupil mask generation etc...
if(0):
    dir =  '/Users/mireland/data/nirc2/131020/n'
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(201,211)]
    run = '2'
    (ftpix, fmask, ptok) = pp.pupil_sampling(files[0:3],dark_file='dark512.fits',flat_file='flat.fits',out_file='kp_Kp9h.fits')
    cube = self.aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
    kp = pp.extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(179,189)]
    run = '1'
    cube = self.aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
    kp = pp.extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(157,167)]
    run = '0'
    cube = self.aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
    kp = pp.extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(225,237)]
    run = '3'
    cube = self.aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube'+run+'.fits')
    kp = pp.extract_kerphase(ptok_file='kp_Kp9h.fits',ftpix=ftpix, ptok=ptok, cube_file='test_cube'+run+'.fits',add_noise=100,out_file='test_kp'+run+'.fits')

#Turn these observables into POISE observables, and find the image-plane representation.
if(0):
    ptok_poise,systematic,cal_kp = pp.poise_kerphase(['test_kp0.fits','test_kp1.fits','test_kp2.fits','test_kp3.fits'],out_file='ptok_poise_FPTauCals.fits')
    kp_implane = pp.kp_to_implane(ptok_file='ptok_poise_FPTauCals.fits', out_file='ptok_poise_implane.fits')


#The target cube files
if(0):
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(168,178)]
    cube = aoinst.clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube_FPTau.fits')
    pp.extract_kerphase(ptok_file='ptok_poise_FPTauCals.fits',cube_file='test_cube_FPTau.fits',use_poise=True,summary_file='poise_FPTau.fits',add_noise=16)
    kp_implane = pp.kp_to_implane(summary_files=['poise_FPTau.fits'], out_file='FPTau_implane.fits')

if (0):
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(144,154)]
#    cube = clean_no_dither(files, 'dark512.fits', 'flat.fits', fmask,out_file='test_cube_V892Tau.fits')
#    pp.extract_kerphase(ptok_file='ptok_poise_hd283477.fits',cube_file='test_cube_V892Tau.fits',use_poise=True,summary_file='poise_V892Tau.fits',add_noise=16)
    d = pyfits.getdata('poise_V892Tau.fits')    
#    kp_implane = pyfits.getdata('ptok_poise_implane.fits')
    kp_implane = pp.kp_to_implane(summary_files=['poise_V892Tau.fits'], out_file='V892_implane.fits')
    im = np.zeros((128,128))
    for i in range(2896): 
        im += d['kp_mn'][i]*kp_implane[i,:,:]
    plt.clf()
    plt.imshow(im[::-1,:], interpolation='nearest')
    plt.title('Simplified reconstructed image (vertical reversed)')
    
if (0):
    #Now lets create a fake binary...
    acube = pyfits.getdata('test_cube2.fits')
    h = pyfits.getheader('test_cube2.fits')
    for i in range(acube.shape[0]):
        acube[i,:,:] += np.roll(np.roll(acube[i,:,:],0,axis=1),7,axis=0)*0.1
    hl = pyfits.HDUList()
    hl.append(pyfits.ImageHDU(acube,h))
    hl.append(pyfits.new_table(pyfits.getdata('test_cube2.fits',1)))
    hl.writeto('fakebin_cube.fits',clobber=True)
    pp.extract_kerphase(ptok_file='ptok_poise_hd283477.fits',cube_file='fakebin_cube.fits',use_poise=True,summary_file='fakebin_summary.fits',add_noise=16)
    d = pyfits.getdata('fakebin_summary.fits')    
    kp_implane = pyfits.getdata('ptok_poise_implane.fits')
    im = np.zeros((128,128))
    for i in range(2850): 
        im += d['kp_mn'][i]*kp_implane[i,:,:]
    plt.clf()
    plt.imshow(im[::-1,:], interpolation='nearest')
    plt.title('Simplified reconstructed image (vertical reversed)')

if (0):
#    summary_file = 'fakebin_summary.fits'
#    implane_file = 'ptok_poise_implane.fits'
#    summary_file = 'poise_V892Tau.fits'
#    implane_file = 'V892_implane.fits'
    summary_file = 'poise_FPTau.fits'
    implane_file = 'FPTau_implane.fits'
    pgrid, crat, crat_sig, chi2 = pp.implane_fit_binary(implane_file, summary_file=summary_file, to_sky_pa=False)
    print "Grid Fit: ", pgrid
    pgrid = np.array(pgrid)
    if (pgrid[2] > 0.5):
        print "Contrast too high to use kerphase for fitting (i.e. near-equal binary)."
    else:
        p = pp.kp_binary_fit([summary_file],pgrid)
    