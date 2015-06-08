"""NIRC2 specific methods and variables for an AOInstrument.
"""
import astropy.io.fits as pyfits
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from aoinstrument import AOInstrument 

class NIRC2(AOInstrument):
 """The NIRC2 Class, that enables processing of NIRC2 images.
 """
 #A global definition, for error-checking downstream
 instrument = 'NIRC2'    

 def make_all_darks(self, ddir='', rdir=''):
    """Make all darks in a current directory. This skeleton routine assumes that
    keywords "SHRNAME", "NAXIS1" and "NAXIS2" exist.
    """
    #Allow over-riding default reduction and data directories.
    if (rdir == ''):
        rdir = self.rdir
    if (ddir == ''):
        ddir = self.ddir
    if len(self.csv_dict) == 0:
        print("Error: Run read_summary_csv first. No darks made.")
        return
    darks = np.where(np.array(self.csv_dict['SHRNAME']) == 'closed')[0]
    #Now we need to find unique values of the following:
    #NAXIS1, NAXIS2 (plus for nirc2... ITIME, COADDS, MULTISAM)
    codes = []
    for d in darks:
        codes.append(self.csv_dict['NAXIS1'][d] + self.csv_dict['NAXIS2'][d] + 
            self.csv_dict['ITIME'][d] + self.csv_dict['COADDS'][d] + self.csv_dict['MULTISAM'][d])
    codes = np.array(codes)
     #For each unique code, find all dark files and call make_dark.
    for c in np.unique(codes):
        w = np.where(codes == c)[0]
        if (len(w) >= 3):
            files = [ddir + self.csv_dict['FILENAME'][darks[ww]] for ww in w]
            self.make_dark(files, rdir=rdir)

 def csv_block_string(self, ix):
    """Find a string from the summary csv file that identifies a unique configuration
    for a set of files to be processed as a block."""
    if len(self.csv_dict) == 0:
        print("Error: Run read_summary_csv first. No string returned.")
        return
    block_string = self.csv_dict['NAXIS1'][ix] + self.csv_dict['NAXIS2'][ix] + \
        self.csv_dict['TARGNAME'][ix] + self.csv_dict['FILTER'][ix] + \
        self.csv_dict['ITIME'][ix] + self.csv_dict['COADDS'][ix]
    return block_string

 def info_from_header(self, h):
    """Find important information from the fits header and store in a common format
    
    Parameters
    ----------
    h:    The fits header
    
    Returns
    -------
    (dark_file, flat_file, filter, wave, rad_pixel)
    """

    #First, sanity check the header
    try: inst=h['CURRINST']
    except: inst=''
    if (len(inst)==0):
        print "Error: could not find instrument in header..."
        raise UserWarning
    if (self.instrument != inst):
        print "Error: software expecting: ", self.instrument, " but instrument is: ", inst
        raise UserWarning

    try: fwo = h['FWONAME']
    except:
        print "No FWONAME in NIRC2 header"
        raise UserWarning
    try: fwi = h['FWINAME']
    except:
        print "No FWINAME in NIRC2 header"
        raise UserWarning

    if (fwi=='Kp'):
        wave = 2.12e-6
        filter='Kp'
    elif (fwi=='CH4_short'):
        wave = 1.60e-6
        filter='CH4_short'
    elif (fwo=='Kcont'):
        wave = 2.27e-6
        filter='Kcont'
    elif (fwi=='Lp'):
        wave = 3.776e-6
        filter='Lp'
    elif (fwi=='H2O_ice'):
        wave = 3.1e-6
        filter='H2O_ice'
    elif (fwo=='PAH'):
        wave = 3.3e-6
        filter='PAH'
    elif (fwi=='Ms'):
        wave = 4.67e-6
        filter = 'Ms'
    else:
        print "Unknown Filter!"
        raise UserWarning
    flat_file = 'flat_' + filter + '.fits'

    try: camname = h['CAMNAME']
    except:
           print "No CAMNAME in header"
    if (camname == 'narrow'):
        #This comes from the Yelda (2010) paper.
         rad_pixel = 0.009952*(np.pi/180.0/3600.0)
    else:
        print "Unknown Camera!"
        raise UserWarning
    #Estimate the expected readout noise directly from the header.
    if h['SAMPMODE'] == 2:
        multisam = 1
    else:
        multisam = h['MULTISAM']
    #The next line comes from the NIRC2 manual home page.
    gain = 4.0
    rnoise = 50.0/gain/np.sqrt(multisam)*np.sqrt(h['COADDS'])
    #Find the appropriate dark file if needed.
    dark_file = self.get_dark_filename(h)
    targname = h['TARGNAME']
    #The pupil orientation...
    vertang_pa = h['ROTPPOSN']-h['EL']-h['INSTANGL'] 
    pa = vertang_pa + h['PARANG']
    #Find the pupil type and parameters for the pupil...
    pupil_params=dict()
    if (fwo == '9holeMsk'):
        pupil_type = 'circ_nrm'
        hole_xy = [[3.44,  4.57,  2.01,  0.20, -1.42, -3.19, -3.65, -3.15,  1.18],
                   [-2.22, -1.00,  2.52,  4.09,  4.46,  0.48, -1.87, -3.46, -3.01]]
        #For some reason... the y-axis is reversed, and we will transpose here
        #to save any transposes later.
        #NB as we aren't doing (u,v) coordinates but only chip coordinates here,
        #there is no difference between reversing the x- and y- axes.
        hole_xy = np.array(hole_xy)
        hole_xy[1,:] = -hole_xy[1,:]
        hole_xy = hole_xy[::-1,:]
        pupil_params['hole_xy'] = hole_xy
        pupil_params['hole_diam'] = 1.1
        pupil_params['mask_rotation'] = -0.01
        pupil_params['mask'] = 'g9'
        ftpix_file = 'ftpix_' + filter + '_'+ pupil_params['mask'] + '.fits'
    elif (fwi == '18holeMsk'):
        pupil_type = 'circ_nrm'
        pupil_params['mask'] = 'g18'
        print "Still to figure out 18 hole mask..."
        raise UserWarning
    else:
        pupil_type = 'keck'
        pupil_params['segment_size'] = 1.558
        pupil_params['obstruction_size'] = 2.0 #Guessed
        ftpix_file = 'ftpix_' + filter + '_fullpupil.fits'
#    else:
#        print "Assuming full pupil..."
#        pupil_type = 'annulus'
#        pupil_params['inner_diam'] = 1.8
#        pupil_params['outer_diam'] = 10.2 #Maximum diameter is really 10.5
#        ftpix_file = 'ftpix_' + filter + '_fullpupil.fits'
    return {'dark_file':dark_file, 'flat_file':flat_file, 'filter':filter, 
        'wave':wave, 'rad_pixel':rad_pixel,'targname':targname, 
        'pupil_type':pupil_type,'pupil_params':pupil_params,'ftpix_file':ftpix_file, 
        'gain':gain, 'rnoise':rnoise, 'vertang_pa':vertang_pa, 'pa':pa}

 def get_dark_filename(self,h):
    """Create a dark fits filename based on a header
    
    Parameters
    ----------
    h: header from astropy.io.fits
    
    Returns
    -------
    dark_file: string
    """
    if h['SAMPMODE'] == 2:
        multisam = 1
    else:
        multisam = h['MULTISAM']
    dark_file = 'dark_' + str(h['NAXIS1']) +'_'+str(h['COADDS']) +'_' +str(multisam)+'_'+ str(int(h['ITIME']*100)) + '.fits'
    return dark_file

 def destripe_nirc2(self,im, subtract_edge=True, subtract_median=False, do_destripe=True):
    """Destripe an image from the NIRC2 camera.
    
    The algorithm is:
    1) Subtract the mode from each quadrant.
    2) For each pixel, find the 24 pixels in other quadrants corresponding to 
    reflections about the chip centre.
    3) Subtract the median of these pixels.
    
    Parameters
    ----------
    im: array_like
        The input image.
    subtract_median: bool, optional
        Whether or not to subtract the median from each quadrant.    
    subtract_edge: bool, optional
        Whether or not to adjust the means of each quadrant by the edge pixels.
    
    Returns
    -------
    im: array_like
        The corrected image.
    """
    s = im.shape
    quads = [im[0:s[0]/2,0:s[1]/2],im[s[0]:s[0]/2-1:-1,0:s[1]/2],
             im[0:s[0]/2,s[1]:s[1]/2-1:-1],im[s[0]:s[0]/2-1:-1,s[1]:s[1]/2-1:-1]]
    quads = np.array(quads, dtype='float')
    #Work through the quadrants, modifying based on the edges.
    if subtract_edge:
        quads[1] += np.median(quads[3][:,s[1]/2-8:s[1]/2])- np.median(quads[1][:,s[1]/2-8:s[1]/2]) 
        quads[2] += np.median(quads[3][s[0]/2-8:s[0]/2,:])- np.median(quads[2][s[0]/2-8:s[0]/2,:])  
        delta = 0.5*(np.median(quads[3][s[0]/2-8:s[0]/2,:]) + np.median(quads[3][:,s[1]/2-8:s[1]/2])
               - np.median(quads[0][s[0]/2-8:s[0]/2,:]) - np.median(quads[0][:,s[1]/2-8:s[1]/2]))
        quads[0] += delta
    #Subtract the background
    if subtract_median:
        print "Subtracting Median..."
        med = np.median(quads)
        dispersion = np.median(np.abs(quads - med))
        MED_DIFF_MULTIPLIER = 4.0
        goodpix = np.where(np.abs(quads - med) < MED_DIFF_MULTIPLIER*dispersion)
        med = np.median(quads[goodpix])
        quads -= med
    if do_destripe:
        quads = quads.reshape((4,s[0]/2,s[1]/16,8))
        stripes = quads.copy()
        for i in range(4):
            for j in range(s[0]/2): #The -1 on  line is because of artifacts
                for k in range(s[0]/16):
                    pix = np.array([stripes[(i+1)%4,j,k,:],stripes[(i+2)%4,j,k,:],stripes[(i+3)%4,j,k,:]])
                    quads[i,j,k,:] -= np.median(pix)
        quads = quads.reshape((4,s[0]/2,s[1]/2))
    im[0:s[0]/2,0:s[1]/2] = quads[0]
    im[s[0]:s[0]/2-1:-1,0:s[1]/2] = quads[1]
    im[0:s[0]/2,s[1]:s[1]/2-1:-1] = quads[2]
    im[s[0]:s[0]/2-1:-1,s[1]:s[1]/2-1:-1] = quads[3]
    return im
    
 def make_dark(self,in_files, out_file='', subtract_median=True, destripe=True, med_threshold=15.0, rdir=''):
    """Create a dark frame and save to a fits file, 
    with an attached bad pixel map as the first fits extension.

    Parameters
    ----------
    in_files : array_like (dtype=string). A list if input filenames.
    out_file: string
        The file to write to.
    subtract_median: bool, optional
        Whether or not to subtract the median from each frame (or quadrants)
    destripe: bool, optional
        Whether or not to destripe the images.
    med_threshold: float, optional
        The threshold for pixels to be considered bad if their absolute
        value differs by more than this multiple of the median difference
        of pixel values from the median.
    
    Returns
    -------
    (optional) out_file: If an empty string is given, it is filled with the default out
    filename
    """
    #Allow over-riding default reduction directory.
    if (rdir == ''):
        rdir = self.rdir
    VAR_THRESHOLD = 10.0
    nf = len(in_files)
    if (nf < 3):
        print "At least 3 dark files sre needed for reliable statistics"
        raise UserWarning
    # Read in the first dark to check the dimensions.
    in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
    h = in_fits[0].header
    instname = ''
    try: instname=h['CURRINST']
    except:
        print "Unknown Header Type"
    #Create the output filename if needed
    if (out_file == ''):
        out_file = self.get_dark_filename(h)
    s = in_fits[0].data.shape
    in_fits.close()
    darks = np.zeros((nf,s[0],s[1]))
    plt.clf()
    for i in range(nf):
        #Read in the data, linearizing as a matter of principle, and also because
        #this routine is used for 
        adark = self.linearize_nirc2(in_files[i])
        if (instname == 'NIRC2'):
            adark = self.destripe_nirc2(adark, subtract_median=subtract_median, do_destripe=destripe)
        if (subtract_median):
            plt.imshow(np.minimum(adark,1e2))
        else:
            plt.imshow(adark)
            print "Median: " + str(np.median(adark))
        plt.draw()
        darks[i,:,:] = adark
    #Now look for weird pixels. 
    med_dark = np.median(darks, axis=0)
    max_dark = np.max(darks, axis=0)
    var_dark = np.zeros((s[0],s[1]))
    for i in range(nf):
        var_dark += (darks[i,:,:] - med_dark)**2
    var_dark -= (max_dark - med_dark)**2
    var_dark /= nf-2
    med_diff = np.median(np.abs(med_dark - np.median(med_dark)))
    print "Median difference: " + str(med_diff)
    med_var_diff = np.median(np.abs(var_dark - np.median(var_dark)))
    bad_med = np.abs(med_dark - np.median(med_dark)) > med_threshold*med_diff
    bad_var = np.abs(var_dark) > np.median(var_dark) + VAR_THRESHOLD*med_var_diff
    print "Pixels with bad mean: " + str(np.sum(bad_med))
    print "Pixels with bad var: " + str(np.sum(bad_var))
    bad = np.logical_or(bad_med, bad_var)
    med_dark[bad] = 0.0
    #Copy the original header to the dark file.
    hl = pyfits.HDUList()
    hl.append(pyfits.ImageHDU(med_dark,h))
    hl.append(pyfits.ImageHDU(np.uint8(bad)))
    hl.writeto(rdir+out_file,clobber=True)
    plt.figure(1)
    plt.imshow(med_dark,cmap=cm.gray, interpolation='nearest')
    plt.title('Median Frame')
    plt.figure(2)
    plt.imshow(bad,cmap=cm.gray, interpolation='nearest')
    plt.title('Bad Pixels')
    plt.draw()

 def linearize_nirc2(self,in_file, out_file=''):
    """Procedure to linearize NIRC2 (treated as a single detector).
    Run on all images nominally before running anything else.
    Originally from IDL code by Stan Metchev with Adam Kraus modifications.

    Parameters
    ----------
    in_file: string
        The input fits file
    out_file: string, optional
        An output fits file
        
    Returns
    -------
    im: (N,N) array
        The linearized image.
    
    Notes
    -----
    This procedure takes COADDS into account, but does not take subreads into account.
    i.e. in MCDS sampling, there is slightly more nonlinearity than accounted for, 
    because the detector is more saturated by the time the last read is made.
    """

    coeff = np.array([1.001,-6.9e-6,-0.70e-10])
    #Get key parameters from the header and get the data
    in_fits = pyfits.open(in_file, ignore_missing_end=True)
    z = in_fits[0].header
    fitsarr = in_fits[0].data
    in_fits.close()
    #See if we've already updated this header
    try:
        lindate = z['LINHIST']
    except:
        print 'Linearizing: ',in_file
        xsub=z['NAXIS1']
        ysub=z['NAXIS2']
        norm=np.array((xsub,ysub))
        coadds = z['COADDS']
        norm = coeff[0]+coeff[1]*fitsarr/coadds+coeff[2]*(fitsarr/coadds)**2
        fitsarr = fitsarr / norm
        z['LINHIST'] = time.asctime()
        if (len(out_file) > 0):
            hl = pyfits.HDUList()
            hl.append(pyfits.ImageHDU(fitsarr,z))
            hl.writeto(out_file,clobber=True)
    return fitsarr
    
 def _calibration_subarr(self, rdir, flat_file, dark_file, szx, szy):
    """A function designed to be used internally only, which chops out the central part
    of calibration data for sub-arrays"""
    if len(flat_file) > 0:
        flat = pyfits.getdata(rdir + flat_file,0)
        flat = flat[flat.shape[0]/2 - szy/2:flat.shape[0]/2 + szy/2,flat.shape[1]/2 - szx/2:flat.shape[1]/2 + szx/2]
        bad = pyfits.getdata(rdir + flat_file,1)
        bad = bad[bad.shape[0]/2 - szy/2:bad.shape[0]/2 + szy/2,bad.shape[1]/2 - szx/2:bad.shape[1]/2 + szx/2]
    else:
        flat = np.ones((szy,szx))
        bad = np.zeros((szy,szx))
    if len(dark_file) > 0:
        dark = pyfits.getdata(rdir + dark_file,0)
        if (szy != dark.shape[0]):
            print("Warning - Dark is of the wrong shape!")
            dark = dark[dark.shape[0]/2 - szy/2:dark.shape[0]/2 + szy/2, \
                       dark.shape[1]/2 - szx/2:dark.shape[1]/2 + szx/2]
    else:
        dark = np.zeros((szy,szx))
    return (flat,dark,bad)
        
 def clean_no_dither(self, in_files, fmask_file='',dark_file='', flat_file='', fmask=[],\
     subarr=128,extra_threshold=7,out_file='',median_cut=0.7, destripe=True, ddir='', rdir='', cdir=''):
    """Clean a series of fits files, including: applying the dark, flat, 
    removing bad pixels and cosmic rays. This can also be used for dithered data, 
    but it will not subtract the dithered positions. There reason for two separate
    programs includes that for dithered data, bad pixel rejection etc has to be done on
    *all* riles.

    Parameters
    ----------
    in_files : array_like (dtype=string). 
        A list if input filenames.
    dark_file: string
        The dark file, previously created with make_dark
    flat_file: string
        The flat file, previously created with make_flat
    ftpix: ( (N) array, (N) array)
        The pixels in the data's Fourier Transform that include all non-zero
        values (created using pupil_sampling)
    subarr: int, optional
        The width of the subarray.
    extra_threshold: float, optional
        A threshold for identifying additional bad pixels and cosmic rays.
    outfile: string,optional
        A filename to save the cube as, including the header of the first
        fits file in the cube plus extra information.
    
    
    Returns
    -------
    The cube of cleaned frames.
    """
    #Allow over-riding default data, cube and analysis directories.
    if (ddir == ''):
        ddir = self.ddir
    if (rdir == ''):
        rdir = self.rdir
    if (cdir == ''):
        cdir = self.cdir
    #Allocate data for the cube
    nf = len(in_files)
    cube = np.zeros((nf,subarr,subarr))
    #Decide on the image size from the first file. !!! are x and y around the right way?
    in_fits = pyfits.open(ddir + in_files[0], ignore_missing_end=True)
    h = in_fits[0].header
    in_fits.close()
    szx = h['NAXIS1']
    szy = h['NAXIS2']
    #Extract important information from the header...
    hinfo = self.info_from_header(h)
    rnoise = hinfo['rnoise']
    gain = hinfo['gain']
    #If we set the fmask manually, then don't use the file.
    if len(fmask) == 0:
        #If no file is given, find it automatically.
        if len(fmask_file) == 0:
            fmask_file = hinfo['ftpix_file']
        try:
            fmask = pyfits.getdata(rdir + fmask_file,1)
        except:
            print("Error - couldn't find kp/Fourier mask file: " +fmask_file+ " in directory: " + rdir)
            raise UserWarning
        h_fmask = pyfits.getheader(rdir + fmask_file)
        if (len(dark_file) == 0):
            dark_file = h_fmask['DARK']
        if (len(flat_file) == 0):
            flat_file = h_fmask['FLAT']

    #Chop out the appropriate part of the flat, dark, bad arrays
    (flat,dark,bad) = self._calibration_subarr(rdir, flat_file, dark_file, szx, szy)

    #Go through the files, cleaning them one at a time
    xpeaks = np.zeros(nf)
    ypeaks = np.zeros(nf)
    maxs = np.zeros(nf)
    pas = np.zeros(nf)
    backgrounds = np.zeros(nf)
    for i in range(nf):
        #First, find the position angles from the header keywords. NB this is the Sky-PA of chip vertical.
        in_fits = pyfits.open(ddir + in_files[i], ignore_missing_end=True)
        h = in_fits[0].header
        in_fits.close()
        pas[i]=360.+h['PARANG']+h['ROTPPOSN']-h['EL']-h['INSTANGL'] 
        #Read in the image - making a nonlinearity correction
        im = self.linearize_nirc2(ddir + in_files[i])
        #Destripe, then clean the data using the dark and the flat. This might change
        #the background, so allow for this.
        backgrounds[i] = np.median(im)
        im = self.destripe_nirc2(im, do_destripe=destripe)
        backgrounds[i] -= np.median(im)
        #!!! It is debatable whether the dark on the next line is really useful... but setting 
        #dark_file='' removes its effect.
        im = (im - dark)/flat
        #Find the star... 
        im *= 1.0 - bad
        im_filt = nd.filters.median_filter(im,size=5)
        max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
        print("Maximum x,y: " + str(max_ix[1])+', '+ str(max_ix[0]))
        xpeaks[i] = max_ix[1]
        ypeaks[i] = max_ix[0]
        maxs[i] = im[max_ix[0],max_ix[1]]
        subim = np.roll(np.roll(im,subarr/2-max_ix[0],axis=0),subarr/2-max_ix[1],axis=1)
        subim = subim[0:subarr,0:subarr]
        subbad = np.roll(np.roll(bad,subarr/2-max_ix[0],axis=0),subarr/2-max_ix[1],axis=1)
        subbad = subbad[0:subarr,0:subarr]
        new_bad = subbad.copy()    
        subim[np.where(subbad)] = 0
        plt.clf()
        plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.title(h['TARGNAME'])
        plt.draw()
        #Iteratively fix the bad pixels and look for more bad pixels...
        for ntry in range(1,15):
            #Correct the known bad pixels
            self.fix_bad_pixels(subim,new_bad,fmask)
            #Search for more bad pixels. Lets use a Fourier technique here...
            extra_bad_ft = np.fft.rfft2(subim)*fmask
            extra_bad = np.real(np.fft.irfft2(extra_bad_ft))
            mim = nd.filters.median_filter(subim,size=5)
            #NB The next line *should* take experimentally determined readout noise into account !!!
            extra_bad = np.abs(extra_bad/np.sqrt(np.maximum(gain*mim + rnoise**2,rnoise**2)))
            unsharp_masked = extra_bad-nd.filters.median_filter(extra_bad,size=3)
            current_threshold = np.max([0.3*np.max(unsharp_masked[new_bad == 0]), extra_threshold*np.median(extra_bad)])
            extra_bad = unsharp_masked > current_threshold
            n_extra_bad = np.sum(extra_bad)
            print(str(n_extra_bad)+" extra bad pixels or cosmic rays identified. Attempt: "+str(ntry))
            subbad += extra_bad
            if (ntry == 1):
                new_bad = extra_bad
            else:
                new_bad += extra_bad
                new_bad = extra_bad>0
            if (n_extra_bad == 0):
                break
        
        #Now re-correct both the known and new bad pixels at once.
        self.fix_bad_pixels(subim,subbad,fmask)
        plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.draw()
        
        #Save the data and move on!
        cube[i,:,:]=subim
    #Fine bad frames based on low peak count.
    good = np.where(maxs > median_cut*np.median(maxs))
    good = good[0]
    if (len(good) < nf):
        print nf-len(good), "  frames rejected due to low peak counts."
    cube = cube[good,:,:]
    nf = np.shape(cube)[0]
    #If a filename is given, save the file.
    if (len(out_file) > 0):
        hl = pyfits.HDUList()
        h['RNOISE'] = rnoise
        h['PGAIN'] = gain #P means python
#        h['PFILTER'] = !!! Not complete !!!
        h['SZX'] = szx
        h['SZY'] = szy
        h['DDIR'] = ddir
        #NB 'TARGNAME' is the standard target name.
        for i in range(nf):
            h['HISTORY'] = 'Input: ' + in_files[i]
        hl.append(pyfits.ImageHDU(cube,h))
        #Add in the original peak pixel values, forming the image centers in the cube.
        #(this is needed for e.g. undistortion)
        col1 = pyfits.Column(name='xpeak', format='E', array=xpeaks)
        col2 = pyfits.Column(name='ypeak', format='E', array=ypeaks)
        col3 = pyfits.Column(name='pa', format='E', array=pas)
        col4 = pyfits.Column(name='max', format='E', array=maxs)
        col5 = pyfits.Column(name='background', format='E', array=backgrounds)
        cols = pyfits.ColDefs([col1, col2,col3,col4,col5])
        hl.append(pyfits.new_table(cols))
        hl.writeto(cdir+out_file,clobber=True)
    return cube
    
 def clean_dithered(self, in_files, fmask_file='',dark_file='', flat_file='', fmask=[],\
    subarr=128,extra_threshold=7,out_file='',median_cut=0.7, destripe=True, \
    manual_click=False, ddir='', rdir='', cdir=''):
    """Clean a series of fits files, including: applying the dark and flat, removing bad pixels and
    cosmic rays, creating a `rough supersky' in order to find a mean image, identifying the target and any 
    secondary targets, identifying appropriate sky frames for each frame and subtracting these off. In
    order to find objects in the image in vertical angle mode, the assumption is made that rotation 
    is much less than interferogram size.
    
    To enhance code readability, many of the options in previous routines have been removed.

    Parameters
    ----------
    in_files : array_like (dtype=string). 
        A list if input filenames.
    dark_file: string
        The dark file, previously created with make_dark
    flat_file: string
        The flat file, previously created with make_flat
    ftpix: ( (N) array, (N) array)
        The pixels in the data's Fourier Transform that include all non-zero
        values (created using pupil_sampling)
    subarr: int, optional
        The width of the subarray.
    extra_threshold: float, optional
        A threshold for identifying additional bad pixels and cosmic rays.
    outfile: string,optional
        A filename to save the cube as, including the header of the first
        fits file in the cube plus extra information.
    
    
    Returns
    -------
    The cube of cleaned frames.
    """
    #Allow over-riding default data, cube and analysis directories.
    if (ddir == ''):
        ddir = self.ddir
    if (rdir == ''):
        rdir = self.rdir
    if (cdir == ''):
        cdir = self.cdir
    #Allocate memory for the cube
    nf = len(in_files)
    cube = np.zeros((nf,subarr,subarr))
    #Decide on the image size from the first file. !!! are x and y around the right way?
    in_fits = pyfits.open(ddir + in_files[0], ignore_missing_end=True)
    h = in_fits[0].header
    in_fits.close()
    szx = h['NAXIS1']
    szy = h['NAXIS2']
    #Allocate memory for the full cube
    full_cube = np.zeros((nf,szy, szx))
    #Extract important information from the header...
    hinfo = self.info_from_header(h)
    rnoise = hinfo['rnoise']
    gain = hinfo['gain']
    rad_pixel = hinfo['rad_pixel']
    #If we set the fmask manually, then don't use the file.
    if len(fmask) == 0:
        #If no file is given, find it automatically.
        if len(fmask_file) == 0:
            fmask_file = hinfo['ftpix_file']
        try:
            fmask = pyfits.getdata(rdir + fmask_file,1)
        except:
            print("Error - couldn't find kp/Fourier mask file: " +fmask_file+ " in directory: " + rdir)
            raise UserWarning
        h_fmask = pyfits.getheader(rdir + fmask_file)
        if (len(dark_file) == 0):
            dark_file = h_fmask['DARK']
        if (len(flat_file) == 0):
            flat_file = h_fmask['FLAT']

    #Chop out the appropriate part of the flat, dark, bad arrays
    (flat,dark,bad) = self._calibration_subarr(rdir, flat_file, dark_file, szx, szy)
    wbad = np.where(bad)

    #Go through the files, cleaning them one at a time and adding to the cube. 
    pas = np.zeros(nf)
    raoffs = np.zeros(nf)
    decoffs = np.zeros(nf)
    decs = np.zeros(nf)
    maxs = np.zeros(nf)
    xpeaks = np.zeros(nf,dtype=int)
    ypeaks = np.zeros(nf,dtype=int)
    backgrounds = np.zeros(nf)
    for i in range(nf):
        #First, find the position angles from the header keywords. NB this is the Sky-PA of chip vertical.
        in_fits = pyfits.open(ddir + in_files[i], ignore_missing_end=True)
        h = in_fits[0].header
        in_fits.close()
        pas[i]=360.+h['PARANG']+h['ROTPPOSN']-h['EL']-h['INSTANGL'] 
        raoffs[i]=h['RAOFF']
        decoffs[i]=h['DECOFF']
        decs[i] =h['DEC']
        #Read in the image - making a nonlinearity correction
        im = self.linearize_nirc2(ddir + in_files[i])
        #Destripe, then clean the data using the dark and the flat. This might change
        #the background, so allow for this.
        backgrounds[i] = np.median(im)
        im = self.destripe_nirc2(im, do_destripe=destripe)
        backgrounds[i] -= np.median(im)
        #!!! It is debatable whether the dark on the next line is really useful... but setting 
        #dark_file='' removes its effect.
        im = (im - dark)/flat
        #For display purposes, we do a dodgy bad pixel correction.
        mim = nd.filters.median_filter(im,size=3)
        im[bad] = mim[bad]
        full_cube[i,:,:] = im
        
    #Find the rough "supersky", by taking the 25th percentile of each pixel.
    rough_supersky = np.percentile(full_cube, 25.0, axis=0)
    #Subtract this supersky off each frame. Don't worry - all strictly pixel-dependent
    #offsets are removed in any case so this doesn't bias the data.
    for i in range(nf):
        full_cube[i,:,:] -= rough_supersky
        
    #Now the NIRC2-specific stuff... 
    raoffs = raoffs - np.mean(raoffs)
    decoffs = decoffs - np.mean(decoffs)
    shifts = np.zeros((nf,2),dtype=int)
    im_mean = np.zeros((szy, szx))
    for i in range(nf):
        th = np.radians(pas[i])
        rot_mat = np.array([[np.cos(th), -np.sin(th)],[-np.sin(th), -np.cos(th)]]) 
        shifts[i,:] = np.dot(rot_mat, np.radians(np.array([raoffs[i]*np.cos(np.radians(decs[i])), decoffs[i]]))/rad_pixel ).astype(int)
        im_mean = im_mean + np.roll(np.roll(full_cube[i,:,:],-shifts[i,0], axis=1), -shifts[i,1], axis=0)
        
    #Find the star...     
    #Show the image, y-axis reversed.
    plt.clf()
    plt.imshow(np.arcsinh(im_mean/100), interpolation='nearest', origin='lower')
    arrow_xy = np.dot(rot_mat, [0,-30])
    plt.arrow(60,60,arrow_xy[0],arrow_xy[1],width=0.2)
    plt.text(60+1.3*arrow_xy[0], 60+1.3*arrow_xy[1], 'N')
    arrow_xy = np.dot(rot_mat, [-30,0])
    plt.arrow(60,60,arrow_xy[0],arrow_xy[1],width=0.2)
    plt.text(60+1.3*arrow_xy[0], 60+1.3*arrow_xy[1], 'E')
    if manual_click:
        plt.title('Click on target...')
        max_ix = plt.ginput(1, timeout=0)[0]
        #To match the (y,x) order below, change this...
        max_ix = int(max_ix[1]), int(max_ix[0])
    else:
        im_filt = nd.filters.median_filter(im_mean,size=5)
        max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
        plt.title('Identified target shown')
    plt.plot(max_ix[1], max_ix[0], 'wx', markersize=20,markeredgewidth=2)
    plt.draw()
    print("Maximum x,y: " + str(max_ix[1])+', '+ str(max_ix[0]))

    #Set the xpeaks and ypeaks values (needed for sub-arraying later)
    for i in range(nf):
        xpeaks[i] = max_ix[1] + shifts[i,0]
        ypeaks[i] = max_ix[0] + shifts[i,1]
    
    subims = np.empty( (nf,subarr,subarr) )
    #Sky subtract and fix bad pixels.
    for i in range(nf):
        #Undo the flat, to minimise the effects of errors in the flat.
        for j in range(nf):
            im = full_cube[j,:,:]*flat
            #Roll all the sub-images, and cut them out.
            subims[j,:,:] = np.roll(np.roll(im,subarr/2-ypeaks[i],axis=0),
                subarr/2-xpeaks[i],axis=1)[0:subarr,0:subarr]
        #Find a flat to re-apply
        subflat = np.roll(np.roll(flat,subarr/2-ypeaks[i],axis=0),subarr/2-xpeaks[i],axis=1)
        subflat = subflat[0:subarr,0:subarr]
        #Find the frames that are appropriate for a dither...
        w = np.where( (xpeaks - xpeaks[i])**2 + (ypeaks - ypeaks[i])**2 > 0.5*subarr**2 )[0]
        if len(w) == 0:
            print "Error: Can not find sky from dithered data - use clean_no_dither for {0:s}".format(in_files[0])
            return
        #To avoid too many extra bad pixels, we'll use a median here.
        sky = np.median(subims[w,:,:], axis=0)
        backgrounds[i] += np.median(sky)
        #Subtract the sky then re-apply the flat.
        subim = (subims[i,:,:] - sky)/subflat
        #Find the peak from the sub-image.
        im_filt = nd.filters.median_filter(subim,size=5)
        max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
        maxs[i] = subim[max_ix[0],max_ix[1]]
        subbad = np.roll(np.roll(bad,subarr/2-ypeaks[i],axis=0),subarr/2-xpeaks[i],axis=1)
        subbad = subbad[0:subarr,0:subarr]
        new_bad = subbad.copy()    
        subim[np.where(subbad)] = 0
        plt.clf()
        plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.title(h['TARGNAME'])
        plt.draw()
        #Iteratively fix the bad pixels and look for more bad pixels...
        for ntry in range(1,15):
            #Correct the known bad pixels
            self.fix_bad_pixels(subim,new_bad,fmask)
            #Search for more bad pixels. Lets use a Fourier technique here...
            extra_bad_ft = np.fft.rfft2(subim)*fmask
            extra_bad = np.real(np.fft.irfft2(extra_bad_ft))
            mim = nd.filters.median_filter(subim,size=5)
            #NB The next line *should* take experimentally determined readout noise into account !!!
            extra_bad = np.abs(extra_bad/np.sqrt(np.maximum(gain*mim + rnoise**2,rnoise**2)))
            unsharp_masked = extra_bad-nd.filters.median_filter(extra_bad,size=3)
            current_threshold = np.max([0.3*np.max(unsharp_masked[new_bad == 0]), extra_threshold*np.median(extra_bad)])
            extra_bad = unsharp_masked > current_threshold
            n_extra_bad = np.sum(extra_bad)
            print(str(n_extra_bad)+" extra bad pixels or cosmic rays identified. Attempt: "+str(ntry))
            subbad += extra_bad
            if (ntry == 1):
                new_bad = extra_bad
            else:
                new_bad += extra_bad
                new_bad = extra_bad>0
            if (n_extra_bad == 0):
                break
        
        #Now re-correct both the known and new bad pixels at once.
        self.fix_bad_pixels(subim,subbad,fmask)
        plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.draw()
        
        #Save the data and move on!
        cube[i,:,:]=subim
        
    #Find bad frames based on low peak count.
    good = np.where(maxs > median_cut*np.median(maxs))
    good = good[0]
    if (len(good) < nf):
        print nf-len(good), "  frames rejected due to low peak counts."
    cube = cube[good,:,:]
    nf = np.shape(cube)[0]
    #If a filename is given, save the file.
    if (len(out_file) > 0):
        hl = pyfits.HDUList()
        h['RNOISE'] = rnoise
        h['PGAIN'] = gain #P means python
#        h['PFILTER'] = !!! Not complete !!!
        h['SZX'] = szx
        h['SZY'] = szy
        h['DDIR'] = ddir
        #NB 'TARGNAME' is the standard target name.
        for i in range(nf):
            h['HISTORY'] = 'Input: ' + in_files[i]
        hl.append(pyfits.ImageHDU(cube,h))
        #Add in the original peak pixel values, forming the image centers in the cube.
        #(this is needed for e.g. undistortion)
        col1 = pyfits.Column(name='xpeak', format='E', array=xpeaks)
        col2 = pyfits.Column(name='ypeak', format='E', array=ypeaks)
        col3 = pyfits.Column(name='pa', format='E', array=pas)
        col4 = pyfits.Column(name='max', format='E', array=maxs)
        col5 = pyfits.Column(name='background', format='E', array=backgrounds)
        cols = pyfits.ColDefs([col1, col2,col3,col4,col5])
        hl.append(pyfits.new_table(cols))
        hl.writeto(cdir+out_file,clobber=True)
    return cube
    

if(0):
    n2 = NIRC2()

#Testing destripe only.
if (0):
    f = pyfits.open(file, ignore_missing_end=True)
    im = f[0].data
    f.close()
    plt.imshow(np.minimum(np.maximum(n2.destripe_nirc2(im),-50),50), interpolation='nearest', cmap=cm.gray)
    f.close()

#Testing darks and flats.
if (0):
    dir =  '/Users/mireland/data/nirc2/131115/n'
    extn = '.fits.gz'
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(40,45)]
    #files.extend([(dir + '{0:04d}' + extn).format(i) for i in range(56,61)])
    n2.make_dark(files,'dark.fits')
    files = [(dir + '{0:04d}' + extn).format(i) for i in range(29,40)]
    n2.make_flat(files,'dark.fits','flat.fits')