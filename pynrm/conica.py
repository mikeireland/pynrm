"""NAOS-Conica specific methods and variables.
"""
from __future__ import division, print_function
import astropy.io.fits as pyfits
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import glob
import pdb
import time

from aoinstrument import AOInstrument 

class CONICA(AOInstrument):
 """The CONICA Class, that enables processing of CONICA images.
 """
 instrument = 'NAOS+CONICA' 
 def is_bad_surrounded(self,bad):
    #Returns matrix of booleans showing which pixels are surrounded by bad pixels
    #"Surrounded" means there is at least one bad pixel in at least two directions
    numPixels = 3
    sz = len(bad)
    is_bad_to_left = np.zeros((sz,sz-numPixels))
    is_bad_to_right = np.zeros((sz,sz-numPixels))
    is_bad_above = np.zeros((sz-numPixels,sz))
    is_bad_below = np.zeros((sz-numPixels,sz))
    for ii in range(0,numPixels):
        is_bad_to_left+=bad[0:sz,numPixels-ii-1:sz-ii-1]
        is_bad_to_right+=bad[0:sz,ii+1:sz-numPixels+ii+1]
        is_bad_above+=bad[numPixels-ii-1:sz-ii-1,0:sz]
        is_bad_below+=bad[ii+1:sz-numPixels+ii+1,0:sz]
    is_bad_to_left = is_bad_to_left>0
    is_bad_to_right = is_bad_to_right>0
    is_bad_above = is_bad_above>0
    is_bad_below = is_bad_below>0
    is_surrounded = np.zeros((sz,sz))
    is_surrounded[0:sz,numPixels:sz]+=is_bad_to_left
    is_surrounded[0:sz,0:sz-numPixels]+=is_bad_to_right
    is_surrounded[numPixels:sz,0:sz]+=is_bad_above
    is_surrounded[0:sz-numPixels,0:sz]+=is_bad_below
    is_surrounded = is_surrounded>2
    return is_surrounded
    
 def saturated_pixels(self,image,header,threshold=7500):
    """Returns coordinates of all saturated pixels
    Uses image and header from file
    
    Parameters
    ----------
    image: numpy array
        The input image

    header: pyfits header
        The header from this image.
    """
    if "COADDS" in header.keys():
        pixels = np.where(image/header["COADDS"]>threshold)
    else:
        pixels = np.where(image>threshold)
    return pixels

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
            self.csv_dict['EXPTIME'][d] + self.csv_dict['COADDS'][d] + self.csv_dict['MULTISAM'][d])
    codes = np.array(codes)
     #For each unique code, find all dark files and call make_dark.
    for c in np.unique(codes):
        w = np.where(codes == c)[0]
        if (len(w) >= 3):
            files = [ddir + self.csv_dict['FILENAME'][darks[ww]] for ww in w]
            self.make_dark(files, rdir=rdir)
            
 def make_all_flats(self, ddir='', rdir=''):
    """Search for sets of files that look like they are a series of flats. If "Lamp Off" 
    files exist within 100 files or so of the flats, call them the darks to go with the
    flats. """
    #Allow over-riding default reduction and data directories.
    if (rdir == ''):
        rdir = self.rdir
    if (ddir == ''):
        ddir = self.ddir
    if len(self.csv_dict) == 0:
        print("Error: Run read_summary_csv first. No flats made.")
        return
    #Fill in elevation with a default value (45, for dome flat position) if there are fits header errors.
    els = self.csv_dict['NAXIS1']
    for i in range(len(els)):
        try:
            this_el = float(els[i])
        except:
            els[i] = '45.0'
    els = els.astype(float)
    #If we're in the dome flat position with more than 1000 counts, this looks
    #like it could be a dome flat!
    codes = []
    flats_maybe = np.where(self.csv_dict['OBJECT']=='flats')[0]
    fluxes = self.csv_dict['MEDIAN_VALUE'][flats_maybe].astype(float)
    for ix in range(len(els)):
        codes.append(self.csv_dict['ESO INS OPTI6 ID'][ix] + self.csv_dict['NAXIS1'][ix] + self.csv_dict['NAXIS2'][ix] + 
            self.csv_dict['EXPTIME'][ix] + self.csv_dict['COADDS'][ix] + self.csv_dict['MULTISAM'][ix] + 
            self.csv_dict['SLITNAME'][ix])
    codes = np.array(codes)
    flat_codes = codes[flats_maybe]
    #For each unique code, find the files with consistent flux
    for c in np.unique(flat_codes):
        #w indexes flats_maybe
        w = np.where(flat_codes == c)[0]
        #Flux has to be within 10% of the median to count.
        this_flat_flux = np.median(fluxes[w])
        good_flats = flats_maybe[w[np.where(np.abs( (fluxes[w] - this_flat_flux)/this_flat_flux < 0.1))[0]]]
        #Less than 2 flats... don't bother.
        if (len(good_flats) >= 2):
            ffiles = [ddir + self.csv_dict['FILENAME'][ww] for ww in good_flats]
            lamp_off = np.where( (codes == c) & (np.array(self.csv_dict['MEDIAN_VALUE'].astype(float) < 600) & \
                    (np.abs(els - 45) < 0.01) ) )[0]
            if (len(lamp_off) >= 3):
                #Use these lamp_off indexes to create a "special" dark.
                dfiles = [ddir + self.csv_dict['FILENAME'][ww] for ww in lamp_off]
                try:
                    hh = pyfits.open(dfiles[0], ignore_missing_end=True)[0].header
                except:
                    hh = pyfits.open(dfiles[0]+'.gz', ignore_missing_end=True)[0].header
                dfilename = str(lamp_off[0]) + '_' + self.get_dark_filename(hh)
                self.make_dark(dfiles, out_file=dfilename)
                self.make_flat(ffiles, dark_file=dfilename)
            #Otherwise, just use default darks. This *will* give an error if they don't exist.
            else:
                self.make_flat(ffiles)
                
 def csv_block_string(self, ix):
    """Find a string from the summary csv file that identifies a unique configuration
    for a set of files to be processed as a block. It isn't *quite* correct 
    because the target name sometimes stays the same with a target change. 
    
    Parameters
    ----------
    ix: int
        The index of the file (in the csv dictionary) that we want to get a block string 
        for"""
    if len(self.csv_dict) == 0:
        print("Error: Run read_summary_csv first. No string returned.")
        return
    block_string = self.csv_dict['NAXIS1'][ix] + self.csv_dict['NAXIS2'][ix] + \
        self.csv_dict['OBJECT'][ix] + self.csv_dict['ESO INS OPTI6 ID'][ix] + \
        self.csv_dict['EXPTIME'][ix] + self.csv_dict['COADDS'][ix]
    return block_string

 def info_from_header(self, h, subarr=None):
    """Find important information from the fits header and store in a common format
    
    Parameters
    ----------
    h:    The fits header
    
    Returns
    -------
    (dark_file, flat_file, filter, wave, rad_pixel)
    """

    #First, sanity check the header
    try: inst=h['INSTRUME']
    except: inst=''
    if (len(inst)==0):
        print("Error: could not find instrument in header...")
        raise UserWarning
    if ((self.instrument != inst) & (inst[0:3] != '###')):
        print("Error: software expecting: ", self.instrument, " but instrument is: ", inst)
        raise UserWarning

    """try: fwo = h['FWONAME']
    except:
        print("No FWONAME in NACO header")
        raise UserWarning"""
    try: fwi = h['ESO INS OPTI6 ID']
    except:
        print("No FWINAME in NACO header")
        raise UserWarning
    try: slit = h['SLITNAME']
    except:
        slit = 'none'
        """print("No SLITNAME in NACO header")
        raise UserWarning"""
    if (fwi=='J'):
        wave = 1.265e-6
        filter='J'
    elif (fwi=='H'):
        wave = 1.66e-6
        filter='H'
    elif (fwi=='Ks'):
        wave = 2.18e-6
        filter='Ks'
    elif (fwi=='L_prime'):
        wave = 3.8e-6
        filter='L_prime'
    elif (fwi=='M_prime'):
        wave = 4.78e-6
        filter='M_prime'
    elif ('NB' in fwi or 'IB' in fwi):
        wave = float(fwi[3:len(fwi)])*1e-6
        filter = fwi
    elif (fwi=='empty'):
        wave = 5e-7
        filter = 'empty'
    else:
        print("Unknown Filter!")
        pdb.set_trace()
    if (slit == 'none'):
        flat_file = 'flat_' + filter + '.fits'
    else:
        flat_file = 'flat_' + filter + '_' + slit + '.fits'

    try: camname = h['CAMNAME']
    except:
           camname = 'narrow_VLT'
           print("No CAMNAME in header")
    if (camname == 'narrow'):
        #This comes from the Yelda (2010) paper.
         rad_pixel = 0.009952*(np.pi/180.0/3600.0)
    elif (camname == 'narrow_VLT'):
         rad_pixel = 0.03*(np.pi/180.0/3600.0)
    else:
        print("Unknown Camera!")
        raise UserWarning
    #Estimate the expected readout noise directly from the header.
    """if h['SAMPMODE'] == 2:
        multisam = 1
    else:
        multisam = h['MULTISAM']"""
    multisam = 1
    #The next line comes from the NACO manual.
    if fwi=='L_prime':
        gain = 9.8
    elif fwi=='M_prime':
        gain = 9.0
    else:
        gain = 11.0
    rnoise = 4.4
    #Find the appropriate dark file if needed.
    dark_file = self.get_dark_filename(h)
    targname = h['ESO OBS NAME']
    #The pupil orientation...
    try:
        el = h['ESO TEL ALT']
    except:
        el = -1
    if (el > 0):
        vertang_pa = (h['ESO ADA ABSROT START']+h['ESO ADA ABSROT END'])/2
        altstart = 90-(180/np.pi)*np.arccos(1./h['ESO TEL AIRM START'])
        altend = 90-(180/np.pi)*np.arccos(1./h['ESO TEL AIRM END'])
        vertang_pa += (altstart+altend)/2
        pa = vertang_pa-(180-(h['ESO TEL PARANG START']+h['ESO TEL PARANG END'])/2)
    else:
        vertang_pa=np.NaN
        pa = np.NaN
    #Find the pupil type and parameters for the pupil...
    pupil_params=dict()
    pupil_type = 'annulus'
    pupil_params['outer_diam'] = 8.2
    #Secondary obstruction guesstimated form picture on ESO webpage.
    pupil_params['inner_diam'] = 1.5

    ftpix_file = 'ftpix_' + filter + '_fullpupil.fits'
    if subarr:
        subarr_string = '_' + str(subarr)
    else:
        subarr_string = ''
    ftpix_file = 'ftpix_' + filter + '_fullpupil' + subarr_string + '.fits'
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
    dark_file = 'dark_' + str(h['NAXIS1']) + '_' + str(int(h['EXPTIME']*100)) + '.fits'
    return dark_file
    
 def destripe_conica(self,im, subtract_edge=True, subtract_median=False, do_destripe=True):
    """Destripe an image from the NACO camera.
    
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
    quads = [im[0:s[0]//2,0:s[1]//2],im[s[0]:s[0]//2-1:-1,0:s[1]//2],
             im[0:s[0]//2,s[1]:s[1]//2-1:-1],im[s[0]:s[0]//2-1:-1,s[1]:s[1]//2-1:-1]]
    #print(quads)
    quads = np.array(quads, dtype='float')
    #Work through the quadrants, modifying based on the edges.
    if subtract_edge:
        quads[1] += np.median(quads[3][:,s[1]//2-8:s[1]//2])- np.median(quads[1][:,s[1]//2-8:s[1]//2]) 
        quads[2] += np.median(quads[3][s[0]//2-8:s[0]//2,:])- np.median(quads[2][s[0]//2-8:s[0]//2,:])  
        delta = 0.5*(np.median(quads[3][s[0]//2-8:s[0]//2,:]) + np.median(quads[3][:,s[1]//2-8:s[1]//2])
               - np.median(quads[0][s[0]//2-8:s[0]//2,:]) - np.median(quads[0][:,s[1]//2-8:s[1]//2]))
        quads[0] += delta
    #Subtract the background
    if subtract_median:
        print("Subtracting Medians...")
        MED_DIFF_MULTIPLIER = 4.0
        for i in range(4):
            quad = quads[i,:,:]
            med = np.median(quad)
            dispersion = np.median(np.abs(quad - med))
            goodpix = np.where(np.abs(quad - med) < MED_DIFF_MULTIPLIER*dispersion)
            med = np.median(quad[goodpix])
            quads[i,:,:] -= med
    if do_destripe:
        quads = quads.reshape((4,s[0]//2,s[1]//16,8))
        stripes = quads.copy()
        for i in range(4):
            for j in range(s[0]//2): #The -1 on  line is because of artifacts
                for k in range(s[0]//16):
                    pix = np.array([stripes[(i+1)%4,j,k,:],stripes[(i+2)%4,j,k,:],stripes[(i+3)%4,j,k,:]])
                    quads[i,j,k,:] -= np.median(pix)
        quads = quads.reshape((4,s[0]//2,s[1]//2))
    im[0:s[0]//2,0:s[1]//2] = quads[0]
    im[s[0]:s[0]//2-1:-1,0:s[1]//2] = quads[1]
    im[0:s[0]//2,s[1]:s[1]//2-1:-1] = quads[2]
    im[s[0]:s[0]//2-1:-1,s[1]:s[1]//2-1:-1] = quads[3]
    return im
    
 def make_dark(self,in_files, out_file='', subtract_median=True, destripe=True, med_threshold=15.0, rdir=''):
    """Create a dark frame and save to a fits file, 
    with an attached bad pixel map as the first fits extension.

    Parameters
    ----------
    in_files : array_like (dtype=string). A list of input filenames.
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
        print("At least 3 dark files sre needed for reliable statistics")
        raise UserWarning
    # Read in the first dark to check the dimensions.
    try:
        in_fits = pyfits.open(in_files[0], ignore_missing_end=True)
    except:
        in_fits = pyfits.open(in_files[0]+'.gz', ignore_missing_end=True)
    h = in_fits[0].header
    instname = ''
    try: instname=h['ESO INS ID']
    except:
        print("Unknown Header Type")
    #Create the output filename if needed
    if (out_file == ''):
        out_file = self.get_dark_filename(h)
    s = in_fits[0].data.shape
    in_fits.close()
    darks = np.zeros((nf,s[0],s[1]))
    plt.clf()
    for i in range(nf):
        #Read in the data
        adark = pyfits.getdata(in_files[i])
        if ('CONICA' in instname):
            adark = self.destripe_conica(adark, subtract_median=subtract_median, do_destripe=destripe)
        if (subtract_median):
            plt.imshow(np.minimum(adark,1e2))
        else:
            plt.imshow(adark)
            print("Median: " + str(np.median(adark)))
        plt.pause(0.001)
        #plt.draw()
        darks[i,:,:] = adark
    #Now look for weird pixels. 
    med_dark = np.median(darks, axis=0)
    max_dark = np.max(darks, axis=0)
    var_dark = np.zeros((s[0],s[1]))
    for i in range(nf):
        var_dark += (darks[i,:,:] - med_dark)**2
    var_dark -= (max_dark - med_dark)**2
    var_dark /= nf-2
    #We need to threshold the med_diff quantity in case of low-noise, many subread images
    med_diff = np.maximum(np.median(np.abs(med_dark - np.median(med_dark))),1.0)
    print("Median difference: " + str(med_diff))
    med_var_diff = np.median(np.abs(var_dark - np.median(var_dark)))
    bad_med = np.abs(med_dark - np.median(med_dark)) > med_threshold*med_diff
    bad_var = np.abs(var_dark) > np.median(var_dark) + VAR_THRESHOLD*med_var_diff
    print("Pixels with bad mean: " + str(np.sum(bad_med)))
    print("Pixels with bad var: " + str(np.sum(bad_var)))
    bad = np.logical_or(bad_med, bad_var)
    med_dark[bad] = 0.0
    #Copy the original header to the dark file.
    hl = pyfits.HDUList()
    hl.append(pyfits.ImageHDU(med_dark,h))
    hl.append(pyfits.ImageHDU(np.uint8(bad)))
    hl.writeto(rdir+out_file,output_verify='ignore',clobber=True)
    """plt.figure(1)
    plt.imshow(med_dark,cmap=cm.gray, interpolation='nearest')
    plt.title('Median Frame')
    plt.figure(2)
    plt.imshow(bad,cmap=cm.gray, interpolation='nearest')
    plt.title('Bad Pixels')
    plt.pause(0.001)"""
    #plt.draw()
    
 def _calibration_subarr(self, rdir, flat_file, dark_file, szx, szy, wave=0):
    """A function designed to be used internally only, which chops out the central part
    of calibration data for sub-arrays. It also automatically finds the nearest wavelength
    flat if an appropriate flat doesn't exist. """
    if len(flat_file) > 0:
        try:
            flat = pyfits.getdata(rdir + flat_file,0)
        except:
            if wave>0:
                #Find the flat file with the nearest wavelengths. In this case, ignore
                #corona flats.
                flat_files = glob.glob(rdir + 'flat*fits')
                if len(flat_files)==0:
                    print("No flat! Are you sure that this is your reduction directory? " + rdir)
                    pdb.set_trace()
                waves = []
                for ffile in flat_files:
                    if ffile.find("corona") > 0:
                        waves.append(-1)
                    else:
                        try: 
                            wave_file = pyfits.getheader(ffile)['WAVE']
                            waves.append(wave_file)
                        except:
                            print("Missing header keyword WAVE!")
                            pdb.set_trace()
                waves = np.array(waves)
                ix = np.argmin(np.abs(waves - wave))
                new_flat_file = flat_files[ix][len(rdir):]
                print("*** Flat file {0:s} not found! Using {1:s} intead. ***".format(flat_file, new_flat_file)) 
                flat_file = new_flat_file
                flat = pyfits.getdata(rdir + flat_file,0)
            else:
                print("ERROR - no flat file!")
                pdb.set_trace()
        flat = flat[flat.shape[0]//2 - szy//2:flat.shape[0]//2 + szy//2,flat.shape[1]//2 - szx//2:flat.shape[1]//2 + szx//2]
        bad = pyfits.getdata(rdir + flat_file,1)
        bad = bad[bad.shape[0]//2 - szy//2:bad.shape[0]//2 + szy//2,bad.shape[1]//2 - szx//2:bad.shape[1]//2 + szx//2]
    else:
        flat = np.ones((szy,szx))
        bad = np.zeros((szy,szx))
    if len(dark_file) > 0:
        try:
            dark = pyfits.getdata(rdir + dark_file,0)
            if (szy != dark.shape[0]):
                print("Warning - Dark is of the wrong shape!")
                dark = dark[dark.shape[0]//2 - szy//2:dark.shape[0]//2 + szy//2, \
                       dark.shape[1]//2 - szx//2:dark.shape[1]//2 + szx//2]
        except:
            print("*** Warning - Dark file {0:s} not found! Using zeros for dark ***".format(dark_file))
            dark = np.zeros((szy,szx))
    else:
        dark = np.zeros((szy,szx))
    return (flat,dark,bad)
    
 def clean_no_dither(self, in_files, fmask_file='',dark_file='', flat_file='', fmask=[],\
     subarr=None,extra_threshold=7,out_file='',median_cut=0.7, destripe=True, ddir='', rdir='', cdir='', manual_click=False):
    """Clean a series of fits files, including: applying the dark, flat, 
    removing bad pixels and cosmic rays. This can also be used for dithered data, 
    but it will not subtract the dithered positions. There reason for two separate
    programs includes that for dithered data, bad pixel rejection etc has to be done on
    *all* riles.

    Parameters
    ----------
    in_files : array_like (dtype=string). 
        A list of input filenames.
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
    return self.clean_dithered(in_files, fmask_file=fmask_file,dark_file=dark_file, flat_file=flat_file, fmask=fmask,\
     subarr=subarr,extra_threshold=extra_threshold,out_file=out_file,median_cut=median_cut, destripe=destripe, \
     ddir=ddir, rdir=rdir, cdir=cdir, manual_click=manual_click, dither=False)
    
 def clean_dithered(self, in_files, fmask_file='',dark_file='', flat_file='', fmask=[],\
    subarr=None, extra_threshold=7,out_file='',median_cut=0.7, destripe=True, \
    manual_click=False, ddir='', rdir='', cdir='', dither=True, show_wait=1, \
    dodgy_badpix_speedup=False, subtract_median=False, extra_diagnostic_plots=False):
    """Clean a series of fits files, including: applying the dark and flat, removing bad pixels and
    cosmic rays, creating a `rough supersky' in order to find a mean image, identifying the target and any 
    secondary targets, identifying appropriate sky frames for each frame and subtracting these off. In
    order to find objects in the image in vertical angle mode, the assumption is made that rotation 
    is much less than interferogram size.
    
    To enhance code readability, many of the options in previous routines have been removed.

    Parameters
    ----------
    in_files : array_like (dtype=string). 
        A list of input filenames.
    fmask_file: string
        The Fourier mask and ftpix fits file - input either this or ftpix and subarr
    dark_file: string
        The dark file, previously created with make_dark
    flat_file: string
        The flat file, previously created with make_flat
    ftpix: ( (N) array, (N) array)
        The pixels in the data's Fourier Transform that include all non-zero
        values (created using pupil_sampling)
    subarr: int
        The size of the sub-array, if ftpix is given manually rather than as an fmask_file
    extra_threshold: float, optional
        A threshold for identifying additional bad pixels and cosmic rays.
    outfile: string,optional
        A filename to save the cube as, including the header of the first
        fits file in the cube plus extra information.
    destripe: bool
        Do we destripe the data? This is *bad* for thermal infrared (e.g. the L' filter)
        TODO: Make this a default depending on filter.
    
    
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
    #Decide on the image size from the first file. !!! are x and y around the right way?
    try:
        in_fits = pyfits.open(ddir + in_files[0], ignore_missing_end=True)
    except:
        in_fits = pyfits.open(ddir + in_files[0] + '.gz', ignore_missing_end=True)
    h = in_fits[0].header
    in_fits.close()
    szx = h['NAXIS1']
    szy = h['NAXIS2']

    #Extract important information from the header...
    hinfo = self.info_from_header(h, subarr=subarr)
    rnoise = hinfo['rnoise']
    gain = hinfo['gain']
    rad_pixel = hinfo['rad_pixel']
    if (len(dark_file) == 0):
        dark_file = hinfo['dark_file']
    if (len(flat_file) == 0):
        flat_file = hinfo['flat_file']
    #If we set the fmask manually, then don't use the file.
    if len(fmask) == 0:
        #If no file is given, find it automatically.
        if len(fmask_file) == 0:
            fmask_file = hinfo['ftpix_file']
        try:
            fmask = pyfits.getdata(rdir + fmask_file,1)
            subarr = pyfits.getheader(rdir + fmask_file)['SUBARR']
        except:
            print("Error - couldn't find kp/Fourier mask file: " +fmask_file+ " in directory: " + rdir)
            raise UserWarning
    elif not subarr:
        raise UserWarning("Must set subarr if fmask is set!")
        

    #Allocate memory for the cube and the full cube
    nf = len(in_files)
    cube = np.zeros((nf,subarr,subarr))
    full_cube = np.zeros((nf,szy, szx))
    bad_cube = np.zeros((nf,subarr,subarr), dtype=np.uint8)

    #Chop out the appropriate part of the flat, dark, bad arrays
    (flat,dark,bad) = self._calibration_subarr(rdir, flat_file, dark_file, szx, szy, wave=hinfo['wave'])
    wbad = np.where(bad)
    #Go through the files, cleaning them one at a time and adding to the cube. 
    pas = np.zeros(nf)
    decs = np.zeros(nf)
    maxs = np.zeros(nf)
    xpeaks = np.zeros(nf,dtype=int)
    ypeaks = np.zeros(nf,dtype=int)
    backgrounds = np.zeros(nf)
    offXs = np.zeros(nf)
    offYs = np.zeros(nf)
    for i in range(nf):
        #First, find the position angles from the header keywords. NB this is the Sky-PA of chip vertical.
        try:
            in_fits = pyfits.open(ddir + in_files[i], ignore_missing_end=True)
        except:
            in_fits = pyfits.open(ddir + in_files[i] + '.gz', ignore_missing_end=True)
        hdr = in_fits[0].header
        in_fits.close()
        rotstart=hdr['ESO ADA ABSROT START']
        rotend=hdr['ESO ADA ABSROT END']
        pastart=hdr['ESO TEL PARANG START']
        paend=hdr['ESO TEL PARANG END']
        alt=hdr['ESO TEL ALT']
        instrument_offset= -0.55
        pas[i]=(rotstart+rotend)/2.+alt-(180.-(pastart+paend)/2.) + instrument_offset
        decs[i] = hdr['DEC']
        offXs[i] = int(hdr['ESO SEQ CUMOFFSETX'])
        offYs[i] = int(hdr['ESO SEQ CUMOFFSETY'])
        #Read in the image
        im = pyfits.getdata(ddir + in_files[i])
        
        #Find saturated pixels and remove them.
        saturation = self.saturated_pixels(im,hdr)
        bad[saturation]=1
        #surrounded = self.is_bad_surrounded(bad)
        #bad+=surrounded
        
        #!!! It is debatable whether the dark on the next line is really useful... but setting 
        #dark_file='' removes its effect.
        im = (im - dark)/flat
        #For display purposes, we do a dodgy bad pixel correction.
        mim = nd.filters.median_filter(im,size=3)
        im[bad] = mim[bad]
        full_cube[i,:,:] = im
        
    #Find the rough "supersky", by taking the 25th percentile of each pixel.
    if (dither):
        rough_supersky = np.percentile(full_cube, 25.0, axis=0)
    else:
        rough_supersky = np.zeros(im.shape)
    #Subtract this supersky off each frame. Don't worry - all strictly pixel-dependent
    #offsets are removed in any case so this doesn't bias the data.
    for i in range(nf):
        full_cube[i,:,:] -= rough_supersky

    shifts = np.zeros((nf,2),dtype=int)
    im_mean = np.zeros((szy, szx))
    for i in range(nf):
        th = np.radians(pas[i])
        rot_mat = np.array([[np.cos(th), -np.sin(th)],[-np.sin(th), -np.cos(th)]])
        shifts[i,:] = np.array([offXs[i],offYs[i]])#np.dot(rot_mat, np.array([offXs[i],offYs[i]]))
        im_mean+=np.roll(np.roll(full_cube[i,:,:],-shifts[i,0], axis=1), -shifts[i,1], axis=0)
    #Find the star...     
    #Show the image, y-axis reversed.
    """plt.clf()
    plt.imshow(np.arcsinh(im_mean/100), interpolation='nearest', origin='lower')
    arrow_xy = np.dot(rot_mat, [0,-30])
    plt.arrow(60,60,arrow_xy[0],arrow_xy[1],width=0.2)
    plt.text(60+1.3*arrow_xy[0], 60+1.3*arrow_xy[1], 'N')
    arrow_xy = np.dot(rot_mat, [-30,0])
    plt.arrow(60,60,arrow_xy[0],arrow_xy[1],width=0.2)
    plt.text(60+1.3*arrow_xy[0], 60+1.3*arrow_xy[1], 'E')"""
    if manual_click:
        #plt.title('Click on target...')
        max_ix = plt.ginput(1, timeout=0)[0]
        #To match the (y,x) order below, change this...
        max_ix = int(max_ix[1]), int(max_ix[0])
    else:
        im_filt = nd.filters.median_filter(im_mean,size=5)
        max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
        #plt.title('Identified target shown')
    """plt.plot(max_ix[1], max_ix[0], 'wx', markersize=20,markeredgewidth=2)
    plt.pause(0.001)"""
    #plt.draw()
    print("Maximum x,y: " + str(max_ix[1])+', '+ str(max_ix[0]))
    time.sleep(show_wait)

    #Set the xpeaks and ypeaks values (needed for sub-arraying later)
    for i in range(nf):
        xpeaks[i] = max_ix[1] + shifts[i,0]
        ypeaks[i] = max_ix[0] + shifts[i,1]
    
    subims = np.empty( (nf,subarr,subarr) )
    #Sky subtract and fix bad pixels.
    for i in range(nf):
        #For each frame, cut out a sub-image centered on this particular (x,y) location
        #corresponding to frame i. 
        for j in range(nf):
            #Undo the flat, to minimise the effects of errors in the flat.
            im = full_cube[j,:,:]*flat
            #Roll all the sub-images, and cut them out.
            subims[j,:,:] = np.roll(np.roll(im,subarr//2-ypeaks[i],axis=0),
                subarr//2-xpeaks[i],axis=1)[0:subarr,0:subarr]
        #Find a flat to re-apply
        subflat = np.roll(np.roll(flat,subarr//2-ypeaks[i],axis=0),subarr//2-xpeaks[i],axis=1)
        subflat = subflat[0:subarr,0:subarr]
        #Find the frames that are appropriate for a dither...
        if (dither):
            w = np.where( (xpeaks - xpeaks[i])**2 + (ypeaks - ypeaks[i])**2 > 0.5*subarr**2 )[0]
            if len(w) == 0:
                print("Error: Can not find sky from dithered data - use dither=False for {0:s}".format(in_files[0]))
                return
            #To avoid too many extra bad pixels, we'll use a median here.
            sky = np.median(subims[w,:,:], axis=0)
            backgrounds[i] += np.median(sky)
            #Subtract the sky then re-apply the flat.
            subim = (subims[i,:,:] - sky)/subflat
        else:
            subim = subims[i,:,:]/subflat
        #Find the peak from the sub-image.
        im_filt = nd.filters.median_filter(subim,size=5)
        max_ix = np.unravel_index(im_filt.argmax(), im_filt.shape)
        maxs[i] = subim[max_ix[0],max_ix[1]]
        
        #subbad is the set of bad pixels in the sub-array.
        subbad = np.roll(np.roll(bad,subarr//2-ypeaks[i],axis=0),subarr//2-xpeaks[i],axis=1)
        subbad = subbad[0:subarr,0:subarr]
        new_bad = subbad.copy()    
        subim[np.where(subbad)] = 0
        """plt.clf()
        plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.title(hinfo['targname']) 
        plt.pause(0.001)"""
        #import pdb; pdb.set_trace()
        #plt.draw()
        #Iteratively fix the bad pixels and look for more bad pixels...
        for ntry in range(1,15):
            # Correct the known bad pixels. Ideally, we self-consistently correct
            # all bad pixels at once.
            if dodgy_badpix_speedup:
                self.fix_bad_pixels(subim,new_bad,fmask)
            else:
                self.fix_bad_pixels(subim,subbad,fmask)
                         
            # Search for more bad pixels. Lets use a Fourier technique here, where we
            # take the inverse Fourier transform of the region of the image Fourier transform
            # that is the null space of the MTF
            extra_bad_ft = np.fft.rfft2(subim)*fmask
            bad_image = np.real(np.fft.irfft2(extra_bad_ft))
            mim = nd.filters.median_filter(subim,size=5)
            
            # NB The next line *should* take experimentally determined readout noise into account
            # rather than a fixed readout noise!!!
            total_noise = np.sqrt(np.maximum((backgrounds[i] + mim)/gain + rnoise**2,rnoise**2))
            bad_image = bad_image/total_noise
            
            #In case of a single bad pixel, we end up with a ringing effect where the 
            #surrounding pixels also look bad. So subtract a median filtered image.
            unsharp_masked = bad_image-nd.filters.median_filter(bad_image,size=3)
            
            # The "extra_threshold" value for extra bad pixels is a scaling of the median 
            # absolute deviation. We set a limit where new bad pixels can't have 
            # absolute values morer than 0.2 times the peak bad pixel.
            current_threshold = np.max([0.25*np.max(np.abs(unsharp_masked[new_bad == 0])), \
                extra_threshold*np.median(np.abs(bad_image))])
            extra_bad = np.abs(unsharp_masked) > current_threshold
            n_extra_bad = np.sum(extra_bad)
            print(str(n_extra_bad)+" extra bad pixels or cosmic rays identified. Attempt: "+str(ntry))
            
            #TESTING - too many bad pixels are identified in L' data, and noise isn't 
            #consistent.
            if extra_diagnostic_plots:
                """plt.clf()
                plt.imshow(np.maximum(subim,0)**0.5, interpolation='nearest', cmap=cm.cubehelix)
                new_bad_yx = np.where(new_bad)
                plt.plot(new_bad_yx[1], new_bad_yx[0], 'wx')
                plt.axis([0,subarr,0,subarr])"""
                import pdb; pdb.set_trace()
            
            subbad += extra_bad
            if (ntry == 1):
                new_bad = extra_bad
            else:
                new_bad += extra_bad
                new_bad = extra_bad>0
            if (n_extra_bad == 0):
                break
        print(str(np.sum(subbad)) + " total bad pixels.")
        
        #Now re-correct both the known and new bad pixels at once.
        self.fix_bad_pixels(subim,subbad,fmask)
        """plt.imshow(np.maximum(subim,0)**0.5,interpolation='nearest')
        plt.pause(0.001)"""
        #plt.draw()
        #Save the data and move on!
        cube[i]=subim
        subbad = subbad>0
        bad_cube[i]=subbad.astype(np.uint8)

    #Find bad frames based on low peak count.
    good = np.where(maxs > median_cut*np.median(maxs))
    good = good[0]
    if (len(good) < nf):
        print(nf-len(good), " frames rejected due to low peak counts.")
    cube = cube[good,:,:]
    nf = np.shape(cube)[0]
    #If a filename is given, save the file.
    if (len(out_file) > 0):
        hl = pyfits.HDUList()
        h['RNOISE'] = rnoise
        h['PGAIN'] = gain #P means python
        h['SZX'] = szx
        h['SZY'] = szy
        h['DDIR'] = ddir
        h['TARGNAME'] = hinfo['targname']
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
        cols = pyfits.ColDefs([col1,col2,col3,col4,col5])
        hl.append(pyfits.BinTableHDU.from_columns(cols))
        hl.append(pyfits.ImageHDU(bad_cube))
        hl.writeto(cdir+out_file,output_verify='ignore',clobber=True)
        print(cube.shape)
    return cube