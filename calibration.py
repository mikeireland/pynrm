#! /usr/bin/env python

#Python script used to calibrate IR data.

from pyraf import iraf
try:
	import pyfits
except:
	import astropy.io.fits as pyfits

import sys, os,commands
import numpy as np
from subprocess import Popen, PIPE
import pdb,time

#Set image type to fits for the purpose of this calibration. 
iraf.set(imtype='fits')
iraf.reset(clobber='yes')

print 'IMPORTANT: All files used in this script must be uncompressed!'

def bad_pix(a,s):
    '''Function designed to return a bad pixel map based on the median and median absolute deviation and replaces anything out of abs(s*mad) with nans'''
    med=np.median(a)
    #calculate median absolute deviation
    mad=np.median(np.abs(a-med))
    b=np.float64(a)
    b[where((b<(med-3*mad))|(b>(med+s*mad)))]=np.nan
    return b

def ask(text,def_val):
    temp=raw_input(text+' = '+str(def_val)+' =')
    if len(temp)==0:
        return def_val
    else:
        return type(def_val)(temp)


#ask which files are going to be calibrated.
data_list=ask('Which data files are to be calibrated? Specify either as a filename with the list of files or as a starting and ending number separated by a space character (for nirc2 data).', 'inlist')


#At this point, test whether the input for the data_list was a couple of numbers stating the starting and final flie to be analysed, or a list of files, and format both such that iraf can use them
if ' ' in data_list:
    try: 
        first,last=data_list.split(' ')
        first=first.zfill(4)
        last=last.zfill(4)
        p=Popen(['/bin/bash','-c','for i in {'+first+'..'+last+'}; do ls n*$i.*; done > datafilelist'])
        p.communicate()
        os.system('cat datafilelist')
        dummy=ask('Is this the list of files that are to be calibrated?','Y')
        if dummy=='n' or dummy=='N':
            sys.exit()
        data_list='datafilelist'
    except Exception: print 'Space character found in data list input but unable to determine which files are going to used'; sys.exit()

#Do the same for darks and flats
dark_list=ask('Which darks are to be used? Specify either as a filename with the list of files or as a starting and ending number separated by a space character.', 'darklist')
if ' ' in dark_list:
    try: 
        first,last=dark_list.split(' ')
        first=first.zfill(4)
        last=last.zfill(4)
        p=Popen(['/bin/bash','-c','for i in {'+first+'..'+last+'}; do ls n*$i.*; done > darkfilelist'])
        p.communicate()
        os.system('cat darkfilelist')
        dummy=ask('Is this the list of files that are to be used as darks?','Y')
        if dummy=='n' or dummy=='N':
            sys.exit()
        data_list='darkfilelist'
    except Exception: print 'Space character found in dark list input but unable to determine which files are going to used'
    
#The flats have the option of not being used
flat_list=ask('Which flats are to be used? Specify either "none" for no flat calibration, as a filename with the list of files or as a starting and ending number separated by a space character.', 'none')

if ' ' in flat_list:
    try: 
        first,last=flat_list.split(' ')
        first=first.zfill(4)
        last=last.zfill(4)
        p=Popen(['/bin/bash','-c','for i in {'+first+'..'+last+'}; do ls n*$i.*; done > flatfilelist'])
        p.communicate()
        os.system('cat flatfilelist')
        dummy=ask('Is this the list of files that are to be calibrated?','Y')
        if dummy=='n' or dummy=='N':
            sys.exit()
        flat_list='flatfilelist'
    except Exception: print 'Space character found in flat list input but unable to determine which files are going to used'

#Decompression if necessary


#Make dark calibration frames using IRAF's imcombine
print 'Combining dark frames into a median and average dark'
try:
    iraf.imcombine(input='@darkfilelist',output='med_dark.fits',combine='median',scale='median',reject='sigclip',lsigma=3.0,hsigma=3.0,mclip='Yes')
    iraf.imcombine(input='@darkfilelist',output='av_dark.fits',combine='average',scale='mean',reject='sigclip',lsigma=3.0,hsigma=3.0,mclip='No')
except Exception:
    print 'The procedure to combine the darks has failed'
    sys.exit()


#Make flat calibration frames using IRAF's imcombine
print 'Combining flat frames into a flat field '
try:
    iraf.imcombine(input='@flatfilelist',output='med_flat.fits',combine='median',scale='median',reject='sigclip',lsigma=3.0,hsigma=3.0,mclip='Yes')
    iraf.imcombine(input='@flatfilelist',output='av_flat.fits',combine='average',scale='mean',reject='sigclip',lsigma=3.0,hsigma=3.0,mclip='No')
    iraf.imcombine(input='@flatfilelist',output='sum_flat.fits',combine='sum',scale='mean',reject='sigclip',lsigma=3.0,hsigma=3.0,mclip='No')
except Exception:
    print 'The procedure to combine the flats has failed'
    sys.exit()

#Decide upon which things are bad pixels and apply to both the median dark and median flat (those that will actually be used to calibrate the images)
mdark=pyfits.getdata('med_dark.fits',ignore_missing_end=True)
mflat=pyfits.getdata('med_flat.fits',ignore_missing_end=True)

bad_dark=bad_pix(mdark,5)
bad_dark_dev=bad_pix(mdark-pyfits.getdata('av_dark.fits',ignore_missing_end=True),5)
bad_flat=bad_pix(mflat,5)

#bad pixel mask will be a set of nans for any bad pixel and ones everywhere else
bad_pix_mask=bad_dark*bad_dark_dev*bad_flat
bad_pix_mask[where(isnan(b)==False)]=1.

#apply bad pixel mask to median flat and dark. Any operation done on nans results in a nan, so I'm multiplying the arrays by the bad pixel mask. The good values remain as they are (multiplied by one), and the bad ones will be nans.
mdark*=bad_pix_mask
mflat*=bad_pix_mask

#Just to have a look at what it looks like.
pyfits.writeto('mdark.fits',mdark)
pyfits.writeto('mflat.fits',mflat)


#ask whether the user wants to specify the darks and flats, or whether he wants the system to find them. 

#ask whether the bad pixel mask is done by us or to specify a mask

#combine darks, flats, apply bad pixel mask, and calibrate 

