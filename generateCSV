#!/usr/bin/env python

#import pyximport; pyximport.install()
import popCSV
import time#, jpb

print 'Program started at time:'
print time.asctime( time.localtime(time.time()) )

path = './'#jpb.ask('Which directory to run this code?','./') # path to root of directory to walk through, relative to location of script
outfile = path + "datainfo.csv" # the output file to write to

keys = ["FILTER", \
        "EFFWAVE", \
        "OBJECT", \
        "TARGNAME", \
        "CAMNAME", \
        "IMAGETYP", \
        "SHRNAME", \
        "SLITNAME",\
        "COADDS", \
        "ITIME", \
        "DATE-OBS", \
        "MJD-OBS", \
        "UTC", \
        "EL", \
        "RA", \
        "DEC", \
        "NAXIS1", \
        "NAXIS2", \
        "PIXSCALE", \
        "SLITNAME", \
        "FWINUM", \
        "FWONUM"]

# The headers for analysis done on the files
# MULTISAM is 1 if SAMPMODE is 2, else write value in header
# PEAKPIX is one operation, but returns 2 values, needs two columns
# SATURATED be sure to set the threshold value for level before saturated
operations = ["MULTISAM", \
              "PEAKPIX(X)", \
              "PEAKPIX(Y)", \
              "PEAK_VALUE", \
              "MEDIAN_VALUE", \
              "SATURATED" \
             ]

# try to be fancy writing the column headers
# Directory and Filename are always going to be the first two columns for the output
colheads = ["DIRECTORY", \
            "FILENAME" \
           ]

colheads=colheads+keys+operations

########################
# Begin main part of the script
########################

popCSV.popCSV(keys,operations,colheads,path,outfile)


print 'Program finished at time:'
print time.asctime( time.localtime(time.time()) )
