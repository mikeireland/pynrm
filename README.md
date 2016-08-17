# pynrm
A python non-redundant masking and AO imaging pipeline based on the POISE algorithm. 

The code can be run in two ways - either from a python prompt, or as a series of
scripts. The key scripts are:

run_csv: Create the comma separated value file.

run_calibration: Makes darks and flats

run_make_ftpix: From a high SNR observation, tweaks the Fourier sampling for a particular
mask (including full pupil).

run_clean: Clean the data and chop out a subarray containing our object of interest.

run_XXX: Missing scripts to do Kernel-phase or other aperture masking analysis.

TODO:
1) Figure out how to modify the nirc2 files natively from python in order to have fits
compliant files. Currently, you need to download the fits files form the Keck archive or
read and write from IDL
2) Document "go.pro" including
rdir: REDUCTION DIRECTORY (with flats and darks etc.)
cdir: CUBE DIRECTORY (contains cleaned data cubes)
ddir: RAW DATA DIRECTORY