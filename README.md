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