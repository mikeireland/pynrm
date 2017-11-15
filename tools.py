import numpy as np
import pdb
def subpixel_roll(im, roll_in_pix):
	"""
	Parameters
	----------
	im: numpy array
		The input image (real)
	roll_in_pix: numpy array(2)
		Floating point amounts to shift in each direction.

	Returns
	-------
	shifted_im: numpy array
		The shifted image.
	"""
	#y and x in conventional screen plotting sense.
	ny = im.shape[0]
	nx = im.shape[1]

	#Create a Fourier domain y co-ordinate uv[0], and a Fourier domain x co-ordinate uv[1]
	uv = np.meshgrid((((np.arange(ny) + ny//2) % ny) - ny//2)/ny, np.arange(nx//2 + 1)/nx, indexing='ij')

	#Fourier transform the image
	ftim = np.fft.rfft2(im)

	#Now the magical Fourier shift, where convolution with a delta function is multiplication
	#by a phase ramp in the Fourier domain.
	ftim *= np.exp(-2j*np.pi*(uv[0]*roll_in_pix[0] + uv[1]*roll_in_pix[1]))

	#Return the inverse transform.
	return np.fft.irfft2(ftim)

def filter_image(image):
        image1 = np.roll(image,1,0)
        image2 = np.roll(image,-1,0)
        cube = np.ndarray((image.shape[0],image.shape[1],9))
        cube[:,:,0] = image
        cube[:,:,1] = image1
        cube[:,:,2] = image2
        cube[:,:,3] = np.roll(image,1,1)
        cube[:,:,4] = np.roll(image1,1,1)
        cube[:,:,5] = np.roll(image2,1,1)
        cube[:,:,6] = np.roll(image,-1,1)
        cube[:,:,7] = np.roll(image1,-1,1)
        cube[:,:,8] = np.roll(image2,-1,1)
        return np.median(cube,2)

def peak_coords(image):
# takes a numpy array and returns the x,y coords of the peak value
# takes the median filtered image from filter_image() as input
    shape = np.shape(image)
    maxindex = image.argmax()
    coords = []
    coords.append(maxindex % shape[1]) # max pixel x value
    coords.append(int(maxindex/shape[1])) # max pixel y value
    coords.append(image.max())
    return coords

# Walk into directories in filesystem
# Ripped from os module and slightly modified
# for alphabetical sorting
#

def sortedWalk(top, topdown=True, onerror=None):
    from os.path import join, isdir, islink
    import os
    names = os.listdir(top)
    names.sort()
    dirs, nondirs = [], []
    for name in names:
        if isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)
    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = join(top, name)
        if not os.path.islink(path):
            for x in sortedWalk(path, topdown, onerror):
                yield x
    if not topdown:
        yield top, dirs, nondirs
