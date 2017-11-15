from __future__ import division
import numpy as np,opticstools as ot,matplotlib.pyplot as plt,math,scipy.ndimage as nd,sys,os
from skimage.measure import block_reduce
from mpl_toolkits.mplot3d import Axes3D
import itertools
sys.path.append('/Users/awallace/Documents/projects/')
sys.path.append('/Users/awallace/Documents/pynrm/')
import conv,astropy.io.fits as pyfits
import psf_marginalise as pm,stats,copy,tools
import scipy.optimize,os.path
def polyfit2d(x, y, z, order):
    """Calculate parameters of 2d polynomial
    
    Parmeters
    ---------
    x: array
       The x data.
    y: array
       The y data.
    z: array
       The result.
    order: int
           The order of the polynomial
    
    Returns
    ---------
    m: array
       The polynomial coefficients from a least squares fit.
    """
    ncols = (order+1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
    	if i!=0 and j!=0:
    		G[:,k] = np.zeros(x.size)
    	else:
        	G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def chiSquared(target,PSF):
	"""Calculate chi squared value between PSF and target.
	
	Parameters
	----------
	target: (sz,sz) array
	        The target image.
	PSF: (sz,sz) array
	     The PSF image.
	
	Returns
	----------
	value: float
	       The chi squared value found by adding up the squares of the differences.
	"""
	sz = target.shape[0]
	value = np.sum((target[0:sz//2-20,0:sz]-PSF[0:sz//2-20,0:sz])**2)+np.sum((target[sz//2+20:sz,0:sz]-PSF[sz//2+20:sz,0:sz])**2)
	return value
	
def sub_background(image):
	sz = image.shape[0]
	imFFT = np.fft.rfft2(image)
	imFFT[int(0.3*imFFT.shape[0]//2):int(1.7*imFFT.shape[0]//2),int(0.3*imFFT.shape[1]//2):int(1.7*imFFT.shape[1]//2)] = 0
	new = np.fft.irfft2(imFFT)
	quarters = np.zeros((4,sz//4,sz//4))
	for ii in range(0,4):
		xLim = np.array([(ii%2)*(3*sz//4),(ii%2)*(3*sz//4)+sz//4])
		yLim = np.array([(ii//2)*(3*sz//4),(ii//2)*(3*sz//4)+sz//4])
		quarters[ii] = new[yLim[0]:yLim[1],xLim[0]:xLim[1]]
	edgePoints = np.zeros((4,3))
	for ii in range(0,4):
		xCoord = (ii%2)*(3*sz//4)+sz//8
		yCoord = (ii//2)*(3*sz//4)+sz//8
		value = np.mean(quarters[ii])#new[yCoord,xCoord]
		edgePoints[ii] = np.array([xCoord,yCoord,value])
	
	vector1 = edgePoints[1]-edgePoints[0]
	vector2 = edgePoints[2]-edgePoints[0]
	point = edgePoints[2]
	normal = np.cross(vector1,vector2)
	plane = np.array([[point[2]+(normal[0]*point[0]+normal[1]*point[1]-normal[0]*jj-normal[1]*ii)/normal[2] for ii in range(sz)] for jj in range(sz)])
	new-=plane
	return new
	
def coarse_tilt(image):
	sz = image.shape[0]
	xMax = np.where(image==np.max(image))[1][0]
	yMax = np.where(image==np.max(image))[0][0]
	xShift = sz//2-xMax
	yShift = sz//2-yMax
	result = np.roll(np.roll(image,yShift,axis=0),xShift,axis=1)
	return result
def tilt(image):
	"""Shift image so the maximum value is in the centre.
	
	Parameters
	----------
	image: (sz,sz) array
	       The image to be shifted.
	       
	Returns
	----------
	result: (sz,sz) array
	        The shifted image.
	"""
	"""maxTargCoords = np.where(target==np.max(target))
	maxPSFCoords = np.where(PSF==np.max(PSF))
	maxTargCoords = np.array([maxTargCoords[0][0],maxTargCoords[1][0]])
	maxPSFCoords = np.array([maxPSFCoords[0][0],maxPSFCoords[1][0]])
	xShift = maxTargCoords[1]-maxPSFCoords[1]
	yShift = maxTargCoords[0]-maxPSFCoords[0]
	init = np.roll(np.roll(PSF,yShift,axis=0),xShift,axis=1)
	chis = []
	arrays = []
	el = 0
	scanSize = 20
	xShift = 0
	yShift = 0
	maxIter = 300
	minFound = 0
	ii = 0
	newArray = init.copy()
	currentChi2 = np.sum((init[maxTargCoords[0]-2:maxTargCoords[0]+3,maxTargCoords[1]-2:maxTargCoords[1]+3]-target[maxTargCoords[0]-2:maxTargCoords[0]+3,maxTargCoords[1]-2:maxTargCoords[1]+3])**2)
	stepCount = 0
	while minFound==0 and ii<maxIter:
		previousChi2 = currentChi2
		previousShift = np.array([xShift,yShift])
		previousArray = newArray.copy()
		xShift += 0.2*np.random.random()-0.1
		yShift += 0.2*np.random.random()-0.1
		newArray = tools.subpixel_roll(newArray,[yShift,xShift])
		currentChi2 = np.sum((newArray[maxTargCoords[0]-2:maxTargCoords[0]+3,maxTargCoords[1]-2:maxTargCoords[1]+3]-target[maxTargCoords[0]-2:maxTargCoords[0]+3,maxTargCoords[1]-2:maxTargCoords[1]+3])**2)
		if currentChi2<previousChi2:
			stepCount+=1
		else:
			stepCount = 0
			xShift = previousShift[0]
			yShift = previousShift[1]
			currentChi2 = previousChi2
			newArray = previousArray.copy()
		if stepCount>5:
			minFound = 1
		ii+=1
	result = newArray.copy()"""
	rad = 3
	sz = image.shape[0]
	maxCoords = np.where(image==np.max(image))
	maxRad = np.sqrt((maxCoords[0][0]-sz//2)**2+(maxCoords[1][0]-sz//2)**2)
	while maxRad>sz//2-2*rad:
		image[maxCoords[0][0],maxCoords[1][0]] = 0
		maxCoords = np.where(image==np.max(image))
		maxRad = np.sqrt((maxCoords[0][0]-sz//2)**2+(maxCoords[1][0]-sz//2)**2)
	zoomIn = image[maxCoords[0][0]-rad:maxCoords[0][0]+rad,maxCoords[1][0]-rad:maxCoords[1][0]+rad]
	newSize = zoomIn.shape[0]
	x1 = np.linspace(0,newSize-1,newSize)
	y1 = np.linspace(0,newSize-1,newSize)
	x = np.meshgrid(x1,y1)[0].reshape(newSize**2)
	y = np.meshgrid(x1,y1)[1].reshape(newSize**2)
	m = polyfit2d(x,y,zoomIn.reshape(newSize**2),2)
	xMax = -m[3]/(2*m[6])+maxCoords[1][0]-rad
	yMax = -m[1]/(2*m[2])+maxCoords[0][0]-rad
	xShift = sz//2-xMax
	yShift = sz//2-yMax
	result = tools.subpixel_roll(image,[yShift,xShift])
	return result

def normalise(array):
	"""Make an array have a maximum value of 1.
	
	Parameters
	----------
	array: (sz,sz) array
	       The input array.
	       
	Returns
	----------
	result: (sz,sz) array
	        The normalised array.
	"""
	maxArray = np.max(array)
	result = array/maxArray
	return result
	
def weight_new(weights,stepsize,unused):
	"""Create a new weight vector by adding a random number multiplied by adding
	   a stepsize to each weight and normalising.
	
	Parameters
	----------
	weights: (nPSFs) array
	         The current weight vector.
	stepsize: float
	          The step size which is multiplied by the random number.
	unused: array
	        The indices of weights that stay zero.
	
	Returns
	----------
	new_weights: (nPSFs) array
	             The new weight vector.
	"""
	new_weights = np.zeros(len(weights))
	for kk in range(0,len(weights)):
		if not kk in unused:
			new_weights[kk] = weights[kk]+stepsize*np.random.normal()
		if new_weights[kk]<0:
			new_weights[kk] = 0
	new_weights/=np.sum(new_weights)
	return new_weights

def interpolate(targets,PSFs,pas,maxIterations=10000,niter=200):
	"""Calculate contrast ratio by interpolating between PSFs.
	
	Parameters
	----------
	targets: (nTargets,sz,sz) array
	         The target cube.
	PSFs: (nPSFs,sz,sz) array
	      The calibrator cube.
	pas: (nTargets) array
	     The position angles of the targets.
	maxIterations: int
	               The maximum number of steps in the minimisation.
	niter: int
	       The number of continuation steps after minimum has been found.
	       
	Returns
	----------
	allCrats:  (nTargets,sz,sz)
	           The cube of contrast ratio maps with north up and east left
	"""
	allCrats = np.zeros(targets.shape)
	badCrats = np.zeros(targets.shape)
	for nFrame in range(0,targets.shape[0]):
		print(nFrame)
		tgtIm = targets[nFrame]
		tgtIm = coarse_tilt(tgtIm)
		tgtIm = sub_background(tgtIm)
		tgtIm = normalise(tgtIm)
		tgtIm = tilt(tgtIm)
		#Calculate which PSF gives the smallest chi2
		chi2s = np.zeros(len(PSFs))
		sz = PSFs.shape[1]
		for ii in range(0,len(chi2s)):
			PSFs[ii] = coarse_tilt(PSFs[ii])
			#PSFs[ii] = sub_background(PSFs[ii])
			PSFs[ii] = normalise(PSFs[ii])
			PSFs[ii] = tilt(PSFs[ii])
			chi2s[ii] = np.sum(chiSquared(tgtIm,PSFs[ii]))

		weights = np.zeros(len(PSFs))
		weights[np.argmin(chi2s)] = 1
		PSF = PSFs[np.argmin(chi2s)]
		diff = tgtIm-PSF
		badCrats[nFrame] = conv.cross_corr(PSF,diff)
		badCrats[nFrame]/=np.sum(PSF**2)
		badCrats[nFrame] = nd.rotate(badCrats[nFrame],-pas[nFrame],reshape=False)
		currentChi2 = np.min(chi2s)
		allChis = []
		allChis.append([weights,currentChi2])
		burnedIn = 0
		ii = 0
		plotVals = []
		derivCombo = 0
		stepCounter = 0
		unused = chi2s.argsort()[-3:]
		while burnedIn==0:
			#Save the previous value of chi2
			previousChi2 = currentChi2
			#Calculate new weights and chi2
			w_new = weight_new(weights,0.01,unused)
			combination = np.sum(w_new[kk]*PSFs[kk] for kk in range(0,len(w_new)))
			combination = coarse_tilt(combination)
			#combination = sub_background(combination)
			combination = normalise(combination)
			combination = tilt(combination)
			currentChi2 = chiSquared(tgtIm,combination)
			allChis.append([weights,currentChi2])
			#Check if chi2 has improved.  If it has, keep result.
			if currentChi2<previousChi2:
				weights = w_new.copy()
				stepCounter = 0
			else:
				currentChi2 = previousChi2
				stepCounter+=1
			if stepCounter>=300 or ii>maxIterations:
				burnedIn = 1
			plotVals.append(currentChi2)
			ii+=1
		goodPSFs = np.zeros((niter,PSFs.shape[1],PSFs.shape[2]))
		goodCrats = np.zeros((niter,PSFs.shape[1],PSFs.shape[2]))
		#Continue iterating to get alternative values and contrast maps
		for ii in range(0,niter):
			previousChi2 = currentChi2
			w_new = weight_new(weights,0.01,unused)
			combination = np.sum(w_new[kk]*PSFs[kk] for kk in range(0,len(w_new)))
			combination = coarse_tilt(combination)
			#combination = sub_background(combination)
			combination = normalise(combination)
			combination = tilt(combination)
			#Calculate contrast ratio
			diff = tgtIm-combination
			goodCrats[ii] = conv.cross_corr(combination,diff)
			goodCrats[ii]/=np.sum(combination**2)
			currentChi2 = chiSquared(tgtIm,combination)
			allChis.append([weights,currentChi2])
			if currentChi2<previousChi2:
				weights = w_new.copy()
			else:
				currentChi2 = previousChi2
		finalCrats = stats.weighted_mean(goodCrats)
		finalCrats = nd.rotate(finalCrats,-pas[nFrame],reshape=False)
		allCrats[nFrame] = finalCrats
	return allCrats