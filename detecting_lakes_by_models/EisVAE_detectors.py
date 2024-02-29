"""
EisVAE: Deep clustering in subglacial radar reflectance reveals subglacial lakes

detecting_lakes_by_models/EisVAE_detectors.py: Use the trained encoder and saved K-means model to detect subglacial lakes from a set of radar image.
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import h5py
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.cluster import KMeans
import joblib
from geopy.distance import geodesic
import scipy.ndimage as sn
from scipy.io import loadmat


# Function to apply Gaussian filter for curve smoothing
def curveSmooth(curve, sigma=1):
	curve_smooth = sn.gaussian_filter1d(curve,sigma)
	return curve_smooth

# Function to normalize the curve
def normlize(curve):
	return (curve-np.min(curve))/(np.max(curve)-np.min(curve))



# Parameters initialization
usedPix = 64
testMode = False
nanThreshold = 9
minLakeLen = 8
n_sigma = .5
n_cluster = 15

# Define save path of preview figures and lake ranges
encodeFigPath = './result_figures/'
encodeH5Path = './result_h5/'


if not os.path.exists(encodeFigPath):
	os.mkdir(encodeFigPath)
	
if not os.path.exists(encodeH5Path):
	os.mkdir(encodeH5Path)

# Color map setting
cmapAll = 'inferno'
colors_map = plt.get_cmap(cmapAll)
cNorm = colors.Normalize(vmin=0, vmax=n_cluster)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colors_map)

# Load the pre-trained encoder model
encoder = keras.models.load_model('../VAE_train/trained_models_for_reference/encoder.h5')
batch_size = 20


# Load h5 file that saves the linear relationship of reflection power and depth
with h5py.File('slope_refl_power_and_depth.h5', 'r') as fr:
	meanPower = np.array(fr['mean'])
	slopePower = np.array(fr['slopeYY'])
	stdPower = np.array(fr['std'])
	
# Load KMeans model
km = joblib.load('../clustering_k-means/save_model_clustering/KmeanK15Model.pkl')


# Specify the directory containing the data files
workPath = '../demo_data_from_AGAP-S_CRESIS/'

# Process each file in the directory
fileList = os.listdir(workPath)
for fileName in fileList:
	if fileName[-4:] == '.mat':

		maxFinalTraceRawSet = []
		fileFullName = os.path.join(workPath, fileName)
		print(fileFullName)
		
		try:
			fr = loadmat(fileFullName)
			
			latSet = np.squeeze(fr['Latitude'])
			lonSet = np.squeeze(fr['Longitude'])
			distAll = geodesic((latSet[0], lonSet[0]), (latSet[-1], lonSet[-1])).km

			latLonName = '(%.4f)'%np.mean(latSet)+',(%.4f).' % np.mean(lonSet)
			

			manualBottom = np.squeeze(fr['Bottom'])
			depthData = np.squeeze(fr['Depth'])
			manualBottomCheck = manualBottom.copy()
			manualBottomCheck[np.isnan(manualBottomCheck)] = 0
			if np.sum(manualBottomCheck) == 0:
				print('[Skip] No Manual Pick Found')
				continue
			#data = np.array(fr['Data'])
			testData = np.array(fr['Data'])
			manualSurface = np.squeeze(fr['Surface'])
			manualElevation = np.squeeze(fr['Elevation'])
			manualBottom = manualBottom + manualSurface
			timeRec = np.squeeze(fr['Time'])
		except NotImplementedError:
			with h5py.File(fileFullName, 'r') as fr:
				
				
				manualBottom = np.squeeze(fr['Bottom'])
				manualBottomCheck = manualBottom.copy()
				manualBottomCheck[np.isnan(manualBottomCheck)] = 0
				if np.sum(manualBottomCheck) == 0:
					print('[Skip] No Manual Pick Found')
					continue
				
				testData = np.array(fr['Data']).T
				timeRec = np.squeeze(fr['Time'])
				manualSurface = np.squeeze(fr['Surface'])
				manualElevation = np.squeeze(fr['Elevation'])
		
		dt = timeRec[2] - timeRec[1]
		if np.max(testData) ==  np.min(testData):
			print('[Skip] No ice bottom data Found.')
			continue
		

		testData = np.log(testData) * 10
		testData[np.isnan(testData)] = 0

		trBottomDepth = -(manualBottom)/dt*(depthData[1]-depthData[0])
				
		outputMat = testData.copy()

		outputMatModi = (testData - np.min(testData))/(np.max(testData) - np.min(testData))
		classResult = []
		
		searchCorRate = 0.5
		maxValueSet = []
		trDepSet = []
		
		fTraceSet = []
		classResultSet = []
		
		nonNanIndxSet = []
		shiftSet = []
		
		# Scan the subglacial lakes trace by trace in reflectors
		for xInd in np.arange(len(manualBottom)):

			if not np.isnan(manualBottom[xInd]):
				yInd = int(manualBottom[xInd]/dt) + 50 
				
				upper = yInd - usedPix
				lower = yInd + usedPix
				
				sampleData = outputMat[yInd - int(usedPix*searchCorRate):yInd + int(usedPix*searchCorRate), xInd]
				
				maxValueInTr = np.max(sampleData)
				maxValueSet.append(maxValueInTr)
				
				trDepSet.append(trBottomDepth[xInd])
				
				sampleData = normlize(curveSmooth(sampleData,4))
				
				outputMat[:upper, xInd] = np.nan
				outputMat[lower:, xInd] = np.nan
				shift = np.argmax(sampleData) - int(usedPix*searchCorRate)
				shiftSet.append(shift)
				yInd += shift
				upper = yInd - usedPix
				lower = yInd + usedPix
				
				outputMatModi[:upper, xInd] = np.nan
				outputMatModi[lower:, xInd] = np.nan
				
				maxFinalTraceRaw = np.max(outputMatModi[upper:lower, xInd])
				maxFinalTraceRawSet.append(maxFinalTraceRaw)
				finalTrace = normlize(curveSmooth(outputMatModi[upper:lower, xInd],4))
				outputMatModi[upper:lower, xInd] = finalTrace
				
				finalTrace = np.squeeze(finalTrace)
				finalTrace[np.isnan(finalTrace)] = 0
				
				fTraceSet.append(finalTrace)
				nonNanIndxSet.append(xInd)
				
			else:
				outputMat[:, xInd] = np.nan
				outputMatModi[:, xInd] = np.nan

				maxValueSet.append(0)
				maxFinalTraceRawSet.append(0)
				shiftSet.append(0)
				trDepSet.append(0)
				
		fTraceSet = np.array(fTraceSet)	
		
		x_decoded_test = encoder.predict(fTraceSet, verbose=1)
		x_decoded_test = np.squeeze(x_decoded_test)
		x_decoded_test = np.array(x_decoded_test, dtype=float)
		classResult = km.predict(x_decoded_test)
		
		classResultWithNan = np.zeros(len(manualBottom)) * np.nan
		classResultWithNan[nonNanIndxSet] = classResult
		
		classResult = classResultWithNan
		
		classResult = np.array(classResult)
		maxFinalTraceRawSet = np.array(maxFinalTraceRawSet)
		
		modiButtom = manualBottom + np.array(shiftSet)*dt
		
		subLakePicked = np.zeros(len(maxFinalTraceRawSet)) * np.nan
		nanFlag = True
		nanLen = 0
		
		nanStartInd = 0
		nanEndInd = 0
		
		
		def jugSubIceLake(ii):
			if classResult[ii] < 0.5 and classResult[ii] > -0.5 and maxValueSet[ii] > slopePower*(-trDepSet[ii])+meanPower+ stdPower * n_sigma:
				return True
			else:
				return False
			
		for ii in range(len(maxFinalTraceRawSet)):
			
			if jugSubIceLake(ii):
									
				if ii - nanStartInd <= nanThreshold:
					subLakePicked[nanStartInd:ii] = 2
					
				nanStartInd = ii
				
		for ii in range(len(maxFinalTraceRawSet)):
			
			if jugSubIceLake(ii):
			
				subLakePicked[ii] = 1
				
				
		nanStartInd = 0
		nanEndInd = 0
				
		for ii in range(1,len(maxFinalTraceRawSet)):
			
			if subLakePicked[ii] > 0.5:
				
				if np.isnan(subLakePicked[ii-1]):
					nanStartInd = ii-1
					
					
			else:
				
				if subLakePicked[ii-1] > 0.5:
					nanEndInd = ii

					if testMode:
						print('Lake Range:', nanStartInd*distAll/len(manualBottom), nanEndInd*distAll/len(manualBottom))
						
						
					if (nanEndInd - nanStartInd) <= minLakeLen:
						subLakePicked[nanStartInd:nanEndInd] = np.nan
						if testMode:
							print('length rejected!')
					else:

						
						
						testTraceTemp = subLakePicked[nanStartInd:nanEndInd]
						testTraceTemp = testTraceTemp[~np.isnan(testTraceTemp)]
						if np.sum(testTraceTemp) > (nanEndInd - nanStartInd)*1.5:
							if testMode:
								print('interp. rejected!')
							subLakePicked[nanStartInd:nanEndInd] = np.nan


						peakValueArr = maxValueSet[nanStartInd:nanEndInd]
						peakDepthArr = trDepSet[nanStartInd:nanEndInd]
						
						meanPeakValue = np.mean(peakValueArr)
						meanDepth = np.mean(peakDepthArr)

						if meanPeakValue <= slopePower*(-meanDepth)+meanPower+ stdPower * (n_sigma):
							if testMode:
								print('mean ref. power rejected:', meanPeakValue, slopePower*(-meanDepth)+meanPower+ stdPower * (n_sigma))
							subLakePicked[nanStartInd:nanEndInd] = np.nan

						
		if (len(maxFinalTraceRawSet)-1 - nanStartInd) <= minLakeLen:
			subLakePicked[nanStartInd:len(maxFinalTraceRawSet)-1] = np.nan
			print('length rejected 2')
		

		subLakePickedNonZero = subLakePicked.copy()
		subLakePickedNonZero[np.isnan(subLakePickedNonZero)] = 0
		
		# if subglacial lake detected, save the lake ranges to h5 file
		
		if np.sum(subLakePickedNonZero) > 1:
				
			h5FileName = os.path.join(encodeH5Path, fileName +'.h5' )
			with h5py.File(h5FileName, 'w') as fw:
				fw.create_dataset('lake', data=subLakePicked)
				fw.create_dataset('buttom', data=manualBottom)
				fw.create_dataset('surface', data=manualSurface)
			
			
		# plot preview image
		plt.figure(figsize=[8,8])
		
		plt.subplot(411)
		plt.gca().set_ylabel('Depth (km)')
		plt.imshow(testData, aspect='auto', cmap='gray', extent=[0, distAll, -depthData[-1]/1e3*1.68/3 ,depthData[0]/1e3*1.68/3])

		plt.plot(np.linspace(0, distAll, len(manualBottom)), -(manualBottom)/1e3*1.68/3/dt*(depthData[1]-depthData[0]), c='navy', label='Labeled Reflection',lw=1, alpha=.5, linestyle='--')

		plt.gca().axes.xaxis.set_ticklabels([])

		plt.gca().set_ylim([-4.8,-1.5])
		plt.subplot(412)
		plt.gca().set_ylabel('Depth (km)')
		plt.imshow(outputMatModi, aspect='auto', cmap='gray',extent=[0, distAll, -depthData[-1]/1e3*1.68/3 ,depthData[0]/1e3*1.68/3])
		
		plt.gca().axes.xaxis.set_ticklabels([])
		plt.gca().set_ylim([-4.8,-1.5])
		plt.subplot(413)
		plt.gca().set_ylabel('Depth (km)')
		
		testData = (testData - np.min(testData))/ (np.max(testData) - np.min(testData))
		plt.imshow(testData, aspect='auto', cmap='gray', extent=[0, distAll, -depthData[-1]/1e3*1.68/3 ,depthData[0]/1e3*1.68/3], vmax=0.95*np.max(testData), vmin=0.4*np.max(testData))
		
		plt.scatter(np.linspace(0, distAll, len(manualBottom)), -(manualBottom)/1e3*1.68/3/dt*(depthData[1]-depthData[0])+.7, c=scalarMap.to_rgba(classResult), label='Picked Bedrock',s=2)
		
		plt.gca().set_xlim([0, distAll])
		plt.gca().axes.xaxis.set_ticklabels([])
		plt.gca().set_ylim([-4.8,-1.5])

		plt.subplot(8,1,7)
	
		plt.imshow([subLakePicked], aspect='auto', cmap='inferno', extent=[0, distAll, 0, 1])
		plt.yticks([])
		
		plt.gca().set_xlabel('Distance(km)')
		if np.sum(subLakePickedNonZero) > 1:
			plt.savefig(os.path.join(encodeFigPath, '+.'+latLonName+fileName+'.pdf'), dpi=600)
			print('+')
			
		else:
			plt.savefig(os.path.join(encodeFigPath, '-.'+latLonName+fileName+'.pdf'), dpi=600)
			print('-')
	
		plt.close()
			
		