"""
EisVAE: Deep clustering in subglacial radar reflectance reveals subglacial lakes

pick_train_data_from_radar_images/pick_data_from_IPR_images.py: Pick the ice bottom reflections from a set of radar images (From CReSIS dataset).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
from scipy.io import loadmat

# Define a function to smooth reflector waveforms using a Gaussian filter
def curveSmooth(curve, sigma=1):
	curve_smooth = sn.gaussian_filter1d(curve,sigma)
	return curve_smooth

# Define a function to normalize waveforms to the range [0, 1]
def normlize(curve):
	return (curve-np.min(curve))/(np.max(curve)-np.min(curve))


# Set the number of points used in the analysis and initialize a list to store reflector data
usedPix = 64
reflectorSet = []

# Flag to control whether to plot previews of the data processing
PLOT_PREVIEW = True
# Create a directory for data previews if it doesn't exist and PLOT_PREVIEW is True
if PLOT_PREVIEW and not os.path.exists('./data_previews'):
	os.mkdir('./data_previews')

# Specify the directory containing the data files
workPath = '../demo_data_from_AGAP-S_CRESIS/'

# Extract reflector waveforms file by file from the specific directory
fileList = os.listdir(workPath)
for fileName in fileList:
	if fileName[-4:] == '.mat':
	
		fileFullName = os.path.join(workPath, fileName)
		print(fileFullName)
		
		fr = loadmat(fileFullName)
		
		testData = np.array(fr['Data'])
		manualSurface = np.squeeze(fr['Surface'])
		manualBottom = np.squeeze(fr['Bottom'])
		
		timeRec = np.squeeze(fr['Time'])
		
		dt = timeRec[2] - timeRec[1]
		
		
		testData = np.log(testData)
		testData = (testData - np.min(testData))/(np.max(testData) - np.min(testData))

		outputMat = testData.copy()
		outputMatModi = testData.copy()
		
		searchCorRate = 0.5
		
		for xInd in np.arange(len(manualBottom)):
			
			if not np.isnan(manualSurface[xInd]+manualBottom[xInd]):
				yInd = int((manualSurface[xInd]+manualBottom[xInd])/dt) + 50
				
				upper = yInd - usedPix
				lower = yInd + usedPix
				
				sampleData = outputMat[yInd - int(usedPix*searchCorRate):yInd + int(usedPix*searchCorRate), xInd]
				
				
				sampleData = normlize(curveSmooth(sampleData,4))
				
				outputMat[:upper, xInd] = np.nan
				outputMat[lower:, xInd] = np.nan
		#		print(np.argmax(sampleData))
				shift = np.argmax(sampleData) - int(usedPix*searchCorRate)
				
				yInd += shift
				upper = yInd - usedPix
				lower = yInd + usedPix
				
				outputMatModi[:upper, xInd] = np.nan
				outputMatModi[lower:, xInd] = np.nan
				
				finalTrace = normlize(curveSmooth(outputMatModi[upper:lower, xInd],4))
				outputMatModi[upper:lower, xInd] = finalTrace
				
				reflectorSet.append(finalTrace)
				
				
			else:
				outputMat[:, xInd] = np.nan
				outputMatModi[:, xInd] = np.nan
				
		# If enabled, plot previews of the data processing results
		if PLOT_PREVIEW:
		
			plt.figure(figsize=[10,10])
			
			plt.subplot(311)
			
			plt.imshow(testData, aspect='auto', cmap='gray')
			
			#plt.plot(bedRockLayerSet[:, 1], bedRockLayerSet[:, 0], c='red', label='EisNet picked Bedrock',lw=2, alpha=.5)
			plt.plot(np.arange(len(manualBottom)), (manualSurface+manualBottom)/dt, c='navy', label='Picked Bedrock',lw=2, alpha=.5, linestyle='--')
			
			plt.legend()
			
			plt.subplot(312)
			plt.imshow(outputMat, aspect='auto', cmap='gray')
			plt.subplot(313)
			
			plt.imshow(outputMatModi, aspect='auto', cmap='gray')
			
			plt.savefig('./data_previews/'+fileName+'.pdf', dpi=300)
			
			plt.close()
	


# Save the collected reflector data to an HDF5 file
reflectorSet = np.array(reflectorSet)
with h5py.File('reflector_extracted_from_IPR.h5', 'w') as fw:
	fw.create_dataset('data', data=reflectorSet)
print(reflectorSet.shape)

# Plot and save a demonstration of all the reflector waveforms picked
plt.figure(figsize=[5, 5])
for ii in range(100):
	randInd = np.squeeze(np.random.randint(0, reflectorSet.shape[0], size=1))
	plt.plot(reflectorSet[randInd, :], c='k', alpha=.1)
plt.savefig('interface_all.pdf', dpi=300)
