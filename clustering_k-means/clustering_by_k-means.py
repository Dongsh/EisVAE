"""
EisVAE: Deep clustering in subglacial radar reflectance reveals subglacial lakes

clustering_k-means/clustering_by_k-means.py: Apply K-means clustering on the encoded radar reflections.
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import h5py

from sklearn.cluster import KMeans
# Used for saving and loading models
import joblib

import matplotlib.colors as colors
import matplotlib.cm as cmx

# Set numbers of clustering
n_cluster = 15

# Define color map for visualization
cmapAll = 'inferno'
# Generate color map for clusters
colors_map = plt.get_cmap(cmapAll)
cNorm = colors.Normalize(vmin=0, vmax=n_cluster)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colors_map)

# Load the trained encoder model
encoder = keras.models.load_model('../VAE_train/trained_models_for_reference/encoder.h5')
batch_size = 20

# Load data from file
with h5py.File('./random_selected_reflectors_for_clustering.h5','r') as data_file:
	x_test = np.array(data_file.get('data'))
	
# Encode data using the encoder
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
x_test_encoded = np.array(x_test_encoded, dtype=float)


km = KMeans(n_clusters=n_cluster, random_state=22).fit(x_test_encoded)  

# Save the trained KMeans model
joblib.dump(km , 'KmeanK'+str(n_cluster)+'_model.pkl')

# Obtain cluster labels and centers
cluster = km.labels_
centers = np.zeros([n_cluster, 2])

# Calculate centers for each cluster
for ni in range(n_cluster):
	encoderInNClass = x_test_encoded[cluster==ni]
	centers[ni,0] = np.mean(encoderInNClass[:,0])
	centers[ni,1] = np.mean(encoderInNClass[:,1])
	
# Plotting the clustering result
plt.figure(figsize=(5,5))
ax=plt.subplot(111)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2, c=km.labels_,cmap=cmapAll)

# Plot the cluster centers
plt.scatter(centers[:,0],centers[:,1], linewidths=1.5, marker='+', s=100, c='black', alpha=.7)

# Plot the grid range of synthetic reflectors
rect = plt.Rectangle((-1.5,-1.5), 3, 3, fill=False, edgecolor = 'gray',linewidth=2, linestyle='--',alpha=.5)
ax.add_patch(rect)

# Set the limits and labels of axes
ax.set_xlim([-3,3])
ax.set_ylim([-2.5,2.5])
ax.set_xlabel('$Z_1$')
ax.set_ylabel('$Z_2$')


# Save the plot of encoded and clustered data
plt.savefig('encoded_and_clustered_reflectors_K'+str(n_cluster)+'.pdf')
plt.close()

# Load the trained generator model
generator = keras.models.load_model('../VAE_train/trained_models_for_reference/generator.h5')

# Number of points for grid
nPoint = 15
# Create grid points
grid_x = np.linspace(-1.5, 1.5, nPoint)
grid_y = np.linspace(-1.5, 1.5, nPoint)

plt.figure(figsize=(10, 10))

# Iterate through grid points, generate, and plot waveforms
for ii, xi in enumerate(grid_x):
	for jj, yj in enumerate(grid_y):
		z_sample = np.array([[yj, xi]])
		x_decoded_test = generator.predict(z_sample)
		
		stf_gen = np.squeeze(x_decoded_test)
		
		ax = plt.subplot(nPoint,nPoint, (nPoint-jj-1)*nPoint + ii+1)
		classLabel = km.predict([[xi, yj]])
		ax.plot(stf_gen, c=scalarMap.to_rgba(classLabel)[0])
		ax.axis('off')

# Save the plot of clustered waveforms
plt.savefig('clustered_waveforms_K'+str(n_cluster)+'.pdf', dpi=300)

