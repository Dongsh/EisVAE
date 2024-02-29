import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('dataset_for_vae_training.h5', 'r') as fr:
	traceAll = np.array(fr['data'])
	
plt.figure(figsize=[7,7])
for trace in traceAll:
	plt.plot(range(-64,64), trace, c='k', alpha=.05)
	
plt.show()
	

	