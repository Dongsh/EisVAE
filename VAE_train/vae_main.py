"""
EisVAE: Deep clustering in subglacial radar reflectance reveals subglacial lakes

VAE_train/vae_main.py: Train Variational Auto-Encoder (VAE) from the picked data set of radar reflections
"""


import tensorflow as tf
import h5py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks


# Set TensorFlow log level to minimize unnecessary information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Select GPU for training (if necessary)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data from an HDF5 file
with h5py.File('../pick_train_data_from_radar_images/dataset_for_vae_training.h5','r') as data_file:
	data = np.array(data_file.get('data')) #[:1488600]

# Print data dimension information
print("[info] Data Loaded: ")
print(data.shape)

# Prepare the test dataset
testSetLength = 2000
showIndList = np.zeros(testSetLength)
set_length = data.shape[0]

testData = []
for ii in range(testSetLength):
	showIndList[ii] = random.randint(0, set_length-1)
	testData.append(data[int(showIndList[ii])])

testData = np.array(testData)

x_test = tf.convert_to_tensor(testData, dtype=tf.float32)
x_train = tf.convert_to_tensor(data, dtype=tf.float32)

# Normalize the data
maxX = np.max(x_train)
minX = np.min(x_train)

x_train = (x_train)/(maxX)
x_test = (x_test)/(maxX)

# Define model parameters
batch_size = 10
original_dim = int(data.shape[1])
intermediate_dim = 128
latent_dim = 2
epochs = 4


# Build the encoder part of the VAE
x = layers.Input(shape=(original_dim,))
print('x:',x.shape)
codes = layers.Dense(intermediate_dim, activation='relu')(x)
print('codes:', codes.shape)
codes = layers.Dense(intermediate_dim, activation='relu')(codes)
print('codes:', codes.shape)
#codes = layers.Dense(intermediate_dim, activation='relu')(codes)
z_mean = layers.Dense(latent_dim)(codes)
z_log_var = layers.Dense(latent_dim)(codes)


# Define a sampling function for sampling from the latent space
@tf.function
def sampling(args):
	z_mean, z_log_var = args
	epsilon = tf.random.normal(shape=(batch_size, latent_dim))
	
	return z_mean + tf.exp(z_log_var * 0.5) * epsilon
	
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
print('z:', z.shape)

# Build the decoder part
decoder_h1 = layers.Dense(intermediate_dim, activation='relu')
decoder_h2 = layers.Dense(intermediate_dim, activation='relu')
#decoder_h3 = layers.Dense(intermediate_dim, activation='relu')
h_decoded = decoder_h1(z)
print('h_decoded:', h_decoded.shape)
h_decoded = decoder_h2(h_decoded)
print('h_decoded:', h_decoded.shape)
#h_decoded = decoder_h3(h_decoded)
decoder_outputer = layers.Dense(original_dim, activation='sigmoid')    #
x_decoded = decoder_outputer(h_decoded)
print('x_decoded:', x_decoded.shape)


# Define the loss function for the VAE
#@tf.function
def vae_loss(x, x_decoded):
	xent_loss = original_dim * keras.losses.mean_squared_error(x, x_decoded)
	kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.exp(z_log_var), axis=-1)
	return xent_loss + kl_loss


# Define the model
vae = keras.Model(x, x_decoded)

# Check different TensorFlow version (>=4 or <4) and apply different compile strategy
def get_tf_minor_version():
	version_str = tf.__version__  
	minor_version = version_str.split(".")[1]  
	return int(minor_version) 

if get_tf_minor_version() >= 4:
	vae.add_loss(vae_loss(x, x_decoded))
	vae.compile(optimizer='adam', loss=None, experimental_run_tf_function = False)
else:
	vae.compile(optimizer='rmsprop', loss=vae_loss, experimental_run_tf_function = False)

# Train the model
model_history = vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=0.1, validation_freq=1)

print("[info] VAE Finish Trainning...")

# Save the trained model
vae.save('vae.h5')

# Plot training and validation loss
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochsX = range(epochs)
plt.figure(figsize=(5, 5))
plt.plot(epochsX, loss, 'r', label='Training loss')
plt.plot(epochsX, val_loss, 'bo', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.gca().set_title('VAE: Training and Validation Loss')
#ylim=[0, max(max(loss),max(val_loss))]
plt.legend()
#	plt.show()
plt.savefig('eventSeriesVAE.pdf')
plt.close()

# Save the encoder
encoder = keras.Model(x, z_mean)
encoder.save('encoder.h5')

# Test the encoder
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c='k', s=20, alpha=.1)
plt.savefig('sample_distributions_encoded.pdf')
plt.close()


# Save the decoder (generator)
decoder_input = layers.Input(shape=(latent_dim,))
h_decoded_in_d = decoder_h1(decoder_input)
h_decoded_in_d = decoder_h2(h_decoded_in_d)
#h_decoded_in_d = decoder_h3(h_decoded_in_d)
x_decoded_in_d = decoder_outputer(h_decoded_in_d)
generator = keras.Model(decoder_input, x_decoded_in_d)
generator.save('generator.h5')

# Test the decoder by generating synthetic waveforms
nPoint = 15
plt.figure(figsize=(10, 10))
grid_x = np.linspace(np.min(x_test_encoded[:, 0]), np.max(x_test_encoded[:, 0]), nPoint)
grid_y = np.linspace(np.min(x_test_encoded[:, 1]), np.max(x_test_encoded[:, 1]) , nPoint)

z_samples = []

for ii, xi in enumerate(grid_x):
	for jj, yj in enumerate(grid_y):
		z_samples.append([yj, xi])
		
		
x_decoded_test = generator.predict(z_samples)
waveforms_gen = np.squeeze(x_decoded_test)


testID = 0
for ii, xi in enumerate(grid_x):
	for jj, yj in enumerate(grid_y):
		synthetic_waveform = waveforms_gen[testID]
		testID += 1
		
		ax = plt.subplot(nPoint,nPoint, (nPoint-jj-1)*nPoint + ii+1)

		ax.plot(synthetic_waveform, c='gray')
		ax.axis('off')
		

plt.savefig('synthetic_waveforms_by_decoder.pdf',dpi=600)


