#  UDACITY CarND P3

#Imports

import os
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

#import lines from Driving_log CSV 

lines = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#Declaring Arrays Images & Measurements, and Correction factor to Left & Right Images

images = []
measurements = []
c_factor = 0.2

#Reading Local paths for center, left, and right images along with measurements


for line in lines:	
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = "./data/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+c_factor)
	measurements.append(measurement-c_factor)

"""  
***** Code segment for Augementing images by flipping data *****

aug_images = []
aug_measurements = []

for image, measuresurement in zip(images, measurements):
	aug_images.append(image)
	aug_measurements.append(measurement)
	flipped_image = np.fliplr(image)
	flipped_measurement = -measurement
	aug_images.append(flipped_image)
	aug_measurements.append(flipped_measurement)

**** End Code segment for Augmenting data ****
"""

print("Number of Images: " + str(len(images)))
print("Number of Measurements: " + str(len(measurements)))

X_train = np.array(images)
y_train = np.array(measurements) 

print("Converted!")
print("Training Data Shape: " + str(X_train.shape))

# *** Training Model Used is similar to NVIDIA *** 
model = Sequential()

#Normalizing data set
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))

#Cropping Images
model.add(Cropping2D(cropping=((50,20),(0,0))))

#Beging Convolution Layers Folowed by Maxpooling
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='ELU'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='ELU'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='ELU'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(64,3,3,activation='ELU'))
model.add(Convolution2D(64,3,3,activation='ELU'))

# Applying Flattening, Dense & Drop Layer
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

#Using Adam optimizer with a learning rate of 1e-4, Loss Function is MSE
model.compile(optimizer=Adam(lr=(1e-4)), loss='mse')

#Using 25 percent of training data for validation set
model.fit(X_train, y_train, validation_split=0.25,shuffle=True, nb_epoch=10)

model.save('model.h5')

"""  End Program """
