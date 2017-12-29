from keras.models import Model
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

import os

import sklearn
from sklearn.model_selection import train_test_split

#################################################################

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)

    return lines

def loadImagesAndMeasurements(dataPath):
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerImages = []
    leftImages = []
    rightImages = []
    measurementsImages = []
    for directory in dataDirectories:
        lines = getLinesFromDrivingLogs(directory, skipHeader=True)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerImages.extend(center)
        leftImages.extend(left)
        rightImages.extend(right)
        measurementsImages.extend(measurements)

    return (centerImages, leftImages, rightImages, measurementsImages)

def combineAllImagesAndMeasurements(centerImages, leftImages, rightImages, measurement, correction):
    images = []
    images.extend(centerImages)
    images.extend(leftImages)
    images.extend(rightImages)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])

    return (images, measurements)

################################################################

# Reading images (center, right, left) and measurements
centerImages, leftImages, rightImages, measurements = loadImagesAndMeasurements('data')
images, measurements = combineAllImagesAndMeasurements(centerImages, leftImages, rightImages, measurements, 0.2)
print('Total Images: {}'.format( len(images)))

# Splitting samples
samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

numBins = 20
avgSamplesPerBin = len(measurements)/numBins
hist, bins = np.histogram(measurements, numBins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.title('angle/measurement data histogram')
plt.ylabel('samples')
plt.xlabel('angles')
plt.bar(center, hist, align='center', width=width)
plt.show()

#################################################################

loss = [0.24500790052019336, 0.085471181150471337, 0.050309754540693905, 0.036409158932793985, 0.030197890161313631, 0.027052960753476397]
validationLoss = [0.12149540784570598, 0.061850274041177621, 0.041372104275590009, 0.033105998999570126, 0.02857181826650429, 0.028098583771499149]

### plot the training and validation loss for each epoch
plt.plot(loss)
plt.plot(validationLoss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.show()
