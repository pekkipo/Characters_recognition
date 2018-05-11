# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import random
import os
from helpers import *
from sklearn.cluster import KMeans
#from NN_classfier import create_model
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
#model = create_model()

def create_model():
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(64, (5, 5), padding="same", input_shape=(128, 128, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(30, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

model=create_model()
model.load_weights(filepath='models/NN_model.hdf5')

# Load the classifier
#clf = joblib.load("models/characters.pkl")

path = '../DR_data/vins'
file = random.choice(os.listdir(path))

file = '1AYEN45963S374568_Agane_light.ttf245.png'

#file = '6TNEF59347P425193_verdana.ttf225.png'
# Read the input image
im = cv2.imread(path + '/' + file)

cv2.imshow("Original Image with Rectangular ROIs {}".format(file), im)
cv2.waitKey()



'''
 VIN CONTAINS 17 numbers
    letters are capital
    1 number
    4 letters
    5 numbers
    1 letter
    6 numbers
    

Perhaps can tran two models for numbers and letters but for now won't do that
number_positions = [0, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
letters_positions = [1, 2, 3, 4, 10]


'''

vin = []
ROIs = detect_characters(im, path + '/' + file)
for roi in ROIs:
    roi = np.expand_dims(roi, axis=0)  # need this if I want to predict on a single image
    prediction = model.predict(roi)
    vin.append(prediction)

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']

vins = np.array(vin)
''.join([str(e) for e in vins])
print(vins)
vin_string = ''
for vin in vins:
    for pred_list in vin:
        for index, pred in enumerate(pred_list):
            if int(pred) == 1:
                predicted_value = classes[index]
                vin_string += predicted_value
                break


print(vin_string)
print(file[:17])

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()



