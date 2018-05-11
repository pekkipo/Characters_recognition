import random
import os
from helpers import *
from keras.models import model_from_json


# load json and create model
json_file = open('saved_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("saved_models/CNN_model.h5")

path = '../DR_data/vins'
#file = random.choice(os.listdir(path))

file = '1AYEN45963S374568_Agane_light.ttf245.png'

# file = '6TNEF59347P425193_verdana.ttf225.png'
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

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']

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



