# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import random
import os

# Load the classifier
clf = joblib.load("models/characters.pkl")

path = '../DR_data/vins'
file = random.choice(os.listdir(path))

file = '6TNEF59347P425193_verdana.ttf225.png'
# Read the input image
im = cv2.imread(path + '/' + file)

def draw_bounding_box(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]
    # cv2.drawContours(image, contours, largest_contour_index, (255, 0, 0), 7)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_bounding_box_rect(image, rect):
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    roi = image[y:y + h, x:x + w]
    # cv2.drawContours(image, contours, largest_contour_index, (255, 0, 0), 7)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

cv2.imshow("Resulting Image with Rectangular ROIs", im_gray)
cv2.waitKey()

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Resulting Image with Rectangular ROIs", im_th)
cv2.waitKey()

# Find contours in the image
cim, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Resulting Image with Rectangular ROIs", cim)
cv2.waitKey()

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for ctr in ctrs:
    draw_bounding_box(im, ctr)


cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

vin = []
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for ctr in ctrs:
    # Draw the rectangles
    draw_bounding_box(im, ctr)

    x, y, w, h = cv2.boundingRect(ctr)

    # Make the rectangular region around the digit
    leng = int(w * 1.6)
    pt1 = int(y + w // 2 - leng // 2)
    pt2 = int(x + y // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    vin.append(nbr)

vins = np.array(vin)
''.join([str(e) for e in vins])
print(vins)
print(file[:17])

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()