# Import the modules
import cv2
from skimage.feature import hog
import numpy as np
from PIL import Image, ImageDraw
from operator import itemgetter

def determine_dominant_color(infile, numcolors=4, swatchsize=20, resize=150):

    image = Image.open(infile).convert('LA')
    result = image.resize((resize, resize))
    #result = image.convert('P', palette=Image.ADAPTIVE, colors=numcolors)
    result.putalpha(0)
    colors = result.getcolors(resize*resize)

    # Find the color intensity with biggest occurence
    m_color = max(colors, key=itemgetter(0))[1]
    del image
    return m_color

def draw_bounding_box(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]
    # cv2.drawContours(image, contours, largest_contour_index, (255, 0, 0), 7)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def detect_characters(image, filepath):

    '''

    :param image: Input image
    :param filepath: need this to get colors with PIL
    :return: list of regions of interest, i.e. characters that will later be fed to the classifier
    '''

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    cv2.imshow("Gray Image with Rectangular ROIs", im_gray)
    cv2.waitKey()

    # find dominant color to determine the background (I cannot use my file naming because it won't work in production)
    file_bg = determine_dominant_color(filepath)

    if file_bg[0] > 100:
        v = np.median(im_gray)
        sigma = 0.33  # 0.33

        # ---- apply optimal Canny edge detection using the computed median----
        lower_threshold = int(max(0, (1.0 - sigma) * v))
        upper_threshold = int(min(255, (1.0 + sigma) * v))

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    else:
        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im_th = cv2.bitwise_not(im_th)
    #ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Thresholded Image with Rectangular ROIs", im_th)
    cv2.waitKey()

    # Find contours in the image
    cim, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("After contouring Image with Rectangular ROIs", cim)
    cv2.waitKey()

   # for ctr in ctrs:
    #    draw_bounding_box(image, ctr)

    #cv2.imshow("Resulting Image with Rectangular ROIs", image)
    #cv2.waitKey()

    ROIs = []
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.

    # First sort the contours from left to right


    for ctr in ctrs:
        # Draw the rectangles
        #draw_bounding_box(image, ctr)

        x, y, w, h = cv2.boundingRect(ctr)

        # Crop the character
        roi = im_gray[y:y+h, x:x+w]

        #cv2.imshow("ROI before resizing", roi)
        #cv2.waitKey()
        # Resize the image
        roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)

        # add margin
        roi = cv2.copyMakeBorder(
            roi,
            40,
            40,
            40,
            40,
            cv2.BORDER_CONSTANT,
            value=[file_bg[0], file_bg[0], file_bg[0]])

        roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)

        cv2.imshow("ROI after resizing", roi)
        cv2.waitKey()
        roi = cv2.dilate(roi, (3, 3))

        ROIs.append(np.array(roi))

    cv2.imshow("Resulting Image with Rectangular ROIs", image)
    cv2.waitKey()

    # PROBLEM. THESE COUNTOURS ARE IN RANDOM ORDER

    return ROIs


