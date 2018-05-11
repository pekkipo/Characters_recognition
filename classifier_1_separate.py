
import glob
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

#data_path = 'images_generation/output/characters_numbers'
data_path = '../DR_data/characters_numbers'
# data_path = 'C:/Users/aleksei.petukhov/Desktop/DR_data/characters_numbers'

def get_pic_ids(paths):
    ids_train = []
    for path in paths:
        ids_in_path = glob.glob("{}/*.png".format(path))
        ids_train.extend(ids_in_path)
    names = [os.path.basename(name) for name in ids_train]
    del ids_train
    return names


ids_train = get_pic_ids([data_path])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

# read image as numpy array
# flatten the image
# read image id to get the label

X = []
Y = []

for id in ids_train_split:
    img = cv2.imread('{}/{}'.format(data_path, id), cv2.IMREAD_GRAYSCALE)
    # I think it is better to convert it to grayscale to reduce features from 12288 to 4096. 64 by 64 imageas are already big enough
    img = img.flatten()
    X.append(img)

    # first character of the filename is a target value
    target = id[0]
    Y.append(target)

features = np.array(X)
labels = np.array(Y)

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((64, 64)), orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')


clf = LinearSVC()
clf.fit(hog_features, labels)

joblib.dump(clf, "models/characters.pkl", compress=3)









