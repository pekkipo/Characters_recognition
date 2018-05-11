
import glob
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import random

#data_path = 'images_generation/output/characters_numbers'
data_path = '../DR_data/numbers'
# data_path = 'C:/Users/aleksei.petukhov/Desktop/DR_data/characters_numbers'

batch_size = 16

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


def create_model(input_shape=(128, 128, 1)):
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(64, (5, 5), padding="same", input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(32, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dropout to fight overfitting
    model.add(Dropout(0.2))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(30, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

##

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image


def decide_on_augmentation():
    return random.choice([True, False])

def create_data(ids):
    X = []
    Y = []
    for id in ids:
        img = cv2.imread('{}/{}'.format(data_path, id), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))

        # Do some augmentation
        if decide_on_augmentation():

            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-50, 50),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))
            img = randomShiftScaleRotate(img,
                                               shift_limit=(-0.0625, 0.0625),
                                               scale_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img = randomHorizontalFlip(img)

        cv2.imshow("Image", img)
        cv2.waitKey()

        # Label is the first letter of the file id
        target = id[0]
        # target = np.expand_dims(target, axis=2)

        X.append(img)
        Y.append(target)

    X = np.array(X, np.float32) / 255  # convert to float32 and normalize
    Y = np.array(Y)

    print(X.shape)
    print(Y.shape)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_enc = np_utils.to_categorical(encoded_Y)

    # I have 30 categories, 9 numbers and 21 letters used

    print(X.shape)
    print(y_enc.shape)

    return X, y_enc


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='models/NN_model_numbers.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]


X_train, Y_train = create_data(ids_train_split)
X_test, Y_test = create_data(ids_valid_split)

model = create_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1, callbacks=callbacks)



