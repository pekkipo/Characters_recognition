
import glob
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from CNN_classifiers.data_augmentation import *

class CNN(object):

    def __init__(self, data_path, save_model_path, num_classes):
        self.data_path = data_path
        self.save_model_path = save_model_path  # I use '../saved_models'
        self.num_classes = num_classes

    def get_pic_ids(self, paths):
        ids_train = []
        for path in paths:
            ids_in_path = glob.glob("{}/*.png".format(path))
            ids_train.extend(ids_in_path)
        names = [os.path.basename(name) for name in ids_train]
        del ids_train
        return names

    def split_dataset(self):
        ids_train = self.get_pic_ids([self.data_path])
        ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2)

        print('Training on {} samples'.format(len(ids_train_split)))
        print('Validating on {} samples'.format(len(ids_valid_split)))

        return ids_train_split, ids_valid_split

    #@classmethod - no need for now
    def create_model(self, num_classes, input_shape=(128, 128, 1)):
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

        # Output layer with num_classes nodes (one for each possible letter/number I predict)
        model.add(Dense(num_classes, activation="softmax"))

        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    def create_data(self, ids):
        X = []
        Y = []
        for id in ids:
            img = cv2.imread('{}/{}'.format(self.data_path, id), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))

            # Do some augmentation
            if decide_on_augmentation():
                img = randomShiftScaleRotate(img,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))

            #cv2.imshow("Image", img)
            #cv2.waitKey()

            # Label is the first letter of the file id
            target = id[0]
            # target = np.expand_dims(target, axis=2)

            X.append(img)
            Y.append(target)




        X = np.array(X, np.float32) / 255  # convert to float32 and normalize
        Y = np.array(Y)

        # gotta reshape X because I have grayscale images. Maybe should have left them RGB
        X = X.reshape(X.shape[0], 128, 128, 1)

        print('Training data shape: {}'.format(X.shape))
        print('Training labels shape: {}'.format(Y.shape))

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_enc = np_utils.to_categorical(encoded_Y)

        # I have 30 categories, 9 numbers and 21 letters used

        print('Training data shape: {}'.format(X.shape))
        print('Training labels shape after encoding: {}'.format(y_enc.shape))

        classes = encoder.classes_

        print('Classes are: {}'.format(classes))



        return X, y_enc, classes

    def train_model(self, type):

        ids_train_split, ids_valid_split = self.split_dataset()

        X_train, Y_train, classes_1 = self.create_data(ids_train_split)
        X_test, Y_test, classes_2 = self.create_data(ids_valid_split)

        # see if classes 1 and 2 are the same

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
                                     filepath='saved_models/NN_model_{}.hdf5'.format(type),
                                     save_best_only=True,
                                     save_weights_only=True),
                     TensorBoard(log_dir='logs')]

        model = self.create_model(self.num_classes)
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=320, epochs=10, verbose=1, callbacks=callbacks)

        self.save_model(model, self.save_model_path)

        return classes_1

    def save_model(self, model, path):
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/model.json".format(path), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("{}/CNN_model.h5".format(path))
        print("Saved model and weights to the following location: {}".format(path))




