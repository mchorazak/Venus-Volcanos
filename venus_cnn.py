from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import pandas as pd

def runCNN():
    # LOAD DATA
    os.listdir('data/Volcanoes_train/')
    train_images = pd.read_csv('data/Volcanoes_train/train_images.csv', header=None)
    train_labels = pd.read_csv('data/Volcanoes_train/train_labels.csv')
    test_images = pd.read_csv('data/Volcanoes_test/test_images.csv', header=None)
    test_labels = pd.read_csv('data/Volcanoes_test/test_labels.csv')
    train_images.shape, train_labels.shape

    # REMOVE FIRST LINE
    train_images.head()
    train_labels.head()

    # NORMALISE
    x_train_normalised = train_images / 256
    y_train_normalised = train_labels['Volcano?']
    x_test_normalised = test_images / 256
    y_test_normalised = test_labels['Volcano?']

    # PREPARE TRAIN/TEST/VAL DATASETS
    img_width, img_height = 110, 110

    x = x_train_normalised.values.reshape(-1, img_width, img_height, 1)
    y = y_train_normalised.values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=3)

    x_test = x_test_normalised.values.reshape(-1, img_width, img_height, 1)
    y_test = y_test_normalised.values

    #CONSTRUCT MODEL
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # FIT MODEL
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    TrainingStart = time()
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
    TrainingEnd = time()
    print('Trained in: {} mins.'.format((TrainingEnd-TrainingStart)/60))


    #PREDICT
    predVali = model.predict_classes(x_val)
    predTest = model.predict_classes(x_test)

    #PRINT RESULTS
    # print('validation report:', '\n', classification_report(y_val, predVali))
    # print('validation accuracy:', accuracy_score(y_val, predVali))
    # print('validation confusion matrix:', '\n', confusion_matrix(y_val, predVali))

    # print('testing report:', '\n', classification_report(y_test, predTest))
    # print('test accuracy:', accuracy_score(y_test, predTest))
    # print('test confusion matrix:', '\n', confusion_matrix(y_test, predTest))
    return classification_report(y_test, predTest)

