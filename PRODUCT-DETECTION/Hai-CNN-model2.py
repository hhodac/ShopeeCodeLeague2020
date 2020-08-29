import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

def test_model4():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(imgSize,imgSize,3)))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(42, activation='softmax'))

    model.summary()
    return model

def visualize_training_model(historical_log):
    history = historical_log
    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return

EPOCHS = 2
imgSize = 28
batch_size = 16
train_data_dir = '../data/input/train/'
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3, vertical_flip=True, featurewise_center=True)

#training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    subset='training',
    target_size=(imgSize, imgSize),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
val_generator = datagen.flow_from_directory(
    train_data_dir,
    subset='validation',
    target_size=(imgSize, imgSize),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
nb_validation_samples = len(val_generator.filenames)
nb_train_samples = len(train_generator.filenames)

# model = test_model4()
model = load_model('test_model4_3.hdf5')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)
save_model_path = 'test_model4_4.hdf5'
save_model(model, save_model_path)
visualize_training_model(history)