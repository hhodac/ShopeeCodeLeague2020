import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
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

epochs = 15
imgSize = 28
batch_size = 16
train_data_dir = '../data/input/train/'
datagen = ImageDataGenerator(rescale=1. / 255)

#training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(imgSize, imgSize),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
nb_train_samples = len(train_generator.filenames)
num_classes = len(train_generator.class_indices)
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

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

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs)
save_model_path = 'test_model.hdf5'
save_model(model, save_model_path)