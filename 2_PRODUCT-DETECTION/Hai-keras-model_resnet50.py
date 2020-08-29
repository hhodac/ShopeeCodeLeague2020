import os
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import vgg16
import matplotlib.pyplot as plt
import efficientnet.keras as efn

# KMP_DUPLICATE_LIB_OK=TRUE

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
imgSize = 224
batch_size = 16
train_data_dir = '../data/input/train/'
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)

#training data
print('Generating training data...')
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
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(val_generator.filenames)
CLASS_COUNT = len(train_generator.class_indices)
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=CLASS_COUNT)

#model
print('Generating training model...')
# base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3),pooling='max')
base_model = efn.EfficientNetB4(weights='imagenet',include_top=False,pooling='avg')
base_model.trainable = False
model = Sequential([
    base_model,
    # Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(CLASS_COUNT, activation='softmax'),
])
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

#training data...
print('Training model...')
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

#save model
print('Saving model...')
save_model_path = 'test_model_resnet50.hdf5'
save_model(model, save_model_path)

#visualise model
visualize_training_model(history)