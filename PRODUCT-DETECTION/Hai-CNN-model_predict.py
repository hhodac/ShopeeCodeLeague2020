import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

model_path = 'test_model4_4.hdf5'
# model_path = 'test_model_resnet50.hdf5'
model = load_model(model_path)
imgSize = 28
df = pd.read_csv('../data/test.csv')
X_test = []
for imageName in tqdm(df['filename']):
    image = cv2.imread('../data/input/test/'+imageName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgSize, imgSize))
    X_test.append(image)
X_test = np.array(X_test).astype('float16')/255
res = model.predict(X_test, batch_size=32)
res = np.argmax(res, axis=1)
df['category'] = res
df['category'] = df.category.apply(lambda c: str(c).zfill(2))
df.to_csv('output_test_model4.csv', index=False)