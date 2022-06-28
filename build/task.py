import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

########## --------------------
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import subprocess
import sys

def install_pk(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_pk('opendatasets')

import opendatasets as od

# ----
text = '''{"username":"abhimanyubhatia","key":"28595f163876d80cea2c8c6443264cc2"}'''

with open('kaggle.json', mode='w') as file:
    file.write(text)
# ----

dataset_url = 'https://www.kaggle.com/datasets/techsash/waste-classification-data'
od.download(dataset_url, force=True)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('waste-classification-data/DATASET/TRAIN',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 color_mode= "rgb",
                                                 class_mode= "categorical")



test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('waste-classification-data/DATASET/TEST',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            color_mode= "rgb",
                                            class_mode= "categorical")

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=2, activation='sigmoid')) #units = 2 because 2 classes
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

model = build_model()


# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(x = training_set, validation_data = test_set, epochs=1, callbacks=[early_stop])

model.save('gs://sanchit909090-bucket/wcm/model')
