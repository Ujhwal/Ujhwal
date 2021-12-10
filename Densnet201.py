import buildDataset

import json
import math
import os

from PIL import Image
import numpy as np
from keras import layers
from tensorflow.keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

x_train, x_val, y_train, y_val = train_test_split(buildDataset.X_train, buildDataset.Y_train, test_size=0.2, random_state=11)

#DataSet Generator
BATCH_SIZE = 16
# Using original generator
train_generator = ImageDataGenerator(zoom_range=2,rotation_range = 90,horizontal_flip=True,vertical_flip=True,)

#Model DensNet201

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
    
    return model

K.clear_session()
gc.collect()

densNet = DenseNet201(weights='imagenet',include_top=False,input_shape=(224,224,3))

model = build_model(densNet ,lr = 1e-4)
model.summary()

# Learning Rate Reducer
learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=5,verbose=1,factor=0.2, min_lr=1e-7)

# Checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint]
)

history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()

