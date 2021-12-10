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

#Data Pre-processesing, removing faulty images
def dataImporter(imgpath, size):
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    imgArr = []
    for i in tqdm(os.listdir(imgpath)):
        PATH = os.path.join(imgpath,i)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            #For handling images that are not in the order of array.
            try:
                img = Image.open(PATH)
                img = img.resize((size,size))
                imgArr.append(np.array(img))
            except:
                continue
    return imgArr

benignTrain = np.array(dataImporter('data/train/benign',224))
malignTrain = np.array(dataImporter('data/train/malignant',224))
benignTest = np.array(dataImporter('data/validation/benign',224))
malignTest = np.array(dataImporter('data/validation/malignant',224))
print(len(benignTrain),len(malignTrain),len(benignTest),len(malignTest))

# Creating Test and Train Arrays, Where each consists of X,Y Co-ordinates. X consists of the corresponding data of both benign and malignant
X_train = np.concatenate((benignTrain, malignTrain), axis = 0)
Y_train = np.concatenate((np.zeros(len(benignTrain)), np.ones(len(malignTrain))), axis = 0)
X_test = np.concatenate((benignTest, malignTest), axis = 0)
Y_test = np.concatenate((np.zeros(len(benignTest)), np.ones(len(malignTest))), axis = 0)

# Shuffle train data and test data
trainShuf = np.arange(X_train.shape[0])
testShuff = np.arange(X_test.shape[0])
np.random.shuffle(trainShuf)
np.random.shuffle(testShuff)
X_train = X_train[trainShuf]
Y_train = Y_train[trainShuf]
X_test = X_test[testShuff]
Y_test = Y_test[testShuff]

# To categorical
Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)